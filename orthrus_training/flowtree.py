"""Proposal-tree primitives for verifier-aware FlowDraft decoding.

FlowTree is deliberately separate from the single-trajectory FlowDraft path:
the frozen AR model remains the only authority that emits tokens.  The drafter
only provides a small tree of alternatives for the verifier to traverse.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class FlowTree:
    """A prefix tree in topological order; node zero is the known anchor."""

    token_ids: torch.Tensor
    parents: torch.Tensor
    depths: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return int(self.token_ids.numel())


def build_flowtree(
    anchor_token_id: int,
    endpoint_logits: torch.Tensor,
    branch_width: int,
    branch_depth: int,
    max_nodes: int,
) -> FlowTree:
    """Build a bounded top-k prefix tree from one parallel flow endpoint.

    The first ``branch_depth`` positions branch.  Every surviving leaf then
    follows the endpoint argmax, which keeps the tree compact while preserving
    several alternatives at the high-value early decisions.
    """

    if endpoint_logits.dim() != 2:
        raise ValueError("endpoint_logits must have shape [depth, vocab]")
    if branch_width < 1 or branch_depth < 0 or max_nodes < 1:
        raise ValueError("invalid FlowTree dimensions")

    depth_limit = int(endpoint_logits.shape[0])
    branch_depth = min(branch_depth, depth_limit)
    tokens = [int(anchor_token_id)]
    parents = [-1]
    depths = [0]
    frontier = [0]

    for depth in range(1, depth_limit + 1):
        width = branch_width if depth <= branch_depth else 1
        candidates = endpoint_logits[depth - 1].topk(width).indices.tolist()
        next_frontier: list[int] = []
        for parent in frontier:
            for token in candidates:
                if len(tokens) >= max_nodes:
                    break
                parents.append(parent)
                tokens.append(int(token))
                depths.append(depth)
                next_frontier.append(len(tokens) - 1)
            if len(tokens) >= max_nodes:
                break
        if not next_frontier:
            break
        frontier = next_frontier

    return FlowTree(
        token_ids=torch.tensor(tokens, dtype=torch.long, device=endpoint_logits.device),
        parents=torch.tensor(parents, dtype=torch.long, device=endpoint_logits.device),
        depths=torch.tensor(depths, dtype=torch.long, device=endpoint_logits.device),
    )


def ancestor_matrix(parents: torch.Tensor) -> torch.Tensor:
    """Return visibility[q, k]: token k is on query q's inclusive path."""

    if parents.dim() != 1 or parents.numel() == 0 or int(parents[0]) != -1:
        raise ValueError("parents must be a non-empty vector with root parent -1")
    size = int(parents.numel())
    visibility = torch.zeros((size, size), dtype=torch.bool, device=parents.device)
    for node in range(size):
        current = node
        while current >= 0:
            visibility[node, current] = True
            current = int(parents[current])
    return visibility


def greedy_path_coverage(tree: FlowTree, teacher_tokens: torch.Tensor) -> int:
    """Number of teacher tokens covered by a root-to-leaf path in ``tree``."""

    if teacher_tokens.dim() != 1:
        raise ValueError("teacher_tokens must be one-dimensional")
    current = 0
    covered = 0
    for expected in teacher_tokens.tolist():
        children = torch.nonzero(tree.parents == current, as_tuple=False).flatten()
        matches = children[tree.token_ids[children] == int(expected)]
        if matches.numel() == 0:
            break
        current = int(matches[0])
        covered += 1
    return covered


def soft_topk_coverage_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    branch_width: int,
    branch_depth: int,
    temperature: float = 1.0,
    margin: float = 0.5,
) -> torch.Tensor:
    """Differentiable surrogate for putting the teacher path in FlowTree.

    It penalizes a teacher token whose smooth rank exceeds the tree's branch
    budget.  Unlike CE, it stops spending gradient after a target is safely in
    the proposal set, leaving capacity for downstream correlated branches.
    """

    if logits.dim() != 3 or target_ids.shape != logits.shape[:2]:
        raise ValueError("expected logits [B, T, V] and target_ids [B, T]")
    if branch_width < 1 or branch_depth < 1 or temperature <= 0:
        raise ValueError("invalid coverage-loss hyperparameters")
    active = min(int(logits.shape[1]), branch_depth)
    active_logits = logits[:, :active, :].float()
    active_targets = target_ids[:, :active]
    target_scores = active_logits.gather(-1, active_targets.unsqueeze(-1))
    # Smooth rank: 1 + number of vocabulary entries that outrank the target.
    outranking = torch.sigmoid((active_logits - target_scores + margin) / temperature)
    soft_rank = outranking.sum(dim=-1) - 0.5
    return F.relu(soft_rank - float(branch_width)).mean()
