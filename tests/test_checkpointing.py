import json

import torch

from orthrus_training.checkpointing import save_trainable_checkpoint
from orthrus_training.modeling import load_trainable_initialization


class DummyConfig:
    block_size = 4
    mask_token_id = 31
    flowdraft_cfm = True
    flowdraft_objective = "ecld"
    flowdraft_time_conditioning_scale = 0.05
    flowdraft_endpoint_topk = 8


class DummyCheckpointModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = torch.nn.Linear(3, 3)
        self.adapter = torch.nn.Linear(3, 2, bias=False)
        self.config = DummyConfig()
        for parameter in self.frozen.parameters():
            parameter.requires_grad = False


def test_trainable_checkpoint_excludes_frozen_weights(tmp_path):
    model = DummyCheckpointModel()
    output = tmp_path / "adapter"
    save_trainable_checkpoint(model, output, "test/base")

    from safetensors.torch import load_file

    state = load_file(output / "adapter_model.safetensors")
    metadata = json.loads((output / "adapter_config.json").read_text(encoding="utf-8"))

    assert set(state) == {"adapter.weight"}
    assert metadata["base_model"] == "test/base"
    assert metadata["flowdraft_cfm"] is True
    assert metadata["trainable_parameter_names"] == ["adapter.weight"]

    restored = DummyCheckpointModel()
    with torch.no_grad():
        restored.adapter.weight.zero_()
    loaded = load_trainable_initialization(restored, output)

    assert loaded == ["adapter.weight"]
    assert torch.equal(restored.adapter.weight, model.adapter.weight)
