# Project Workflow

## Local Checks

Run syntax checks:

```bash
python -m py_compile scripts/*.py orthrus_training/*.py
```

Run tests:

```bash
python -m pytest -q
```

## DataSphere Jobs

Run the current FlowDraft hard-CE experiment:

```bash
scripts/remote-run.sh
```

Use a custom job config:

```bash
scripts/remote-run.sh datasphere/jobs/flowdraft-hardce.yaml
```

List jobs:

```bash
scripts/remote-list.sh
```

Attach to logs:

```bash
scripts/remote-attach.sh <JOB_ID>
```

Download results:

```bash
scripts/remote-download.sh <JOB_ID>
```

Required local environment:

```bash
export DATASPHERE_PROJECT_ID=<project_id>
export HF_TOKEN=<huggingface_token>
```

## Rules

- Keep executable experiment logic in scripts and `orthrus_training/`.
- Before launching a GPU job, run available local tests and static checks.
- Job outputs should go under `outputs/`.
- Do not print credentials, tokens, or secret values.
- Do not delete remote jobs or remote results unless explicitly requested.
- Avoid launching multiple GPU jobs concurrently unless parallel execution is intentional.
