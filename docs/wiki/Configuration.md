# Configuration

The server uses repository-relative paths by default. You can override them with environment variables.

## Project Paths

| Variable | Default |
| --- | --- |
| `DRONE_PROJECT_ROOT` | parent directory of `merging/` |
| `MERGING_DATA_ROOT` | `merging/data/jobs` |
| `LINGBOT_MODEL_PATH` | `lingbot-map/lingbot-map-long.pt` |
| `TRIMAMBA_CKPT` | `Trimamba/checkpoints/best_model_ckpt.pth` |

## Conda Paths

The app locates conda environments with this order:

1. Explicit interpreter variables.
2. `CONDA_ROOT`.
3. `CONDA_EXE`.
4. `CONDA_PREFIX`.
5. `~/miniconda3`.

| Variable | Default env |
| --- | --- |
| `LINGBOT_PYTHON` | `lingbot-map` |
| `TRIMAMBA_PYTHON` | `triple` |
| `RETRIEVAL_PYTHON` | `retrieval` |

Example:

```bash
export CONDA_ROOT=$HOME/miniconda3
export RETRIEVAL_PYTHON=$CONDA_ROOT/envs/retrieval/bin/python
```

## Generated Data

Each upload creates a job directory:

```text
merging/data/jobs/<job_id>/
```

This directory contains videos, NPZ files, PCD files, summaries, retrieval indexes, logs, and viewer logs. It is ignored by Git.
