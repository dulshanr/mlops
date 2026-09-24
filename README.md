# MLOps Playground

A small Python project for experimenting with MLflow tracking, model registry
workflows, and Hugging Face model artifacts. It includes two examples:

- Train and log a scikit-learn Iris classifier with MLflow.
- Import a Hugging Face language model into the MLflow Model Registry, then
  download it again for local Transformers use.

## Prerequisites

- Python 3.11 or newer
- [uv](https://docs.astral.sh/uv/) (recommended), or another Python environment manager
- An MLflow tracking server available at `http://127.0.0.1:5001`

## Setup

Create the environment and install the locked dependencies:

```bash
uv sync
```

Start ML flow in docker using docker compose

https://github.com/mlflow/mlflow/blob/master/docker-compose/README.md

Open the MLflow UI at <http://127.0.0.1:5001>.

## Iris classification example

Run the autologging example:

```bash
uv run python main.py
```

`main.py` trains a logistic-regression classifier on the Iris dataset with
MLflow's scikit-learn autologging enabled. The run, parameters, metrics, and
model are recorded in the `MLflow Quickstart` experiment.

For the equivalent explicit logging flow, run:

```bash
uv run python manual.py
```

## Model registry inference

`inference.py` loads `models:/model_v2/1`, generates predictions for the Iris
test split, and prints sample actual and predicted classes:

```bash
uv run python inference.py
```

Before running it, register a compatible model as version `1` of `model_v2`, or
change `model_uri` in `inference.py` to the name and version in your registry.

## Hugging Face model workflow

Import Qwen 2.5 0.5B Instruct from Hugging Face and create an MLflow registered
model version:

```bash
uv run python hugging-face-push.py
```

The script downloads `Qwen/Qwen2.5-0.5B-Instruct`, logs the files under the
`hf_model` artifact path, and registers them as `qwen2.5-0.5b-instruct`.

Download registered version `1` back into a local cache and load it with
Transformers:

```bash
uv run python hugging-face-pull.py
```

The model artifacts are placed in `.model_cache/`. Both `.model_cache/` and the
intermediate `_hf_download/` directory are excluded from Git.

## Project files

| File | Purpose |
| --- | --- |
| `main.py` | Iris training with MLflow autologging |
| `manual.py` | Iris training with explicit MLflow logging |
| `inference.py` | Load a registered model and run Iris predictions |
| `hugging-face-push.py` | Download and register a Hugging Face model |
| `hugging-face-pull.py` | Download a registered Hugging Face model locally |

## Notes

- The scripts currently use a local tracking URI. If your server is elsewhere,
  update the `mlflow.set_tracking_uri(...)` value in the scripts.
- Hugging Face model downloads can be several gigabytes. They are intentionally
  kept out of version control.
