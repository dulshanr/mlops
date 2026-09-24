import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REGISTERED_NAME = "qwen2.5-0.5b-instruct"

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("llm-registry")
client = MlflowClient()

local_dir = snapshot_download(repo_id=MODEL_ID, local_dir="./_hf_download")

with mlflow.start_run(run_name=f"import-{MODEL_ID}") as run:
    mlflow.set_tags({"source": "huggingface", "hf_repo": MODEL_ID})
    mlflow.log_artifacts(local_dir, artifact_path="hf_model")

# create the registered model if it doesn't exist yet
try:
    client.create_registered_model(REGISTERED_NAME)
except MlflowException:
    pass

mv = client.create_model_version(
    name=REGISTERED_NAME,
    source=f"{run.info.artifact_uri}/hf_model",
    run_id=run.info.run_id,
)
print("Registered version:", mv.version)