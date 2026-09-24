import mlflow
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

mlflow.set_tracking_uri("http://localhost:5001")

CACHE_DIR = Path("./.model_cache")   # temp location inside your project dir
CACHE_DIR.mkdir(exist_ok=True)

local_path = mlflow.artifacts.download_artifacts(
    artifact_uri="models:/qwen2.5-0.5b-instruct/1",   # or /latest, or an alias like @champion
    dst_path=str(CACHE_DIR),
)

tokenizer = AutoTokenizer.from_pretrained(local_path)
model = AutoModelForCausalLM.from_pretrained(local_path)