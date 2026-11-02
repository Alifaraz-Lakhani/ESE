from huggingface_hub import snapshot_download
import os

# os.environ["HF_TOKEN"] = "hf_YourTokenHere"  # Optional

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
local_dir = "./models/Qwen2.5-1.5B-Instruct"

snapshot_download(
    repo_id=model_name,
    local_dir=local_dir,
    local_dir_use_symlinks=False
)
print(f"Model downloaded to {local_dir}")