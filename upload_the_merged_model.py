# upload_to_hf.py
from huggingface_hub import HfApi, create_repo, upload_folder
from pathlib import Path
import os, sys

# --- EDIT THESE ---
HF_TOKEN  = "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"  # write-scoped token
HF_USER   = "qbao775"                                                    # your HF username (must match token)
REPO_NAME = "qwen2p5-3b-mi-dpo-merged"                                   # repo name you want
OUT_DIR   = "/mnt/gptcoaching_mi_training/runs/qwen2p5-3b-mi-dpo-merged" # folder to upload
PRIVATE   = True
# ------------------

api = HfApi(token=HF_TOKEN)

me = api.whoami()  # sanity check the token
if me["name"] != HF_USER and HF_USER not in me.get("orgs", []):
    raise SystemExit(f"Token belongs to '{me['name']}', not '{HF_USER}'. Fix HF_USER or token.")

repo_id = f"{HF_USER}/{REPO_NAME}"
create_repo(repo_id=repo_id, private=PRIVATE, exist_ok=True, token=HF_TOKEN)

readme = Path(OUT_DIR) / "README.md"
if not readme.exists():
    readme.write_text(
        f"# {REPO_NAME}\n\nMerged weights derived from `Qwen/Qwen2.5-3B-Instruct` after SFT/DPO for MI-style coaching.\n"
        "Check the base model license before redistribution.\n",
        encoding="utf-8"
    )

upload_folder(
    repo_id=repo_id,
    folder_path=OUT_DIR,
    path_in_repo=".",
    commit_message="Upload merged full model",
    token=HF_TOKEN,
    ignore_patterns=[
        "optimizer.pt","scheduler.pt","rng_state.pth",
        "training_args.bin","trainer_state.json",".ipynb_checkpoints/*"
    ],
)

print(f"✅ Uploaded: https://huggingface.co/{repo_id}")
