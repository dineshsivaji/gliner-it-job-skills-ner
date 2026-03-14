"""
train_gliner_resume.py
----------------------
Fine-tunes GLiNER medium on resume parsing using 2-label taxonomy.

  - Uses gliner_medium-v2.1 instead of large
  - Trains on re-labelled dataset (run relabel_dataset.py first)
  - 800 steps (medium needs more than large for same domain coverage)
  - Only 2 trained labels: TECHNICAL_SKILL, JOB_TITLE
    (sparse labels handled zero-shot by base model at inference)
  - Multi-GPU support via torchrun (T4 x2 on Kaggle)
  - fp16 enabled for Tensor Core acceleration on T4
  - num_epochs calculation accounts for effective batch size across GPUs

Launch (single GPU):
    python train_gliner_resume.py

Launch (multi-GPU, e.g. T4 x2 on Kaggle):
    !torchrun --nproc_per_node=2 train_gliner_resume.py
"""

import json
import os
import random

import torch

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# ── Config ───────────────────────────────────────────────────────────────────

TRAIN_PATH   = "/kaggle/working/synthetic_gliner_relabelled.json"  # output of relabel_dataset.py
OUTPUT_DIR   = "/kaggle/working/gliner-it-job-skills-ner"
HF_REPO_ID   = "dineshsivaji/gliner-it-job-skills-ner"

NUM_STEPS        = 800
BATCH_SIZE       = 8   # per device — effective = BATCH_SIZE * num_gpus
LR               = 5e-6
OTHERS_LR        = 1e-5
WARMUP_RATIO     = 0.1

# ── Load & split ─────────────────────────────────────────────────────────────

print(f"Loading dataset from: {TRAIN_PATH}")
with open(TRAIN_PATH) as f:
    data = json.load(f)

print(f"  Total examples: {len(data)}")

random.seed(42)
random.shuffle(data)

split = int(len(data) * 0.9)
train_dataset = data[:split]
test_dataset  = data[split:]

print(f"  Train: {len(train_dataset)}  |  Test: {len(test_dataset)}")

# ── Model ────────────────────────────────────────────────────────────────────

from gliner import GLiNER

num_gpus       = torch.cuda.device_count()
use_cuda       = num_gpus > 0
device         = torch.device("cuda:0") if use_cuda else torch.device("cpu")
use_fp16       = use_cuda  # fp16 only on GPU — uses T4 Tensor Cores

print(f"\nDevice       : {device}")
print(f"GPUs found   : {num_gpus}  {'(multi-GPU enabled)' if num_gpus > 1 else '(single GPU)'}")
print(f"fp16 enabled : {use_fp16}")

print("\nLoading urchade/gliner_medium-v2.1 ...")
model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

# ── Training config ──────────────────────────────────────────────────────────
# With torchrun, HuggingFace Trainer splits batches across GPUs automatically.
# effective_batch = per_device_batch * num_gpus
# num_batches must reflect this so num_epochs is correct.

data_size      = len(train_dataset)
effective_batch = BATCH_SIZE * max(1, num_gpus)
num_batches    = data_size // effective_batch
num_epochs     = max(1, NUM_STEPS // num_batches)

print(f"\nTraining config:")
print(f"  num_steps        : {NUM_STEPS}")
print(f"  per_device_batch : {BATCH_SIZE}")
print(f"  num_gpus         : {max(1, num_gpus)}")
print(f"  effective_batch  : {effective_batch}")
print(f"  data_size        : {data_size}")
print(f"  num_batches      : {num_batches}")
print(f"  num_epochs       : {num_epochs}")

trainer = model.train_model(
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    output_dir="models",
    learning_rate=LR,
    weight_decay=0.01,
    others_lr=OTHERS_LR,
    others_weight_decay=0.01,
    lr_scheduler_type="linear",
    warmup_ratio=WARMUP_RATIO,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    focal_loss_alpha=0.75,
    focal_loss_gamma=2,
    num_train_epochs=num_epochs,
    save_steps=10000,
    save_total_limit=1,
    dataloader_num_workers=0,
    use_cpu=False,
    fp16=use_fp16,       # Tensor Core acceleration on T4
    report_to="none",
)

# ── Save & push to HuggingFace ───────────────────────────────────────────────

print(f"\nSaving model to: {OUTPUT_DIR}")
model.save_pretrained(OUTPUT_DIR)

from huggingface_hub import login, HfApi
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HF_TOKEN")
login(token=hf_token)

api = HfApi()
api.create_repo(repo_id=HF_REPO_ID, repo_type="model", exist_ok=True)
api.upload_folder(folder_path=OUTPUT_DIR, repo_id=HF_REPO_ID, repo_type="model")

print(f"\n✅ Model uploaded to: https://huggingface.co/{HF_REPO_ID}")
