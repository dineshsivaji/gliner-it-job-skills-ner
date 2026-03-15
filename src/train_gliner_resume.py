"""
train_gliner_resume.py
----------------------
Fine-tunes GLiNER medium on resume parsing using 2-label taxonomy.

  - Uses gliner_medium-v2.1 instead of large
  - Trains on re-labelled dataset (run relabel_dataset.py first)
  - Only 2 trained labels: TECHNICAL_SKILL, JOB_TITLE
    (sparse labels handled zero-shot by base model at inference)
  - Multi-GPU support via torchrun (T4 x2 on Kaggle)
  - fp16 enabled for Tensor Core acceleration on T4
  - Early stopping + best-model checkpointing

Launch (single GPU):
    python -m src.train_gliner_resume

Launch (multi-GPU, e.g. T4 x2 on Kaggle):
    torchrun --nproc_per_node=2 -m src.train_gliner_resume

All paths are configurable via CLI args or env vars.
"""

import argparse
import json
import os
import random

import torch

os.environ["TOKENIZERS_PARALLELISM"] = "true"

# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_TRAIN_PATH = os.getenv(
    "TRAIN_PATH", "src/training_data/synthetic_gliner_relabelled.json"
)
DEFAULT_OUTPUT_DIR = os.getenv(
    "OUTPUT_DIR", "models/gliner-it-job-skills-ner"
)
DEFAULT_HF_REPO = "dineshsivaji/gliner-it-job-skills-ner"
DEFAULT_BASE_MODEL = "urchade/gliner_medium-v2.1"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune GLiNER medium on IT resume data"
    )
    parser.add_argument("--data", default=DEFAULT_TRAIN_PATH,
                        help="Path to relabelled dataset JSON")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR,
                        help="Local directory to save the fine-tuned model")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                        help="Base GLiNER model to fine-tune")
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO,
                        help="HuggingFace repo ID to upload the model")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=5e-6,
                        help="Learning rate for the main model")
    parser.add_argument("--others-lr", type=float, default=1e-5,
                        help="Learning rate for other params")
    parser.add_argument("--warmup-ratio", type=float, default=0.1,
                        help="Warmup ratio for LR scheduler")
    parser.add_argument("--eval-steps", type=int, default=200,
                        help="Run eval every N steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-upload", action="store_true",
                        help="Skip HuggingFace upload")
    args = parser.parse_args()

    # ── Load & split ─────────────────────────────────────────────────────────

    print(f"Loading dataset from: {args.data}")
    with open(args.data) as f:
        data = json.load(f)
    print(f"  Total examples: {len(data)}")

    random.seed(args.seed)
    random.shuffle(data)

    split = int(len(data) * 0.9)
    train_dataset = data[:split]
    test_dataset = data[split:]
    print(f"  Train: {len(train_dataset)}  |  Test: {len(test_dataset)}")

    # ── Model ────────────────────────────────────────────────────────────────

    from gliner import GLiNER

    num_gpus = torch.cuda.device_count()
    use_cuda = num_gpus > 0
    device = torch.device("cuda:0") if use_cuda else torch.device("cpu")
    use_fp16 = use_cuda  # fp16 only on GPU — uses Tensor Cores

    print(f"\nDevice       : {device}")
    print(f"GPUs found   : {num_gpus}  "
          f"{'(multi-GPU)' if num_gpus > 1 else '(single GPU)' if num_gpus == 1 else '(CPU)'}")
    print(f"fp16 enabled : {use_fp16}")

    print(f"\nLoading {args.base_model} ...")
    model = GLiNER.from_pretrained(args.base_model)

    # ── Training config ──────────────────────────────────────────────────────

    effective_batch = args.batch_size * max(1, num_gpus)
    num_batches = len(train_dataset) // effective_batch

    print(f"\nTraining config:")
    print(f"  num_epochs       : {args.epochs}")
    print(f"  per_device_batch : {args.batch_size}")
    print(f"  num_gpus         : {max(1, num_gpus)}")
    print(f"  effective_batch  : {effective_batch}")
    print(f"  data_size        : {len(train_dataset)}")
    print(f"  batches/epoch    : {num_batches}")
    print(f"  total_steps      : ~{num_batches * args.epochs}")
    print(f"  eval_steps       : {args.eval_steps}")

    trainer = model.train_model(
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        output_dir="checkpoints",
        learning_rate=args.lr,
        weight_decay=0.01,
        others_lr=args.others_lr,
        others_weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        focal_loss_alpha=0.75,
        focal_loss_gamma=2,
        num_train_epochs=args.epochs,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=0,
        use_cpu=not use_cuda,
        fp16=use_fp16,
        report_to="none",
    )

    # ── Save ─────────────────────────────────────────────────────────────────

    print(f"\nSaving best model to: {args.output}")
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)

    # ── Upload to HuggingFace (optional) ─────────────────────────────────────

    if args.no_upload:
        print("Skipping HuggingFace upload (--no-upload flag).")
        return

    try:
        from huggingface_hub import login, HfApi

        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            try:
                from kaggle_secrets import UserSecretsClient
                hf_token = UserSecretsClient().get_secret("HF_TOKEN")
            except Exception:
                pass

        if not hf_token:
            print("⚠️  HF_TOKEN not found. Skipping upload.")
            return

        login(token=hf_token)
        api = HfApi()
        api.create_repo(repo_id=args.hf_repo, repo_type="model", exist_ok=True)
        api.upload_folder(folder_path=args.output, repo_id=args.hf_repo, repo_type="model")
        print(f"\n✅ Model uploaded to: https://huggingface.co/{args.hf_repo}")

    except ImportError:
        print("⚠️  huggingface_hub not installed. Skipping upload.")


if __name__ == "__main__":
    main()
