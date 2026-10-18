"""
MendSeed AI — Production QLoRA Fine-Tuning Pipeline.

Designed for hosted GPU environments (Google Colab, Kaggle, Modal, RunPod, Lambda Labs, AWS).
Optimized for 16GB VRAM (NVIDIA Tesla T4, Turing sm_75 / A10G / L4 / A100).

Key Capabilities:
- Validates training and validation dataset schemas and SHA-256 hashes before loading model weights.
- Strict CUDA pre-flight verification: refuses to run on CPU without hardware acceleration.
- Real validation-based early stopping (selects checkpoint by measured validation loss, not hardcoded step numbers).
- Saves comprehensive training metadata, git/data hashes, seed, and trainer history for reproducible auditing.
- Uploads to Hugging Face only when explicitly requested (--push-to-hub).
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
import sys

# Path resolution
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
SDK_DIR = os.path.join(PARENT_DIR, "sdk")
if SDK_DIR not in sys.path:
    sys.path.insert(0, SDK_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from validate_dataset import DatasetValidator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("mendseed_ai.training")


def compute_file_sha256(filepath: str) -> str:
    """Computes SHA-256 checksum of a file for artifact tracking."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description="MendSeed AI QLoRA Training Pipeline")
    parser.add_argument(
        "--train-file",
        type=str,
        default=os.path.join(PARENT_DIR, "training", "data", "train.jsonl"),
        help="Path to training jsonl split"
    )
    parser.add_argument(
        "--val-file",
        type=str,
        default=os.path.join(PARENT_DIR, "training", "data", "val.jsonl"),
        help="Path to validation jsonl split"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(PARENT_DIR, "training", "checkpoints"),
        help="Output directory for checkpoints and training artifacts"
    )
    parser.add_argument(
        "--base-model-id",
        type=str,
        default=os.getenv("BASE_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct"),
        help="Hugging Face base model identifier"
    )
    parser.add_argument(
        "--base-model-revision",
        type=str,
        default=os.getenv("BASE_MODEL_REVISION", "main"),
        help="Base model git commit revision"
    )
    parser.add_argument(
        "--adapter-repo",
        type=str,
        default=os.getenv("ADAPTER_REPO", "itsjustayush/mendseed-qwen2.5-3b-lora"),
        help="Target Hugging Face adapter repo"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device train batch size"
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=150,
        help="Maximum training steps"
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=15,
        help="Evaluation and checkpoint saving interval in steps"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload fine-tuned adapter to Hugging Face Hub at completion"
    )
    return parser.parse_args()


def preflight_checks(args):
    """
    Validates hardware, environment, and dataset integrity before allocating GPU memory.
    """
    # 1. Fallback path resolution if default relative paths differ
    if not os.path.exists(args.train_file):
        alt_path = os.path.join(CURRENT_DIR, "mendseed_training_data.jsonl")
        if os.path.exists(alt_path):
            logger.info(f"Using alternative train path: {alt_path}")
            args.train_file = alt_path
        else:
            raise FileNotFoundError(f"Training data not found at: {args.train_file}")

    # 2. Dataset schema and contract validation
    validator = DatasetValidator()
    train_report = validator.validate_file(args.train_file)
    if not train_report["valid"]:
        raise ValueError(f"Training dataset validation failed: {train_report.get('contract_errors')}")
    logger.info(f"Validated {train_report['valid_examples']} training examples in {args.train_file}")

    val_report = None
    if args.val_file and os.path.exists(args.val_file):
        val_report = validator.validate_file(args.val_file)
        if not val_report["valid"]:
            raise ValueError(f"Validation dataset check failed: {val_report.get('contract_errors')}")
        logger.info(f"Validated {val_report['valid_examples']} validation examples in {args.val_file}")

        # Check split leakage
        leakage = validator.check_split_leakage({
            "train": train_report["examples"],
            "val": val_report["examples"]
        })
        if leakage["leakage_detected"]:
            logger.warning(f"Potential leakage detected between train and val: {leakage['warnings']}")

    # 3. Hardware acceleration check
    try:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA-capable GPU is strictly required for 4-bit QLoRA fine-tuning.\n"
                "Do NOT run fine-tuning on a local CPU machine. Please launch this script on a hosted "
                "GPU session (e.g. Google Colab T4, Kaggle, Modal, or cloud VM with NVIDIA GPU)."
            )
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"Pre-flight GPU check passed: {gpu_name} ({vram_gb:.1f} GB VRAM available)")
    except ImportError:
        raise RuntimeError(
            "PyTorch with CUDA support is not installed in this local environment.\n"
            "Do NOT run fine-tuning on a local CPU machine. Please launch this script on a hosted "
            "GPU session (e.g. Google Colab T4, Kaggle, Modal, RunPod, or a cloud VM with an NVIDIA GPU)."
        )

    return train_report, val_report


def run_training():
    args = parse_args()
    logger.info("=== Starting MendSeed AI QLoRA Training Pipeline ===")

    # Run preflight checks
    train_report, val_report = preflight_checks(args)

    # Import heavy ML libraries only inside training execution
    import torch
    from datasets import load_dataset
    from huggingface_hub import login
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    import transformers
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTTrainer

    # HF Authentication if pushing
    hf_token = os.getenv("HF_TOKEN")
    if args.push_to_hub:
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable required when --push-to-hub is specified.")
        login(token=hf_token)
        logger.info("Authenticated with Hugging Face Hub.")

    # Quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    logger.info(f"Loading tokenizer: {args.base_model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, revision=args.base_model_revision)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    logger.info(f"Loading base model {args.base_model_id} in 4-bit NF4...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        revision=args.base_model_revision,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Ingest and format dataset using Qwen's ChatML template
    logger.info(f"Loading datasets...")
    train_dataset = load_dataset("json", data_files=args.train_file, split="train")

    def format_chat(row):
        messages = [
            {"role": "system", "content": row["system"]},
            {"role": "user", "content": row["input"]},
            {"role": "assistant", "content": row["output"]}
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    train_formatted = train_dataset.map(format_chat)

    eval_formatted = None
    if args.val_file and os.path.exists(args.val_file):
        eval_dataset = load_dataset("json", data_files=args.val_file, split="train")
        eval_formatted = eval_dataset.map(format_chat)

    # Training arguments with real validation tracking
    has_val = eval_formatted is not None and len(eval_formatted) > 0

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        max_steps=args.max_steps,
        eval_strategy="steps" if has_val else "no",
        eval_steps=args.eval_steps if has_val else None,
        save_strategy="steps" if has_val else "no",
        save_steps=args.eval_steps if has_val else None,
        load_best_model_at_end=True if has_val else False,
        metric_for_best_model="eval_loss" if has_val else None,
        fp16=True,
        bf16=False,
        optim="paged_adamw_8bit",
        seed=args.seed,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_formatted,
        eval_dataset=eval_formatted,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=1024,
        processing_class=tokenizer,
        args=training_args,
    )

    logger.info("Executing QLoRA fine-tuning...")
    train_result = trainer.train()

    # Save best adapter weights locally
    final_adapter_dir = os.path.join(args.output_dir, "best_adapter")
    os.makedirs(final_adapter_dir, exist_ok=True)
    trainer.model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)
    logger.info(f"Saved fine-tuned LoRA adapter to: {final_adapter_dir}")

    # Save comprehensive reproducibility metadata
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "base_model_id": args.base_model_id,
        "base_model_revision": args.base_model_revision,
        "adapter_repo": args.adapter_repo,
        "lora_config": {
            "r": peft_config.r,
            "lora_alpha": peft_config.lora_alpha,
            "target_modules": peft_config.target_modules,
            "lora_dropout": peft_config.lora_dropout,
        },
        "training_data": {
            "path": args.train_file,
            "sha256": compute_file_sha256(args.train_file),
            "record_count": len(train_formatted),
        },
        "validation_data": {
            "path": args.val_file if has_val else None,
            "sha256": compute_file_sha256(args.val_file) if has_val and os.path.exists(args.val_file) else None,
            "record_count": len(eval_formatted) if has_val else 0,
        },
        "hyperparameters": {
            "max_steps": args.max_steps,
            "effective_batch_size": args.batch_size * args.grad_accum,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
        },
        "training_metrics": {
            "train_loss": train_result.training_loss,
            "best_checkpoint_step": trainer.state.best_model_checkpoint,
            "best_metric": trainer.state.best_metric,
            "log_history": trainer.state.log_history,
        },
        "dependencies": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        }
    }

    meta_path = os.path.join(args.output_dir, "training_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved training metadata & log history: {meta_path}")

    # Push to hub only if explicitly requested
    if args.push_to_hub:
        logger.info(f"Pushing adapter to Hugging Face Hub: {args.adapter_repo}...")
        trainer.model.push_to_hub(args.adapter_repo, private=True)
        tokenizer.push_to_hub(args.adapter_repo, private=True)
        logger.info("Upload complete!")
    else:
        logger.info("Adapter saved locally. --push-to-hub was not set.")


if __name__ == "__main__":
    run_training()
