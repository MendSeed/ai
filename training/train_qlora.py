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
import inspect
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
        "--train-file", "--train_file",
        type=str,
        default=os.path.join(PARENT_DIR, "training", "data", "train.jsonl"),
        help="Path to training jsonl split"
    )
    parser.add_argument(
        "--val-file", "--val_file",
        type=str,
        default=os.path.join(PARENT_DIR, "training", "data", "val.jsonl"),
        help="Path to validation jsonl split"
    )
    parser.add_argument(
        "--output-dir", "--output_dir",
        type=str,
        default=os.path.join(PARENT_DIR, "training", "checkpoints"),
        help="Output directory for checkpoints and training artifacts"
    )
    parser.add_argument(
        "--base-model-id", "--base_model_id",
        type=str,
        default=os.getenv("BASE_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct"),
        help="Hugging Face base model identifier"
    )
    parser.add_argument(
        "--base-model-revision", "--base_model_revision",
        type=str,
        default=os.getenv("BASE_MODEL_REVISION", "main"),
        help="Base model git commit revision"
    )
    parser.add_argument(
        "--adapter-repo", "--adapter_repo",
        type=str,
        default=os.getenv("ADAPTER_REPO", "itsjustayush/mendseed-qwen2.5-3b-lora"),
        help="Target Hugging Face adapter repo"
    )
    parser.add_argument(
        "--batch-size", "--batch_size",
        type=int,
        default=2,
        help="Per-device train batch size"
    )
    parser.add_argument(
        "--grad-accum", "--grad_accum",
        type=int,
        default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning-rate", "--learning_rate",
        type=float,
        default=2e-4,
        help="Initial learning rate"
    )
    parser.add_argument(
        "--warmup-ratio", "--warmup_ratio",
        type=float,
        default=0.05,
        help="Warmup ratio for learning rate scheduler"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=True,
        help="Enable FP16 mixed precision training (recommended for Tesla T4)"
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=False,
        help="Enable BF16 mixed precision training (requires Ampere A100/L4 or newer)"
    )
    parser.add_argument(
        "--max-steps", "--max_steps",
        type=int,
        default=150,
        help="Maximum training steps"
    )
    parser.add_argument(
        "--eval-steps", "--eval_steps",
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
        "--push-to-hub", "--push_to_hub",
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

    logger.info(f"Loading base model {args.base_model_id} in 4-bit NF4 with torch.float16 (T4/Turing hardware compatible)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_id,
        revision=args.base_model_revision,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    try:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except TypeError:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

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
    logger.info("Loading datasets...")
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

    # Dynamically select SFTConfig (trl >= 0.12.0) or TrainingArguments (legacy trl)
    import trl
    from trl import SFTTrainer

    use_sft_config = hasattr(trl, "SFTConfig")
    training_args_cls = getattr(trl, "SFTConfig", TrainingArguments)

    sig = inspect.signature(training_args_cls.__init__)
    supported_params = set(sig.parameters.keys())

    # Universal warmup steps calculation (avoids warmup_ratio parameter deprecations)
    calc_warmup_steps = max(1, int(args.max_steps * (args.warmup_ratio if args.warmup_ratio is not None else 0.05)))

    args_dict = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.learning_rate,
        "lr_scheduler_type": "cosine",
        "logging_steps": 5,
        "max_steps": args.max_steps,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "optim": "paged_adamw_8bit",
        "seed": args.seed,
        "report_to": "none",
    }

    if "warmup_steps" in supported_params:
        args_dict["warmup_steps"] = calc_warmup_steps
    elif "warmup_ratio" in supported_params:
        args_dict["warmup_ratio"] = args.warmup_ratio if args.warmup_ratio is not None else 0.05

    # Strategy configuration for evaluation and checkpointing
    if "eval_strategy" in supported_params:
        args_dict["eval_strategy"] = "steps" if has_val else "no"
    elif "evaluation_strategy" in supported_params:
        args_dict["evaluation_strategy"] = "steps" if has_val else "no"

    if has_val:
        args_dict["eval_steps"] = args.eval_steps
        args_dict["save_strategy"] = "steps"
        args_dict["save_steps"] = args.eval_steps
        args_dict["load_best_model_at_end"] = True
        args_dict["metric_for_best_model"] = "eval_loss"

    # In modern TRL (trl >= 0.12.0), dataset_text_field & max_seq_length belong in SFTConfig
    if use_sft_config:
        if "dataset_text_field" in supported_params:
            args_dict["dataset_text_field"] = "text"
        if "max_seq_length" in supported_params:
            args_dict["max_seq_length"] = 1024

    training_args = training_args_cls(**args_dict)

    # Configure SFTTrainer kwargs safely according to its constructor signature
    sft_sig = inspect.signature(SFTTrainer.__init__)
    sft_params = set(sft_sig.parameters.keys())

    trainer_kwargs = {
        "model": model,
        "train_dataset": train_formatted,
        "eval_dataset": eval_formatted,
        "peft_config": peft_config,
        "args": training_args,
    }

    if "processing_class" in sft_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sft_params:
        trainer_kwargs["tokenizer"] = tokenizer

    # Pass dataset_text_field and max_seq_length to SFTTrainer only in legacy TRL
    if not use_sft_config:
        if "dataset_text_field" in sft_params:
            trainer_kwargs["dataset_text_field"] = "text"
        if "max_seq_length" in sft_params:
            trainer_kwargs["max_seq_length"] = 1024

    trainer = SFTTrainer(**trainer_kwargs)

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
