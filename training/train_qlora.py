"""
Production QLoRA Fine-Tuning Script for Qwen/Qwen2.5-3B-Instruct.
Optimized for NVIDIA Tesla T4 GPU (Turing architecture, sm_75) on Google Colab or Kaggle.

Key Settings for T4:
- Quantization: 4-bit NF4 with double quantization (BitsAndBytesConfig)
- Compute Dtype: torch.float16 (T4 does not have native bfloat16 tensor cores)
- Optimizer: paged_adamw_8bit (prevents out-of-memory spikes)
- Target Modules: All 7 linear projection layers of Qwen's transformer blocks
- Artifact: Uploads ONLY the lightweight LoRA adapter (~45 MB) to private Hugging Face repo
"""

import os
import torch
from datasets import load_dataset
from huggingface_hub import login
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

# 1. Configuration & Security
MODEL_ID = os.getenv("BASE_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
ADAPTER_REPO = os.getenv("ADAPTER_REPO", "itsjustayush/mendseed-qwen2.5-3b-lora")
DATA_FILE = os.getenv("TRAINING_DATA", "mendseed_training_data.jsonl")

# Authenticate with Hugging Face token
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    try:
        from google.colab import userdata
        hf_token = userdata.get("HF_TOKEN")
    except Exception:
        pass

if hf_token:
    login(token=hf_token)
    print("Authenticated successfully with Hugging Face Hub.")
else:
    print("Warning: HF_TOKEN not found in environment or Colab secrets.")


def train():
    print(f"Initializing QLoRA fine-tuning for {MODEL_ID} on T4 GPU...")

    # 2. 4-bit NF4 Configuration with float16 compute for Tesla T4
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading base model in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    # 3. LoRA configuration targeting all linear layers for strict JSON schema output
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
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 4. Ingest and format dataset using Qwen's ChatML template
    print(f"Loading training data from {DATA_FILE}...")
    dataset = load_dataset("json", data_files=DATA_FILE, split="train")

    def format_chat(row):
        messages = [
            {"role": "system", "content": row["system"]},
            {"role": "user", "content": row["input"]},
            {"role": "assistant", "content": row["output"]}
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    formatted_dataset = dataset.map(format_chat)

    # 5. Training Arguments configured for T4 FP16 and memory efficiency
    training_args = TrainingArguments(
        output_dir="./mendseed-qlora-checkpoints",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        max_steps=100,
        fp16=True,
        bf16=False,
        optim="paged_adamw_8bit",
        save_strategy="no",
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args,
    )

    print("Starting QLoRA training loop...")
    trainer.train()

    # 6. Upload only the lightweight (~45 MB) LoRA adapter
    print(f"Uploading private LoRA adapter to Hugging Face: {ADAPTER_REPO}...")
    trainer.model.push_to_hub(ADAPTER_REPO, private=True)
    tokenizer.push_to_hub(ADAPTER_REPO, private=True)
    print("Fine-tuning complete! Weights safely uploaded to Hugging Face.")


if __name__ == "__main__":
    train()
