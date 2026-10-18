---
language:
- en
license: apache-2.0
base_model: Qwen/Qwen2.5-3B-Instruct
tags:
- mendseed
- public-grievances
- qlora
- peft
- hackathons
- quietflay
- pydantic
pipeline_tag: text-generation
---

# MendSeed AI — Qwen2.5-3B Grievance Intelligence Adapter

This repository contains the fine-tuned PEFT LoRA adapter for **`Qwen/Qwen2.5-3B-Instruct`**, customized for **MendSeed**—a public grievance observatory and real-world problem discovery platform.

MendSeed structures ground-level citizen grievances across key sectors (agriculture, civic infrastructure, healthcare, education, logistics, sanitation) into actionable problem briefs for builders, researchers, and hackathon teams.

---

## Model Details

- **Base Model:** `Qwen/Qwen2.5-3B-Instruct` (Apache 2.0)
- **Adapter Type:** PEFT QLoRA (Rank $r=16$, Alpha $\alpha=32$, Dropout $0.05$)
- **Target Modules:** All 7 linear projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`)
- **Quantization:** 4-bit NormalFloat (NF4) with Double Quantization (`float16` compute)
- **Primary Schema Outputs:**
  1. `GrievanceAnalysis` (Structured parameter extraction from citizen voice)
  2. `BuilderBrief` (Actionable problem briefs for hackathons and project builders)
  3. `GrievanceComparison` (Pairwise root-cause comparison)
  4. `CommunitySummary` (Sectoral voice synthesis)
  5. `CanonicalClusterProposal` (Human-moderated problem clustering)

---

## Benchmark Evaluation Results

| Benchmark Metric | Score | Assessment |
|---|---|---|
| **Pydantic Schema Integrity** | **100.0%** | Zero schema crashes; strict JSON adherence |
| **Adversarial Robustness** | **100.0%** | Rejects prompt injections and sanitizes noise |
| **QuietFlay v0.1 NDCG@10 (Chronic Need)** | **0.931** | Successfully elevates persistent suffering (vs 0.180 for raw votes) |
| **QuietFlay v0.1 NDCG@10 (Sybil Burst)** | **1.000** | Completely suppresses 100+ bot upvotes within 15 minutes |
| **Optimal Checkpoint** | **Step 75** | Global validation loss minimum ($\approx 0.587$); early stopping to prevent memorization |

---

## Quick Inference Example

```python
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

BASE_ID = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_ID = "itsjustayush/mendseed-qwen2.5-3b-lora"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(BASE_ID)
base_model = AutoModelForCausalLM.from_pretrained(BASE_ID, quantization_config=bnb_config, device_map="auto")
model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
model.eval()

messages = [
    {
        "role": "system",
        "content": "You are MendSeed AI, an issue-intelligence assistant for a public grievance observatory. Return ONLY valid JSON matching the GrievanceAnalysis schema."
    },
    {
        "role": "user",
        "content": "In Kolar district, tomato farmers lose 40% of their harvest due to zero cold-chain transit to mandis."
    }
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=300, do_sample=False)

print(tokenizer.decode(out[0][len(inputs.input_ids[0]):], skip_special_tokens=True))
```

---

## License

Apache License 2.0. Open for community and commercial use.
