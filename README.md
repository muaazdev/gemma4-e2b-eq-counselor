# 🧠 Gemma 4 E2B — Empathetic Counseling Assistant

Fine-tuning Google's **Gemma 4 E2B** for emotionally intelligent, bilingual (English/Arabic) mental health counseling support using **QLoRA** on a free Google Colab T4 GPU.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Gemma 4](https://img.shields.io/badge/Model-Gemma%204%20E2B-orange)
![Unsloth](https://img.shields.io/badge/Framework-Unsloth-green)
![License](https://img.shields.io/badge/License-Apache%202.0-red)

---

## 🎯 Project Overview

This project demonstrates an **end-to-end LLM fine-tuning pipeline** — from dataset selection and preprocessing through training, inference, and deployment — targeting the underserved domain of **Arabic-English empathetic counseling**.

### Why This Matters

- **Arabic NLP is a strategic priority** — Saudi Arabia's Vision 2030 and SDAIA are investing heavily in Arabic language AI, yet empathetic/counseling AI remains underexplored in Arabic
- **Gemma 4 E2B supports 140 languages** natively, making it ideal for bilingual deployment
- **Mental health AI** is a growing field with real-world impact — the WHO estimates 1 in 8 people globally live with a mental health condition

### Key Results

| Metric | Value |
|--------|-------|
| Base Model | `google/gemma-4-e2b-it` |
| Trainable Parameters | 31M / 5.15B (0.60%) |
| Training Dataset | MentalChat16K (16,052 samples) |
| Training Time | ~32 min (200 steps, T4 GPU) |
| VRAM Usage | 7.6 GB |
| Languages | English, Arabic (zero-shot cross-lingual) |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────┐
│              Gemma 4 E2B (5.15B params)         │
│              4-bit Quantized (bitsandbytes)     │
├─────────────────────────────────────────────────┤
│  LoRA Adapters (rank=16, alpha=16)              │
│  Target: q/k/v/o_proj, gate/up/down_proj        │
│  Trainable: 31M params (0.60%)                  │
├─────────────────────────────────────────────────┤
│  Training: SFTTrainer + adamw_8bit              │
│  Gradient Checkpointing: Unsloth custom         │
│  Effective Batch Size: 8 (1 × 8 accum)          │
└─────────────────────────────────────────────────┘
```

---

## 📊 Dataset

**[MentalChat16K](https://huggingface.co/datasets/ShenLab/MentalChat16K)** — A benchmark dataset for conversational mental health assistance.

| Component | Samples | Source |
|-----------|---------|--------|
| Synthetic Counseling | 9,774 | GPT-3.5 Turbo generated, 33 mental health topics |
| Clinical Interviews | 6,278 | Real anonymized transcripts (PISCES clinical trial) |
| **Total** | **16,052** | After filtering (min 20 chars per field) |

**Topics covered:** Anxiety, Depression, Relationships, Grief, Substance Abuse, Family Conflict, Intimacy, and 26 more.

**Data format** — Converted to Gemma 4 chat template:
```
<bos><|turn>system
You are a compassionate and emotionally intelligent counseling assistant...
<|turn>user
[Client message]
<|turn>model
[Counselor response]
```

---

## 🚀 Quick Start

### Run the Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/gemma4-e2b-eq-counselor/blob/main/gemma4_e2b_eq_finetune.ipynb)


### Load the Model from HuggingFace

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="muaaz/gemma4-e2b-eq-counselor",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

messages = [
    {"role": "system", "content": [{"type": "text", "text": "You are a compassionate counseling assistant."}]},
    {"role": "user", "content": [{"type": "text", "text": "I've been feeling anxious about my future."}]},
]

inputs = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
).to(model.device)

outputs = model.generate(input_ids=inputs, max_new_tokens=512, temperature=0.7)
print(tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True))
```

---

## 💬 Sample Outputs

### English
**User:** *I've been feeling really anxious about starting a new job. I can't sleep and I keep thinking about everything that could go wrong.*

**Assistant:** *I completely understand how overwhelming this must feel. Anxiety about new beginnings is incredibly common, and it shows that you care deeply about doing well. Let me share some strategies that might help...*

### Arabic
**User:** *أشعر بالقلق الشديد من المستقبل ولا أستطيع النوم. ماذا أفعل؟*

**Assistant:** *أتفهم تمامًا مدى صعوبة هذا الشعور. القلق بشأن المستقبل وصعوبة النوم يمكن أن يكونا مرهقين جدًا. أريدك أن تعرفي أن مشاعرك هذه طبيعية جدًا...*

> The model provides detailed, culturally appropriate Arabic responses with grounding techniques (5-4-3-2-1), self-compassion exercises, and validation — leveraging Gemma 4's native multilingual capability.

---

## 📁 Repository Structure

```
gemma4-e2b-eq-counselor/
├── README.md                          # This file
├── gemma4_e2b_eq_finetune.ipynb       # Complete Colab training notebook
├── config/
│   └── training_config.yaml           # Hyperparameters reference
├── LICENSE                            # Apache 2.0
└── .gitignore
```

---

## 🔧 Training Details

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| LoRA rank (r) | 16 | Sweet spot for E2B — enough capacity without overfitting |
| LoRA alpha | 16 | alpha = r → scaling factor 1.0 |
| Learning rate | 5e-5 | Lower than typical 2e-4 to prevent NaN on T4 float32 |
| Batch size | 1 × 8 accum | Memory-constrained on T4 16GB |
| Optimizer | adamw_8bit | Saves ~30% VRAM vs standard AdamW |
| Gradient clipping | 0.3 | Tighter clipping for 4-bit stability |
| Scheduler | Cosine | Standard for short fine-tuning runs |
| Precision | float32 | T4 fallback (no bf16 support for this model) |

### Training Observations

- **Loss plateau at ~12.36** — Identified as a chat template tokenization mismatch between `apply_chat_template` output and the model's expected format during training. The Gemma 4 Processor handles multimodal content types, which creates friction with text-only SFT pipelines.
- **Mitigation path**: Use raw text formatting instead of `apply_chat_template`, or switch to `FastVisionModel` which handles the Processor correctly.
- **Key insight**: Gemma 4 E2B's base model already demonstrates strong empathetic counseling capability in both English and Arabic — the zero-shot cross-lingual transfer is remarkably good, suggesting the pre-training data includes substantial counseling/empathy-related content.

---

## 🔮 Future Improvements

- [ ] Fix chat template alignment for effective fine-tuning (use raw template strings)
- [ ] Train on A100 with bf16 for stable, faster convergence
- [ ] Add Arabic counseling datasets (e.g., Arabic Empathetic Chatbot corpus)
- [ ] Implement GRPO/DPO for preference-based alignment
- [ ] Evaluate with MentalChat16K's 7-metric benchmark (empathy, relevance, safety, etc.)
- [ ] Deploy via Ollama with GGUF export for local inference

---

## ⚠️ Disclaimer

This model is for **research and educational purposes only**. It is **not a substitute for professional mental health care**. If you or someone you know is in crisis, please contact a licensed mental health professional or a crisis helpline.

---

## 📚 References

- [Gemma 4 E2B](https://ai.google.dev/gemma) — Google DeepMind
- [MentalChat16K](https://huggingface.co/datasets/ShenLab/MentalChat16K) — Xu et al., 2025
- [Unsloth](https://github.com/unslothai/unsloth) — 2x faster fine-tuning
- [LoRA](https://arxiv.org/abs/2106.09685) — Hu et al., 2021
- [QLoRA](https://arxiv.org/abs/2305.14314) — Dettmers et al., 2023

---

## 📄 License

This project is licensed under the [Apache License 2.0](LICENSE).
