---
title: CascadeDebug GRPO Training
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
suggested_hardware: l4x1
license: mit
---

# CascadeDebug — GRPO Training (Docker Space)

This Space uses the **Docker SDK** (not Gradio SDK) so we control the runtime:

- **Python 3.11** (Gradio SDK on HF was on **3.13**, which breaks many Unsloth / Triton / bitsandbytes paths)
- **CUDA 12.1** base image + **PyTorch 2.5.1+cu121**
- **Unsloth `2025.11.4`** with extra **`[cu121-torch251]`** so xformers, bitsandbytes, and triton match the torch build

Training still starts from `app.py` (Gradio monitor); the image runs `python3.11 app.py`.

- **Model:** Qwen2.5-3B-Instruct (4-bit via Unsloth)
- **Algorithm:** GRPO
- **GPU:** L4 recommended (`suggested_hardware: l4x1`)

Set **`HF_TOKEN`** in Space → Settings → Repository secrets if you want Hub push of results.

### If training still hits dtype errors

Add a secret or env var **`UNSLOTH_FORCE_FLOAT32=1`** — slower, but forces Unsloth off the fp16 fast-LORA autocast path.
