---
title: CascadeDebug GRPO Training
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.25.0
app_file: app.py
pinned: false
hardware: l4x1
license: mit
---

# CascadeDebug — GRPO Training Space

This Space runs GRPO training for the CascadeDebug RL environment.

- **Model:** Qwen2.5-3B-Instruct (4-bit via Unsloth)
- **Algorithm:** GRPO (Group Relative Policy Optimization)
- **Steps:** 300 | **Group size:** 4
- **GPU:** L4 (24 GB VRAM)

Training starts automatically on boot. Reward plots refresh every 10 seconds.
Results are pushed to `Dikshita2026/cascadedebug` on completion.
