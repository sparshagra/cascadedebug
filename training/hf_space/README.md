---
title: CascadeDebug Training
emoji: 🔬
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.25.0
app_file: app.py
pinned: true
hardware: l4x1
license: mit
---

# 🔬 CascadeDebug — GRPO Training Space

This HuggingFace Space runs **GRPO training** for the CascadeDebug RL environment.

- **Model:** `Qwen2.5-7B-Instruct` (4-bit via Unsloth)
- **Algorithm:** GRPO (Group Relative Policy Optimization)
- **Task:** Multi-agent pipeline fault localization
- **GPU:** L4 (24GB VRAM, Ada Lovelace)

Training starts automatically when the Space boots.
Results are pushed to `Dikshita2026/cascadedebug` upon completion.
