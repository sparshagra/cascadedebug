---
title: CascadeDebug
emoji: 🔍
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: bsd-3-clause
---

# CascadeDebug 🔍

> **First RL training environment for multi-agent pipeline fault localization**

[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-blue)](https://huggingface.co/spaces/Dikshita2026/cascadedebug)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-green)](https://github.com/meta-pytorch/OpenEnv)

---

## Problem

Modern AI deployments chain multiple LLM calls together — one agent researches, another writes code, another validates. A single bad output at step 2 of a 6-step pipeline doesn't just cause step 2 to fail. It silently corrupts every downstream step. By the time the final output is wrong, the error has been **amplified, reinterpreted, and baked into four subsequent decisions**.

**NeurIPS 2025 (MAST-Data):** Multi-agent pipelines amplify errors up to **17.2×** vs single-agent baselines. Failure rate: **41–86.7%** across 7 state-of-the-art open-source systems.

**The gap:** Benchmarks exist to evaluate this failure mode (MAST-Data). No open RL environment exists to *train* a model to fix it.

---

## Solution: CascadeDebug

CascadeDebug is the first RL training environment that:

1. **Simulates** a 3–6 step professional pipeline (Researcher → Coder → Analyst)
2. **Injects** one silent error at a uniformly random step each episode
3. **Rewards** the agent for:
   - Localizing the exact faulty step
   - Attributing blame to the correct role
   - Negotiating a fix with a deterministic gatekeeper
   - Surgically repairing *only* the faulty step (no full restart)

**Themes covered:** Theme 1 (multi-agent negotiation), Theme 2 (long-horizon recovery), Theme 3.1 (professional tool-using pipeline)

---

## Environment

### What the agent sees
```json
{
  "pipeline_id": "ep_042",
  "steps": [
    {"role": "Researcher", "output": "Python uses 1-based indexing.", "step_id": 1},
    {"role": "Coder",       "output": "for i in range(1, n+1): ...", "step_id": 2},
    {"role": "Analyst",     "output": "Code output validated.",      "step_id": 3}
  ],
  "task_brief": "Explain Python indexing and benchmark a loop.",
  "turn": 1,
  "gatekeeper_feedback": null
}
```

### What the agent must do
```json
{
  "fault_step_id": 1,
  "blame_role": "Researcher",
  "fix_content": "Python uses 0-based indexing. Index 0 is the first element.",
  "action_type": "propose"
}
```

### Reward Functions (4 independent signals)
| Signal | Weight | Description |
|--------|--------|-------------|
| Fault Localization | 0.35 | `fault_step_id == injected_step` |
| Blame Attribution | 0.20 | `blame_role == injected_role` |
| Fix Correctness | 0.35 | Verifier score on `fix_content` |
| Surgical Precision | 0.10 | No unmodified steps changed + turn efficiency |

### Curriculum
| Level | Episodes | Pipeline Length | Error Types |
|-------|----------|-----------------|-------------|
| 1 | 0–500 | 3 steps | Obvious (wrong type, false fact) |
| 2 | 500–1500 | 4 steps | Subtle (off-by-one, embedded false claim) |
| 3 | 1500+ | 5-6 steps | Cross-step dependency errors |

---

## Results

> 📊 Training results will be added here after Phase 7 (onsite, 25–26 April 2026).

| Metric | Untrained Baseline | After Level 1 | After Level 2 |
|--------|-------------------|---------------|---------------|
| Avg total reward | ~0.10–0.20 | ~0.40–0.55 | ~0.55–0.70 |
| Localization accuracy | ~0.33 (random) | ~0.60 | ~0.70 |
| Gatekeeper accept rate | ~0.20 | ~0.50 | ~0.65 |

**Reward curve:** *(image to be added after training)*

**Baseline vs trained:** *(image to be added after training)*

---

## Links

| Resource | URL |
|----------|-----|
| 🤗 HuggingFace Space | https://huggingface.co/spaces/Dikshita2026/cascadedebug |
| 📓 Colab Training Notebook | *TBD — Phase 5* |
| 🎬 YouTube Demo (<2 min) | *TBD — Phase 9* |
| 📝 HuggingFace Mini-Blog | *TBD — Phase 9* |

---

## Structure

```
cascade_debug/
├── openenv.yaml              # OpenEnv manifest (3 tasks + graders)
├── README.md                 # This file
├── inference.py              # Baseline agent (OpenEnv-compliant logging)
├── graders.py                # Deterministic graders for Phase 2 compliance
├── models.py                 # Action, Observation dataclasses
├── client.py                 # CascadeDebugEnv (EnvClient — no server imports)
├── CASCADE_DEBUG_CONTEXT.md  # AI agent handoff doc — read before coding
├── server/
│   ├── cascade_debug_environment.py  # Environment logic (reset/step/state)
│   ├── app.py                         # FastAPI server
│   ├── Dockerfile                     # HF Spaces deployment
│   └── requirements.txt
├── data/
│   ├── pipeline_bank.json             # 1000 pre-computed episodes (Phase 1)
│   └── generate_pipeline_bank.py      # Offline generation script
├── training/
│   └── train_grpo.ipynb               # GRPO training notebook (Phase 5)
└── results/
    ├── reward_curve.png               # Main hero plot
    ├── baseline_vs_trained.png        # Comparison plot
    └── component_rewards.png          # 4 reward signals over training
```

---

## Quick Start (Local)

```bash
# Install
pip install openenv-core
pip install -e .

# Run local server
cd server && uvicorn app:app --host 0.0.0.0 --port 7860

# Run baseline inference
TASK_NAME=localize_level1 python inference.py
TASK_NAME=all python inference.py
```

---

## Anti-Hacking Design

- **Injection is uniformly random** — cannot game by always guessing step N
- **Gatekeeper is deterministic rule-based** — not an LLM (no oracle exploitation)
- **Pipeline bank is pre-computed** — no LLM at training time (deterministic, reproducible)
- **Code execution in sandboxed subprocess** — hard 5s timeout, no raw exec()
- **No LLM-as-judge** — all verifiers are programmatic

---

*CascadeDebug | OpenEnv Hackathon 2026 | Meta × HuggingFace × PyTorch*
