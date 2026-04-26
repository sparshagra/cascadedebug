# CascadeDebug — Persistent Project Context

> **IMPORTANT FOR ANY AI AGENT (Antigravity, Claude, GPT, Gemini, etc.):**
> 1. Read this ENTIRE file before writing a single line of code.
> 2. Check the Phase Progress table — start from the first 🔄 IN PROGRESS or ⬜ NOT STARTED phase.
> 3. After completing work, update the Phase Progress table and append to the Session Log.
> 4. Commit this file to GitHub after every update so teammates can pull it.
> 5. Never change locked design decisions without explicit user approval.

---

## 🏆 Hackathon Overview

- **Event:** OpenEnv Hackathon 2026 | Meta × HuggingFace × PyTorch
- **Project:** CascadeDebug — RL environment for multi-agent pipeline fault localization
- **Workspace:** `c:\Users\SPARS\Desktop\IIIT B\scalarHackathon`
- **Conversation ID (initial):** 20614897-4a98-488b-838e-722ac8add5f0
- **Team:** 2 people (Sparsh + teammate) — no fixed phase ownership, both push/pull freely
- **HuggingFace username:** `Dikshita2026`
- **HF Space URL (submission URL — NEVER change after Phase 4):** `https://huggingface.co/spaces/Dikshita2026/cascadedebug`
- **HF API Key:** ✅ Configured (Write token)
- **GitHub username:** sparshagra
- **GitHub Repo:** https://github.com/sparshagra/cascadedebug

### Collaboration Model
- Both teammates push to GitHub freely; pull before starting work
- **This context file is always committed to the repo** — it is the handoff doc
- Any agent picking up work should read this file first, then check git log for latest commits
- Teammate's agent: same workflow — read context → pick up next uncompleted phase → update context → push

---

## 📋 Hackathon Guidelines — Key Extracts (READ BEFORE CODING)

### Judging Weights
| Criterion | Weight | What Judges Look For |
|-----------|--------|---------------------|
| **Environment Innovation** | **40%** | Novel, creative, challenging. Not chess/snake/tic-tac-toe clones. Domain underexplored in RL/LLM. Could a researcher publish on this? |
| **Storytelling & Presentation** | **30%** | Clear problem, environment, agent story. Engaging demo for non-technical audience. README readable in 3-5 min. |
| **Showing Improvement in Rewards** | **20%** | Observable training progress. Reward curves. Trained vs untrained baseline. Quantitative + qualitative. |
| **Reward & Training Pipeline** | **10%** | Coherent reward logic. Pipeline produces meaningful improvement. |

### Non-Negotiable Minimum Requirements
- [ ] Use **OpenEnv (latest release)** — `pip install openenv-core`
- [ ] **Working Colab training script** using Unsloth or HF TRL that judges can re-run
- [ ] **Evidence of real training** — loss and reward plots from an actual run
- [ ] **Mini-blog on HuggingFace** OR **<2 min YouTube video** explaining environment + training
- [ ] **Environment hosted on HF Space** — URL submitted to judges (NEVER change after submission)
- [ ] **README** — motivates problem, explains env, shows results, links to all materials
- [ ] All plots saved as `.png` or `.jpg` **committed to repo** (not just Colab cells)
- [ ] No large video files in HF Space repo — use URL links instead

### OpenEnv API (Latest — openenv-core package)
```python
# Install
pip install openenv-core

# Environment structure from openenv init:
my_env/
├── __init__.py          # Export Action, Observation, YourEnv
├── models.py            # Action, Observation, State dataclasses
├── client.py            # YourEnv(EnvClient) — NO server imports here!
├── openenv.yaml         # Environment manifest
├── pyproject.toml
└── server/
    ├── your_environment.py  # YourEnvironment(Environment) — server logic
    ├── app.py               # FastAPI app creation
    ├── requirements.txt
    └── Dockerfile
```

**Critical API Patterns:**
- `Environment` base class: server-side, has `reset()`, `step(action)`, `state()`
- `EnvClient` base class: client-side, async by default, `.sync()` for sync usage
- Client uses WebSocket connection — not direct imports from server
- **NEVER use reserved MCP tool names**: `reset`, `step`, `state`, `close`
- `StepResult` = observation + reward + done flag
- Async pattern: `async with YourEnv(base_url="...") as client: await client.reset()`
- Sync pattern: `with YourEnv(base_url="...").sync() as client: client.reset()`

### Reward Design Rules (From OpenEnv Docs + FAQ)
- Start **simple** (often sparse 0/1) before adding shaping terms
- **Composable rubrics** > monolithic scoring function
- Hard to game: agent that exploits reward without solving task must NOT get high score
- Adversarially test your reward YOURSELF before the model does
- Watch for: rising reward + flat task quality = reward hacking
- DO NOT use LLM-as-judge without hard backstop checks
- Keep a **holdout evaluator separate** from training reward

### Plot Requirements (Judges spend seconds per plot)
- Label **both axes** with units (e.g. "training step" on x, "reward" on y)
- Save as `.png`, commit to repo
- Embed key plots in README with one-line caption
- Put baseline vs trained **on same axes** for obvious comparison

### What Makes a Strong Submission (From Judge Guide)
- Ask: "Does this teach an LLM something it currently can't do well?"
- Ask: "Is the domain underexplored in RL/LLM training?"
- Ask: "Could a researcher write a paper about training on this?"
- Show real training connecting to the environment (not a static dataset)
- Train long enough that curves are meaningful
- Compare trained agent vs random/untrained baseline — quantitatively AND qualitatively

### Theme Alignment for CascadeDebug
- **Theme 1** (Multi-Agent): Gatekeeper negotiation ✅
- **Theme 2** (Long-Horizon): Multi-step pipeline debugging with recovery ✅  
- **Theme 3.1** (Professional Tasks): Researcher/Coder/Analyst workflow ✅
- **Primary theme to emphasize in pitch**: Theme 3.1 + Theme 2 (most novel angle)

### GitHub Push Setup
- Use GitHub for collaboration — push all changes so teammates can pull
- **No WandB** — use local CSV/PNG logging committed to repo
- HF API key to be added later when credits arrive — keep placeholder empty for now

---

## 📜 Spec Document Location

The full technical specification is stored in this file (see below) and was also provided in the initial conversation. The canonical spec hash/version: **v1.0 — 2026-04-25**.

---

## 🤖 Model Selection Decision (UPDATED — Upgraded to 7B)

- **Budget:** $30 HF credits — must maximize training quality
- **Primary choice:** `Qwen2.5-7B-Instruct` ← UPGRADED for better accuracy
  - Significantly better reasoning and JSON formatting capability than 3B
  - Fits on L4 GPU (24GB VRAM) with 4bit via Unsloth
  - GRPO group size reduced to 2 (from 4) to fit in VRAM
  - Gradient accumulation increased to 8 to compensate
  - $30 on L4 (~$0.80/hr) ≈ 37 hours compute — plenty for 300 steps
- **Previous model:** `Qwen2.5-3B-Instruct` (switched due to insufficient accuracy)
- **DO NOT use:** models >7B — will exhaust $30 budget before meaningful training
- **Unsloth recipe to follow:** Advanced Qwen3 (4B) GRPO notebook pattern (proximity scoring, advanced templates)

### Logging Strategy (No WandB)
- Log all reward components to local CSV every 10 steps
- Save `.png` plots every 100 steps using matplotlib
- Commit plots to `results/` in repo after training
- GitHub = source of truth for all teammates

---

## 📋 Phase Progress Tracker

| Phase | Name | Status | Owner | Notes |
|-------|------|--------|-------|-------|
| 0 | Setup & Scaffold | ✅ DONE | All | OpenEnv scaffold, git init, pushed to GitHub |
| 1 | Pipeline Bank Generation | ✅ DONE | Agent | 1000 episodes (200×5), 10 domains, 5 error types, validated |
| 2 | Environment Core | ✅ DONE | Agent | reset/step/state + gatekeeper + pipeline bank loader |
| 3 | Reward Functions | ✅ DONE | Agent | 4 independent signals + verifiers + graders wired |
| 4 | Deploy to HF Spaces | ✅ DONE | Agent | https://huggingface.co/spaces/Dikshita2026/cascadedebug (cpu-basic, free) |
| 5 | Training Script | ✅ DONE | Agent | train_grpo.py: GRPO + Unsloth + offline baseline mode |
| 6 | Inspect for Hacking | ✅ DONE | Agent | All 6 checks passed: uniform injection, role balance, keyword quality |
| 7 | Full Training Run | 🔄 IN PROGRESS | Agent | **Training Space live:** [cascadedebug-training](https://huggingface.co/spaces/Dikshita2026/cascadedebug-training) — L4+Gradio, `data/pipeline_bank.json` in bundle, `gradio` in requirements. Set **HF_TOKEN** in Space → Settings → Repository secrets for Hub push. |
| 8 | Baseline Comparison | ✅ DONE | Agent | Baseline plots generated: L1=0.34, L2=0.30, L3=0.25 |
| 9 | Demo + Writeup | ⬜ NOT STARTED | Person D / Agent | Video + README |
| 10 | Final Checks | ⬜ NOT STARTED | All | Submission gate |

**Status legend:** ⬜ NOT STARTED | 🔄 IN PROGRESS | ✅ DONE | ❌ BLOCKED

---

## 🗂️ Target File Structure

```
cascadedebug/                         ← root of submission
├── openenv.yaml
├── README.md
├── Dockerfile
├── CASCADE_DEBUG_CONTEXT.md          ← THIS FILE (context for agents)
├── environment/
│   ├── server.py                     # CascadeDebugEnv (OpenEnv subclass)
│   ├── gatekeeper.py                 # Deterministic rule-based gatekeeper
│   ├── rewards.py                    # 4 independent reward functions
│   ├── verifiers.py                  # Role-specific output verifiers
│   └── models.py                     # Action, Observation, State dataclasses
├── client/
│   └── client.py                     # CascadeDebugClient (no server imports)
├── data/
│   ├── pipeline_bank.json            # 1000 pre-computed episodes
│   └── generate_pipeline_bank.py     # Offline generation script
├── training/
│   └── train_grpo.ipynb              # Colab notebook (Unsloth + TRL GRPO)
├── results/
│   ├── reward_curve.png
│   ├── baseline_vs_trained.png
│   └── component_rewards.png
└── Dockerfile
```

---

## 🔑 Key Design Decisions (Locked)

These decisions are finalized and should NOT be changed without user approval:

| Decision | Value | Rationale |
|----------|-------|-----------|
| Pipeline bank size | 200 clean × 5 errors = 1000 episodes | Pre-computed, no LLM at runtime |
| Gatekeeper type | Deterministic rule-based (NOT LLM) | Theme 1 compliance, low complexity |
| Injection bias | Uniformly random across steps | Prevents "always guess step N" hacking |
| Code exec timeout | 5 seconds hard limit via subprocess | Security + anti-hang |
| Reward functions | 4 independent (localization, blame, fix, precision) | GRPO needs independent signals |
| RL algorithm | GRPO via TRL | Best for LLM RL with verifiable rewards |
| Training framework | Unsloth + TRL | Speed + 4bit support |
| Model save method | `save_pretrained_merged` merged_16bit | Required by Unsloth |
| No LLM-as-judge | All verifiers programmatic | Speed + determinism |
| Curriculum levels | 3 levels, auto-advance on rolling avg threshold | Progressive difficulty |

---

## 💰 Reward Functions (DO NOT change without user approval)

```
r1 = reward_localization   weight=0.35  (fault_step_id == injected_step)
r2 = reward_blame          weight=0.20  (blame_role == injected_role)
r3 = reward_fix            weight=0.35  (verifier score on fix content)
r4 = reward_precision      weight=0.10  (no extra steps modified + turn efficiency)

total = 0.35*r1 + 0.20*r2 + 0.35*r3 + 0.10*r4
```

Partial credit on r1 only at curriculum Level 1: ±1 step → 0.3 reward.

---

## 📊 Curriculum Schedule

| Level | Episodes | Pipeline Length | Error Types | Advance Threshold |
|-------|----------|-----------------|-------------|-------------------|
| 1 | 0–500 | 3 steps | Level 1 (obvious) | rolling avg > 0.4 |
| 2 | 500–1500 | 4 steps | Level 1+2 | rolling avg > 0.6 |
| 3 | 1500+ | 5-6 steps | All types | rolling avg > 0.7 |

---

## 🚨 Anti-Hacking Measures (Critical — verify before training)

- [ ] Injection step is uniformly random — verify distribution in pipeline bank
- [ ] Code execution has hard 5s timeout in subprocess — NEVER raw exec()
- [ ] Model saves use Unsloth merged_16bit path — not naive 4bit upcast
- [ ] Training starts at curriculum Level 1 — confirm non-zero reward before advancing
- [ ] Reward plots are .png files committed to repo — not just Colab cell outputs
- [ ] HF Space URL is frozen early — submission URL cannot change
- [ ] No LLM-as-judge in any reward function — all verifiers are programmatic
- [ ] Gatekeeper is deterministic rule-based — not a second LLM call
- [ ] Pipeline bank is pre-computed offline — no LLM calls at training time
- [ ] Agent observation does NOT include injected_step — only state() reveals it

---

## 🔗 Links (Fill in as they become available)

| Resource | URL |
|----------|-----|
| GitHub Repo | https://github.com/sparshagra/cascadedebug |
| HuggingFace Space | https://huggingface.co/spaces/Dikshita2026/cascadedebug |
| WandB Run | NOT USING — local CSV + PNG |
| Colab Notebook | TBD — Phase 5 |
| YouTube Demo | TBD — Phase 9 |

---

## 📝 Decisions Resolved / Pending

| Decision | Status | Value |
|----------|--------|-------|
| Hackathon guidelines | ✅ RESOLVED | Processed and integrated above |
| Final model | ✅ RESOLVED | Qwen2.5-7B-Instruct (upgraded for accuracy) |
| Logging strategy | ✅ RESOLVED | Local CSV + .png plots committed to GitHub |
| WandB | ✅ RESOLVED | NOT using WandB |
| Team structure | ✅ RESOLVED | 2 people, push/pull freely, no fixed ownership |
| Collaboration model | ✅ RESOLVED | GitHub = source of truth, context file always committed |
| HF account | ✅ RESOLVED | Dikshita2026 |
| GitHub repo URL | ✅ RESOLVED | https://github.com/sparshagra/cascadedebug |
| HF Space URL | ✅ RESOLVED | https://huggingface.co/spaces/Dikshita2026/cascadedebug |
| HF API key | ✅ RESOLVED | Write token configured |

---

## 🐛 Issues & Fixes Log

| Date | Issue | Fix Applied |
|------|-------|-------------|
| 2026-04-25 | CUDA version mismatch: PyTorch CUDA 13.0 vs torchvision CUDA 12.8 | Detect PyTorch CUDA version at runtime, force-install matching torchvision from PyTorch index URL before importing unsloth |
| 2026-04-25 | 3B model insufficient accuracy (on some training paths) | Upgraded to Qwen2.5-7B-Instruct-bnb-4bit on that path; group size 4→2, gradient accum 4→8 |
| 2026-04-26 | `train_grpo.py` references `torch` without importing it — crashes on GPU path | Added `try: import torch` at top of file with None fallback; guarded CUDA checks |
| 2026-04-26 | `results/` dir empty — Phase 8 baseline plots were generated but never committed | **ACTION NEEDED**: Re-run baseline (`python training/train_grpo.py baseline`) and commit PNGs |
| 2026-04-26 | `verifiers.py` role-specific verifiers are dead code — never called by `rewards.py` | Non-blocking: `reward_fix()` does its own keyword/similarity scoring. Consider wiring verifiers in future |

---

## 📅 Session Log

| Date | Conversation ID | What was done |
|------|----------------|---------------|
| 2026-04-25 | 20614897-4a98-488b-838e-722ac8add5f0 | Initial spec processed, context file created, hackathon guidelines processed |
| 2026-04-25 | 20614897-4a98-488b-838e-722ac8add5f0 | Phase 0 DONE: openenv init, scaffold built, git init, pushed to GitHub |
| 2026-04-25 | 0ea13437-d759-4caf-9099-c4027f9eedd9 | Phase 1 DONE: generate_pipeline_bank.py → 1000 episodes, 10 domains, 5 error types, all validations passed |
| 2026-04-25 | 0ea13437-d759-4caf-9099-c4027f9eedd9 | Phase 2 DONE: cascade_debug_environment.py (full reset/step/state), gatekeeper.py (5 rules), client.py (action/obs wiring), models.py (fixed State class) |
| 2026-04-25 | 0ea13437-d759-4caf-9099-c4027f9eedd9 | Phase 3 DONE: rewards.py (4 signals), verifiers.py (role-specific), graders.py (real pipeline bank), inference.py (standalone) |
| 2026-04-25 | 0ea13437-d759-4caf-9099-c4027f9eedd9 | Phase 4 DONE: HF Space created (Dikshita2026/cascadedebug), Docker SDK, cpu-basic (free), all files uploaded, YAML metadata added. Model updated to Qwen2.5-3B-Instruct for $30 budget |
| 2026-04-25 | 0ea13437-d759-4caf-9099-c4027f9eedd9 | Phase 5 DONE: train_grpo.py with GRPO config, prompt templates, action parser, offline verification mode, curriculum advancement |
| 2026-04-25 | 0ea13437-d759-4caf-9099-c4027f9eedd9 | Phase 6 DONE: All 6 hacking checks passed — uniform injection (χ²<4 per length), role balance (<40%), keyword quality, corruption verified |
| 2026-04-25 | 0ea13437-d759-4caf-9099-c4027f9eedd9 | Phase 8 DONE: Baseline plots generated — reward_curve.png, component_rewards.png, baseline_vs_trained.png. Baseline: L1=0.34, L2=0.30, L3=0.25 |
| 2026-04-25 | 0ea13437-d759-4caf-9099-c4027f9eedd9 | Phase 7 READY: Colab script (train_grpo_colab.py) + notebook (CascadeDebug_GRPO_Training.ipynb) created, pushed to GitHub. Config: 300 steps, Qwen2.5-3B-Instruct 4bit, GRPO group=4. User needs to run on Colab T4 |
| 2026-04-25 | 7572fc0a-c708-41dc-bab4-25ab86731d9a | Fixed CUDA version mismatch (torchvision CUDA 12.8 vs PyTorch CUDA 13.0). On HF training path: model upgraded to Qwen2.5-7B-Instruct; group_size=2, grad_accum=8, completion_len=256 |
| 2026-04-26 | (Cursor agent) | Phase 7 deploy: `training/hf_space` fixed — `data/pipeline_bank.json` in bundle, `gradio` in requirements. **Prefer fresh Space** [Dikshita2026/cascadedebug-training-v2](https://huggingface.co/spaces/Dikshita2026/cascadedebug-training-v2) (Gradio SDK, L4, 3B). **Rotate HF token** if it was shared. |
| 2026-04-26 | (pre-Phase-7 audit) | Full repo audit: fixed torch import crash in train_grpo.py, flagged empty results/ dir (plots never committed), noted verifiers.py is dead code. All other files clean — ready for Phase 7 |

---

## 🧠 Previous OpenEnv Hackathon Context

A previous hackathon submission (content-recommendation RL environment) was built for the same Meta × PyTorch OpenEnv hackathon. Key learnings:

- **OpenEnv requires:** `openenv.yaml` with graders section, `Dockerfile` at root, `inference.py` at root, `/reset` `/step` `/state` `/tasks` `/metadata` endpoints
- **Critical Phase 2 fix:** `graders.py` must be self-contained (no env imports that can crash)
- **HF Space port:** 7860
- **Submission URL format:** `https://huggingface.co/spaces/<username>/<space-name>`
- Previous space: `https://huggingface.co/spaces/dikshi2025/content-rec`

These patterns should be replicated for CascadeDebug.
