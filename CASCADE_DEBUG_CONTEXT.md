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
- **HuggingFace username:** `sparshagra51` (NEW account for credits)
- **HF Space URL (submission URL — NEVER change after Phase 4):** `https://huggingface.co/spaces/sparshagra51/cascadedebug`
- **HF API Key:** Placeholder — user to add after claiming credits onsite (25th/26th April)
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

## 🤖 Model Selection Decision (FINALIZED)

- **HuggingFace credits available onsite (25th & 26th)** — not constrained to cheapest model
- **Primary choice:** `Qwen2.5-7B-Instruct`
  - Best balance of reasoning quality + rollout speed for structured output tasks
  - 4bit via Unsloth fits on A10G; strong instruction following → non-zero r1 from episode 1
  - Strong at JSON/structured format → critical for parsing `fault_step_id`, `blame_role`, `fix_content`
- **Upgrade option (if A100 available):** `Llama-3.1-8B-Instruct` or `Qwen2.5-14B-Instruct`
- **Fallback (if VRAM constrained):** `Qwen2.5-3B-Instruct`
- **DO NOT use:** models >14B without A100 — rollout speed bottlenecks GRPO training
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
| 1 | Pipeline Bank Generation | ⬜ NOT STARTED | Agent | 200 pipelines × 5 errors — NEXT PHASE |
| 2 | Environment Core | ⬜ NOT STARTED | Person A / Agent | reset/step/state/gatekeeper |
| 3 | Reward Functions | ⬜ NOT STARTED | Person B / Agent | 4 independent signals |
| 4 | Deploy to HF Spaces | ⬜ NOT STARTED | Person D / Agent | Get URL early |
| 5 | Training Script | ⬜ NOT STARTED | Person C / Agent | GRPO + Unsloth |
| 6 | Inspect for Hacking | ⬜ NOT STARTED | C + B / Agent | Bias check |
| 7 | Full Training Run | ⬜ NOT STARTED | Person C / Agent | ~1500 episodes |
| 8 | Baseline Comparison | ⬜ NOT STARTED | Person B / Agent | Hero chart |
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
| GitHub Repo | TBD |
| HuggingFace Space | TBD |
| WandB Run | TBD |
| Colab Notebook | TBD |
| YouTube Demo | TBD |

---

## 📝 Decisions Resolved / Pending

| Decision | Status | Value |
|----------|--------|-------|
| Hackathon guidelines | ✅ RESOLVED | Processed and integrated above |
| Final model | ✅ RESOLVED | Qwen2.5-7B-Instruct (upgrade on credits) |
| Logging strategy | ✅ RESOLVED | Local CSV + .png plots committed to GitHub |
| WandB | ✅ RESOLVED | NOT using WandB |
| Team structure | ✅ RESOLVED | 2 people, push/pull freely, no fixed ownership |
| Collaboration model | ✅ RESOLVED | GitHub = source of truth, context file always committed |
| HF account | ⬜ PENDING | NEW account for credits — username to be confirmed |
| GitHub repo URL | ✅ RESOLVED | https://github.com/sparshagra/cascadedebug |
| HF Space URL | ⬜ PENDING | https://huggingface.co/spaces/sparshagra51/cascadedebug — lock in Phase 4 |
| HF API key | ⬜ PENDING | User to add after claiming credits onsite (25th/26th) |

---

## 🐛 Issues & Fixes Log

| Date | Issue | Fix Applied |
|------|-------|-------------|
| — | — | — |

---

## 📅 Session Log

| Date | Conversation ID | What was done |
|------|----------------|---------------|
| 2026-04-25 | 20614897-4a98-488b-838e-722ac8add5f0 | Initial spec processed, context file created, hackathon guidelines processed |
| 2026-04-25 | 20614897-4a98-488b-838e-722ac8add5f0 | Phase 0 DONE: openenv init, scaffold built, git init, pushed to GitHub |

---

## 🧠 Previous OpenEnv Hackathon Context

A previous hackathon submission (content-recommendation RL environment) was built for the same Meta × PyTorch OpenEnv hackathon. Key learnings:

- **OpenEnv requires:** `openenv.yaml` with graders section, `Dockerfile` at root, `inference.py` at root, `/reset` `/step` `/state` `/tasks` `/metadata` endpoints
- **Critical Phase 2 fix:** `graders.py` must be self-contained (no env imports that can crash)
- **HF Space port:** 7860
- **Submission URL format:** `https://huggingface.co/spaces/<username>/<space-name>`
- Previous space: `https://huggingface.co/spaces/dikshi2025/content-rec`

These patterns should be replicated for CascadeDebug.
