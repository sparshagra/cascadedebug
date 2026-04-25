# CascadeDebug — Hackathon Guidelines, Pitfalls & Grading Reference
# Saved: 2026-04-25 | Source: Official hackathon docs

---

## 🏆 Judging Criteria (Weights)

| Criterion | Weight | What Judges Look For |
|-----------|--------|---------------------|
| **Environment Innovation** | **40%** | Novel, creative, challenging. Not clones. Could a researcher write a paper on this? |
| **Storytelling & Presentation** | **30%** | Clear problem/env/agent story. Engaging demo for non-technical audience. README in 3-5 min. |
| **Showing Improvement in Rewards** | **20%** | Observable training progress. Reward curves. Trained vs untrained baseline. Quantitative + qualitative. |
| **Reward & Training Pipeline** | **10%** | Coherent reward logic. Pipeline produces meaningful improvement. |

---

## ✅ Non-Negotiable Minimum Requirements

- [ ] Use **OpenEnv (latest release)** — `pip install openenv-core`
- [ ] **Working Colab training script** using Unsloth or HF TRL that judges can re-run
- [ ] **Evidence of real training** — loss and reward plots from an actual run
- [ ] **Mini-blog on HuggingFace** OR **<2 min YouTube video** explaining environment + training
- [ ] **Environment hosted on HF Space** — URL submitted (NEVER change after submission)
- [ ] **README** — motivates problem, explains env, shows results, links all materials
- [ ] All plots saved as `.png` or `.jpg` **committed to repo** (not just Colab cells)
- [ ] No large video files in HF Space repo — use URL links instead

---

## 🔴 Critical Pitfalls Checklist

### Environment Design Pitfalls
- [ ] P1: Weak verifier → rule-based verifiers reject correct answers; model-based ones exploited
- [ ] P2: "Just use LLM as judge" — judge becomes part of optimization target, gets gamed
- [ ] P3: Static task difficulty → too easy = no signal; too hard = zero reward, no learning
- [ ] P4: Too few task types → model learns narrow strategies, fails to transfer
- [ ] P5: Environment too toylike → optimizes benchmark, fails real workflow

### Reward Engineering Pitfalls
- [ ] R1: Using proxy metric AS the goal (Goodhart's Law) — token count, format, test count
- [ ] R2: Starting with complicated reward → start SIMPLE (sparse 0/1), add shaping carefully
- [ ] R3: Conflicting reward components → results in oscillation or brittle shortcuts
- [ ] R4: Binary reward only → too sparse for long-horizon tasks
- [ ] R5: Dense rewards creating wrong local optima and incentive misalignment

### RL Post-Training Pitfalls
- [ ] T1: Using RL before base model is ready → GRPO needs model to occasionally succeed
- [ ] T2: Monitoring only headline reward → rising reward ≠ improving real task quality
- [ ] T3: Not auditing actual generations → reward hacking invisible in aggregate metrics
- [ ] T4: Saving LoRA/QLoRA models incorrectly → naive 4-bit→16-bit upcast damages quality
- [ ] T5: Training forever without inspecting outputs → drift undetected

### Anti-Hacking Measures
- [ ] H1: Injection step must be **uniformly random** — verify chi-squared distribution
- [ ] H2: Code execution must have **hard timeout** (5s subprocess) — never raw exec()
- [ ] H3: No LLM-as-judge in reward functions — all verifiers programmatic
- [ ] H4: Gatekeeper must be **deterministic rule-based** — not a second LLM call
- [ ] H5: Pipeline bank **pre-computed offline** — no LLM calls at training time
- [ ] H6: Agent observation **must NOT include injected_step** — only state() reveals it
- [ ] H7: "Always guess step 1" exploit check — step 1 correct rate must be < 40%

---

## 🧠 Key RL Concepts Reference

### Why Rewards Matter
> "RL gives you what you asked for, not what you meant." — DeepMind specification gaming
The reward IS the task specification. Incomplete or gameable reward → model optimizes wrong thing.

### GRPO vs PPO
- **PPO**: Classic policy optimization, constrains how much policy changes between iterations
- **GRPO**: Group-relative variant — compares outputs within a group, estimates relative advantage, more memory-efficient (removes value model from PPO)

### RLVR vs RLVE
- **RLVR**: Verifiable rewards on fixed/semi-fixed prompts (our setup)
- **RLVE**: Dynamic environments that procedurally generate tasks and adjust difficulty (prevents static dataset saturation)

### Curriculum Learning
- Too hard at start → never gets reward → learning stalls
- Progression: short horizons → fewer tools → simpler state → stronger hints → harder tasks
- Advance ONLY after rolling average reward crosses threshold

### Process Supervision vs Outcome-Only
- Outcome-only: same reward to every token → wasteful, blurs which steps were good
- Process supervision: feedback on intermediate steps → better sample efficiency
- Risk: step labels can be noisy or exploitable

### When is RL Appropriate?
✅ You can verify success programmatically
✅ Exploration is meaningful
✅ Multi-step interaction matters
✅ No abundant high-quality demonstrations exist
❌ Task so hard model NEVER gets any reward (probability of good answer = 0)

---

## 📊 What to Monitor During Training

```
Do NOT monitor only aggregate reward. Track separately:
- reward_localization   → is fault localization improving?
- reward_blame          → is attribution improving?
- reward_fix            → is fix quality improving?
- reward_precision      → is agent getting more surgical?
- gatekeeper_accept_rate → is negotiation improving?
- curriculum_level      → is curriculum advancing?
- rollout diversity     → are solutions getting varied or converging to one hack?
- sample actual generations every 50 steps (NOT just metrics)
```

**Red flags:**
- Rising reward + flat localization accuracy = reward hacking
- Model always predicts same step/role = bias (hacking)
- Fix content is empty or trivial = gaming fix verifier
- Suspiciously high judge scores = LLM judge exploitation

---

## 💾 Model Saving (CRITICAL — Do Not Get Wrong)

```python
# CORRECT — Unsloth merged save
model.save_pretrained_merged(
    "cascadedebug-final",
    tokenizer,
    save_method="merged_16bit",  # NOT naive 4bit upcast
)
# Test inference IMMEDIATELY after saving

# WRONG — naive 4bit upcast damages model quality
model.half()  # DO NOT DO THIS
model.merge_and_unload()  # DO NOT DO THIS on 4bit models without Unsloth path
```

---

## 📈 Plot Requirements (Judges Spend Seconds Per Plot)

- Label **both axes** with units (x = "training step", y = "reward")
- Save as `.png`, commit to repo — NOT just Colab cells or WandB
- Embed key plots in README with one-line caption
- Put baseline vs trained **on same axes** for obvious comparison
- Key plots needed: reward_curve.png, component_rewards.png, baseline_vs_trained.png, localization_accuracy.png

---

## 🎯 Best Unsloth GRPO Recipes (Reference)

| Use Case | Recipe |
|----------|--------|
| Simplest starting point | Qwen2.5-3B GRPO or Gemma-3-1B GRPO notebook |
| Care about reward engineering | **Advanced Qwen3 (4B) GRPO** — proximity scoring, advanced templates |
| Environment-style RL | GPT-OSS 20B 2048 game notebook |
| Guided learning path | HF Practical Exercise with Unsloth + GRPO |
| Full recipe collection | Unsloth notebooks repo (250+ notebooks) |

Advanced Qwen3 features to adopt:
- **Proximity scoring** for more nuanced rewards (instead of binary)
- **Prefinetuning** to skip GRPO format learning phase
- **OpenR1 dataset support**

---

## 📋 Strong Submission Template

```
README structure:
1. Problem (2 paragraphs) — what capability gap?
2. Environment — what does agent see, do, get rewarded for?
3. Results — hero chart embedded as .png, before/after behavior
4. Links — HF Space, YouTube video, Colab notebook

Demo video structure (< 2 min):
0:00–0:20 — "Multi-agent pipelines fail 17x due to error cascading"
0:20–0:50 — Show environment (what agent sees, what it must do)
0:50–1:30 — Show reward curve (untrained vs trained)
1:30–2:00 — Show before/after: untrained guesses wrong step, trained pinpoints & fixes
```

---

## 🔗 Key Resources

| Resource | URL |
|----------|-----|
| TRL GRPO Docs | https://huggingface.co/docs/trl/grpo_trainer |
| TRL GRPO Cookbook | https://huggingface.co/docs/trl/en/grpo_trainer#grpo-cookbook |
| OpenEnv Repo | https://github.com/meta-pytorch/OpenEnv |
| OpenEnv Reward Guide | https://meta-pytorch.org/OpenEnv/ |
| OpenEnv Environments Hub | https://huggingface.co/openenv |
| DeepMind Specification Gaming | https://deepmind.google/research/publications/specification-gaming/ |
| Lilian Weng Reward Hacking | https://lilianweng.github.io/posts/2024-11-28-reward-hacking/ |
| PPO Paper | https://arxiv.org/abs/1707.06347 |
| DeepSeekMath / GRPO | https://arxiv.org/abs/2402.03300 |
| Unsloth Repo | https://github.com/unslothai/unsloth |
| RLVE Paper | https://arxiv.org/abs/2410.02916 |
| BrowserGym | https://github.com/ServiceNow/BrowserGym |
| Mega Lecture (RL+OpenEnv) | https://www.youtube.com/watch?v=Jew4lhAiqnw (Recommended) |
| Workshop Video | https://www.youtube.com/watch?v=1jU05MlENOI |
| RLVE / Verifier Pitfalls Paper | https://arxiv.org/abs/2410.02916 |

---

## 🎬 Lecture Modules Reference

| Module | Topic | Timestamp |
|--------|-------|-----------|
| 1 | Why OpenEnv? | Mega Lecture 40:01–46:00 |
| 2 | Using Existing Envs | Mega Lecture 1:24:11–1:30:00 |
| 3 | Deploying Envs | Mega Lecture 1:30:00–1:39:07 |
| 4 | Building Your Own | Workshop 43:45–50:20 |
| 5 | Training + TRL (Wordle GRPO walkthrough) | Mega Lecture 1:53:20–2:07:12 |

---

*Last updated: 2026-04-25. Training is in progress on HF Spaces T4 (Dikshita2026/cascadedebug-training).*
