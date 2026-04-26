"""
CascadeDebug GRPO Training — Google Colab (Phase 7)
===================================================

Use `training/colab_phase7/Phase7_GRPO_Colab.ipynb` in Google Colab, or run
this file with:  python training/train_grpo_colab.py

- **mergekit** is required: TRL's GRPOTrainer imports it.
- Set `PROFILE` below: `"submission"` = 3B, 90 steps (typically under 3h on Colab T4). `"full"` = 7B+300
  steps (aligned with `train_gpu.py`, long on T4).
"""

# ═══════════════════════════════════════════════════════════════════════
# CELL 1: Install dependencies
# ═══════════════════════════════════════════════════════════════════════
import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

print("📦 Installing dependencies...")
# Unsloth for fast 4bit training
subprocess.check_call([sys.executable, "-m", "pip", "install", "unsloth", "-q"])
install("trl>=0.12.0")
install("mergekit")
install("huggingface_hub")
install("datasets")
install("matplotlib")
install("numpy")
print("✅ Dependencies installed!")

# ═══════════════════════════════════════════════════════════════════════
# CELL 2: Clone repo & load data
# ═══════════════════════════════════════════════════════════════════════
import os
import json
import random
import csv
import re
import time
from pathlib import Path
from collections import Counter
from datetime import datetime

# Detect environment and locate project root
# Priority: /app (HF Spaces), /content/cascadedebug (Colab), script's parent dir (local)
_script_dir = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path(".")

if Path("/app/data/pipeline_bank.json").exists():
    # HF Spaces Docker environment
    REPO_DIR = "/app"
    print("✅ Running in HF Spaces")
elif Path("/content/cascadedebug").exists():
    REPO_DIR = "/content/cascadedebug"
    os.system(f"cd {REPO_DIR} && git pull")
    print("✅ Repo updated (Colab)!")
elif (_script_dir / "data" / "pipeline_bank.json").exists():
    REPO_DIR = str(_script_dir)
    print("✅ Running locally")
else:
    # Colab first run — clone
    REPO_DIR = "/content/cascadedebug"
    os.system("git clone https://github.com/sparshagra/cascadedebug.git /content/cascadedebug")
    print("✅ Repo cloned!")

os.chdir(REPO_DIR)

# Load pipeline bank
PIPELINE_BANK_PATH = Path(REPO_DIR) / "data" / "pipeline_bank.json"
RESULTS_DIR = Path(REPO_DIR) / "results"
RESULTS_DIR.mkdir(exist_ok=True)
LOG_PATH = RESULTS_DIR / "training_log.csv"

with open(PIPELINE_BANK_PATH, "r") as f:
    PIPELINE_BANK = json.load(f)

BANK_BY_LEVEL = {}
for ep in PIPELINE_BANK:
    level = ep["curriculum_level"]
    BANK_BY_LEVEL.setdefault(level, []).append(ep)

print(f"📦 Loaded {len(PIPELINE_BANK)} episodes")
for level, eps in sorted(BANK_BY_LEVEL.items()):
    print(f"   Level {level}: {len(eps)} episodes")

# ═══════════════════════════════════════════════════════════════════════
# CELL 3: Configuration
# ═══════════════════════════════════════════════════════════════════════

# "submission" = 3B + 90 steps + shorter completions (typ. ~1–2.5h T4, under 3h with margin)
# "full"       = 7B + 300 steps (context / train_gpu parity; 5–7h+ on T4 is common)
PROFILE = "submission"

if PROFILE not in ("submission", "full"):
    raise ValueError('PROFILE must be "submission" or "full"')

MAX_SEQ_LENGTH = 2048
LORA_R = 16
LORA_ALPHA = 16
NUM_GENERATIONS = 2  # GRPO group size; keep at 2
LEARNING_RATE = 5e-6
LOG_EVERY = 5

if PROFILE == "full":
    MODEL_NAME = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
    MAX_STEPS = 300
    MAX_COMPLETION_LENGTH = 256
    GRADIENT_ACCUMULATION = 8
    SAVE_EVERY = 50
    DATASET_SAMPLES = 500
else:
    MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
    MAX_STEPS = 90
    MAX_COMPLETION_LENGTH = 160
    GRADIENT_ACCUMULATION = 4
    SAVE_EVERY = 20
    DATASET_SAMPLES = 250

# Curriculum
CURRICULUM_THRESHOLDS = {1: 0.4, 2: 0.6, 3: 0.7}
ROLLING_WINDOW = 30

# HF config (for pushing model)
HF_TOKEN = os.getenv("HF_TOKEN", "")  # Set in Colab: os.environ["HF_TOKEN"] = "hf_..."
HF_REPO = "Dikshita2026/cascadedebug"

print(f"🤖 Model: {MODEL_NAME}")
print(f"⚙️  Profile: {PROFILE} | Max steps: {MAX_STEPS} | Group: {NUM_GENERATIONS} | "
      f"completion_len: {MAX_COMPLETION_LENGTH} | grad_accum: {GRADIENT_ACCUMULATION}")
print(f"💾 Results: {RESULTS_DIR}")

# ═══════════════════════════════════════════════════════════════════════
# CELL 4: Reward functions (self-contained)
# ═══════════════════════════════════════════════════════════════════════

def reward_localization(predicted_step, true_step, curriculum_level=1):
    if predicted_step == true_step:
        return 1.0
    if curriculum_level == 1 and abs(predicted_step - true_step) == 1:
        return 0.3
    return 0.0

def reward_blame(predicted_role, true_role):
    return 1.0 if predicted_role == true_role else 0.0

def reward_fix(fix_content, original_output, expected_keywords, corrupted_output):
    if not fix_content or not fix_content.strip():
        return 0.0
    fix_lower = fix_content.lower()
    # Keyword match (0.5)
    if expected_keywords:
        matches = sum(1 for kw in expected_keywords if kw.lower() in fix_lower)
        kw_score = matches / len(expected_keywords)
    else:
        kw_score = 0.5
    # Not-corrupt (0.2)
    nc_score = 0.0 if (corrupted_output and fix_lower == corrupted_output.lower()) else 1.0
    # Length (0.15)
    if original_output:
        ratio = len(fix_content) / max(len(original_output), 1)
        length_score = 1.0 if 0.3 <= ratio <= 3.0 else max(0, min(ratio / 0.3, 1.0 - (ratio - 3.0) / 7.0))
    else:
        length_score = 0.5
    # Similarity (0.15)
    if original_output:
        orig_words = set(original_output.lower().split())
        fix_words = set(fix_lower.split())
        sim_score = len(orig_words & fix_words) / max(len(orig_words), 1)
    else:
        sim_score = 0.0
    return min(max(0.50 * kw_score + 0.20 * nc_score + 0.15 * length_score + 0.15 * sim_score, 0), 1)

def reward_precision(turn, gatekeeper_accepted=False):
    if gatekeeper_accepted:
        return 1.0 if turn == 1 else 0.6
    return 0.2

def compute_episode_reward(action, episode, curriculum_level=1):
    r1 = reward_localization(action["fault_step_id"], episode["injected_step"], curriculum_level)
    r2 = reward_blame(action["blame_role"], episode["injected_role"])
    r3 = reward_fix(
        action["fix_content"],
        episode.get("original_output", ""),
        episode.get("expected_fix_keywords", []),
        episode.get("corrupted_output", ""),
    )
    r4 = reward_precision(1, False)
    total = 0.35 * r1 + 0.20 * r2 + 0.35 * r3 + 0.10 * r4
    return {"localization": r1, "blame": r2, "fix": r3, "precision": r4, "total": total}

print("✅ Reward functions loaded")

# ═══════════════════════════════════════════════════════════════════════
# CELL 5: Prompt template & action parser
# ═══════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are CascadeDebug, an expert at finding and fixing errors in multi-step AI pipelines.

You will see a pipeline of steps, each produced by a different role (Researcher, Coder, or Analyst).
Exactly ONE step contains a silent error. Your job:
1. Identify which step has the error (fault_step_id)
2. Identify which role produced it (blame_role)
3. Provide the corrected output (fix_content)

Respond ONLY with valid JSON:
{"fault_step_id": <int>, "blame_role": "<Researcher|Coder|Analyst>", "fix_content": "<corrected output>", "action_type": "submit"}"""


def format_episode_prompt(episode):
    pipeline = episode["corrupted_pipeline"]
    steps_text = ""
    for step in pipeline:
        steps_text += f"\n--- Step {step['step_id']} ({step['role']}) ---\n{step['output']}\n"
    user_msg = f"Task: {episode['task_brief']}\n\nPipeline output (one step has a silent error):\n{steps_text}\nFind the faulty step and provide corrected output as JSON."
    return SYSTEM_PROMPT, user_msg


def parse_action(text):
    """Parse model output into action dict with robust fallback."""
    try:
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            action = json.loads(match.group())
            if all(k in action for k in ["fault_step_id", "blame_role", "fix_content"]):
                action.setdefault("action_type", "submit")
                action["fault_step_id"] = int(action["fault_step_id"])
                if action["blame_role"] in ["Researcher", "Coder", "Analyst"]:
                    return action
    except:
        pass
    try:
        step_match = re.search(r'fault_step_id["\s:]+(\d+)', text)
        role_match = re.search(r'blame_role["\s:]+["\']?(Researcher|Coder|Analyst)', text)
        fix_match = re.search(r'fix_content["\s:]+["\'](.+?)["\']', text, re.DOTALL)
        return {
            "fault_step_id": int(step_match.group(1)) if step_match else 1,
            "blame_role": role_match.group(1) if role_match else "Researcher",
            "fix_content": fix_match.group(1) if fix_match else "default fix",
            "action_type": "submit",
        }
    except:
        return {"fault_step_id": 1, "blame_role": "Researcher", "fix_content": "parse error", "action_type": "submit"}


print("✅ Prompt template & parser ready")

# ═══════════════════════════════════════════════════════════════════════
# CELL 6: Load model
# ═══════════════════════════════════════════════════════════════════════
import torch
print(f"🔧 CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"🔧 GPU: {torch.cuda.get_device_name(0)}")
    print(f"🔧 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

from unsloth import FastLanguageModel

print(f"\n🤖 Loading {MODEL_NAME}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    load_in_4bit=True,
    dtype=None,  # auto-detect
)

# Add LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
print("✅ Model loaded with LoRA adapters!")

# ═══════════════════════════════════════════════════════════════════════
# CELL 7: Build training dataset
# ═══════════════════════════════════════════════════════════════════════
from datasets import Dataset

curriculum_level = 1
reward_history = []
all_log_rows = []

def build_dataset(curriculum_level, n_samples=500):
    """Build dataset from pipeline bank for current curriculum level."""
    episodes = BANK_BY_LEVEL.get(curriculum_level, [])
    if not episodes:
        episodes = PIPELINE_BANK

    data = []
    for i in range(n_samples):
        ep = episodes[i % len(episodes)]
        sys_prompt, user_msg = format_episode_prompt(ep)

        # Format as chat messages
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        data.append({
            "prompt": prompt,
            "episode_idx": i % len(episodes),
        })

    return Dataset.from_list(data)

train_dataset = build_dataset(curriculum_level, n_samples=DATASET_SAMPLES)
print(f"✅ Training dataset: {len(train_dataset)} samples (Level {curriculum_level}), profile={PROFILE!r}")

# ═══════════════════════════════════════════════════════════════════════
# CELL 8: GRPO Reward Wrapper
# ═══════════════════════════════════════════════════════════════════════

# Map episode indices for reward computation
episode_list = BANK_BY_LEVEL.get(curriculum_level, PIPELINE_BANK)
step_counter = {"count": 0}

def cascadedebug_reward(completions, **kwargs):
    """
    GRPO reward function.
    Takes model completions and returns reward scores.
    """
    global curriculum_level, reward_history

    rewards = []
    prompts = kwargs.get("prompts", [None] * len(completions))

    for i, completion in enumerate(completions):
        # Extract text from completion
        if isinstance(completion, list):
            text = completion[-1].get("content", "") if completion else ""
        elif isinstance(completion, str):
            text = completion
        else:
            text = str(completion)

        # Pick episode (cycle through)
        ep_idx = (step_counter["count"] + i) % len(episode_list)
        ep = episode_list[ep_idx]

        # Parse action
        action = parse_action(text)

        # Compute reward
        result = compute_episode_reward(action, ep, curriculum_level)
        reward = result["total"]
        rewards.append(reward)
        reward_history.append(reward)

        # Log
        all_log_rows.append({
            "step": len(reward_history),
            "level": curriculum_level,
            "pipeline_id": ep["pipeline_id"],
            "r_loc": result["localization"],
            "r_blame": result["blame"],
            "r_fix": result["fix"],
            "r_prec": result["precision"],
            "r_total": result["total"],
            "predicted_step": action["fault_step_id"],
            "true_step": ep["injected_step"],
            "predicted_role": action["blame_role"],
            "true_role": ep["injected_role"],
        })

    step_counter["count"] += len(completions)

    # Curriculum advancement check (every 50 steps)
    if len(reward_history) >= ROLLING_WINDOW and len(reward_history) % 20 == 0:
        avg = sum(reward_history[-ROLLING_WINDOW:]) / ROLLING_WINDOW
        threshold = CURRICULUM_THRESHOLDS.get(curriculum_level, 1.0)
        if avg > threshold and curriculum_level < 3:
            curriculum_level += 1
            print(f"\n🎓 CURRICULUM ADVANCED → Level {curriculum_level}! (rolling avg={avg:.3f})")

    return rewards

print("✅ GRPO reward function ready")

# ═══════════════════════════════════════════════════════════════════════
# CELL 9: Train!
# ═══════════════════════════════════════════════════════════════════════
from trl import GRPOConfig, GRPOTrainer

print("\n" + "=" * 60)
print("🚀 STARTING GRPO TRAINING")
print(f"   Model: {MODEL_NAME}")
print(f"   Steps: {MAX_STEPS}")
print(f"   Group size: {NUM_GENERATIONS}")
print(f"   Curriculum: Level {curriculum_level}")
print("=" * 60 + "\n")

training_args = GRPOConfig(
    output_dir=str(RESULTS_DIR / "grpo_checkpoints"),
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=1,
    num_generations=NUM_GENERATIONS,
    max_completion_length=MAX_COMPLETION_LENGTH,
    max_prompt_length=MAX_SEQ_LENGTH - MAX_COMPLETION_LENGTH,
    max_steps=MAX_STEPS,
    logging_steps=LOG_EVERY,
    save_steps=SAVE_EVERY,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    warmup_ratio=0.1,
    optim="adamw_8bit",
    seed=42,
    report_to="none",
    remove_unused_columns=False,
    log_level="info",
)

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=cascadedebug_reward,
    args=training_args,
    train_dataset=train_dataset,
)

start_time = time.time()
trainer.train()
elapsed = time.time() - start_time

print(f"\n✅ Training complete in {elapsed/60:.1f} minutes!")
print(f"   Total reward samples: {len(reward_history)}")
if reward_history:
    print(f"   Final rolling avg: {sum(reward_history[-50:]) / min(len(reward_history), 50):.4f}")
    print(f"   Best rolling avg: {max(sum(reward_history[i:i+50]) / 50 for i in range(max(1, len(reward_history)-50))):.4f}")

# ═══════════════════════════════════════════════════════════════════════
# CELL 10: Save model
# ═══════════════════════════════════════════════════════════════════════
print("\n💾 Saving model...")

# Save LoRA adapter
model.save_pretrained(str(RESULTS_DIR / "lora_adapter"))
tokenizer.save_pretrained(str(RESULTS_DIR / "lora_adapter"))
print("   ✅ LoRA adapter saved")

# Save merged 16bit model
try:
    model.save_pretrained_merged(
        str(RESULTS_DIR / "merged_model"),
        tokenizer,
        save_method="merged_16bit",
    )
    print("   ✅ Merged 16bit model saved")
except Exception as e:
    print(f"   ⚠️ Merged save failed (non-critical): {e}")

# ═══════════════════════════════════════════════════════════════════════
# CELL 11: Save training log
# ═══════════════════════════════════════════════════════════════════════
if all_log_rows:
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_log_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_log_rows)
    print(f"\n📝 Training log saved: {LOG_PATH} ({len(all_log_rows)} rows)")

# ═══════════════════════════════════════════════════════════════════════
# CELL 12: Generate plots
# ═══════════════════════════════════════════════════════════════════════
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

print("\n📈 Generating plots...")

# Plot 1: Reward curve
fig, ax = plt.subplots(figsize=(12, 5))
steps = list(range(1, len(reward_history) + 1))
ax.plot(steps, reward_history, alpha=0.2, color='#3b82f6', label='Per-completion', linewidth=0.5)

window = min(50, len(reward_history) // 5) if len(reward_history) > 20 else 5
if window > 1 and len(reward_history) > window:
    rolling = np.convolve(reward_history, np.ones(window)/window, mode='valid')
    ax.plot(range(window, len(reward_history) + 1), rolling,
            color='#ef4444', linewidth=2.5, label=f'Rolling avg (w={window})')

ax.set_xlabel('Training Step', fontsize=13)
ax.set_ylabel('Total Reward', fontsize=13)
ax.set_title('CascadeDebug — GRPO Training Reward Curve', fontsize=15, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'reward_curve.png', dpi=150)
plt.close()
print("   ✅ reward_curve.png")

# Plot 2: Component rewards
if all_log_rows:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    components = [
        ('r_loc', 'Fault Localization (r1, w=0.35)', '#3b82f6'),
        ('r_blame', 'Blame Attribution (r2, w=0.20)', '#10b981'),
        ('r_fix', 'Fix Correctness (r3, w=0.35)', '#f59e0b'),
        ('r_prec', 'Surgical Precision (r4, w=0.10)', '#8b5cf6'),
    ]
    for ax, (key, title, color) in zip(axes.flat, components):
        values = [r[key] for r in all_log_rows]
        ax.plot(range(len(values)), values, alpha=0.2, color=color, linewidth=0.5)
        if len(values) > 20:
            w = min(50, len(values) // 5)
            rolling = np.convolve(values, np.ones(w)/w, mode='valid')
            ax.plot(range(w-1, len(values)), rolling, color=color, linewidth=2)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Step')
        ax.set_ylabel('Reward')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
    plt.suptitle('CascadeDebug — Component Reward Signals', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'component_rewards.png', dpi=150)
    plt.close()
    print("   ✅ component_rewards.png")

# Plot 3: Baseline vs trained comparison
if all_log_rows:
    fig, ax = plt.subplots(figsize=(10, 6))

    # Baseline (first 20% of training)
    n_baseline = max(len(all_log_rows) // 5, 10)
    baseline_rows = all_log_rows[:n_baseline]
    trained_rows = all_log_rows[-n_baseline:]

    # Group by level
    for label, rows, offset, color in [
        ("Untrained (first 20%)", baseline_rows, -0.15, '#94a3b8'),
        ("Trained (last 20%)", trained_rows, 0.15, '#3b82f6'),
    ]:
        by_level = {}
        for r in rows:
            by_level.setdefault(r["level"], []).append(r["r_total"])

        levels = sorted(by_level.keys())
        avgs = [sum(by_level.get(l, [0])) / max(len(by_level.get(l, [1])), 1) for l in [1, 2, 3]]

        x = np.arange(3) + offset
        bars = ax.bar(x, avgs, width=0.28, label=label, color=color)
        for bar, avg in zip(bars, avgs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{avg:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax.set_xticks(range(3))
    ax.set_xticklabels(['Level 1\n(3-step, easy)', 'Level 2\n(4-step, medium)', 'Level 3\n(5-6 step, hard)'])
    ax.set_ylabel('Average Reward', fontsize=13)
    ax.set_title('CascadeDebug — Untrained vs Trained Agent', fontsize=15, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'baseline_vs_trained.png', dpi=150)
    plt.close()
    print("   ✅ baseline_vs_trained.png")

# Plot 4: Localization accuracy over training
if all_log_rows:
    fig, ax = plt.subplots(figsize=(10, 5))
    correct = [1 if r["predicted_step"] == r["true_step"] else 0 for r in all_log_rows]
    if len(correct) > 20:
        w = min(50, len(correct) // 5)
        rolling_acc = np.convolve(correct, np.ones(w)/w, mode='valid')
        ax.plot(range(w-1, len(correct)), rolling_acc, color='#3b82f6', linewidth=2)
    ax.axhline(y=1/3, color='#ef4444', linestyle='--', label='Random baseline (1/3)', linewidth=1.5)
    ax.set_xlabel('Training Step', fontsize=13)
    ax.set_ylabel('Localization Accuracy', fontsize=13)
    ax.set_title('CascadeDebug — Fault Localization Accuracy Over Training', fontsize=15, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'localization_accuracy.png', dpi=150)
    plt.close()
    print("   ✅ localization_accuracy.png")

print("\n✅ All plots saved to results/")

# ═══════════════════════════════════════════════════════════════════════
# CELL 13: Push results to GitHub & HF
# ═══════════════════════════════════════════════════════════════════════
print("\n📤 Pushing results...")

# Optional: Git push (Colab usually has no GitHub credentials — use Files sidebar → Download results/)
print("   (Optional) GitHub push — skipped by default; download results/ if needed.")
try:
    _c = subprocess.run(
        f'cd {REPO_DIR} && git add results/ 2>/dev/null && git -c user.email=colab@local -c user.name=Colab '
        f'commit -m "Phase 7: GRPO {MAX_STEPS} steps" 2>/dev/null && git push 2>&1',
        shell=True, capture_output=True, text=True, timeout=120,
    )
    if _c.returncode == 0:
        print("   ✅ Pushed to GitHub")
    else:
        print("   ℹ️ No push (add GitHub token or run git locally on your machine).")
except Exception as _e:
    print("   ℹ️ Git not configured:", str(_e)[:120])

# Upload plots to HF Space (optional; needs HF write token)
if HF_TOKEN:
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        for plot_file in ["reward_curve.png", "component_rewards.png", "baseline_vs_trained.png", "localization_accuracy.png", "training_log.csv"]:
            plot_path = RESULTS_DIR / plot_file
            if plot_path.exists():
                api.upload_file(
                    path_or_fileobj=str(plot_path),
                    path_in_repo=f"results/{plot_file}",
                    repo_id=HF_REPO,
                    repo_type="space",
                )
        print("   ✅ Results uploaded to Hugging Face Space")
    except Exception as e:
        print(f"   ⚠️ HF upload failed: {e}")
else:
    print("   ℹ️ HF_TOKEN not set — upload skipped (plots remain under results/)")

# ═══════════════════════════════════════════════════════════════════════
# CELL 14: Final summary
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("🏆 TRAINING COMPLETE — Final Summary")
print("=" * 60)
print(f"   Model: {MODEL_NAME}")
print(f"   Steps: {MAX_STEPS}")
print(f"   Duration: {elapsed/60:.1f} minutes")
print(f"   Total reward samples: {len(reward_history)}")

if reward_history:
    # First 20% vs last 20%
    n = len(reward_history)
    early = reward_history[:n//5]
    late = reward_history[-n//5:]
    print(f"\n   📊 Early avg reward:  {sum(early)/len(early):.4f}")
    print(f"   📊 Late avg reward:   {sum(late)/len(late):.4f}")
    print(f"   📊 Improvement:       {(sum(late)/len(late) - sum(early)/len(early)):.4f}")

    # Localization accuracy
    if all_log_rows:
        early_rows = all_log_rows[:len(all_log_rows)//5]
        late_rows = all_log_rows[-len(all_log_rows)//5:]
        early_acc = sum(1 for r in early_rows if r["predicted_step"] == r["true_step"]) / max(len(early_rows), 1)
        late_acc = sum(1 for r in late_rows if r["predicted_step"] == r["true_step"]) / max(len(late_rows), 1)
        print(f"\n   🎯 Early localization accuracy: {early_acc:.2%}")
        print(f"   🎯 Late localization accuracy:  {late_acc:.2%}")

print(f"\n   📁 Results: {RESULTS_DIR}")
print(f"   🌐 HF Space: https://huggingface.co/spaces/{HF_REPO}")
print("=" * 60)
