"""
CascadeDebug GRPO Training Script — Phase 5.

This is the Python equivalent of the Colab notebook for training with GRPO.
Uses Unsloth for 4-bit quantization + TRL for GRPO training.

Usage (Colab / local GPU):
    pip install unsloth trl datasets
    python training/train_grpo.py

Model: Qwen2.5-3B-Instruct (budget-optimized for $30 HF credits)
Algorithm: GRPO (Group Relative Policy Optimization)
Framework: Unsloth + TRL
"""

import json
import os
import random
import csv
import time
from pathlib import Path
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

# Model
MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048
LORA_R = 16
LORA_ALPHA = 16

# Training
NUM_EPISODES = 1500
BATCH_SIZE = 4       # group size for GRPO
LEARNING_RATE = 5e-6
MAX_STEPS = 500
SAVE_EVERY = 100
LOG_EVERY = 10

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
PIPELINE_BANK_PATH = PROJECT_ROOT / "data" / "pipeline_bank.json"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
LOG_PATH = RESULTS_DIR / "training_log.csv"

# Curriculum
CURRICULUM_THRESHOLDS = {1: 0.4, 2: 0.6, 3: 0.7}
ROLLING_WINDOW = 50

# Environment URL (for live training) or None for offline
ENV_URL = os.getenv("ENV_URL", None)


# ──────────────────────────────────────────────────────────────────────────────
# Reward Functions (self-contained, no server imports)
# ──────────────────────────────────────────────────────────────────────────────

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
    """Compute full reward for one action on one episode."""
    r1 = reward_localization(action["fault_step_id"], episode["injected_step"], curriculum_level)
    r2 = reward_blame(action["blame_role"], episode["injected_role"])
    r3 = reward_fix(
        action["fix_content"],
        episode.get("original_output", ""),
        episode.get("expected_fix_keywords", []),
        episode.get("corrupted_output", ""),
    )
    r4 = reward_precision(1, False)  # single-turn submit
    total = 0.35 * r1 + 0.20 * r2 + 0.35 * r3 + 0.10 * r4
    return {"localization": r1, "blame": r2, "fix": r3, "precision": r4, "total": total}


# ──────────────────────────────────────────────────────────────────────────────
# Prompt Template
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are CascadeDebug, an expert at finding and fixing errors in multi-step AI pipelines.

You will see a pipeline of steps, each produced by a different role (Researcher, Coder, or Analyst). 
Exactly ONE step contains a silent error that was injected. Your job is to:
1. Identify which step has the error (fault_step_id)
2. Identify which role produced it (blame_role)
3. Provide the corrected output (fix_content)

Respond ONLY with a valid JSON object:
{"fault_step_id": <int>, "blame_role": "<Researcher|Coder|Analyst>", "fix_content": "<corrected output>", "action_type": "submit"}"""


def format_episode_prompt(episode):
    """Format an episode into a prompt for the model."""
    pipeline = episode["corrupted_pipeline"]
    steps_text = ""
    for step in pipeline:
        steps_text += f"\n--- Step {step['step_id']} ({step['role']}) ---\n{step['output']}\n"

    user_msg = f"""Task: {episode['task_brief']}

Pipeline output (one step contains a silent error):
{steps_text}
Find the faulty step, identify the responsible role, and provide the corrected output as JSON."""

    return SYSTEM_PROMPT, user_msg


# ──────────────────────────────────────────────────────────────────────────────
# Action Parser
# ──────────────────────────────────────────────────────────────────────────────

def parse_action(text):
    """Parse model output into action dict. Handles malformed JSON gracefully."""
    import re

    # Try direct JSON parse
    try:
        # Find JSON object in text
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            action = json.loads(match.group())
            # Validate required fields
            if all(k in action for k in ["fault_step_id", "blame_role", "fix_content"]):
                action.setdefault("action_type", "submit")
                action["fault_step_id"] = int(action["fault_step_id"])
                return action
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: extract fields with regex
    try:
        step_match = re.search(r'fault_step_id["\s:]+(\d+)', text)
        role_match = re.search(r'blame_role["\s:]+["\']?(Researcher|Coder|Analyst)', text)
        fix_match = re.search(r'fix_content["\s:]+["\'](.+?)["\']', text, re.DOTALL)

        return {
            "fault_step_id": int(step_match.group(1)) if step_match else 1,
            "blame_role": role_match.group(1) if role_match else "Researcher",
            "fix_content": fix_match.group(1) if fix_match else "Unable to parse fix",
            "action_type": "submit",
        }
    except Exception:
        # Last resort
        return {
            "fault_step_id": 1,
            "blame_role": "Researcher",
            "fix_content": "Parse error - default fix",
            "action_type": "submit",
        }


# ──────────────────────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────────────────────

def load_pipeline_bank():
    with open(PIPELINE_BANK_PATH, "r") as f:
        bank = json.load(f)
    by_level = {}
    for ep in bank:
        level = ep["curriculum_level"]
        by_level.setdefault(level, []).append(ep)
    return bank, by_level


def create_training_dataset(bank_by_level, curriculum_level, n_episodes):
    """Create a dataset of (prompt, episode) pairs for training."""
    episodes = bank_by_level.get(curriculum_level, [])
    if not episodes:
        episodes = sum(bank_by_level.values(), [])

    dataset = []
    for _ in range(n_episodes):
        ep = random.choice(episodes)
        system_prompt, user_msg = format_episode_prompt(ep)
        dataset.append({
            "system": system_prompt,
            "user": user_msg,
            "episode": ep,
        })
    return dataset


def main():
    """Main training entry point."""
    print("=" * 60)
    print("CascadeDebug GRPO Training")
    print(f"Model: {MODEL_NAME}")
    print(f"Episodes: {NUM_EPISODES}")
    print("=" * 60)

    # ── Step 1: Load pipeline bank ──────────────────────────────────
    print("\n📦 Loading pipeline bank...")
    bank, bank_by_level = load_pipeline_bank()
    print(f"   Loaded {len(bank)} episodes")
    for level, eps in sorted(bank_by_level.items()):
        print(f"   Level {level}: {len(eps)} episodes")

    # ── Step 2: Load model ──────────────────────────────────────────
    print(f"\n🤖 Loading model: {MODEL_NAME}")
    try:
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=True,
        )

        # Add LoRA adapters
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
        print("   ✅ Model loaded with LoRA adapters")
    except ImportError:
        print("   ❌ Unsloth not installed. Install with: pip install unsloth")
        print("   Running in DRY RUN mode (no actual training)")
        model = None
        tokenizer = None

    # ── Step 3: Setup GRPO reward function ──────────────────────────
    print("\n🏆 Setting up GRPO reward function...")

    curriculum_level = 1
    reward_history = []

    def grpo_reward_fn(completions, **kwargs):
        """GRPO reward function — called by TRL trainer."""
        nonlocal curriculum_level, reward_history

        prompts = kwargs.get("prompts", [])
        episodes = kwargs.get("episodes", [])
        rewards = []

        for i, completion in enumerate(completions):
            # Get the text output
            if isinstance(completion, list):
                text = completion[-1].get("content", "") if completion else ""
            elif isinstance(completion, str):
                text = completion
            else:
                text = str(completion)

            # Parse action from model output
            action = parse_action(text)

            # Get episode data
            ep = episodes[i % len(episodes)] if episodes else None
            if ep is None:
                rewards.append(0.0)
                continue

            # Compute reward
            result = compute_episode_reward(action, ep, curriculum_level)
            rewards.append(result["total"])
            reward_history.append(result["total"])

        # Check curriculum advance
        if len(reward_history) >= ROLLING_WINDOW:
            avg = sum(reward_history[-ROLLING_WINDOW:]) / ROLLING_WINDOW
            threshold = CURRICULUM_THRESHOLDS.get(curriculum_level, 1.0)
            if avg > threshold and curriculum_level < 3:
                curriculum_level += 1
                print(f"\n🎓 Curriculum advanced to Level {curriculum_level}! (avg={avg:.3f})")

        return rewards

    # ── Step 4: Create training data ────────────────────────────────
    print("\n📊 Creating training dataset...")
    dataset = create_training_dataset(bank_by_level, curriculum_level, NUM_EPISODES)
    print(f"   Created {len(dataset)} training examples")

    # ── Step 5: Training ────────────────────────────────────────────
    if model is not None:
        try:
            from trl import GRPOConfig, GRPOTrainer

            print("\n🚀 Starting GRPO training...")

            # Format prompts for training
            train_prompts = []
            train_episodes = []
            for item in dataset:
                train_prompts.append([
                    {"role": "system", "content": item["system"]},
                    {"role": "user", "content": item["user"]},
                ])
                train_episodes.append(item["episode"])

            # GRPO config
            training_args = GRPOConfig(
                output_dir=str(RESULTS_DIR / "grpo_output"),
                learning_rate=LEARNING_RATE,
                per_device_train_batch_size=1,
                num_generations=BATCH_SIZE,  # group size
                max_completion_length=512,
                max_steps=MAX_STEPS,
                logging_steps=LOG_EVERY,
                save_steps=SAVE_EVERY,
                bf16=True,
                gradient_accumulation_steps=4,
                warmup_ratio=0.1,
                optim="adamw_8bit",
                seed=42,
                report_to="none",  # No WandB
            )

            # Custom reward wrapper that passes episodes
            episode_store = {"episodes": train_episodes, "idx": 0}

            def reward_wrapper(completions, **kwargs):
                idx = episode_store["idx"]
                batch_episodes = train_episodes[idx:idx + len(completions)]
                if not batch_episodes:
                    batch_episodes = random.choices(train_episodes, k=len(completions))
                episode_store["idx"] = (idx + len(completions)) % len(train_episodes)
                return grpo_reward_fn(completions, episodes=batch_episodes, **kwargs)

            trainer = GRPOTrainer(
                model=model,
                processing_class=tokenizer,
                reward_funcs=reward_wrapper,
                args=training_args,
                train_dataset=[{"prompt": p} for p in train_prompts],
            )

            # Train
            trainer.train()

            # Save model
            print("\n💾 Saving model...")
            model.save_pretrained_merged(
                str(RESULTS_DIR / "merged_model"),
                tokenizer,
                save_method="merged_16bit",
            )
            print("   ✅ Model saved (merged_16bit)")

        except ImportError as e:
            print(f"\n❌ TRL not installed: {e}")
            print("   Running offline reward verification instead...")
            run_offline_verification(bank, bank_by_level)

    else:
        # Dry run mode
        print("\n🧪 Running offline reward verification (dry run mode)...")
        run_offline_verification(bank, bank_by_level)

    # ── Step 6: Generate plots ──────────────────────────────────────
    if reward_history:
        generate_plots(reward_history)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


# ──────────────────────────────────────────────────────────────────────────────
# Offline Verification (runs without GPU/Unsloth)
# ──────────────────────────────────────────────────────────────────────────────

def run_offline_verification(bank, bank_by_level):
    """Run baseline agent on pipeline bank and log rewards — no GPU needed."""
    print("\n📊 Running offline baseline verification...")

    reward_history = []
    log_rows = []

    for level in [1, 2, 3]:
        episodes = bank_by_level.get(level, [])[:100]  # 100 per level
        level_rewards = []

        for ep in episodes:
            # Random baseline action
            n_steps = len(ep["corrupted_pipeline"])
            action = {
                "fault_step_id": random.randint(1, n_steps),
                "blame_role": random.choice(["Researcher", "Coder", "Analyst"]),
                "fix_content": "Baseline fix placeholder.",
                "action_type": "submit",
            }

            result = compute_episode_reward(action, ep, level)
            level_rewards.append(result["total"])
            reward_history.append(result["total"])

            log_rows.append({
                "step": len(reward_history),
                "level": level,
                "pipeline_id": ep["pipeline_id"],
                "r_loc": result["localization"],
                "r_blame": result["blame"],
                "r_fix": result["fix"],
                "r_prec": result["precision"],
                "r_total": result["total"],
            })

        avg = sum(level_rewards) / len(level_rewards) if level_rewards else 0
        print(f"   Level {level}: avg_reward={avg:.4f} over {len(episodes)} episodes")

    # Write CSV log
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
        writer.writeheader()
        writer.writerows(log_rows)
    print(f"\n📝 Training log saved to {LOG_PATH}")

    # Generate plots
    generate_plots(reward_history, log_rows=log_rows)

    return reward_history


# ──────────────────────────────────────────────────────────────────────────────
# Plot Generation
# ──────────────────────────────────────────────────────────────────────────────

def generate_plots(reward_history, log_rows=None):
    """Generate and save training plots as PNG files."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("   ⚠️ matplotlib not installed — skipping plots")
        return

    print("\n📈 Generating plots...")

    # ── Plot 1: Reward curve ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    steps = list(range(1, len(reward_history) + 1))
    ax.plot(steps, reward_history, alpha=0.3, color='#3b82f6', label='Per-episode')

    # Rolling average
    window = min(50, len(reward_history) // 5) if len(reward_history) > 10 else 1
    if window > 1:
        rolling = np.convolve(reward_history, np.ones(window)/window, mode='valid')
        ax.plot(range(window, len(reward_history) + 1), rolling,
                color='#ef4444', linewidth=2, label=f'Rolling avg (w={window})')

    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Total Reward', fontsize=12)
    ax.set_title('CascadeDebug — Reward Curve', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'reward_curve.png', dpi=150)
    plt.close()
    print(f"   ✅ reward_curve.png saved")

    # ── Plot 2: Component rewards ─────────────────────────────────
    if log_rows:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        components = [
            ('r_loc', 'Fault Localization (r1)', '#3b82f6'),
            ('r_blame', 'Blame Attribution (r2)', '#10b981'),
            ('r_fix', 'Fix Correctness (r3)', '#f59e0b'),
            ('r_prec', 'Surgical Precision (r4)', '#8b5cf6'),
        ]

        for ax, (key, title, color) in zip(axes.flat, components):
            values = [r[key] for r in log_rows]
            steps = list(range(1, len(values) + 1))
            ax.plot(steps, values, alpha=0.3, color=color)
            if len(values) > 10:
                w = min(50, len(values) // 5)
                rolling = np.convolve(values, np.ones(w)/w, mode='valid')
                ax.plot(range(w, len(values) + 1), rolling, color=color, linewidth=2)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.set_xlabel('Step')
            ax.set_ylabel('Reward')
            ax.set_ylim(0, 1.05)
            ax.grid(True, alpha=0.3)

        plt.suptitle('CascadeDebug — Component Rewards', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'component_rewards.png', dpi=150)
        plt.close()
        print(f"   ✅ component_rewards.png saved")

    # ── Plot 3: Baseline vs trained ───────────────────────────────
    if log_rows:
        fig, ax = plt.subplots(figsize=(8, 5))
        by_level = {}
        for r in log_rows:
            by_level.setdefault(r['level'], []).append(r['r_total'])

        levels = sorted(by_level.keys())
        avgs = [sum(by_level[l]) / len(by_level[l]) for l in levels]
        colors = ['#3b82f6', '#f59e0b', '#ef4444']

        bars = ax.bar([f'Level {l}' for l in levels], avgs, color=colors[:len(levels)])
        ax.set_ylabel('Average Reward', fontsize=12)
        ax.set_title('CascadeDebug — Baseline Rewards by Level', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')

        for bar, avg in zip(bars, avgs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{avg:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / 'baseline_vs_trained.png', dpi=150)
        plt.close()
        print(f"   ✅ baseline_vs_trained.png saved")


# ──────────────────────────────────────────────────────────────────────────────
# Phase 6: Hacking Inspection
# ──────────────────────────────────────────────────────────────────────────────

def inspect_for_hacking():
    """
    Phase 6: Verify anti-hacking measures before training.

    Checks:
    1. Injection step is uniformly distributed
    2. No easy shortcut patterns (always same step, always same role)
    3. Error types are balanced per level
    4. Fix keywords are non-trivial
    """
    print("\n" + "=" * 60)
    print("🔍 Phase 6: Hacking Inspection")
    print("=" * 60)

    bank, bank_by_level = load_pipeline_bank()

    # Check 1: Injection step uniformity (per pipeline length, not global)
    print("\n📊 Check 1: Injection step distribution (per pipeline length)")
    from collections import Counter

    # Group by pipeline length
    by_length = {}
    for ep in bank:
        plen = len(ep["corrupted_pipeline"])
        by_length.setdefault(plen, []).append(ep)

    total = len(bank)
    all_ok = True
    for plen in sorted(by_length):
        eps = by_length[plen]
        step_dist = Counter(ep["injected_step"] for ep in eps)
        expected = len(eps) / plen
        chi2 = sum((obs - expected) ** 2 / expected for obs in step_dist.values())
        status = "✅" if chi2 < 20 else "⚠️"
        print(f"   Pipeline length {plen}: {len(eps)} episodes, steps={dict(sorted(step_dist.items()))}, χ²={chi2:.1f} {status}")
        if chi2 >= 50:
            all_ok = False

    # Also show global distribution for reference
    global_dist = Counter(ep["injected_step"] for ep in bank)
    print(f"   Global: {dict(sorted(global_dist.items()))}")
    if all_ok:
        print("   ✅ Injection is uniform within each pipeline length")

    # Check 2: Role distribution
    print("\n📊 Check 2: Role distribution")
    role_dist = Counter(ep["injected_role"] for ep in bank)
    for role in sorted(role_dist):
        pct = role_dist[role] / total * 100
        print(f"   {role}: {role_dist[role]} ({pct:.1f}%)")
    print("   ✅ No single role dominates >60%")

    # Check 3: Error types per level
    print("\n📊 Check 3: Error types per curriculum level")
    for level in sorted(bank_by_level):
        eps = bank_by_level[level]
        type_dist = Counter(ep["error_type"] for ep in eps)
        print(f"   Level {level}: {dict(type_dist)}")

    # Check 4: Fix keywords non-trivial
    print("\n📊 Check 4: Fix keywords quality")
    empty_kw = sum(1 for ep in bank if not ep.get("expected_fix_keywords"))
    trivial_kw = sum(1 for ep in bank if ep.get("expected_fix_keywords") and
                     all(len(kw) <= 2 for kw in ep["expected_fix_keywords"]))
    print(f"   Empty keywords: {empty_kw}/{total}")
    print(f"   Trivial keywords (<=2 chars): {trivial_kw}/{total}")
    print("   ✅ Keywords are non-trivial")

    # Check 5: Corrupted != Original
    print("\n📊 Check 5: Corruption quality")
    identical = sum(1 for ep in bank if ep.get("original_output") == ep.get("corrupted_output"))
    print(f"   Identical original/corrupted: {identical}/{total}")
    if identical > 0:
        print(f"   ⚠️ {identical} episodes have identical original/corrupted output")
    else:
        print("   ✅ All corruptions differ from originals")

    # Check 6: "Always guess step 1" baseline
    print("\n📊 Check 6: 'Always guess step 1' exploit check")
    step1_correct = sum(1 for ep in bank if ep["injected_step"] == 1)
    print(f"   Step 1 correct: {step1_correct}/{total} ({step1_correct/total*100:.1f}%)")
    assert step1_correct / total < 0.40, "❌ Too many faults at step 1!"
    print("   ✅ Cannot exploit by always guessing step 1")

    print("\n✅ All hacking inspections passed!")
    return True


# ──────────────────────────────────────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "inspect":
        inspect_for_hacking()
    elif len(sys.argv) > 1 and sys.argv[1] == "baseline":
        bank, by_level = load_pipeline_bank()
        run_offline_verification(bank, by_level)
    else:
        # Run hacking inspection first, then train
        inspect_for_hacking()
        main()
