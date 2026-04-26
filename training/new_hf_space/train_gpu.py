"""
CascadeDebug GRPO Training — HF Space (Docker SDK + L4)
========================================================
Runtime: Python 3.11, Unsloth 2025.11.4, PyTorch cu121 (see Dockerfile).
"""

import json
import os
import sys
import csv
import random
import re
import shutil
import time
import queue as _queue
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Stabilizers (before `import unsloth`). Docker uses Python 3.11 + Unsloth
# 2025.11.4; bf16 on L4 is the default path below. If you still see Half/Float
# in fast_lora, set secret UNSLOTH_FORCE_FLOAT32=1 on the Space.
# ---------------------------------------------------------------------------
os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")
os.environ.setdefault("UNSLOTH_DISABLE_AUTO_UPDATES", "1")

_STALE_CACHES = (
    Path("/app/unsloth_compiled_cache"),
    Path(__file__).parent / "unsloth_compiled_cache",
)
for _p in _STALE_CACHES:
    try:
        if _p.is_dir():
            shutil.rmtree(_p, ignore_errors=True)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Shared state (read by app.py Gradio UI)
# ---------------------------------------------------------------------------
log_queue: _queue.Queue = _queue.Queue()

training_state: dict = {
    "status": "idle",
    "elapsed": 0.0,
}

reward_history: list = []
all_log_rows: list = []

PIPELINE_BANK: list = []
BANK_BY_LEVEL: dict = {}

curriculum_level: int = 1
step_counter: dict = {"count": 0}


def log(msg: str) -> None:
    print(msg, flush=True)
    log_queue.put(str(msg))


def _resolve_grpo_precision(torch_module):
    """
    Prefer bf16 on L4 with Py3.11 + Unsloth 2025.11.4. Never fp16 in TRL for
    this 4-bit GRPO stack. Optional: UNSLOTH_FORCE_FLOAT32=1 -> full fp32.
    """
    if os.environ.get("UNSLOTH_FORCE_FLOAT32") == "1":
        return torch_module.float32, False, False
    if torch_module.cuda.is_bf16_supported():
        return torch_module.bfloat16, True, False
    return torch_module.float32, False, False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_NAME = "unsloth/Qwen2.5-3B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048
LORA_R = 16
LORA_ALPHA = 16

MAX_STEPS = 500
NUM_GENERATIONS = 4
MAX_COMPLETION_LEN = 384
LEARNING_RATE = 5e-6
GRADIENT_ACCUM = 4
LOG_EVERY = 5
SAVE_EVERY = 50
PLOT_EVERY = 50

CURRICULUM_THRESHOLDS = {1: 0.4, 2: 0.6, 3: 0.7}
ROLLING_WINDOW = 30

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_RESULTS_REPO = "Dikshita2026/cascadedebug"

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are CascadeDebug, an expert at finding and fixing errors in "
    "multi-step AI pipelines.\n\n"
    "You will see a pipeline of steps, each produced by a different role "
    "(Researcher, Coder, or Analyst). Exactly ONE step contains a silent "
    "error. Your job:\n"
    "1. Identify which step has the error (fault_step_id)\n"
    "2. Identify which role produced it (blame_role)\n"
    "3. Provide the corrected output (fix_content)\n\n"
    "Respond ONLY with valid JSON:\n"
    '{"fault_step_id": <int>, "blame_role": "<Researcher|Coder|Analyst>", '
    '"fix_content": "<corrected output>", "action_type": "submit"}'
)


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------
def format_episode_prompt(episode: dict, tokenizer) -> str:
    pipeline = episode["corrupted_pipeline"]
    steps_text = ""
    for step in pipeline:
        steps_text += (
            f"\n--- Step {step['step_id']} ({step['role']}) ---\n"
            f"{step['output']}\n"
        )
    user_msg = (
        f"Task: {episode['task_brief']}\n\n"
        f"Pipeline output (one step has a silent error):\n{steps_text}\n"
        "Find the faulty step and provide the corrected output as JSON."
    )
    return tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def build_dataset(cur_level: int, tokenizer, n_samples: int = 500):
    from datasets import Dataset

    episodes = BANK_BY_LEVEL.get(cur_level, PIPELINE_BANK)
    data = []
    for i in range(n_samples):
        ep = episodes[i % len(episodes)]
        prompt = format_episode_prompt(ep, tokenizer)
        data.append({"prompt": prompt, "episode_idx": i % len(episodes)})
    return Dataset.from_list(data)


# ---------------------------------------------------------------------------
# Action parser
# ---------------------------------------------------------------------------
def parse_action(text: str) -> dict:
    try:
        match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if match:
            action = json.loads(match.group())
            if all(k in action for k in ("fault_step_id", "blame_role", "fix_content")):
                action.setdefault("action_type", "submit")
                action["fault_step_id"] = int(action["fault_step_id"])
                if action["blame_role"] in ("Researcher", "Coder", "Analyst"):
                    return action
    except Exception:
        pass
    try:
        sm = re.search(r'fault_step_id["\s:]+(\d+)', text)
        rm = re.search(r'blame_role["\s:]+["\']?(Researcher|Coder|Analyst)', text)
        fm = re.search(r'fix_content["\s:]+["\'](.+?)["\']', text, re.DOTALL)
        return {
            "fault_step_id": int(sm.group(1)) if sm else 1,
            "blame_role": rm.group(1) if rm else "Researcher",
            "fix_content": fm.group(1) if fm else "default fix",
            "action_type": "submit",
        }
    except Exception:
        return {
            "fault_step_id": 1,
            "blame_role": "Researcher",
            "fix_content": "parse error",
            "action_type": "submit",
        }


# ---------------------------------------------------------------------------
# Reward functions (self-contained, identical to server/rewards.py weights)
# ---------------------------------------------------------------------------
def reward_localization(predicted: int, true: int, level: int = 1) -> float:
    if predicted == true:
        return 1.0
    if level == 1 and abs(predicted - true) == 1:
        return 0.3
    return 0.0


def reward_blame(predicted: str, true: str) -> float:
    return 1.0 if predicted == true else 0.0


def reward_fix(
    fix_content: str,
    original_output: str,
    expected_keywords: list,
    corrupted_output: str,
) -> float:
    if not fix_content or not fix_content.strip():
        return 0.0
    fl = fix_content.lower()
    kw_score = (
        sum(1 for kw in expected_keywords if kw.lower() in fl) / len(expected_keywords)
        if expected_keywords
        else 0.5
    )
    nc_score = 0.0 if (corrupted_output and fl == corrupted_output.lower()) else 1.0
    if original_output:
        ratio = len(fix_content) / max(len(original_output), 1)
        length_score = (
            1.0
            if 0.3 <= ratio <= 3.0
            else max(0.0, min(ratio / 0.3, 1.0 - (ratio - 3.0) / 7.0))
        )
        orig_words = set(original_output.lower().split())
        fix_words = set(fl.split())
        sim_score = len(orig_words & fix_words) / max(len(orig_words), 1)
    else:
        length_score = 0.5
        sim_score = 0.0
    return min(
        max(0.50 * kw_score + 0.20 * nc_score + 0.15 * length_score + 0.15 * sim_score, 0.0),
        1.0,
    )


def reward_precision(turn: int = 1, gatekeeper_accepted: bool = False) -> float:
    if gatekeeper_accepted:
        return 1.0 if turn == 1 else 0.6
    return 0.2


def compute_episode_reward(action: dict, episode: dict, level: int = 1) -> dict:
    r1 = reward_localization(action["fault_step_id"], episode["injected_step"], level)
    r2 = reward_blame(action["blame_role"], episode["injected_role"])
    r3 = reward_fix(
        action["fix_content"],
        episode.get("original_output", ""),
        episode.get("expected_fix_keywords", []),
        episode.get("corrupted_output", ""),
    )
    r4 = reward_precision(1, False)
    total = 0.35 * r1 + 0.20 * r2 + 0.35 * r3 + 0.10 * r4
    return {
        "localization": r1,
        "blame": r2,
        "fix": r3,
        "precision": r4,
        "total": total,
    }


# ---------------------------------------------------------------------------
# GRPO reward callback (called by TRL trainer)
# ---------------------------------------------------------------------------
def cascadedebug_reward(completions, **kwargs):
    global curriculum_level

    episode_list = BANK_BY_LEVEL.get(curriculum_level, PIPELINE_BANK)
    rewards: list = []

    for i, completion in enumerate(completions):
        if isinstance(completion, list):
            text = completion[-1].get("content", "") if completion else ""
        elif isinstance(completion, str):
            text = completion
        else:
            text = str(completion)

        ep_idx = (step_counter["count"] + i) % len(episode_list)
        ep = episode_list[ep_idx]

        action = parse_action(text)
        result = compute_episode_reward(action, ep, curriculum_level)
        total = result["total"]

        rewards.append(total)
        reward_history.append(total)

        all_log_rows.append(
            {
                "step": len(reward_history),
                "level": curriculum_level,
                "r_loc": result["localization"],
                "r_blame": result["blame"],
                "r_fix": result["fix"],
                "r_prec": result["precision"],
                "r_total": total,
                "predicted_step": action["fault_step_id"],
                "true_step": ep["injected_step"],
                "predicted_role": action["blame_role"],
                "true_role": ep["injected_role"],
            }
        )

    step_counter["count"] += len(completions)

    if len(reward_history) >= ROLLING_WINDOW and len(reward_history) % 20 == 0:
        avg = sum(reward_history[-ROLLING_WINDOW:]) / ROLLING_WINDOW
        threshold = CURRICULUM_THRESHOLDS.get(curriculum_level, 1.0)
        if avg > threshold and curriculum_level < 3:
            curriculum_level += 1
            log(f"🎓 CURRICULUM ADVANCED → Level {curriculum_level}! (rolling avg={avg:.3f})")

    if (
        len(reward_history) >= (PLOT_EVERY * NUM_GENERATIONS)
        and len(reward_history) % (PLOT_EVERY * NUM_GENERATIONS) < NUM_GENERATIONS
    ):
        try:
            _save_interim_plot()
        except Exception as exc:
            log(f"⚠️ Interim plot error: {exc}")

    return rewards


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _save_interim_plot() -> None:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(reward_history) < 5:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(reward_history, alpha=0.25, color="#3b82f6", linewidth=0.8, label="Per-completion")
    if len(reward_history) > 10:
        w = min(30, len(reward_history) // 3)
        rolling = np.convolve(reward_history, np.ones(w) / w, mode="valid")
        ax.plot(
            range(w - 1, len(reward_history)),
            rolling,
            color="#ef4444",
            linewidth=2,
            label=f"Rolling avg (w={w})",
        )
    avg = sum(reward_history[-ROLLING_WINDOW:]) / min(len(reward_history), ROLLING_WINDOW)
    ax.set_xlabel("Training Completion", fontsize=12)
    ax.set_ylabel("Total Reward", fontsize=12)
    ax.set_title(
        f"CascadeDebug — GRPO  |  Completions: {len(reward_history)}  "
        f"|  Avg: {avg:.3f}  |  Level: {curriculum_level}",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "reward_curve.png", dpi=110)
    plt.close(fig)


def generate_final_plots() -> None:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log("Generating final plots...")
    plt.close("all")

    # Plot 1: Full reward curve
    if reward_history:
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(reward_history, alpha=0.2, color="#3b82f6", linewidth=0.5, label="Per-completion")
        if len(reward_history) > 20:
            w = min(50, len(reward_history) // 5)
            rolling = np.convolve(reward_history, np.ones(w) / w, mode="valid")
            ax.plot(
                range(w - 1, len(reward_history)),
                rolling,
                color="#ef4444",
                linewidth=2.5,
                label=f"Rolling avg (w={w})",
            )
        ax.set_xlabel("Training Completion", fontsize=13)
        ax.set_ylabel("Total Reward", fontsize=13)
        ax.set_title("CascadeDebug — GRPO Training Reward Curve", fontsize=15, fontweight="bold")
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "reward_curve.png", dpi=150)
        plt.close(fig)
        log("   ✅ reward_curve.png")

    # Plot 2: Component rewards
    if all_log_rows:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        components = [
            ("r_loc", "Fault Localization (r1, w=0.35)", "#3b82f6"),
            ("r_blame", "Blame Attribution (r2, w=0.20)", "#10b981"),
            ("r_fix", "Fix Correctness (r3, w=0.35)", "#f59e0b"),
            ("r_prec", "Surgical Precision (r4, w=0.10)", "#8b5cf6"),
        ]
        for ax, (key, title, color) in zip(axes.flat, components):
            vals = [r[key] for r in all_log_rows]
            ax.plot(vals, alpha=0.2, color=color, linewidth=0.5)
            if len(vals) > 20:
                w = min(50, len(vals) // 5)
                rolling = np.convolve(vals, np.ones(w) / w, mode="valid")
                ax.plot(range(w - 1, len(vals)), rolling, color=color, linewidth=2)
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_xlabel("Step")
            ax.set_ylabel("Reward")
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
        plt.suptitle("CascadeDebug — Component Reward Signals", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "component_rewards.png", dpi=150)
        plt.close(fig)
        log("   ✅ component_rewards.png")

    # Plot 3: Localization accuracy
    if all_log_rows:
        fig, ax = plt.subplots(figsize=(10, 5))
        correct = [1 if r["predicted_step"] == r["true_step"] else 0 for r in all_log_rows]
        if len(correct) > 20:
            w = min(50, len(correct) // 5)
            rolling = np.convolve(correct, np.ones(w) / w, mode="valid")
            ax.plot(
                range(w - 1, len(correct)),
                rolling,
                color="#3b82f6",
                linewidth=2,
                label="Rolling accuracy",
            )
        ax.axhline(y=1 / 3, color="#ef4444", linestyle="--", linewidth=1.5, label="Random (1/3)")
        ax.set_xlabel("Training Step", fontsize=13)
        ax.set_ylabel("Localization Accuracy", fontsize=13)
        ax.set_title(
            "CascadeDebug — Fault Localization Accuracy",
            fontsize=15,
            fontweight="bold",
        )
        ax.set_ylim(0, 1)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "localization_accuracy.png", dpi=150)
        plt.close(fig)
        log("   ✅ localization_accuracy.png")

    # Plot 4: Baseline vs trained
    if all_log_rows and len(all_log_rows) > 20:
        fig, ax = plt.subplots(figsize=(10, 6))
        n = len(all_log_rows)
        n5 = max(n // 5, 10)
        for label, rows, offset, color in [
            ("Untrained (first 20%)", all_log_rows[:n5], -0.15, "#94a3b8"),
            ("Trained (last 20%)", all_log_rows[-n5:], 0.15, "#3b82f6"),
        ]:
            by_level = {}
            for r in rows:
                by_level.setdefault(r["level"], []).append(r["r_total"])
            avgs = [
                sum(by_level.get(l, [0])) / max(len(by_level.get(l, [1])), 1)
                for l in [1, 2, 3]
            ]
            x = np.arange(3) + offset
            bars = ax.bar(x, avgs, width=0.28, label=label, color=color)
            for bar, avg in zip(bars, avgs):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{avg:.3f}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=10,
                )
        ax.set_xticks(range(3))
        ax.set_xticklabels(
            ["Level 1\n(3-step, easy)", "Level 2\n(4-step, medium)", "Level 3\n(5-6 step, hard)"]
        )
        ax.set_ylabel("Average Reward", fontsize=13)
        ax.set_title("CascadeDebug — Untrained vs Trained Agent", fontsize=15, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "baseline_vs_trained.png", dpi=150)
        plt.close(fig)
        log("   ✅ baseline_vs_trained.png")

    plt.close("all")


# ---------------------------------------------------------------------------
# Push results to HF Hub
# ---------------------------------------------------------------------------
def push_to_hub(model, tokenizer) -> None:
    if not HF_TOKEN:
        log("⚠️ HF_TOKEN not set — skipping Hub push")
        return
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=HF_TOKEN)
        log("\n📤 Pushing results to HF Hub...")

        for fname in [
            "reward_curve.png",
            "component_rewards.png",
            "localization_accuracy.png",
            "baseline_vs_trained.png",
            "training_log.csv",
        ]:
            fpath = RESULTS_DIR / fname
            if fpath.exists():
                try:
                    api.upload_file(
                        path_or_fileobj=str(fpath),
                        path_in_repo=f"results/{fname}",
                        repo_id=HF_RESULTS_REPO,
                        repo_type="space",
                        commit_message=f"Training results: {fname}",
                    )
                    log(f"   ✅ {fname} → {HF_RESULTS_REPO}")
                except Exception as e:
                    log(f"   ⚠️ {fname} push failed: {e}")

        merged_dir = str(RESULTS_DIR / "merged_model")
        try:
            model.save_pretrained_merged(
                merged_dir,
                tokenizer,
                save_method="merged_16bit",
            )
            log(f"   Merged 16bit model saved -> {merged_dir}")
            try:
                api.create_repo(
                    repo_id="Dikshita2026/cascadedebug-model",
                    repo_type="model",
                    exist_ok=True,
                )
                api.upload_folder(
                    folder_path=merged_dir,
                    repo_id="Dikshita2026/cascadedebug-model",
                    repo_type="model",
                    commit_message="CascadeDebug: merged GRPO-trained model",
                )
                log("   Merged model → Dikshita2026/cascadedebug-model")
            except Exception as hub_err:
                log(f"   Model Hub push skipped: {hub_err}")
        except Exception as merge_err:
            log(f"   Merged save failed, saving LoRA adapter: {merge_err}")
            lora_dir = str(RESULTS_DIR / "lora_adapter")
            model.save_pretrained(lora_dir)
            tokenizer.save_pretrained(lora_dir)
            log(f"   LoRA adapter saved -> {lora_dir}")

    except Exception as e:
        log(f"⚠️ Hub push error: {e}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def run() -> None:
    global curriculum_level

    training_state["status"] = "running"

    try:
        log("=" * 60)
        log("CascadeDebug GRPO Training — HF Spaces L4 GPU")
        log("=" * 60)

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "No CUDA GPU detected. Make sure the Space hardware is set to L4 or T4."
            )

        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        log(f"✅ GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()

        compute_dtype, use_bf16, use_fp16 = _resolve_grpo_precision(torch)
        if os.environ.get("UNSLOTH_FORCE_FLOAT32") == "1":
            log("   Precision: float32 (UNSLOTH_FORCE_FLOAT32=1)")
        elif use_bf16:
            log("   Precision: bf16 (Py3.11 + Unsloth 2025.11.x, TRL bf16=True)")
        else:
            log(f"   Precision: {compute_dtype} | TRL bf16={use_bf16} fp16={use_fp16}")

        # Load pipeline bank
        log("\n📦 Loading pipeline bank...")
        bank_path = DATA_DIR / "pipeline_bank.json"
        if not bank_path.exists():
            raise FileNotFoundError(f"Pipeline bank not found at {bank_path}")

        with open(bank_path) as f:
            PIPELINE_BANK[:] = json.load(f)

        BANK_BY_LEVEL.clear()
        for ep in PIPELINE_BANK:
            BANK_BY_LEVEL.setdefault(ep["curriculum_level"], []).append(ep)

        for lvl, eps in sorted(BANK_BY_LEVEL.items()):
            log(f"   Level {lvl}: {len(eps)} episodes")

        # Load model
        log(f"\n🤖 Loading {MODEL_NAME} (4-bit via Unsloth)...")
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=True,
            dtype=compute_dtype,
            fast_inference=False,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )

        free_gb = (
            torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        ) / 1024**3
        log(f"✅ Model + LoRA ready | Free VRAM: {free_gb:.1f} GB")

        # Build training dataset
        log(f"\n📊 Building dataset (Level {curriculum_level}, 500 samples)...")
        train_dataset = build_dataset(curriculum_level, tokenizer, n_samples=500)
        log(f"   {len(train_dataset)} prompts ready")

        # GRPO trainer
        from trl import GRPOConfig, GRPOTrainer

        training_args = GRPOConfig(
            output_dir=str(RESULTS_DIR / "grpo_checkpoints"),
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=1,
            num_generations=NUM_GENERATIONS,
            max_completion_length=MAX_COMPLETION_LEN,
            max_prompt_length=MAX_SEQ_LENGTH - MAX_COMPLETION_LEN,
            max_steps=MAX_STEPS,
            logging_steps=LOG_EVERY,
            save_steps=SAVE_EVERY,
            fp16=use_fp16,
            bf16=use_bf16,
            gradient_accumulation_steps=GRADIENT_ACCUM,
            warmup_ratio=0.1,
            optim="adamw_8bit",
            seed=42,
            report_to="none",
            remove_unused_columns=False,
            log_level="warning",
            dataloader_num_workers=2,
            dataloader_pin_memory=True,
        )

        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=cascadedebug_reward,
            args=training_args,
            train_dataset=train_dataset,
        )

        log(f"\n🚀 GRPO Training started!")
        log(f"   Steps: {MAX_STEPS} | Group size: {NUM_GENERATIONS}")
        log(f"   Total completions: {MAX_STEPS * NUM_GENERATIONS}")
        log(f"   Estimated: ~2-4 hours on L4\n")

        start = time.time()
        trainer.train()
        elapsed = time.time() - start
        training_state["elapsed"] = elapsed

        log(f"\n✅ Training done in {elapsed / 60:.1f} min ({len(reward_history)} samples)")

        # Save CSV log
        if all_log_rows:
            log_path = RESULTS_DIR / "training_log.csv"
            with open(log_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_log_rows[0].keys())
                writer.writeheader()
                writer.writerows(all_log_rows)
            log(f"   📝 CSV: {len(all_log_rows)} rows")

        # Save LoRA adapter
        log("\n💾 Saving LoRA adapter...")
        model.save_pretrained(str(RESULTS_DIR / "lora_adapter"))
        tokenizer.save_pretrained(str(RESULTS_DIR / "lora_adapter"))
        log("   ✅ LoRA adapter saved")

        generate_final_plots()
        push_to_hub(model, tokenizer)

        # Summary
        if reward_history:
            n = len(reward_history)
            n5 = max(n // 5, 1)
            eavg = sum(reward_history[:n5]) / n5
            lavg = sum(reward_history[-n5:]) / n5
            log("\n" + "=" * 60)
            log("🏆 TRAINING SUMMARY")
            log("=" * 60)
            log(f"   Steps:          {MAX_STEPS}")
            log(f"   Duration:       {elapsed / 60:.1f} min")
            log(f"   Early avg:      {eavg:.4f}")
            log(f"   Late avg:       {lavg:.4f}")
            log(f"   Improvement:    {lavg - eavg:+.4f}")
            if all_log_rows:
                late_rows = all_log_rows[-len(all_log_rows) // 5 :]
                loc_acc = sum(
                    1 for r in late_rows if r["predicted_step"] == r["true_step"]
                ) / max(len(late_rows), 1)
                log(f"   Final loc acc:  {loc_acc:.2%}")
            log("=" * 60)

        training_state["status"] = "done"

    except Exception:
        err = traceback.format_exc()
        log(f"❌ Fatal error:\n{err}")
        training_state["status"] = "error"
        raise
