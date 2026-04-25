"""
CascadeDebug Training Monitor — Gradio 5.x UI
==============================================
• Imports train_gpu and launches training in a background daemon thread 3 s after boot.
• Streams logs and refreshes reward plots every 5 s via gr.Timer.
• Compatible with Gradio 5.x (sdk_version: 5.25.0).
"""

import sys
import io
import threading
from pathlib import Path

import gradio as gr

# Import the training module (sets up log_queue, training_state, etc.)
import train_gpu

RESULTS_DIR = Path(__file__).parent / "results"


# ── Capture plain print() calls from trainer / HF libraries ──────────────────
class _TeeOutput(io.TextIOBase):
    def __init__(self, original):
        self._orig = original

    def write(self, text: str) -> int:
        self._orig.write(text)
        stripped = text.strip()
        if stripped:
            train_gpu.log_queue.put(stripped)
        return len(text)

    def flush(self):
        self._orig.flush()


sys.stdout = _TeeOutput(sys.stdout)


# ── Training thread ───────────────────────────────────────────────────────────
_thread = None  # type: threading.Thread | None


def _launch_training():
    global _thread
    if _thread is None or not _thread.is_alive():
        _thread = threading.Thread(target=train_gpu.run, daemon=True, name="grpo-train")
        _thread.start()


threading.Timer(3.0, _launch_training).start()


# ── UI helper functions ───────────────────────────────────────────────────────
def _status() -> str:
    s       = train_gpu.training_state["status"]
    history = train_gpu.reward_history
    n       = len(history)

    if s == "idle":
        return "Initializing — training starts in ~3 s..."
    if s == "running":
        avg = sum(history[-30:]) / max(len(history[-30:]), 1) if history else 0.0
        return (
            f"Training — {n} completions / "
            f"{train_gpu.MAX_STEPS * train_gpu.NUM_GENERATIONS} total  |  "
            f"Rolling avg reward: {avg:.3f}  |  "
            f"Curriculum Level: {train_gpu.curriculum_level}"
        )
    if s == "done":
        elapsed = train_gpu.training_state.get("elapsed", 0)
        return f"Training Complete!  Duration: {elapsed/60:.1f} min  |  {n} reward samples"
    if s == "error":
        return "Error — inspect logs below"
    return "Unknown"


def _collect_logs(current: str) -> str:
    lines = []
    while not train_gpu.log_queue.empty():
        try:
            lines.append(train_gpu.log_queue.get_nowait())
        except Exception:
            break
    if lines:
        combined = current + "\n" + "\n".join(lines)
        return combined[-12_000:] if len(combined) > 12_000 else combined
    return current


def _plot(name: str):
    p = RESULTS_DIR / name
    return str(p) if p.exists() else None


def ui_tick(current_log: str):
    return (
        _collect_logs(current_log),
        _status(),
        _plot("reward_curve.png"),
        _plot("component_rewards.png"),
        _plot("localization_accuracy.png"),
        _plot("baseline_vs_trained.png"),
    )


# ── Gradio 5.x layout ─────────────────────────────────────────────────────────
_css = """
.log-box textarea {
    font-family: 'Courier New', Consolas, monospace !important;
    font-size: 12px !important;
    background: #0d1117 !important;
    color: #8b949e !important;
    border: 1px solid #30363d !important;
    border-radius: 6px;
    line-height: 1.5;
}
footer { display: none !important; }
"""

with gr.Blocks(
    title="CascadeDebug — GRPO Training",
    theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"),
    css=_css,
) as demo:

    gr.Markdown(
        """
        # CascadeDebug — GRPO Training Monitor

        **Model:** `Qwen2.5-7B-Instruct` (4-bit Unsloth) &nbsp;|&nbsp;
        **Algorithm:** GRPO &nbsp;|&nbsp;
        **GPU:** L4 (24GB VRAM) &nbsp;|&nbsp;
        **Steps:** 300 &nbsp;|&nbsp;
        **Group size:** 2

        > Training starts automatically. Logs and plots refresh every 5 seconds.
        > Results are pushed to `Dikshita2026/cascadedebug` on completion.
        """
    )

    status_box = gr.Textbox(
        value="Initializing — training starts in ~3 s...",
        label="Training Status",
        interactive=False,
    )

    with gr.Row():
        with gr.Column(scale=5):
            log_box = gr.Textbox(
                value="Waiting for training to start...",
                label="Training Logs",
                lines=30,
                max_lines=38,
                interactive=False,
                elem_classes=["log-box"],
            )
        with gr.Column(scale=3):
            reward_plot    = gr.Image(label="Reward Curve",          height=230, show_download_button=True)
            component_plot = gr.Image(label="Component Rewards",     height=230, show_download_button=True)

    with gr.Row():
        loc_plot      = gr.Image(label="Localization Accuracy",  height=230, show_download_button=True)
        baseline_plot = gr.Image(label="Untrained vs Trained",   height=230, show_download_button=True)

    timer = gr.Timer(5.0)
    timer.tick(
        fn=ui_tick,
        inputs=[log_box],
        outputs=[log_box, status_box, reward_plot, component_plot, loc_plot, baseline_plot],
    )


demo.launch(server_name="0.0.0.0", server_port=7860)
