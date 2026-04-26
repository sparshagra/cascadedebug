"""
CascadeDebug Training Monitor — Gradio 5.x UI
===============================================
Launches GRPO training in a daemon thread 5s after boot.
Streams logs and refreshes reward plots via gr.Timer.
"""

import sys
import io
import threading
from pathlib import Path

import gradio as gr
import train_gpu

RESULTS_DIR = Path(__file__).parent / "results"


class _TeeOutput(io.TextIOBase):
    """Mirror stdout to the Gradio log queue."""

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

# ---------------------------------------------------------------------------
# Training thread
# ---------------------------------------------------------------------------
_thread = None


def _launch_training():
    global _thread
    if _thread is None or not _thread.is_alive():
        _thread = threading.Thread(target=train_gpu.run, daemon=True, name="grpo-train")
        _thread.start()


threading.Timer(5.0, _launch_training).start()

# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _status() -> str:
    s = train_gpu.training_state["status"]
    history = train_gpu.reward_history
    n = len(history)

    if s == "idle":
        return "⏳ Initializing — training starts in ~5 s..."
    if s == "running":
        avg = sum(history[-30:]) / max(len(history[-30:]), 1) if history else 0.0
        return (
            f"🏃 Training — {n} completions / "
            f"{train_gpu.MAX_STEPS * train_gpu.NUM_GENERATIONS} total  |  "
            f"Avg: {avg:.3f}  |  Level: {train_gpu.curriculum_level}"
        )
    if s == "done":
        elapsed = train_gpu.training_state.get("elapsed", 0)
        return f"✅ Complete!  {elapsed / 60:.1f} min  |  {n} reward samples"
    if s == "error":
        return "❌ Error — check logs below"
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
        return combined[-15_000:] if len(combined) > 15_000 else combined
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


# ---------------------------------------------------------------------------
# Gradio layout
# ---------------------------------------------------------------------------
_css = """
.log-box textarea {
    font-family: 'Courier New', Consolas, monospace !important;
    font-size: 12px !important;
    background: #0d1117 !important;
    color: #c9d1d9 !important;
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
        # 🔬 CascadeDebug — GRPO Training Monitor

        **Model:** `Qwen2.5-3B-Instruct` (4-bit via Unsloth) &nbsp;|&nbsp;
        **Algorithm:** GRPO &nbsp;|&nbsp;
        **Steps:** 300 &nbsp;|&nbsp;
        **Group size:** 4

        > Training starts automatically. Logs and plots refresh every 10 seconds.
        """
    )

    status_box = gr.Textbox(
        value="⏳ Initializing...",
        label="Status",
        interactive=False,
    )

    with gr.Row():
        with gr.Column(scale=5):
            log_box = gr.Textbox(
                value="Waiting for training to start...",
                label="Training Logs",
                lines=28,
                max_lines=35,
                interactive=False,
                elem_classes=["log-box"],
            )
        with gr.Column(scale=3):
            reward_plot = gr.Image(label="Reward Curve", height=220, show_download_button=True)
            component_plot = gr.Image(label="Component Rewards", height=220, show_download_button=True)

    with gr.Row():
        loc_plot = gr.Image(label="Localization Accuracy", height=220, show_download_button=True)
        baseline_plot = gr.Image(label="Untrained vs Trained", height=220, show_download_button=True)

    timer = gr.Timer(10.0)
    timer.tick(
        fn=ui_tick,
        inputs=[log_box],
        outputs=[log_box, status_box, reward_plot, component_plot, loc_plot, baseline_plot],
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
