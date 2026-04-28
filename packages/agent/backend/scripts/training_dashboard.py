"""Human-readable training data dashboard. Called via `make training-stats`."""
import sys
from pathlib import Path

from scripts.extract_e2e_sessions import extract_sessions


def format_dashboard(stats: dict, target_sft: int = 800) -> str:
    sft = stats.get("sft_count", 0)
    dpo = stats.get("dpo_count", 0)
    threads = stats.get("threads_processed", 0)
    feedback_threads = stats.get("threads_with_feedback", 0)
    pct = round(sft * 100 / target_sft) if target_sft else 0
    feedback_rate = round(feedback_threads * 100 / threads) if threads else 0
    remaining = max(0, target_sft - sft)

    bar_width = 30
    filled = int(bar_width * min(sft, target_sft) / target_sft) if target_sft else 0
    bar = "█" * filled + "░" * (bar_width - filled)

    return f"""
训练数据飞轮日报
================================
SFT 样本:  {sft:>4} / {target_sft}  [{bar}]  {pct}%
DPO 对:    {dpo:>4}
线程处理:  {threads:>4}
有反馈的:  {feedback_threads:>4}  ({feedback_rate}%)
================================
距离 Phase 1 目标还差 {remaining} 条 SFT 样本
"""


if __name__ == "__main__":
    default_base = Path(__file__).resolve().parent.parent / ".deer-flow"
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else default_base
    stats = extract_sessions(base)
    print(format_dashboard(stats))
