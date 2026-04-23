"""Post-process training data: join auto-collected samples with feedback.

Outputs:
- training-data/processed/sft.jsonl  — Fireworks ChatML
- training-data/processed/dpo.jsonl  — preference pairs
- training-data/processed/stats.json — counts for daily dashboard

Usage:
    python scripts/extract_e2e_sessions.py [base_dir]
    base_dir defaults to backend/.deer-flow/
"""
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sft_record(sample: dict, output_text: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": sample["input"]},
            {
                "role": "assistant",
                "content": output_text,
                "thinking": sample.get("thinking", ""),
            },
        ],
        "metadata": {
            "thread_id": sample["thread_id"],
            "role": sample["role"],
        },
    }


def extract_sessions(base_dir: Path) -> dict:
    """Join recorded samples with feedback and write processed JSONL files.

    Returns a stats dict with sft_count, dpo_count, threads_processed,
    threads_with_feedback.
    """
    collected_dir = base_dir / "training-data" / "auto-collected"
    feedback_dir = base_dir / "training-data" / "feedback"
    out_dir = base_dir / "training-data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build per-thread feedback index: {thread_id: [items]}
    feedback_by_thread: dict[str, list[dict]] = {}
    if feedback_dir.exists():
        for f in sorted(feedback_dir.glob("*.jsonl")):
            feedback_by_thread[f.stem] = _read_jsonl(f)

    sft: list[dict] = []
    dpo: list[dict] = []

    threads_processed = 0
    if collected_dir.exists():
        for f in sorted(collected_dir.glob("*.jsonl")):
            thread_id = f.stem
            samples = _read_jsonl(f)
            if not samples:
                continue
            threads_processed += 1
            feedbacks = feedback_by_thread.get(thread_id, [])

            # v0.1 join: first feedback applies to all samples in the thread.
            # (frontend sends real message_id but we don't yet match per-message;
            # improve once we have enough data to warrant the join logic.)
            fb = next(iter(feedbacks), None)

            for sample in samples:
                if fb is None:
                    # No feedback yet — include as SFT with no quality signal
                    sft.append(_sft_record(sample, sample["output"]))
                    continue

                verdict = fb["verdict"]
                if verdict == "correct":
                    sft.append(_sft_record(sample, sample["output"]))
                elif fb.get("revised_text"):
                    # Both SFT (revised text) AND DPO preference pair
                    sft.append(_sft_record(sample, fb["revised_text"]))
                    dpo.append({
                        "prompt": sample["input"],
                        "chosen": fb["revised_text"],
                        "rejected": sample["output"],
                        "metadata": {"thread_id": thread_id, "verdict": verdict},
                    })
                # verdict == "wrong" with no revision → discard entirely

    def _write(path: Path, records: list[dict]) -> None:
        path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
            + ("\n" if records else ""),
            encoding="utf-8",
        )

    _write(out_dir / "sft.jsonl", sft)
    _write(out_dir / "dpo.jsonl", dpo)

    stats = {
        "generated_at": datetime.now(UTC).isoformat(),
        "sft_count": len(sft),
        "dpo_count": len(dpo),
        "threads_processed": threads_processed,
        "threads_with_feedback": len(feedback_by_thread),
    }
    (out_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return stats


if __name__ == "__main__":
    default_base = Path(__file__).resolve().parent.parent / ".deer-flow"
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else default_base
    result = extract_sessions(base)
    print(json.dumps(result, indent=2, ensure_ascii=False))
