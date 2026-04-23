import json
from pathlib import Path

from scripts.extract_e2e_sessions import extract_sessions


def _write_jsonl(path: Path, items: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def test_extract_joins_lead_samples_and_correct_feedback(tmp_path):
    _write_jsonl(
        tmp_path / "training-data" / "auto-collected" / "t-1.jsonl",
        [
            {
                "role": "lead",
                "thread_id": "t-1",
                "input": "hi",
                "output": "ok",
                "thinking": "",
                "recorded_at": "2026-04-23T00:00:00+00:00",
            },
        ],
    )
    _write_jsonl(
        tmp_path / "training-data" / "feedback" / "t-1.jsonl",
        [
            {
                "thread_id": "t-1",
                "message_id": "m-1",
                "verdict": "correct",
                "revised_text": None,
                "note": None,
                "submitted_at": "2026-04-23T00:01:00+00:00",
            }
        ],
    )

    stats = extract_sessions(tmp_path)

    sft = [
        json.loads(l)
        for l in (tmp_path / "training-data" / "processed" / "sft.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(sft) >= 1
    assert stats["sft_count"] >= 1
    assert stats["dpo_count"] == 0


def test_extract_generates_dpo_pair_from_needs_fix(tmp_path):
    _write_jsonl(
        tmp_path / "training-data" / "auto-collected" / "t-2.jsonl",
        [
            {
                "role": "lead",
                "thread_id": "t-2",
                "input": "分析",
                "output": "泛泛回答",
                "thinking": "",
                "recorded_at": "2026-04-23T00:00:00+00:00",
            },
        ],
    )
    _write_jsonl(
        tmp_path / "training-data" / "feedback" / "t-2.jsonl",
        [
            {
                "thread_id": "t-2",
                "message_id": "m-1",
                "verdict": "needs_fix",
                "revised_text": "专家级解读",
                "note": "加上 p 值",
                "submitted_at": "2026-04-23T00:01:00+00:00",
            }
        ],
    )

    stats = extract_sessions(tmp_path)

    dpo = [
        json.loads(l)
        for l in (tmp_path / "training-data" / "processed" / "dpo.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(dpo) == 1
    assert dpo[0]["chosen"] == "专家级解读"
    assert dpo[0]["rejected"] == "泛泛回答"
    assert stats["dpo_count"] == 1


def test_extract_excludes_wrong_verdict_without_revision(tmp_path):
    """verdict=wrong + no revised_text → sample dropped entirely (not in SFT or DPO)."""
    _write_jsonl(
        tmp_path / "training-data" / "auto-collected" / "t-3.jsonl",
        [
            {
                "role": "lead",
                "thread_id": "t-3",
                "input": "q",
                "output": "bad answer",
                "thinking": "",
                "recorded_at": "2026-04-23T00:00:00+00:00",
            },
        ],
    )
    _write_jsonl(
        tmp_path / "training-data" / "feedback" / "t-3.jsonl",
        [
            {
                "thread_id": "t-3",
                "message_id": "m-1",
                "verdict": "wrong",
                "revised_text": None,
                "note": None,
                "submitted_at": "2026-04-23T00:01:00+00:00",
            }
        ],
    )

    stats = extract_sessions(tmp_path)
    assert stats["sft_count"] == 0
    assert stats["dpo_count"] == 0


def test_extract_no_feedback_still_writes_sft(tmp_path):
    """Samples with no feedback are still included in SFT (low-confidence label)."""
    _write_jsonl(
        tmp_path / "training-data" / "auto-collected" / "t-4.jsonl",
        [
            {
                "role": "lead",
                "thread_id": "t-4",
                "input": "q",
                "output": "answer",
                "thinking": "",
                "recorded_at": "2026-04-23T00:00:00+00:00",
            },
        ],
    )

    stats = extract_sessions(tmp_path)
    assert stats["sft_count"] >= 1
