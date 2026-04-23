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
                "message_id": "m-1",
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
        json.loads(line)
        for line in (tmp_path / "training-data" / "processed" / "sft.jsonl")
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
                "message_id": "m-1",
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
        json.loads(line)
        for line in (tmp_path / "training-data" / "processed" / "dpo.jsonl")
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
                "message_id": "m-1",
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


def test_extract_joins_feedback_by_message_id(tmp_path):
    """Feedback with message_id=m-1 applies ONLY to the sample with that id.
    Other samples in the same thread (m-2, m-3) must go the no-feedback path."""
    _write_jsonl(
        tmp_path / "training-data" / "auto-collected" / "t-multi.jsonl",
        [
            {
                "role": "lead",
                "thread_id": "t-multi",
                "message_id": "m-1",
                "input": "q1",
                "output": "bad answer",
                "thinking": "",
                "recorded_at": "2026-04-23T00:00:00+00:00",
            },
            {
                "role": "lead",
                "thread_id": "t-multi",
                "message_id": "m-2",
                "input": "q2",
                "output": "answer 2",
                "thinking": "",
                "recorded_at": "2026-04-23T00:01:00+00:00",
            },
            {
                "role": "subagent",
                "thread_id": "t-multi",
                "message_id": "subtask-call-99",
                "subagent_type": "code-executor",
                "input": "{}",
                "output": "subagent output",
                "recorded_at": "2026-04-23T00:02:00+00:00",
            },
        ],
    )
    _write_jsonl(
        tmp_path / "training-data" / "feedback" / "t-multi.jsonl",
        [
            {
                "thread_id": "t-multi",
                "message_id": "m-1",
                "verdict": "needs_fix",
                "revised_text": "good answer",
                "note": None,
                "submitted_at": "2026-04-23T00:03:00+00:00",
            },
            {
                "thread_id": "t-multi",
                "message_id": "subtask-call-99",
                "verdict": "wrong",
                "revised_text": None,
                "note": None,
                "submitted_at": "2026-04-23T00:04:00+00:00",
            },
        ],
    )

    stats = extract_sessions(tmp_path)

    sft = [
        json.loads(line)
        for line in (tmp_path / "training-data" / "processed" / "sft.jsonl")
        .read_text()
        .splitlines()
    ]
    dpo = [
        json.loads(line)
        for line in (tmp_path / "training-data" / "processed" / "dpo.jsonl")
        .read_text()
        .splitlines()
    ]

    # m-1: needs_fix with revision -> SFT (revised) + DPO
    # m-2: no feedback -> SFT (raw output)
    # subtask-call-99: wrong, no revision -> dropped
    outputs = sorted(record["messages"][1]["content"] for record in sft)
    assert outputs == ["answer 2", "good answer"]
    assert len(dpo) == 1
    assert dpo[0]["chosen"] == "good answer"
    assert dpo[0]["rejected"] == "bad answer"
    assert stats["sft_count"] == 2
    assert stats["dpo_count"] == 1


def test_extract_latest_feedback_wins_for_same_message(tmp_path):
    """If the expert clicks feedback twice on the same message, the most
    recent verdict wins (submitted_at is monotonic; later overrides)."""
    _write_jsonl(
        tmp_path / "training-data" / "auto-collected" / "t-redo.jsonl",
        [
            {
                "role": "lead",
                "thread_id": "t-redo",
                "message_id": "m-1",
                "input": "q",
                "output": "draft",
                "thinking": "",
                "recorded_at": "2026-04-23T00:00:00+00:00",
            },
        ],
    )
    _write_jsonl(
        tmp_path / "training-data" / "feedback" / "t-redo.jsonl",
        [
            {
                "thread_id": "t-redo",
                "message_id": "m-1",
                "verdict": "wrong",
                "revised_text": None,
                "note": None,
                "submitted_at": "2026-04-23T00:01:00+00:00",
            },
            {
                "thread_id": "t-redo",
                "message_id": "m-1",
                "verdict": "correct",
                "revised_text": None,
                "note": None,
                "submitted_at": "2026-04-23T00:02:00+00:00",
            },
        ],
    )

    stats = extract_sessions(tmp_path)
    # Latest verdict is "correct", so sample should be kept as SFT with raw output
    assert stats["sft_count"] == 1
    assert stats["dpo_count"] == 0


def test_extract_legacy_samples_without_message_id_treated_as_no_feedback(tmp_path):
    """Old recorded samples don't have message_id — they still get included
    as SFT with no feedback signal rather than being silently dropped."""
    _write_jsonl(
        tmp_path / "training-data" / "auto-collected" / "t-legacy.jsonl",
        [
            {
                "role": "lead",
                "thread_id": "t-legacy",
                # note: no message_id
                "input": "q",
                "output": "a",
                "thinking": "",
                "recorded_at": "2026-04-23T00:00:00+00:00",
            },
        ],
    )
    _write_jsonl(
        tmp_path / "training-data" / "feedback" / "t-legacy.jsonl",
        [
            {
                "thread_id": "t-legacy",
                "message_id": "m-does-not-match",
                "verdict": "wrong",
                "revised_text": None,
                "note": None,
                "submitted_at": "2026-04-23T00:01:00+00:00",
            }
        ],
    )

    stats = extract_sessions(tmp_path)
    # Legacy sample has no message_id so the feedback does not apply to it;
    # we keep the sample as SFT (raw output) rather than losing it.
    assert stats["sft_count"] == 1
    assert stats["dpo_count"] == 0
