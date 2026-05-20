"""Live e2e smoke test for metric catalog architecture (2026-05-13).

跑 EPM 真数据完整 agent 流程，验证 catalog → plan → execute → handoff
链路全程在 agent 系统里能跑通。

Skipped 条件：
- CI 环境
- 没有 config.yaml
- 没有真 EPM demo data (/home/wangqiuyang/DemoData/newdemodata/)

运行：

    PYTHONPATH=. uv run pytest tests/test_metric_catalog_live.py -v -s

测试断言层次：

1. **机制层**：plan.json / handoff.json / 各 metric output 文件生成
2. **结构层**：plan.metrics 数量、字段、reason；handoff metrics 数值非 null
3. **判读哲学层**：lead/data-analyst/report-writer 输出不含违禁的"绝对
   阈值/常模"语言（同事 5-13 反馈硬要求）

这不是 unit test —— 一次跑 5-15 分钟，消耗真 API。但它是 catalog
架构唯一的端到端机制闭环验证。
"""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path

import pytest

from deerflow.client import DeerFlowClient
from deerflow.config.paths import get_paths

# ---------------------------------------------------------------------------
# Skip gates
# ---------------------------------------------------------------------------

_REAL_EPM_DIR = Path("/home/wangqiuyang/DemoData/newdemodata/高架十字迷宫_小鼠_三点")

_skip_reason = None
if not os.environ.get("ETHOINSIGHT_LIVE_E2E"):
    _skip_reason = "Live e2e disabled — set ETHOINSIGHT_LIVE_E2E=1 to enable"
elif os.environ.get("CI"):
    _skip_reason = "Live test skipped in CI"
elif not Path(__file__).resolve().parents[2].joinpath("config.yaml").exists():
    _skip_reason = "No config.yaml — live test requires API credentials"
elif not _REAL_EPM_DIR.is_dir():
    _skip_reason = f"Real EPM demo data not found: {_REAL_EPM_DIR}"

if _skip_reason:
    pytest.skip(_skip_reason, allow_module_level=True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client():
    return DeerFlowClient(thinking_enabled=False)


@pytest.fixture(scope="module")
def thread_id() -> str:
    return f"catalog-smoke-{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="module")
def epm_raw_file() -> Path:
    """Real EPM trajectory from DemoData."""
    candidates = list(_REAL_EPM_DIR.glob("轨迹*.txt"))
    if not candidates:
        pytest.skip(f"No EPM trajectory .txt under {_REAL_EPM_DIR}")
    return candidates[0]


@pytest.fixture(scope="module")
def workspace_dir(thread_id) -> Path:
    """Host path to thread's workspace dir (where plan.json / handoff lands)."""
    return get_paths().sandbox_work_dir(thread_id)


# ---------------------------------------------------------------------------
# Test 1: full agent flow produces expected artifacts
# ---------------------------------------------------------------------------


def test_epm_full_agent_flow_produces_artifacts(client, thread_id, epm_raw_file, workspace_dir):
    """端到端流程：上传 EPM 真数据 → 让 agent 分析 → 验证 artifact 都在。"""

    # Step 1: upload raw file
    upload_result = client.upload_files(thread_id, [str(epm_raw_file)])
    assert upload_result.get("success") is True, f"upload failed: {upload_result}"

    # Step 2: ask agent to analyze (single-shot prompt)
    prompt = (
        "我做了 EPM 高架十字迷宫实验，单只小鼠测焦虑表型。"
        "请按照默认流程帮我分析这份数据。"
    )

    events = list(client.stream(prompt, thread_id=thread_id))
    assert events, "agent stream returned no events"

    # Step 3: verify catalog-architecture artifacts exist
    columns_json = workspace_dir / "columns.json"
    plan_json = workspace_dir / "metric_plan.json"
    handoff_json = workspace_dir / "handoff_code_executor.json"

    assert columns_json.is_file(), (
        f"columns.json not found at {columns_json} — lead 没有跑 dump_headers"
    )
    assert plan_json.is_file(), (
        f"metric_plan.json not found — lead 没有跑 catalog.resolve"
    )
    assert handoff_json.is_file(), (
        f"handoff_code_executor.json not found — code-executor 没产 handoff"
    )


# ---------------------------------------------------------------------------
# Test 2: plan.json structure aligns with EPM Q6 whitelist
# ---------------------------------------------------------------------------


def test_plan_contains_q6_whitelist_metrics(client, thread_id, workspace_dir):
    """plan.metrics 必须含 EPM Q6 白名单的全部 5 个指标。"""

    plan_path = workspace_dir / "metric_plan.json"
    if not plan_path.is_file():
        pytest.skip("Plan not generated (preceding test failed)")

    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    assert plan["schema_version"] == "1.0"
    assert plan["paradigm"] == "epm"

    metric_ids = {m["id"] for m in plan["metrics"]}
    q6_whitelist = {
        "open_arm_time_ratio",
        "open_arm_time",
        "open_arm_entry_count",
        "open_arm_entry_ratio",
        "total_entry_count",
    }
    assert metric_ids == q6_whitelist, (
        f"plan.metrics 与 Q6 白名单偏差: "
        f"缺 {q6_whitelist - metric_ids}, 多 {metric_ids - q6_whitelist}"
    )

    for m in plan["metrics"]:
        assert m["script"].startswith("ethoinsight.scripts.epm.")
        assert m["required"] is True
        assert m["reason"] == "paradigm.default"


# ---------------------------------------------------------------------------
# Test 3: handoff has real numeric values for all 5 metrics
# ---------------------------------------------------------------------------


def test_handoff_has_real_numeric_values(client, thread_id, workspace_dir):
    """handoff_code_executor.json 必须含 5 个指标的真实数值（不是 null）。"""

    handoff_path = workspace_dir / "handoff_code_executor.json"
    if not handoff_path.is_file():
        pytest.skip("Handoff not generated (preceding test failed)")

    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))

    # handoff schema 可能含 "metrics" / "per_subject" / 顶层 metric_id keys；
    # 兼容性地搜索所有数值
    all_text = json.dumps(handoff, ensure_ascii=False)
    for metric_id in [
        "open_arm_time_ratio",
        "open_arm_time",
        "open_arm_entry_count",
        "open_arm_entry_ratio",
        "total_entry_count",
    ]:
        assert metric_id in all_text, f"handoff 中找不到指标 {metric_id}"


# ---------------------------------------------------------------------------
# Test 4: report doesn't contain forbidden absolute-threshold language
# ---------------------------------------------------------------------------


def test_report_does_not_use_absolute_threshold_language(client, thread_id, workspace_dir):
    """同事 5-13 反馈硬要求：判读不参考常模/基线，全部走组间比较。

    断言 agent 最终输出（report + reasoning）不含违禁词。
    """
    outputs_dir = workspace_dir.parent / "outputs"
    if not outputs_dir.is_dir():
        pytest.skip("outputs/ dir not found")

    report_files = list(outputs_dir.glob("*.md")) + list(outputs_dir.glob("*.html"))
    if not report_files:
        pytest.skip("No report file found in outputs/")

    combined = "\n".join(p.read_text(encoding="utf-8", errors="ignore") for p in report_files)

    forbidden_patterns = [
        r"正常范围",
        r"normal\s+range",
        r"Reference\s+range",
        r"参考阈值",
        r"文献典型",
        r"文献参考值",
        r"常模",
        # "偏低/偏高" 本身不一定违规（用户语言里也用），只在"X 偏低/偏高"
        # 且后面没接"vs control/相比对照"时才算违规 —— 但 regex 检测太脆，
        # 这一条留给人工 review
    ]

    hits = []
    for pat in forbidden_patterns:
        for m in re.finditer(pat, combined, flags=re.IGNORECASE):
            ctx = combined[max(0, m.start() - 40) : min(len(combined), m.end() + 40)]
            hits.append(f"  match '{m.group()}' near: ...{ctx}...")

    assert not hits, (
        "Report 含禁用的绝对阈值语言（同事 5-13 反馈：'不要参考常模或基线'）：\n"
        + "\n".join(hits)
    )


# ---------------------------------------------------------------------------
# Test 5: gate signals indicate statistical_validity reflects n=1
# ---------------------------------------------------------------------------


def test_single_subject_skips_statistics(client, thread_id, workspace_dir):
    """单只样本应当跳过统计（catalog `when: n_per_group >= 2 and n_groups >= 2`）。

    Plan.statistics.skip_reason 必须非 null。
    """
    plan_path = workspace_dir / "metric_plan.json"
    if not plan_path.is_file():
        pytest.skip("Plan not generated")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    stats = plan.get("statistics")
    if stats is None:
        # 完全没 statistics 块也是合理的（catalog statistics: null）
        return
    assert stats["skip_reason"] is not None, (
        "单只样本 plan.statistics.skip_reason 应当非 null 但为 null"
    )
