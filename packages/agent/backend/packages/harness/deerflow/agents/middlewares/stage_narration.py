"""stage_narration — A1 后端事件分轨地基的纯映射模块（承重墙的「脑」）。

Spec: docs/superpowers/specs/2026-06-30-generative-ux-roadmap-and-a1-event-track-foundation-design.md

本模块是「意图 → 人话阶段」的**唯一真相源**（SSOT）。它只做确定性映射，不读 LLM、
不发 I/O；发射（往 custom 轨写）由 ``StageNarrationMiddleware``（stage_plan）和
``task`` 工具（stage_update，复用既有派遣观测点）负责。

设计原则（守 spec 模块 3 + single-source-of-truth）：
- 阶段名是 SSOT：定义在本模块一处，前端不维护第二份阶段字典。
- 意图→阶段集是固定表（不靠 LLM）：只有「多阶段流水线」意图才发 stage_plan；
  知识问答 / 闲聊 / 单步追问 / 单步出图报告不发（前端据此无 stepper）。
- 人话字段（stages / skipped / stage / narration）**永不携带内脏**（工具名、
  subagent 名、gate 关键字）——它们是面向用户的，内脏留在 reasoning/内部轨。
- 与 PR#213 / SealGateMiddleware 同构：真实状态机驱动，不靠 LLM 自报。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

# ---------------------------------------------------------------------------
# §1 阶段名 SSOT —— 人话阶段字典（唯一一处，前端不复刻）
#
# 这些是面向研究员的阶段名，不是机器侧 subagent 名。
# subagent 名（code-executor 等）永不直接出现在 stage_* 事件里。
# ---------------------------------------------------------------------------
STAGE_IDENTIFY = "识别范式"
STAGE_COMPUTE = "计算指标"
STAGE_INTERPRET = "数据解读"
STAGE_CHART = "生成图表"
STAGE_REPORT = "撰写报告"

# 意图 → 有序阶段集。只有多阶段流水线意图在此登记；
# 不在表里的意图（CHART/REPORT/QA_*/CLARIFY/None/未知）→ 不发 stage_plan。
_INTENT_STAGES: dict[str, list[str]] = {
    "E2E_FULL": [STAGE_IDENTIFY, STAGE_COMPUTE, STAGE_INTERPRET, STAGE_CHART, STAGE_REPORT],
    "E2E_FULL_ASKVIZ": [STAGE_IDENTIFY, STAGE_COMPUTE, STAGE_INTERPRET, STAGE_CHART, STAGE_REPORT],
    "E2E_MIN": [STAGE_IDENTIFY, STAGE_COMPUTE],
}

# ---------------------------------------------------------------------------
# §2 subagent_type → 人话阶段名（用于 stage_update）
#
# 复用 task 工具的派遣观测点：task 进/出某 subagent → 发对应阶段的 active/completed。
# 未登记的 subagent 返回 None（不猜阶段名 → 不发 stage_update）。
# ---------------------------------------------------------------------------
_DISPATCH_STAGE: dict[str, str] = {
    "code-executor": STAGE_COMPUTE,
    "data-analyst": STAGE_INTERPRET,
    "chart-maker": STAGE_CHART,
    "report-writer": STAGE_REPORT,
}

# n=1 单样本时被跳过的阶段（无统计基础，不做组间比较解读）。
# 守 CLAUDE.md 第 9 条：判读只看 control vs treatment 显著差异，n=1 无统计基础。
_N1_SKIPPED: tuple[str, ...] = (STAGE_INTERPRET,)

# ---------------------------------------------------------------------------
# §3 默认人话 narration 模板
#
# stage_update 的 narration 可由调用方覆盖（如「正在为 28 个文件生成 113 张图表…」）；
# 缺省时用这里的模板。**纯人话，无内脏词**（防泄漏断言守护）。
# ---------------------------------------------------------------------------
DEFAULT_NARRATIONS: dict[str, str] = {
    "active": "进行中…",
    "completed": "完成",
}


def intent_to_stage_plan(intent: str | None, *, n: int | None) -> dict | None:
    """意图 + 样本数 → stage_plan payload；非流水线意图返回 None（不发）。

    Args:
        intent: lead 声明的意图（E2E_FULL / E2E_FULL_ASKVIZ / E2E_MIN / ...）。
                None / 未知 / 非流水线意图 → 返回 None。
        n: 本批样本数（subject 数）。n<2 时标注被跳过的解读阶段；
            None（未知）→ skipped 为空（不臆测 n=1）。

    Returns:
        ``{"kind": "stage_plan", "stages": [...], "skipped": [...]}`` 或 None。
        skipped 只含「在 stages 里、但因 n=1 被跳过」的阶段。
    """
    stages = _INTENT_STAGES.get(intent) if intent is not None else None
    if stages is None:
        return None
    skipped = [s for s in _N1_SKIPPED if n is not None and n < 2 and s in stages]
    return {"kind": "stage_plan", "stages": list(stages), "skipped": skipped}


def stage_for_dispatch(subagent_type: str) -> str | None:
    """subagent_type → 人话阶段名；未登记返回 None。"""
    return _DISPATCH_STAGE.get(subagent_type)


StageStatus = Literal["active", "completed"]


def stage_update(stage: str, status: StageStatus, narration: str | None = None) -> dict:
    """构造 stage_update payload。

    Args:
        stage: 人话阶段名（来自 stage_for_dispatch / stage_plan.stages）。
        status: ``active``（subagent 进入）或 ``completed``（subagent 成功退出）。
                失败/超时/取消**不**调本函数发 completed（grounded：叙事不撒谎）。
        narration: 人话叙述，可空（缺省用 DEFAULT_NARRATIONS）。
    """
    if status not in ("active", "completed"):
        raise ValueError(f"stage_update status must be 'active' or 'completed', got {status!r}")
    return {
        "kind": "stage_update",
        "stage": stage,
        "status": status,
        "narration": narration if narration is not None else DEFAULT_NARRATIONS[status],
    }


# ---------------------------------------------------------------------------
# §4 派遣边界发射助手 —— 由 task 工具在既有 task_started / task_completed /
#    task_failed 旁同源调用（复用观测点，不重复造）。
#
# grounded：是否发 completed 完全由真实 executor 结果决定（succeeded=True/False），
# 不靠 LLM 自报。失败/超时/取消 → 不发 completed（叙事不撒谎，spec 验收 4）。
#
# writer 默认 None（调用方传 get_stream_writer 的包装）；发射失败被吞，永不崩 turn。
# ---------------------------------------------------------------------------
TerminalStatus = Literal["failed", "timed_out", "cancelled", "disappeared"]


def _safe_write(writer: Callable[[dict], None] | None, payload: dict) -> None:
    if writer is None:
        return
    try:
        writer(payload)
    except Exception:  # noqa: BLE001
        # Best-effort signal: never crash the agent turn over narration.
        pass


def emit_dispatch_enter(subagent_type: str, *, writer: Callable[[dict], None] | None = None) -> None:
    """subagent 进入（task_started 边界）→ 发 stage_update(active)。

    未登记的 subagent_type → 不发（不猜阶段名）。
    """
    stage = stage_for_dispatch(subagent_type)
    if stage is None:
        return
    _safe_write(writer, stage_update(stage, "active"))


def emit_dispatch_exit(
    subagent_type: str,
    *,
    succeeded: bool,
    terminal_status: TerminalStatus | str | None = None,
    writer: Callable[[dict], None] | None = None,
) -> None:
    """subagent 退出边界 → 仅成功时发 stage_update(completed)。

    Args:
        subagent_type: 派遣的 subagent 名（prompt 侧连字符）。
        succeeded: 真实 executor 是否成功（COMPLETED=True；FAILED/TIMED_OUT/
                   CANCELLED/disappeared=False）。**grounded 信号**。
        terminal_status: 终态名（用于日志/未来扩展，当前不影响发射决策——
                         只要 succeeded=False 就不发 completed）。
        writer: custom 轨写入回调。

    失败/超时/取消/消失 → **不发 completed**（叙事不撒谎）。active 已在进入时发过，
    这里不发「失败」事件——留空（前端 stepper 停在 active）比撒谎说完成好。
    """
    if not succeeded:
        return
    stage = stage_for_dispatch(subagent_type)
    if stage is None:
        return
    _safe_write(writer, stage_update(stage, "completed"))
