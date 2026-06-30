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
# 知识问答阶段（非流水线意图的「独立活动提示」，spec 2026-06-30-a1-stage-narration-coverage-gap-fix 缺口 2）。
# knowledge-assistant 派遣发它的 stage_update；因 QA 意图不发 stage_plan，前端把它渲染成一句活动提示而非 stepper 节点（A2 范围）。
STAGE_KNOWLEDGE = "查阅领域知识"

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
    # 缺口 2：知识问答派遣 → 独立活动提示（非流水线阶段，QA 意图不发 stage_plan）。
    "knowledge-assistant": STAGE_KNOWLEDGE,
}

# n=1 单样本时被跳过的阶段（无统计基础，不做组间比较解读）。
# 守 CLAUDE.md 第 9 条：判读只看 control vs treatment 显著差异，n=1 无统计基础。
_N1_SKIPPED: tuple[str, ...] = (STAGE_INTERPRET,)

# ---------------------------------------------------------------------------
# §2b 「识别范式」阶段的工具观测点（缺口 1）
#
# 「识别范式」由 lead 自调工具完成（不派 subagent），故 task 派遣观测点收不到它。
# 这里登记「哪些工具名 = 识别阶段的进入/完成信号」，供 StageNarrationMiddleware.wrap_tool_call
# 在真实工具调用边界同源发射（与 emit_dispatch_enter/exit 同构：真实事件驱动，不靠 LLM 自报）。
#
# - _IDENTIFY_ENTER_TOOLS：lead 开始识别（进入识别范式）→ 发 active。
# - _IDENTIFY_DONE_TOOL：识别完成、即将派 code-executor 的 grounded 信号 →
#   仅当其返回 status=="ok" 才发 completed（叙事不撒谎）。
# ---------------------------------------------------------------------------
_IDENTIFY_ENTER_TOOLS: frozenset[str] = frozenset(
    {
        "identify_ev19_template",
        "inspect_uploaded_file",
    }
)
_IDENTIFY_DONE_TOOL = "prep_metric_plan"

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


def is_identify_enter_tool(tool_name: str) -> bool:
    """该工具名是否表示 lead 已进入「识别范式」阶段（缺口 1）。

    用于 StageNarrationMiddleware.wrap_tool_call 在调 handler 前发 ``识别范式`` active。
    """
    return tool_name in _IDENTIFY_ENTER_TOOLS


def is_identify_done_tool(tool_name: str) -> bool:
    """该工具名是否是「识别范式」完成的 grounded 信号源（缺口 1）。

    ``prep_metric_plan`` 成功（status=ok）= 识别+计划完成、即将派 code-executor。
    是否真发 completed 由 ``identify_done_succeeded`` 据真实返回值再判（不撒谎）。
    """
    return tool_name == _IDENTIFY_DONE_TOOL


def identify_done_succeeded(tool_result_content: str | object) -> bool:
    """解析 identify-done 工具（prep_metric_plan）的 ToolMessage content，判断识别是否真成功。

    grounded（spec 验收 4 / 缺口 1）：只有真实返回 ``status == "ok"`` 才算成功。
    content 非 str / 非 JSON / 解析失败 / status 缺失或非 ok → 返回 False（安全侧倒，不发 completed）。
    """
    if not isinstance(tool_result_content, str) or not tool_result_content:
        return False
    import json

    try:
        parsed = json.loads(tool_result_content)
    except (ValueError, TypeError):
        return False
    return isinstance(parsed, dict) and parsed.get("status") == "ok"


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
