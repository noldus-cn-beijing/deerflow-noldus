"""编排路径 SSOT — 8 条 INTENT 路径的声明式定义。

唯一真相源：prompt 箭头图、派遣顺序 provider、ask 点 provider 三个消费者
都从本模块读取，不得各自硬编码路径逻辑。改路径 = 改本文件，三处自动同步，
CI 哨兵（tests/test_path_registry_ssot.py）验证一致。

声明式数据，非命令式代码：这里没有 await/while/if 控制流；路径的"执行"由
deerflow 现有 middleware/provider 读本数据驱动，不在此运行。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

StepKind = Literal["dispatch", "ask"]


@dataclass(frozen=True)
class Step:
    """One step in an intent's dispatch chain.

    Attributes:
        kind: "dispatch" (delegate to subagent) or "ask" (ask user a question).
        target: dispatch → subagent name in prompt-side hyphenated form ("code-executor").
                ask → ask key ("viz" / "report" / "four_choice" / "clarify").
        condition: Optional condition string for conditional dispatch (e.g. "viz==yes").
                   None means unconditional.
    """

    kind: StepKind
    target: str
    condition: str | None = None


# ---------------------------------------------------------------------------
# 8 个 INTENT → 有序 step 序列。
# 从 prompt.py 箭头图逐条转写而来。每个 INTENT 对应一条从起始到终态的
# 有向路径；Step("dispatch", X) 代表 task(X)，Step("ask", k) 代表
# ask_clarification 反问用户。
# ---------------------------------------------------------------------------
PATHS: dict[str, list[Step]] = {
    "E2E_FULL_ASKVIZ": [
        Step("dispatch", "code-executor"),
        Step("dispatch", "data-analyst"),
        Step("ask", "viz"),
        Step("dispatch", "chart-maker", condition="viz==yes"),
        Step("ask", "report"),
    ],
    "E2E_FULL": [
        Step("dispatch", "code-executor"),
        Step("dispatch", "data-analyst"),
        Step("dispatch", "chart-maker"),
        Step("ask", "report"),
    ],
    "E2E_MIN": [
        Step("dispatch", "code-executor"),
        Step("ask", "four_choice"),
    ],
    "CHART": [Step("dispatch", "chart-maker")],
    "REPORT": [Step("dispatch", "report-writer")],
    "QA_FACT": [Step("dispatch", "knowledge-assistant")],
    "QA_KNOWLEDGE": [Step("dispatch", "knowledge-assistant")],
    "CLARIFY": [Step("ask", "clarify")],
}

VALID_INTENTS = frozenset(PATHS.keys())


# ---------------------------------------------------------------------------
# §1.1 命名映射：prompt 侧连字符 ↔ handoff 侧下划线
# ---------------------------------------------------------------------------
def to_handoff_name(target: str) -> str:
    """Convert prompt-side hyphenated name to handoff-side underscore name.

    >>> to_handoff_name("code-executor")
    'code_executor'
    """
    return target.replace("-", "_")


def to_prompt_name(handoff_name: str) -> str:
    """Convert handoff-side underscore name to prompt-side hyphenated name.

    >>> to_prompt_name("code_executor")
    'code-executor'
    """
    return handoff_name.replace("_", "-")


# ---------------------------------------------------------------------------
# §1.2 ask key → gate_completed 映射
# ---------------------------------------------------------------------------
ASK_GATE_MAP: dict[str, str] = {
    "viz": "gate3_viz_acknowledged",
    "report": "gate4_report_acknowledged",
    "four_choice": "gate_min_choice_acknowledged",
    # "clarify" 不走 gate，直接 ask_clarification 中断，不在此映射
}


# ---------------------------------------------------------------------------
# §1.3 ask key → 落盘该 gate 的工具名（gate-setter）
#
# 用于消除「同一轮并行发 set_X + task(下游)」的竞态误拦：当 lead 在同一个
# AIMessage 里同时调 gate-setter 和下游 task 时，guardrail 可能在 gate-setter
# 落盘前就评估 task → 误判 gate 未完成。provider 检测到同批 tool_calls 已含
# 对应 gate-setter 时，视该 ask 点已满足（落盘 in-flight，deny 是假阳性）。
#
# 仅登记真实存在的 setter 工具。viz 有 set_viz_choice；report / four_choice
# 暂无专用 setter 工具（ASK_GATE_MAP 已声明 gate 名，待 setter 工具落地后补这里）。
# ---------------------------------------------------------------------------
ASK_GATE_SETTER_TOOL: dict[str, str] = {
    "viz": "set_viz_choice",
}


# ---------------------------------------------------------------------------
# §1.1 dispatch target 校验（lazy，避免 circular import）
#
# 不能在 module import-time 调用，因为 deerflow.subagents.builtins 的
# import chain 会触发 deerflow.subagents.__init__ → executor → thread_state →
# tools → task_tool → deerflow.subagents 循环。
# 由调用方在真正需要校验时调用（provider 初始化 / CI sentinel 测试）。
# ---------------------------------------------------------------------------
_validated = False


def ensure_dispatch_targets_validated() -> None:
    """Lazily validate that every dispatch Step.target is a registered BUILTIN_SUBAGENTS key.

    Safe to call multiple times — only validates once. Called by:
    - PathSequenceProvider.__init__
    - IntentPostStepAskGateProvider (after generalization)
    - tests/test_path_registry_ssot.py (CI sentinel)
    """
    global _validated
    if _validated:
        return
    from deerflow.subagents.builtins import BUILTIN_SUBAGENTS

    registered = set(BUILTIN_SUBAGENTS.keys())
    for intent, steps in PATHS.items():
        for step in steps:
            if step.kind == "dispatch" and step.target not in registered:
                raise ValueError(
                    f"PATHS[{intent!r}] dispatch target {step.target!r} "
                    f"not in BUILTIN_SUBAGENTS. Known: {sorted(registered)}"
                )
    _validated = True
