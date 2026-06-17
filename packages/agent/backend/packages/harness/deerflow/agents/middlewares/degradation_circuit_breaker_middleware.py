"""DegradationCircuitBreakerMiddleware — lead after_model gate for statistics crashes (P7).

P7 (spec 2026-06-17-data-degradation-circuit-breaker): 数据降级熔断器。code-executor 是
lead 通过 ``task`` 工具派遣的 subagent，其 handoff (``handoff_code_executor.json``) 由 lead
消费。当 ``gate_signals.statistics_status == "crashed"`` 时，statistics 子步骤崩溃（损害可复现性/
准确性），本 middleware 在 lead 的 ``after_model`` 拦截：

  1. 自救一次：注入正面 reminder（"重派 code-executor 重跑 statistics"）+ ``jump_to='model'``，
     计数 +1。一次为限（``_MAX_SELF_HELP = 1``，spec §5 "figure out 时间不应长"）。
  2. 自救超限转 HITL：再 crashed 则注入 reminder 引导模型调 ``ask_clarification`` 向用户确认
     （检查数据/接受仅描述性/调参数），由 ClarificationMiddleware 拦截该 tool_call 并中断。

只熔断 ``crashed``。``absent_by_design``（单组/单样本合理 skip，正常描述性 partial）与
``ok`` 只通知不熔断（通知 = 已写进 handoff，lead 自读）。

防同一条 crashed 反复触发：按 handoff 文件 mtime 去重（per-run）。同一文件 mtime 只触发一次；
handoff 被 code-executor 重写（mtime 变）才允许新一条触发，此时 ``_self_help_counts`` 已 +1，
直接进 HITL 分支。

范式复刻：
  - ``SealGateMiddleware``：``@hook_config(can_jump_to=["model"])`` + per-run 计数（runtime.run_id
    作 key）+ fail-open（except 返回 None）+ ``{"messages":[HumanMessage(reminder)], "jump_to":"model"}``。
  - ``QualityWarningBroadcastMiddleware``：last AIMessage 无 tool_calls（lead 的 broadcast turn）
    才触发；``resolve_workspace_from_state`` 取 workspace 后直接 ``json.loads`` 读 handoff（不走
    ``read_handoff``，避免 get_app_config 依赖与 schema 校验副作用——信号字段 gate_signals.statistics_status
    不需要整 schema 校验，读原始 dict 即可）。

HITL 中断：``ask_clarification`` 的中断由 ``ClarificationMiddleware.wrap_tool_call`` 拦截**模型
发出的 tool_call** 实现（``Command(goto=END)``）。本 middleware 不自造 tool_call，而是注入 reminder
+ jump_to=model 让模型自己调 ask_clarification，由 ClarificationMiddleware 接管——最干净的范式。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, hook_config
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

# 产品决策（spec §2.2/§5 "1-2 次"，用户确认取 1）：一次重试足以区分偶发崩溃 vs 必然崩溃。
# 自救一次后转 HITL，避免在 crashed 上反复重试浪费 token、避免变相纵容手算螺旋。
_MAX_SELF_HELP = 1
_HANDOFF_FILENAME = "handoff_code_executor.json"


class DegradationCircuitBreakerMiddleware(AgentMiddleware):
    """after_model gate: detect statistics crash from code-executor handoff, bounded self-help then HITL."""

    def __init__(self) -> None:
        super().__init__()
        # per-run 自救计数：{run_id: count}（仿 SealGate._reminder_counts）
        self._self_help_counts: dict[str, int] = {}
        # per-run 已处理的 handoff mtime：防同一条 crashed 反复触发。
        # {run_id: last_handoff_mtime}（handoff 被重写 → mtime 变 → 允许新一条触发）
        self._processed_mtime: dict[str, float] = {}

    def _run_key(self, runtime: Runtime) -> str:
        return str(getattr(runtime, "run_id", None) or id(runtime))

    def _get_count(self, runtime: Runtime) -> int:
        return self._self_help_counts.get(self._run_key(runtime), 0)

    def _bump_count(self, runtime: Runtime) -> None:
        k = self._run_key(runtime)
        self._self_help_counts[k] = self._self_help_counts.get(k, 0) + 1

    @hook_config(can_jump_to=["model"])
    def after_model(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        """Check after lead model output whether a statistics crash needs handling."""
        try:
            return self._check(state, runtime)
        except Exception:
            logger.debug("DegradationCircuitBreaker: check failed, fail-open", exc_info=True)
            return None

    @hook_config(can_jump_to=["model"])
    async def aafter_model(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        """Async version — delegates to sync."""
        return self.after_model(state, runtime)

    def _check(self, state: Any, runtime: Runtime) -> dict[str, Any] | None:
        # 惰性 import（harness 导入环铁律：experiment_context 顶层 import 会闭环）
        from deerflow.agents.middlewares.experiment_context import resolve_workspace_from_state

        messages = state.get("messages", []) if hasattr(state, "get") else []
        if not messages:
            return None
        last = messages[-1]
        # 仿 QualityWarningBroadcast: last AIMessage 有 tool_calls = lead 还在派遣/工作，不抢断。
        if getattr(last, "type", None) != "ai":
            return None
        if getattr(last, "tool_calls", None):
            return None

        workspace = resolve_workspace_from_state(state)
        if not workspace:
            return None

        # 直接 json.loads 读 handoff（不走 read_handoff：避免 get_app_config 依赖 + schema 校验副作用；
        # 信号字段 gate_signals.statistics_status 读原始 dict 即可，仿 QualityWarningBroadcast 范式）。
        handoff_path = Path(str(workspace)) / _HANDOFF_FILENAME
        try:
            if not handoff_path.exists():
                return None
            handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("DegradationCircuitBreaker: 读 %s 失败: %s", _HANDOFF_FILENAME, e)
            return None
        if not isinstance(handoff, dict):
            return None

        gate_signals = handoff.get("gate_signals") or {}
        if not isinstance(gate_signals, dict):
            return None
        # 只熔断 crashed（损害可复现性）。absent_by_design / ok 只通知（已落 handoff，lead 自读）。
        if gate_signals.get("statistics_status") != "crashed":
            return None

        # 防同一条反复触发：按 handoff 文件 mtime 去重（per-run）。
        # handoff 被 code-executor 重写（mtime 变）才允许新一条触发。
        run_key = self._run_key(runtime)
        try:
            mtime = handoff_path.stat().st_mtime
        except OSError:
            mtime = 0.0
        if self._processed_mtime.get(run_key) == mtime:
            return None

        error_summary = (gate_signals.get("statistics_error") or "")[:200]
        count = self._get_count(runtime)
        self._bump_count(runtime)
        self._processed_mtime[run_key] = mtime

        if count < _MAX_SELF_HELP:
            logger.info(
                "DegradationCircuitBreaker: self-help #%d for statistics crash (err=%s)",
                count + 1,
                error_summary,
            )
            reminder = _self_help_reminder(error_summary)
        else:
            logger.info(
                "DegradationCircuitBreaker: self-help cap reached (%d), escalate to HITL",
                count + 1,
            )
            reminder = _hitl_reminder(error_summary)

        return {
            "messages": [
                HumanMessage(
                    content=reminder,
                    name="degradation_circuit_breaker",
                    additional_kwargs={"hide_from_ui": True},
                )
            ],
            "jump_to": "model",
        }


# ---- 正面指令 reminder 文案（CLAUDE.md §6 deepseek：用"请"不用"不要/禁止"）----


def _self_help_reminder(error_summary: str) -> str:
    """自救 reminder：引导重派 code-executor 重跑 statistics（正面动作，单次为限）。"""
    return (
        "<system_reminder>\n"
        "[降级熔断] statistics 子步骤崩溃，这会损害分析的可复现性，需要修复。\n"
        f"崩溃摘要：{error_summary}\n"
        "请采取一次自救：检查 metric plan 的 statistics 段参数是否合理，"
        "然后用 task 工具重新派遣 code-executor（subagent_type=code-executor）重跑，"
        "让 statistics 子步骤正常产出。一次为限，若重跑仍崩溃，下一步转人工确认。\n"
        "</system_reminder>"
    )


def _hitl_reminder(error_summary: str) -> str:
    """HITL reminder：引导调 ask_clarification 向用户确认（正面动作 + 具体选项）。"""
    return (
        "<system_reminder>\n"
        "[降级熔断] statistics 子步骤再次崩溃，自救已超限，需要向用户确认。\n"
        f"崩溃摘要：{error_summary}\n"
        "请调用 ask_clarification 工具向用户确认如何继续，提供以下选项让用户选择：\n"
        "  ① 用户检查输入数据后，由你重新派遣 code-executor 重跑 statistics；\n"
        "  ② 接受仅描述性结果（无统计检验），继续后续 data-analyst/report-writer；\n"
        "  ③ 用户调整统计参数后重跑。\n"
        "</system_reminder>"
    )
