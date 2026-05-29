"""Subagent execution engine."""

import asyncio
import atexit
import logging
import threading
import uuid
from collections.abc import Callable, Coroutine
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from contextvars import Context, copy_context
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain.tools import BaseTool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from deerflow.agents.thread_state import SandboxState, ThreadDataState, ThreadState
from deerflow.models import create_chat_model
from deerflow.subagents.config import SubagentConfig
from deerflow.subagents.token_collector import SubagentTokenCollector

logger = logging.getLogger(__name__)


# Hook pairs (sync, async) on AgentMiddleware. Each pair becomes a graph node
# in langchain.agents.create_agent if EITHER sync or async is overridden.
# See langchain.agents.factory ~L1198-1290 for the wiring.
_BEFORE_AGENT_HOOKS = (("before_agent", "abefore_agent"),)
_BEFORE_MODEL_HOOKS = (("before_model", "abefore_model"),)
_AFTER_MODEL_HOOKS = (("after_model", "aafter_model"),)
_AFTER_AGENT_HOOKS = (("after_agent", "aafter_agent"),)


def _middleware_implements(m: AgentMiddleware, sync_name: str, async_name: str) -> bool:
    """Return True if middleware ``m`` overrides either the sync or async hook."""
    base = AgentMiddleware
    return getattr(m.__class__, sync_name) is not getattr(base, sync_name) or getattr(m.__class__, async_name) is not getattr(base, async_name)


def calculate_subagent_recursion_limit(middlewares: list[AgentMiddleware], max_turns: int) -> int:
    """Compute LangGraph recursion_limit for a subagent run from its middleware chain.

    LangGraph counts node steps. ``langchain.agents.create_agent`` adds one node
    per overridden middleware hook (before_agent / before_model / after_model /
    after_agent), plus the fixed ``model`` and ``tools`` nodes. Per turn:

        before_model_hooks + 1 (model) + after_model_hooks + 1 (tools)

    before_agent / after_agent fire exactly once at entry / exit.

    Adds +3 margin so an extra middleware hook landing during sync does not
    immediately re-break the limit.
    """
    if max_turns <= 0:
        raise ValueError(f"max_turns must be positive, got {max_turns}")

    before_agent = sum(1 for m in middlewares if _middleware_implements(m, *_BEFORE_AGENT_HOOKS[0]))
    before_model = sum(1 for m in middlewares if _middleware_implements(m, *_BEFORE_MODEL_HOOKS[0]))
    after_model = sum(1 for m in middlewares if _middleware_implements(m, *_AFTER_MODEL_HOOKS[0]))
    after_agent = sum(1 for m in middlewares if _middleware_implements(m, *_AFTER_AGENT_HOOKS[0]))

    per_turn = before_model + 1 + after_model + 1
    one_off = before_agent + after_agent
    margin = 3
    return max_turns * per_turn + one_off + margin


# ---------------------------------------------------------------------------
# Sprint 5.7: Handoff emission validation
#
# Ethoinsight subagents that MUST seal a handoff file before terminating
# successfully. Keys are SubagentConfig.name (hyphenated, matching config.name);
# values are the expected handoff filename in the thread workspace.
#
# General-purpose subagents (general-purpose / bash / knowledge-assistant) are
# intentionally NOT in this map — they have no seal_*_handoff contract and may
# legitimately complete without producing a handoff file.
# ---------------------------------------------------------------------------
_HANDOFF_EMISSION_REQUIRED: dict[str, str] = {
    "code-executor": "handoff_code_executor.json",
    "data-analyst": "handoff_data_analyst.json",
    "chart-maker": "handoff_chart_maker.json",
    "report-writer": "handoff_report_writer.json",
}


# Sprint 5.8: seal-resume 只补 1 轮。这是深思过的决策，不是任意值。
# 理由：补轮失败的主因更可能是【系统性】(分析结论本身残缺 / LLM 卡在理解错误)
# 而非【随机】——再补一轮多半是同样结果。补不上时，交给 5.7 兜底重派【整个
# subagent】(全新上下文)反而更可能成功。纵深防御已足够：补轮(便宜捞一次) →
# 5.7 重派(全新上下文) → lead 最多 2 次 retry。不需要在补轮内部再加循环。
# 改大此值的代价：消耗更多 turn + token + 延迟（补轮是独立 astream，受 recursion_limit 管）。
_SEAL_RESUME_MAX_ATTEMPTS = 1


def _validate_handoff_emitted(
    subagent_name: str,
    workspace_path: str | None,
) -> str | None:
    """Return None if the subagent's handoff file is present, else a diagnostic.

    Called just before set_terminal(COMPLETED). If subagent_name is in
    _HANDOFF_EMISSION_REQUIRED but its handoff file is absent in workspace,
    returns an error string explaining the failure (used as
    try_set_terminal(FAILED, error=...)). Otherwise returns None (pass / N/A).

    ROBUSTNESS (Sprint 5.7 spec §1 C3): this function MUST NOT raise. The call
    site sits inside executor's try/except — any exception here would be caught
    and mis-attributed as a generic FAILED, clobbering our diagnostic. So all
    filesystem access is wrapped; on unexpected error we fail-open (return None).

    Args:
        subagent_name: SubagentConfig.name, e.g. "data-analyst".
        workspace_path: Host-side workspace dir from thread_data["workspace_path"].
            None when ThreadDataMiddleware didn't set it (old threads / dev paths).

    Returns:
        None  -> validation passes, or subagent not in white-list, or
                 unresolvable workspace (fail-open).
        str   -> diagnostic explaining the missing handoff.
    """
    expected_filename = _HANDOFF_EMISSION_REQUIRED.get(subagent_name)
    if expected_filename is None:
        # Not a handoff-producing ethoinsight subagent — no contract to enforce.
        return None

    if not workspace_path:
        # No workspace resolvable — can't validate. Fail-open (do NOT block);
        # this preserves pre-5.7 behavior for old threads and dev paths.
        logger.warning(
            "[handoff_emission] Subagent %s: no workspace_path; skipping "
            "handoff validation (fail-open).",
            subagent_name,
        )
        return None

    try:
        handoff_path = Path(workspace_path) / expected_filename
        if handoff_path.exists():
            return None
    except Exception:  # noqa: BLE001 — must never raise; see docstring ROBUSTNESS.
        logger.exception(
            "[handoff_emission] Subagent %s: error while checking %s; "
            "failing open.",
            subagent_name,
            expected_filename,
        )
        return None

    seal_tool = f"seal_{subagent_name.replace('-', '_')}_handoff"
    return (
        f"Subagent '{subagent_name}' terminated without emitting "
        f"'{expected_filename}'. This means the LLM finished its reasoning "
        f"but forgot to call the {seal_tool} tool. The handoff file is "
        f"REQUIRED by downstream subagents. Lead should re-dispatch this task "
        f"with an explicit reminder to invoke the seal tool at the end."
    )


class SubagentStatus(Enum):
    """Status of a subagent execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"

    @property
    def is_terminal(self) -> bool:
        """True for statuses that mark a subagent as done (success or failure).

        Compares by ``.value`` (string) rather than enum identity so this stays
        correct across ``importlib.reload`` boundaries — under reload, the enum
        class identity changes but ``.value`` is stable.
        """
        return self.value in {"completed", "failed", "cancelled", "timed_out"}


@dataclass
class SubagentResult:
    """Result of a subagent execution.

    Attributes:
        task_id: Unique identifier for this execution.
        trace_id: Trace ID for distributed tracing (links parent and subagent logs).
        status: Current status of the execution.
        result: The final result message (if completed).
        error: Error message (if failed).
        started_at: When execution started.
        completed_at: When execution completed.
        ai_messages: List of complete AI messages (as dicts) generated during execution.
    """

    task_id: str
    trace_id: str
    status: SubagentStatus
    result: str | None = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    ai_messages: list[dict[str, Any]] | None = None
    token_usage_records: list[dict[str, int | str]] = field(default_factory=list)
    usage_reported: bool = False
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _state_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.ai_messages is None:
            self.ai_messages = []

    def try_set_terminal(
        self,
        status: SubagentStatus,
        *,
        result: str | None = None,
        error: str | None = None,
        completed_at: datetime | None = None,
        ai_messages: list[dict[str, Any]] | None = None,
        token_usage_records: list[dict[str, int | str]] | None = None,
    ) -> bool:
        """Set a terminal status exactly once (CAS — first writer wins).

        Background timeout/cancellation and the execution worker can race on
        the same result holder. The first terminal transition wins; late
        terminal writes are silently ignored and return False (so callers can
        detect they were beaten to the punch).

        Borrowed from upstream deer-flow.

        Args:
            status: A terminal status (COMPLETED/FAILED/CANCELLED/TIMED_OUT).
            result/error/ai_messages/token_usage_records: Optional payload
                fields to set atomically with the status transition.
            completed_at: Override for the completion timestamp. Defaults to
                ``datetime.now()`` when omitted.

        Returns:
            True if the transition was applied, False if the status was
            already terminal.
        """
        if not status.is_terminal:
            raise ValueError(f"Status {status} is not terminal")

        with self._state_lock:
            if self.status.is_terminal:
                return False

            if result is not None:
                self.result = result
            if error is not None:
                self.error = error
            if ai_messages is not None:
                self.ai_messages = ai_messages
            if token_usage_records is not None:
                self.token_usage_records = token_usage_records
            self.completed_at = completed_at or datetime.now()
            self.status = status
            return True


# Global storage for background task results
_background_tasks: dict[str, SubagentResult] = {}
_background_tasks_lock = threading.Lock()

# Thread pool for background task scheduling and orchestration
_scheduler_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="subagent-scheduler-")

# Thread pool for actual subagent execution (with timeout support)
# Larger pool to avoid blocking when scheduler submits execution tasks
_execution_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="subagent-exec-")

# Persistent event loop for isolated subagent executions triggered from an
# already-running parent loop. Reusing one long-lived loop avoids creating a
# fresh loop per execution and then closing async resources bound to it.
_isolated_subagent_loop: asyncio.AbstractEventLoop | None = None
_isolated_subagent_loop_thread: threading.Thread | None = None
_isolated_subagent_loop_started: threading.Event | None = None
_isolated_subagent_loop_lock = threading.Lock()


def _run_isolated_subagent_loop(
    loop: asyncio.AbstractEventLoop,
    started_event: threading.Event,
) -> None:
    """Run the persistent isolated subagent loop in a dedicated daemon thread."""
    asyncio.set_event_loop(loop)
    loop.call_soon(started_event.set)
    try:
        loop.run_forever()
    finally:
        started_event.clear()


def _shutdown_isolated_subagent_loop() -> None:
    """Stop and close the persistent isolated subagent loop."""
    global _isolated_subagent_loop, _isolated_subagent_loop_thread, _isolated_subagent_loop_started

    with _isolated_subagent_loop_lock:
        loop = _isolated_subagent_loop
        thread = _isolated_subagent_loop_thread
        _isolated_subagent_loop = None
        _isolated_subagent_loop_thread = None
        _isolated_subagent_loop_started = None

    if loop is None:
        return

    if loop.is_running():
        loop.call_soon_threadsafe(loop.stop)

    if thread is not None and thread.is_alive() and thread is not threading.current_thread():
        thread.join(timeout=1)

    thread_stopped = thread is None or not thread.is_alive()
    loop_stopped = not loop.is_running()

    if not loop.is_closed():
        if thread_stopped and loop_stopped:
            loop.close()
        else:
            logger.warning(
                "Skipping close of isolated subagent loop because shutdown did not complete within timeout (thread_alive=%s, loop_running=%s)",
                thread is not None and thread.is_alive(),
                loop.is_running(),
            )


atexit.register(_shutdown_isolated_subagent_loop)


# Module reload safety: if a previous version of this module also registered
# _shutdown_isolated_subagent_loop with atexit (e.g. via importlib.reload in
# tests, or hot-reload), the old callback would still fire on process exit
# referencing a stale loop reference. Unregister the previous version so we
# only ever have one active shutdown hook per process. Borrowed from upstream
# deer-flow e19bec1.
_previous_shutdown_isolated_subagent_loop = globals().get("_previous_shutdown_isolated_subagent_loop")
if callable(_previous_shutdown_isolated_subagent_loop) and _previous_shutdown_isolated_subagent_loop is not _shutdown_isolated_subagent_loop:
    atexit.unregister(_previous_shutdown_isolated_subagent_loop)
_previous_shutdown_isolated_subagent_loop = _shutdown_isolated_subagent_loop


def _get_isolated_subagent_loop() -> asyncio.AbstractEventLoop:
    """Return the persistent event loop used by isolated subagent executions."""
    global _isolated_subagent_loop, _isolated_subagent_loop_thread, _isolated_subagent_loop_started
    with _isolated_subagent_loop_lock:
        thread_is_alive = _isolated_subagent_loop_thread is not None and _isolated_subagent_loop_thread.is_alive()
        loop_is_usable = _isolated_subagent_loop is not None and not _isolated_subagent_loop.is_closed() and _isolated_subagent_loop.is_running() and thread_is_alive

        if not loop_is_usable:
            loop = asyncio.new_event_loop()
            started_event = threading.Event()
            thread = threading.Thread(
                target=_run_isolated_subagent_loop,
                args=(loop, started_event),
                name="subagent-persistent-loop",
                daemon=True,
            )
            thread.start()
            if not started_event.wait(timeout=5):
                loop.call_soon_threadsafe(loop.stop)
                thread.join(timeout=1)
                loop.close()
                raise RuntimeError("Timed out starting isolated subagent event loop")
            _isolated_subagent_loop = loop
            _isolated_subagent_loop_thread = thread
            _isolated_subagent_loop_started = started_event

        if _isolated_subagent_loop is None:
            raise RuntimeError("Isolated subagent event loop is not initialized")
        return _isolated_subagent_loop


def _submit_to_isolated_loop_in_context(
    context: Context,
    coro_factory: Callable[[], Coroutine[Any, Any, "SubagentResult"]],
) -> Future["SubagentResult"]:
    """Submit a coroutine to the isolated loop while preserving ContextVar state."""
    return context.run(
        lambda: asyncio.run_coroutine_threadsafe(
            coro_factory(),
            _get_isolated_subagent_loop(),
        )
    )


def _filter_tools(
    all_tools: list[BaseTool],
    allowed: list[str] | None,
    disallowed: list[str] | None,
) -> list[BaseTool]:
    """Filter tools based on subagent configuration.

    Args:
        all_tools: List of all available tools.
        allowed: Optional allowlist of tool names. If provided, only these tools are included.
        disallowed: Optional denylist of tool names. These tools are always excluded.

    Returns:
        Filtered list of tools.
    """
    filtered = all_tools

    # Apply allowlist if specified
    if allowed is not None:
        allowed_set = set(allowed)
        filtered = [t for t in filtered if t.name in allowed_set]

    # Apply denylist
    if disallowed is not None:
        disallowed_set = set(disallowed)
        filtered = [t for t in filtered if t.name not in disallowed_set]

    return filtered


def _load_skill_contents(skill_names: list[str]) -> str:
    """Load and inline skill file contents for subagent prompt injection.

    Delegates to ``deerflow.skills.render.render_skill_sections`` which is
    independently unit-tested (see ``tests/test_subagent_skill_path_rewrite.py``).
    Kept here as a thin wrapper so existing call sites stay untouched.

    Returns:
        Concatenated skill content string, or empty string if no skills loaded.
    """
    from deerflow.config import get_app_config
    from deerflow.skills.render import render_skill_sections

    try:
        container_base_path = get_app_config().skills.container_path
    except Exception:
        container_base_path = "/mnt/skills"
    return render_skill_sections(skill_names, container_base_path)


def _get_model_name(config: SubagentConfig, parent_model: str | None) -> str | None:
    """Resolve the model name for a subagent.

    Args:
        config: Subagent configuration.
        parent_model: The parent agent's model name.

    Returns:
        Model name to use, or None to use default.
    """
    if config.model == "inherit":
        return parent_model
    return config.model


class SubagentExecutor:
    """Executor for running subagents."""

    def __init__(
        self,
        config: SubagentConfig,
        tools: list[BaseTool],
        parent_model: str | None = None,
        sandbox_state: SandboxState | None = None,
        thread_data: ThreadDataState | None = None,
        thread_id: str | None = None,
        trace_id: str | None = None,
        authorized_handoff_paths: set[str] | None = None,
        app_config=None,
    ):
        """Initialize the executor.

        Args:
            config: Subagent configuration.
            tools: List of all available tools (will be filtered).
            parent_model: The parent agent's model name for inheritance.
            sandbox_state: Sandbox state from parent agent.
            thread_data: Thread data from parent agent.
            thread_id: Thread ID for sandbox operations.
            trace_id: Trace ID from parent for distributed tracing.
            authorized_handoff_paths: Set of absolute workspace paths the
                subagent is authorized to read via read_file. Populated by
                task_tool from {{handoff://<name>}} placeholders in the lead's
                task prompt. None or empty set = no authorization (subagent
                cannot read peer handoff files). Consumed by
                HandoffIsolationProvider attached to the subagent's middleware
                chain (see Task 11).
            app_config: Reserved for upstream parity (resolved AppConfig).
                Ignored locally — config is fetched via the global cache.
        """
        self.config = config
        self.parent_model = parent_model
        self.sandbox_state = sandbox_state
        self.thread_data = thread_data
        self.thread_id = thread_id
        # Generate trace_id if not provided (for top-level calls)
        self.trace_id = trace_id or str(uuid.uuid4())[:8]
        self.authorized_handoff_paths = authorized_handoff_paths or set()
        self.app_config = app_config

        # Filter tools based on config
        self.tools = _filter_tools(
            tools,
            config.tools,
            config.disallowed_tools,
        )
        logger.info(f"[trace={self.trace_id}] SubagentExecutor initialized: {config.name} with {len(self.tools)} tools")

    def _build_middlewares(self) -> list[AgentMiddleware]:
        """Build the subagent's middleware chain.

        Split out from ``_create_agent`` so callers can inspect the chain (e.g.,
        to size ``recursion_limit`` based on the actual hook count) without
        instantiating the LLM model.
        """
        from deerflow.agents.middlewares.tool_error_handling_middleware import build_subagent_runtime_middlewares

        # Reuse shared middleware composition with lead agent.
        middlewares = build_subagent_runtime_middlewares(lazy_init=True)

        # Attach HandoffIsolationProvider so subagent's read_file on
        # handoff_*.json files is gated by lead's {{handoff://}} authorization.
        from deerflow.guardrails.handoff_isolation_provider import HandoffIsolationProvider
        from deerflow.guardrails.middleware import GuardrailMiddleware

        handoff_isolation = HandoffIsolationProvider(
            authorized_paths=self.authorized_handoff_paths,
            self_outbox_subagent_name=self.config.name,
        )
        middlewares.append(GuardrailMiddleware(
            provider=handoff_isolation,
            passport=f"subagent:{self.config.name}",
        ))

        # Attach ScriptInvocationOnlyProvider so code-executor's bash tool is
        # whitelisted to ethoinsight.scripts.* invocations + safe file ops.
        # Non-code-executor subagents pass through (provider self-gates by agent_id).
        from deerflow.guardrails.script_invocation_only_provider import (
            ScriptInvocationOnlyProvider,
        )

        middlewares.append(GuardrailMiddleware(
            provider=ScriptInvocationOnlyProvider(),
            passport=f"subagent:{self.config.name}",
        ))

        return middlewares

    def _create_agent(self, tools: list[BaseTool] | None = None):
        """Create the agent instance.

        Args:
            tools: Optional tool list to override ``self.tools`` (e.g., after
                applying skill-allowed-tools filtering in ``_build_initial_state``).

        Side effect: caches the built middleware chain on ``self._last_middlewares``
        so ``_aexecute`` can size LangGraph's ``recursion_limit`` against the actual
        hook count without rebuilding (or, when tests ``patch.object`` this method
        wholesale, fall back to a legacy formula).
        """
        model_name = _get_model_name(self.config, self.parent_model)
        model = create_chat_model(name=model_name, thinking_enabled=False)

        middlewares = self._build_middlewares()
        self._last_middlewares = middlewares

        # A-10: system_prompt is injected as a SystemMessage in the initial
        # state (see _build_initial_state), merged with any skill content into
        # a single SystemMessage. Passing system_prompt=None here prevents
        # create_agent from prepending a second SystemMessage — some LLM APIs
        # reject multiple system messages with "System message must be at the
        # beginning."
        return create_agent(
            model=model,
            tools=tools if tools is not None else self.tools,
            middleware=middlewares,
            system_prompt=None,
            state_schema=ThreadState,
        )

    def _build_initial_state(self, task: str) -> tuple[dict[str, Any], list[BaseTool]]:
        """Build the initial state for agent execution.

        Combines the subagent's system_prompt with any skill content into a
        single SystemMessage to avoid the multi-SystemMessage rejection some
        LLM APIs raise (vLLM, Xinference, Chinese providers).

        Args:
            task: The task description.

        Returns:
            Tuple of (initial_state, filtered_tools) — filtered_tools mirrors
            the upstream contract; locally we have no skill-allowed-tools
            policy, so it always equals ``self.tools``.
        """
        system_parts: list[str] = []
        if self.config.system_prompt:
            system_parts.append(self.config.system_prompt)
        if self.config.skills:
            skill_sections = _load_skill_contents(self.config.skills)
            if skill_sections:
                system_parts.append(skill_sections)

        messages: list[Any] = []
        if system_parts:
            messages.append(SystemMessage(content="\n\n".join(system_parts)))
        messages.append(HumanMessage(content=task))

        state: dict[str, Any] = {
            "messages": messages,
        }

        # Pass through sandbox and thread data from parent
        if self.sandbox_state is not None:
            state["sandbox"] = self.sandbox_state
        if self.thread_data is not None:
            state["thread_data"] = self.thread_data

        return state, self.tools

    async def _attempt_seal_resume(
        self,
        agent,
        final_state: dict,
        run_config,
        context: dict,
        result,
        collector,
    ) -> dict | None:
        """漏调 seal 时，就地补一轮聚焦收尾。返回更新后的 final_state，异常时返回 None。

        不强制 tool_choice（探针证明会产空 args），不用 structured output（会 strip
        runtime 注入）。只是给 subagent 一个上下文聚焦的"现在请 seal"轮次——探针实证
        聚焦上下文里 LLM 会正确调 seal 且 args 完整。模型无关（Qwen 切换零影响）。

        ROBUSTNESS: 绝不抛异常。补轮失败 → 返回 None → 外层走 5.7 FAILED 兜底。
        """
        seal_tool = f"seal_{self.config.name.replace('-', '_')}_handoff"
        # 补轮 prompt 措辞（grill 2026-05-29 锁定）：用 HumanMessage 追加（不是第二个
        # SystemMessage——dashscope/create_agent 不接受中途插第二条 SystemMessage，见
        # executor.py:567 已踩坑规避）。措辞保持下方现状即可：它已是"指令 + 终结步骤"
        # 框定（"这是最后一步"），且 2026-05-29 E2E dogfood 实证补轮用它成功调对 seal。
        # 【实施 agent 不要"优化"这段措辞】——能跑通就不动；正面措辞，勿加"不要重新分析/
        # 不要解释"之类反向激活词（CLAUDE.md §6 deepseek 正面提示原则）。
        resume_prompt = (
            f"你上面的分析已经完成。现在请调用 {seal_tool} 工具，"
            f"把你刚才得出的结论（key_findings / 各结构化字段）填入工具参数并落库。"
            f"这是完成本次任务的最后一步——调用该工具后任务即完成。"
        )
        try:
            messages = list(final_state.get("messages", []))

            # Sprint 5.8 前置守卫（grill 锁定 2026-05-29）：必须有实质分析内容
            # (至少一条 AIMessage) 才值得补轮。补轮的全部价值是"让 LLM 把【已经做好的】
            # 分析 seal 掉"；若 history 里连一条 AIMessage 都没有(subagent 根本没产生过
            # 实质分析，如第一轮就被截断/异常早退)，补轮无米下锅——强行补只会让 LLM 用空
            # args 调 seal，产出空 handoff(比漏调更糟)。这种情况返回 None，交给 5.7
            # 兜底重派【整个 subagent】(全新跑一次)。语义：只补"漏了 seal"，不补"没分析"。
            if not any(isinstance(m, AIMessage) for m in messages):
                logger.warning(
                    "[trace=%s] Subagent %s: no AIMessage in history, skip seal-resume "
                    "(no analysis to seal; let 5.7 re-dispatch)",
                    self.trace_id,
                    self.config.name,
                )
                return None

            messages.append(HumanMessage(content=resume_prompt))
            resume_state = {**final_state, "messages": messages}
            # Sprint 5.8 显式重注入 thread_data + sandbox（grill 2026-05-29 锁定）：
            # 补轮里 subagent 调 seal_*_handoff 时，seal tool 从 runtime.state["thread_data"]
            # ["workspace_path"] 取目录写文件（seal_handoff_tools.py:_resolve_workspace）。
            # 探针已实证 graph 的 final_state 确实透传了 thread_data，所以【当前】不注入
            # 也能 seal。但仍显式重注入——零成本(executor 手里就有
            # self.thread_data/self.sandbox_state)、不依赖"graph 透传自定义 state 字段"
            # 这个 langchain 版本相关的隐式行为(版本/middleware 变动可能改变)。显式 > 隐式透传
            # ([[feedback_dev_prod_behavior_alignment]] 精神)。缺 workspace 时 seal 会抛
            # RuntimeError → 被 try/except 吞 → 补轮静默白做，极难诊断，故防御性补这一行。
            if self.thread_data is not None:
                resume_state["thread_data"] = self.thread_data
            if self.sandbox_state is not None:
                resume_state["sandbox"] = self.sandbox_state

            logger.info(
                "[trace=%s] Subagent %s: handoff missing, attempting seal-resume (1 focused turn)",
                self.trace_id,
                self.config.name,
            )
            new_final = None
            async for chunk in agent.astream(
                resume_state, config=run_config, context=context, stream_mode="values"
            ):
                if result.cancel_event.is_set():
                    return new_final  # 尊重取消
                new_final = chunk
                # 收集补轮的 AI message（复用主循环的去重模式）
                msgs = chunk.get("messages", [])
                if msgs and isinstance(msgs[-1], AIMessage):
                    md = msgs[-1].model_dump()
                    mid = md.get("id")
                    dup = (
                        any(m.get("id") == mid for m in result.ai_messages)
                        if mid
                        else md in result.ai_messages
                    )
                    if not dup:
                        result.ai_messages.append(md)
            return new_final if new_final is not None else final_state
        except Exception:
            logger.exception(
                "[trace=%s] Subagent %s: seal-resume attempt failed; falling back to 5.7 FAILED",
                self.trace_id,
                self.config.name,
            )
            return None

    async def _aexecute(self, task: str, result_holder: SubagentResult | None = None) -> SubagentResult:
        """Execute a task asynchronously.

        Args:
            task: The task description for the subagent.
            result_holder: Optional pre-created result object to update during execution.

        Returns:
            SubagentResult with the execution result.
        """
        if result_holder is not None:
            # Use the provided result holder (for async execution with real-time updates)
            result = result_holder
        else:
            # Create a new result for synchronous execution
            task_id = str(uuid.uuid4())[:8]
            result = SubagentResult(
                task_id=task_id,
                trace_id=self.trace_id,
                status=SubagentStatus.RUNNING,
                started_at=datetime.now(),
            )

        collector: SubagentTokenCollector | None = None
        try:
            state, filtered_tools = self._build_initial_state(task)
            agent = self._create_agent(filtered_tools)
            middlewares = getattr(self, "_last_middlewares", None)

            # Token collector for subagent LLM calls — records flow into
            # SubagentResult.token_usage_records and are reported to the
            # parent RunJournal by task_tool when the subagent terminates.
            collector_caller = f"subagent:{self.config.name}"
            collector = SubagentTokenCollector(caller=collector_caller)

            # LangGraph recursion_limit counts node steps. With the current
            # middleware chain each turn dispatches: before_model hooks + model +
            # after_model hooks + tools. The actual turn limit is still enforced
            # by AI message counting below; this just keeps LangGraph from
            # tripping before we get to ``max_turns`` AI messages.
            #
            # ``middlewares`` is cached by ``_create_agent``. When tests mock
            # ``_create_agent`` wholesale, the cache is empty and we fall back
            # to the legacy 2x formula (which never triggers because the mock
            # path doesn't go through real LangGraph either).
            if middlewares is not None:
                recursion_limit = calculate_subagent_recursion_limit(middlewares, self.config.max_turns)
            else:
                recursion_limit = self.config.max_turns * 2 + 1
            run_config: RunnableConfig = {
                "recursion_limit": recursion_limit,
                "callbacks": [collector],
                "tags": [collector_caller],
            }
            context = {}
            if self.thread_id:
                run_config["configurable"] = {"thread_id": self.thread_id}
                context["thread_id"] = self.thread_id

            logger.info(
                f"[trace={self.trace_id}] Subagent {self.config.name} starting async execution with max_turns={self.config.max_turns} (recursion_limit={recursion_limit}, middlewares={len(middlewares) if middlewares is not None else 'unknown'})"
            )

            # Use stream instead of invoke to get real-time updates
            # This allows us to collect AI messages as they are generated
            final_state = None

            # Pre-check: bail out immediately if already cancelled before streaming starts
            if result.cancel_event.is_set():
                logger.info(f"[trace={self.trace_id}] Subagent {self.config.name} cancelled before streaming")
                with _background_tasks_lock:
                    result.try_set_terminal(SubagentStatus.CANCELLED, error="Cancelled by user")
                if collector is not None:
                    result.token_usage_records = collector.snapshot_records()
                return result

            async for chunk in agent.astream(state, config=run_config, context=context, stream_mode="values"):  # type: ignore[arg-type]
                # Cooperative cancellation: check if parent requested stop.
                # Note: cancellation is only detected at astream iteration boundaries,
                # so long-running tool calls within a single iteration will not be
                # interrupted until the next chunk is yielded.
                if result.cancel_event.is_set():
                    logger.info(f"[trace={self.trace_id}] Subagent {self.config.name} cancelled by parent")
                    with _background_tasks_lock:
                        result.try_set_terminal(SubagentStatus.CANCELLED, error="Cancelled by user")
                    result.token_usage_records = collector.snapshot_records()
                    return result

                final_state = chunk

                # Extract AI messages from the current state
                messages = chunk.get("messages", [])
                if messages:
                    last_message = messages[-1]
                    # Check if this is a new AI message
                    if isinstance(last_message, AIMessage):
                        # Convert message to dict for serialization
                        message_dict = last_message.model_dump()
                        # Only add if it's not already in the list (avoid duplicates)
                        # Check by comparing message IDs if available, otherwise compare full dict
                        message_id = message_dict.get("id")
                        is_duplicate = False
                        if message_id:
                            is_duplicate = any(msg.get("id") == message_id for msg in result.ai_messages)
                        else:
                            is_duplicate = message_dict in result.ai_messages

                        if not is_duplicate:
                            result.ai_messages.append(message_dict)
                            logger.info(f"[trace={self.trace_id}] Subagent {self.config.name} captured AI message #{len(result.ai_messages)}")

                            # Hard limit: terminate early when AI message count reaches max_turns
                            if len(result.ai_messages) >= self.config.max_turns:
                                logger.warning(f"[trace={self.trace_id}] Subagent {self.config.name} reached max_turns={self.config.max_turns} AI messages, terminating early")
                                break

            logger.info(f"[trace={self.trace_id}] Subagent {self.config.name} completed async execution")
            result.token_usage_records = collector.snapshot_records()

            if final_state is None:
                logger.warning(f"[trace={self.trace_id}] Subagent {self.config.name} no final state")
                result.result = "No response generated"
            else:
                # Extract the final message - find the last AIMessage
                messages = final_state.get("messages", [])
                logger.info(f"[trace={self.trace_id}] Subagent {self.config.name} final messages count: {len(messages)}")

                # Find the last AIMessage in the conversation
                last_ai_message = None
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage):
                        last_ai_message = msg
                        break

                if last_ai_message is not None:
                    content = last_ai_message.content
                    # Handle both str and list content types for the final result
                    if isinstance(content, str):
                        result.result = content
                    elif isinstance(content, list):
                        # Extract text from list of content blocks for final result only.
                        # Concatenate raw string chunks directly, but preserve separation
                        # between full text blocks for readability.
                        text_parts = []
                        pending_str_parts = []
                        for block in content:
                            if isinstance(block, str):
                                pending_str_parts.append(block)
                            elif isinstance(block, dict):
                                if pending_str_parts:
                                    text_parts.append("".join(pending_str_parts))
                                    pending_str_parts.clear()
                                text_val = block.get("text")
                                if isinstance(text_val, str):
                                    text_parts.append(text_val)
                        if pending_str_parts:
                            text_parts.append("".join(pending_str_parts))
                        result.result = "\n".join(text_parts) if text_parts else "No text content in response"
                    else:
                        result.result = str(content)
                elif messages:
                    # Fallback: use the last message if no AIMessage found
                    last_message = messages[-1]
                    logger.warning(f"[trace={self.trace_id}] Subagent {self.config.name} no AIMessage found, using last message: {type(last_message)}")
                    raw_content = last_message.content if hasattr(last_message, "content") else str(last_message)
                    if isinstance(raw_content, str):
                        result.result = raw_content
                    elif isinstance(raw_content, list):
                        parts = []
                        pending_str_parts = []
                        for block in raw_content:
                            if isinstance(block, str):
                                pending_str_parts.append(block)
                            elif isinstance(block, dict):
                                if pending_str_parts:
                                    parts.append("".join(pending_str_parts))
                                    pending_str_parts.clear()
                                text_val = block.get("text")
                                if isinstance(text_val, str):
                                    parts.append(text_val)
                        if pending_str_parts:
                            parts.append("".join(pending_str_parts))
                        result.result = "\n".join(parts) if parts else "No text content in response"
                    else:
                        result.result = str(raw_content)
                else:
                    logger.warning(f"[trace={self.trace_id}] Subagent {self.config.name} no messages in final state")
                    result.result = "No response generated"

            # Sprint 5.7 + 5.8: validate handoff emission before marking COMPLETED.
            # For ethoinsight subagents (code-executor / data-analyst /
            # chart-maker / report-writer), the expected handoff file MUST exist
            # in the thread workspace. An LLM can finish its thinking turn
            # without emitting the seal_*_handoff tool_call — executor would
            # otherwise mark the task COMPLETED falsely and break downstream.
            #
            # Sprint 5.8: 漏调 seal → 先就地补一轮聚焦收尾，补不上才走 5.7 FAILED。
            # 仅对白名单 subagent（_validate 返回非 None 即在白名单且文件缺失）+ 有
            # final_state 可续跑 + 未被取消时尝试。
            _workspace = (
                self.thread_data.get("workspace_path")
                if isinstance(self.thread_data, dict)
                else None
            )
            _handoff_error = _validate_handoff_emitted(self.config.name, _workspace)

            if (
                _handoff_error is not None
                and final_state is not None
                and not result.cancel_event.is_set()
            ):
                _resumed_state = await self._attempt_seal_resume(
                    agent, final_state, run_config, context, result, collector,
                )
                if _resumed_state is not None:
                    final_state = _resumed_state
                # 补轮后重新校验。
                # ⚠️ 已知未覆盖失败模式（grill 2026-05-29 锁定 B，已立案后续修复）：
                # _validate_handoff_emitted 只查【文件存在】，不查内容。补轮是"被催着
                # 调 seal"的场景，LLM 极小概率会调了 seal 但 args 残缺/空(key_findings=[]
                # 等)→ 文件产出 → 此处判 COMPLETED → 但下游读到空内容 handoff，一路绿灯
                # 产垃圾。探针实证补轮 args 完整(概率低)，且"空内容 handoff"是 seal tool
                # 的【普遍】问题(正常调用也可能偶发，非补轮引入)，在补轮里单独堵不治本。
                # 5.8 不处理；留给独立的「handoff 内容非空/schema 完整性校验」后续 fix
                # (见 spec §8)。
                _handoff_error = _validate_handoff_emitted(self.config.name, _workspace)
                # Sprint 5.8 §3.4: 补轮 token 计入 collector snapshot
                if collector is not None:
                    result.token_usage_records = collector.snapshot_records()

            if _handoff_error is not None:
                logger.warning(
                    "[trace=%s] Subagent %s terminated without emitting handoff "
                    "(seal-resume did not recover): %s",
                    self.trace_id,
                    self.config.name,
                    _handoff_error,
                )
                result.try_set_terminal(SubagentStatus.FAILED, error=_handoff_error)
            else:
                result.try_set_terminal(SubagentStatus.COMPLETED)

        except Exception as e:
            logger.exception(f"[trace={self.trace_id}] Subagent {self.config.name} async execution failed")
            result.try_set_terminal(SubagentStatus.FAILED, error=str(e))
            if collector is not None:
                result.token_usage_records = collector.snapshot_records()

        return result

    def execute(self, task: str, result_holder: SubagentResult | None = None) -> SubagentResult:
        """Execute a task synchronously (wrapper around async execution).

        This method runs the async execution in a new event loop, allowing
        asynchronous tools (like MCP tools) to be used within the thread pool.

        Args:
            task: The task description for the subagent.
            result_holder: Optional pre-created result object to update during execution.

        Returns:
            SubagentResult with the execution result.
        """
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None and loop.is_running():
                logger.debug(f"[trace={self.trace_id}] Subagent {self.config.name} detected running event loop, using persistent isolated loop")
                # Snapshot parent ContextVar state (e.g., user_id) so that the
                # subagent coroutine running on the isolated loop sees the same
                # auth context as the lead agent. Without copy_context() here, the
                # coroutine would run with an empty context and any
                # ContextVar.get() (such as runtime/user_context._current_user)
                # would fall back to its default ("default" user_id).
                parent_context = copy_context()
                future: Future[SubagentResult] | None = None
                try:
                    future = _submit_to_isolated_loop_in_context(
                        parent_context,
                        lambda: self._aexecute(task, result_holder),
                    )
                    return future.result(timeout=self.config.timeout_seconds)
                except FuturesTimeoutError:
                    if result_holder is not None:
                        result_holder.cancel_event.set()
                    if future is not None:
                        future.cancel()
                    raise

            # Standard path: no running event loop, use asyncio.run
            # asyncio.run preserves the current context for the coroutine, so
            # ContextVars set in the calling thread are visible inside _aexecute.
            return asyncio.run(self._aexecute(task, result_holder))
        except Exception as e:
            logger.exception(f"[trace={self.trace_id}] Subagent {self.config.name} execution failed")
            # Create a result with error if we don't have one
            if result_holder is not None:
                result = result_holder
            else:
                # Initialise as RUNNING so try_set_terminal can transition it
                # to FAILED; if we initialised as FAILED directly, the CAS
                # guard would reject the error/completed_at write.
                result = SubagentResult(
                    task_id=str(uuid.uuid4())[:8],
                    trace_id=self.trace_id,
                    status=SubagentStatus.RUNNING,
                )
            result.try_set_terminal(SubagentStatus.FAILED, error=str(e))
            return result

    def execute_async(self, task: str, task_id: str | None = None) -> str:
        """Start a task execution in the background.

        Args:
            task: The task description for the subagent.
            task_id: Optional task ID to use. If not provided, a random UUID will be generated.

        Returns:
            Task ID that can be used to check status later.
        """
        # Use provided task_id or generate a new one
        if task_id is None:
            task_id = str(uuid.uuid4())[:8]

        # Create initial pending result
        result = SubagentResult(
            task_id=task_id,
            trace_id=self.trace_id,
            status=SubagentStatus.PENDING,
        )

        logger.info(f"[trace={self.trace_id}] Subagent {self.config.name} starting async execution, task_id={task_id}, timeout={self.config.timeout_seconds}s")

        with _background_tasks_lock:
            _background_tasks[task_id] = result

        # Snapshot parent ContextVar state BEFORE submitting to thread pool.
        # _scheduler_pool is a ThreadPoolExecutor, which does NOT propagate
        # contextvars across thread boundaries. We must capture the parent
        # context here (still on the calling task's thread) and explicitly
        # replay it inside _submit_to_isolated_loop_in_context.
        parent_context = copy_context()

        def run_task():
            with _background_tasks_lock:
                _background_tasks[task_id].status = SubagentStatus.RUNNING
                _background_tasks[task_id].started_at = datetime.now()
                result_holder = _background_tasks[task_id]

            try:
                # Submit execution directly to the persistent isolated loop so the
                # background path does not create a temporary loop via execute().
                # Pass the captured parent_context so user_id and other
                # ContextVars propagate into the subagent coroutine.
                execution_future = _submit_to_isolated_loop_in_context(
                    parent_context,
                    lambda: self._aexecute(task, result_holder),
                )
                try:
                    # Wait for execution with timeout
                    exec_result = execution_future.result(timeout=self.config.timeout_seconds)
                    with _background_tasks_lock:
                        # exec_result already set terminal status via try_set_terminal
                        # inside _aexecute. Mirror its final fields into the
                        # shared _background_tasks entry. We use try_set_terminal
                        # so the timeout path (if it raced) wins instead of being
                        # overwritten.
                        _background_tasks[task_id].try_set_terminal(
                            exec_result.status,
                            result=exec_result.result,
                            error=exec_result.error,
                            ai_messages=exec_result.ai_messages,
                        )
                except FuturesTimeoutError:
                    logger.error(f"[trace={self.trace_id}] Subagent {self.config.name} execution timed out after {self.config.timeout_seconds}s")
                    with _background_tasks_lock:
                        _background_tasks[task_id].try_set_terminal(
                            SubagentStatus.TIMED_OUT,
                            error=f"Execution timed out after {self.config.timeout_seconds} seconds",
                        )
                    # Signal cooperative cancellation and cancel the future
                    result_holder.cancel_event.set()
                    execution_future.cancel()
            except Exception as e:
                logger.exception(f"[trace={self.trace_id}] Subagent {self.config.name} async execution failed")
                with _background_tasks_lock:
                    _background_tasks[task_id].try_set_terminal(SubagentStatus.FAILED, error=str(e))

        _scheduler_pool.submit(run_task)
        return task_id


MAX_CONCURRENT_SUBAGENTS = 3


def request_cancel_background_task(task_id: str) -> None:
    """Signal a running background task to stop.

    Sets the cancel_event on the task, which is checked cooperatively
    by ``_aexecute`` during ``agent.astream()`` iteration.  This allows
    subagent threads — which cannot be force-killed via ``Future.cancel()``
    — to stop at the next iteration boundary.

    Args:
        task_id: The task ID to cancel.
    """
    with _background_tasks_lock:
        result = _background_tasks.get(task_id)
        if result is not None:
            result.cancel_event.set()
            logger.info("Requested cancellation for background task %s", task_id)


def get_background_task_result(task_id: str) -> SubagentResult | None:
    """Get the result of a background task.

    Args:
        task_id: The task ID returned by execute_async.

    Returns:
        SubagentResult if found, None otherwise.
    """
    with _background_tasks_lock:
        return _background_tasks.get(task_id)


def list_background_tasks() -> list[SubagentResult]:
    """List all background tasks.

    Returns:
        List of all SubagentResult instances.
    """
    with _background_tasks_lock:
        return list(_background_tasks.values())


def cleanup_background_task(task_id: str) -> None:
    """Remove a completed task from background tasks.

    Should be called by task_tool after it finishes polling and returns the result.
    This prevents memory leaks from accumulated completed tasks.

    Only removes tasks that are in a terminal state (COMPLETED/FAILED/TIMED_OUT)
    to avoid race conditions with the background executor still updating the task entry.

    Args:
        task_id: The task ID to remove.
    """
    with _background_tasks_lock:
        result = _background_tasks.get(task_id)
        if result is None:
            # Nothing to clean up; may have been removed already.
            logger.debug("Requested cleanup for unknown background task %s", task_id)
            return

        # Only clean up tasks that are in a terminal state to avoid races with
        # the background executor still updating the task entry.
        is_terminal_status = result.status in {
            SubagentStatus.COMPLETED,
            SubagentStatus.FAILED,
            SubagentStatus.CANCELLED,
            SubagentStatus.TIMED_OUT,
        }
        if is_terminal_status or result.completed_at is not None:
            del _background_tasks[task_id]
            logger.debug("Cleaned up background task: %s", task_id)
        else:
            logger.debug(
                "Skipping cleanup for non-terminal background task %s (status=%s)",
                task_id,
                result.status.value if hasattr(result.status, "value") else result.status,
            )
