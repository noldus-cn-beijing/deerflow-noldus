"""Lead agent factory.

INVARIANT — tracing callback placement
======================================

Tracing callbacks (Langfuse, LangSmith) are attached at the **graph
invocation root** in :func:`make_lead_agent` (see the
``build_tracing_callbacks()`` block that appends to ``config["callbacks"]``).
Every ``create_chat_model(...)`` call inside this module — and inside any
middleware reachable from this graph (e.g. ``TitleMiddleware``) — MUST pass
``attach_tracing=False``.

Forgetting that flag emits duplicate spans (one rooted at the graph, one at
the model) AND prevents the Langfuse handler's ``propagate_attributes``
path from firing, so ``session_id`` / ``user_id`` never reach the trace.
The four current sites are: bootstrap agent, default agent, summarization
middleware, and the async path inside ``TitleMiddleware``. Any new in-graph
``create_chat_model`` call must add to this list and pass the flag.
"""

import logging

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.runnables import RunnableConfig

from deerflow.agents.lead_agent.prompt import apply_prompt_template
from deerflow.agents.middlewares.archiving_summarization import ArchivingSummarizationMiddleware
from deerflow.agents.middlewares.clarification_middleware import ClarificationMiddleware
from deerflow.agents.middlewares.loop_detection_middleware import LoopDetectionMiddleware
from deerflow.agents.middlewares.memory_middleware import MemoryMiddleware
from deerflow.agents.middlewares.paradigm_identification_gate_middleware import ParadigmIdentificationGateMiddleware
from deerflow.agents.middlewares.safety_finish_reason_middleware import SafetyFinishReasonMiddleware
from deerflow.agents.middlewares.subagent_limit_middleware import SubagentLimitMiddleware
from deerflow.agents.middlewares.think_tag_middleware import ThinkTagMiddleware
from deerflow.agents.middlewares.title_middleware import TitleMiddleware
from deerflow.agents.middlewares.todo_middleware import TodoMiddleware
from deerflow.agents.middlewares.token_usage_middleware import TokenUsageMiddleware
from deerflow.agents.middlewares.tool_error_handling_middleware import build_lead_runtime_middlewares
from deerflow.agents.middlewares.training_data_middleware import TrainingDataMiddleware
from deerflow.agents.middlewares.view_image_middleware import ViewImageMiddleware
from deerflow.agents.thread_state import ThreadState
from deerflow.config.agents_config import load_agent_config, validate_agent_name
from deerflow.config.app_config import get_app_config
from deerflow.config.summarization_config import get_summarization_config
from deerflow.guardrails.middleware import GuardrailMiddleware
from deerflow.models import create_chat_model
from deerflow.runtime.user_context import set_current_user
from deerflow.tracing import build_tracing_callbacks

logger = logging.getLogger(__name__)


class _AuthUser:
    """Minimal duck-typed user satisfying the CurrentUser Protocol.

    Used to bridge LangGraph's `langgraph_auth_user_id` (a plain str) into
    deerflow's user_context, which expects an object with an `.id` attribute.
    """

    __slots__ = ("id",)

    def __init__(self, user_id: str) -> None:
        self.id = user_id


def _resolve_model_name(requested_model_name: str | None = None) -> str:
    """Resolve a runtime model name safely, falling back to default if invalid. Returns None if no models are configured."""
    app_config = get_app_config()
    default_model_name = app_config.models[0].name if app_config.models else None
    if default_model_name is None:
        raise ValueError("No chat models are configured. Please configure at least one model in config.yaml.")

    if requested_model_name and app_config.get_model_config(requested_model_name):
        return requested_model_name

    if requested_model_name and requested_model_name != default_model_name:
        logger.warning(f"Model '{requested_model_name}' not found in config; fallback to default model '{default_model_name}'.")
    return default_model_name


def _create_summarization_middleware() -> ArchivingSummarizationMiddleware | None:
    """Create and configure the summarization middleware from config."""
    config = get_summarization_config()

    if not config.enabled:
        return None

    # Prepare trigger parameter
    trigger = None
    if config.trigger is not None:
        if isinstance(config.trigger, list):
            trigger = [t.to_tuple() for t in config.trigger]
        else:
            trigger = config.trigger.to_tuple()

    # Prepare keep parameter
    keep = config.keep.to_tuple()

    # Prepare model parameter
    # attach_tracing=False because the graph-level RunnableConfig (set in
    # ``make_lead_agent``) already carries tracing callbacks; binding them
    # again at the model level would emit duplicate spans and break
    # ``session_id`` / ``user_id`` propagation.
    if config.model_name:
        model = create_chat_model(name=config.model_name, thinking_enabled=False, attach_tracing=False)
    else:
        # Use a lightweight model for summarization to save costs
        # Falls back to default model if not explicitly specified
        model = create_chat_model(thinking_enabled=False, attach_tracing=False)

    # Prepare kwargs
    kwargs = {
        "model": model,
        "trigger": trigger,
        "keep": keep,
    }

    if config.trim_tokens_to_summarize is not None:
        kwargs["trim_tokens_to_summarize"] = config.trim_tokens_to_summarize

    if config.summary_prompt is not None:
        kwargs["summary_prompt"] = config.summary_prompt

    kwargs["preserve_recent_skill_count"] = config.preserve_recent_skill_count
    kwargs["preserve_recent_skill_tokens"] = config.preserve_recent_skill_tokens
    kwargs["preserve_recent_skill_tokens_per_skill"] = config.preserve_recent_skill_tokens_per_skill
    kwargs["skills_container_path"] = get_app_config().skills.container_path

    return ArchivingSummarizationMiddleware(**kwargs)


def _create_todo_list_middleware(is_plan_mode: bool) -> TodoMiddleware | None:
    """Create and configure the TodoList middleware.

    Args:
        is_plan_mode: Whether to enable plan mode with TodoList middleware.

    Returns:
        TodoMiddleware instance if plan mode is enabled, None otherwise.
    """
    if not is_plan_mode:
        return None

    # Custom prompts matching DeerFlow's style
    system_prompt = """
<todo_list_system>
You have access to the `write_todos` tool to help you manage and track complex multi-step objectives.

**CRITICAL RULES:**
- Mark todos as completed IMMEDIATELY after finishing each step - do NOT batch completions
- Keep EXACTLY ONE task as `in_progress` at any time (unless tasks can run in parallel)
- Update the todo list in REAL-TIME as you work - this gives users visibility into your progress
- DO NOT use this tool for simple tasks (< 3 steps) - just complete them directly

**When to Use:**
This tool is designed for complex objectives that require systematic tracking:
- Complex multi-step tasks requiring 3+ distinct steps
- Non-trivial tasks needing careful planning and execution
- User explicitly requests a todo list
- User provides multiple tasks (numbered or comma-separated list)
- The plan may need revisions based on intermediate results

**When NOT to Use:**
- Single, straightforward tasks
- Trivial tasks (< 3 steps)
- Purely conversational or informational requests
- Simple tool calls where the approach is obvious

**Best Practices:**
- Break down complex tasks into smaller, actionable steps
- Use clear, descriptive task names
- Remove tasks that become irrelevant
- Add new tasks discovered during implementation
- Don't be afraid to revise the todo list as you learn more

**Task Management:**
Writing todos takes time and tokens - use it when helpful for managing complex problems, not for simple requests.
</todo_list_system>
"""

    tool_description = """Use this tool to create and manage a structured task list for complex work sessions.

**IMPORTANT: Only use this tool for complex tasks (3+ steps). For simple requests, just do the work directly.**

## When to Use

Use this tool in these scenarios:
1. **Complex multi-step tasks**: When a task requires 3 or more distinct steps or actions
2. **Non-trivial tasks**: Tasks requiring careful planning or multiple operations
3. **User explicitly requests todo list**: When the user directly asks you to track tasks
4. **Multiple tasks**: When users provide a list of things to be done
5. **Dynamic planning**: When the plan may need updates based on intermediate results

## When NOT to Use

Skip this tool when:
1. The task is straightforward and takes less than 3 steps
2. The task is trivial and tracking provides no benefit
3. The task is purely conversational or informational
4. It's clear what needs to be done and you can just do it

## How to Use

1. **Starting a task**: Mark it as `in_progress` BEFORE beginning work
2. **Completing a task**: Mark it as `completed` IMMEDIATELY after finishing
3. **Updating the list**: Add new tasks, remove irrelevant ones, or update descriptions as needed
4. **Multiple updates**: You can make several updates at once (e.g., complete one task and start the next)

## Task States

- `pending`: Task not yet started
- `in_progress`: Currently working on (can have multiple if tasks run in parallel)
- `completed`: Task finished successfully

## Task Completion Requirements

**CRITICAL: Only mark a task as completed when you have FULLY accomplished it.**

Never mark a task as completed if:
- There are unresolved issues or errors
- Work is partial or incomplete
- You encountered blockers preventing completion
- You couldn't find necessary resources or dependencies
- Quality standards haven't been met

If blocked, keep the task as `in_progress` and create a new task describing what needs to be resolved.

## Best Practices

- Create specific, actionable items
- Break complex tasks into smaller, manageable steps
- Use clear, descriptive task names
- Update task status in real-time as you work
- Mark tasks complete IMMEDIATELY after finishing (don't batch completions)
- Remove tasks that are no longer relevant
- **IMPORTANT**: When you write the todo list, mark your first task(s) as `in_progress` immediately
- **IMPORTANT**: Unless all tasks are completed, always have at least one task `in_progress` to show progress

Being proactive with task management demonstrates thoroughness and ensures all requirements are completed successfully.

**Remember**: If you only need a few tool calls to complete a task and it's clear what to do, it's better to just do the task directly and NOT use this tool at all.
"""

    return TodoMiddleware(system_prompt=system_prompt, tool_description=tool_description)


# ThreadDataMiddleware must be before SandboxMiddleware to ensure thread_id is available
# UploadsMiddleware should be after ThreadDataMiddleware to access thread_id
# DanglingToolCallMiddleware patches missing ToolMessages before model sees the history
# SummarizationMiddleware should be early to reduce context before other processing
# TodoListMiddleware should be before ClarificationMiddleware to allow todo management
# TitleMiddleware generates title after first exchange
# MemoryMiddleware queues conversation for memory update (after TitleMiddleware)
# ViewImageMiddleware should be before ClarificationMiddleware to inject image details before LLM
# ToolErrorHandlingMiddleware should be before ClarificationMiddleware to convert tool exceptions to ToolMessages
# ClarificationMiddleware should be last to intercept clarification requests after model calls
# Lead agent 不该有 bash/write_file/str_replace —— 所有 ethoinsight CLI
# 调用走 prep_metric_plan 工具，所有写文件操作走 code-executor 子代理。
# 这是 P0 修复：lead 无 bash → 无 quoting retry → 无 recursion 100 耗尽。
# (subagent 通过 SubagentConfig.tools 显式声明 bash，不受此过滤影响)
_LEAD_EXCLUDED_TOOLS: frozenset[str] = frozenset({"bash", "write_file", "str_replace"})


def _filter_lead_tools(tools: list, excluded: frozenset[str]) -> list:
    """Drop tools whose .name is in excluded set. Pure function — single source of truth for the lead exclusion policy."""
    return [t for t in tools if t.name not in excluded]


def build_middlewares(config: RunnableConfig, model_name: str | None, agent_name: str | None = None, custom_middlewares: list[AgentMiddleware] | None = None, *, deferred_setup=None):
    """Build the lead-agent middleware chain based on runtime configuration.

    Public entry point for the lead agent's full middleware composition. Used by
    ``make_lead_agent`` and by the embedded ``DeerFlowClient`` (a lead-agent variant
    that needs the identical chain).

    Args:
        config: Runtime configuration containing configurable options like is_plan_mode.
        model_name: Resolved runtime model name; gates vision-only middleware.
        agent_name: If provided, MemoryMiddleware will use per-agent memory storage.
        custom_middlewares: Optional list of custom middlewares to inject into the chain.
        deferred_setup: Optional deferred-MCP-tool setup that attaches
            ``DeferredToolFilterMiddleware`` when ``tool_search`` is enabled.

    Returns:
        List of middleware instances.
    """
    middlewares = build_lead_runtime_middlewares(app_config=get_app_config(), lazy_init=True)

    # LoopDetectionMiddleware — detect and break repetitive tool call loops.
    # Created early so it can be passed to summarization middleware for reset-on-compact.
    loop_detection = LoopDetectionMiddleware.from_config(get_app_config().loop_detection)

    # Add summarization middleware if enabled
    summarization_middleware = _create_summarization_middleware()
    if summarization_middleware is not None:
        summarization_middleware._loop_detection = loop_detection
        middlewares.append(summarization_middleware)

    # Add TodoList middleware if plan mode is enabled
    is_plan_mode = config.get("configurable", {}).get("is_plan_mode", False)
    todo_list_middleware = _create_todo_list_middleware(is_plan_mode)
    if todo_list_middleware is not None:
        middlewares.append(todo_list_middleware)

    # Add TokenUsageMiddleware when token_usage tracking is enabled
    if get_app_config().token_usage.enabled:
        middlewares.append(TokenUsageMiddleware())

    # Add TitleMiddleware
    middlewares.append(TitleMiddleware())

    # Add MemoryMiddleware (after TitleMiddleware)
    middlewares.append(MemoryMiddleware(agent_name=agent_name))

    # Add TrainingDataMiddleware (records every turn for SFT/DPO dataset).
    # Sits alongside MemoryMiddleware as an after_agent observer. Failures are
    # swallowed internally so recording errors never crash the agent turn.
    middlewares.append(TrainingDataMiddleware())

    # Add ViewImageMiddleware only if the current model supports vision.
    # Use the resolved runtime model_name from make_lead_agent to avoid stale config values.
    app_config = get_app_config()
    model_config = app_config.get_model_config(model_name) if model_name else None
    if model_config is not None and model_config.supports_vision:
        middlewares.append(ViewImageMiddleware())

    # Hide deferred tool schemas from model binding until tool_search promotes them.
    # The deferred set + catalog hash come from the build-time setup (assembled
    # after tool-policy filtering); promotion is read from graph state.
    if deferred_setup is not None and deferred_setup.deferred_names:
        from deerflow.agents.middlewares.deferred_tool_filter_middleware import DeferredToolFilterMiddleware

        middlewares.append(DeferredToolFilterMiddleware(deferred_setup.deferred_names, deferred_setup.catalog_hash))

    # Add SubagentLimitMiddleware to truncate excess parallel task calls
    subagent_enabled = config.get("configurable", {}).get("subagent_enabled", True)
    if subagent_enabled:
        max_concurrent_subagents = config.get("configurable", {}).get("max_concurrent_subagents", 3)
        middlewares.append(SubagentLimitMiddleware(max_concurrent=max_concurrent_subagents))

    # ParadigmIdentificationGateMiddleware — after_model gate to force
    # identify_ev19_template call when uploaded data is present (layer 3a).
    # Placed before LoopDetection so reminders are counted properly.
    middlewares.append(ParadigmIdentificationGateMiddleware())

    # LoopDetectionMiddleware — append the instance created earlier
    middlewares.append(loop_detection)

    # ThinkTagMiddleware — route inline <think>...</think> content from
    # assistant text into additional_kwargs.reasoning_content so the frontend
    # renders it in a collapsible Reasoning block rather than the main bubble.
    middlewares.append(ThinkTagMiddleware())

    # Ev19TemplateGuardrail — block task(code-executor) when ev19_template is unset
    from deerflow.config.guardrails_config import get_guardrails_config

    guardrails_cfg = get_guardrails_config()
    if guardrails_cfg.enabled:
        from deerflow.guardrails.ev19_template_provider import (
            Ev19TemplateGuardrailProvider,
            Ev19WorkspaceBridgeMiddleware,
        )

        provider = Ev19TemplateGuardrailProvider()
        middlewares.append(Ev19WorkspaceBridgeMiddleware())
        middlewares.append(GuardrailMiddleware(provider=provider, fail_closed=guardrails_cfg.fail_closed))

        # InspectGateGuardrail — block ask_clarification when identify_ev19_template
        # hasn't been called with uploaded data (layer 3b safety net).
        from deerflow.guardrails.inspect_gate_provider import (
            InspectGateBridgeMiddleware,
            InspectGateGuardrailProvider,
        )

        middlewares.append(InspectGateBridgeMiddleware())
        middlewares.append(GuardrailMiddleware(provider=InspectGateGuardrailProvider(), fail_closed=guardrails_cfg.fail_closed))

        # S5: DataQualityGuardrailProvider — block downstream subagents on
        # critical+blocks_downstream warnings (manual mode only)
        workflow_mode_dq = config.get("configurable", {}).get("workflow_mode", "auto")
        if workflow_mode_dq == "manual":
            from deerflow.guardrails.data_quality_provider import DataQualityGuardrailProvider

            dq_provider = DataQualityGuardrailProvider()
            middlewares.append(GuardrailMiddleware(provider=dq_provider, fail_closed=guardrails_cfg.fail_closed))

        # W17: Intent classification — block non-read_file calls before lead declares [intent]
        from deerflow.guardrails.intent_classification_provider import (
            IntentBridgeMiddleware,
            IntentClassificationGuardrailProvider,
        )

        middlewares.append(IntentBridgeMiddleware())
        middlewares.append(GuardrailMiddleware(
            provider=IntentClassificationGuardrailProvider(),
            fail_closed=guardrails_cfg.fail_closed,
        ))

        # W20: IntentPostStepAskGate — 拦截 ASKVIZ 流程中跳过 ask(viz?) 直接派 chart-maker
        from deerflow.guardrails.intent_post_step_ask_gate_provider import (
            IntentPostStepAskGateBridge,
            IntentPostStepAskGateProvider,
        )

        middlewares.append(IntentPostStepAskGateBridge())
        middlewares.append(GuardrailMiddleware(
            provider=IntentPostStepAskGateProvider(),
            fail_closed=guardrails_cfg.fail_closed,
        ))

        # W19 (relocated): Inject {{handoff://X}} placeholders into task() prompts
        # BEFORE the W18 guardrail evaluates them. Without this middleware, lead
        # tasks that omit the placeholder are unconditionally denied by the W18
        # guardrail because task_tool's own injection runs strictly after the
        # guardrail (root cause of 2026-05-19 dogfood thread 51b00ac8).
        # Must be appended BEFORE W18 (first-appended = outermost in wrap_tool_call).
        from deerflow.agents.middlewares.handoff_placeholder_injection_middleware import (
            HandoffPlaceholderInjectionMiddleware,
        )

        middlewares.append(HandoffPlaceholderInjectionMiddleware())

        # W18a: Path sequence — 校验 task(X) 按路径顺序派遣(前序 dispatch 已完成)
        # 在 W18 之前: 顺序校验先于占位符校验,给出更早更具体的反馈
        from deerflow.guardrails.path_sequence_provider import (
            PathSequenceBridge,
            PathSequenceProvider,
        )

        middlewares.append(PathSequenceBridge())
        middlewares.append(GuardrailMiddleware(
            provider=PathSequenceProvider(),
            fail_closed=guardrails_cfg.fail_closed,
        ))

        # W18: Task handoff authorization — 校验 task(subagent_type=X) prompt 含必需 {{handoff://X}} 占位符
        from deerflow.guardrails.task_handoff_authorization_provider import (
            TaskHandoffAuthorizationProvider,
        )

        middlewares.append(GuardrailMiddleware(
            provider=TaskHandoffAuthorizationProvider(),
            fail_closed=guardrails_cfg.fail_closed,
        ))

    # Inject custom middlewares before ClarificationMiddleware
    if custom_middlewares:
        middlewares.extend(custom_middlewares)

    # TodoPlanningDisciplineProvider — rate-limit write_todos after initial planning.
    # Bridge middleware must be placed BEFORE the GuardrailMiddleware so the
    # contextvar is set when the provider evaluates the tool call.
    from deerflow.guardrails.todo_planning_discipline_provider import (
        TodoDisciplineBridgeMiddleware,
        TodoPlanningDisciplineProvider,
    )

    middlewares.append(TodoDisciplineBridgeMiddleware())
    middlewares.append(GuardrailMiddleware(provider=TodoPlanningDisciplineProvider(), fail_closed=False, name="GuardrailMiddleware[todo-planning-discipline]"))

    # GateEnforcementMiddleware — block task() before Gate 1 in manual mode
    workflow_mode = config.get("configurable", {}).get("workflow_mode", "auto")
    if workflow_mode == "manual":
        from deerflow.agents.middlewares.gate_enforcement_middleware import GateEnforcementMiddleware

        middlewares.append(GateEnforcementMiddleware(enabled=True))

    # QualityWarningBroadcastMiddleware — surface data-analyst handoff
    # quality_warnings onto the broadcast AIMessage so the frontend banner
    # can render. Active in both auto and manual modes.
    from deerflow.agents.middlewares.quality_warning_broadcast_middleware import (
        QualityWarningBroadcastMiddleware,
    )

    middlewares.append(QualityWarningBroadcastMiddleware())

    # DegradationCircuitBreakerMiddleware (P7) — detect statistics crash from
    # code-executor handoff (gate_signals.statistics_status=='crashed'), bounded
    # self-help (jump_to=model, re-dispatch code-executor) then HITL (let model call
    # ask_clarification). Placed after QualityWarningBroadcast (both read
    # handoff_code_executor.json on the lead's broadcast/content-only turn, same
    # precondition) and before SafetyFinishReason / ClarificationMiddleware (so its
    # jump_to='model' reminder is not stripped/short-circuited, and a model-issued
    # ask_clarification can be intercepted by ClarificationMiddleware). Lazy import —
    # harness import-cycle 铁律。
    from deerflow.agents.middlewares.degradation_circuit_breaker_middleware import (
        DegradationCircuitBreakerMiddleware,
    )

    middlewares.append(DegradationCircuitBreakerMiddleware())

    # SafetyFinishReasonMiddleware — suppress tool execution when the provider
    # safety-terminates the response (OpenAI content_filter / Anthropic refusal /
    # Gemini SAFETY). Appended near the end so LangChain's reverse-order
    # after_model dispatch lets it strip stale tool_calls *first*; downstream
    # Loop / Subagent accounting then sees a clean AIMessage. See
    # safety_finish_reason_middleware.py docstring.
    safety_config = app_config.safety_finish_reason
    if safety_config.enabled:
        middlewares.append(SafetyFinishReasonMiddleware.from_config(safety_config))

    # ClarificationMiddleware should always be last
    middlewares.append(ClarificationMiddleware())
    return middlewares


def make_lead_agent(config: RunnableConfig):
    # Lazy import to avoid circular dependency
    from deerflow.tools import get_available_tools
    from deerflow.tools.builtins import setup_agent

    cfg = config.get("configurable", {})

    # Copy LangGraph's auth user_id into the deerflow ContextVar.
    # `make_lead_agent` runs on the bg-loop worker task that subsequently
    # invokes the middlewares (UploadsMiddleware, ThreadDataMiddleware, …),
    # so a ContextVar set here is task-local and visible to every middleware
    # in this run. It cannot be set in `authenticate()` / `@auth.on` because
    # those run in the request-handling thread, where ContextVars do not
    # propagate to the bg-loop asyncio task.
    auth_user_id = cfg.get("langgraph_auth_user_id")
    if auth_user_id:
        set_current_user(_AuthUser(str(auth_user_id)))

    thinking_enabled = cfg.get("thinking_enabled", True)
    reasoning_effort = cfg.get("reasoning_effort", None)

    # Downgrade reasoning_effort based on Gate completion phase.
    # Gate 1 done → step down once (high→medium); Gate 2 done → step down twice (high→low).
    # This avoids wasting reasoning tokens on dispatch tasks that don't need deep thinking.
    def _step_down(effort: str | None) -> str | None:
        if effort == "high":
            return "medium"
        if effort == "medium":
            return "low"
        return effort  # "low", None, or unknown → keep

    thread_id = cfg.get("thread_id")
    # Resolve the current paradigm + user from the thread workspace once, so it can
    # feed both reasoning_effort downgrade and Sprint 8 prior-corrections injection.
    lead_paradigm: str | None = None
    lead_user_id: str | None = None
    if thread_id:
        try:
            from deerflow.agents.middlewares.experiment_context import read_context
            from deerflow.runtime.user_context import get_effective_user_id

            lead_user_id = get_effective_user_id()
            app_config = get_app_config()
            workspace = app_config.paths.sandbox_work_dir(thread_id, user_id=lead_user_id)
            ctx = read_context(str(workspace))
            if ctx:
                lead_paradigm = ctx.get("paradigm")
                gate_completed = ctx.get("gate_completed", [])
                if reasoning_effort and isinstance(gate_completed, list):
                    if "gate2_quality_acknowledged" in gate_completed:
                        reasoning_effort = _step_down(_step_down(reasoning_effort))
                    elif "gate1_paradigm" in gate_completed:
                        reasoning_effort = _step_down(reasoning_effort)
        except Exception:
            pass  # fail-safe: keep configured reasoning_effort, no paradigm context

    requested_model_name: str | None = cfg.get("model_name") or cfg.get("model")
    is_plan_mode = cfg.get("is_plan_mode", False)
    workflow_mode = cfg.get("workflow_mode", "auto")  # "manual" | "auto"
    subagent_enabled = cfg.get("subagent_enabled", True)
    max_concurrent_subagents = cfg.get("max_concurrent_subagents", 3)
    is_bootstrap = cfg.get("is_bootstrap", False)
    agent_name = validate_agent_name(cfg.get("agent_name"))

    agent_config = load_agent_config(agent_name) if not is_bootstrap else None
    # Custom agent model from agent config (if any), or None to let _resolve_model_name pick the default
    agent_model_name = agent_config.model if agent_config and agent_config.model else None

    # Final model name resolution: request → agent config → global default, with fallback for unknown names
    model_name = _resolve_model_name(requested_model_name or agent_model_name)

    app_config = get_app_config()
    model_config = app_config.get_model_config(model_name)

    if model_config is None:
        raise ValueError("No chat model could be resolved. Please configure at least one model in config.yaml or provide a valid 'model_name'/'model' in the request.")
    if thinking_enabled and not model_config.supports_thinking:
        logger.warning(f"Thinking mode is enabled but model '{model_name}' does not support it; fallback to non-thinking mode.")
        thinking_enabled = False

    logger.info(
        "Create Agent(%s) -> thinking_enabled: %s, reasoning_effort: %s, model_name: %s, is_plan_mode: %s, subagent_enabled: %s, max_concurrent_subagents: %s",
        agent_name or "default",
        thinking_enabled,
        reasoning_effort,
        model_name,
        is_plan_mode,
        subagent_enabled,
        max_concurrent_subagents,
    )

    # Inject run metadata for LangSmith trace tagging
    if "metadata" not in config:
        config["metadata"] = {}

    config["metadata"].update(
        {
            "agent_name": agent_name or "default",
            "model_name": model_name or "default",
            "thinking_enabled": thinking_enabled,
            "reasoning_effort": reasoning_effort,
            "is_plan_mode": is_plan_mode,
            "subagent_enabled": subagent_enabled,
        }
    )

    # Inject tracing callbacks at the graph invocation root so a single LangGraph
    # run produces one trace with all node / LLM / tool calls as child spans,
    # AND so the Langfuse handler sees ``on_chain_start(parent_run_id=None)`` and
    # actually propagates ``langfuse_session_id`` / ``langfuse_user_id`` from
    # ``config["metadata"]`` onto the trace. Without root-level attachment the
    # model is a nested observation and the handler strips ``langfuse_*`` keys.
    tracing_callbacks = build_tracing_callbacks()
    if tracing_callbacks:
        existing = config.get("callbacks") or []
        if not isinstance(existing, list):
            existing = list(existing)
        config["callbacks"] = [*existing, *tracing_callbacks]

    if is_bootstrap:
        # Special bootstrap agent with minimal prompt for initial custom agent creation flow
        raw_tools = get_available_tools(model_name=model_name, subagent_enabled=subagent_enabled) + [setup_agent]
        from deerflow.tools.builtins.tool_search import assemble_deferred_tools
        final_tools, setup = assemble_deferred_tools(raw_tools, enabled=app_config.tool_search.enabled)
        return create_agent(
            model=create_chat_model(name=model_name, thinking_enabled=thinking_enabled, attach_tracing=False),
            tools=final_tools,
            middleware=build_middlewares(config, model_name=model_name, deferred_setup=setup),
            system_prompt=apply_prompt_template(subagent_enabled=subagent_enabled, max_concurrent_subagents=max_concurrent_subagents, available_skills=set(["bootstrap"]), thread_id=thread_id, deferred_names=setup.deferred_names),
            state_schema=ThreadState,
        )

    # Default lead agent (unchanged behavior)
    # Resolve tool groups: custom agent config > config.yaml tool_groups > None (all)
    lead_tool_groups = None
    if agent_config and agent_config.tool_groups:
        lead_tool_groups = agent_config.tool_groups
    elif not agent_config:
        # Use declared tool_groups from config.yaml as default filter for lead agent,
        # so tools in undeclared groups (e.g. ethoinsight:executor) are not visible.
        app_config = get_app_config()
        declared_groups = [g.name for g in app_config.tool_groups] if app_config.tool_groups else None
        lead_tool_groups = declared_groups if declared_groups else None

    all_lead_tools = get_available_tools(model_name=model_name, groups=lead_tool_groups, subagent_enabled=subagent_enabled)
    filtered_lead_tools = _filter_lead_tools(all_lead_tools, _LEAD_EXCLUDED_TOOLS)
    logger.info(
        "Lead tools after filtering: %d→%d (excluded: %s)",
        len(all_lead_tools),
        len(filtered_lead_tools),
        sorted(_LEAD_EXCLUDED_TOOLS),
    )

    from deerflow.tools.builtins.tool_search import assemble_deferred_tools
    final_tools, setup = assemble_deferred_tools(filtered_lead_tools, enabled=app_config.tool_search.enabled)

    return create_agent(
        model=create_chat_model(name=model_name, thinking_enabled=thinking_enabled, reasoning_effort=reasoning_effort, attach_tracing=False),
        tools=final_tools,
        middleware=build_middlewares(config, model_name=model_name, agent_name=agent_name, deferred_setup=setup),
        system_prompt=apply_prompt_template(
            subagent_enabled=subagent_enabled,
            max_concurrent_subagents=max_concurrent_subagents,
            agent_name=agent_name,
            available_skills=set(agent_config.skills) if agent_config and agent_config.skills is not None else None,
            paradigm=lead_paradigm,
            user_id=lead_user_id,
            thread_id=thread_id,
            deferred_names=setup.deferred_names,
        ),
        state_schema=ThreadState,
    )
