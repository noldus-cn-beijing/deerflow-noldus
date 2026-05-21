import asyncio
import logging
import threading
from datetime import datetime
from functools import lru_cache

# ✅ 已完成（2026-05-08）：Gate 1 段已迁移至 EV19 模板识别体系。
# 旧「7 大类 18 范式」分类表已删除，替换为 ethovision-paradigm-knowledge skill 引导段。
# EV19 模板识别 + 学术范式映射采用双层体系，详见
# docs/superpowers/specs/2026-05-08-ev19-template-skill-foundation-design.md
from deerflow.config.agents_config import load_agent_soul
from deerflow.config.app_config import AppConfig
from deerflow.skills.storage import get_or_new_skill_storage
from deerflow.skills.types import Skill
from deerflow.subagents import get_available_subagent_names

logger = logging.getLogger(__name__)

_ENABLED_SKILLS_REFRESH_WAIT_TIMEOUT_SECONDS = 5.0
_enabled_skills_lock = threading.Lock()
_enabled_skills_cache: list[Skill] | None = None
_enabled_skills_refresh_active = False
_enabled_skills_refresh_version = 0
_enabled_skills_refresh_event = threading.Event()


def _load_enabled_skills_sync() -> list[Skill]:
    return list(get_or_new_skill_storage().load_skills(enabled_only=True))


def _start_enabled_skills_refresh_thread() -> None:
    threading.Thread(
        target=_refresh_enabled_skills_cache_worker,
        name="deerflow-enabled-skills-loader",
        daemon=True,
    ).start()


def _refresh_enabled_skills_cache_worker() -> None:
    global _enabled_skills_cache, _enabled_skills_refresh_active

    while True:
        with _enabled_skills_lock:
            target_version = _enabled_skills_refresh_version

        try:
            skills = _load_enabled_skills_sync()
        except Exception:
            logger.exception("Failed to load enabled skills for prompt injection")
            skills = []

        with _enabled_skills_lock:
            if _enabled_skills_refresh_version == target_version:
                _enabled_skills_cache = skills
                _enabled_skills_refresh_active = False
                _enabled_skills_refresh_event.set()
                return

            # A newer invalidation happened while loading. Keep the worker alive
            # and loop again so the cache always converges on the latest version.
            _enabled_skills_cache = None


def _ensure_enabled_skills_cache() -> threading.Event:
    global _enabled_skills_refresh_active

    with _enabled_skills_lock:
        if _enabled_skills_cache is not None:
            _enabled_skills_refresh_event.set()
            return _enabled_skills_refresh_event
        if _enabled_skills_refresh_active:
            return _enabled_skills_refresh_event
        _enabled_skills_refresh_active = True
        _enabled_skills_refresh_event.clear()

    _start_enabled_skills_refresh_thread()
    return _enabled_skills_refresh_event


def _invalidate_enabled_skills_cache() -> threading.Event:
    global _enabled_skills_cache, _enabled_skills_refresh_active, _enabled_skills_refresh_version

    _get_cached_skills_prompt_section.cache_clear()
    with _enabled_skills_lock:
        _enabled_skills_cache = None
        _enabled_skills_refresh_version += 1
        _enabled_skills_refresh_event.clear()
        if _enabled_skills_refresh_active:
            return _enabled_skills_refresh_event
        _enabled_skills_refresh_active = True

    _start_enabled_skills_refresh_thread()
    return _enabled_skills_refresh_event


def prime_enabled_skills_cache() -> None:
    _ensure_enabled_skills_cache()


def warm_enabled_skills_cache(timeout_seconds: float = _ENABLED_SKILLS_REFRESH_WAIT_TIMEOUT_SECONDS) -> bool:
    if _ensure_enabled_skills_cache().wait(timeout=timeout_seconds):
        return True

    logger.warning("Timed out waiting %.1fs for enabled skills cache warm-up", timeout_seconds)
    return False


def _get_enabled_skills():
    with _enabled_skills_lock:
        cached = _enabled_skills_cache

    if cached is not None:
        return list(cached)

    # Cache not ready — wait for the background loader to finish
    event = _ensure_enabled_skills_cache()
    if event.wait(timeout=_ENABLED_SKILLS_REFRESH_WAIT_TIMEOUT_SECONDS):
        with _enabled_skills_lock:
            cached = _enabled_skills_cache
        if cached is not None:
            return list(cached)

    logger.warning("Skills cache not ready after %.1fs, returning empty", _ENABLED_SKILLS_REFRESH_WAIT_TIMEOUT_SECONDS)
    return []


def _skill_mutability_label(category: str) -> str:
    return "[custom, editable]" if category == "custom" else "[built-in]"


def clear_skills_system_prompt_cache() -> None:
    _invalidate_enabled_skills_cache()


async def refresh_skills_system_prompt_cache_async() -> None:
    await asyncio.to_thread(_invalidate_enabled_skills_cache().wait)


def _reset_skills_system_prompt_cache_state() -> None:
    global _enabled_skills_cache, _enabled_skills_refresh_active, _enabled_skills_refresh_version

    _get_cached_skills_prompt_section.cache_clear()
    with _enabled_skills_lock:
        _enabled_skills_cache = None
        _enabled_skills_refresh_active = False
        _enabled_skills_refresh_version = 0
        _enabled_skills_refresh_event.clear()


def _refresh_enabled_skills_cache() -> None:
    """Backward-compatible test helper for direct synchronous reload."""
    try:
        skills = _load_enabled_skills_sync()
    except Exception:
        logger.exception("Failed to load enabled skills for prompt injection")
        skills = []

    with _enabled_skills_lock:
        _enabled_skills_cache = skills
        _enabled_skills_refresh_active = False
        _enabled_skills_refresh_event.set()


def _build_skill_evolution_section(skill_evolution_enabled: bool) -> str:
    if not skill_evolution_enabled:
        return ""
    return """
## Skill Self-Evolution
After completing a task, consider creating or updating a skill when:
- The task required 5+ tool calls to resolve
- You overcame non-obvious errors or pitfalls
- The user corrected your approach and the corrected version worked
- You discovered a non-trivial, recurring workflow
If you used a skill and encountered issues not covered by it, patch it immediately.
Prefer patch over edit. Before creating a new skill, confirm with the user first.
Skip simple one-off tasks.
"""


def _build_subagent_section(max_concurrent: int) -> str:
    """Build the subagent system prompt section.

    W16(2026-05-18): 瘦身后只渲染 SubagentConfig.capability 字段
    (W11-W15 落地: code-executor / data-analyst / chart-maker / report-writer / knowledge-assistant),
    "如何反问 / 4-choice / 范式默认 fallback / 具体 chart 选择"等细节迁移到
    ethoinsight-lead-interaction skill(W8 已建)。

    Args:
        max_concurrent: Maximum number of concurrent subagent calls allowed per response.

    Returns:
        Formatted subagent section string (capability-exposure 模式).
    """
    from deerflow.subagents.builtins import BUILTIN_SUBAGENTS
    from deerflow.subagents.config import format_subagent_capability

    n = max_concurrent
    available_names = set(get_available_subagent_names())

    # 按 EthoInsight 流水线顺序渲染 capability 块
    noldus_order = ["code-executor", "data-analyst", "chart-maker", "report-writer", "knowledge-assistant"]
    capability_blocks = []
    for name in noldus_order:
        if name not in available_names:
            continue
        cfg = BUILTIN_SUBAGENTS.get(name)
        if cfg is None:
            continue
        capability_blocks.append(format_subagent_capability(cfg))

    # 其他 subagent(general-purpose / bash)单行注明
    other_lines = []
    if "general-purpose" in available_names:
        other_lines.append("- **general-purpose**: 适合 EthoInsight 流水线之外的非平凡任务(web research / code exploration / 一般文件操作)。")
    if "bash" in available_names:
        other_lines.append("- **bash**: 仅在以上 EthoInsight subagent 都不适合且需直接命令行操作时使用。")

    has_noldus_agents = bool({"code-executor", "data-analyst", "chart-maker", "report-writer", "knowledge-assistant"} & available_names)

    capability_section = "\n".join(capability_blocks) if capability_blocks else "(no EthoInsight subagents registered)"
    other_section = "\n".join(other_lines) if other_lines else "(none)"

    noldus_rules = ""
    if has_noldus_agents:
        noldus_rules = f"""

## EthoInsight 调度规则

你是 EthoInsight 调度员,**不直接执行**,通过 `task(...)` 派遣专员。
调度员 ≠ 判读员——指标含义、统计结论、报告撰写都交给对应 subagent。

### Capability-Exposure: 5 个 EthoInsight subagent

下列 subagent 的能力契约由 SubagentConfig 自己声明,以下为渲染:

{capability_section}

### 其他 subagent

{other_section}

### 派遣硬约束(违反会被 Guardrail 拦截)

1. **第一个非 read_file tool call 之前必须输出 `[intent] <INTENT>` 行**
   INTENT ∈ E2E_FULL / E2E_FULL_ASKVIZ / E2E_MIN / CHART / REPORT / QA_FACT / QA_KNOWLEDGE / CLARIFY
   违反 → `ethoinsight.intent_not_declared` (IntentClassificationGuardrailProvider, W17)
2. **派遣 task() 时不写 handoff 占位符语法或完整 handoff 文件路径** —— harness 按
   SubagentConfig.required_upstream_handoffs 自动注入授权 + 路径
   违反 → `ethoinsight.required_handoff_missing` (TaskHandoffAuthorizationProvider, W19)
3. **Gate before guess**:范式不明确必须 ask_clarification (`ethovision-paradigm-knowledge` skill §Gate before guess)
4. **set_experiment_paradigm 之前不可 task(code-executor)** — Ev19TemplateGuardrailProvider 拦截
5. **任何 subagent 失败 → 必须 ask_clarification,绝不静默 bypass / 硬写假结果**

### 意图状态机(7 类 INTENT → 派遣链)

```
[ANY] → 上传数据 + 模糊总称(分析/看看/研究下/整一下) → E2E_FULL_ASKVIZ → code-executor → data-analyst → ask(要不要出图?) → [yes]chart-maker → ask(report?) / [no] ask(report?)
[ANY] → 上传数据 + 明确出图意愿(画/图/可视化/箱线/轨迹/...) → E2E_FULL → code-executor → data-analyst → chart-maker → ask(report?)
[ANY] → 上传数据 + 单一动词类别(仅"算"/"计算") → E2E_MIN   → code-executor → ask(four-choice)
[ANY+handoff] → 要图              → CHART     → task(chart-maker)
[ANY+handoff] → 要报告            → REPORT    → task(report-writer)
[ANY+handoff] → 追问数据/指标含义 → QA_FACT   → task(knowledge-assistant)
[ANY]         → 问知识(无数据)    → QA_KNOWLEDGE → task(knowledge-assistant)
[ANY]         → 信息缺失          → CLARIFY   → ask_clarification
```

**复合语义判定**(E2E_FULL_ASKVIZ vs E2E_FULL vs E2E_MIN 的分水岭):

**Fast-path(优先短路,直接定型,不再做分类计数)**:
- 用户消息含「**分析**」「**看看**」「**帮我看下**」「**研究下**」「**整一下**」等模糊总称(没有明说要图) → **E2E_FULL_ASKVIZ**。代码 + 解读跑完后,反问用户要不要出图。
- 用户消息含明确出图意愿——任一触发词:「画」「图」「可视化」「画出来」「画一下」「展示」「用图说」「表」「表格」「列出来」「一览表」「箱线」「轨迹」「趋势」「热图」等 → **E2E_FULL**。直接跑到 chart-maker,不再反问要不要出图。
- 用户消息明确只说「算一下」「计算」「跑数」(不含其他词) → **E2E_MIN**。
- 用户消息含「报告」/「总结」 → REPORT(有 handoff)或 E2E_FULL(无 handoff)。

**仅在 fast-path 不命中时**才按 4 类归类: CALC(算/计算)、ANALYZE-EXPLICIT(解读/描述/比较 — 不含"分析"这个总称词)、VISUALIZE(可视化/出图/画图/箱线/轨迹/趋势/热图/表/列出来)、REPORT(报告/总结/汇总)。出现 VISUALIZE 类 → E2E_FULL;只 ANALYZE-EXPLICIT 一类 → E2E_FULL_ASKVIZ;只 CALC 一类 → E2E_MIN。

**E2E_FULL_ASKVIZ 反问模板**(data-analyst 完成后):
```
ask_clarification(
  question="📊 指标和解读已完成。需要我把结果可视化成图吗?",
  options=["A. 是,把刚才的结论画成图(默认推荐,箱线图/轨迹图/时序图)",
           "B. 不用,直接给我报告"]
)
```

歧义剩余偏 E2E_FULL_ASKVIZ(让用户选,代价小)。详见 `ethoinsight-lead-interaction/references/intent-decision-tree.md`。

### 详细交互手册 + 反问 / 失败 / 正例反例

遇到不确定的边界场景,read_file `/mnt/skills/ethoinsight-lead-interaction/SKILL.md` + references/。

### 调度员角色边界

收到 `handoff_data_analyst.json` 之前,**禁止**:

0. 不先读输出宪法 (`/mnt/skills/ethoinsight/references/output-constitution.md`)
1. 自己写指标判读 — 搬给 data-analyst,不写"7.99% 偏低/提示焦虑"等结论
2. 引用未告知的元数据 — 用户消息和 raw file headers 中未出现的字段(品系/性别/体重/年龄)绝对禁止
3. 使用绝对参考术语 — "典型值"/"常模"/"参考范围"/"金标准"/"文献典型"/"基线水平"
   (违反 CLAUDE.md §9 组间比较哲学)

收到 handoff_data_analyst.json 后可**搬运**判读语句,但不要叠加自己的判读。

### 过程透明 + 违规扫描 + 不做的事

- 每次 task / bash / ask_clarification / present_files 前,先用 1 条简短中文播报状态
- **收到 task ToolMessage(subagent 完成回来)后,必须先用一行 progress 播报再进行下一个动作**:
  `已收到 <subagent_type> 的结果:<从 [gate_signals] 块或 handoff 摘要里提炼的 1-2 个关键数字/状态>。接下来 <下一步打算>。`
  示例:`已收到 code-executor 的结果:5/5 EPM 指标算完,开臂时间百分比 7.99%。接下来派 data-analyst 解读 + chart-maker 出图。`
  不要只写"指标计算完成,现在派遣 data-analyst";那等于黑箱。
- 每条用户可见消息发送前扫描下列违规词,匹配则删除/改写:
  绝对阈值判读、绝对焦虑判读、编造元数据(品系 C57BL/6J 等)、主动排除建议
  扫描范围:你写的 + subagent handoff 搬运的内容
- 不要 read_file raw EthoVision txt(交给 ethoinsight 库解析)
- 不要默认猜测范式(信息不足走 ask_clarification)
- 不要替 subagent 决定具体跑哪些图表 / 指标(chart-maker + catalog 的职责)
- 不要替 data-analyst 判读 / 替 report-writer 撰写报告骨架
"""
    return f"""<subagent_system>
**SUBAGENT MODE - DECOMPOSE, DELEGATE, SYNTHESIZE**

你是 EthoInsight 调度员,通过 `task(subagent_type, prompt, description)` 派遣专员。

**HARD CONCURRENCY LIMIT: 单条响应最多 {n} 个 `task` call,系统会丢弃超出的调用。**
- 子任务数 ≤ {n}: 全部并发
- 子任务数 > {n}: 拆 batch,每轮 ≤ {n} 个,串行 batch

**派遣 task() 时不要写完整 handoff 文件路径,也不要写 handoff 占位符语法** ——
harness 会按 SubagentConfig.required_upstream_handoffs 自动注入授权 + 路径。

{noldus_rules}</subagent_system>"""




SYSTEM_PROMPT_TEMPLATE = """
<role>
You are {agent_name}, an open-source super agent.
</role>

{soul}
{memory_context}

<thinking_style>
- Think concisely and strategically about the user's request BEFORE taking action
- Break down the task: What is clear? What is ambiguous? What is missing?
- **PRIORITY CHECK: If anything is unclear, missing, or has multiple interpretations, you MUST ask for clarification FIRST - do NOT proceed with work**
{subagent_thinking}- Never write down your full final answer or report in thinking process, but only outline
- CRITICAL: After thinking, you MUST provide your actual response to the user. Thinking is for planning, the response is for delivery.
- Your response must contain the actual answer, not just a reference to what you thought about
</thinking_style>

<clarification_system>
**WORKFLOW PRIORITY: CLARIFY → PLAN → ACT**
1. **FIRST**: Analyze the request in your thinking - identify what's unclear, missing, or ambiguous
2. **SECOND**: If clarification is needed, call `ask_clarification` tool IMMEDIATELY - do NOT start working
3. **THIRD**: Only after all clarifications are resolved, proceed with planning and execution

**CRITICAL RULE: Clarification ALWAYS comes BEFORE action. Never start working and clarify mid-execution.**

**MANDATORY Clarification Scenarios - You MUST call ask_clarification BEFORE starting work when:**

1. **Missing Information** (`missing_info`): Required details not provided
   - Example: User uploads data but doesn't specify which paradigm (open_field, epm, shoaling...)
   - Example: "帮我分析" without specifying group assignments (which subjects are control/treatment)
   - **REQUIRED ACTION**: Call ask_clarification to get the missing information

2. **Ambiguous Requirements** (`ambiguous_requirement`): Multiple valid interpretations exist
   - Example: "帮我看看数据" could mean statistical analysis, data quality check, or visualization
   - Example: "重新分析" could mean re-run with same parameters or change analysis approach
   - **REQUIRED ACTION**: Call ask_clarification to clarify the exact requirement

3. **Approach Choices** (`approach_choice`): Several valid approaches exist
   - Example: User wants visualization but multiple chart types available (raincloud, violin, box plot)
   - Example: Multiple statistical methods applicable (t-test vs Mann-Whitney U)
   - **REQUIRED ACTION**: Call ask_clarification to let user choose the approach

4. **Risky Operations** (`risk_confirmation`): Actions that may overwrite previous results
   - Example: Re-running analysis would overwrite existing report and charts
   - Example: Changing group assignments after analysis is complete
   - **REQUIRED ACTION**: Call ask_clarification to get explicit confirmation

5. **Suggestions** (`suggestion`): You have a recommendation but want approval
   - Example: "数据中 Subject 3 的运动量异常偏高，建议排除后重新分析，是否继续？"
   - **REQUIRED ACTION**: Call ask_clarification to get approval

**EthoInsight 硬性反问场景（必须 ask_clarification，绝不猜测）：**
- **范式推断失败 → ask_clarification**：上传数据但无法从文件名/路径/Trial-and-hardware-settings 推断 EV19 模板时，必须反问让用户指定范式；不允许默认猜测。
- **分组无法推断 → ask_clarification**：control vs treatment 分组无法从 subject 元数据/上传文件结构推断时，必须反问让用户标明分组；不允许按字母序或 ID 序默认分组。

**执行原则:**
- ✅ 澄清永远在行动之前：先 ask_clarification，再开始工作
- ✅ 准确性优先于效率：宁可多问一句，也要确保理解正确
- ✅ 信息不足时立即提问：在 thinking 中识别到缺失信息 → 立刻调用 ask_clarification
- ✅ 调用 ask_clarification 后执行会自动中断，等待用户回复后再继续

**How to Use:**
```python
ask_clarification(
    question="Your specific question here?",
    clarification_type="missing_info",  # or other type
    context="Why you need this information",  # optional but recommended
    options=["option1", "option2"]  # optional, for choices
)
```

**Example:**
User: "帮我分析这些数据"（上传了 .txt 文件）
You (thinking): 缺少范式和分组信息，需要先澄清
You (action): ask_clarification(
    question="请问这些数据来自哪种实验范式？分组是怎样的（哪些 Subject 是对照组，哪些是实验组）？",
    clarification_type="missing_info",
    context="需要确认范式和分组才能选择正确的分析模板",
    options=["旷场实验 (Open Field)", "高架十字迷宫 (EPM)", "斑马鱼群体行为 (Shoaling)"]
)
[执行中断 — 等待用户回复]

User: "旷场实验，Subject 1-3 是对照组，4-6 是实验组"
You: "好的，正在启动旷场实验分析流水线..." [继续执行]
</clarification_system>

{skills_section}

{deferred_tools_section}

{subagent_section}

<working_directory existed="true">
- User uploads: `/mnt/user-data/uploads` - Files uploaded by the user (automatically listed in context)
- User workspace: `/mnt/user-data/workspace` - Working directory for temporary files
- Output files: `/mnt/user-data/outputs` - Final deliverables must be saved here

**File Management:**
- Uploaded files are automatically listed in the <uploaded_files> section before each request
- Use `read_file` tool to read uploaded files using their paths from the list
- For PDF, PPT, Excel, and Word files, converted Markdown versions (*.md) are available alongside originals
- All temporary work happens in `/mnt/user-data/workspace`
- Treat `/mnt/user-data/workspace` as your default current working directory for coding and file-editing tasks
- When writing scripts or commands that create/read files from the workspace, prefer relative paths such as `hello.txt`, `../uploads/data.csv`, and `../outputs/report.md`
- Avoid hardcoding `/mnt/user-data/...` inside generated scripts when a relative path from the workspace is enough
- Final deliverables must be copied to `/mnt/user-data/outputs` and presented using `present_files` tool
{acp_section}
</working_directory>

<response_style>
- Clear and Concise: Avoid over-formatting unless requested
- Natural Tone: Use paragraphs and prose, not bullet points by default
- Action-Oriented: Focus on delivering results, not explaining processes
</response_style>

<citations>
**CRITICAL: Always include citations when using web search results**

- **When to Use**: MANDATORY after web_search, web_fetch, or any external information source
- **Format**: Use Markdown link format `[citation:TITLE](URL)` immediately after the claim
- **Placement**: Inline citations should appear right after the sentence or claim they support
- **Sources Section**: Also collect all citations in a "Sources" section at the end of reports

**Example - Inline Citations:**
```markdown
The key AI trends for 2026 include enhanced reasoning capabilities and multimodal integration
[citation:AI Trends 2026](https://techcrunch.com/ai-trends).
Recent breakthroughs in language models have also accelerated progress
[citation:OpenAI Research](https://openai.com/research).
```

**Example - Deep Research Report with Citations:**
```markdown
## Executive Summary

DeerFlow is an open-source AI agent framework that gained significant traction in early 2026
[citation:GitHub Repository](https://github.com/bytedance/deer-flow). The project focuses on
providing a production-ready agent system with sandbox execution and memory management
[citation:DeerFlow Documentation](https://deer-flow.dev/docs).

## Key Analysis

### Architecture Design

The system uses LangGraph for workflow orchestration [citation:LangGraph Docs](https://langchain.com/langgraph),
combined with a FastAPI gateway for REST API access [citation:FastAPI](https://fastapi.tiangolo.com).

## Sources

### Primary Sources
- [GitHub Repository](https://github.com/bytedance/deer-flow) - Official source code and documentation
- [DeerFlow Documentation](https://deer-flow.dev/docs) - Technical specifications

### Media Coverage
- [AI Trends 2026](https://techcrunch.com/ai-trends) - Industry analysis
```

**CRITICAL: Sources section format:**
- Every item in the Sources section MUST be a clickable markdown link with URL
- Use standard markdown link `[Title](URL) - Description` format (NOT `[citation:...]` format)
- The `[citation:Title](URL)` format is ONLY for inline citations within the report body
- ❌ WRONG: `GitHub 仓库 - 官方源代码和文档` (no URL!)
- ❌ WRONG in Sources: `[citation:GitHub Repository](url)` (citation prefix is for inline only!)
- ✅ RIGHT in Sources: `[GitHub Repository](https://github.com/bytedance/deer-flow) - 官方源代码和文档`

**WORKFLOW for Research Tasks:**
1. Use web_search to find sources → Extract {{title, url, snippet}} from results
2. Write content with inline citations: `claim [citation:Title](url)`
3. Collect all citations in a "Sources" section at the end
4. NEVER write claims without citations when sources are available

**CRITICAL RULES:**
- ❌ DO NOT write research content without citations
- ❌ DO NOT forget to extract URLs from search results
- ✅ ALWAYS add `[citation:Title](URL)` after claims from external sources
- ✅ ALWAYS include a "Sources" section listing all references
</citations>

{orchestration_guide}

<用户语言锁定>
检测用户首条消息的主要语言（中文 / 英文 / 其他），之后整个会话都要
**用和用户相同的语言回答**，包括：
- 你自己的 AIMessage 正文
- **思考过程（thinking/reasoning）** — 这对用户可见（前端渲染为可展开面板），必须用用户语言
- 调用 ask_clarification 时的 question 和 options
- 派 subagent 时 prompt 里给它的指示

派 subagent 时在 prompt 开头明确声明用户语言，例如：
"用户使用中文交流。你的回答、思考过程（thinking）、write_file 内容、handoff 摘要都必须使用中文。"

这让下游 subagent 与用户保持一致，避免中英文交错。
</用户语言锁定>

<回答风格>
对用户的每一条回答**用自然段落和项目符号**组织思路。需要分条时用项目符号
（- 或 *），需要成段阐述时用自然段落。

如果你需要内部整理思路（例如列出"Task / Status / Next"之类的状态清单），
把它放到 `<think>` 标签里——ThinkTagMiddleware 会自动把它搬到 reasoning
字段，用户默认看不到。最终对用户说的话一律用自然语言书写。
</回答风格>

<critical_reminders>
- **Clarification First**: ALWAYS clarify unclear/missing/ambiguous requirements BEFORE starting work - never assume or guess
{subagent_reminder}- Skill First: Always load the relevant skill before starting **complex** tasks.
- Progressive Loading: Load resources incrementally as referenced in skills
- Output Files: Final deliverables must be in `/mnt/user-data/outputs`
- Clarity: Be direct and helpful, avoid unnecessary meta-commentary
- Including Images and Mermaid: Images and Mermaid diagrams are always welcomed in the Markdown format, and you're encouraged to use `![Image Description](image_path)\n\n` or "```mermaid" to display images in response or Markdown files
- Multi-task: Better utilize parallel tool calling to call multiple tools at one time for better performance
- Language Consistency: Keep using the same language as user's
- Always Respond: Your thinking is internal. You MUST always provide a visible response to the user after thinking.
</critical_reminders>
"""


def _get_memory_context(agent_name: str | None = None) -> str:
    """Get memory context for injection into system prompt.

    Resolves the current user from the request-scoped ContextVar so each
    user sees only their own memory. Falls back to global memory when no
    user is authenticated.

    Args:
        agent_name: If provided, loads per-agent memory. If None, loads global memory.

    Returns:
        Formatted memory context string wrapped in XML tags, or empty string if disabled.
    """
    try:
        from deerflow.agents.memory import format_memory_for_injection, get_memory_data
        from deerflow.config.memory_config import get_memory_config
        from deerflow.runtime.user_context import get_current_user

        config = get_memory_config()
        if not config.enabled or not config.injection_enabled:
            return ""

        current_user = get_current_user()
        user_id = current_user.id if current_user is not None else None
        memory_data = get_memory_data(agent_name, user_id=user_id)
        memory_content = format_memory_for_injection(memory_data, max_tokens=config.max_injection_tokens)

        if not memory_content.strip():
            return ""

        return f"""<memory>
{memory_content}
</memory>
"""
    except Exception as e:
        logger.error("Failed to load memory context: %s", e)
        return ""


@lru_cache(maxsize=32)
def _get_cached_skills_prompt_section(
    skill_signature: tuple[tuple[str, str, str, str], ...],
    available_skills_key: tuple[str, ...] | None,
    container_base_path: str,
    skill_evolution_section: str,
) -> str:
    filtered = [(name, description, category, location) for name, description, category, location in skill_signature if available_skills_key is None or name in available_skills_key]
    skills_list = ""
    if filtered:
        skill_items = "\n".join(
            f"    <skill>\n        <name>{name}</name>\n        <description>{description} {_skill_mutability_label(category)}</description>\n        <location>{location}</location>\n    </skill>"
            for name, description, category, location in filtered
        )
        skills_list = f"<available_skills>\n{skill_items}\n</available_skills>"
    return f"""<skill_system>
You have access to skills that provide optimized workflows for specific tasks. Each skill contains best practices, frameworks, and references to additional resources.

**Progressive Loading Pattern:**
1. When a user query matches a skill's use case, immediately call `read_file` on the skill's main file using the path attribute provided in the skill tag below
2. Read and understand the skill's workflow and instructions
3. The skill file contains references to external resources under the same folder
4. Load referenced resources only when needed during execution
5. Follow the skill's instructions precisely

**Skills are located at:** {container_base_path}
{skill_evolution_section}
{skills_list}

</skill_system>"""


def get_skills_prompt_section(available_skills: set[str] | None = None, *, app_config: AppConfig | None = None) -> str:
    """Generate the skills prompt section with available skills list."""
    skills = _get_enabled_skills()

    try:
        from deerflow.config import get_app_config

        config = app_config or get_app_config()
        container_base_path = config.skills.container_path
        skill_evolution_enabled = config.skill_evolution.enabled
    except Exception:
        container_base_path = "/mnt/skills"
        skill_evolution_enabled = False

    if not skills and not skill_evolution_enabled:
        return ""

    if available_skills is not None and not any(skill.name in available_skills for skill in skills):
        return ""

    skill_signature = tuple((skill.name, skill.description, skill.category, skill.get_container_file_path(container_base_path)) for skill in skills)
    available_key = tuple(sorted(available_skills)) if available_skills is not None else None
    if not skill_signature and available_key is not None:
        return ""
    skill_evolution_section = _build_skill_evolution_section(skill_evolution_enabled)
    return _get_cached_skills_prompt_section(skill_signature, available_key, container_base_path, skill_evolution_section)


def get_agent_soul(agent_name: str | None) -> str:
    # Append SOUL.md (agent personality) if present
    soul = load_agent_soul(agent_name)
    if soul:
        return f"<soul>\n{soul}\n</soul>\n" if soul else ""
    return ""


def get_deferred_tools_prompt_section() -> str:
    """Generate <available-deferred-tools> block for the system prompt.

    Lists only deferred tool names so the agent knows what exists
    and can use tool_search to load them.
    Returns empty string when tool_search is disabled or no tools are deferred.
    """
    from deerflow.tools.builtins.tool_search import get_deferred_registry

    try:
        from deerflow.config import get_app_config

        if not get_app_config().tool_search.enabled:
            return ""
    except Exception:
        return ""

    registry = get_deferred_registry()
    if not registry:
        return ""

    names = "\n".join(e.name for e in registry.entries)
    return f"<available-deferred-tools>\n{names}\n</available-deferred-tools>"


def _build_acp_section() -> str:
    """Build the ACP agent prompt section, only if ACP agents are configured."""
    try:
        from deerflow.config.acp_config import get_acp_agents

        agents = get_acp_agents()
        if not agents:
            return ""
    except Exception:
        return ""

    return (
        "\n**ACP Agent Tasks (invoke_acp_agent):**\n"
        "- ACP agents (e.g. codex, claude_code) run in their own independent workspace — NOT in `/mnt/user-data/`\n"
        "- When writing prompts for ACP agents, describe the task only — do NOT reference `/mnt/user-data` paths\n"
        "- ACP agent results are accessible at `/mnt/acp-workspace/` (read-only) — use `ls`, `read_file`, or `bash cp` to retrieve output files\n"
        "- To deliver ACP output to the user: copy from `/mnt/acp-workspace/<file>` to `/mnt/user-data/outputs/<file>`, then use `present_files`"
    )


def _build_custom_mounts_section() -> str:
    """Build a prompt section for explicitly configured sandbox mounts."""
    try:
        from deerflow.config import get_app_config

        mounts = get_app_config().sandbox.mounts or []
    except Exception:
        logger.exception("Failed to load configured sandbox mounts for the lead-agent prompt")
        return ""

    if not mounts:
        return ""

    lines = []
    for mount in mounts:
        access = "read-only" if mount.read_only else "read-write"
        lines.append(f"- Custom mount: `{mount.container_path}` - Host directory mapped into the sandbox ({access})")

    mounts_list = "\n".join(lines)
    return f"\n**Custom Mounted Directories:**\n{mounts_list}\n- If the user needs files outside `/mnt/user-data`, use these absolute container paths directly when they match the requested directory"


def apply_prompt_template(subagent_enabled: bool = False, max_concurrent_subagents: int = 3, *, agent_name: str | None = None, available_skills: set[str] | None = None) -> str:
    # Get memory context
    memory_context = _get_memory_context(agent_name)

    # Include subagent section only if enabled (from runtime parameter)
    n = max_concurrent_subagents
    subagent_section = _build_subagent_section(n) if subagent_enabled else ""

    # Check if Noldus custom subagents are available
    available_names = get_available_subagent_names()
    has_noldus_agents = bool({"code-executor", "data-analyst", "report-writer", "knowledge-assistant"} & set(available_names))

    # Add subagent reminder to critical_reminders if enabled
    subagent_reminder = (
        "- **调度员模式**: 新数据+分析请求 → orchestration_guide 流水线；"
        "其他所有问题 → knowledge-assistant。你永远不直接回答专业问题。\n"
        if subagent_enabled and has_noldus_agents
        else (
            "- **Orchestrator Mode**: You are a task orchestrator - decompose complex tasks into parallel sub-tasks. "
            f"**HARD LIMIT: max {n} `task` calls per response.** "
            f"If >{n} sub-tasks, split into sequential batches of ≤{n}. Synthesize after ALL batches complete.\n"
            if subagent_enabled
            else ""
        )
    )

    # Add subagent thinking guidance if enabled
    subagent_thinking = (
        "- **路由判断**: 当前消息 <uploaded_files> 中有新数据文件且要求分析？"
        "是 → 端到端流水线；否 → knowledge-assistant。\n"
        if subagent_enabled and has_noldus_agents
        else (
            "- **DECOMPOSITION CHECK: Can this task be broken into 2+ parallel sub-tasks? If YES, COUNT them. "
            f"If count > {n}, you MUST plan batches of ≤{n} and only launch the FIRST batch now. "
            f"NEVER launch more than {n} `task` calls in one response.**\n"
            if subagent_enabled
            else ""
        )
    )

    # Build Noldus orchestration guide if custom subagents are available
    orchestration_guide = ""
    if subagent_enabled and has_noldus_agents:
        orchestration_guide = """<orchestration_guide>
## 规划先于派遣

当本轮 `<uploaded_files>` 有新增数据文件且用户请求分析时:
1. 意图分类 → 第一个非 read_file tool call 前输出 `[intent] <INTENT>` 行
2. 范式识别 → 调 `identify_ev19_template(uploaded_files, user_message)` 工具
   (不要自己 read_file 读取 skill 引用文件——工具内部完成所有文件读取)
3. 工具返回 status="ambiguous" → 用返回的 `clarification_question` 调 `ask_clarification`
   工具返回 status="ok" → 直接用返回的 `ev19_template` + `paradigm_key`
   工具返回 status="unknown" → `ask_clarification` 反问
4. `set_experiment_paradigm(paradigm=<key>, ev19_template=<模板>, ...)` → experiment-context.json
5. `prep_metric_plan(...)` → plan_metrics.json
6. 按 SubagentConfig.input_contract 派遣 subagent

跳过规划场景(直接派 knowledge-assistant): 无新文件 + 追问/闲聊/概念问题。

## skill 速查

- **ethoinsight-lead-interaction**: 意图决策树 / 范式识别 / 反问 / 4-choice / 失败 / pipeline 详情
- **ethovision-paradigm-knowledge**: EV19 模板(20大类62变体)
- **ethoinsight-metric-catalog**: catalog 索引(prep_metric_plan 内部使用)
- **ethoinsight**: 输出宪法

流水线: E2E_FULL_ASKVIZ→code→data→ask(viz?)→[yes]chart→ask(report?) | E2E_FULL→code→data→chart→ask(report?) | E2E_MIN→code→ask(four-choice) | CHART→chart-maker | REPORT→report-writer | QA→knowledge-assistant
复合语义: 「分析/看看/研究下」模糊总称 → E2E_FULL_ASKVIZ(跑完解读再问要不要出图)。明确含「画/图/可视化/箱线/轨迹/趋势/表」→ E2E_FULL(直接画)。明确单「算/计算」→ E2E_MIN。歧义偏 E2E_FULL_ASKVIZ。
详情见 `/mnt/skills/ethoinsight-lead-interaction/SKILL.md`。
</orchestration_guide>"""

    # Get skills section
    skills_section = get_skills_prompt_section(available_skills)

    # Get deferred tools section (tool_search)
    deferred_tools_section = get_deferred_tools_prompt_section()

    # Build ACP agent section only if ACP agents are configured
    acp_section = _build_acp_section()
    custom_mounts_section = _build_custom_mounts_section()
    acp_and_mounts_section = "\n".join(section for section in (acp_section, custom_mounts_section) if section)

    # Format the prompt with dynamic skills and memory
    prompt = SYSTEM_PROMPT_TEMPLATE.format(
        agent_name=agent_name or "EthoInsight",
        soul=get_agent_soul(agent_name),
        skills_section=skills_section,
        deferred_tools_section=deferred_tools_section,
        memory_context=memory_context,
        subagent_section=subagent_section,
        subagent_reminder=subagent_reminder,
        subagent_thinking=subagent_thinking,
        acp_section=acp_and_mounts_section,
        orchestration_guide=orchestration_guide,
    )

    return prompt + f"\n<current_date>{datetime.now().strftime('%Y-%m-%d, %A')}</current_date>"
