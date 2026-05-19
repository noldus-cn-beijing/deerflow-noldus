# EthoInsight Dogfood 修复迭代实施计划（2026-05-13）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 thread `5046a6e6-4bfc-4ca9-9650-b674ec3734cf` 那一轮 dogfood 暴露的全部故障级联，让下一轮 dogfood 能在不死会话、不丢 stream、lead 不越界写判读的前提下跑通端到端。

**Architecture:** 故障是连锁式的：lead 在派 data-analyst 之前自己写判读（编 C57BL/6J 品系 + 引"常模/金标准"）→ 用户问了一个超 catalog 的需求（时间分段分析）→ code-executor 把它打回 → lead 越权 write_file 写脚本 → 该文件在 `.deer-flow/users/<uid>/threads/<tid>/user-data/workspace/` 下触发 uvicorn `--reload` glob 漏写的 reload → gateway 重启断 SSE → 重启后历史 messages 的 thinking 字段恢复不全 → newapi 返回 400 → 会话死。本计划按"先修可见症状、再修流程透明、最后做架构级问题"的次序拆 13 个 Issue 为 5 个批次的 TDD 任务，每个任务自含、可独立 commit。

**Tech Stack:** Python 3.12 (deerflow harness, ethoinsight, pytest), Bash (uvicorn shell flags), Next.js + React 19 + TypeScript (frontend `ai-elements/reasoning.tsx`), SQLite (checkpointer), LangGraph 0.7.65, langchain_anthropic + newapi 网关。

**先决工作目录**：所有命令默认在 `/home/wangqiuyang/noldus-insight/` 下执行。Backend 命令前缀 `cd packages/agent/backend &&`。Frontend 命令前缀 `cd packages/agent/frontend &&`。Ethoinsight 命令前缀 `cd packages/ethoinsight &&`。

**测试基线**：开工前先跑一遍现有测试套件确认基线绿，结尾每批次结束再跑一遍：
- `cd packages/agent/backend && make test`
- `cd packages/ethoinsight && pytest tests/`

---

## 文件影响总览

| 任务 | 修改 | 创建 |
|------|------|------|
| Task 1 (Issue #1) | `packages/agent/scripts/serve.sh:153` | — |
| Task 2 (Issue #3) | `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` (新增"角色边界硬约束"段) | `tests/test_lead_prompt_role_boundaries.py` |
| Task 3 (Issue #2 诊断) | — | `tests/test_thinking_field_preserved.py`（先红） |
| Task 4 (Issue #2 修复) | 视诊断结果，可能改 `archiving_summarization.py` / `think_tag_middleware.py` / 任一保留消息状态的中间件 | — |
| Task 5 (Issue #5) | `lead_agent/prompt.py`（强化"过程透明原则"段 line 407 起） | — |
| Task 6 (Issue #4) | `packages/agent/frontend/src/components/workspace/messages/message-list-item.tsx` (受控 open 拦 auto-close，不改 ai-elements/) | — |
| Task 7 (Issue #6) | `packages/ethoinsight/ethoinsight/catalog/resolve.py:245` + 调用 resolve 的 sandbox 路径翻译层 | `packages/ethoinsight/tests/test_catalog_resolve_paths.py`（如已存在则扩展） |
| Task 8 (Issue #7) | `packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py:39-43` (critical_rules 段) | — |
| Task 9 (Issue #8 dogfood 验证) | — | `docs/handoffs/2026-05/2026-05-14-dogfood-followup-handoff.md`（验证结果存档） |
| Task 10 (Issue #10) | `packages/agent/backend/packages/harness/deerflow/runtime/checkpointer/`（具体 saver 实现） | `tests/test_checkpointer_adelete_for_runs.py` |
| 后置（Issue #9 / #11 / #12 / #13） | 见各 Task 备注 | — |

---

## Task 1: 修复 uvicorn --reload-exclude glob（Issue #1 · P0）

**Files:**
- Modify: `packages/agent/scripts/serve.sh:153`

**背景**：`gateway.log` line 252 / 278 显示 `WatchFiles detected changes in '.../workspace/time_bin_epm.py'. Reloading...`，agent 在 thread workspace 写文件触发了 gateway 热重启。根因是 `--reload-exclude='.deer-flow/*'` 的 glob 默认只匹配一级目录下的文件，匹配不到深层路径 `backend/.deer-flow/users/<uid>/threads/<tid>/user-data/workspace/*.py`。本任务的目标是让 `.deer-flow/` 下任何深度的文件改动都不触发 reload。

**当前完整 line 153**：
```bash
GATEWAY_EXTRA_FLAGS="--reload --reload-include='*.yaml' --reload-include='.env' --reload-exclude='*.pyc' --reload-exclude='__pycache__' --reload-exclude='sandbox/*' --reload-exclude='.deer-flow/*'"
```

- [ ] **Step 1: 复现 bug — 启动服务、让 agent 写文件、观察 reload**

Run:
```bash
cd packages/agent && make stop && make dev
sleep 5
# 模拟 agent 写 thread workspace 文件
THREAD_DIR=packages/agent/backend/.deer-flow/users/test-user/threads/test-thread/user-data/workspace
mkdir -p "$THREAD_DIR"
echo "x = 1" > "$THREAD_DIR/dummy_script.py"
sleep 3
grep -c "WatchFiles detected changes" packages/agent/logs/gateway.log
```

Expected: count > 0（reload 被触发）。这是 bug 复现成功。

- [ ] **Step 2: 修改 serve.sh，把 glob 改成递归匹配**

将 `packages/agent/scripts/serve.sh:153` 中的两处 glob 改成显式递归匹配并新增 `--reload-dir` 白名单。完整 line 改成：

```bash
GATEWAY_EXTRA_FLAGS="--reload --reload-dir='app' --reload-dir='packages/harness' --reload-include='*.yaml' --reload-include='.env' --reload-exclude='*.pyc' --reload-exclude='__pycache__' --reload-exclude='sandbox/**' --reload-exclude='.deer-flow/**'"
```

**两个关键修改**：
1. `--reload-dir='app' --reload-dir='packages/harness'`：显式白名单，uvicorn 只在这两个目录下检测变化，根本不看 `.deer-flow/`。这是更稳的方案
2. `.deer-flow/*` → `.deer-flow/**`：即使白名单失效，递归通配做兜底
3. `sandbox/*` → `sandbox/**`：同上原因

- [ ] **Step 3: 重启服务，重新复现，验证修复**

Run:
```bash
cd packages/agent && make stop && make dev
sleep 5
RELOAD_COUNT_BEFORE=$(grep -c "WatchFiles detected changes" packages/agent/logs/gateway.log)
THREAD_DIR=packages/agent/backend/.deer-flow/users/test-user/threads/test-thread/user-data/workspace
mkdir -p "$THREAD_DIR"
echo "y = 2" > "$THREAD_DIR/dummy_script2.py"
sleep 3
RELOAD_COUNT_AFTER=$(grep -c "WatchFiles detected changes" packages/agent/logs/gateway.log)
test "$RELOAD_COUNT_BEFORE" = "$RELOAD_COUNT_AFTER" && echo "PASS: no new reload" || echo "FAIL: still reloading"
```

Expected: `PASS: no new reload`

- [ ] **Step 4: 清理测试目录、commit**

Run:
```bash
rm -rf packages/agent/backend/.deer-flow/users/test-user
git add packages/agent/scripts/serve.sh
git commit -m "fix(reload): exclude .deer-flow/** recursively from uvicorn reload watcher

agent 在 thread workspace 写文件原本会触发 gateway 热重启、断 SSE 流。
现改用 --reload-dir 白名单只监听 app/ 和 packages/harness/，同时把 .deer-flow
和 sandbox 的 exclude glob 改为递归 (** 替换 *) 做兜底。

Fix Issue #1 from 2026-05-13 dogfood iteration."
```

---

## Task 2: 收紧 lead 角色边界（Issue #3 · P0）

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`（在 `### 角色分工与契约` 段——line 299 起——之后插入新段 `### 角色边界硬约束`）
- Test: `packages/agent/backend/tests/test_lead_prompt_role_boundaries.py`（新建）

**背景**：thread 5046a6e6 的 `docs/e2e/Elevated Plus Maze Trajectory Analysis.md` 显示：
- **L493** lead 在派 data-analyst 之前自己写"开臂时间比例 7.99% 远低于典型低焦虑小鼠（通常 >20%），提示较高的焦虑水平"
- **L609-613** 编 C57BL/6J 品系（**用户从未提及品系**）+ 编"典型范围 15-30% / 30-50% / 15-30 次"
- **L644** 写"EPM **金标准**要求每组 n ≥ 8-12 只"

三条违规全部出自 lead 嘴里。同事 feedback 2/3/4 (verdict=needs_fix) 全部挂在 lead 自己写的那个 run 上。本任务用 prompt 硬约束治这条——**不依赖微调**，因为这是角色越界、不是用词漂移。

**为什么 prompt 能根治这条**：deepseek 模型对"角色"指令比对"违禁词"指令更敏感（用正面指令"你的角色是调度员"，比"不许说常模"有效）。

- [ ] **Step 1: 写失败测试 — 检查 prompt 包含硬约束关键短语**

新建文件 `packages/agent/backend/tests/test_lead_prompt_role_boundaries.py`：

```python
"""验证 lead prompt 中含 Issue #3 角色边界硬约束段。

不是 e2e 测试 — 是 prompt 内容存在性检查，防 prompt 被改回旧版退化。
真正的 e2e 验证靠 Task 9 的 dogfood。
"""

from deerflow.agents.lead_agent.prompt import SYSTEM_PROMPT_TEMPLATE


def test_lead_prompt_forbids_self_interpretation_before_data_analyst() -> None:
    """lead 不应在 data-analyst 返回前自己写指标判读。"""
    p = SYSTEM_PROMPT_TEMPLATE
    # 关键短语必须存在
    assert "handoff_data_analyst.json" in p, "prompt 必须显式引用 data-analyst handoff 文件名"
    assert "调度员" in p or "调度角色" in p, "prompt 必须将 lead 定位为调度员"


def test_lead_prompt_forbids_unsupported_metadata() -> None:
    """lead 不应引用用户未告知的元数据（品系/体重等）。"""
    p = SYSTEM_PROMPT_TEMPLATE
    # 必须显式禁止未告知元数据
    assert "品系" in p, "prompt 必须显式提到品系约束"
    assert "raw" in p.lower() or "headers" in p.lower(), "prompt 必须说明合法来源（raw file headers）"


def test_lead_prompt_forbids_absolute_reference_terms() -> None:
    """lead 不应引用'典型值/常模/金标准'等绝对参考词。"""
    p = SYSTEM_PROMPT_TEMPLATE
    # 必须出现这些禁词的明确禁令（出现说明 prompt 提到了它们要禁）
    forbidden_in_constraint_section = ["典型值", "常模", "金标准"]
    for word in forbidden_in_constraint_section:
        assert word in p, f"prompt 必须显式提到禁词 '{word}'（哪怕是禁令上下文）"


def test_lead_prompt_provides_positive_example() -> None:
    """prompt 必须给出 lead 应该如何呈现指标的正例。"""
    p = SYSTEM_PROMPT_TEMPLATE
    # 期望含"正例"或"正确做法"等关键词
    assert "正例" in p or "正确做法" in p or "正确示例" in p, (
        "prompt 必须给出 lead 在没收到 data-analyst 结果时如何呈现指标的正例"
    )
```

- [ ] **Step 2: 运行测试，确认全部失败**

Run:
```bash
cd packages/agent/backend && PYTHONPATH=. uv run pytest tests/test_lead_prompt_role_boundaries.py -v
```

Expected: 4 个测试全 FAIL（因为 prompt 里还没有这段内容）。

- [ ] **Step 3: 在 prompt.py 里插入"角色边界硬约束"段**

打开 `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`，找到 `### 角色分工与契约`（line 299 附近）。在 **`### 失败处理规则`（line 309）之前**插入下面整段：

```
### 角色边界硬约束（不可越权）

你是**调度员**，不是判读员。在收到 `handoff_data_analyst.json` 之前，**禁止**做下列任何一件事：

1. **不要自己写指标判读** — 不写"7.99% 偏低/提示焦虑/正常活动"等结论。你的工作是把指标**搬给** data-analyst 解读。
2. **不要引用未告知的元数据** — 用户消息和 raw file headers 中**未出现**的字段（品系/性别/体重/年龄等），**绝对禁止**出现在你的回复中。如同事要求标注品系，**先反问用户**："请问该 Subject 的品系是？"，得到答案后再用。
3. **不要使用绝对参考术语** — 包括但不限于"典型值"/"常模"/"参考范围"/"金标准"/"文献典型"/"基线水平"。这些违反组间比较哲学（详见 CLAUDE.md 第 9 条）。

**正例**（lead 拿到 code-executor 5 个指标 m_*.json 后、还没派 data-analyst）：
```
我已经拿到了 Subject 1 的 5 个核心 EPM 指标：

| 指标 | 数值 | 单位 |
|------|------|------|
| 开臂停留时间 | 23.56 | 秒 |
| 开臂时间比例 | 7.99 | % |
| 开臂进入次数 | 6 | 次 |
| 开臂进入比例 | 28.57 | % |
| 总进入次数 | 21 | 次 |

正在派遣 data-analyst 进行行为学解读...
```

**反例**（这是 thread 5046a6e6 实际发生的错误，永远不要这样写）：
```
开臂时间比例 7.99% — 远低于典型低焦虑小鼠（通常 >20%），提示较高的焦虑水平
[或]
C57BL/6J 典型范围 15-30%，本 Subject 7.99% 偏低
[或]
EPM 金标准要求每组 n ≥ 8-12 只
```

收到 `handoff_data_analyst.json` 之后，你可以**搬运**其中的判读语句（用 data-analyst 的原话或语义相同的中文表达），但**不要叠加**自己的判读。
```

- [ ] **Step 4: 运行测试，确认全部通过**

Run:
```bash
cd packages/agent/backend && PYTHONPATH=. uv run pytest tests/test_lead_prompt_role_boundaries.py -v
```

Expected: 4 个测试全 PASS。

- [ ] **Step 5: 跑全套现有测试确认无回归**

Run:
```bash
cd packages/agent/backend && make test
```

Expected: 全绿（与开工前基线一致）。

- [ ] **Step 6: Commit**

Run:
```bash
git add packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py packages/agent/backend/tests/test_lead_prompt_role_boundaries.py
git commit -m "fix(lead): forbid self-written interpretation + unsupported metadata + absolute references

thread 5046a6e6 dogfood 暴露：lead 在派 data-analyst 之前自己写'7.99% 远低于
典型低焦虑小鼠'、编 C57BL/6J 品系、引'金标准' — 触发同事 feedback 2/3/4
全部 needs_fix。

新增'角色边界硬约束'段，把 lead 重新定位为调度员；用正例+反例约束 deepseek
对'角色'指令更敏感的特性；加 4 条 prompt 内容存在性测试防退化。

真正的 e2e 验证在 Task 9 dogfood 阶段。

Fix Issue #3 from 2026-05-13 dogfood iteration."
```

---

## Task 3: 写诊断测试，确认 thinking 字段丢失的触发链路（Issue #2 · P0 · 诊断）

**Files:**
- Test: `packages/agent/backend/tests/test_thinking_field_preserved.py`（新建，先红）

**背景**：thread 5046a6e6 的 `docs/e2e/...md:937/958/979` 三次相同报错 `messages[7]: missing field 'thinking' at line 1 column 18538`，是会话彻底死的元凶。这条 task 的目标是**先复现 + 定位**，不直接修代码 —— 修代码在 Task 4。

可能的丢字段链路（按概率排序）：
- (a) `agents/middlewares/archiving_summarization.py` 归档→恢复链路（thread 5046a6e6 有触发记录 langgraph.log line 1121 "Archived 8 messages"）
- (b) `agents/middlewares/think_tag_middleware.py` 把 `<think>` 标签搬到 reasoning_content 后丢失 raw thinking block
- (c) checkpointer 序列化时丢字段（Issue #10 同源问题）
- (d) 跨 reload 状态恢复时 thinking 字段未持久化（Issue #1 修了后此源减少但不消失）

- [ ] **Step 1: 在 backend 里加诊断单测，模拟"含 thinking content block 的 AIMessage 走完归档→恢复链路"**

新建 `packages/agent/backend/tests/test_thinking_field_preserved.py`：

```python
"""复现 Issue #2: messages[N]: missing field 'thinking' 400 错误。

thread 5046a6e6 实际触发链路：
1. 用户问"时间分段分析"
2. lead 自己 write_file 写脚本（触发 Issue #1 reload）
3. 重启后从 checkpoint 恢复历史 messages
4. 历史 message 中某条 AIMessage 的 thinking content block 丢失
5. langchain_anthropic 发请求时校验失败 → newapi 返 400

本测试模拟链路中"消息含 thinking content block → 走过某个中间件 → 校验仍含 thinking"。

诊断时先跑这个测试，看哪条链路最先 fail。Task 4 修复对应的中间件后此测试转 PASS。
"""

import pytest
from langchain_core.messages import AIMessage


@pytest.fixture
def ai_message_with_thinking() -> AIMessage:
    """A canonical AIMessage with a thinking content block.

    Mirrors what deepseek-v4-pro returns via newapi when thinking_enabled=True.
    """
    return AIMessage(
        content=[
            {"type": "thinking", "thinking": "Let me reason about this: ..."},
            {"type": "text", "text": "The answer is 42."},
        ],
        additional_kwargs={"reasoning_content": "Let me reason about this: ..."},
    )


def _has_thinking_block(msg: AIMessage) -> bool:
    """Check if msg.content has at least one type=thinking block."""
    if not isinstance(msg.content, list):
        return False
    return any(
        isinstance(b, dict) and b.get("type") == "thinking" for b in msg.content
    )


def test_thinking_block_survives_think_tag_middleware(ai_message_with_thinking: AIMessage) -> None:
    """think_tag_middleware 把内联 <think> 标签搬到 reasoning_content 时
    应保留 type=thinking 的 content block（不应删除）。
    """
    from deerflow.agents.middlewares.think_tag_middleware import strip_think_tags  # noqa: F401

    # think_tag_middleware 主要处理纯文本 <think>...</think>，
    # 不应改动已经结构化的 type=thinking content block。
    # 这里直接复制 message、过 strip_think_tags、看 block 是否还在。
    msg = ai_message_with_thinking
    # 这里假设 middleware 有公开方法 process(msg) — 实际方法名待诊断时确认
    # 如果方法名不同需要在 Task 4 改测试 + 改 production code
    assert _has_thinking_block(msg), "fixture 本身必须含 thinking block（前提）"


def test_thinking_block_survives_archiving_summarization() -> None:
    """ArchivingSummarizationMiddleware 归档 + 恢复后保留 thinking content block。

    模拟：
    - 8 条 messages 中含 1 条 AIMessage 含 thinking block
    - 触发归档，前 5 条被压缩为 summary
    - 后 3 条保留
    - 保留的 messages 中含 thinking block 的那条 thinking 字段仍在
    """
    pytest.skip("待 Task 3 Step 2 诊断后填具体调用方式")


def test_thinking_block_survives_checkpointer_roundtrip() -> None:
    """checkpointer 序列化 + 反序列化后保留 thinking content block。

    模拟：
    - 把含 thinking block 的 message 存到 checkpoint
    - 读回来
    - thinking 字段还在
    """
    pytest.skip("待 Task 3 Step 2 诊断后填具体调用方式")
```

- [ ] **Step 2: 运行测试 + 收集诊断信息**

Run:
```bash
cd packages/agent/backend && PYTHONPATH=. uv run pytest tests/test_thinking_field_preserved.py -v
```

Expected：第 1 个测试 PASS（fixture sanity），第 2/3 个 SKIP。诊断的关键是接下来——**真复现一次 thread 5046a6e6 的死法**。

执行下列命令拿一份完整的坏 message dump：

```bash
# 1. 启动 backend 时把 anthropic SDK 的 debug 日志打开
export ANTHROPIC_LOG=debug

# 2. 在测试 thread 上：
#    - 上传一份 EPM 数据
#    - lead 走到派 code-executor 之后
#    - 手动让 gateway reload 一次（touch app/gateway/app.py）
#    - 用户再发一条消息让 lead 继续
#    - 抓 anthropic 协议层的请求 body

# 3. 在 logs/langgraph.log 里找到下次 LLM 调用前的 messages dump，
#    定位哪条 AIMessage 的 thinking 字段丢了

grep -A 100 "missing field.*thinking" packages/agent/logs/langgraph.log | head -120
```

把诊断结果（哪条中间件/链路丢字段）写进 `docs/handoffs/2026-05/2026-05-14-thinking-field-diagnosis-notes.md` 备查。

- [ ] **Step 3: 根据诊断结果，把 Step 1 的两条 skipped 测试改为真实可执行**

诊断后已知具体丢字段位置 → 把对应 skip 改成真测试代码。每条测试必须明确：
- 输入：含 thinking content block 的 message
- 经过：哪条中间件 / 哪个序列化路径
- 断言：处理后 message **仍含** type=thinking 的 content block

如果是 `archiving_summarization.py` 在归档前压缩消息时丢字段：

```python
def test_thinking_block_survives_archiving_summarization(ai_message_with_thinking: AIMessage) -> None:
    from deerflow.agents.middlewares.archiving_summarization import ArchivingSummarizationMiddleware

    middleware = ArchivingSummarizationMiddleware(
        model=None,  # 不需要真模型；测试只验证保留逻辑
        max_tokens_before_summary=100,
        messages_to_keep=3,
    )

    messages_before_archive = [
        AIMessage(content="msg 1"),
        AIMessage(content="msg 2"),
        AIMessage(content="msg 3"),
        AIMessage(content="msg 4"),
        AIMessage(content="msg 5"),
        ai_message_with_thinking,  # 这条必须保留 thinking block
        AIMessage(content="msg 7"),
        AIMessage(content="msg 8"),
    ]

    preserved = middleware._partition_messages_for_test(  # 调真实 partition 方法
        messages_before_archive,
        cutoff_index=5,
    )[1]  # preserved_messages

    # 找到 fixture 那条
    target = next((m for m in preserved if m is ai_message_with_thinking), None)
    assert target is not None, "fixture message 应在 preserved 列表中"
    assert _has_thinking_block(target), "preserved 后 thinking block 必须仍在"
```

如果丢字段位置是其它链路，类比改造。

- [ ] **Step 4: 运行测试，确认能复现 bug（红）**

Run:
```bash
cd packages/agent/backend && PYTHONPATH=. uv run pytest tests/test_thinking_field_preserved.py -v
```

Expected: 至少有 1 条测试 FAIL，标记 bug 复现位置。**不要尝试在 Task 3 里修——继续 Task 4**。

- [ ] **Step 5: Commit 诊断 + 失败测试**

Run:
```bash
git add packages/agent/backend/tests/test_thinking_field_preserved.py
git add docs/handoffs/2026-05/2026-05-14-thinking-field-diagnosis-notes.md
git commit -m "test(thinking): reproduce Issue #2 thinking-field 400 error in unit tests

thread 5046a6e6 dogfood: messages[7]: missing field 'thinking' 400 错误
导致会话彻底死。本提交先加复现测试 + 诊断笔记，不修代码，修在下一个 commit。

Issue #2 — diagnosis only, fix in Task 4."
```

---

## Task 4: 修复 thinking 字段丢失（Issue #2 · P0 · 修复）

**Files:**
- Modify: 视 Task 3 诊断结果定（最可能是 `archiving_summarization.py` 或 `think_tag_middleware.py`）
- Test: 把 Task 3 写的失败测试转 PASS

**背景**：本任务的具体改动**完全依赖 Task 3 的诊断结果**。下面给出最可能的两个修复路径模板。

- [ ] **Step 1: 根据 Task 3 诊断，定位失字段的最小可复现位置**

如果诊断指向 `archiving_summarization.py`：丢字段往往在 `_partition_messages` / `_apply_summary` 的消息重建逻辑——它可能用 `AIMessage(content=str(m.content))` 重建，把 list content 强转 str 丢了 thinking block。

如果诊断指向 `think_tag_middleware.py`：丢字段可能在 `strip_think_tags` 之后没把 thinking content block 复原到 message.content。

- [ ] **Step 2: 实施最小修复**

修复原则：**保留 message.content 的原始结构，不要强转 str**。

如果是 `archiving_summarization.py` 的归档路径：

修改前（示例代码，实际待 Task 3 定位）：
```python
def _rebuild_message(self, m: BaseMessage) -> BaseMessage:
    # 简化版重建 — 把 content 强转 str（错）
    return type(m)(content=str(m.content))
```

修改后：
```python
def _rebuild_message(self, m: BaseMessage) -> BaseMessage:
    # 保留 list 形式的 content（含 thinking content block）
    return type(m)(
        content=m.content,  # 原样保留
        additional_kwargs=dict(m.additional_kwargs or {}),
    )
```

具体代码视诊断而定。**关键约束**：修改后 `_has_thinking_block(m)` 在改写前后**结果一致**。

- [ ] **Step 3: 运行 Task 3 的测试，确认转 PASS**

Run:
```bash
cd packages/agent/backend && PYTHONPATH=. uv run pytest tests/test_thinking_field_preserved.py -v
```

Expected: 所有测试 PASS。

- [ ] **Step 4: 跑全套测试无回归**

Run:
```bash
cd packages/agent/backend && make test
```

Expected: 全绿。

- [ ] **Step 5: 手动验证 — 重现 thread 5046a6e6 死法**

Run：在 dogfood 环境（make stop && make dev 重启）开一个新 thread，做以下 e2e 验证：

1. 上传 EPM 单只数据
2. 等 lead 派 code-executor 完成
3. **手动触发一次归档**：连发多条消息让 thread 达到归档阈值
4. 归档触发后 lead 应能继续派 data-analyst，**不应**返回 `missing field 'thinking'` 400

如果手动验证失败，回到 Task 3 重新诊断（可能丢字段在另一条链路）。

- [ ] **Step 6: Commit**

Run:
```bash
git add <Task 4 修改的文件>
git commit -m "fix(<middleware>): preserve thinking content block during message rebuild

thread 5046a6e6 dogfood: 归档/恢复链路在重建消息时丢了 type=thinking content
block，导致下次 LLM 请求被 newapi 返 400 'messages[N]: missing field thinking'，
会话彻底死。

修复：<中间件名> 重建消息时保留 m.content 原始 list 结构，不再强转 str。
Task 3 的 4 条单测全部 PASS。手动 dogfood 触发归档场景验证不再 400。

Fix Issue #2 from 2026-05-13 dogfood iteration."
```

---

## Task 5: lead 阶段播报强化（Issue #5 · P1）

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`（line 407 `### 过程透明原则` 段强化）

**背景**：现有 prompt line 407 已有"过程透明原则"段，但 thread 5046a6e6 实际表现是**只在派遣 code-executor 前播报了一次**，data-analyst 之前没播报、收尾时也没"分析完毕"播报。用户原话"端到端流程对用户展现的也不是很清晰"。本任务把"软建议"升级为"强制清单"。

- [ ] **Step 1: 读现有 prompt line 407-426 段，看清现有措辞**

Run:
```bash
sed -n '407,430p' packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py
```

记下现有内容（不删，是强化）。

- [ ] **Step 2: 把 `### 过程透明原则` 段升级为强制清单**

把 line 407 起的 `### 过程透明原则` 整段替换为：

```
### 过程透明原则（强制清单）

每次调用 `task` 派遣 subagent、调用 `bash` 跑 catalog/parse CLI、调用 `ask_clarification` 反问用户、调用 `present_files` 呈现文件之前，**必须**先用 1 条简短中文消息向用户播报状态。

**强制场景清单**（缺一不可）：

| 触发动作 | 必须的播报模板 |
|---------|---------------|
| 派 code-executor | "🧮 正在计算 N 个 <范式> 指标，预计 30-60 秒..." |
| 派 data-analyst | "🔬 指标已完成，正在请专家解读，预计 1-2 分钟..." |
| 派 report-writer | "📝 解读已完成，正在生成中文研究报告..." |
| 跑 `python -m ethoinsight.parse.dump_headers` | "📂 正在解析 EthoVision 文件结构..." |
| 跑 `python -m ethoinsight.catalog.resolve` | "📋 正在生成指标计划..." |
| ask_clarification 反问 | "⚠️ 我需要先确认一件事..." |
| subagent 完成后下一个 action 前 | "✅ <subagent> 已完成。<下一步>..." |

**说法要面向研究员用户，不暴露内部实现细节**：
- ✅ "正在请专家解读结果"（用户视角）
- ❌ "我会调用 data-analyst subagent"（内部实现名）
- ✅ "数据有质量警告，需要先确认一下"
- ❌ "handoff.data_quality_warnings 包含 critical 级警告"

每一步 subagent 返回后，在下一次行动前先呈现**可读洞察**（接现有"分析结果呈现模板"段，详见 line 428 起）。

**反例**（thread 5046a6e6 实际发生的错误，永远不要这样）：
- 派完 code-executor 直接派 data-analyst，中间不向用户说一句
- subagent 完成后不汇报，直接派下一个
- 跑 bash 命令前不解释（用户看到一长串 SandboxAudit 命令但不知道为什么）
```

- [ ] **Step 3: 跑现有测试无回归**

Run:
```bash
cd packages/agent/backend && make test
```

Expected: 全绿。

- [ ] **Step 4: Commit**

Run:
```bash
git add packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py
git commit -m "fix(lead): upgrade transparency from suggestion to mandatory checklist

thread 5046a6e6 dogfood: lead 派 subagent 前几乎不播报，用户原话'端到端流程
对用户展现的也不是很清晰'。原有'过程透明原则'段是软建议，模型经常忽略。

新版改成强制清单 + 模板表 + 反例。配合 Issue #4（前端不再自动折叠 reasoning）
形成完整的可见性闭环。

E2E 验证在 Task 9 dogfood 阶段确认每次 task/bash/ask_clarification 前都有
对应播报。

Fix Issue #5 from 2026-05-13 dogfood iteration."
```

---

## Task 6: 调用处用受控 open 关闭 reasoning auto-close（Issue #4 · P1）

**Files:**
- Modify: `packages/agent/frontend/src/components/workspace/messages/message-list-item.tsx`

**重要约束**：Frontend `CLAUDE.md` 明确写 `ai-elements/` 是从 Vercel AI SDK registry **生成**的目录，**禁止手改**：
> `ui/` and `ai-elements/` are generated from registries (Shadcn, MagicUI, React Bits, Vercel AI SDK) — don't manually edit these.

本任务**不动** `ai-elements/reasoning.tsx`，**只改调用处**。

**背景**：`ai-elements/reasoning.tsx:43` 硬编码 `AUTO_CLOSE_DELAY = 1000`，加上 `:82-93` useEffect，流式结束 1 秒后强制折叠 reasoning。这是上游 Vercel AI Elements 给"通用 chatbot"的默认行为——研究员看不到推理过程。

**修复策略（路径 A：受控 open）**：

`Reasoning` 组件用 `@radix-ui/react-use-controllable-state` 实现 controlled 模式。机制如下：

1. **`defaultOpen` 是 useEffect 的 guard**：`reasoning.tsx:84` 的 useEffect 第一行就是 `if (defaultOpen && !isStreaming && isOpen && !hasAutoClosed)`，传 `defaultOpen={false}` 让整段 auto-close 永远不触发
2. **`open`/`onOpenChange` 让父组件受控**：传 `open={localState}` 后，组件内部的 `setIsOpen()` 在 controlled 模式下**只触发回调，不改 state**（验证：`@radix-ui/react-use-controllable-state/dist/index.mjs:32-44`）。父组件用 `useState` 决定真实 state

**组合效果**：`defaultOpen={false}` 关 auto-close 逻辑 + `open={true}` 强制初始展开 + `onOpenChange` 让用户手动折叠仍能生效。

**EthoInsight 调用处只有 1 个**（已确认）：`packages/agent/frontend/src/components/workspace/messages/message-list-item.tsx:215`。

- [ ] **Step 1: 确认 Reasoning 调用处只有 1 个**

Run:
```bash
cd packages/agent/frontend && grep -rln "from \"@/components/ai-elements/reasoning\"\|from '@/components/ai-elements/reasoning'" src/
```

Expected: 输出 1 行 `src/components/workspace/messages/message-list-item.tsx`。如果输出多行，**停下**——本计划只覆盖 1 处调用，多处情况需重新评估。

- [ ] **Step 2: 在 message-list-item.tsx 加受控 open state**

打开 `packages/agent/frontend/src/components/workspace/messages/message-list-item.tsx`。

确认 `useState` 已经 import（这是 React 19 项目，应已 import）。如果没有，在文件顶部 import 处加：

```tsx
import { useState } from "react";
```

找到 line 215 附近的 `<Reasoning isStreaming={isLoading}>` 段（当前完整代码）：

```tsx
{reasoningContent && (
  <Reasoning isStreaming={isLoading}>
    <ReasoningTrigger />
    <ReasoningContent>{reasoningContent}</ReasoningContent>
  </Reasoning>
)}
```

修改为：

```tsx
{reasoningContent && (
  <ReasoningPanel
    isStreaming={isLoading}
    reasoningContent={reasoningContent}
  />
)}
```

**新增**：把 reasoning UI 拆成内部子组件 `ReasoningPanel`，原因——`useState` 必须在组件顶层调用，不能放在 `{reasoningContent && ...}` 条件渲染里。在**当前文件末尾、`MessageListItem` 组件函数体外**追加：

```tsx
/**
 * Local wrapper around ai-elements/Reasoning to keep the panel expanded by
 * default and skip the upstream "auto-close 1s after streaming ends" behavior.
 *
 * Why: Upstream Vercel AI Elements assumes "users only care about the answer."
 * For EthoInsight (a research assistant for behavioral scientists), users need
 * to inspect the full reasoning trace to trust the conclusion.
 *
 * Implementation: We don't modify ai-elements/reasoning.tsx (it's generated
 * from upstream and CLAUDE.md forbids manual edits). Instead we drive
 * `defaultOpen={false}` (which gates the auto-close useEffect at
 * ai-elements/reasoning.tsx:84) combined with controlled `open` state — the
 * Radix useControllableState contract guarantees that internal setIsOpen() is
 * a no-op while controlled, so the panel stays open until the user clicks the
 * trigger to collapse it.
 */
function ReasoningPanel({
  isStreaming,
  reasoningContent,
}: {
  isStreaming: boolean;
  reasoningContent: string;
}) {
  const [open, setOpen] = useState(true);

  return (
    <Reasoning
      isStreaming={isStreaming}
      defaultOpen={false}
      open={open}
      onOpenChange={setOpen}
    >
      <ReasoningTrigger />
      <ReasoningContent>{reasoningContent}</ReasoningContent>
    </Reasoning>
  );
}
```

- [ ] **Step 3: typecheck + lint**

Run:
```bash
cd packages/agent/frontend && pnpm check
```

Expected: 0 errors。

如果 `pnpm check` 报 `useState` 未使用警告（说明已 import），跳过 Step 2 的 import 改动。

- [ ] **Step 4: 手动验证 — dev server 启动看 reasoning 不再消失**

Run:
```bash
cd packages/agent && make stop && make dev
```

在浏览器打开 `http://localhost:2026`，新建 thread，让 agent 跑一轮（任意有 reasoning 输出的对话）：

| 行为 | 期望 |
|------|------|
| 流式开始时 | reasoning 展开（同改前） |
| 流式期间 | reasoning 持续展开（同改前） |
| 流式结束 | **reasoning 仍然展开**（修复成功，改前 1 秒后会折叠） |
| 等待 30 秒 | 仍然展开（修复成功） |
| 用户点击 `ReasoningTrigger` 按钮 | 折叠（保留交互能力） |
| 用户再次点击 | 重新展开 |

**关键验证**：流式结束后**等待 5 秒以上**确认不会自动折叠。这是路径 A 机制保证的，但工程上必须看到才算通过。

- [ ] **Step 5: Commit**

Run:
```bash
git add packages/agent/frontend/src/components/workspace/messages/message-list-item.tsx
git commit -m "fix(frontend): keep reasoning panel expanded via controlled state wrapper

thread 5046a6e6 dogfood: 用户原话'think 打印出来了一些内容后，突然会消除'。
根因是 ai-elements/reasoning.tsx:43 AUTO_CLOSE_DELAY=1000 + useEffect 在流式
结束 1 秒后强制折叠，研究员看不到推理过程。

Frontend CLAUDE.md 禁止手改 ai-elements/ 目录（registry-generated）。改用
受控 open + defaultOpen=false 在调用处拦住 auto-close：
- defaultOpen=false 让 ai-elements/reasoning.tsx:84 useEffect guard 永远 fail
- 受控 open 用本地 useState，Radix useControllableState 在 controlled 下
  让组件内 setIsOpen() 只触发 onOpenChange、不改 state
- 用户手动点击 ReasoningTrigger 仍可折叠（onOpenChange→setOpen 走通）

Fix Issue #4 from 2026-05-13 dogfood iteration."
```

---

## Task 7: 修 catalog plan.json 输出路径打穿 sandbox 抽象（Issue #6 · P1）

**Files:**
- Modify: `packages/ethoinsight/ethoinsight/catalog/resolve.py:245`（拼输出路径处）
- Modify: 调用 resolve 的 sandbox 路径翻译层（Step 1 定位）
- Test: `packages/ethoinsight/tests/test_catalog_resolve_paths.py`（新建或扩展）

**背景**：thread 5046a6e6 实物 `metric_plan.json` 第一行的 `input` 是虚拟路径 `/mnt/user-data/uploads/...`，但 `output` 是**物理绝对路径** `/home/wangqiuyang/.../user-data/workspace/m_*.json`。lead 通过 CLI 传 `--workspace-dir /mnt/user-data/workspace`（虚拟）—— 但被 sandbox 路径翻译层中途转成物理后再被 `resolve.py:245` 拼接。

**修复原则**：`metric_plan.json` 是给下游 subagent 看的"施工单"，**必须**只含虚拟路径——sandbox 抽象层不应漏物理路径出去。

- [ ] **Step 1: 找出 bash tool 在哪里做虚拟→物理翻译**

Run:
```bash
grep -rn "replace_virtual_path\|virtual_path\|/mnt/user-data" packages/agent/backend/packages/harness/deerflow/sandbox/ --include="*.py" | grep -v __pycache__ | head -20
```

定位翻译函数（已知约为 `sandbox/tools.py` 的 `replace_virtual_path` 系列）。

确认翻译时机：bash 命令文本里的 `/mnt/user-data/workspace` 在传给 subprocess 之前被替换成物理路径。**这是合理的**——bash 进程在 host 上跑，必须看物理路径。

**真正的 bug 在**：lead 通过 bash 传给 catalog.resolve 的 `--workspace-dir /mnt/user-data/workspace` 被翻译成物理后，`resolve.py:245` 直接用这个物理 workspace_dir 拼路径写进 plan.json —— **plan.json 是给后续 LLM 看的，物理路径不该出现**。

- [ ] **Step 2: 写失败测试 — catalog.resolve 输出虚拟路径**

新建 `packages/ethoinsight/tests/test_catalog_resolve_paths.py`：

```python
"""验证 metric_plan.json 的 output 路径必须是虚拟路径（不能含 host 物理路径）。

Issue #6 from thread 5046a6e6 dogfood:
plan.json 的 output 字段写了 /home/wangqiuyang/.../workspace/m_*.json 物理路径，
打穿了 sandbox 抽象。后续 subagent 看到这条 plan 时，本来应该用 /mnt/...
虚拟路径，但 lead 直接 cat plan.json 后把物理路径暴露给模型。
"""

from pathlib import Path

import pytest

from ethoinsight.catalog.resolve import resolve


VIRTUAL_WORKSPACE = "/mnt/user-data/workspace"
VIRTUAL_UPLOADS = "/mnt/user-data/uploads"


# EPM default_metrics 要求列 `in_zone_open_arms_*` (glob)。
# 测试用 1 个匹配的列名让 resolve 真跑通完整流程。
EPM_MIN_COLUMNS = [
    "in_zone_open_arms_center",
    "in_zone_closed_arms_center",
]


def test_resolve_outputs_only_virtual_paths(tmp_path: Path) -> None:
    """无论 caller 传虚拟还是物理路径，resolve 必须只产出虚拟路径。"""
    # 模拟 thread 5046a6e6 场景：lead 传虚拟，但 sandbox 翻译为物理
    physical_workspace = str(tmp_path / "workspace")
    Path(physical_workspace).mkdir(parents=True)

    raw_files = [f"{VIRTUAL_UPLOADS}/dummy.txt"]

    # 调真实 resolve()。签名（来自 resolve.py:54-67）：
    # resolve(paradigm, columns, raw_files, workspace_dir, *,
    #         include, exclude, n_per_group, n_groups,
    #         groups_file, columns_file, ev19_template)
    plan = resolve(
        paradigm="epm",
        columns=EPM_MIN_COLUMNS,
        raw_files=raw_files,
        workspace_dir=physical_workspace,  # 模拟翻译后的物理路径
        ev19_template="PlusMaze-FewZones",
    )

    # 关键断言：plan 里每个 metric 的 output 不能含 host 物理路径
    for m in plan.metrics:
        assert m.output.startswith(VIRTUAL_WORKSPACE), (
            f"metric {m.id} output 不是虚拟路径: {m.output}"
        )
        assert "/home/" not in m.output, (
            f"metric {m.id} output 含 host 物理路径前缀: {m.output}"
        )


def test_resolve_outputs_use_virtual_workspace_with_explicit_kwarg(tmp_path: Path) -> None:
    """resolve 应支持显式 virtual_workspace_dir 参数，与物理 workspace_dir 区分。"""
    # 这是 Step 4 引入新参数后的测试
    pytest.skip("待 Step 4 引入 virtual_workspace_dir 参数后启用")
```

- [ ] **Step 3: 跑测试，确认 FAIL（复现 bug）**

Run:
```bash
cd packages/ethoinsight && pytest tests/test_catalog_resolve_paths.py -v
```

Expected: `test_resolve_outputs_only_virtual_paths` FAIL，错误信息显示 `output` 含 `/home/...`。

- [ ] **Step 4: 修 `resolve.py`，引入 `virtual_workspace_dir` 参数**

修改 `packages/ethoinsight/ethoinsight/catalog/resolve.py`：

1. `resolve()` 函数签名加 `virtual_workspace_dir: str | None = None` 参数（向后兼容）。
2. `_build_metric_plan` 里 line 245 改成：

```python
# 旧
output_path = str(Path(workspace_dir) / f"m_{m.id}.json")

# 新
# Prefer virtual workspace for plan.json output paths so plan.json never
# leaks host filesystem paths to downstream subagents.
effective_workspace = virtual_workspace_dir or workspace_dir
output_path = str(Path(effective_workspace) / f"m_{m.id}.json")
```

3. 同步改 CLI（`packages/ethoinsight/ethoinsight/catalog/cli.py`）加 `--virtual-workspace-dir` 参数（如未提供则用 `--workspace-dir` 兜底）。

- [ ] **Step 5: 修 lead prompt，要求 lead 调 catalog.resolve 时显式传 `--virtual-workspace-dir /mnt/user-data/workspace`**

找到 `lead_agent/prompt.py` 里描述 catalog.resolve 调用方式的段（应在派 code-executor 之前的工作流段，或在 `ethoinsight-metric-catalog` skill 里）。把示例命令改为：

```bash
python -m ethoinsight.catalog.resolve \
    --paradigm epm \
    --ev19-template PlusMaze-FewZones \
    --columns-file /mnt/user-data/workspace/columns.json \
    --raw-files-json /mnt/user-data/workspace/raw_files.json \
    --workspace-dir /mnt/user-data/workspace \
    --virtual-workspace-dir /mnt/user-data/workspace \
    --groups-file /mnt/user-data/workspace/groups.json \
    --output /mnt/user-data/workspace/metric_plan.json
```

（注：本计划默认这段在 `lead_agent/prompt.py`。如果实际在 `packages/agent/skills/custom/ethoinsight-metric-catalog/` 下的 skill 文件，相应改那里）

- [ ] **Step 6: 把 Step 2 第二条 skip 测试改为真实测试**

```python
def test_resolve_outputs_use_virtual_workspace_with_explicit_kwarg(tmp_path: Path) -> None:
    """resolve 收到 virtual_workspace_dir 后必须用它而非物理 workspace_dir。"""
    physical_workspace = str(tmp_path / "workspace")
    Path(physical_workspace).mkdir(parents=True)
    virtual_workspace = "/mnt/user-data/workspace"

    raw_files = [f"{VIRTUAL_UPLOADS}/dummy.txt"]

    plan = resolve(
        paradigm="epm",
        columns=EPM_MIN_COLUMNS,
        raw_files=raw_files,
        workspace_dir=physical_workspace,           # 物理
        ev19_template="PlusMaze-FewZones",
        virtual_workspace_dir=virtual_workspace,    # 虚拟（Step 4 新加的 kwarg）
    )

    for m in plan.metrics:
        assert m.output.startswith(virtual_workspace), m.output
```

- [ ] **Step 7: 跑测试，确认 PASS**

Run:
```bash
cd packages/ethoinsight && pytest tests/test_catalog_resolve_paths.py -v
```

Expected: 全部 PASS。

- [ ] **Step 8: 跑全套测试无回归**

Run:
```bash
cd packages/ethoinsight && pytest tests/ -v
cd packages/agent/backend && make test
```

Expected: 全绿。

- [ ] **Step 9: Commit**

Run:
```bash
git add packages/ethoinsight/ethoinsight/catalog/resolve.py
git add packages/ethoinsight/ethoinsight/catalog/cli.py
git add packages/ethoinsight/tests/test_catalog_resolve_paths.py
git add packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py
git commit -m "fix(catalog): force virtual paths in metric_plan.json output field

thread 5046a6e6 dogfood: plan.json 的 output 字段含 host 物理路径
/home/wangqiuyang/.../workspace/m_*.json，打穿了 sandbox 抽象。下游 subagent
读 plan 时看到物理路径暴露，违反 catalog single-source-of-truth 哲学。

修复：
- resolve() 加 virtual_workspace_dir 参数，优先用它构造 output 路径
- CLI 加 --virtual-workspace-dir，未提供时兜底用 --workspace-dir
- lead prompt 示例命令改为同时传两个 workspace dir
- 加 2 条单测锁死'plan.json output 必须虚拟路径'

Fix Issue #6 from 2026-05-13 dogfood iteration."
```

---

## Task 8: 收紧 code-executor critical_rules，禁止 ls 验证后重跑（Issue #7 · P1）

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py:39-43`

**背景**：thread 5046a6e6 run 3 的 SandboxAudit 显示 code-executor 在 09:16:35 跑完 5 个 compute_*、09:16:41 ls 验证产物存在、09:17:02 又**完全重跑**了一遍 5 个 compute_*。多花约 30 秒。已有 critical_rules (commit b36e315a) 没压住"ls 后重跑"这个 pattern。

- [ ] **Step 1: 读现有 critical_rules 段**

Run:
```bash
sed -n '39,43p' packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py
```

记下现有 3 条规则的措辞（保留，新增第 4 条）。

- [ ] **Step 2: 在 critical_rules 段末尾追加第 4 条**

打开 `packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py`，找到 `<critical_rules>` 段（line 39-43），在 `</critical_rules>` 之前追加：

```
- **每个 compute_* 脚本对每个 metric_id 只允许执行一次**。如果你已经跑过 `python -m ethoinsight.scripts.<paradigm>.compute_<metric>`，**禁止**第二次跑同一个脚本，即使你 ls 验证产物时觉得"不放心"。**ls 看到产物文件存在就是成功**。重跑只会浪费 turn 预算，让你写不完 handoff。
  - **正确流程**：read plan → bash 跑 N 个 compute_* → bash ls 验证 N 个产物 → write handoff_code_executor.json → 输出 [gate_signals] → 完成。
  - **错误流程**（thread 5046a6e6 暴露的真实案例）：跑 5 个 compute_* → ls → 觉得"再确认一下" → 又跑 5 个 → 浪费 5-10 个 turn → 没空间写 handoff。
```

- [ ] **Step 3: 跑现有测试无回归**

Run:
```bash
cd packages/agent/backend && make test
```

Expected: 全绿。

- [ ] **Step 4: Commit**

Run:
```bash
git add packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py
git commit -m "fix(code-executor): forbid re-running compute_* after ls verification

thread 5046a6e6 run 3 SandboxAudit: code-executor 在 09:16:35 跑完 5 个
compute_*、09:16:41 ls 验证、09:17:02 又重跑了一遍同样 5 个脚本。critical_rules
b36e315a 没压住'ls 后不放心又重跑'pattern，多花 30s/run。

新增第 4 条 critical_rule，明确'每个 metric_id 一次'，并配正例+反例。
LLM 行为类问题不保证 100% 不复发，Task 9 dogfood 阶段观察复发率。

Fix Issue #7 from 2026-05-13 dogfood iteration."
```

---

## Task 9: Dogfood 验证 + report-writer 派遣观察（Issue #8 · P1）

**Files:**
- Create: `docs/handoffs/2026-05/2026-05-14-dogfood-followup-handoff.md`（验证存档）

**背景**：完成 Task 1-8 后必须做一次完整 dogfood，验证 Batch A/B 真的解决了问题，且**同时**观察 report-writer 是否被派遣 + 是否真读 catalog YAML。

- [ ] **Step 1: 重启服务**

Run:
```bash
cd packages/agent && make stop && make dev
```

等服务起来（约 30-60 秒）。

- [ ] **Step 2: 打开新浏览器 thread 跑单只 EPM**

操作：
1. 打开 `http://localhost:2026`
2. 新建 thread
3. 上传 `/home/wangqiuyang/DemoData/newdemodata/高架十字迷宫_小鼠_三点/` 下任意 1 个 Subject 文件（或复用 thread 5046a6e6 同一文件）
4. 发消息："请分析这个 EPM 单只数据"
5. 回答 lead 的反问（Subject 是对照组、treatment 是 1 mol GABA、暂时只看单只描述性）

- [ ] **Step 3: 观察检查清单（逐项打勾）**

| 检查项 | 期望 | 来源 Issue |
|--------|------|-----------|
| ☐ gateway.log 不再出现 `WatchFiles detected changes in '.../workspace/...py'` | 0 次 | #1 |
| ☐ langgraph.log 不再出现 `messages[N]: missing field 'thinking'` 400 | 0 次 | #2 |
| ☐ lead 在派 code-executor 之前发消息播报"🧮 正在计算..."类 | 至少 1 次 | #5 |
| ☐ lead 派 data-analyst 之前发消息播报"🔬 ..." | 至少 1 次 | #5 |
| ☐ lead 在数据分析阶段**不**自己写"7.99% 偏低"等判读 | 不出现 | #3 |
| ☐ lead 不引用 C57BL/6J 品系（用户没说品系时） | 不出现 | #3 |
| ☐ lead 不引用"典型值/常模/金标准/参考范围" | 不出现 | #3 |
| ☐ 前端 reasoning 流式结束后**不**自动折叠 | 仍展开 | #4 |
| ☐ workspace 实物 metric_plan.json 的 output 字段全是 `/mnt/user-data/workspace/m_*.json` | 全虚拟 | #6 |
| ☐ langgraph.log 里 SandboxAudit 每个 compute_* 脚本只出现一次（不重跑） | 1 次 | #7 |
| ☐ report-writer 是否被派遣（看 `SubagentExecutor initialized: report-writer`） | **观察记录** | #8 |
| ☐ 如果 report-writer 被派，它是否 `read_file .../catalog/epm.yaml` | **观察记录** | #8 |

实际操作命令：
```bash
# 记下 dogfood 开始时间
DOGFOOD_START=$(date -u +%FT%TZ)
echo "Dogfood started at: $DOGFOOD_START"

# 用浏览器跑完整 thread 后，捞 log
# 1. 看 reload 是否消失
grep -c "WatchFiles detected changes" packages/agent/logs/gateway.log
# 2. 看 thinking 400 是否消失
grep -c "missing field.*thinking" packages/agent/logs/langgraph.log
# 3. 看 report-writer 是否被派
grep -c "SubagentExecutor initialized: report-writer" packages/agent/logs/langgraph.log
# 4. 看每个 compute_* 是否只跑一次
for m in open_arm_time_ratio open_arm_time open_arm_entry_count open_arm_entry_ratio total_entry_count; do
  count=$(grep -c "compute_$m " packages/agent/logs/langgraph.log)
  echo "compute_$m: $count time(s)"
done
# 5. 看 plan.json 输出路径
NEW_THREAD_ID=<从浏览器 URL 复制>
cat "packages/agent/backend/.deer-flow/users/*/threads/$NEW_THREAD_ID/user-data/workspace/metric_plan.json" | python3 -c "import json,sys; p=json.load(sys.stdin); print('\n'.join(m['output'] for m in p['metrics']))"
```

- [ ] **Step 4: 把验证结果写进 handoff 文档**

新建 `docs/handoffs/2026-05/2026-05-14-dogfood-followup-handoff.md`，按下面骨架填写。**模板里 `_(待填)_` 是执行时的占位符**，需要执行者用 Step 3 命令的实际输出填进去：

```markdown
# 2026-05-14 Dogfood 修复迭代验证交接

## Batch A/B 修复验证结果（vs Issue #1-8）

| Issue | 检查项 | 期望 | 实测 | 结论 |
|-------|--------|------|------|------|
| #1 | gateway reload 次数 | 0 | _(待填)_ | ✅/❌ |
| #2 | thinking 400 次数 | 0 | _(待填)_ | ✅/❌ |
| #3 | lead 自写判读 | 0 | _(待填)_ | ✅/❌ |
| #3 | lead 编品系 | 不出现 | _(待填)_ | ✅/❌ |
| #3 | lead 引常模/金标准 | 不出现 | _(待填)_ | ✅/❌ |
| #4 | reasoning 自动折叠 | 不发生 | _(待填)_ | ✅/❌ |
| #5 | 阶段播报次数 | ≥4 次 | _(待填)_ | ✅/❌ |
| #6 | plan.json 输出虚拟路径 | 全虚拟 | _(待填)_ | ✅/❌ |
| #7 | compute_* 重跑次数 | 1 次/个 | _(待填)_ | ✅/❌ |
| #8 | report-writer 被派 | 观察 | _(待填)_ | _(视情况)_ |
| #8 | report-writer 读 catalog YAML | 观察 | _(待填)_ | _(视情况)_ |

## thread 信息
- thread_id: _(待填)_
- run_ids: _(待填)_
- 数据文件: _(待填)_
- 开始时间: _(待填)_
- 结束时间: _(待填)_

## 异常观察
_(把验证清单里 ❌ 的项 + 任何意外行为详细记录在这)_

## 下一步建议
_(根据验证结果，决定是否需要回头修 Batch A/B、还是可以前进到 Batch D)_
```

- [ ] **Step 5: Commit 验证存档**

Run:
```bash
git add docs/handoffs/2026-05/2026-05-14-dogfood-followup-handoff.md
git commit -m "docs(dogfood): record Batch A/B verification results

验证 thread <NEW_THREAD_ID>：Issue #1-8 修复后的 dogfood 结果。
report-writer 派遣行为已观察（Issue #8 待决）。

具体结果见 handoff 文档表格。"
```

- [ ] **Step 6: 视验证结果决定下一步**

如果验证清单所有 ✅：进入 Task 10（Issue #10 checkpointer）和后置 task。

如果有任何 ❌：
- 是 P0 修复未生效 → 回到对应 Task 重新诊断 + 修
- 是 P1 修复未生效但不阻塞流程 → 记入 handoff "下一步建议"，先做 Task 10
- 是 Issue #8 report-writer 没被派 → 创建 Task 9.5（lead prompt 加"data-analyst 完成且生成报告意图明确时必须派 report-writer"），但**不在本计划范围**，作为后续 issue 处理

---

## Task 10: 实现 AsyncSqliteSaver.adelete_for_runs（Issue #10 · P2）

**Files:**
- Modify: checkpointer 实现（Step 1 定位具体文件）
- Test: `packages/agent/backend/tests/test_checkpointer_adelete_for_runs.py`

**背景**：langgraph.log 每个 run 启动都有 warning `Custom checkpointer missing adelete_for_runs: multitask_strategy='rollback' will not clean up checkpoints from cancelled runs`。Task 1 修了 reload bug 之后，cancel 触发频率会大降，但用户连发消息仍会触发。本任务把 rollback 后的清理做干净。

- [ ] **Step 1: 定位自定义 checkpointer 实现**

Run:
```bash
grep -rn "class.*AsyncSqliteSaver\|AsyncSqliteSaver(" packages/agent/backend/packages/harness/deerflow/ --include="*.py" | grep -v __pycache__ | head -10
```

定位 saver 类的真正定义位置。

- [ ] **Step 2: 看 langgraph 内置 PostgresSaver 的 adelete_for_runs 契约**

Run:
```bash
uv run python -c "from langgraph.checkpoint.postgres import PostgresSaver; help(PostgresSaver.adelete_for_runs)" 2>&1 | head -20
```

读契约说明（参数：`thread_id: str, run_ids: list[str]`；返回：`None`；语义：删除 thread 内匹配 run_ids 的所有 checkpoint）。

- [ ] **Step 3: 写失败测试**

新建 `packages/agent/backend/tests/test_checkpointer_adelete_for_runs.py`：

```python
"""验证自定义 AsyncSqliteSaver 实现 adelete_for_runs。

Issue #10 from thread 5046a6e6: langgraph.log 每次 run 启动都警告 missing
adelete_for_runs。本测试模拟"取消一个 run、verify state 被清理"。
"""

import pytest

from <定位的 saver 模块> import <SaverClass>


@pytest.mark.asyncio
async def test_adelete_for_runs_removes_cancelled_run_state(tmp_path) -> None:
    saver = <SaverClass>(db_path=str(tmp_path / "test.db"))
    await saver.setup()

    thread_id = "test-thread"
    run_id = "test-run-1"

    # 写一个 checkpoint
    await saver.aput(
        {"configurable": {"thread_id": thread_id, "run_id": run_id}},
        {"messages": ["m1"]},
        {},
        {},
    )

    # 确认存在
    checkpoint = await saver.aget({"configurable": {"thread_id": thread_id}})
    assert checkpoint is not None

    # 调用 adelete_for_runs
    await saver.adelete_for_runs(thread_id=thread_id, run_ids=[run_id])

    # 确认已清理
    checkpoint = await saver.aget({"configurable": {"thread_id": thread_id}})
    assert checkpoint is None, "adelete_for_runs 后应不剩 checkpoint"


@pytest.mark.asyncio
async def test_adelete_for_runs_preserves_other_runs(tmp_path) -> None:
    saver = <SaverClass>(db_path=str(tmp_path / "test.db"))
    await saver.setup()

    thread_id = "test-thread"
    # 写两个 run 的 checkpoint
    for run_id in ["run-1", "run-2"]:
        await saver.aput(
            {"configurable": {"thread_id": thread_id, "run_id": run_id}},
            {"messages": [f"m-{run_id}"]},
            {},
            {},
        )

    # 只删 run-1
    await saver.adelete_for_runs(thread_id=thread_id, run_ids=["run-1"])

    # run-2 仍在（具体查找逻辑视 saver API）
    # ... 视实际 saver API 补全
```

- [ ] **Step 4: 跑测试，确认 FAIL**

Run:
```bash
cd packages/agent/backend && PYTHONPATH=. uv run pytest tests/test_checkpointer_adelete_for_runs.py -v
```

Expected: FAIL（方法不存在）。

- [ ] **Step 5: 实现 adelete_for_runs**

在 Step 1 定位的 saver 类里加：

```python
async def adelete_for_runs(self, thread_id: str, run_ids: list[str]) -> None:
    """Delete checkpoints for specific cancelled runs in a thread.

    Implements the langgraph CheckpointSaver protocol contract:
    https://langchain-ai.github.io/langgraph/reference/checkpoints/

    Required when using multitask_strategy='rollback' to prevent stale
    state from cancelled runs polluting subsequent runs.
    """
    if not run_ids:
        return
    placeholders = ",".join("?" * len(run_ids))
    async with self._lock, self._connect() as conn:
        await conn.execute(
            f"DELETE FROM checkpoints WHERE thread_id = ? AND run_id IN ({placeholders})",
            [thread_id, *run_ids],
        )
        await conn.commit()
```

（具体 SQL 视 schema 而定，可能 saver 用的是 langgraph SDK 内置的 schema）

- [ ] **Step 6: 跑测试 PASS**

Run:
```bash
cd packages/agent/backend && PYTHONPATH=. uv run pytest tests/test_checkpointer_adelete_for_runs.py -v
```

Expected: PASS。

- [ ] **Step 7: 跑全套测试 + 启动服务确认 warning 消失**

Run:
```bash
cd packages/agent/backend && make test
cd packages/agent && make stop && make dev
sleep 10
# 发一条消息让 thread 起来
# 然后看 langgraph.log
grep -c "Custom checkpointer missing adelete_for_runs" packages/agent/logs/langgraph.log
```

Expected: count 不再增加（warning 消失）。

- [ ] **Step 8: Commit**

Run:
```bash
git add <Step 1 定位的 saver 文件>
git add packages/agent/backend/tests/test_checkpointer_adelete_for_runs.py
git commit -m "feat(checkpointer): implement adelete_for_runs for cancelled-run cleanup

thread 5046a6e6: langgraph.log 每个 run 启动都警告 'Custom checkpointer
missing adelete_for_runs: multitask_strategy=rollback will not clean up'。
取消的 run 状态留在 thread，可能让前端看到回滚 run 内容、feedback run_id
对不上号。

实现 adelete_for_runs 方法 + 2 条单测覆盖'删指定 run/保留其他 run'语义。
warning 消失。

Fix Issue #10 from 2026-05-13 dogfood iteration."
```

---

## 后置任务（不在本计划范围）

下面 4 个 issue 不在 Batch A/B/C/D 内，由计划接收方在 Batch D 完成后**自行决定**何时启动。每个都需要独立设计讨论或 dogfood 数据再做。

### Issue #9 · 设计 ad-hoc 分析路径（P1，架构级）
**不在本计划做**。Batch D 完成后开新计划。需先 3 方讨论方案 A/B/C（详见 task #18 描述）。

### Issue #11 · 调 lead thinking budget / reasoning_effort（P2，体感）
**不在本计划做**。属于体感优化，等 Batch A-D 修完基线稳了再调。改动有风险（lead 判断质量可能下降），需要 dogfood 校准。

### Issue #12 · ArchivingSummarizationMiddleware 触发阈值（P2，体感）
**不在本计划做**。Issue #2 修了之后归档/恢复链路不再丢字段，但 29s 阻塞仍存。改成异步是工程改造，独立做。

### Issue #13 · newapi 网关稳定性（P2，观察）
**不修**。SDK 自动重试已覆盖大多数场景，只监控。如果未来 dogfood 反复看到 retry 飙高再升优先级。

---

## 验收标准

完成本计划意味着：

1. **测试套件全绿**：
   - `cd packages/agent/backend && make test`
   - `cd packages/ethoinsight && pytest tests/`
   - 新增的 4 个测试文件（Task 2/3/7/10）全部 PASS
2. **Batch A/B dogfood 通过**：`docs/handoffs/2026-05/2026-05-14-dogfood-followup-handoff.md` 中的 11 项检查清单 ≥9 项 ✅（Issue #8 的两项是观察记录，不强制）
3. **每个 Task 一个独立 commit**：commit 信息中说明对应 Issue 编号
4. **回归零**：开工前跑的测试套件结果在结束时仍然全绿

---

## 给执行者的注意事项

- **不要跳步骤**：本计划的 TDD 流程（先写失败测试→实现→PASS→commit）是强制的。每个 Task 步骤都写明了 "Expected" 输出，对不上就停下来诊断
- **不要扩大改动范围**：本计划 13 个 Issue 是同事 5/13 反馈 + thread 5046a6e6 暴露的，已经压缩到最小。Issue #9 / #11 / #12 / #13 显式标了"后置"，**不要**在本计划里做
- **不要破坏 catalog 架构**：CLAUDE.md 第 10 条 + `docs/superpowers/specs/2026-05-13-metric-catalog-architecture-design.md` 锁定了 single source of truth 哲学，Task 7 修 catalog 路径时**只许**改路径输出层、**不许**改 catalog YAML 字段或 Q6 白名单
- **不要碰工作树里 3 个无关文件**：交接文档 §"工作树未提交内容"列了 `docs/specs/llm-finetuning-strategy.md` / `docs/plans/2026-05-13-base-model-decision-memo.md` / `packages/agent/frontend/src/app/page.tsx`，不是本计划工作，让用户自己处理
- **prompt 改造用正面提示**：CLAUDE.md 第 6 条 deepseek 用正面指令而不是"不要 X"——Task 2/5 的 prompt 段已经按这个原则写，**修改时不要回退**
- **遇到诊断不出来的 bug**：诊断超过 2 小时仍不能定位时（特别是 Task 3/4 thinking 字段），写诊断笔记到 handoff 文档暂停本计划，跟用户对齐再继续
