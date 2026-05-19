# DeerFlow 上游同步 Plan B：剩余 13 个 commit（含 DynamicContextMiddleware 重头戏）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **前置硬约束**：本 plan 假设 Plan A（`2026-05-15-deerflow-upstream-sync-plan-A-safe-batch.md`）的全部 11 个 commit 已经合入 dev、make test 全绿、dogfood 验证通过。如果 Plan A 没完成，**不要开始 Plan B**。

**Goal:** 把 deerflow 上游 2026-05-08 ~ 2026-05-14 之间 Plan A 没碰的 13 个 commit 处理掉。其中 4 个组成"DynamicContextMiddleware 链"（高价值 prefix-cache 优化，3 个修复跟着主 feature 走）、5 个独立中风险修复，4 个建议跳过 / 单独评估。

**Architecture:** 分两个 phase。Phase 1 是 DynamicContextMiddleware 链（4 个 commit 当作一个事务做：先合主 feature `c1b7f1d1`，再叠加 3 个 fix），核心是改 `lead_agent/prompt.py`（受保护 + 全中文调度规则）。Phase 2 是 9 个独立 commit，每个一个 task，含明确"是否要做 / 如何做 / 风险点 / 跳过理由"。

**Tech Stack:** git surgical merge、pytest、ruff、本地 harness、上游 `deerflow/main`

**基线：** 本地 dev HEAD = Plan A 完成后的 dev tip，上游 `deerflow/main` = fetch 后取最新。

**受保护文件列表（同 Plan A）：**

```
agents/lead_agent/prompt.py        ← Phase 1 重点
agents/lead_agent/agent.py         ← Phase 1
agents/middlewares/llm_error_handling_middleware.py
agents/middlewares/thread_data_middleware.py
agents/thread_state.py
config/paths.py
mcp/tools.py                       ← Phase 2 Task B-2.5
sandbox/tools.py                   ← Phase 2 Task B-2.6
sandbox/sandbox.py
sandbox/local/local_sandbox.py
subagents/builtins/__init__.py
subagents/executor.py
tools/builtins/task_tool.py        ← Phase 2 Task B-2.9 会动
```

路径前缀均为 `packages/agent/backend/packages/harness/deerflow/`。

---

## 全局准备

### Step P.1：建独立 worktree（与 Plan A 完全隔离）

```bash
cd /home/wangqiuyang/noldus-insight
git fetch deerflow main
git worktree add .claude/worktrees/deerflow-sync-plan-B -b sync/deerflow-plan-B dev
cd .claude/worktrees/deerflow-sync-plan-B
git status --short  # 期望: 空
git log -1 --format='%H %s'  # 期望: Plan A 最后一个 commit 或 dev tip
```

### Step P.2：建 progress 文档

```bash
mkdir -p docs/handoffs/2026-05
touch docs/handoffs/2026-05/2026-05-15-deerflow-sync-plan-B-progress.md
```

```markdown
# 2026-05-15 DeerFlow Sync Plan B 进度

## Phase 1: DynamicContextMiddleware 链（4 commit 视作一个事务）

| Task | Commit | 状态 | commit hash | 备注 |
|---|---|---|---|---|
| B-1.1 | c1b7f1d1 | ⏳ | | 主 feature，改 prompt.py + agent.py |
| B-1.2 | 08ee7ade | ⏳ | | 删重复定义 |
| B-1.3 | 881ff712 | ⏳ | | summarization 保持 context |
| B-1.4 | f76e4e35 | ⏳ | | title middleware 修复 |

## Phase 2: 独立 commit

| Task | Commit | 状态 | 决策 | commit hash | 备注 |
|---|---|---|---|---|---|
| B-2.1 | de253e4a | ⏳ | TODO | | model_name 链路 |
| B-2.2 | 68d8caec | ⏳ | TODO | | update_agent user_id |
| B-2.3 | 94da8f67 | ⏳ | TODO | | uv extras + serve.sh |
| B-2.4 | bedbf229 | ⏳ | TODO | | mcp sync wrapper |
| B-2.5 | 30a58462 | ⏳ | TODO | | write_file append |
| B-2.6 | 0d1053ca | ⏳ | SKIP? | | Windows uploads |
| B-2.7 | 5127f08e | ⏳ | SKIP? | | 默认开 token usage |
| B-2.8 | 2b1fcb3e | ⏳ | SKIP | | 删 max_turns，跟本地冲突 |
| B-2.9 | 1c96a6af | ⏳ | EVAL | | bootstrap user scope |
```

### Step P.3：跑基线测试

```bash
cd packages/agent/backend
source .venv/bin/activate
make test 2>&1 | tail -20
make lint 2>&1 | tail -10
```

期望全绿。

---

# Phase 1: DynamicContextMiddleware 链（4 commit）

## 重要说明

`c1b7f1d1` 是本 plan 最难、价值最高的 commit。它改了：

- 新建 `dynamic_context_middleware.py`（193 行）
- 改 🛡 `lead_agent/prompt.py` 16 行 — **本地这文件全中文调度规则 + 4 个 ethoinsight subagent 描述 + Gate 反问机制**
- 改 🛡 `lead_agent/agent.py` 6 行 — **本地这文件含 ArchivingSummarizationMiddleware / ThinkTagMiddleware / TrainingDataMiddleware / GateEnforcementMiddleware 等 Noldus 中间件链顺序**
- 改 `token_usage_middleware.py` 11 行
- 改 `models/claude_provider.py` 4 行

**收益**：让 system_prompt 变成"静态前缀 + 动态尾部"形式，prefix cache 命中率提升、长会话 token 成本显著下降。

**风险**：改 prompt.py / agent.py 是高侵入操作。每一处改动必须人工判断"是上游加的还是 Noldus 的"，不能直接覆盖。

**单事务**：Phase 1 的 4 个 commit 应被视作一个原子事务——只有把全部 4 个合完且 make test 全绿才算完成。中途测试红了可以单 task commit，但**不能只合 c1b7f1d1 不合 3 个 fix 就停下**（fix 是补主 feature 的 bug，少合 fix 会让线上一直带 bug）。

## Task B-1.1: `c1b7f1d1` — DynamicContextMiddleware 主 feature

**上游 PR**：#2801 `feat: static system prompt with DynamicContextMiddleware for prefix-cache optimization`

**Files：**
- Add: `packages/agent/backend/packages/harness/deerflow/agents/middlewares/dynamic_context_middleware.py`（新建 193 行）
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py`（🛡 6 行）
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`（🛡 16 行）
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/middlewares/token_usage_middleware.py`（11 行）
- Modify: `packages/agent/backend/packages/harness/deerflow/models/claude_provider.py`（4 行）
- Add: `packages/agent/backend/tests/test_dynamic_context_middleware.py`（新建 312 行）
- Modify: `packages/agent/backend/tests/test_token_usage_middleware.py`（77 行新增）
- Modify: `packages/agent/backend/tests/test_csrf_middleware.py`（16 行）

- [ ] **Step B-1.1.1：深度阅读上游 c1b7f1d1 整体改动**

```bash
git show c1b7f1d1 --stat
git show c1b7f1d1 -- backend/packages/harness/deerflow/agents/middlewares/dynamic_context_middleware.py
git show c1b7f1d1 -- backend/packages/harness/deerflow/agents/lead_agent/
git show c1b7f1d1 -- backend/packages/harness/deerflow/models/claude_provider.py
```

理解整个 feature 的设计意图：

1. `dynamic_context_middleware.py` 拦截 system_prompt，把"动态部分（如时间戳、用户上下文）"剥离出来作为 reminder message 放在用户消息流里
2. `lead_agent/agent.py` 把该 middleware 加入中间件链
3. `lead_agent/prompt.py` 改的是 system_prompt 模板，把动态字段（如 `{current_time}`）从模板里删掉
4. `claude_provider.py` 改的是 Claude API 调用层（可能加 `cache_control` 标记）
5. `token_usage_middleware.py` 配合统计 cache 命中数据

- [ ] **Step B-1.1.2：识别本地受保护文件的所有 Noldus 改动**

```bash
git diff f0dd8cb HEAD -- packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py > /tmp/local-noldus-prompt-diff.txt
git diff f0dd8cb HEAD -- packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py > /tmp/local-noldus-agent-diff.txt
wc -l /tmp/local-noldus-*.txt
```

仔细读 `/tmp/local-noldus-prompt-diff.txt`，列出所有 Noldus 改动：
- 中文调度规则
- Gate 反问机制
- 4 个 ethoinsight subagent 描述（code-executor、data-analyst、report-writer、knowledge-assistant）
- 实验上下文 set_experiment_paradigm 相关
- 等等

把列表写进 progress 表的备注列，作为 surgical merge 时的"必须保留"清单。

- [ ] **Step B-1.1.3：整文件拉 `dynamic_context_middleware.py`**

```bash
git show deerflow/main:backend/packages/harness/deerflow/agents/middlewares/dynamic_context_middleware.py > packages/agent/backend/packages/harness/deerflow/agents/middlewares/dynamic_context_middleware.py
```

- [ ] **Step B-1.1.4：surgical merge 🛡 `lead_agent/agent.py`（6 行）**

上游 6 行改动一般是：

1. import DynamicContextMiddleware
2. 把它加入 middleware list 的合适位置

读上游 c1b7f1d1 在 agent.py 的具体 hunk：

```bash
git show c1b7f1d1 -- backend/packages/harness/deerflow/agents/lead_agent/agent.py
```

然后手工把 DynamicContextMiddleware import + 加入本地 middleware list。位置很关键：

**位置规则**：DynamicContextMiddleware 应在 `ArchivingSummarizationMiddleware` 之前（先剥离动态字段再压缩历史），在 `GateEnforcementMiddleware` 之后（先 Gate 反问、再剥离）。如果不确定，参考上游 agent.py 中 middleware 的相对顺序。

- [ ] **Step B-1.1.5：surgical merge 🛡 `lead_agent/prompt.py`（16 行 — 本 task 最难步骤）**

上游 16 行改动主要是：

1. 把 `system_prompt` 模板里的动态占位符（如 `{current_time}`、`{user_id}`）替换为 reminder 文案
2. 在文件末尾加 `STATIC_SYSTEM_PROMPT_PREFIX` / `DYNAMIC_CONTEXT_TEMPLATE` 之类的常量

读上游 hunk：

```bash
git show c1b7f1d1 -- backend/packages/harness/deerflow/agents/lead_agent/prompt.py
```

surgical merge 策略：

- 上游删除的"动态字段引用"如果在 Noldus 中文 prompt 里**也有**：跟着删（同步行为）
- 上游删除的"动态字段引用"如果 Noldus prompt **没有**（中文版本写法不同）：保留 Noldus 写法，但注意该字段已经被 DynamicContextMiddleware 走另一路注入，**Noldus 模板里也不能再写死动态值**
- 上游新增的"静态 prefix / 动态模板常量"：照搬到本地 prompt.py 文件末尾，但常量内容需要本地化（如果是中文 prompt）

**如果合并到一半不确定 Noldus prompt 哪段该跟、哪段不该跟**：STOP，把决策点写进 progress，让用户拍板。

- [ ] **Step B-1.1.6：surgical merge `token_usage_middleware.py`（11 行）**

合入 token usage 配合 cache 统计的逻辑。该文件本地 ⚠ DIFFERS 但不在受保护列表，应是叠加在 Plan A Task 8/9 之后。

- [ ] **Step B-1.1.7：surgical merge `models/claude_provider.py`（4 行）**

合入 claude_provider 的改动（应该是把 system_prompt 标 `cache_control: ephemeral` 之类的小动作）。

- [ ] **Step B-1.1.8：合入 3 个测试**

```bash
git show deerflow/main:backend/tests/test_dynamic_context_middleware.py > packages/agent/backend/tests/test_dynamic_context_middleware.py
git show c1b7f1d1 -- backend/tests/test_token_usage_middleware.py | head -100  # 77 行 patch
git show c1b7f1d1 -- backend/tests/test_csrf_middleware.py  # 16 行 patch
```

手工合入 test_token_usage_middleware.py / test_csrf_middleware.py 的 patch；test_dynamic_context_middleware.py 整文件拉。

- [ ] **Step B-1.1.9：跑测试**

```bash
cd packages/agent/backend
pytest tests/test_dynamic_context_middleware.py tests/test_token_usage_middleware.py tests/test_csrf_middleware.py -v 2>&1 | tail -40
make test 2>&1 | tail -30
make lint 2>&1 | tail -10
```

期望全绿。如果某个 test 红了：

- 看 traceback：是 prompt.py 改动不一致？是 middleware 顺序？是 cache_control 没标？
- 修一处再跑一次
- 实在卡住，STOP 上报

- [ ] **Step B-1.1.10：dogfood mini 验证（不退出 worktree）**

```bash
cd /home/wangqiuyang/noldus-insight/.claude/worktrees/deerflow-sync-plan-B/packages/agent
make stop && make dev
```

等服务起来后用 Playwright 或浏览器跑一次最简对话（"你好"），看：

- Lead agent 能否正常应答
- 后端日志里能看到 DynamicContextMiddleware 工作（动态字段被剥离到 reminder）
- 没有 Pydantic warning / KeyError / NameError

如果跑通：继续；如果跑挂：回头看 surgical merge 漏哪了。

```bash
make stop
```

- [ ] **Step B-1.1.11：commit**

```bash
cd /home/wangqiuyang/noldus-insight/.claude/worktrees/deerflow-sync-plan-B
git add packages/agent/backend/packages/harness/deerflow/agents/middlewares/dynamic_context_middleware.py \
        packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py \
        packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py \
        packages/agent/backend/packages/harness/deerflow/agents/middlewares/token_usage_middleware.py \
        packages/agent/backend/packages/harness/deerflow/models/claude_provider.py \
        packages/agent/backend/tests/test_dynamic_context_middleware.py \
        packages/agent/backend/tests/test_token_usage_middleware.py \
        packages/agent/backend/tests/test_csrf_middleware.py
git commit -m "$(cat <<'EOF'
sync(deerflow): DynamicContextMiddleware 静态 system_prompt for prefix-cache（上游 #2801 / c1b7f1d1）

把 system_prompt 拆成"静态 prefix + 动态 reminder"，动态字段（时间、用户）走 reminder
注入用户消息流，让 Anthropic prefix cache 命中率上去，长会话 token 成本显著下降。
本地 lead_agent prompt.py / agent.py 含大量 Noldus 定制，已做 surgical merge 保留所有
中文调度规则、Gate 反问机制、4 个 ethoinsight subagent 描述、Noldus 中间件链。

上游 PR: https://github.com/bytedance/deer-flow/pull/2801

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step B-1.1.12：更新 progress**

---

## Task B-1.2: `08ee7ade` — 删除 dynamic_context_middleware 重复定义

**上游 PR**：#2837 `fix(lint): remove duplicate is_dynamic_context_reminder definition`

**改动范围**：`dynamic_context_middleware.py` -5 行（删一个重复函数定义）。

**Files：** Modify: `packages/agent/backend/packages/harness/deerflow/agents/middlewares/dynamic_context_middleware.py`

- [ ] **Step B-1.2.1：看上游 diff**

```bash
git show 08ee7ade
```

读改动：上游删了一个重复的 `is_dynamic_context_reminder` 函数定义（lint warning）。

- [ ] **Step B-1.2.2：在本地 dynamic_context_middleware.py 找重复定义并删除**

如果 Task B-1.1 拉的版本里有这个重复定义（应该有），按 patch 删掉。如果上游本来就只有一份（说明 c1b7f1d1 之后被立即修复），跳过。

- [ ] **Step B-1.2.3：跑测试**

```bash
cd packages/agent/backend
pytest tests/test_dynamic_context_middleware.py -v 2>&1 | tail -20
make lint 2>&1 | tail -10  # 验证 lint warning 消失
```

- [ ] **Step B-1.2.4：commit**

```bash
cd /home/wangqiuyang/noldus-insight/.claude/worktrees/deerflow-sync-plan-B
git add packages/agent/backend/packages/harness/deerflow/agents/middlewares/dynamic_context_middleware.py
git commit -m "$(cat <<'EOF'
sync(deerflow): 删除 dynamic_context_middleware 重复函数定义（上游 #2837 / 08ee7ade）

is_dynamic_context_reminder 在 c1b7f1d1 引入时不慎重复定义两次，修。
上游 PR: https://github.com/bytedance/deer-flow/pull/2837

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step B-1.2.5：更新 progress**

---

## Task B-1.3: `881ff712` — 保持 dynamic context 跨 summarization

**上游 PR**：#2823 `fix(harness): preserve dynamic context across summarization`

**改动范围**：`dynamic_context_middleware.py` 15 行 + `summarization_middleware.py` 21 行 + 2 个测试。

**Files：**
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/middlewares/dynamic_context_middleware.py`（15 行）
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/middlewares/summarization_middleware.py`（21 行）
- Modify: `packages/agent/backend/tests/test_dynamic_context_middleware.py`（24 行）
- Modify: `packages/agent/backend/tests/test_summarization_middleware.py`（42 行）

- [ ] **Step B-1.3.1：看上游 diff**

```bash
git show 881ff712 --stat
git show 881ff712 -- backend/packages/harness/deerflow/agents/middlewares/
git show 881ff712 -- backend/tests/
```

- [ ] **Step B-1.3.2：surgical merge 两个 middleware**

`dynamic_context_middleware.py`：加 15 行处理 summarization 场景。
`summarization_middleware.py`：加 21 行配合（保留 dynamic context reminder 不被压缩掉）。

注意本地 `ArchivingSummarizationMiddleware` 是 Noldus 自己的(在 `archiving_summarization.py`)；本 commit 改的是上游 `summarization_middleware.py`，两个不同文件，互不冲突。但要确认本地是否真的还在用上游 summarization_middleware.py（应该在用，因为 dev tip 上 dev/summarization 与上游 DIFFERS 但存在）。

- [ ] **Step B-1.3.3：合入两个测试**

```bash
git show 881ff712 -- backend/tests/test_dynamic_context_middleware.py
git show 881ff712 -- backend/tests/test_summarization_middleware.py
```

- [ ] **Step B-1.3.4：跑测试**

```bash
cd packages/agent/backend
pytest tests/test_dynamic_context_middleware.py tests/test_summarization_middleware.py -v 2>&1 | tail -30
make test 2>&1 | tail -30
make lint 2>&1 | tail -10
```

- [ ] **Step B-1.3.5：commit**

```bash
cd /home/wangqiuyang/noldus-insight/.claude/worktrees/deerflow-sync-plan-B
git add packages/agent/backend/packages/harness/deerflow/agents/middlewares/dynamic_context_middleware.py \
        packages/agent/backend/packages/harness/deerflow/agents/middlewares/summarization_middleware.py \
        packages/agent/backend/tests/test_dynamic_context_middleware.py \
        packages/agent/backend/tests/test_summarization_middleware.py
git commit -m "$(cat <<'EOF'
sync(deerflow): summarization 保留 dynamic context reminder（上游 #2823 / 881ff712）

长会话 summarization 压缩时，dynamic context reminder 不应被压掉，否则下一轮就丢上下文。
上游 PR: https://github.com/bytedance/deer-flow/pull/2823

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step B-1.3.6：更新 progress**

---

## Task B-1.4: `f76e4e35` — title middleware 与 dynamic context reminder 配合

**上游 PR**：#2830 `fix title generation with dynamic context reminder`

**改动范围**：`dynamic_context_middleware.py` 7 行 + `title_middleware.py` 9 行 + 36 行新测试。

**Files：**
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/middlewares/dynamic_context_middleware.py`（7 行）
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/middlewares/title_middleware.py`（9 行）
- Modify: `packages/agent/backend/tests/test_title_middleware_core_logic.py`（36 行）

- [ ] **Step B-1.4.1：看上游 diff**

```bash
git show f76e4e35 --stat
git show f76e4e35 -- backend/packages/harness/deerflow/agents/middlewares/
git show f76e4e35 -- backend/tests/test_title_middleware_core_logic.py
```

- [ ] **Step B-1.4.2：surgical merge**

`title_middleware.py` 加 9 行（生成 title 时跳过 dynamic context reminder 消息）。`dynamic_context_middleware.py` 加 7 行（标记 reminder 让 title middleware 能识别）。

- [ ] **Step B-1.4.3：合入测试**

```bash
git show f76e4e35 -- backend/tests/test_title_middleware_core_logic.py
```

- [ ] **Step B-1.4.4：跑测试**

```bash
cd packages/agent/backend
pytest tests/test_title_middleware_core_logic.py tests/test_dynamic_context_middleware.py -v 2>&1 | tail -30
make test 2>&1 | tail -30
make lint 2>&1 | tail -10
```

- [ ] **Step B-1.4.5：commit**

```bash
cd /home/wangqiuyang/noldus-insight/.claude/worktrees/deerflow-sync-plan-B
git add packages/agent/backend/packages/harness/deerflow/agents/middlewares/dynamic_context_middleware.py \
        packages/agent/backend/packages/harness/deerflow/agents/middlewares/title_middleware.py \
        packages/agent/backend/tests/test_title_middleware_core_logic.py
git commit -m "$(cat <<'EOF'
sync(deerflow): title middleware 跳过 dynamic context reminder（上游 #2830 / f76e4e35）

生成对话 title 时不应把 dynamic context reminder 当成用户输入算进去，否则 title 会错。
上游 PR: https://github.com/bytedance/deer-flow/pull/2830

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step B-1.4.6：更新 progress**

---

## Phase 1 收尾

### Step B-1.F.1：跑完整 dogfood 验证

```bash
cd /home/wangqiuyang/noldus-insight/.claude/worktrees/deerflow-sync-plan-B/packages/agent
make stop && make dev
```

用 Playwright 或浏览器跑一次完整 EPM 分析。重点验证：

- Lead agent 长会话不丢上下文（DynamicContextMiddleware 工作）
- summarization 触发后下一轮仍能用当前时间 / 用户上下文
- title 生成正常（不被 reminder 干扰）
- 后端日志 / token usage 能看到 prefix cache 命中率 > 0

```bash
make stop
```

如果发现回归：回到 Task B-1.1 ~ B-1.4 中的某一个 task，单独 revert 那个 commit 并修，**不要 force push、不要跳过单测**。

---

# Phase 2: 9 个独立 commit

## Task B-2.1: `de253e4a` — model_name 从 gateway 一路传到 SQLite

**上游 PR**：#2775 `feat(run): Propagates model_name from the gateway request through the runtime and persistence stack to the SQLite database`

**决策**：**做**。原因：与本地 training-data 飞轮（`packages/agent/backend/.deer-flow/training-data/auto-collected/` + `/api/threads/{tid}/runs/{rid}/feedback`）联动有用——飞轮采集训练数据时能知道每条样本用的哪个 model。

**Files：**
- Modify: `packages/agent/backend/app/gateway/services.py`（19 行 — 本地 Noldus 应用层）
- Modify: `packages/agent/backend/packages/harness/deerflow/persistence/run/sql.py`（14 行）
- Modify: `packages/agent/backend/packages/harness/deerflow/runtime/runs/manager.py`（16 行）
- Modify: `packages/agent/backend/packages/harness/deerflow/runtime/runs/store/base.py`（1 行）
- Modify: `packages/agent/backend/packages/harness/deerflow/runtime/runs/store/memory.py`（2 行）
- Modify: `packages/agent/backend/packages/harness/deerflow/runtime/runs/worker.py`（11 行）
- Modify: `packages/agent/backend/tests/test_run_manager.py`（51 行）
- Modify: `packages/agent/backend/tests/test_run_repository.py`（29 行）

- [ ] **Step B-2.1.1：看上游 diff**

```bash
git show de253e4a --stat
git show de253e4a -- backend/app/gateway/services.py
git show de253e4a -- backend/packages/harness/deerflow/
```

- [ ] **Step B-2.1.2：⚠ 重点检查 `services.py` 本地 Noldus 改动**

```bash
git log --oneline -- packages/agent/backend/app/gateway/services.py | head -10
git diff f0dd8cb HEAD -- packages/agent/backend/app/gateway/services.py
```

如果本地 services.py 有 better-auth / @require_permission / 多用户隔离改动，surgical merge 时务必保留。

- [ ] **Step B-2.1.3：surgical merge 6 个文件**

按 commit diff 逐个合入。schema 变化：runs 表加 `model_name` 列、`RunStore` 接口加 `model_name` 参数、worker 调用时传入。

- [ ] **Step B-2.1.4：检查是否需要 Alembic migration**

```bash
ls packages/agent/backend/packages/harness/deerflow/persistence/migrations/versions/
```

上游 PR 内是否包含 Alembic migration？看：

```bash
git show de253e4a -- backend/packages/harness/deerflow/persistence/migrations/
```

如果有，整文件拉。如果没有，**STOP** 上报——加列必须有 migration，否则线上数据库不会变。

- [ ] **Step B-2.1.5：合入测试**

```bash
git show de253e4a -- backend/tests/test_run_manager.py
git show de253e4a -- backend/tests/test_run_repository.py
```

注意 test_run_repository.py 在 Plan A Task 1 (2a1ac06b) 已被改过，本次叠加。

- [ ] **Step B-2.1.6：跑测试**

```bash
cd packages/agent/backend
pytest tests/test_run_manager.py tests/test_run_repository.py -v 2>&1 | tail -30
make test 2>&1 | tail -30
make lint 2>&1 | tail -10
```

- [ ] **Step B-2.1.7：commit**

```bash
cd /home/wangqiuyang/noldus-insight/.claude/worktrees/deerflow-sync-plan-B
git add packages/agent/backend/app/gateway/services.py \
        packages/agent/backend/packages/harness/deerflow/persistence/run/sql.py \
        packages/agent/backend/packages/harness/deerflow/runtime/runs/manager.py \
        packages/agent/backend/packages/harness/deerflow/runtime/runs/store/base.py \
        packages/agent/backend/packages/harness/deerflow/runtime/runs/store/memory.py \
        packages/agent/backend/packages/harness/deerflow/runtime/runs/worker.py \
        packages/agent/backend/tests/test_run_manager.py \
        packages/agent/backend/tests/test_run_repository.py
# 如果有 migration，也 add 进去
git commit -m "$(cat <<'EOF'
sync(deerflow): model_name 从 gateway 传到 SQLite（上游 #2775 / de253e4a）

把 model_name 从 gateway request 一路传到 run worker 再写入 SQLite，
飞轮采集训练数据时能知道每条样本用的是哪个模型。
上游 PR: https://github.com/bytedance/deer-flow/pull/2775

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step B-2.1.8：更新 progress**

---

## Task B-2.2: `68d8caec` — update_agent 走 runtime.context user_id

**上游 PR**：#2867 `fix(agents): make update_agent honor runtime.context user_id like setup_agent`

**决策**：**先 EVAL，再决定**。本地 `update_agent_tool.py` 现在不存在（Plan A Task 11 已确认）。本 commit 假设有这个 tool，意味着：

- 选项 A：把本 commit + Plan A Task 11 跳过的 `update_agent_tool.py` 整文件拉同时引入这个 tool，并接入 lead agent 的 builtin tools 列表
- 选项 B：本地不打算引入这个 tool（v0.1 单团队多用户，可能不需要 agent 自我修改），整个 commit 跳过

**Files（如果选 A）：**
- Modify: `packages/agent/backend/packages/harness/deerflow/runtime/user_context.py`（28 行）
- Modify: `packages/agent/backend/packages/harness/deerflow/tools/builtins/setup_agent_tool.py`（11 行）
- Add: `packages/agent/backend/packages/harness/deerflow/tools/builtins/update_agent_tool.py`（整文件拉）
- Add: 3 个 e2e 测试

- [ ] **Step B-2.2.1：跟用户确认决策**

发消息给主会话：

> Task B-2.2: 是否要引入 `update_agent_tool`（agent 自我修改能力，spec §"复用 deerflow 现成功能"提到过、v0.1 后才启用）？
> - A：引入（合 #2867 + Plan A 跳过的 #2774 中的 update_agent_tool.py + 接入 builtins）
> - B：跳过（本 commit 整个跳过）

等用户答复。**不要自己决定**。

- [ ] **Step B-2.2.2：如选 A，按完整流程合入**

（具体步骤参考 Plan A Task 8 的格式，这里不展开）

- [ ] **Step B-2.2.3：如选 B，在 progress 表标 SKIP 并附理由**

---

## Task B-2.3: `94da8f67` — uv extras + dev entrypoint 改造

**上游 PR**：#2754 + #2767 `fix(scripts): preserve uv extras across make dev restarts`

**决策**：**EVAL**。改动范围超出 harness：

- `persistence/engine.py` 9 行（harness）
- `scripts/serve.sh` 34 行 — 跟用户工作树里那个未 commit 的 serve.sh 30→90s 改动**直接冲突**
- `scripts/detect_uv_extras.py` 180 行（新脚本）
- `docker/dev-entrypoint.sh` 85 行（新脚本）
- `docker/docker-compose-dev.yaml` 9 行
- `config.example.yaml` 22 行
- 2 个测试

**冲突点**：本地 serve.sh 已有未 commit 改动（gateway timeout 30→90s，疑似 G4 方案 C agent 跑 dogfood 时改的）。要先解决这个冲突再合本 commit。

- [ ] **Step B-2.3.1：先看本地 serve.sh 未 commit 改动**

```bash
cd /home/wangqiuyang/noldus-insight
git diff packages/agent/scripts/serve.sh | head -40
git log --oneline packages/agent/scripts/serve.sh | head -5
```

- [ ] **Step B-2.3.2：跟用户确认 serve.sh 决策**

发消息给主会话：

> Task B-2.3: 本地 packages/agent/scripts/serve.sh 有未 commit 改动（gateway timeout 30→90s），上游 #2754 也改 serve.sh（preserve uv extras 逻辑）。三种处理：
> - A：先 commit 本地 30→90s 改动（用 G4 方案 C agent 名义），再合上游 #2754
> - B：放弃本地 30→90s 改动（接受上游 serve.sh），重新评估 gateway 启动慢的根因
> - C：本 commit 整个跳过（保留本地 serve.sh）

等用户答复。

- [ ] **Step B-2.3.3：按用户决策执行**

---

## Task B-2.4: `bedbf229` — async-only config tools 用 sync wrapper

**上游 PR**：#2878 `fix(harness): wrap async-only config tools for sync client execution`

**决策**：**做**。改动包含 🛡 `mcp/tools.py`（本地有 4096 截断），但本 commit 改动是从 mcp/tools.py 提取 helper 到新 `tools/sync.py`，**减负而非加负**，对本地 4096 截断兼容性好。

**Files：**
- Modify: `packages/agent/backend/packages/harness/deerflow/mcp/tools.py`（🛡 -45 行，主要是删/搬运）
- Modify: `packages/agent/backend/packages/harness/deerflow/tools/skill_manage_tool.py`（4 行）
- Add: `packages/agent/backend/packages/harness/deerflow/tools/sync.py`（新 36 行）
- Modify: `packages/agent/backend/packages/harness/deerflow/tools/tools.py`（10 行）
- Modify: `packages/agent/backend/tests/test_mcp_sync_wrapper.py`（16 行）
- Modify: `packages/agent/backend/tests/test_tool_deduplication.py`（42 行）

- [ ] **Step B-2.4.1：看上游 diff**

```bash
git show bedbf229 --stat
git show bedbf229 -- backend/packages/harness/deerflow/mcp/tools.py
git show bedbf229 -- backend/packages/harness/deerflow/tools/sync.py
```

- [ ] **Step B-2.4.2：surgical merge 🛡 `mcp/tools.py`**

上游主要是删 45 行（把 sync wrapper 逻辑搬到 tools/sync.py）。**本地 Noldus 4096 截断改动必须保留**——它跟搬走的 sync wrapper 逻辑应该正交。

- [ ] **Step B-2.4.3：整文件拉 `tools/sync.py`**

```bash
git show deerflow/main:backend/packages/harness/deerflow/tools/sync.py > packages/agent/backend/packages/harness/deerflow/tools/sync.py
```

- [ ] **Step B-2.4.4：surgical merge 2 个非受保护文件**

`skill_manage_tool.py` + `tools.py` 改动小，按 patch 合。

- [ ] **Step B-2.4.5：合入测试 + 跑测试**

```bash
git show bedbf229 -- backend/tests/test_mcp_sync_wrapper.py
git show bedbf229 -- backend/tests/test_tool_deduplication.py

cd packages/agent/backend
pytest tests/test_mcp_sync_wrapper.py tests/test_tool_deduplication.py -v 2>&1 | tail -30
make test 2>&1 | tail -30
make lint 2>&1 | tail -10
```

注意 test_tool_deduplication.py 在 Plan A Task 3 (f1a0ab69) 已被改过，本次叠加。

- [ ] **Step B-2.4.6：commit**

```bash
cd /home/wangqiuyang/noldus-insight/.claude/worktrees/deerflow-sync-plan-B
git add packages/agent/backend/packages/harness/deerflow/mcp/tools.py \
        packages/agent/backend/packages/harness/deerflow/tools/skill_manage_tool.py \
        packages/agent/backend/packages/harness/deerflow/tools/sync.py \
        packages/agent/backend/packages/harness/deerflow/tools/tools.py \
        packages/agent/backend/tests/test_mcp_sync_wrapper.py \
        packages/agent/backend/tests/test_tool_deduplication.py
git commit -m "$(cat <<'EOF'
sync(deerflow): async-only config tools 走 sync wrapper（上游 #2878 / bedbf229）

把 sync wrapper 逻辑从 mcp/tools.py 抽到 tools/sync.py，
保留本地 mcp/tools.py 的 4096 字符截断 Noldus 改动。
上游 PR: https://github.com/bytedance/deer-flow/pull/2878

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step B-2.4.7：更新 progress**

---

## Task B-2.5: `30a58462` — write_file --append 在 model-facing schema 里可见

**上游 PR**：#2843 `fix(tools): make write_file append discoverable in model-facing schema`

**决策**：**做**。让模型能看到 `write_file` 有 `--append` 选项，避免重复 `read_file → modify → write_file` 整文件覆盖。

**改动小（3-2 行级）**，但触及 🛡 `sandbox/tools.py`、跟 Plan A Task 11 (7de9b582) 在 sandbox/tools.py 上有叠加。

**Files：**
- Modify: `packages/agent/backend/packages/harness/deerflow/sandbox/tools.py`（🛡 3 行）
- Modify: `packages/agent/backend/packages/harness/deerflow/tools/builtins/setup_agent_tool.py`（2 行）
- Modify: `packages/agent/backend/packages/harness/deerflow/tools/builtins/update_agent_tool.py`（2 行 — **若 Task B-2.2 选 A 才合**）
- Modify: `packages/agent/backend/tests/test_tool_args_schema_no_pydantic_warning.py`（17 行 — 在 Plan A Task 11 的测试基础上叠加）

- [ ] **Step B-2.5.1：看上游 diff**

```bash
git show 30a58462
```

- [ ] **Step B-2.5.2：surgical merge 🛡 `sandbox/tools.py`（3 行）**

把 `write_file` tool schema 的 `append: bool = False` 参数描述加进去（模型能看到）。**保留本地 {{shared://}} / extra_env 改动**。

- [ ] **Step B-2.5.3：surgical merge setup_agent_tool.py / update_agent_tool.py（若适用）**

- [ ] **Step B-2.5.4：合入测试 + 跑**

```bash
git show 30a58462 -- backend/tests/test_tool_args_schema_no_pydantic_warning.py

cd packages/agent/backend
pytest tests/test_tool_args_schema_no_pydantic_warning.py -v 2>&1 | tail -20
make test 2>&1 | tail -30
make lint 2>&1 | tail -10
```

- [ ] **Step B-2.5.5：commit**

```bash
cd /home/wangqiuyang/noldus-insight/.claude/worktrees/deerflow-sync-plan-B
# 视 Task B-2.2 决策调整 add 列表
git add packages/agent/backend/packages/harness/deerflow/sandbox/tools.py \
        packages/agent/backend/packages/harness/deerflow/tools/builtins/setup_agent_tool.py \
        packages/agent/backend/tests/test_tool_args_schema_no_pydantic_warning.py
git commit -m "$(cat <<'EOF'
sync(deerflow): write_file --append 在 model-facing schema 里可见（上游 #2843 / 30a58462）

模型现在能看到 write_file 有 --append 选项，避免重复读改写整文件。
上游 PR: https://github.com/bytedance/deer-flow/pull/2843

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step B-2.5.6：更新 progress**

---

## Task B-2.6: `0d1053ca` — Windows symlink-protected uploads

**上游 PR**：#2794 `fix(uploads): add Windows support for safe symlink-protected uploads`

**决策**：**SKIP**。本地不跑 Windows 部署。改 uploads/manager.py 64 行只为 Windows 兼容，对 Linux 本地无价值，并引入 Windows 分支代码增加维护成本。

- [ ] **Step B-2.6.1：在 progress 表标 SKIP**

理由：Linux 部署不需要 Windows symlink 路径处理。如果未来要支持 Windows 客户端单独 evaluate。

无需 commit。

---

## Task B-2.7: `5127f08e` — 默认开启 token usage

**上游 PR**：#2841 `enable token usage by default`

**决策**：**EVAL → 大概率 SKIP**。改动是把 `config/token_usage_config.py` 的默认值翻成 `enabled=True`。本地大概率**已经是 True**（否则 Plan A Task 8/9 的飞轮就没数据），是 noop。

- [ ] **Step B-2.7.1：验证本地是否已经默认开**

```bash
cat packages/agent/backend/packages/harness/deerflow/config/token_usage_config.py | grep -i "enable\|default"
```

如果已经是 True：标 SKIP，理由 "already true locally, no-op"。
如果是 False：合入（按 Plan A 模板）。前端 mdx 文档 + 测试一起合。

- [ ] **Step B-2.7.2：更新 progress**

---

## Task B-2.8: `2b1fcb3e` — 删除 task_tool 的 max_turns 参数

**上游 PR**：#2783 `fix(task): remove max_turns parameter from task tool interface`

**决策**：**SKIP（硬冲突）**。本地正是依赖 max_turns 做 subagent 硬限制（`docs/sop/deerflow-sync-sop.md` 把 task_tool 列为受保护文件就是因为这个）。删 max_turns 会让本地失去防 subagent 失控的能力。

- [ ] **Step B-2.8.1：在 progress 表标 SKIP 并附详细理由**

```markdown
| B-2.8 | 2b1fcb3e | ⏭ SKIP | | 删 max_turns 跟本地硬限制冲突，CLAUDE.md fork 策略 / sync SOP 都列 task_tool.py 为受保护文件就是因为这个。未来要重新设计 subagent 限制机制时再回看 |
```

无需 commit。

---

## Task B-2.9: `1c96a6af` — bootstrap 走 user scope

**上游 PR**：#2784 `fix: keep new agent bootstrap in user scope`

**决策**：**EVAL**。本地是多团队多用户研究助手，bootstrap 走 user scope **理论上对**，但需要确认本地 setup_agent_tool.py 当前的 bootstrap 行为是否已经走 user scope（Tier 4 之后应该已经是）。

**Files：**
- Modify: `packages/agent/backend/app/gateway/services.py`（19 行 — 跟 B-2.1 在同一文件，可能叠加）
- Modify: `packages/agent/backend/packages/harness/deerflow/tools/builtins/setup_agent_tool.py`（9 行）
- Modify: `packages/agent/backend/tests/test_gateway_services.py`（15 行）
- Modify: `packages/agent/backend/tests/test_setup_agent_tool.py`（22 行）
- Modify: `packages/agent/frontend/src/app/workspace/agents/new/page.tsx`（56 行）
- Modify: `packages/agent/frontend/src/core/i18n/locales/{en-US,zh-CN}.ts`（各 2 行）

- [ ] **Step B-2.9.1：跟用户确认决策**

发消息给主会话：

> Task B-2.9: 上游 #2784 让新 agent bootstrap 走 user scope。本地 setup_agent_tool.py 在 Tier 4 之后应该已经走 user scope 了？要不要合？
> - A：合入（让本地行为跟上游对齐，避免未来漂移）
> - B：跳过（本地行为已正确）
> - C：先验证本地 bootstrap 是不是 user scope，再决定

等用户答复。

- [ ] **Step B-2.9.2：按用户决策执行**

---

# Phase 2 收尾

### Step B-2.F.1：跑完整 dogfood 验证

```bash
cd /home/wangqiuyang/noldus-insight/.claude/worktrees/deerflow-sync-plan-B/packages/agent
make stop && make dev
```

Playwright 跑完整 EPM 单只分析 + 验证：
- Plan A 全部收益（token usage header、subagent token bucket、tool_search promotion 等）仍在
- Plan B Phase 1 收益（prefix cache 命中率提高、长会话 dynamic context 不丢、title 生成正常）
- Plan B Phase 2 各 commit 的具体收益（model_name 入库、write_file append 可见、mcp sync wrapper 工作）

### Step B-2.F.2：跑完整测试

```bash
cd packages/agent/backend
make test 2>&1 | tail -50
make lint 2>&1 | tail -20

cd ../frontend
pnpm check 2>&1 | tail -20
```

### Step B-2.F.3：归档进度文档

补全 progress 表，commit。

```bash
git add docs/handoffs/2026-05/2026-05-15-deerflow-sync-plan-B-progress.md
git commit -m "docs: deerflow sync plan B 完成交接（13 commit 处理结果）"
```

### Step B-2.F.4：等用户授权后 merge / push

**⚠ 不要主动 push**。完成后回报主会话：

> Plan B 处理完毕。Phase 1 DynamicContextMiddleware 链 4 个 commit 已落 sync/deerflow-plan-B 分支。Phase 2: 5 个合入、4 个 SKIP（理由见 progress）、其中 N 个根据用户决策跳过。本地 make test + make lint 全绿，dogfood 验证通过。等用户授权 merge。

---

## 风险与中断恢复

### 风险 1：Phase 1 surgical merge `prompt.py` / `agent.py` 时迷失

**症状**：上游 16 行 prompt.py 改动一半看不懂跟本地中文调度规则的对应关系。

**处理**：

1. 不要盲目合
2. 先把上游英文 prompt 的"动态字段引用方式" + 本地中文 prompt 的对应位置 列在 progress 表的备注列
3. 列完后再判断：上游每一行改动属于"删除动态字段（要跟）"还是"加新常量（要本地化照搬）"
4. 仍不清楚的：STOP 上报

### 风险 2：dogfood 端到端验证发现回归

**症状**：Phase 1 跑完 dogfood 后发现 lead agent 不调度、subagent 不响应、Gate 反问失效之一。

**处理**：

1. 用 `git log --oneline` 看 Phase 1 的 4 个 commit
2. 逐个 `git revert` 反向定位（先 revert B-1.4，跑 dogfood 看是否恢复；不行再 revert B-1.3...）
3. 找到回归 commit 后，重新 surgical merge 那一个 commit，重点排查 prompt.py 改动是否动了 Noldus 调度规则
4. 修完重新跑测试 + dogfood

### 风险 3：Plan A 和 Plan B 并行（不应发生）

**不允许**两个 plan 同时执行。Plan B 假设 Plan A 已 merge 进 dev 才开始。如果发现并行：STOP，让用户决定先合谁。

### 中断恢复

中断后：

1. 读 `docs/handoffs/2026-05/2026-05-15-deerflow-sync-plan-B-progress.md`
2. `cd .claude/worktrees/deerflow-sync-plan-B`
3. `git log --oneline | head -20` 看哪些已 commit
4. 从下一个 ⏳ task 接

---

## Plan B 完成后下一步

向主会话回报，并 flag：

- 是否更新 `scripts/sync-deerflow.sh` 加 worktree 兼容性（`.git` 是文件不是目录的场景）+ 重置 baseline tracking（当前因为大量 cherry-pick squash subtree commit 已经失效）
- 是否更新 `CLAUDE.md` 第 11 条（已被第 12 条修正但原文仍在）和 `docs/sop/deerflow-sync-sop.md` 第 145 行（"Tier 4 整 PR 跳过"已过期）
- 是否更新 `docs/handoffs/2026-05/2026-05-15-spec-phase-1-and-g4-dual-track-handoff.md` 的"已完成"段加上"Plan A + B sync 完成"
- 评估上游 5-14 之后新出现的 commit（fetch 一次看有没有新增）是否要追上
