# 2026-06-02 设计 spec — DeerFlow 上游 sync(f9b70713 → 74e3e80c,34 commit delta)

**类型**:可实施版(已核验真实 delta、受保护文件影响面、subtree squash 历史坑、Tier4 风险已消解;实施前 `git fetch deerflow && git pull` 复核 HEAD)
**对应**:5-25 sync(`f9b70713`)后上游 `deerflow/main` 前进到 `74e3e80c`,**全量跟随上游** + surgical 保护 Noldus 定制
**估期**:~1 天(34 commit / 23 个 harness 文件 +1321 行;全量合,surgical 仅 2-3 处受保护文件)
**前置**:无硬前置;建议在反幻觉 spec(`2026-06-02-lead-tool-invocation-reliability`)之前做(sync 会顺带带入 #2135 的 after_model+jump_to 范式,反幻觉层3 可直接复用,不必手工复刻)

> **策略(2026-06-02 用户锁定)**:**deerflow 是我们的 infra 底座,不是外部参考库**。我们整个 agent harness 站在它肩上,它的 bug fix / 性能改进 / 新能力默认**全要**——因为它修的坑就是我们迟早会踩的坑(event-loop 阻塞、SSE、checkpointer、中间件全是地基)。
>
> **默认全量跟随,例外是受保护文件做 surgical**(而非"默认跳过、挑着合")。理由:挑着合 = 主动让底座和上游分叉,分叉越久未来 sync 越痛,且会错过"还没意识到需要"的修复。即使某 commit 我们当前用不上(如 postgres pool / MiMo provider),只要落在非受保护文件,**合进来零害**(不走那条路就不触发)且**保持一致性、消除分叉点**——为"用不上"主动跳过等于人为制造下次 sync 的冲突。
>
> **唯一守住的是 22 个受保护文件里的 Noldus 定制**(中文 prompt、4096 截断、subagent 注册、extra_env、定制中间件链等)——全量拉取时这些绝不能被覆盖,做 surgical merge。遵循 [[feedback_sync_protected_files_registry_loss]] + CLAUDE.md「同步核心规则」。

---

## 0. 关键事实(实施前必读,纠正一个吓人的假象)

### ⚠️ "2192 commit" 是假象,真实 delta = 34
`git log origin/dev..deerflow/main` 显示 2192 commit,**这是 subtree 没 squash 导致的"全量回退对比"假象**(`.deerflow-sync-state` 注释记录:5-21/5-25 sync 都没 squash,LAST_SYNC_COMMIT 检测失效)。

**真实 delta**:从记录的基准 `f9b70713`(5-25)算起:
```
git log --oneline f9b70713..deerflow/main   # = 34 commit
```
- 34 个新增 commit
- 其中碰 `backend/packages/harness`(我们关心的核心)= **14 个**
- 其中碰我们 **22 个受保护文件 = 仅 2 处**(`mcp/tools.py`、`mcp/cache.py`)
- 纯 frontend/docs/chore(deps)= **9 个**(低风险,可批量或跳过)

**这是一次常规增量 sync,不是史诗工程。**

### sync 基准 & 状态文件
- `.deerflow-sync-state`:`last_sync_commit: f9b70713` / `2026-05-25` / 5 PR 15 commit
- **本次 sync 完成后必须更新此文件**到新 HEAD(SOP Step 6),否则下次又回退全量对比

### 受保护文件清单(22 个,`scripts/sync-deerflow.sh:51`)
含 Noldus 定制,**绝不整文件接受上游**:`agents/lead_agent/{prompt,agent}.py`、`subagents/builtins/__init__.py`、`tools/tools.py`、`tools/builtins/__init__.py`、`subagents/executor.py`、`mcp/tools.py`、`sandbox/tools.py`、`guardrails/middleware.py`、`agents/middlewares/loop_detection_middleware.py` 等(完整见脚本)。

---

## 1. 34 个 commit 分类(实施 = 按类处理)

### 默认:全部合。下面只标出"需要特别处理"的,其余 23 个 harness 文件 + app/frontend 改动**全量跟随上游**。

### ⚠️ 必须 surgical 的文件(全量拉取时这几个不能整文件覆盖,逐处保 Noldus 定制)
这是全量 sync 里**唯一需要人工逐处对比**的部分。其余文件 `git show 上游 > 本地` 直接覆盖即可。

| 文件 | 受保护 | 上游改动 | Noldus 定制(必保) | surgical 做法 |
|---|---|---|---|---|
| `mcp/tools.py` | ✅ | +17 行 HTTP/SSE 跳过 session pooling(避免 anyio RuntimeError,`162fb214`) | `MCP_TOOL_RESULT_MAX_CHARS=4096` + `_truncate_result`(`:24-30`) | 正交:合 pooling fix,保截断 |
| `agents/middlewares/llm_error_handling_middleware.py` | ✅ | +44 行 | 总超时上限 + 多种 timeout 关键字识别 | 逐处对比,合上游 fix 保超时定制 |
| `agents/middlewares/todo_middleware.py` | 🟡(我们有 plan-mode 定制) | +22 行 = **#2135 的 after_model+jump_to**(防过早退出) | 我们的 context-loss 检测 | 对比合;**这正是反幻觉 spec 层3 要的范式,合进来即白赚** |
| `agents/thread_state.py`(若被 `92905e9e` 碰) | ✅ | thread state schema 复用 | Noldus 字段(shared_path/artifacts/todos 等) | 确认无字段丢失 |

> **判定**:CLAUDE.md 血泪教训点名的 Tier4 致命文件(`runtime/user_context.py`/`runs/manager.py`/`memory/storage.py`/`setup_agent_tool.py`)**这批 34 commit 一个都没碰**;`runtime/checkpointer`/`runtime/events` 虽被碰,但**我们已吃 Tier4**(persistence 模块齐全),不再有 ImportError 风险 → **可全量跟随**(见 §3 风险更新)。

### 顺带带入的红利(全量跟随的额外收益,不是"挑"来的)
- `todo_middleware.py` 的 **#2135 after_model+jump_to 范式** → 反幻觉 spec 层3 直接复用
- `ToolOutputBudgetMiddleware`(`ca487578`,新文件 `tool_output_budget_middleware.py` +489 行)→ 超大 tool 输出保护,和我们 MCP 4096 截断互补
- `11dd5b06` strip 未闭合 `<think>` 标签 → 关联我们前端 thinking 渲染史([[project_2026-05-20_e2e_3fix_ttft_test_cleanup]])
- `e344be8d` Blockbuster event-loop 阻塞 IO 测试门 → 关联 [[feedback_async_io_blocks_event_loop]],防回归
- `9f3be2a9` UploadsMiddleware 异步化、`cbf8b194` JSONL IO 加固、`3cb75887` memory json、`e683ed6a` write_file 恢复 等一批 runtime/中间件鲁棒性 fix

### 即使"当前用不上"也合(消除分叉点,非主动跳过)
- `031d6fbc` postgres AsyncConnectionPool:我们用 SQLite,**不触发**,但合进来保持 checkpointer 模块和上游一致
- `44677c5e` MiMo provider(新文件 `patched_mimo.py`):我们用 deepseek,**不加载**,但合进来零害 + 下次 sync 不冲突
- `46ddc346`/`e8e9edcb` Feishu channels:我们主走 web,但 channels 模块跟随上游

### 唯一真可跳的(纯文档,合不合都行)
`74e3e80c`/`872079b8`(docs 注释清理)、`8330b244`/`da41701f`(blocking IO 文档)、`e0280194`(PR 模板)、`d6a604d5`(Windows makefile,我们 linux/amd64)。**docs 跳过零损失;但若 subtree 整体 merge,这些也一起进来,无害。**

---

## 2. 实施步骤(全量跟随 + surgical 保护)

### Step 1 — dry-run 看脚本分类
```bash
./scripts/sync-deerflow.sh --dry-run    # 看脚本对 34 commit 的"安全/受保护"分类
```
⚠️ **脚本的"安全文件"分类仅用于"哪些要 surgical"的提示,不是"哪些要跳过"**。我们的策略是全量合,脚本对非受保护文件默认 `git show 上游 > 本地`(全量覆盖)正是我们要的。

### Step 2 — 全量合 + surgical(分 PR 仅为 review 粒度,不为筛选)
- **PR-1(全量非受保护文件)**:让脚本全量合所有非受保护 harness + app/frontend 改动。跑全量测试。
- **PR-2(surgical 受保护文件)**:§1 表里那 3-4 个文件,逐处 `diff <(git show deerflow/main:<file>) <local>`,合上游 fix **保 Noldus 定制**。
- ⚠️ **仍不用 `--auto-apply`**:不是因为要筛选,而是受保护文件需人工 surgical;非受保护的可在交互里全部 Y。

### Step 3 — 每个 PR 改完跑全量
```bash
cd packages/agent/backend && source .venv/bin/activate && make test
```
[[feedback_pr_merge_must_run_full_suite_on_shared_logic]]:碰共享逻辑/受保护文件必跑全量。

### Step 4 — head-to-head 验证受保护文件无定制丢失
对 `mcp/tools.py` 等:合入后 grep 确认 Noldus 定制仍在([[feedback_head_to_head_before_claiming_no_merge]]):
```bash
grep -n "MCP_TOOL_RESULT_MAX_CHARS\|_truncate_result" mcp/tools.py   # 必须仍命中
```

### Step 5 — 更新 sync-state(关键,别漏)
sync 完成后更新 `.deerflow-sync-state`:
```
last_sync_commit: 74e3e80c   # 或实际 sync 到的 HEAD
last_sync_date: 2026-06-02
last_sync_commits_count: 34
last_sync_prs: "#XX, #XX, #XX"
```
**否则下次 sync 又回退到 2192 全量对比假象。**

### Step 6 —(可选但推荐)做一次 subtree squash
`.deerflow-sync-state` 注释指出历史坑根源是没 squash。考虑这次 sync 后做一次 squash,让未来 `git log origin/dev..deerflow/main` 显示真实 delta 而非累积假象。⚠️ squash 风险高,需单独评估,可推迟。

---

## 3. 风险与铁律(CLAUDE.md「同步核心规则」)

1. **绝不整文件覆盖受保护文件**:`git show deerflow/main:<file> > <local>` 永远禁止。22 个受保护文件 + 任何含集合字面量/聚合 import 的文件([[feedback_sync_protected_files_registry_loss]])。
2. **不用 `--auto-apply`**:脚本"安全文件"分类不识别间接 Tier 4 依赖(血泪教训:`runtime/user_context.py` 等被标安全实则污染)。
3. **Tier 4 判定**:上游文件 import `persistence.*`/`runtime.user_context`/`runtime.events` 等 → 整文件不能直接拉(本仓库虽已吃 Tier 4,但 surgical 仍要逐处确认,见 CLAUDE.md 第 13 条)。
4. **每个受保护文件合入后 head-to-head grep 定制**:确认 Noldus 改动(prompt 中文规则、4096 截断、subagent 注册、extra_env 等)未丢。
5. **skip 的 commit 在 PR/handoff 记录原因**:尤其 postgres pool(我们用 SQLite)、MiMo provider(我们用 deepseek)这类"上游有但我们用不上"的。

---

## 4. 验收
- [ ] 真实 delta 确认 = 34(从 `f9b70713` 算起,非 2192)
- [ ] **全量合**:23 个非受保护 harness 文件 + app/frontend 改动全部跟随上游(非挑选)
- [ ] surgical 受保护文件:`mcp/tools.py` 合 session pooling fix **且** 4096 截断 grep 仍命中;`llm_error_handling_middleware.py` 保超时定制;`todo_middleware.py` 合 #2135 范式 + 保 context-loss 定制
- [ ] 顺带红利确认进来:`todo_middleware` 的 after_model+jump_to、`ToolOutputBudgetMiddleware`、`11dd5b06` think 标签、`e344be8d` Blockbuster gate
- [ ] 全量 make test 不退化(每个 PR)
- [ ] `.deerflow-sync-state` 更新到新 HEAD + date + count + PR 号
- [ ] 真正跳过的仅纯 docs(若有),在 handoff 记录
- [ ] (可选)评估 subtree squash 消除未来全量对比假象

## 5. 不在范围
- ❌ 整文件接受任何受保护文件(唯一例外:surgical 保定制)
- ❌ `--auto-apply`(受保护文件需人工 surgical;非受保护可交互全 Y)
- ❌ **为"当前用不上"主动跳过非受保护 commit**(postgres pool/MiMo 落非受保护文件 → 全量合,消除分叉点)
- ❌ 强行 squash(高风险,单独评估)
- ❌ 改 Noldus 定制本身(只合上游正交 fix)

## 6. 关联
- SOP:`docs/sop/deerflow-sync-sop.md`
- 脚本:`scripts/sync-deerflow.sh`(PROTECTED_FILES `:51`,dry-run/auto-apply)
- 上次 sync:[[project_2026-05-25_deerflow_sync_all_prs_merged]] + [[project_2026-05-25_deerflow_sync_cleanup_pr_pending]]
- 反幻觉 spec:`2026-06-02-lead-tool-invocation-reliability-design.md`(其层 3 拉的 #2135 范式也来自上游,但属"复刻范式"非本 sync 的"合 commit")
