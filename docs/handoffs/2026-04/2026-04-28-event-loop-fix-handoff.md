# Handoff: Event loop is closed 修复 + DeerFlow 上游同步评估 + E2E 验证

**日期**: 2026-04-28
**状态**: 主 bug 已修复 + E2E 跑通 + 上游 issue 草稿就绪，等待提交 + 暴露的次级问题待修
**上一会话**: [2026-04-28 lead-agent-workflow-handoff.md](2026-04-28-lead-agent-workflow-handoff.md)

---

## 1. 本会话目标与产出

主要解决两件事：

1. ✅ DeerFlow 上游同步评估（57 个 commits 待评估）
2. ✅ 用户 E2E 测试中遇到的 `RuntimeError: Event loop is closed` 修复

附带产出：

3. ✅ 上游 issue 草稿（中文，按上游模板格式）
4. ⚠️ 跑出一次完整 E2E 但暴露了几个次级问题待修

---

## 2. 关键改动（已实施）

### 2.1 Event loop is closed 根因修复 — memory updater 清 langchain client cache

**文件**: [agents/memory/updater.py](packages/agent/backend/packages/harness/deerflow/agents/memory/updater.py)

**根因**（**重要：上游 deerflow main 也有此 bug，未修复**）：

三层独立设计叠加成 race：

1. `langchain_anthropic._client_utils._get_default_async_httpx_client` 用 `@lru_cache` **全进程缓存** httpx.AsyncClient
2. httpx 连接在创建时绑定 event loop（`anyio TLSStream._transport._loop`），**不能跨 loop 复用**
3. deerflow `_run_async_update_sync` 用 `asyncio.run(coro)` 模式，每次 memory update **创建/销毁短命 loop**

时序：memory daemon thread 在 loop_mem 上调 LLM → 连接 conn_2 进入连接池（绑 loop_mem）→ memory update 完成 loop_mem 关闭 → conn_2 留池中成"僵尸"→ 下个 lead run 复用 conn_2 → 流读完后 SDK 自动 aclose → `transport._loop.call_soon()` 触发 `Event loop is closed`。

**关键陷阱**：错误堆栈完全指向 lead run 的 streaming，**没有任何 memory 帧**——但污染源是 memory update。

**修复**: 在 `_run_async_update_sync` 的 `finally` 中调用新增的 `_evict_provider_async_client_caches()`，清掉 langchain anthropic 的两个全局 lru_cache（`_get_default_async_httpx_client` + `_get_default_httpx_client`）。下次 LLM 调用时 langchain 重建一个全新的 AsyncClient + 空连接池，污染连接随旧 client GC 回收。

代价：每次 memory update 后第一个 LLM 调用要重做 SSL 握手（~200ms），memory 默认 30s debounce 影响可忽略。

### 2.2 BG_JOB_ISOLATED_LOOPS=true（不解决主 bug，但是必要的卫生）

**文件**: [backend/Makefile:6](packages/agent/backend/Makefile#L6) 和 [scripts/serve.sh:269](packages/agent/scripts/serve.sh#L269)

启动 langgraph dev 时显式设 `BG_JOB_ISOLATED_LOOPS=true`，让每个 background run 独立 event loop（langgraph 框架文档明确建议 `--allow-blocking` 必须配套此变量）。

**注意**：这只解决 langgraph 框架级 run 之间的隔离，**不影响**deerflow 自己起的 daemon thread（memory queue）—— 所以单靠这个不能修主 bug，2.1 才是关键。

启动日志可看到 `Starting queue with isolated loops`（之前是 `shared loop`）确认生效。

### 2.3 ReadError 加入可重试列表（防御性补丁）

**文件**: [llm_error_handling_middleware.py:169](packages/agent/backend/packages/harness/deerflow/agents/middlewares/llm_error_handling_middleware.py#L169)

在 `_classify_error` 的可重试异常 set 中增加 `"ReadError"`（对齐上游 #2309）。本地之前已有 `ReadTimeout`/`ConnectTimeout`/`RemoteProtocolError`，缺 `ReadError`。

### 2.4 单元测试

**文件**: [tests/test_memory_updater.py](packages/agent/backend/tests/test_memory_updater.py)

新增 4 个测试覆盖 `_evict_provider_async_client_caches` + `_run_async_update_sync` 的 finally 路径：
- `test_evict_provider_async_client_caches_clears_lru_caches`
- `test_evict_provider_async_client_caches_swallows_import_errors`
- `test_run_async_update_sync_evicts_caches_on_success`
- `test_run_async_update_sync_evicts_caches_on_failure`

---

## 3. 测试基线

```
1708 PASS, 2 FAIL (pre-existing), 14 skipped
```

2 个已有失败（与本次改动无关，沿用 2026-04-27 交接文档已说明）：
1. `test_planning_skill_is_enabled_in_config` — `extensions_config.json` 文件不存在
2. `test_usage_example_shows_ask_clarification_between_analyst_and_writer` — prompt 文本已改但测试未更新

新增测试 4 个全部通过；总数从 1704 → 1708。

---

## 4. E2E 验证结果

**测试场景**: 用户上传 5 个鱼群行为轨迹文件（对照组 1-2，实验组 3-5），完整跑通 lead → code-executor → data-analyst → report-writer 流程。

**记录**: [docs/e2e/Shoaling Behavior Trajectory Analysis  4-28端到端.md](../e2e/Shoaling%20Behavior%20Trajectory%20Analysis%20%204-28%E7%AB%AF%E5%88%B0%E7%AB%AF.md)

### 4.1 ✅ 主 bug 已修

LLM 调用密度高（lead + 3 个 subagent + 3 次 memory update），**没有任何 `Event loop is closed` 错误**。修复有效。

### 4.2 ⚠️ 暴露的次级问题（按严重程度排）

#### P0：`set_experiment_paradigm` tool 报 permission error
- 现象（E2E 文档 line 181）："The set_experiment_paradigm tool failed with a permission error"
- agent fallback 到手写 `experiment-context.json` 才走通
- 上一会话刚把这个 tool 注册到 `BUILTIN_TOOLS`，注册成功但**实际调用失败**
- 需查 [middlewares/experiment_context.py](packages/agent/backend/packages/harness/deerflow/agents/middlewares/experiment_context.py) 的工具实现 + sandbox/gate enforcement 中间件是否拦截了它

#### P0：`/mnt/shared/` 写入失败
- 现象（E2E 文档 line 345-348）："Permission denied to /mnt/shared/. Let me try writing to /mnt/user-data/workspace/ instead"
- agent 想写 `code_summary.json` 到 `/mnt/shared/` 但被拒
- 应该是 prompt 提到了 `/mnt/shared/` 但 sandbox 没暴露这个虚拟路径
- 需查 [agents/lead_agent/prompt.py](packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py) 找 `/mnt/shared` 引用，要么改 prompt 要么在 sandbox 加路径映射

#### P1：Gate 1 两级范式选择被跳过
- 设计：flywheel/manual 模式应强制走两级 UI（7 大类 → 细分范式）
- 实际：agent 看到用户消息提到"鱼群行为"就直接推断 paradigm=shoaling，跳过 UI
- agent 思考过程（line 33-35）："The user has already specified the paradigm... I don't need to go through the full Gate 1 two-step confirmation"
- 这是 prompt 设计偏弱 / GLM-5.1 倾向走捷径
- 修法：加强 [prompt.py:259-350](packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py#L259-L350) 的 Gate 1 描述，**正面**强调"无论用户是否说明范式，都必须先调用范式选择工具让 UI 显示选项"

#### P1：data quality warning 强制 ask_clarification 流程未走
- 设计：handoff_code_executor.json 有 critical warning（n=2 < 3）应触发 `ask_clarification`
- 实际：agent 思考过程（line 274-279）有犹豫："Let me follow the process. Actually, let me present the results first..."，最后选择不阻塞
- 同样是 prompt 偏弱
- 修法：与 P1 同源，加强 orchestration guide 的强制约束

### 4.3 ✅ 业务质量评估

抛开技术问题，**分析质量很好**：data-analyst 准确识别 "中等效应量 + 不显著 p 值 = power 不足而非无差异"；识别 Subject 3 是组间差异的单点驱动者；给出可操作的实验设计建议；report-writer 6 章节结构完整。

---

## 5. 上游同步评估结论

`./scripts/sync-deerflow.sh --dry-run` 跑完。**fork 仓库已迁移**：

- 旧 remote `git@github.com:Dimples-ai/deer-flow.git`（404 不存在）
- 新 remote `git@github.com:noldus-cn-beijing/deerflow-noldus.git`（fork 自 bytedance/deer-flow）
- 已在本地 `git remote set-url deerflow ...` 切换

**上游基准**: `f0dd8cb` → `9dc2598`，**57 个 commits**横跨约 3 个月。

**分类**:
- 51 个安全文件（你没改过）→ 可自动合入
- 1 个新增文件
- 8 个受保护文件（你改过 + 上游改过）需人工合并：

| 文件 | 你的 diff | 上游关键改动 |
|---|---|---|
| `subagents/executor.py` | 212 行 | #1965 event loop fix（**本地已有此实现**）, #2253 per-subagent skill |
| `agents/lead_agent/agent.py` | 202 行 | #2515 read options from context, #2034 setup wizard |
| `agents/lead_agent/prompt.py` | 864 行 | 杂项 prompt 修复 |
| `agents/middlewares/llm_error_handling_middleware.py` | 131 行 | #2309 catch ReadError（**本地刚补**）, #2095 circuit breaker |
| `tools/builtins/task_tool.py` | 125 行 | #2253 / #2305 / #2514 subagent 工具继承 |
| `sandbox/tools.py` | 167 行 | #2317 ls_tool 路径掩码（安全）|
| `sandbox/local/local_sandbox.py` | 79 行 | #2494 / #1935 路径处理 |
| `mcp/tools.py` | 83 行 | #2451 自定义 tool interceptor |

**关键发现**：原以为可以从上游 cherry-pick #1965 修今天的 bug —— **错的**。
- #1965 (`e5b14906`) 修的是 `SubagentExecutor.execute()` 在 running loop 里调用 `asyncio.run` 的冲突，**本地已实现完整**（`_isolated_loop_pool` + `_execute_in_isolated_loop` 在 [executor.py:80](packages/agent/backend/packages/harness/deerflow/subagents/executor.py#L80) 都存在）
- #2309 修的是 ReadError 加入重试列表 —— **本地已比上游更全**（多了 `retry_total_timeout_s` 等），仅缺 `ReadError` 名字，已补上
- 今天的 `Event loop is closed` 是**第三个独立的 race 路径**（memory updater + langchain lru_cache），上游 main 也有此 bug 但**未修复**

**memory 相关文件全部已与上游对齐**：`memory/queue.py`、`storage.py`、`message_processing.py` 等 0 行 diff；`updater.py` 仅 1 行（`run_name="memory_agent"` trace 字段）。所以**今天的修复是 memory 路径 race 的首次修复**。

---

## 6. 上游 Issue 草稿（已起草，待提交）

**位置**: [docs/handoffs/2026-04-28-deerflow-upstream-issue-draft.md](2026-04-28-deerflow-upstream-issue-draft.md)

**目标仓库**: https://github.com/bytedance/deer-flow/issues/new/choose（用 `Runtime Information` 模板）

**核心内容**:
- 标题：`[runtime] memory updater 触发 RuntimeError: Event loop is closed（langchain provider 全局 client 缓存被跨 loop 复用）`
- Problem summary + Expected/Actual + 完整堆栈节选
- 三层根因分析（带链接到 langchain-anthropic 源码 + deerflow 源码）
- 时序图（T0/T1/T2）
- 三个建议修法（A 选项已在我们 fork 验证有效）
- 平台 / 版本 / 相关 issue（#2302、langgraph#4218、LiteLLM 同类事后分析）

**未提交**。用户希望自己拿草稿去提。

**提交前需用户确认/调整**:
1. 复现频率描述（"在交互密集的 E2E 测试中可稳定复现" 是否准确）
2. 是否同时提配套 PR
3. 是否在 OpenAI 等 provider 也复现一遍以增强说服力

---

## 7. 未完成事项（按优先级）

### P0 - 修 E2E 暴露的两个 permission 错误

1. **修 `set_experiment_paradigm` tool 的 permission error**
   - 入手：先看 [experiment_context.py](packages/agent/backend/packages/harness/deerflow/agents/middlewares/experiment_context.py) 的 `set_experiment_paradigm_tool` 实现
   - 验证：写一个 E2E 测试或 unit test 直接调用这个 tool 看错误堆栈
   - 怀疑点：sandbox 路径权限 / GateEnforcementMiddleware 的允许工具列表

2. **修 `/mnt/shared/` 路径不可写**
   - 入手：grep `"/mnt/shared"` 找所有 prompt 引用
   - 选 A：改 prompt 把所有 `/mnt/shared/` 改成 `/mnt/user-data/workspace/`
   - 选 B：在 sandbox 路径映射里加 `/mnt/shared/` → 物理路径

### P1 - 加强 Gate 1 / data quality 强制流程

3. **加强 Gate 1 prompt**
   - 文件：[lead_agent/prompt.py:259-350](packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py#L259-L350)
   - 用**正面**指令（按 CLAUDE.md 第 6 条规则——GLM-5.1 不能用"禁止 X"）
   - 例如："flywheel 模式下，**第一步永远是**调用范式选择工具显示 7 大类 UI，让用户在 UI 上选择，**即使用户已经在消息中提到了范式名称**"

4. **加强 data quality warning 强制 ask_clarification**
   - 同源问题，prompt 强化即可

### P1 - 提交上游 issue

5. **提交 deerflow upstream issue**
   - 草稿就绪：`docs/handoffs/2026-04-28-deerflow-upstream-issue-draft.md`
   - 用户希望自己提

### P2 - 后续上游同步（可选）

6. **批量评估 51 个安全文件 + 6 个剩余受保护文件**
   - 当前不阻塞业务，可在 v0.1 后再做
   - 注意 `mcp/tools.py` 的 #2451（自定义 tool interceptor）可能对 noldus-kb MCP 集成有用

---

## 8. 文件位置速查

| 内容 | 路径 |
|------|------|
| 本次修复主入口 | [agents/memory/updater.py:220-285](packages/agent/backend/packages/harness/deerflow/agents/memory/updater.py#L220-L285) |
| 新增 `_evict_provider_async_client_caches` | [updater.py:220](packages/agent/backend/packages/harness/deerflow/agents/memory/updater.py#L220) |
| BG_JOB_ISOLATED_LOOPS 注入点 | [Makefile:6](packages/agent/backend/Makefile#L6) + [serve.sh:269](packages/agent/scripts/serve.sh#L269) |
| ReadError 补丁 | [llm_error_handling_middleware.py:169](packages/agent/backend/packages/harness/deerflow/agents/middlewares/llm_error_handling_middleware.py#L169) |
| 新增测试 | [test_memory_updater.py](packages/agent/backend/tests/test_memory_updater.py)（文件末尾 4 个测试）|
| 上游 issue 草稿 | [docs/handoffs/2026-04-28-deerflow-upstream-issue-draft.md](2026-04-28-deerflow-upstream-issue-draft.md) |
| 本次 E2E 记录 | [docs/e2e/Shoaling Behavior Trajectory Analysis  4-28端到端.md](../e2e/Shoaling%20Behavior%20Trajectory%20Analysis%20%204-28%E7%AB%AF%E5%88%B0%E7%AB%AF.md) |
| dry-run 报告 | `/tmp/deerflow-sync-report/`（临时目录，会清掉）|
| Sandbox 路径映射逻辑 | [sandbox/local/local_sandbox.py](packages/agent/backend/packages/harness/deerflow/sandbox/local/local_sandbox.py) 的 `replace_virtual_path` |
| Gate 1 prompt 节 | [lead_agent/prompt.py:259-350](packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py#L259-L350) |

---

## 9. 建议接手路径

```
第一步：确认本次改动状态（git diff 5 个文件，应该全部 staged 但未 commit）
  cd /home/wangqiuyang/noldus-insight
  git status -s
  # 预期：M Makefile, M updater.py, M llm_error_handling_middleware.py,
  #       M test_memory_updater.py, M serve.sh
  # 加 ?? 两个新文件（issue 草稿 + E2E 记录）

第二步：commit 本次修复（建议 1 个 commit）
  git add packages/agent/backend/Makefile \
          packages/agent/scripts/serve.sh \
          packages/agent/backend/packages/harness/deerflow/agents/memory/updater.py \
          packages/agent/backend/packages/harness/deerflow/agents/middlewares/llm_error_handling_middleware.py \
          packages/agent/backend/tests/test_memory_updater.py
  git commit -m "fix(memory): evict langchain provider client cache after asyncio.run cycle

修复 RuntimeError: Event loop is closed —— memory updater 用 asyncio.run 模式
跑短命 loop，langchain_anthropic 的 @lru_cache 全局 httpx.AsyncClient 在跨 loop
共享时残留绑定已关闭 loop 的连接，下游 LLM 调用偶发崩溃。
- 在 _run_async_update_sync 的 finally 中清空 langchain client lru_cache
- 启动脚本补 BG_JOB_ISOLATED_LOOPS=true（langgraph 推荐配置）
- ReadError 加入可重试异常列表（对齐上游 #2309）"

第三步：提交上游 issue（用户自己做，文档草稿已就绪）
  cat docs/handoffs/2026-04-28-deerflow-upstream-issue-draft.md
  # 复制内容到 https://github.com/bytedance/deer-flow/issues/new/choose

第四步：开始修 P0 - set_experiment_paradigm permission error
  # 入手点：
  cat packages/agent/backend/packages/harness/deerflow/agents/middlewares/experiment_context.py
  # 找到 set_experiment_paradigm_tool 定义，看是不是 sandbox 路径写入权限问题
  # 跑 unit test 复现：
  cd packages/agent/backend && PYTHONPATH=. .venv/bin/pytest tests/ -k "experiment" -v
```

---

## 10. 风险与注意事项

| 风险 | 应对 |
|------|------|
| **langchain-anthropic 私有模块依赖** | 我们的 fix 用了 `langchain_anthropic._client_utils._get_default_async_httpx_client` 这是私有 API，未来 langchain 升级可能改名/删除。已加 try/except 兜底（导入失败只记 debug log 不报错）。但若 langchain 改了实现仍用 lru_cache 但函数名变，bug 会复发。建议在 langchain 升级时验证 |
| **修复仅覆盖 anthropic 一家 provider** | 如果未来用 OpenAI/DeepSeek 等也走 langchain，且它们的 client 实现也有同款 lru_cache 问题，需要扩展 `_evict_provider_async_client_caches` 加更多 provider 分支 |
| **E2E 暴露的 P0 问题** | `set_experiment_paradigm` 失败 + `/mnt/shared/` 不可写 —— 这两个都是上一会话引入的功能没完全跑通的副作用。下个 session 必修 |
| **GLM-5.1 不严格遵循 prompt** | E2E 暴露 Gate 1 / data quality 流程被 agent 简化掉。CLAUDE.md 第 6 条规则提醒过"必须用正面指令"——加强 prompt 时记得用正面语气，不要用"禁止" |
| **Docker 部署没改** | `docker/docker-compose-dev.yaml` 和 `docker-compose.yaml` 还没设 `BG_JOB_ISOLATED_LOOPS=true`。如果团队用 docker 跑也要修，但需重建镜像 |
| **上游同步策略不变** | fork 仓库已切到 `noldus-cn-beijing/deerflow-noldus`，下次同步前先在 fork 本地跑 `git fetch upstream && git merge upstream/main && git push`，再回 noldus-insight 跑 `./scripts/sync-deerflow.sh` |

---

## 11. 决策记录

- ❌ **不 cherry-pick 上游 #1965**：本地 executor.py 已实现完整 isolated loop 逻辑，#1965 修的不是今天的 bug
- ❌ **不 cherry-pick 上游 #2309**：本地 llm_error_handling_middleware 已比上游更全，仅补一个 `ReadError` 名字
- ❌ **不批量合 51 个安全文件**：当前不紧急，且 v0.1 deadline 9 月，避免引入未测过的改动
- ✅ **采用清 lru_cache 方案而非改架构**：路径 A 改动小（5 行核心代码），代价可控（每次 memory update 后多 200ms SSL 握手），已验证有效。路径 B（memory updater 改长生命周期 loop）更彻底但风险大，留待提 PR 时再考虑
- ✅ **同时设 `BG_JOB_ISOLATED_LOOPS=true`**：单独不能修 bug，但是 langgraph 框架文档明确推荐，作为卫生措施加上

---

## 12. 用户对核心 bug 的理解程度（备注）

用户在本会话中**深入追问了 race 的细节**，问了这些核心问题：

1. "memory 系统结束之后 connection 没正确关闭吗"——纠偏：是故意保留（HTTP keep-alive），不是漏关
2. "httpx 和 https 区别"——库 vs 协议
3. "连接池里到底有什么"——SSL 通道对象，跨 thread/agent 全进程共享
4. "不同 thread 隔离怎么做的"——业务层（messages/checkpointer/sandbox）按 thread_id 隔离，网络层不隔离
5. "SSL 握手时间是验证的 cost"——主要是 TLS 密钥协商 80ms，证书验证只几毫秒
6. "checkpoint 是捞 JSON 塞上下文吗"——是的，但是 LangGraph 框架自动做且 state 不只 messages

**已经完全理解 bug 机制**。下个 session 不需要重复解释，直接执行修复。
