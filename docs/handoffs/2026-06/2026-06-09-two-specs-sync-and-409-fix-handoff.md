# Handoff — 写两份 spec：①全量 sync 上游 deerflow（21 commit）②修 Gateway 多 worker 切 thread 409

> 日期：2026-06-09 ｜ 主仓库 dev HEAD：`1a819e7e`（含 5 spec + 循环导入 fix + 文档/部署脚本对齐）
> 上游 deerflow/main HEAD：`f92a26d5` ｜ 上次 sync 基准：`74e3e80c`（落后 **21 个 harness commit**）
> 这是给下一位 AI Agent 的施工交接：**产出两份 spec**（不是直接改代码）。不是给用户的总结。

---

## 0. 任务（用户原话）

> "先写一个 spec 全量拉取，然后再写一个 spec 对于这个进行修复。"

- **Spec ①**：全量 sync 上游 deerflow 的 21 个 harness commit（默认全量跟随 + 受保护文件 surgical）。
- **Spec ②**：修「线上部署后，从正在运行的 thread 切走再切回 → HTTP 409 `Run ... is not active on this worker and cannot be streamed`」。

两份 spec **正交、各自独立 worktree/PR**。先写 spec，**不要直接实施**（用户要先 review spec）。

---

## 1. ⚠️ 接手第一步：现场核实（别信本文档的文字结论，自己 grep + 解码）

memory 铁律 `feedback_grill_handoff_must_be_verified` + `feedback_head_to_head_before_claiming_no_merge`：本文档所有"已核实"项你都应抽查复核。尤其：
- **建 worktree 必须显式基于 dev**：`git worktree add <path> -b <name> dev`（EnterWorktree / 裸 `git worktree add` 默认 origin/main，落后一大截 → PR 卷入误删除。本仓库这坑反复踩，见 memory `project_2026-06-08_three_specs_review_acdf`）。
- **跑全量测试**：worktree 需 `ln -sf /home/wangqiuyang/noldus-insight/packages/agent/config.yaml <worktree>/packages/agent/config.yaml`，否则 ~5 个 config 依赖测试假红。
- **已知基线 5 红**（非新增，别归因自己）：`test_chart_maker_config_basic_fields` + `deferred_tool_registry_promotion×2` + `inspect_gate_guardrail`/`paradigm_identification_gate` 的 `test_async_delegates_to_sync`。

---

## 2. Spec ① — 全量 sync 上游 deerflow（21 commit）

### 2.1 已核实的事实

- **sync 工具**：`scripts/sync-deerflow.sh`，上游 remote=`deerflow`（=`git@github.com:noldus-cn-beijing/deerflow-noldus.git`），分支 `main`。SOP：`docs/sop/deerflow-sync-sop.md`。
- **基准**：`.deerflow-sync-state` 记录上次 sync 到 `74e3e80c`。`74e3e80c..deerflow/main` = **21 个 harness commit**。
- **sync 范围**：只同步 `backend/packages/harness/deerflow/`（subtree）。
- **受保护文件 27 个**（脚本 `PROTECTED_FILES`，行 51-116）。

### 2.2 21 个 commit 全清单（已 fetch，`git log --oneline 74e3e80c..deerflow/main -- backend/packages/harness/deerflow`）

```
f92a26d5 fix(web_fetch): support proxy for Jina reader in restricted networks
3b6dd0a4 feat(subagents): extend deferred MCP tool loading to subagents
cd5bedaa feat: MiniMax provider for image/video/podcast skills + new music-generation skill
64d923b0 fix(middleware): externalize oversized tool output into sandbox for non-mounted sandboxes
51920072 fix(middleware): offload memory injection off event loop to prevent tiktoken blocking  ← 直接相关 event-loop 阻塞
f725a963 fix(runtime): protect sync singleton init and reset
10c1d9f4 fix(search): fix DDGS Wikipedia region handling
8d2e55a0 fix(subagent): structured subagent_status field over text parsing
d8b728f7 fix(mcp): close stdio sessions on their owning loop to avoid cross-task cancel-scope error
befe334f fix(config): make the reload boundary discoverable from code
d133b111 fix(summarization): tag summary LLM calls nostream to stop phantom stream messages
88e36d96 fix(#3189): prevent write_file streaming timeout on long reports
268fdd69 fix(gateway): drain in-flight runs before closing checkpointer on shutdown  ← run 生命周期，spec② 关联
1aac408d fix upload file size contract
2bbc7879 refactor(tool-search): consolidate MCP metadata tag and harden deferred-tool setup
28b1da21 fix(agents): harden update_agent null-like args
89ae74d4 fix(skills): surface offending line and quoting hint on SKILL.md YAML
8fca56cf fix(mcp): accept transport field as alias for type
3ae82dc6 fix(mcp): add auth interceptor with channel user_id and keep header propagation to mcp tools
5dc2d6cb fix(sandbox): close AioSandbox HTTP client during provider teardown
d9f47249 fix(tool-search): reliably hide deferred MCP schemas by removing the ContextVar
```

### 2.3 ⚠️ 触及的受保护文件（9 个，必须 surgical，绝不整文件覆盖）

21 commit 触及 ~42 个 harness 文件，其中**这 9 个撞受保护清单**：

| 受保护文件 | 为何危险 |
|-----------|---------|
| `agents/lead_agent/agent.py` | 中间件链顺序 + Noldus 定制中间件，整文件覆盖丢定制 |
| `agents/lead_agent/prompt.py` | 中文调度规则 + ethoinsight subagent 描述 + Gate 反问 + **6-08 刚加的 n=1 路由/初次判读归 data-analyst** |
| `agents/middlewares/llm_error_handling_middleware.py` | 总超时上限 + 多 timeout 关键字 |
| `agents/thread_state.py` | `shared_path` 字段 + merge_artifacts/merge_viewed_images reducer |
| `config/paths.py` | `/mnt/shared`、`shared_dir()` |
| `mcp/tools.py` | 4096 字符截断 + BUILTIN 注册类（含集合字面量，绝不整文件接受） |
| `sandbox/tools.py` | `{{shared://}}` 占位符 + `replace_virtual_path` + `DEERFLOW_PATH_*` env 生成（**spec② / validate_catalog 依赖此处 env key 生成规则**） |
| `subagents/executor.py` | **🔴🔴 最敏感**：见 §2.4 |
| `tools/tools.py` | BUILTIN_TOOLS 注册集合 |

### 2.4 🔴🔴 executor.py 的特别警告（sync 最易翻车点）

`subagents/executor.py` 上游这次也改了（`8d2e55a0` structured subagent_status / 可能还有别的），**而我们昨天刚改了它**，必须 surgical 保住**全部**本地定制：
1. **循环导入 fix（commit `3ffaf672`）**：`_attempt_auto_seal_from_artifacts` 函数体内**惰性 import** `from deerflow.tools.builtins.seal_handoff_tools import _seal_handoff_to_workspace`——**绝不能让上游把这行挪回模块顶层**，否则生产启动崩 Gateway（见 memory `feedback_conftest_mock_hides_circular_import_verify_bare_prod_import`）。
2. **auto-seal 兜底（commit `a9622070`）**：`_attempt_auto_seal_from_artifacts` + `_AUTO_SEALABLE` 全套。
3. **既有 Noldus 定制**：`recursion_limit` 修复、`max_turns` 硬限制、`_attempt_seal_resume`、`_validate_handoff_emitted`、`_HANDOFF_CONTENT_CHECKS`、`_SEAL_RESUME_MAX_ATTEMPTS=1`（Sprint 5.8 决策，resume_prompt 措辞别动）。
4. 上游 `8d2e55a0` 引入 `subagents/status_contract.py`（新文件，非受保护）——若 executor 要用它，把 import 合进来但保留上述定制。

**sync executor.py 后必做**（memory `feedback_conftest_mock_hides_circular_import_verify_bare_prod_import`）：
```bash
cd packages/agent/backend && source .venv/bin/activate
PYTHONPATH=. python -c "import app.gateway"                      # 必须 0 退出
PYTHONPATH=. python -c "from deerflow.agents import make_lead_agent"  # 必须 0 退出
PYTHONPATH=. python -m pytest tests/test_gateway_import_no_cycle.py -q  # 必须绿
```

### 2.5 非受保护文件（~33 个）：默认全量跟随

`runtime/runs/manager.py`、`runtime/store/provider.py`、`runtime/checkpointer/provider.py`、各 middleware（summarization/tool_error/tool_output_budget/view_image/deferred_tool_filter/dynamic_context）、mcp/*（cache/session_pool/__init__）、community/*、config/*（app_config/model_config/checkpointer_config/extensions_config/reload_boundary）、models/factory.py + 新 `models/patched_minimax.py`、skills/parser.py、tools/builtins/tool_search.py、uploads/manager.py、client.py 等——**全量接受**（memory `feedback_sync_full_follow_upstream_infra`：底座默认全合，"用不上"也合进零害、消除下次冲突点）。

> ⚠️ 注意新增文件可能引新依赖：`cd5bedaa` MiniMax provider 可能需要新 pip 包（minimax sdk）。sync 后跑 `import app.gateway` 裸导入 + 全量测试，缺包会立刻暴露（memory `feedback_harness_must_import_without_ethoinsight`）。MiniMax 用不上可评估跳过该 commit 或惰性 import 隔离。

### 2.6 Spec ① 应包含的内容

1. **执行步骤**：`./scripts/sync-deerflow.sh --dry-run` 生成报告 → 非受保护文件全量接受 → 9 个受保护文件逐个 `diff <(git show deerflow/main:<UH>/<file>) <local>` surgical 合入上游 fix 点、保留全部 Noldus 定制 → 跑全量回归 + 裸导入验证 → 更新 `.deerflow-sync-state` 到 `f92a26d5`。
2. **每个受保护文件的 surgical 清单**：spec 里逐个列"上游改了什么 / 本地要保住什么"。executor.py 用 §2.4 红线。
3. **验收**：全量测试（认 5 基线红）+ `test_gateway_import_no_cycle` 绿 + `import app.gateway`/`make_lead_agent` 裸导入 0 退出 + `make dev` 实起 Gateway 健康。
4. **可能分批 PR**：21 commit 若太大，可按主题分 PR（如先非受保护全量一批、受保护 surgical 一批）。参考 memory `project_2026-05-25_deerflow_sync_3_pr_batch` 的分批经验。
5. **SOP 对齐**：按 `docs/sop/deerflow-sync-sop.md` Step 1-6。

---

## 3. Spec ② — 修 Gateway 多 worker 切 thread 409

### 3.1 现象（用户实测，线上部署后）

从正在运行的 thread 页面切走（切到别的 thread）→ 再切回 → 前端报：
```
Failed to load resource: the server responded with a status of 409
HTTP 409: {"detail":"Run 7fb55505-... is not active on this worker and cannot be streamed"}
```

### 3.2 ✅ 已核实的根因（现场读码确证，非推测）

1. **409 来源**：`packages/agent/backend/app/gateway/routers/thread_runs.py`
   - line 268（`join_run`）+ line 308（`stream_existing_run`）：`if record.store_only ... raise HTTPException(409, "is not active on this worker and cannot be streamed")`。
2. **根因 = Gateway-embedded 多 worker + 内存 StreamBridge**：
   - compose（`docker/docker-compose.yaml:69`）：`uvicorn app.gateway.app:app --workers ${GATEWAY_WORKERS:-4}` → **4 个 worker 进程**。
   - run 的实时 SSE 流由 `StreamBridge` 承载，**只有内存实现**（`runtime/stream_bridge/memory.py` `MemoryStreamBridge` + `async_provider.py`，无 Redis/DB 共享实现）。流是 **worker 本地内存**。
   - run 元数据有持久化 RunStore（`runtime/runs/store/memory.py` 是默认；`manager.py` 支持 `store: RunStore` 注入）。当 `get(run_id)` 在**当前 worker 内存找不到** run、但能从持久 store 查到行时，`_record_from_store`（`manager.py:227`）重建一个 **`store_only=True`**（`manager.py:93/246`）的 record。
   - **结果**：切回来时 nginx/uvicorn 把 re-stream 请求轮询到**另一个 worker**（不是当初跑 run 的那个）→ 该 worker 内存无此 run 的 bridge → record 标 `store_only=True` → **409**（不是 404，因为元数据从 store 查得到）。
3. **这是上游有意设计**：`store_only` = "元数据可查、但实时流不在本 worker、无法 join"。上游**没有**跨 worker 共享流的实现（已核实 21 commit 无此修复，见 §3.4）。

### 3.3 ⚠️ 下一个 agent 仍需现场核实/补查（写 spec② 前必做）

1. **前端怎么发起 re-stream**：grep `frontend/src/` 哪个组件在切回 thread 时调 `/runs/{id}/stream` 或 `/join`（LangGraph SDK `joinStream`/`useStream`）。确认：切回来是**盲目重连**，还是已有部分降级。报错栈 `9cf7c8666eaa7a30.js` 是打包后文件，要回源码定位。**关键**：前端拿到 409 后当前行为（抛 console error + 加载失败 UI？还是静默？）。
2. **DB 里这个 run 的最终状态/历史消息能否拿到**：确认有"读 run 最终结果 / thread 历史消息"的端点（`store_only` 的 record 仍可查状态）。这是"前端优雅降级"方案的依据——409 时改读历史而非流。
3. **`GATEWAY_WORKERS` 真实值**：compose 默认 4，但**生产 `.env` / 部署可能覆盖**。确认线上实际几个 worker（`DEPLOY_AGENT_ENV` 指的 .env）。若已是 1 还报 409，则根因另有他因，需重查。
4. **上游 run 持久化改进是否减轻**：spec① 若先合入 `39f901d3`（restore historical runs from store）/`268fdd69`（drain in-flight runs on shutdown）/`f725a963`（singleton 保护），确认它们**是否改变 store_only 行为**（大概率不消除 409，只让元数据更可恢复，但要核实）。

### 3.4 ✅ 已核实：上游 21 commit 无直接修复

grep 21 commit 的 stream/worker/join/409/store_only/sse 关键词，**没有一条修"跨 worker SSE join"**。最相关的 `268fdd69`（shutdown drain）/`39f901d3`（restart restore）都是 run 元数据持久化，不消除流的 409。**结论：spec② 不能靠 sync 上游解决，是独立修复。**

### 3.5 修复方向（用户已了解三选项，spec② 要把取舍写清，最终选型可留给 review）

用户问过怎么修，我给过三个选项（用户当时转向先看上游有没有修，未拍板具体方案）。spec② 应把三条路的**真实代价**写清，建议默认走"前端优雅降级"（最小、最稳，不引基础设施依赖）：

| 方案 | 做法 | 代价 | 评价 |
|------|------|------|------|
| **A 前端优雅降级**（建议默认） | 切回收到 409（run 不在本 worker）时，前端不再盲目 re-stream，改拉该 run 最终状态 + 历史消息渲染（DB 已有），消除 console error/加载失败 | 最小、最稳、不动后端基础设施 | 缺点：切回时「还在跑的 run」看不到实时增量，要等它结束或手动刷新。对当前体验已是大幅改善 |
| **B GATEWAY_WORKERS=1** | compose env 设 1，单 worker 所有 run 同进程内存，re-stream 必命中，409 消失 | 一行 env | 缺点：丢多 worker 并发吞吐（4→1），高并发 Gateway 成瓶颈。**适合当前用户量小的过渡**，但不是终态 |
| **C 共享 StreamBridge** | Redis pub/sub 或 DB 轮询实现 `StreamBridge`，任意 worker 可 join 别 worker 的 run 流 | 引 Redis 依赖 + 写新 bridge 实现 + 改部署 + 充分测试 | 彻底解决跨 worker 实时续流，但重。**真正终态**，工作量大 |

> spec② 建议：**A + B 组合**作为短期（前端降级兜底 + 过渡期可选 workers=1 彻底无 409），**C 留作 backlog**（真要多 worker 实时续流时再做，可能届时跟随上游若上游补了共享 bridge）。让 review 拍板。

### 3.6 Spec ② 应包含的内容

1. **根因确证段**：§3.2 的链 + 前端 re-stream 定位（§3.3 补查后填）。
2. **选型 + 理由**：三方案取舍表，建议默认 A（或 A+B），C 列 backlog。
3. **A 的具体改动**：前端 stream 错误处理——409（`is not active on this worker`）时 catch → 转读历史/最终状态端点 → 渲染，不抛 console error。列具体文件（`frontend/src/` 待 §3.3 定位）。
4. **B 的具体改动**（若纳入）：`docker/docker-compose.yaml` 的 `GATEWAY_WORKERS` 默认值 / 生产 .env 文档。⚠️ memory `feedback_dev_prod_behavior_alignment`：env/flag 是产品属性，要落 compose 不能只放 .env 让用户配。
5. **测试**：前端无测试框架（`frontend/CLAUDE.md` "No test framework configured")→ playwright 行为验证（切 thread 复现 409 → 改后无 console error、能看到历史）。后端若动则 `make test`。
6. **dogfood 验证（必做）**：线上/本地多 worker 起，切走再切回正在跑的 thread，确认无 409 报错、UI 不再加载失败。

### 3.7 关键文件清单（spec②）

| 文件 | 说明 |
|------|------|
| `app/gateway/routers/thread_runs.py:268,308` | 409 抛出点（`store_only` 判据）|
| `runtime/runs/manager.py:93,227,246` | `store_only` 字段 + `_record_from_store` 置位 |
| `runtime/stream_bridge/memory.py` + `async_provider.py` | 内存 StreamBridge（唯一实现，C 方案要新增共享版）|
| `runtime/runs/store/memory.py` | 默认内存 RunStore（元数据；DB-backed 是 store_only 元数据来源）|
| `docker/docker-compose.yaml:69` | `--workers ${GATEWAY_WORKERS:-4}`（B 方案）|
| `frontend/src/`（待定位）| 切 thread 时发起 re-stream 的组件（A 方案）|

---

## 4. 本会话已落地的相关改动（背景，spec 别重复）

- dev HEAD `1a819e7e`：①循环导入 fix（`3ffaf672`，executor.py 惰性 import + `test_gateway_import_no_cycle.py` guard）②文档对齐 Gateway-embedded 现状（根 + backend CLAUDE.md）③`deploy-via-tar.sh` 删 langgraph 残留镜像。
- 五份 EPM dogfood spec（A 路由/B 术语/C seal auto-seal/D validate_catalog 路径/E present_files 拒幻影）全已合 dev。
- **Gateway-embedded = 现部署形态**（无独立 langgraph 容器，跟随上游），这是 spec② 409 的架构前提。

---

## 5. 关联 memory（接手必读）

- `feedback_conftest_mock_hides_circular_import_verify_bare_prod_import` — sync executor.py 后必裸导入验证（spec① §2.4 命门）
- `feedback_sync_full_follow_upstream_infra` / `feedback_sync_protected_files_registry_loss` / `feedback_head_to_head_before_claiming_no_merge` — sync 原则（spec①）
- `feedback_harness_must_import_without_ethoinsight` — 新依赖/import 假绿陷阱（spec① MiniMax）
- `feedback_dev_prod_behavior_alignment` / `feedback_deploy_compose_per_service_image_tag` — env/部署对齐（spec② B 方案）
- `project_2026-06-08_three_specs_review_acdf` + `project_2026-06-08_epm_dogfood_findings` — 本批 spec 上下文 + worktree base 坑
- `feedback_grill_handoff_must_be_verified` — 别信文字、现场核实

---

## 6. 下一位 agent 第一步

1. 读上述 memory + 本文档 §1。
2. **现场抽查复核** §2.3（9 受保护文件）+ §3.2（409 链）几个关键点，确认本文档没误判。
3. 补查 §3.3 的 4 项（尤其前端 re-stream 定位 + 生产实际 worker 数）。
4. 写 **spec①**（`docs/superpowers/specs/2026-06-09-deerflow-sync-21-commits-spec.md`）+ **spec②**（`docs/superpowers/specs/2026-06-09-gateway-multiworker-stream-409-fix-spec.md`）。
5. 交用户 review，**不要直接实施**。
