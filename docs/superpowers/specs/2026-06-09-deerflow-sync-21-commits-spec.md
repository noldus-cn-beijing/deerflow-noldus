# Spec ① — 全量 sync 上游 deerflow（21 commit，74e3e80c → f92a26d5）

> 日期：2026-06-09 ｜ 作者：实施前 spec（待 review，**不直接实施**）
> 主仓库 dev HEAD：`1a819e7e` ｜ 上游 `deerflow/main` HEAD：`f92a26d5` ｜ 上次 sync 基准：`74e3e80c`
> 配套：spec ②（`2026-06-09-gateway-multiworker-stream-409-fix-spec.md`）正交、独立 worktree/PR
> 来源 handoff：`docs/handoffs/2026-06/2026-06-09-two-specs-sync-and-409-fix-handoff.md`

---

## 0. 目标与边界

**目标**：把上游 deerflow（infra 底座）`74e3e80c..f92a26d5` 的 **21 个 harness commit** 合入本仓库 subtree（`packages/agent/backend/packages/harness/deerflow/`），遵循「**默认全量跟随 + 受保护文件 surgical**」原则（CLAUDE.md「同步核心规则」，memory `feedback_sync_full_follow_upstream_infra`）。

**非目标**：
- 不修 409 跨 worker 流问题（spec ②；本 spec §6 已核实：21 commit **无**该修复）。
- 不动 ethoinsight 库、不动业务逻辑、不引入新范式。

**红线（违反即回滚）**：
- 9 个受保护文件**逐处 diff、绝不整文件覆盖**（除非走 SOP 选项 C：整文件接受后用 Edit 把全部 Noldus 定制重新打回 + grep 验证）。
- `subagents/executor.py` 的循环导入 fix（`3ffaf672`）+ auto-seal 兜底（`a9622070`）**全部保住**（§4.1 红线）。
- sync 后**除 `make test` 外必须裸导入两生产入口**（`import app.gateway` + `make_lead_agent`），否则循环导入回归会被 conftest mock 假绿掩盖（memory `feedback_conftest_mock_hides_circular_import_verify_bare_prod_import`）。

---

## 1. 现场核实结论（本 spec 作者已复核，修正 handoff 两处）

接手 agent 已按 memory `feedback_grill_handoff_must_be_verified` 现场 grep + 解码复核，结论：

| 项 | handoff 说法 | 核实结果 |
|---|---|---|
| 21 commit 清单 | §2.2 列 21 个 | ✅ 准确，`git log 74e3e80c..deerflow/main -- backend/packages/harness/deerflow` = 21，HEAD `f92a26d5` |
| 受保护文件总数 | §2.1「27 个」 | ⚠️ **修正：实际 22 个**（`scripts/sync-deerflow.sh:51-116` 数组）。handoff §2.1 内部不一致（§2.3 表只列 9）。本 spec 用核实值 **22** |
| 受保护 ∩ 触及 | §2.3「9 个」 | ✅ 准确，`comm -12 <(触及42) <(保护22)` = **正好 9 个**，与 §2.3 表一一对应 |
| 触及文件总数 | §2.1「~42」 | ✅ 准确，42 个 |
| MiniMax 新 pip 依赖风险 | §2.5「`cd5bedaa` 可能需 minimax sdk」 | ⚠️ **修正：无新依赖**。`models/patched_minimax.py` 在 base 已存在，上游只 +19 行，imports 仅 `langchain_openai`（已有）+ stdlib。subtree 内 `backend/pyproject.toml` **无 minimax 依赖新增**。安全（详 §5.3） |
| executor.py 循环导入 fix 在位 | §2.4 命门 | ✅ 已核实：`executor.py:314` `from deerflow.tools.builtins.seal_handoff_tools import _seal_handoff_to_workspace` 在 `_attempt_auto_seal_from_artifacts` **函数体内**（缩进 8 空格 + 解环注释），未在顶层 |

---

## 2. 21 个 commit 全清单（已 fetch）

```
f92a26d5 fix(web_fetch): support proxy for Jina reader in restricted networks
3b6dd0a4 feat(subagents): extend deferred MCP tool loading to subagents      ← 核心重构(§3)
cd5bedaa feat: MiniMax provider for image/video/podcast skills + music-gen   ← 非新依赖(§5.3)
64d923b0 fix(middleware): externalize oversized tool output into sandbox for non-mounted sandboxes
51920072 fix(middleware): offload memory injection off event loop to prevent tiktoken blocking
f725a963 fix(runtime): protect sync singleton init and reset
10c1d9f4 fix(search): fix DDGS Wikipedia region handling
8d2e55a0 fix(subagent): structured subagent_status field over text parsing   ← 新文件 status_contract.py + 前端契约(§5.2)
d8b728f7 fix(mcp): close stdio sessions on their owning loop to avoid cross-task cancel-scope error
befe334f fix(config): make the reload boundary discoverable from code
d133b111 fix(summarization): tag summary LLM calls nostream to stop phantom stream messages
88e36d96 fix(#3189): prevent write_file streaming timeout on long reports     ← sandbox/tools.py write_file 80KB 守卫(§4.6)
268fdd69 fix(gateway): drain in-flight runs before closing checkpointer on shutdown
1aac408d fix upload file size contract
2bbc7879 refactor(tool-search): consolidate MCP metadata tag and harden deferred-tool setup  ← 核心重构(§3)
28b1da21 fix(agents): harden update_agent null-like args
89ae74d4 fix(skills): surface offending line and quoting hint on SKILL.md YAML
8fca56cf fix(mcp): accept transport field as alias for type
3ae82dc6 fix(mcp): add auth interceptor with channel user_id + header propagation  ← mcp/tools.py + config/paths.py(§4.4/§4.7)
5dc2d6cb fix(sandbox): close AioSandbox HTTP client during provider teardown
d9f47249 fix(tool-search): reliably hide deferred MCP schemas by removing the ContextVar  ← 核心重构(§3)
```

**主题分组**（决定分批 PR，见 §7）：
- **A. deferred-tool 重构**（跨 4 受保护 + ~5 非受保护，最大风险）：`2bbc7879` + `3b6dd0a4` + `d9f47249`。
- **B. 流式/超时硬化**（#3189 系列）：`88e36d96`（write_file 守卫）+ `d133b111`（summary nostream）+ 配套 `llm_error_handling`（StreamChunkTimeout）。
- **C. event-loop / runtime 稳定性**：`51920072`（memory 注入离开 event loop）+ `f725a963`（singleton 保护）+ `64d923b0`（大 tool 输出外置）+ `268fdd69`（shutdown drain）。
- **D. MCP 健壮性**：`3ae82dc6`（auth interceptor）+ `8fca56cf`（transport alias）+ `d8b728f7`（stdio loop）+ `5dc2d6cb`（AioSandbox teardown）。
- **E. 杂项独立 fix**：`8d2e55a0`（结构化 subagent_status）+ `28b1da21` + `89ae74d4` + `befe334f` + `10c1d9f4` + `1aac408d` + `cd5bedaa`（MiniMax）+ `f92a26d5`（Jina proxy）。

---

## 3. ⚠️ 头号风险专章：deferred-tool 加载重构（`2bbc7879`/`3b6dd0a4`/`d9f47249`）

这是 21 commit 里**唯一的跨文件协同重构**，横跨 **4 个受保护文件 + 4 个非受保护文件**作为一个不可分割的整体。单独合任一文件会留下半截状态、启动即崩。必须**整组一起合**。

### 3.1 重构做了什么（语义）

**旧模型（我们当前所在）**：MCP 工具的「延迟加载」用一个**全局 ContextVar 注册表**（`get_deferred_registry`/`set_deferred_registry`），在 `get_available_tools` 里构建；`DeferredToolFilterMiddleware()` 无参，从 ContextVar 读；prompt 的 `get_deferred_tools_prompt_section()` 无参，从 ContextVar 读。

**新模型（上游）**：删除 ContextVar 注册表，改为**每个 agent 在构建点（tool-policy 过滤之后）装配** `assemble_deferred_tools(filtered_tools, enabled=...)` → 返回 `(final_tools, deferred_setup)`；`deferred_setup` 携带 `deferred_names` + `catalog_hash`，显式传给 `DeferredToolFilterMiddleware(names, hash)` 和 prompt section。promotion 状态改存 graph state（`ThreadState.promoted` + `merge_promoted` reducer）。**好处**：subagent 也能拿到同样的 deferred 过滤（`3b6dd0a4`），且 re-entrant 调用不再互相擦除 promotion（修 issue #2884）。

### 3.2 涉及的 8 个文件（整组合入）

| 文件 | 受保护? | 上游改动 | 合入动作 |
|---|---|---|---|
| `tools/builtins/tool_search.py` | 否 | 新增 `assemble_deferred_tools` / `get_deferred_tools_prompt_section(deferred_names=...)` / `tag` 逻辑，删 ContextVar 注册表 | **全量接受** |
| `tools/mcp_metadata.py` | 否 | 新增 `tag_mcp_tool` / MCP 元数据 tag | **全量接受** |
| `agents/middlewares/deferred_tool_filter_middleware.py` | 否 | 改签名收 `(deferred_names, catalog_hash)`，从 graph state 读 promotion | **全量接受** |
| `agents/thread_state.py` | **是** | 新增 `PromotedTools` TypedDict + `merge_promoted` reducer + `promoted` 字段（**§4.5：与我们 `shared_path` 字段不冲突，纯加在 disjoint 区域**） | **surgical**：加上游 3 块，保 `shared_path`/`merge_artifacts`/`merge_viewed_images` |
| `tools/tools.py` | **是** | `get_available_tools` 里 -52 行（删 ContextVar 注册）+ 改用 `tag_mcp_tool`（**§4.8：与我们 12 个 ethoinsight 工具注册在 disjoint 区域**） | **surgical**：取上游 MCP-deferral 重构 + 保我们 12 工具 import/注册 |
| `agents/lead_agent/agent.py` | **是** | `_make_lead_agent` 两处（bootstrap + default）改用 `assemble_deferred_tools`；`_build_middlewares` 加 `deferred_setup` 参；`apply_prompt_template` 传 `deferred_names`（**§4.2：中间件链顺序 + Noldus 自定义中间件必须保**） | **surgical**：见 §4.2 |
| `agents/lead_agent/prompt.py` | **是** | **删除** prompt.py 里的 `get_deferred_tools_prompt_section`（移到 tool_search.py），改 import；`apply_prompt_template` 加 `deferred_names` 参（**§4.3：1066 行中文定制 + Noldus 专有 `paradigm`/`user_id` 参必须保**） | **surgical**：见 §4.3（**最易翻车**：我们本地仍在 prompt.py:851 定义旧版函数） |
| `subagents/executor.py` | **是 🔴🔴** | `_build_initial_state` 改返回 `(state, final_tools, deferred_setup)`；`_create_agent` 收 `deferred_setup`；TYPE_CHECKING 惰性 import `DeferredToolSetup`（**§4.1 红线：循环导入 fix + auto-seal 全部保**） | **surgical**：见 §4.1 |

### 3.3 合入顺序（组内）

先合非受保护 4 个（全量），再 surgical 4 个受保护，最后**一次性**裸导入 + 全量测试验证整组。**中途不要单独提交半组**——半组状态启动必崩（`assemble_deferred_tools` 不存在 / `merge_promoted` 字段缺）。

### 3.4 组内验收

```bash
cd packages/agent/backend && source .venv/bin/activate
PYTHONPATH=. python -c "import app.gateway"                          # 0 退出
PYTHONPATH=. python -c "from deerflow.agents import make_lead_agent" # 0 退出
PYTHONPATH=. python -m pytest tests/test_gateway_import_no_cycle.py -q
PYTHONPATH=. python -m pytest tests/ -k "tool_search or deferred or promoted" -q
```
grep 确认旧 ContextVar API 已全删干净（不留半截）：
```bash
grep -rn "get_deferred_registry\|set_deferred_registry\|DeferredToolRegistry" packages/agent/backend/packages/harness/deerflow/ \
  | grep -v "tool_search.py"   # 期望：除 tool_search.py 内部，无其他引用残留
```

---

## 4. 9 个受保护文件逐个 surgical 清单

> 通用方法（SOP Step 3 选项 B/C）：先 `diff <(git show deerflow/main:<上游路径>) <本地路径>` 看差异，手工把上游 fix 点合入、保全 Noldus 定制，grep markers 验证。下表「上游改了什么」「本地要保住什么」已现场核实。

### 4.1 🔴🔴 `subagents/executor.py`（最敏感）

**上游改了什么**（仅 deferred-tool 重构，122 diff 行）：
- `from typing import Any` → `from typing import TYPE_CHECKING, Any`；新增 `if TYPE_CHECKING: from deerflow.tools.builtins.tool_search import DeferredToolSetup`（类型注解专用，注释说明 eager import 会重入本包）。
- `_create_agent(self, tools=None)` → `_create_agent(self, tools=None, *, deferred_setup=None)`，并把 `deferred_setup` 透传给 `build_subagent_runtime_middlewares(..., deferred_setup=deferred_setup)`。
- `_build_initial_state` 返回 `(state, filtered_tools)` → `(state, final_tools, deferred_setup)`；函数体**惰性 import** `from deerflow.tools.builtins.tool_search import assemble_deferred_tools, get_deferred_tools_prompt_section`（注释强调惰性是为破环）；装配 `final_tools, deferred_setup = assemble_deferred_tools(filtered_tools, enabled=...)`；append `<available-deferred-tools>` section。
- `_aexecute` 里 `state, filtered_tools = ...; agent = self._create_agent(filtered_tools)` → `state, final_tools, deferred_setup = ...; agent = self._create_agent(final_tools, deferred_setup=deferred_setup)`。

> 上游这个重构**本身就用 TYPE_CHECKING + 函数体惰性 import** 来破同一个环——与我们的循环导入处方同源、不冲突。合入即可。

**本地要保住什么**（LOCAL vs base = **1155 diff 行**，全部保留）：
1. **🔴 循环导入 fix（`3ffaf672`）**：`_attempt_auto_seal_from_artifacts` 函数体内 `from deerflow.tools.builtins.seal_handoff_tools import _seal_handoff_to_workspace`（executor.py:314）**绝不能挪回顶层**。
2. **🔴 auto-seal 兜底（`a9622070`）**：`_AUTO_SEALABLE`（dict，executor.py:279）+ `_attempt_auto_seal_from_artifacts`（executor.py:285）全套；调用点 executor.py:1195。**仅 report-writer/chart-maker 可 auto-seal**，code-executor/data-analyst 永不（CLAUDE.md 第 15 条）。
3. **recursion_limit 计算**：`calculate_subagent_recursion_limit` + `_middleware_implements` + `_BEFORE/AFTER_*_HOOKS`。
4. **handoff 校验**：`_HANDOFF_EMISSION_REQUIRED`、`_HANDOFF_CONTENT_CHECKS`、`_validate_handoff_emitted`、`_check_*_content`、`_CODE_EXECUTOR_METRICS_FIELDS`（三字段等价 + warning）。
5. **seal-resume**：`_attempt_seal_resume`、`_SEAL_RESUME_MAX_ATTEMPTS = 1`（Sprint 5.8 决策，resume_prompt 措辞**别动**）、`max_turns` 硬限制。
6. 顶层额外 import：`from pathlib import Path`、`from langchain.agents.middleware import AgentMiddleware`、`from deerflow.subagents.handoff_schemas import ChartMakerHandoff, ReportWriterHandoff`、`from typing import Callable`（保留我们的，**同时**加上游 `TYPE_CHECKING`）。

**冲突点提示**：`_build_initial_state` / `_create_agent` / `_aexecute` 三处签名同时被上游和本地改过区域接近——逐行对齐，上游的 deferred 透传与我们的 max_turns/recursion 逻辑在不同代码块，可共存。

**合后必做**（§0 红线 + memory）：
```bash
cd packages/agent/backend && source .venv/bin/activate
PYTHONPATH=. python -c "import app.gateway"                          # 必须 0 退出
PYTHONPATH=. python -c "from deerflow.agents import make_lead_agent" # 必须 0 退出
PYTHONPATH=. python -m pytest tests/test_gateway_import_no_cycle.py -q
# 再 grep 确认惰性 import 仍在函数体（缩进 >0）：
grep -n "from deerflow.tools.builtins.seal_handoff_tools import" packages/agent/backend/packages/harness/deerflow/subagents/executor.py
```

### 4.2 `agents/lead_agent/agent.py`

**上游改了什么**（+21/-9，全是 deferred-tool 重构）：`from __future__ import annotations`；`_build_middlewares(..., deferred_setup=None)`；`DeferredToolFilterMiddleware()` → `DeferredToolFilterMiddleware(deferred_setup.deferred_names, deferred_setup.catalog_hash)`，且条件从 `if app_config.tool_search.enabled` → `if deferred_setup is not None and deferred_setup.deferred_names`；`_make_lead_agent` 两处（bootstrap + default）改 `raw_tools → filter → assemble_deferred_tools → final_tools`，并把 `deferred_setup=setup` 传给 `_build_middlewares`、`deferred_names=setup.deferred_names` 传给 `apply_prompt_template`；惰性 import `assemble_deferred_tools`。

**本地要保住什么**：**中间件链顺序 + Noldus 自定义中间件**（`ArchivingSummarizationMiddleware`/`ThinkTagMiddleware`/`TrainingDataMiddleware`/`GateEnforcementMiddleware` 等出现在 `_build_middlewares` 里的定制）。in-graph `create_chat_model` 必须保 `attach_tracing=False`（memory `feedback_in_graph_create_chat_model_attach_tracing_false`，SOP 教训 4）。`get_available_tools` 调用的 Noldus 参数（若有 `groups`/`subagent_enabled` 等）保留。

**做法**：把上游对 `_build_middlewares` 签名 + `DeferredToolFilterMiddleware` 构造 + 两处 `_make_lead_agent` 装配逻辑的改动合入，**中间件 append 的顺序与我们定制中间件原样不动**（上游只动 `DeferredToolFilterMiddleware` 那一处 append 的参数，别顺手重排）。

### 4.3 `agents/lead_agent/prompt.py`（deferred-tool 重构最易翻车点）

**上游改了什么**（+12/-31）：
1. 顶部加 `from deerflow.tools.builtins.tool_search import get_deferred_tools_prompt_section`。
2. **删除** prompt.py 内的 `def get_deferred_tools_prompt_section(*, app_config=None)`（整函数移到 `tool_search.py`，且新签名为 `get_deferred_tools_prompt_section(*, deferred_names=...)`）。
3. `apply_prompt_template(...)` 加形参 `deferred_names: frozenset[str] = frozenset()`；调用点 `deferred_tools_section = get_deferred_tools_prompt_section(deferred_names=deferred_names)`。
4. 文档/提示串增量：write_file 工作流提示（str_replace/append 分段，issue #3189）、`update_agent` 防 `"null"`/`"none"` 字面量。

**本地要保住什么**（**LOCAL = 1066 行，53 处中文定制命中**）：全部中文调度规则、ethoinsight subagent 描述、Gate 反问、**6-08 刚加的 n=1 路由 / 初次判读归 data-analyst / partial 三态**、`apply_prompt_template` 的 Noldus 专有形参（**核实：本地签名含 `paradigm: str | None`、`user_id: str | None`**，上游没有，必须保）。

**⚠️ 翻车点**：本地 prompt.py:851 **仍定义着旧版 `get_deferred_tools_prompt_section()`（无参，读 ContextVar）**，调用点在 prompt.py:1044/1056。surgical 时必须：
- **删**本地 prompt.py:851 那份旧函数（它依赖即将删除的 ContextVar API）。
- **加** import `from deerflow.tools.builtins.tool_search import get_deferred_tools_prompt_section`。
- `apply_prompt_template` 签名：在保留 `paradigm`/`user_id` 等 Noldus 形参的**同时**加 `deferred_names: frozenset[str] = frozenset()`。
- 调用点（本地 1044/1056）改为 `get_deferred_tools_prompt_section(deferred_names=deferred_names)`。
- 同步 §4.2：agent.py 的 `apply_prompt_template(...)` 调用要传 `deferred_names=setup.deferred_names`（caller 与 callee 签名对齐，否则 TypeError）。
- 增量文档串（write_file 提示 / update_agent null 提示）可合入（中性、有益）。

**做法建议**：此文件改动交织重，优先 **SOP 选项 B**（手工合 4 点 import/签名/调用 + 删旧函数），**不要**用选项 C 整文件接受（会瞬间丢 1066 行中文）。改后 grep：
```bash
grep -cE "data-analyst|code-executor|反问|焦虑|n=1|partial" packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py  # 应 ≈53
grep -n "def get_deferred_tools_prompt_section" packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py             # 应为空(已移走)
grep -n "paradigm\|user_id" packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py | head                          # Noldus 专有形参仍在
```

### 4.4 `mcp/tools.py`

**上游改了什么**（+10/-1，`3ae82dc6`）：`from collections.abc import Mapping`；`_make_session_pool_tool` 的 `base_handler` 把 interceptor 注入的 headers 通过 MCP `call_tool(..., meta={"headers": ...})` 透传（stdio MCP 保 header）。

**本地要保住什么**：**4096 字符截断** + BUILTIN 注册类（含集合字面量）。

**核实**：上游 diff **不触及** 4096/BUILTIN/truncate（grep 已确认）。改动在 `_make_session_pool_tool` 内，与我们的截断/注册在 disjoint 区域 → **干净 additive 合入**。

### 4.5 `agents/thread_state.py`

**上游改了什么**（+27，deferred-tool 重构的 state 部分）：新增 `PromotedTools` TypedDict、`merge_promoted` reducer、`ThreadState.promoted: Annotated[PromotedTools | None, merge_promoted]` 字段。

**本地要保住什么**：`shared_path` 字段 + `merge_artifacts` / `merge_viewed_images` reducer。

**核实**：上游纯加在 disjoint 区域，**不触及** `shared_path`/`merge_artifacts` → 几乎 additive。把上游 3 块加进来即可。

### 4.6 `sandbox/tools.py`

**上游改了什么**（+63/-2，`88e36d96` issue #3189）：`import os`；常量 `_WRITE_FILE_CONTENT_MAX_BYTES = 80*1024` + `_WRITE_FILE_MAX_BYTES_ENV = "DEERFLOW_WRITE_FILE_MAX_BYTES"`；`_effective_write_file_max_bytes()`；`write_file_tool` 在 `append=False` 且超 80KB 时返回可操作的拒绝消息 + 更新 docstring（SIZE POLICY）。

**本地要保住什么**：`{{shared://}}` 占位符、`replace_virtual_path`、`DEERFLOW_PATH_*` env 生成规则（**spec ② / validate_catalog 依赖此处 env key 生成**）。

**核实**：上游 diff **不触及** `DEERFLOW_PATH`/`shared://`/`replace_virtual_path`（grep 已确认）。write_file 守卫是 additive → **干净合入**。

### 4.7 `config/paths.py`

**上游改了什么**（+20，`3ae82dc6`）：`import hashlib`；`_UNSAFE_USER_ID_CHAR_RE` + `_SAFE_USER_ID_DIGEST_HEX_LEN`；`make_safe_user_id(raw)`（把 IM 渠道 id 规范化进 user-id 字符集，lossy 时加 sha1 短摘要后缀）。

**本地要保住什么**：`/mnt/shared`、`shared_dir()`。

**核实**：上游纯加新函数，**不触及** `/mnt/shared`/`shared_dir` → additive。

### 4.8 `tools/tools.py`

**上游改了什么**（+8/-52，deferred-tool 重构）：见 §3.2。`get_available_tools` 删 ContextVar 注册块、改 `for t in mcp_tools: tag_mcp_tool(t)`；import 从 `get_deferred_registry` 改为 `from deerflow.tools.mcp_metadata import tag_mcp_tool`。

**本地要保住什么**（**LOCAL vs base = 28 diff 行**）：我们在 imports + BUILTIN 注册块加的 **12 个 ethoinsight 工具**：`set_experiment_paradigm_tool`、`set_viz_choice_tool`、`identify_ev19_template_tool`、`inspect_uploaded_file_tool`、`prep_metric_plan_tool`、`seal_code_executor_handoff`、`seal_data_analyst_handoff`、`seal_chart_maker_handoff`、`seal_report_writer_handoff`、`present_file_tool`、`view_image_tool`、`task_tool`。

**做法**：上游重构在 `get_available_tools` 函数体（MCP 分支），我们的 12 工具在 import 段 + 注册集合——disjoint。surgical：取上游 `get_available_tools` 的 MCP-deferral 改动 + import 行（`tag_mcp_tool`），**保我们 12 工具的 import 与注册原样**。grep：
```bash
grep -cE "seal_code_executor_handoff|identify_ev19_template_tool|set_experiment_paradigm_tool" packages/agent/backend/packages/harness/deerflow/tools/tools.py  # 应 ≥3
grep -n "tag_mcp_tool" packages/agent/backend/packages/harness/deerflow/tools/tools.py  # 上游新 import 在位
```

### 4.9 `agents/middlewares/llm_error_handling_middleware.py`（受保护，但 **非** deferred-tool 重构组）

**上游改了什么**（+66/-2，#3189 流式系列）：新增 `_RETRY_BUDGET_OVERRIDES = {"StreamChunkTimeoutError": 2}`、`_STREAM_DROP_EXCEPTIONS`、方法 `_max_attempts_for(exc)`；可重试集合加 `"StreamChunkTimeoutError"`；为 stream-drop 异常返回「请拆小输出」的专门用户消息。

**本地要保住什么**：总超时上限 + 多 timeout 关键字识别。

**做法**：纯 additive（新常量 + 新方法 + 在既有可重试集合里加一项）。合入上游增量，确认我们的 timeout 关键字集合与总超时逻辑未被覆盖。grep 我们的关键字仍在。

> 注：此文件 + `prompt.py`(write_file 提示) + `sandbox/tools.py`(write_file 守卫) + `summarization`(nostream) 同属 **#3189 流式硬化主题**，建议同一 PR（§7 批 B）。

---

## 5. 非受保护文件（33 个）：默认全量跟随

`comm -23 <(触及42) <(保护22)` = 33 个，**全量接受**（memory `feedback_sync_full_follow_upstream_infra`：底座默认全合，用不上也合进、零害、消下次冲突点）。清单：

```
agents/memory/prompt.py
agents/middlewares/deferred_tool_filter_middleware.py    ← deferred-tool 重构组(§3)
agents/middlewares/dynamic_context_middleware.py
agents/middlewares/summarization_middleware.py           ← d133b111 nostream
agents/middlewares/tool_error_handling_middleware.py     ← 8d2e55a0 stamp subagent_status
agents/middlewares/tool_output_budget_middleware.py      ← 64d923b0
agents/middlewares/view_image_middleware.py
client.py
community/aio_sandbox/aio_sandbox_provider.py            ← 5dc2d6cb
community/aio_sandbox/aio_sandbox.py
community/ddg_search/tools.py                            ← 10c1d9f4
community/jina_ai/jina_client.py                         ← f92a26d5 proxy
community/jina_ai/tools.py
config/app_config.py
config/checkpointer_config.py
config/extensions_config.py
config/model_config.py                                   ← cd5bedaa MiniMax provider 接线
config/reload_boundary.py                                ← befe334f
mcp/cache.py
mcp/__init__.py
mcp/session_pool.py                                      ← 8fca56cf transport alias / d8b728f7 stdio loop
models/factory.py                                        ← cd5bedaa
models/patched_minimax.py                                ← cd5bedaa(+19,非新依赖 §5.3)
runtime/checkpointer/provider.py                         ← f725a963 singleton 保护
runtime/runs/manager.py                                  ← 268fdd69 drain(注意:与 spec② store_only 相关,但不消除 409 §6)
runtime/store/provider.py
skills/parser.py                                         ← 89ae74d4 YAML 报错行
subagents/builtins/general_purpose.py                    ← 3b6dd0a4
subagents/status_contract.py                             ← 8d2e55a0 新文件(§5.2)
tools/builtins/tool_search.py                            ← deferred-tool 重构组(§3)
tools/builtins/update_agent_tool.py                      ← 28b1da21 null-like args
tools/mcp_metadata.py                                    ← deferred-tool 重构组(§3)
uploads/manager.py                                       ← 1aac408d upload size
```

### 5.1 ⚠️ `runtime/runs/manager.py` 注意

上游 `268fdd69`（shutdown drain in-flight runs）改此文件——它**与 spec ② 的 `store_only` 机制同文件**（`store_only` 字段 manager.py:93、`_record_from_store` ~227/246）。全量接受**不会**消除 409（§6 已核实），但合入后**复核** `store_only` 字段语义与 `_record_from_store` 未被上游改动破坏（spec ② 依赖这段不变）。grep：
```bash
grep -n "store_only" packages/agent/backend/packages/harness/deerflow/runtime/runs/manager.py  # 字段 + 置位点仍在
```

### 5.2 `subagents/status_contract.py`（`8d2e55a0` 新文件，必须带配套 fixture）

新增 backend↔frontend 契约模块（结构化 `subagent_status` 取代文本前缀解析）。**关键**：它声明 single source of truth 为共享 fixture `contracts/subagent_status_contract.json`——sync 时**必须确认该 JSON fixture 也一并合入**（grep 上游 `contracts/` 路径），否则 backend stamper（`tool_error_handling_middleware.py`）与前端 fallback parser 的契约测试会缺基准。**前端配套**（`subagent_status` 读取）属 deerflow 前端（subtree `frontend/`）——若本次 sync 范围只含 backend harness，记录为 follow-up（前端 sync 另算），但 backend 侧的 stamp 不会破坏现有文本路径（向后兼容）。

```bash
# 确认 fixture 与契约文件一起来了
ls packages/agent/backend/packages/harness/deerflow/contracts/subagent_status_contract.json 2>/dev/null \
  || git show deerflow/main --stat 8d2e55a0 | grep contracts
```

### 5.3 `cd5bedaa` MiniMax —— 已核实非新依赖（修正 handoff §2.5）

`models/patched_minimax.py` 在 base 已存在，本 commit 对 harness 只 **+19 行**；其 imports 仅 `langchain_openai`（既有）+ `langchain_core` + stdlib（`re`/`Mapping`/`typing`）。subtree 内 `backend/pyproject.toml` 在 21 commit 中**无 minimax 依赖新增**（grep 已确认）。**结论：可直接全量合，无需跳过、无需惰性隔离**。（MiniMax 的 image/video/podcast skill 主体在 subtree 之外，不随本次 harness sync 进来；用不上也不触发。）兜底仍按 §8 验收：合后裸导入会立刻暴露任何缺包（memory `feedback_harness_must_import_without_ethoinsight`）。

---

## 6. ✅ 已核实：21 commit 无 spec ②（409）修复 → 两 spec 正交

grep 21 commit 全量 diff 的 `stream/worker/join/409/store_only/sse/StreamBridge` 关键词，**无一条修「跨 worker SSE join」**。最相关的 `268fdd69`（shutdown drain）/`f725a963`（singleton 保护）只是 run 元数据持久化/启动稳定性，**不消除流的 409**。上游 `runtime/stream_bridge/async_provider.py:55-56` 仍是 `if config.type == "redis": raise NotImplementedError("Redis stream bridge planned for Phase 2")`——**跨 worker 共享流上游自己都标 Phase 2 未实现**。

**结论**：spec ① 合完 409 依旧存在，必须靠 spec ② 独立修。两 spec 互不阻塞、各自 PR。

---

## 7. 分批 PR 建议（21 commit 较大，参考 `project_2026-05-25_deerflow_sync_3_pr_batch`）

| 批 | 内容 | 风险 | 验收重点 |
|---|---|---|---|
| **PR-1 deferred-tool 重构**（§3） | `2bbc7879`+`3b6dd0a4`+`d9f47249` 整组（4 受保护 + 4 非受保护，**不可拆**） | 🔴 高 | 裸导入 + `test_gateway_import_no_cycle` + tool_search/promoted 测试 + grep 旧 ContextVar API 清零 |
| **PR-2 流式硬化 #3189**（批 B） | `88e36d96`(sandbox write_file 守卫) + `d133b111`(summary nostream) + `llm_error_handling`(StreamChunkTimeout) + `prompt.py` write_file 提示 | 🟡 中（含 2 受保护：sandbox/tools.py additive、llm_error additive、prompt.py 已在 PR-1 处理则此处只增文档串） | write_file 80KB 拒绝路径测试 + 全量 |
| **PR-3 runtime/MCP 稳定性 + 杂项**（批 C/D/E） | 其余 commit（全量非受保护为主 + `mcp/tools.py`/`config/paths.py`/`thread_state.py` 的 additive surgical + `status_contract.py` 新文件 + MiniMax） | 🟢 低 | 全量 + 裸导入 + status_contract fixture 在位 |

> 若 review 倾向单 PR，也可一次合（21 commit），但**受保护文件必须仍逐个 surgical**，且 PR 描述按上表分组列清「每文件上游改了什么 / 保了什么」。
>
> **顺序约束**：PR-1 必须先合（它建立 `assemble_deferred_tools` 等新 API，PR-2/3 不依赖但 dev 上半组状态不能停留）。PR-2 的 prompt.py write_file 提示若与 PR-1 的 prompt.py surgical 撞同文件，合进 PR-1 一起做（避免二次碰 1066 行文件）。

---

## 8. 验收（整体）

按 SOP `docs/sop/deerflow-sync-sop.md` Step 4-6：

1. **裸导入两生产入口**（§0 红线，每个受保护文件改后 + 整体完成后都跑）：
   ```bash
   cd packages/agent/backend && source .venv/bin/activate
   PYTHONPATH=. python -c "import app.gateway"                          # 0 退出
   PYTHONPATH=. python -c "from deerflow.agents import make_lead_agent" # 0 退出
   PYTHONPATH=. python -m pytest tests/test_gateway_import_no_cycle.py -q
   ```
2. **全量测试**：`make test`（backend）。**认 5 个已知基线红**（非新增，handoff §1 + memory `feedback_known_full_suite_test_pollution_4_tests`）：`test_chart_maker_config_basic_fields` + `deferred_tool_registry_promotion×2` + `inspect_gate_guardrail`/`paradigm_identification_gate` 的 `test_async_delegates_to_sync`。worktree 需 `ln -sf <主仓>/packages/agent/config.yaml <worktree>/packages/agent/config.yaml`，否则 ~5 个 config 依赖测试假红。
   > ⚠️ deferred-tool 重构可能影响 `deferred_tool_registry_promotion` 两个基线红测试（它们测旧 ContextVar API）。合 PR-1 后这两个测试**预期会变**——确认是「测试随上游 API 演进而更新/移除」而非「我们改坏」：head-to-head 看上游是否同步改了这两个测试，把上游版本一并取来（SOP 教训：上游修复带配套 test 就一起拿）。
3. **lint**：`make lint`（ruff）。
4. **`make dev` 实起**：Gateway 健康（`make dev` 不卡在 `Waiting for Gateway on port 8001`——卡住=循环导入回归），前端可访问 localhost:2026，发一条消息走通 lead → subagent。
5. **grep 守护 markers 全在**（§4 各文件的 grep 命令）。
6. **更新 `.deerflow-sync-state`** 到 `f92a26d5`（last_sync_commit / last_sync_date=2026-06-09 / last_sync_commits_count=21 / 填实际 PR 号），commit（中文 message）。

---

## 9. 关联 memory（实施 agent 必读）

- `feedback_conftest_mock_hides_circular_import_verify_bare_prod_import` — 改 executor/tools/agents 后必裸导入两入口（§0 红线 / §4.1）
- `feedback_sync_full_follow_upstream_infra` — 默认全量跟随（§5）
- `feedback_sync_protected_files_registry_loss` — 注册类文件（tools/tools.py 等）绝不整文件接受（§4.8）
- `feedback_head_to_head_before_claiming_no_merge` — 受保护文件先 diff 再判（§4 通用方法）
- `feedback_harness_must_import_without_ethoinsight` — 新依赖假绿陷阱（§5.3 MiniMax）
- `feedback_ethoinsight_must_be_backend_workspace_dep` — 全量跟随勿洗掉 pyproject 的 ethoinsight workspace 声明（SOP 教训 5）
- `feedback_in_graph_create_chat_model_attach_tracing_false` — agent.py 保 attach_tracing=False（§4.2）
- `project_2026-05-25_deerflow_sync_3_pr_batch` / `project_2026-05-25_deerflow_sync_all_prs_merged` — 分批 PR 经验（§7）
- `project_2026-06-08_three_specs_review_acdf` — **worktree 必须显式 `git worktree add <path> -b <name> dev`**（默认 origin/main 落后一大截、PR 卷入误删）
- `feedback_grill_handoff_must_be_verified` — 别信文字、现场核实（§1）

---

## 10. 实施 agent 第一步

1. 读 §9 memory + 本 spec。
2. **建 worktree（红线）**：`git worktree add <path> -b sync-deerflow-21-commits dev`（**显式基于 dev**，非默认 origin/main）；`ln -sf <主仓>/packages/agent/config.yaml <worktree>/packages/agent/config.yaml`。
3. `./scripts/sync-deerflow.sh --dry-run` 生成报告，对照本 spec §2-5 核对触及文件分类。
4. 按 §7 分批（建议先 PR-1 deferred-tool 整组）：非受保护全量接受 → 9 受保护逐个 surgical（§4 清单）→ §8 验收。
5. 更新 `.deerflow-sync-state` → `f92a26d5`，提交，**交 review，不直接 merge 到 main**（先进 dev，再 PR）。
