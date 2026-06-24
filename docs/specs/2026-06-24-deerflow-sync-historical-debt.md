# 2026-06-24 DeerFlow 同步历史债清单（可执行版）

> **状态**：已对照 `deerflow/main`（当前 HEAD `11415875`，已 fetch）逐符号核验，2026-06-24 修订为可直接执行。
>
> **触发**：sync PR #191（`e418d729`→`11415875`）合入上游测试时，5 个测试文件（`test_lead_agent_prompt` / `test_gateway_services` / `test_subagent_executor` / `test_uploads_router` / `test_feishu_parser`）的**上游版**假设了本地没有的生产接口；整文件覆盖后暴露这些债。#191 已把这 5 个测试**回退到本地版**（故它们不在 #191 diff 里），债单独立项清理。
>
> **定性**：这是 **e418d729 之前就欠的历史债**（本地 harness 在更早的某次 sync 漏合了上游几批重构）。`.deerflow-sync-state` 现标 `last_sync_commit: 11415875`（=上游 tip），与本债单不矛盾——**#191 只是「暴露者」，不是「制造者」**。清完这 5 条，本地 harness 才真正追平上游接口面，后续 sync 不会再因预存偏离冒红测试。
>
> **执行进度（2026-06-24）**：债 A 已实施 PR #194。债 B/C/D 已从本母 spec 拆成独立可执行 spec，可交给不同 agent 并行：
> - **债 B**（executor async）：[`2026-06-24-debt-b-executor-async-spec.md`](2026-06-24-debt-b-executor-async-spec.md) — 可与 A 并行（文件不重叠）。
> - **债 D**（app 层 D1/D2/D3）：[`2026-06-24-debt-d-app-layer-spec.md`](2026-06-24-debt-d-app-layer-spec.md) — 完全独立，可放心并行。
> - **债 C**（langchain/langgraph 升级）：[`2026-06-24-debt-c-langchain-upgrade-spec.md`](2026-06-24-debt-c-langchain-upgrade-spec.md) — **最高风险，A/B/D 全合入 dev 后最后做、独立 PR、独立验证**。
>
> ---

## ⚠️ 给执行 agent 的前置须知（务必先读）

1. **路径前缀**：
   - 本地 harness：`packages/agent/backend/packages/harness/deerflow/`
   - 本地 app 层：`packages/agent/backend/app/`
   - 本地测试：`packages/agent/backend/tests/`
   - 上游对应路径去掉 `packages/agent/`，即 `backend/...`，用 `git show deerflow/main:backend/<path>` 读取。
2. **commit SHA 不可信，一律以「符号/签名 diff」为准**。`packages/agent/` 是 DeerFlow 的 subtree fork，历次 sync **没做 squash**（见 `.deerflow-sync-state` 历史背景），`git log` 上的 SHA 对不上上游逻辑 commit。**本文不再给「根因 commit」**——`git show deerflow/main:<file>` 看上游当前态即可，不要去 `git cherry-pick` 任何 SHA。
   - 反例（旧版 spec 的错）：曾把债 2 溯到 commit `0ee9ad9e`，实际那是个 **1072 文件、22 万行的 squash 产物**（前端 Copilot fix），毫无参考价值。
3. **改 harness 核心后必须裸导入两生产入口**（CLAUDE.md「harness 模块顶层 import 闭环风险」铁律）：
   ```bash
   cd packages/agent/backend
   PYTHONPATH=. python -c "import app.gateway"
   PYTHONPATH=. python -c "from deerflow.agents import make_lead_agent"
   ```
   两者 exit 0 才算过。新 helper 的 import 放函数体内惰性，别放模块顶层。
4. **改共享组件先 grep 所有消费者，再全量跑回归**（三大病理之 catastrophic forgetting）。`apply_prompt_template` 这种签名改动尤其要先列全调用点。
5. **本地已知 baseline 污染**：backend 全量跑有确定性失败（deferred_tool / inspect_gate / paradigm_gate 等的 test isolation 污染，单跑全绿、全量必红）。判断「我有没有引入新红」的方法是**对比 baseline 数字**，不是「全量全绿」。本债单全部完成前，先记下当前 baseline。
6. **运行测试的解释器**：用 `packages/agent/backend/.venv/bin/python -m pytest`（或 `make test`）。本仓库 harness 跑测试已知需 `PYTHONPATH=.`（在 backend/ 目录下）。

### 开工前先记 baseline

```bash
cd packages/agent/backend
source .venv/bin/activate   # 或直接用 .venv/bin/python
# 记录当前全量结果（数字写进 PR 描述，作为「未引入新红」的对账基准）
make test 2>&1 | tail -30
# 单独确认两生产入口当前可裸导入（应 exit 0）
PYTHONPATH=. python -c "import app.gateway" && echo "app.gateway OK"
PYTHONPATH=. python -c "from deerflow.agents import make_lead_agent" && echo "make_lead_agent OK"
```

---

## 债务总览

| 债 | 名称 | 规模 | 受保护文件 | 解锁的上游测试 |
|---|---|---|---|---|
| **A（原债1+债2 合并）** | `app_config` 贯通 + skills per-config cache | **中** | `lead_agent/prompt.py`、`lead_agent/agent.py`、`client.py` | `test_lead_agent_prompt.py` |
| **B（原债4）** | executor skill 加载 async 化 | **中** | `subagents/executor.py` | `test_subagent_executor.py`（+ 依赖 A 的 skill-storage app_config） |
| **C（原债3）** | langchain/langgraph 生态版本升级 | **中（有 blast radius，非「小」）** | 无源码；改 `packages/harness/pyproject.toml` + `uv.lock` | 仅解锁 1 个被 `@pytest.mark.skip` 的本地测试（**不在 5 个回退文件之列**） |
| **D（原债5）** | app 层三类缺口（sandbox readable / Feishu user_id / internal auth + checkpoint） | **中** | 无 harness 受保护文件，但 `app/` 需单独同步 | `test_gateway_services.py` + `test_uploads_router.py` + `test_feishu_parser.py` |

> **为什么把原债 1、2 合并**：两者都改**同一个** `apply_prompt_template` 签名和**同一个** `_build_subagent_section`。拆两个 PR 必然在同几行冲突。合并为一个「prompt.py app_config 体系对齐」PR。

> **为什么把原债 3 重新定性**：旧版 spec 说「17 个测试调用点加 `tools=None`，规模小」——**方向反了且规模错**。真相见债 C：根因是 langchain 库版本 skew，本地只有 **1 个**测试受影响（且已被显式 skip），修法是**升级整个 langchain/langgraph 生态**（langgraph 还跨了 1.0→1.1 minor），blast radius 不小。

### 推荐执行顺序

1. **债 A**（prompt.py app_config 对齐）→ 解锁 `test_lead_agent_prompt`
2. **债 B**（executor async）→ 解锁 `test_subagent_executor`（**注意：还需债 A 的 skill-storage app_config 通路**，见债 B 依赖说明）
3. **债 D**（app 层）→ 解锁其余 3 个测试，可拆 3 个小 PR
4. **债 C**（langchain 升级）→ **最后做、独立 PR、独立验证**。版本升级风险最高，放最后避免污染其他债的回归判断。

---

## 债 A：`app_config` 贯通 + skills per-config cache

**受保护文件**：`agents/lead_agent/prompt.py`、`agents/lead_agent/agent.py`、`client.py`（全部含 Noldus 定制，**逐处对比、绝不整文件覆盖**）

### A.1 现状核验（已确认）

**本地缺 `app_config` 参数的 4 个签名**（行号为本地 `prompt.py`）：

| 函数 | 本地签名 | 上游签名 | Noldus 冲突 |
|---|---|---|---|
| `_get_memory_context`（行 730） | `(agent_name: str \| None = None)` | `(agent_name=None, *, app_config: AppConfig \| None = None)` | 无（上游 body 内 `if app_config is None: ... else: config = app_config.memory` 驱动 per-config memory 选项；本地 body 走隐式 `get_app_config()`，纯数据层） |
| `_build_acp_section`（行 979） | `()` | `(*, app_config: AppConfig \| None = None)` | 无 |
| `_build_custom_mounts_section`（行 999） | `()` | `(*, app_config: AppConfig \| None = None)` | 无（本地 body 行 1002 已用 `get_app_config()`） |
| `apply_prompt_template`（行 1052） | `(…, *, agent_name, available_skills, paradigm, user_id, thread_id, deferred_names)` | `(…, *, agent_name, available_skills, app_config=None, deferred_names)` | **有**——见 A.2 核心断裂 |

**本地缺的 skills per-config 符号**（全部 grep=0，上游均存在，定义在**上游 `prompt.py`**，不在 `skills/storage/`）：
- `_enabled_skills_by_config_cache: dict[int, tuple[object, list[Skill]]] = {}`（模块级 per-AppConfig 缓存，按 `id(app_config)` 索引）
- `get_enabled_skills_for_config(app_config: AppConfig \| None = None) -> list[Skill]`
- `get_cached_enabled_skills() -> list[Skill]`
- `_build_self_update_section(agent_name)`（custom agent 自更新 prompt 段）
- `_build_available_subagents_description(available_names, bash_available, *, app_config)`（动态 subagent 描述）

**已确认本地具备的前置**（降低难度）：
- 本地 `prompt.py` **已 import** `AppConfig`（行 12）并已在 `get_skills_prompt_section`（行 943）/`_build_custom_mounts_section`（行 1002）用 `get_app_config()`。
- **skills storage 包 `skills/storage/`（`skill_storage.py` / `local_skill_storage.py` / `__init__.py`）本地与上游已字节级一致**，且 `get_or_new_skill_storage(**kwargs)` / `LocalSkillStorage.__init__(..., app_config=None)` **已接受 `app_config`**。→ **债 A 不需要改 storage 包**，只改 `prompt.py` + 调用点。
- 本地 `get_skills_prompt_section`（行 943）**签名已有 `*, app_config=None`**，但 body 仍调 `_get_enabled_skills()`（忽略参数）；上游 body 调 `get_enabled_skills_for_config(app_config)`。→ 这是「补线」不是「加参数」。

### A.2 核心断裂（必须协调，不能二选一）

本地 `apply_prompt_template` 用 `paradigm / user_id / thread_id` 三参数注入 Noldus 的 **prior_corrections + resolved_facts**（本地 body 行 1064-1071：`_get_prior_corrections_context` + `_get_resolved_facts_context`）；上游用 `app_config`。**两套参数都要保留**：
- **不能**删掉 `paradigm/user_id/thread_id`（Noldus 业务依赖）。
- **要**新增 `app_config`，并让 config 获取走「参数 > 全局」：`config = app_config or get_app_config()`。

本地 `_build_subagent_section(max_concurrent)`（行 181，Noldus 大幅定制）上游是 `(max_concurrent, *, app_config=None)`。本地 `apply_prompt_template` body 调 `_build_subagent_section(n)`（行 1075），上游调 `_build_subagent_section(n, app_config=app_config)`。

### A.3 全部 3 个调用点（旧版 spec 漏了 client.py）

`apply_prompt_template` 共 **3 处**调用（用 `grep -rn 'apply_prompt_template(' packages/agent/backend/packages/harness/deerflow/` 复核）：

1. **`client.py:259`** — **旧版 spec 完全漏掉这处**。它隔壁就传了 `app_config=self._app_config`（给 `make_lead_agent`），所以 `apply_prompt_template` 调用补 `app_config=self._app_config` 最自然。当前它没传 `paradigm/user_id/thread_id`。
2. **`agent.py:662`**（bootstrap 路径）— 当前传 `available_skills=set(["bootstrap"]), thread_id=thread_id`。
3. **`agent.py:694`**（主路径）— 当前传 `paradigm=lead_paradigm, user_id=lead_user_id, thread_id=thread_id`。`agent.py` 已 import `AppConfig`/`get_app_config`，且该函数上下文有 `agent_config` 可取 app_config。

### A.4 执行步骤

1. `grep -rn 'apply_prompt_template(' packages/agent/backend/packages/harness/deerflow/` 确认仍是 3 处生产调用（防代码漂移）。**注意还有 9+ 个本地测试文件调用 `apply_prompt_template`**（`test_lead_prompt_capability_render`、`test_lead_prompt_role_boundaries`、`test_resolved_facts_readback` 等）——本次签名改动是**纯增量**（新增 optional `app_config`，保留 `paradigm/user_id/thread_id`），它们应无需改动，但**必须作为回归全跑**（见 A.5）。
2. 打开上游 `prompt.py` 作参照：`git show deerflow/main:backend/packages/harness/deerflow/agents/lead_agent/prompt.py`。
3. 在本地 `prompt.py`：
   - 把上游的 `_enabled_skills_by_config_cache`、`get_enabled_skills_for_config`、`get_cached_enabled_skills`、`_build_self_update_section`、`_build_available_subagents_description` **按上游实现搬入**（这些是新增、与 Noldus 段正交，可直接照搬上游 body）。
   - 4 个签名加 `app_config` 参数（`_get_memory_context` / `_build_acp_section` / `_build_custom_mounts_section` / `apply_prompt_template`）。`apply_prompt_template` **保留** `paradigm/user_id/thread_id`，**新增** `app_config`。
   - `get_skills_prompt_section` body 把 `_get_enabled_skills()` 改成 `get_enabled_skills_for_config(app_config)`（补线）。
   - `apply_prompt_template` body 内 `_build_subagent_section(n)` → `_build_subagent_section(n, app_config=app_config)`；其他原本用 `get_app_config()` 的地方改 `app_config or get_app_config()`。
   - `_build_subagent_section` 签名加 `*, app_config=None` 并向下传——**保留 Noldus 定制 body**（EthoInsight subagent rendering / orchestration_guide / clarification_system 原样不动）。
   - **保留** Noldus 的 `_get_prior_corrections_context` / `_get_resolved_facts_context` 调用与 `<memory>` 注入。
4. 在 3 个调用点补 `app_config=`：`client.py:259` 传 `self._app_config`；`agent.py` 两处传当前上下文可得的 app_config（无则 `None`，签名默认值兜底）。
5. 验证（见 A.5）。

### A.5 债 A 验收

```bash
cd packages/agent/backend
# 1) 裸导入两生产入口（必 exit 0）
PYTHONPATH=. python -c "import app.gateway"
PYTHONPATH=. python -c "from deerflow.agents import make_lead_agent"
# 2) 采用上游 test_lead_agent_prompt.py（它假设上述符号全在）
git show deerflow/main:backend/tests/test_lead_agent_prompt.py > tests/test_lead_agent_prompt.py
PYTHONPATH=. python -m pytest tests/test_lead_agent_prompt.py -q
# 3) grep 回归：确认 Noldus 段没丢
grep -nE 'prior_corrections|resolved_facts|_build_subagent_section' packages/harness/deerflow/agents/lead_agent/prompt.py | head
# 3b) 本地 prompt 相关测试全跑（9+ 文件调 apply_prompt_template，验签名增量未破它们）
PYTHONPATH=. python -m pytest tests/ -q -k "lead_prompt or lead_agent or resolved_facts" 2>&1 | tail -15
# 4) 全量对比 baseline（不得新增红）
make test 2>&1 | tail -30
```
> 上游 `test_lead_agent_prompt.py` 上游 425 行 / 本地 152 行，含 11 个新测试（`test_build_self_update_section_*`、`test_*_uses_explicit_app_config_*` 等）。这些是债 A 的验收断言。若个别上游测试断言了 Noldus **没有**的上游专属文案，surgical 跳过该断言并在 PR 注明（守「保留本地定制」）。

---

## 债 B：executor skill 加载 async 化

**受保护文件**：`subagents/executor.py`（含 Noldus 定制：`_attempt_auto_seal_from_artifacts`@310、`_preset_handoff_template_if_needed`@1085、`recursion_limit` 计算、handoff validation——这些与 async 化**正交**，原样保留）

### B.1 现状核验（已确认，行号为本地 executor.py）

| 符号 | 本地 | 上游 |
|---|---|---|
| `_build_initial_state` | **sync** def，行 1029，返回 `tuple[dict, list[BaseTool], Any]` | **async** def，上游行 440，返回 `tuple[..., "DeferredToolSetup"]` |
| `_load_skills` | **无** | `async def _load_skills`，上游行 378 |
| `_load_skill_messages` | **无** | `async def _load_skill_messages`，上游行 409 |
| skill 加载 helper | `_load_skill_contents`（sync，行 824） | 被 `_load_skill_messages` 取代 |
| `_build_initial_state` 调用点 | 行 1259，**无 `await`** | 上游行 539，**有 `await`** |

### B.2 依赖说明（关键，旧版 spec 漏了）

上游 `_load_skills` body（上游行 387）调 `get_or_new_skill_storage(app_config=self.app_config)`，`test_subagent_executor.py` 里有 `test_load_skill_messages_uses_explicit_app_config_for_skill_storage`。→ **债 B 解锁 `test_subagent_executor` 需要 `self.app_config` 在 executor 内可用 + skill storage 接受 `app_config`**。**两个前置本地都已具备**：
- 本地 `SubagentExecutor.__init__` **已有** `app_config=None`（executor.py 行 877）+ `self.app_config = app_config`（行 921，docstring 标「Reserved for upstream parity」）。→ **债 B 不需要改构造器**，新的 `_load_skills` 直接用 `self.app_config` 即可。
- skill storage 包已接受 `app_config`（见 A.1）。

**注意**：`test_load_skill_messages_uses_explicit_app_config_for_skill_storage` 这类测试还依赖 `get_or_new_skill_storage(app_config=)` 与 prompt 层 per-config cache 走**同一条** app_config 通路（债 A 的 `get_enabled_skills_for_config`）才语义自洽。**建议债 A 合入后再做债 B。**

### B.3 执行步骤

1. 参照上游：`git show deerflow/main:backend/packages/harness/deerflow/subagents/executor.py`。
2. 搬入 `async def _load_skills` + `async def _load_skill_messages`（照上游 body，含 `asyncio.to_thread(get_or_new_skill_storage, **storage_kwargs)`）。
3. `_build_initial_state` 改 `async def`，body 内对 skill 加载改 `await self._load_skills()` / `await self._load_skill_messages(...)`，替换原 `_load_skill_contents` 用法。
4. 行 1259 调用处 `self._build_initial_state(task)` → `await self._build_initial_state(task)`（它在 `async def _aexecute` 行 1234 内，可直接 await）。
5. 确认 `_aexecute` 之外没有 sync 路径再调 `_build_initial_state`（`grep -n '_build_initial_state(' executor.py`，应只剩 1259 这一处）。
6. **保留** `_attempt_auto_seal_from_artifacts` / `_preset_handoff_template_if_needed` / handoff validation 原样。
7. 验证（见 B.4）。

### B.4 债 B 验收

```bash
cd packages/agent/backend
PYTHONPATH=. python -c "import app.gateway"
PYTHONPATH=. python -c "from deerflow.agents import make_lead_agent"
# 采用上游 test_subagent_executor.py（上游 2410 行 / 本地 1007 行，async 重写）
git show deerflow/main:backend/tests/test_subagent_executor.py > tests/test_subagent_executor.py
PYTHONPATH=. python -m pytest tests/test_subagent_executor.py -q
# 回归：Noldus seal 兜底测试仍绿（先 grep 找出文件名）
grep -rln '_attempt_auto_seal_from_artifacts\|preset_handoff_template' tests/ | xargs -r -I{} sh -c 'PYTHONPATH=. python -m pytest {} -q'
make test 2>&1 | tail -30
```
> ⚠️ 这是改 `subagents/executor.py`——CLAUDE.md 反复强调的导入环高危区。`conftest.py` 为破环 mock 了 `deerflow.subagents.executor`，**pytest 全绿是假绿**；步骤里的裸导入两入口是真正的环检测，**不能省**。

---

## 债 C：langchain / langgraph 生态版本升级（原债3，重新定性）

**受保护文件**：无源码改动。改 `packages/agent/backend/packages/harness/pyproject.toml`（版本下界）+ `packages/agent/backend/uv.lock`（重解析）。

### C.1 真实根因（已坐实，推翻旧版 spec）

旧版 spec 说「`ToolRuntime` 上游接受 `tools`、本地不接受，17 个测试调用点加 `tools=None`，规模小」——**三处错**：

1. **方向反了**：`ToolRuntime` 来自 `langchain.tools`（**库**，不是 deerflow 源码）。本地装的 `langchain==1.2.3` 的 `ToolRuntime.__init__` 签名是 `(state, context, config, stream_writer, tool_call_id, store)`——**不接受 `tools`**。是**上游升级到 `langchain>=1.2.15` 后** `ToolRuntime` 新增了 `tools` 字段。所以是「本地库太旧」，不是「本地代码缺参数」。
2. **不是 17 处、是 1 处**：本地真正受影响的只有 `tests/test_sandbox_middleware.py`（唯一传 `tools=[]` 的测试），而且**已被 #191 显式 skip**：
   ```python
   @pytest.mark.skip(reason="Requires langchain >= 1.2.15 (ToolRuntime tools= kwarg); local pin is 1.2.3. Re-enable after langchain upgrade in a follow-up PR.")
   ```
   实测 `ToolRuntime(..., tools=[])` 在本地 1.2.3 抛 `TypeError: got an unexpected keyword argument 'tools'`。**5 个回退测试文件无一依赖债 C**。
3. **规模不小**：本地 vs 上游是**整个 langchain/langgraph 生态的版本差**，不是单个库。`packages/harness/pyproject.toml` 对比：

   | 包 | 本地下界 | 上游下界 |
   |---|---|---|
   | `langchain` | `>=1.2.3`（锁定解析到 1.2.3） | `>=1.2.15`（解析到 1.2.15） |
   | `langchain-core`（传递） | 1.2.17 | 1.3.3 |
   | `langgraph` | `>=1.0.6,**<1.0.10**` | `>=1.1.9`（**跨 1.0→1.1 minor**） |
   | `langchain-anthropic` | `>=1.3.4` | `>=1.4.1` |
   | `langchain-openai` | `>=1.1.7` | `>=1.2.1` |
   | `langchain-mcp-adapters` | `>=0.1.0` | `>=0.2.2` |
   | `langgraph-api` | `>=0.7.0,<0.8.0` | `>=0.8.1` |
   | `langgraph-runtime-inmem` | `>=0.22.1` | `>=0.28.0` |
   | `langgraph-cli` | `>=0.4.14` | `>=0.4.24` |

   本地 `langgraph<1.0.10` 上界**主动挡住**了新栈。要升 langchain 到 1.2.15+ 很可能连带要放开 langgraph 上界到 1.1.x——这是个 **minor 版本跳跃**，harness 中间件 / GuardrailProvider / checkpointer 都站在 langgraph API 上，blast radius 真实存在。

### C.2 执行步骤（**最后做，独立 PR**）

1. **决策对齐版本**：是「最小升级」（只把 `langchain`/`langchain-core` 升到能用 `tools=`，langgraph 尽量不动）还是「跟齐上游」（langchain+langgraph 全家对齐上游下界）。
   - 跟齐上游更彻底、消除未来 sync 冲突（符合「全量跟随上游底座」策略），但要吃 langgraph 1.0→1.1 的 API 变更。
   - 最小升级 blast radius 小，但可能 langchain 1.2.15 的依赖约束本身就要求 langgraph 1.1（需 `uv` 解析时才知道）。**先试最小升级，解析失败再跟齐。**
2. 改 `packages/agent/backend/packages/harness/pyproject.toml` 对应下界。
3. 重解析锁：
   ```bash
   cd packages/agent/backend
   uv lock --upgrade-package langchain --upgrade-package langchain-core  # 最小升级
   # 若解析报 langgraph 冲突，再放开 langgraph 上界并：
   # uv lock --upgrade-package langgraph --upgrade-package langgraph-api ...
   uv sync
   ```
4. **回归整个 harness**（langgraph 升级是高危）：
   ```bash
   PYTHONPATH=. python -c "import app.gateway"
   PYTHONPATH=. python -c "from deerflow.agents import make_lead_agent"
   make test 2>&1 | tail -40   # 重点看中间件 / guardrail / checkpointer 相关测试有无新红
   ```
5. 升级生效后**移除那条 skip**并跑：
   ```bash
   # 删掉 test_sandbox_middleware.py 里 test_default_lazy_tool_acquisition_uses_async_provider 上的 @pytest.mark.skip
   PYTHONPATH=. python -m pytest tests/test_sandbox_middleware.py -q   # 期望 10 passed, 0 skipped
   ```
6. **务必本地 `make dev` 起一次**确认 langgraph 升级没破运行时（Gateway 起得来、能跑一轮）。
7. 在 PR 注明实际落定的版本组合 + langgraph 是否跨 minor。

### C.3 债 C 验收
- `make test` 全量 ≤ baseline 红数（不新增）。
- `test_sandbox_middleware.py` 的 skip 移除后 10 passed 0 skipped。
- `make dev` 起得来，能完成一轮对话（langgraph minor 升级的运行时验证）。

---

## 债 D：app 层三类缺口

**受保护文件**：无 harness 受保护文件。涉及 `app/gateway/routers/uploads.py`、`app/channels/feishu.py`、`app/channels/manager.py`、`app/gateway/services.py`、`app/gateway/internal_auth.py`。**`sync-deerflow.sh` 不覆盖 `app/`，需手动同步。** 可拆 3 个小 PR（D1/D2/D3），也可合一个。

### D.1 sandbox readable（`uploads.py`）

**现状**：本地有 `_make_file_sandbox_writable`（行 82），**缺** `_make_file_sandbox_readable`；上游有 `_make_file_sandbox_readable`（行 101，加 `S_IRGRP|S_IROTH`），且在上游行 331 `sync_to_sandbox` 路径调 `readable`、行 335 调 `writable`。

**步骤**：
1. 照上游搬入 `_make_file_sandbox_readable`（`git show deerflow/main:backend/app/gateway/routers/uploads.py` 看 body）。
2. 在本地对应的文件写入/同步路径，按上游时序补 `_make_file_sandbox_readable(file_path)` 调用（**先 diff 本地 vs 上游该路径的循环结构**，本地循环可能与上游不完全一致，对齐调用时序而非照抄行号）。
3. 验收：`git show deerflow/main:backend/tests/test_uploads_router.py > tests/test_uploads_router.py`（上游 766 / 本地 640，含 `test_make_file_sandbox_readable_*`）→ `PYTHONPATH=. python -m pytest tests/test_uploads_router.py -q`。

### D.2 Feishu `user_id` 显式传参（`feishu.py` + `manager.py`）

**现状 + 裁决（已调研，结论：采用上游传参，非保留本地）**：
- 本地 `receive_file`（行 311）/`_receive_single_file`（行 334）**无 `user_id` 参数**，内部用 `get_effective_user_id()`（行 373）解析。
- 上游加 `*, user_id: str | None = None`，body 改 `effective_user_id = user_id or get_effective_user_id()`，并在 `manager.py` 新增 `_channel_storage_user_id(msg)` resolver，调用处传 `user_id=storage_user_id`。
- **为什么必须采用上游传参（不是冗余）**：上游 commit 文档明确这修一个真实 bug——dispatcher 线程的 contextvar 未设时 `get_effective_user_id()` 退化成 `"default"`，文件落进 `users/default/...` 而 agent 从 `users/{真实}/...` 读，**文件对不上**。`user_id` 参数携带的是「消息所属/channel owner 的 user」，与 contextvar 不同源。

**步骤**：
1. `receive_file` / `_receive_single_file` 加 `*, user_id: str | None = None`（keyword-only，默认 None，向后兼容）。
2. body 行 373 改 `effective_user_id = user_id or get_effective_user_id()`，下游 `ensure_thread_dirs` / `sandbox_uploads_dir` 用 `effective_user_id`。
3. `manager.py`：照上游搬 `_channel_storage_user_id(msg)`（+其依赖 `_effective_owner_user_id` / `_safe_user_id_for_run`，`git show deerflow/main:backend/app/channels/manager.py` 看），调用处 `channel.receive_file(msg, thread_id, user_id=storage_user_id)`。**若 Noldus 当前没有 per-user IM ownership 映射**，`_channel_storage_user_id` 可先退化为 `_safe_user_id_for_run(msg.user_id)`（无 owner 分支），保证不退化成 `"default"`。
4. 验收：`git show deerflow/main:backend/tests/test_feishu_parser.py > tests/test_feishu_parser.py`（上游 502 / 本地 436，含 `test_feishu_receive_file_syncs_sandbox_with_explicit_user_id`）→ pytest。

### D.3 internal auth + checkpoint regenerate（`internal_auth.py` + `services.py`）

**现状**（已确认本地全缺）：

| 符号 | 本地 | 上游 |
|---|---|---|
| `INTERNAL_OWNER_USER_ID_HEADER_NAME` | **无** | `internal_auth.py:13` = `"X-DeerFlow-Owner-User-Id"` |
| `get_trusted_internal_owner_user_id(request)` | **无** | `internal_auth.py:46` |
| `create_internal_auth_headers` | `()`（行 26，无参） | `(*, owner_user_id: str \| None = None)`（行 28） |
| `apply_checkpoint_to_run_config` | **无** | `services.py:292`（async），调用点 `services.py:467` |
| `INTERNAL_SYSTEM_ROLE` | **已有**（`internal_auth.py:13`，`services.py` 已消费）→ **无需新增** | `internal_auth.py:15` |

> 注：`INTERNAL_SYSTEM_ROLE` 本地已存在，**不在缺口之列**；本债只需补 `INTERNAL_OWNER_USER_ID_HEADER_NAME` + `get_trusted_internal_owner_user_id` + 给 `create_internal_auth_headers` 加 `owner_user_id` 参数 + `apply_checkpoint_to_run_config`。

**步骤**：
1. 照上游搬 `internal_auth.py` 的常量 + `get_trusted_internal_owner_user_id` + 给 `create_internal_auth_headers` 加 `*, owner_user_id=None`（`git show deerflow/main:backend/app/gateway/internal_auth.py`）。
2. 照上游搬 `services.py` 的 `apply_checkpoint_to_run_config`（async）+ 在 start-run 路径（上游行 467）补 `await apply_checkpoint_to_run_config(config, body=body, thread_id=thread_id, request=request)`。**先 diff 本地 `services.py` 该函数上下文**，本地 start-run 逻辑可能有 Noldus 定制，对齐插入点。
3. grep `create_internal_auth_headers(` 所有调用点，确认加 `owner_user_id` 默认值后不破现有调用。
4. 验收：`git show deerflow/main:backend/tests/test_gateway_services.py > tests/test_gateway_services.py`（上游 884 / 本地 346，含 `test_apply_checkpoint_to_run_config_*`、`test_start_run_uses_internal_owner_header_*`）→ pytest。

### D.4 债 D 整体验收
```bash
cd packages/agent/backend
PYTHONPATH=. python -c "import app.gateway"   # app 层改动也要确认 Gateway 可导入
for f in test_uploads_router test_feishu_parser test_gateway_services; do
  PYTHONPATH=. python -m pytest tests/$f.py -q
done
make test 2>&1 | tail -30
```

---

## 测试 → 债 依赖映射（执行 agent 用此判断「采用哪个上游测试需要先做哪条债」）

| 上游测试文件（本地现回退） | 解锁所需债 | 关键新符号 |
|---|---|---|
| `test_lead_agent_prompt.py`（425/152 行，+11 测试） | **A** | `_build_self_update_section`、`get_enabled_skills_for_config`、`get_cached_enabled_skills`、`_enabled_skills_by_config_cache`、`apply_prompt_template(app_config=)`、`get_skills_prompt_section` 接通 app_config |
| `test_subagent_executor.py`（2410/1007 行，async 重写） | **B**（+ A 的 skill-storage app_config 通路） | `async _load_skills`、`async _load_skill_messages`、`async _build_initial_state`、构造器 `app_config=` |
| `test_gateway_services.py`（884/346 行，+34 测试） | **D.3** | `apply_checkpoint_to_run_config`、`INTERNAL_OWNER_USER_ID_HEADER_NAME`、`INTERNAL_SYSTEM_ROLE`、`create_internal_auth_headers(owner_user_id=)` |
| `test_uploads_router.py`（766/640 行，+7 测试） | **D.1** | `_make_file_sandbox_readable` |
| `test_feishu_parser.py`（502/436 行，+3 测试） | **D.2** | `receive_file(user_id=)`、`_receive_single_file(user_id=)` |
| `test_sandbox_middleware.py` 的 1 个 skip 测试（**不在回退之列**） | **C** | langchain≥1.2.15 的 `ToolRuntime(tools=)` |

> **采用上游测试的统一手法**：`git show deerflow/main:backend/tests/<file> > packages/agent/backend/tests/<file>`，跑通后若个别断言依赖 Noldus **没有**的上游专属行为，surgical 删该断言并在 PR 注明（守「保留本地定制」铁律，不为迁就上游测试改 Noldus 业务）。

---

## 完成标准（DoD）

1. 4 条债各自 PR 合入 dev，每个 PR：
   - 裸导入两生产入口 exit 0（debt A/B/D 必跑）；
   - 对应上游测试文件采用后全绿（或 surgical 注明跳过的断言）；
   - `make test` 全量 ≤ 开工 baseline 红数（无新增红）；
   - 债 C 额外：`make dev` 起得来 + 跑通一轮。
2. 5 个回退测试文件全部替换回上游版并通过（债 C 不涉及这 5 个，单独验 skip 解除）。
3. 本地 harness 接口面追平 `deerflow/main`，后续 sync（11415875 之后）不再因预存偏离暴露红测试。

## 附：核验命令速查（执行前复跑，防代码漂移）

```bash
cd /home/wangqiuyang/noldus-insight
git fetch deerflow main
P=packages/agent/backend/packages/harness/deerflow
# 债 A：4 签名 + 5 符号
grep -nE 'def (apply_prompt_template|_get_memory_context|_build_acp_section|_build_custom_mounts_section)\b' $P/agents/lead_agent/prompt.py
for s in _enabled_skills_by_config_cache get_enabled_skills_for_config get_cached_enabled_skills _build_self_update_section _build_available_subagents_description; do printf "%s local=" "$s"; grep -c "$s" $P/agents/lead_agent/prompt.py; done
grep -rn 'apply_prompt_template(' packages/agent/backend/packages/harness/deerflow/ | grep -v 'def apply_prompt_template'   # 应 3 处（client.py + agent.py×2）
# 债 B：executor async
grep -nE '(async +def|def) _build_initial_state|_load_skills|_load_skill_messages|_build_initial_state\(' $P/subagents/executor.py
# 债 C：版本
grep -nE 'langchain|langgraph' packages/agent/backend/packages/harness/pyproject.toml
git show deerflow/main:backend/packages/harness/pyproject.toml | grep -nE 'langchain|langgraph'
# 债 D：app 层符号
grep -nE '_make_file_sandbox_readable|_make_file_sandbox_writable' packages/agent/backend/app/gateway/routers/uploads.py
grep -nE 'def receive_file|def _receive_single_file' packages/agent/backend/app/channels/feishu.py
grep -nE 'INTERNAL_OWNER_USER_ID_HEADER_NAME|get_trusted_internal_owner_user_id|apply_checkpoint_to_run_config' packages/agent/backend/app/gateway/internal_auth.py packages/agent/backend/app/gateway/services.py
```
