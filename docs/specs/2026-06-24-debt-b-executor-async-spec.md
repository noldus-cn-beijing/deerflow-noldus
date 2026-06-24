# 债 B：executor skill 加载 async 化（独立可执行 spec）

> **母 spec**：`docs/specs/2026-06-24-deerflow-sync-historical-debt.md` 的「债 B」小节（§B.1–B.4）。本文从母 spec 拆出，自包含、可直接交给 agent 执行。
>
> **解锁目标**：采用上游 `tests/test_subagent_executor.py`（上游 2410 行 / 本地 1007 行，async 重写）并通过。
>
> **规模**：中。改 1 个受保护文件 `subagents/executor.py`。

---

## 〇、给实施 agent 的一句话

把 `SubagentExecutor` 的 skill 加载与 `_build_initial_state` 从 sync 改成 async（搬入上游 `async _load_skills` + `async _load_skill_messages`，`_build_initial_state` 加 `await`），让本地 executor 接口面追平上游，解锁上游 `test_subagent_executor.py`——**保 Noldus 定制（auto-seal / preset_handoff / recursion_limit / handoff validation）原样不动**。

---

## ⚠️ 给执行 agent 的前置须知（务必先读）

1. **路径前缀**：本地 harness 是 `packages/agent/backend/packages/harness/deerflow/`；上游对应去掉 `packages/agent/`，用 `git show deerflow/main:backend/<path>` 读取。
2. **commit SHA 不可信，一律以「符号/签名 diff」为准**。`packages/agent/` 是 subtree fork，历次 sync 没 squash，`git log` 上的 SHA 对不上上游逻辑 commit。**别 `git cherry-pick` 任何 SHA**，看 `git show deerflow/main:<file>` 当前态即可。
3. **改 executor 是 CLAUDE.md 反复强调的导入环高危区**。`conftest.py` 为破环 mock 了 `deerflow.subagents.executor`，**pytest 全绿是假绿**。改完必须裸导入两生产入口（backend/ 下，无 conftest）：
   ```bash
   PYTHONPATH=packages/harness:. python -c "import app.gateway"
   PYTHONPATH=packages/harness:. python -c "from deerflow.agents import make_lead_agent"
   ```
   两者 exit 0 才算过。新 helper 的 import 放函数体内惰性，别放模块顶层。
4. **本地已知 baseline 污染**：backend 全量跑有确定性失败（test isolation 污染 + 需 live server 的 test_client_live）。开工前先记 baseline，判断「我有没有引入新红」用**对比 baseline 数字**，不是「全量全绿」。当前 baseline = **18 红**（债 A 已合后）。
5. **运行测试的解释器**：`packages/agent/backend/.venv/bin/python -m pytest`，在 `packages/agent/backend/` 目录下，需 `PYTHONPATH=packages/harness:.`。
6. **worktree 跑测试必带 config env**（否则 `config.yaml not found` 假红污染判断）：
   ```bash
   DEER_FLOW_CONFIG_PATH=/home/wangqiuyang/noldus-insight/packages/agent/config.yaml
   ```

---

## 一、现状核验（2026-06-24 已逐符号确认，行号为本地 executor.py）

| 符号 | 本地 | 上游 |
|---|---|---|
| `_build_initial_state` | **sync** def，行 1029，返回 `tuple[dict, list[BaseTool], Any]` | **async** def，返回 `tuple[..., "DeferredToolSetup"]` |
| `_load_skills` | **无** | `async def _load_skills`（上游行 378） |
| `_load_skill_messages` | **无** | `async def _load_skill_messages`（上游行 409） |
| skill 加载 helper | `_load_skill_contents`（sync，行 824，**纯文本 append**） | 被 `_load_skill_messages` 取代（**per-session SystemMessage 注入**，Codex 模式） |
| `_build_initial_state` 调用点 | 行 1259，**无 `await`**（在 `async _aexecute` 内） | 上游有 `await` |

**前置依赖已具备（债 A 合入后）**：
- `SubagentExecutor.__init__` 已有 `app_config=None`（行 877）+ `self.app_config = app_config`（行 921，docstring 标「Reserved for upstream parity」）。→ **不需要改构造器**，新的 `_load_skills` 直接用 `self.app_config`。
- skill storage 包已接受 `app_config`（`get_or_new_skill_storage(app_config=)`）。

---

## 二、⚠️ 最大风险：上游 `_build_initial_state` 是**重写**不是**微调**（受保护文件）

本地 `_build_initial_state`（行 1029）与上游差异**很大**：

| 维度 | 本地 | 上游 |
|---|---|---|
| skill 加载方式 | `_load_skill_contents(self.config.skills)` → 纯文本 append 进 system_prompt | `_load_skills()` 拉元数据 + `_load_skill_messages()` → **每条 skill 一个 `<skill>` SystemMessage**（Codex 模式） |
| deferred tools 过滤时机 | 先 skill 文本、后 deferred（`enabled = get_app_config().tool_search.enabled`） | 先 skill 元数据 → `_apply_skill_allowed_tools` **policy 过滤** → 再 deferred（`enabled = (self.app_config or get_app_config()).tool_search.enabled`） |
| 工具过滤 | `assemble_deferred_tools(self.tools, ...)`（不过滤） | `assemble_deferred_tools(filtered_tools, ...)`（经 `_apply_skill_allowed_tools` 过滤） |

**这意味上游对 skill 注入做了一次架构升级（text-append → message-injection + tool policy 过滤）**。本债要决定：

- **方案 B-min（推荐，守受保护）**：只做「async 化」这一层——`_build_initial_state` 改 `async def`，内部 skill 加载改 `await self._load_skill_messages(await self._load_skills())`（搬入上游两个 async helper），调用点加 `await`。**保留本地的 system_prompt 组装顺序 / deferred 段逻辑 / Noldus 定制**。`_apply_skill_allowed_tools` 的 tool policy 过滤可作为可选增强（若上游测试断言要求）。
- **方案 B-full（风险高）**：整段照搬上游 `_build_initial_state` body（含 message-injection + tool policy 过滤）。**不推荐**——上游 body 与本地 Noldus system_prompt 组装逻辑交织，整段替换易丢定制。

**裁决依据**：先采用上游 `test_subagent_executor.py`，跑通；若个别测试断言依赖 message-injection 模式（如断言 `<skill>` 标签独立 SystemMessage），surgical 升级到 B-full 的对应片段，**在 PR 注明**。先 B-min 跑绿多少算多少，别一上来 B-full。

---

## 三、改动清单（`packages/agent/backend/packages/harness/deerflow/subagents/executor.py`）

1. **搬入上游两个 async helper**（照上游 body）：
   - `async def _load_skills(self) -> list[Skill]`：`asyncio.to_thread(get_or_new_skill_storage, **({"app_config": self.app_config} if self.app_config else {}))` → `storage.load_skills(enabled_only=True)` → 按 `self.config.skills` 白名单过滤。
   - `async def _load_skill_messages(self, skills) -> list[SystemMessage]`：每条 skill 读 `skill.skill_file.read_text` 包 `<skill name="...">`。
   - 上游还有 `_apply_skill_allowed_tools`（`filter_tools_by_skill_allowed_tools(self._base_tools, skills)`）——若走 B-full 才需要，先确认 `filter_tools_by_skill_allowed_tools` 本地是否已存在（grep），缺则一并搬。
2. **`_build_initial_state` 改 `async def`**：内部 skill 加载段从 `_load_skill_contents(self.config.skills)` 改为 `await self._load_skill_messages(await self._load_skills())`，把返回的 `list[SystemMessage]` 内容 append 进 `system_parts`（保留本地「合并成单个 SystemMessage 避免 multi-SystemMessage 被拒」的组装策略）。deferred 段 `enabled` 改 `(self.app_config or get_app_config()).tool_search.enabled`。
3. **调用点加 `await`**：行 1259 `self._build_initial_state(task)` → `await self._build_initial_state(task)`（它在 `async def _aexecute` 内，可直接 await）。
4. **确认无 sync 路径再调 `_build_initial_state`**：`grep -n '_build_initial_state(' executor.py`，应只剩 1259 这一处（在 `_aexecute` 内）。
5. **保留 Noldus 定制原样不动**（这些与 async 化正交）：
   - `_attempt_auto_seal_from_artifacts`（@310 区域）
   - `_preset_handoff_template_if_needed`（@1085 区域）
   - `recursion_limit` 计算
   - handoff validation
6. **`_load_skill_contents`（行 824）去留**：若新 async helper 取代了它，且全文件再无调用（grep 确认），删除；若仍被别处用（如 `get_deferred_tools_prompt_section` 之类），保留。**别为删而删，grep 决定**。

---

## 四、测试（TDD，红→绿）

1. **采用上游测试**：
   ```bash
   git show deerflow/main:backend/tests/test_subagent_executor.py > tests/test_subagent_executor.py
   PYTHONPATH=packages/harness:. DEER_FLOW_CONFIG_PATH=<abs config.yaml> python -m pytest tests/test_subagent_executor.py -q
   ```
   上游 2410 行 / 本地 1007 行。关键新符号断言：`async _load_skills`、`async _load_skill_messages`、`async _build_initial_state`、构造器 `app_config=`、`test_load_skill_messages_uses_explicit_app_config_for_skill_storage`。
2. **若个别上游测试断言依赖 Noldus 没有的上游专属行为**（如 message-injection 独立 SystemMessage 的精确格式、`_apply_skill_allowed_tools` policy 过滤后的工具集），surgical 跳过该断言/测试并 **PR 注明**（守「保留本地定制」铁律）。**优先级**：先 B-min 跑，失败测试逐个看是「async 化没做对」（要修代码）还是「上游架构升级 Noldus 没跟」（要 surgical skip）。
3. **回归 Noldus seal 兜底测试**：
   ```bash
   grep -rln '_attempt_auto_seal_from_artifacts\|preset_handoff_template\|_RECONSTRUCTABLE' tests/ | \
     xargs -r -I{} sh -c 'PYTHONPATH=packages/harness:. python -m pytest {} -q'
   ```
   全绿才算 Noldus 定制没被 async 化碰坏。

---

## 五、验收标准

1. ✅ 裸导入两生产入口 exit 0（`import app.gateway` / `from deerflow.agents import make_lead_agent`）。
2. ✅ `test_subagent_executor.py` 采用上游版后全绿（或 surgical 注明跳过的断言）。
3. ✅ Noldus seal 兜底相关测试（`_attempt_auto_seal_from_artifacts` / `preset_handoff_template`）全绿。
4. ✅ `make test` 全量 ≤ 开工 baseline 红数（无新增红）。判断法：worktree 全量红集 ⊆ baseline 红集（comm 对比，wt-only 新增 = 0）。
5. ⏳ 可选 manual：若走 B-full（message-injection），dogfood 一轮确认 subagent skill 注入正常。

---

## 六、风险与注意事项

1. **导入环**（CLAUDE.md 铁律）：新 helper 里 `from deerflow.skills.storage import get_or_new_skill_storage` 已在上游放函数体内（lazy），**照搬，别提到模块顶层**。改完必裸导入两入口。
2. **受保护文件 surgical**：`executor.py` 含 Noldus auto-seal / preset_handoff / recursion_limit / handoff validation。**绝不整文件覆盖**，只改 skill 加载段 + async 化 + 调用点 await。
3. **B-min vs B-full 取舍**：先 B-min 跑测试，失败再判是否需 B-full 片段。**别预先上 B-full**（Catastrophic forgetting 风险：丢 Noldus system_prompt 组装）。
4. **`test_load_skill_messages_uses_explicit_app_config_for_skill_storage`** 依赖 `self.app_config` → `get_or_new_skill_storage(app_config=)` 链路。本债不改构造器（已有），但若该测试要求 storage 返回带 app_config 标记的 skills，确认 storage 包行为（债 A 已确认 storage 接 app_config）。
5. **与债 A 的关系**：本债代码依赖（`self.app_config` + storage 接 app_config）债 A 之前就具备，**不阻塞**。但若债 A 还没合，从 dev 分支做，A 合后 rebase（文件不重叠，无冲突）。

---

## 附：核验命令速查（开工前复跑，防代码漂移）

```bash
cd /home/wangqiuyang/noldus-insight
P=packages/agent/backend/packages/harness/deerflow
grep -nE '(async +def|def) _build_initial_state|_load_skill_contents|_load_skills|_load_skill_messages' $P/subagents/executor.py
grep -n '_build_initial_state(' $P/subagents/executor.py   # 应只剩调用点 1 处 + 定义 1 处
grep -nE 'app_config=None|self\.app_config' $P/subagents/executor.py
git show deerflow/main:backend/packages/harness/deerflow/subagents/executor.py | sed -n '378,545p'  # 上游 _load_skills / _load_skill_messages / _build_initial_state
```
