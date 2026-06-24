# 2026-06-24 DeerFlow 同步历史债清理 — 债 A 已合，B/D/C 待做（交接）

> **交接对象**：接手债 B / D / C 的 agent（可并行）。
> **母 spec**：[`docs/specs/2026-06-24-deerflow-sync-historical-debt.md`](../../specs/2026-06-24-deerflow-sync-historical-debt.md)
> **三份独立可执行 spec**（都已合 dev `624258f5`）：
> - 债 B：[`2026-06-24-debt-b-executor-async-spec.md`](../../specs/2026-06-24-debt-b-executor-async-spec.md)
> - 债 D：[`2026-06-24-debt-d-app-layer-spec.md`](../../specs/2026-06-24-debt-d-app-layer-spec.md)
> - 债 C：[`2026-06-24-debt-c-langchain-upgrade-spec.md`](../../specs/2026-06-24-debt-c-langchain-upgrade-spec.md)

---

## 1. 当前任务目标

清理 sync PR #191 暴露的 5 条历史债（本地 harness 落后 `deerflow/main` `11415875`）。本地 harness 接口面追平上游后，后续 sync 不再因预存偏离冒红测试。4 债各独立 PR：

- **债 A**（app_config 贯通 + skills per-config cache）— ✅ **已合 dev**（PR #194，b1a4a4c6）
- **债 B**（executor skill 加载 async 化）— ⏳ 待做，**worktree 已建**（`worktree-debt-b-executor-async`，分支已从 dev 拉但**尚无任何改动**）
- **债 D**（app 层 D1 sandbox readable / D2 Feishu user_id / D3 internal auth + checkpoint）— ⏳ 待做，可拆 3 小 PR
- **债 C**（langchain/langgraph 生态版本升级）— ⏳ **最后做**，最高风险，须 B/D 全合后

---

## 2. 当前进展

### ✅ 已完成（债 A，PR #194 已 MERGED）
- `app_config` 参数从 `DeerFlowClient → make_lead_agent → apply_prompt_template → 各 prompt 段`全链贯通（config 获取走 `app_config or get_app_config()`）。
- prompt.py 搬入 5 个 per-config skills 符号 + 4 签名加 app_config；保留 Noldus 的 `paradigm/user_id/thread_id`（注入 prior_corrections + resolved_facts + `<memory>`）。
- **顺手修一处 #191 暴露的 `build_middlewares` 单边合债**：`client.py` 早已传 `build_middlewares(..., available_skills=, app_config=)` 但函数不收 → `DeerFlowClient`（Gateway-embedded 默认部署）每次建 agent `TypeError`。按上游对齐签名 + body 内 `get_app_config()` 改 `resolved_app_config`。`make_lead_agent` 两处调用同步补 `app_config=`。（此教训已进 memory：`feedback_sync_signature_drift_caller_exists_but_callee_missing_kwarg`）
- 采用上游 `test_lead_agent_prompt.py`（13 passed 4 skipped）+ 修两处旧 no-arg lambda。
- 验收：裸导入两入口 exit 0；全量无新增红（wt-only=0）。

### ✅ 已完成（spec 拆分）
- 三份独立 spec 已合 dev（`624258f5`），母 spec 顶部加了 cross-ref 指针。

### ⏳ 进行中 / 待做
- **债 B**：worktree `worktree-debt-b-executor-async` 已建（分支 = dev HEAD），**尚未动代码**。接手 agent 直接进该 worktree 干。
- **债 D**：无 worktree，需新建（从 dev）。
- **债 C**：无 worktree，**须等 B/D 全合 dev 后再建**。

---

## 3. 关键上下文

### 路径前缀
- 本地 harness：`packages/agent/backend/packages/harness/deerflow/`
- 本地 app 层：`packages/agent/backend/app/`
- 本地测试：`packages/agent/backend/tests/`
- 上游对应：去掉 `packages/agent/`，用 `git show deerflow/main:backend/<path>` 读。

### commit SHA 不可信
`packages/agent/` 是 subtree fork，历次 sync 没 squash，`git log` SHA 对不上上游逻辑 commit。**别 `git cherry-pick`，看 `git show deerflow/main:<file>` 当前态。**

### 测试姿势（worktree 必读，否则假红/假绿）
```bash
cd packages/agent/backend
VENV=.venv  # 或绝对路径 /home/wangqiuyang/noldus-insight/packages/agent/backend/.venv
# worktree 必带 config env（否则 config.yaml not found 假红）
DEER_FLOW_CONFIG_PATH=/home/wangqiuyang/noldus-insight/packages/agent/config.yaml \
PYTHONPATH=packages/harness:. $VENV/bin/python -m pytest tests/<file>.py -q -p no:cacheprovider
```

### 改 harness 核心后必须裸导入两生产入口（CLAUDE.md 铁律）
`conftest.py` mock 了 `deerflow.subagents.executor` 破导入环，**pytest 全绿是假绿**。改完 `subagents/` / `tools/builtins/` / `agents/` 后必跑：
```bash
PYTHONPATH=packages/harness:. python -c "import app.gateway"
PYTHONPATH=packages/harness:. python -c "from deerflow.agents import make_lead_agent"
```
两者 exit 0 才算过。永久 guard：`tests/test_gateway_import_no_cycle.py`。

### baseline（判断"新红"的对账基准）
当前全量 baseline = **18 红**（含 test isolation 污染 + 需 live server 的 `test_client_live`）。判"我有没有引入新红"用**对比红集**（worktree 全量红 vs baseline 红，wt-only 新增应 = 0），不是"全量全绿"。单跑全绿、全量必红的测试（deferred_tool / inspect_gate / paradigm_gate 的 test_async_delegates_to_sync 等）是已知污染，别归因自己改坏。

---

## 4. 关键发现（务必读，避免重蹈覆辙）

1. **债 B 的 `_build_initial_state` 是重写不是微调**：上游把 skill 加载从 text-append 升级到 message-injection（每条 skill 一个 `<skill>` SystemMessage）+ tool policy 过滤。本地是受保护文件（含 Noldus auto-seal / preset_handoff / recursion_limit / handoff validation）。spec 给了 **B-min（只 async 化保 Noldus 组装）/ B-full（整段照搬）** 两方案，**先 B-min 跑测试再按需升级**，别一上来 B-full（catastrophic forgetting 风险）。

2. **债 D.2 Feishu user_id 必须采用上游传参**（修真实 bug：dispatcher 线程 contextvar 未设 → `get_effective_user_id()` 退化 `"default"` → 文件落 `users/default/` 而 agent 读 `users/{真实}/`，对不上）。但前提是**先 grep 确认 Noldus 有无 per-user IM ownership 映射**——无则退化分支（`_safe_user_id_for_run(msg.user_id)`），别假设。

3. **债 C 最高风险**：langgraph 跨 **1.0→1.1 minor**，harness 中间件 / GuardrailProvider / checkpointer 全站在 langgraph API 上。`uv.lock` 解析会被并发改依赖的 PR 干扰。**顺序硬约束：B/D 全合 dev 后才做 C**，否则全量红数对账分不清"C 升级破坏"vs"B/D 改动"。`make dev` 运行时验证不可省（import 不崩 ≠ 运行时不崩）。

4. **采用上游测试的统一手法**：`git show deerflow/main:backend/tests/<file> > tests/<file>`，跑通后若断言依赖 Noldus 没有的上游专属行为，surgical 删该断言 + PR 注明（守"保留本地定制"）。

5. **三债文件零重叠，B+D 可并行，C 串行最后**：
   - 债 B 改 `subagents/executor.py`（harness）
   - 债 D 改 `app/channels/{feishu,manager}.py` + `app/gateway/{internal_auth,services}.py` + `app/gateway/routers/uploads.py`（app 层）
   - 债 C 改 `packages/harness/pyproject.toml` + `uv.lock`

---

## 5. 未完成事项（按优先级）

| 优先级 | 债 | 状态 | 动作 |
|---|---|---|---|
| **P0** | B | worktree 已建未动 | 进 `worktree-debt-b-executor-async`，按 spec §三 改 executor.py（先 B-min） |
| **P0** | D | 无 worktree | 从 dev 建 worktree，按 spec D1→D2→D3（可拆 3 PR） |
| **P1** | C | 等 B/D 合后 | 最高风险，独立 PR + `make dev` 验运行时 |

---

## 6. 建议接手路径

### 接债 B 的 agent
1. `cd .claude/worktrees/worktree-debt-b-executor-async`（worktree 已存在，分支已从 dev 拉）。
2. **先 `git pull --rebase origin dev`** 把刚合的 spec（624258f5）拉进来，读 `docs/specs/2026-06-24-debt-b-executor-async-spec.md`。
3. 记 baseline：`DEER_FLOW_CONFIG_PATH=<abs> PYTHONPATH=packages/harness:. .venv/bin/python -m pytest tests/ -q | grep FAILED` → 存红集。
4. 按 spec §三 改 `subagents/executor.py`（搬 `async _load_skills` + `async _load_skill_messages`，`_build_initial_state` 改 async + 调用点 await）。**先 B-min**。
5. 采用上游 `test_subagent_executor.py` 跑；裸导入两入口；回归 Noldus seal 兜底测试；全量对比 baseline。

### 接债 D 的 agent
1. `git worktree add .claude/worktrees/worktree-debt-d-app-layer -b worktree-debt-d-app-layer origin/dev`（从含 spec 的 dev 拉）。
2. 读 `docs/specs/2026-06-24-debt-d-app-layer-spec.md`。D 完全独立，可放心与 B 并行。
3. 按 D1→D2→D3 顺序，每个子债采用上游对应测试验收。

### 接债 C 的 agent
1. **先确认 B 和 D 都已合 dev**（`git log dev --oneline | grep -E 'debt-b|debt-d'`）。
2. `git worktree add .claude/worktrees/worktree-debt-c-langchain -b worktree-debt-c-langchain origin/dev`。
3. 读 `docs/specs/2026-06-24-debt-c-langchain-upgrade-spec.md`，**严格按 §二 先试最小升级**。
4. `make dev` 起一次跑一轮对话（不可省）。

---

## 7. 风险与注意事项

- **executor.py 是导入环高危区**（债 B）：新 helper 的 `from deerflow.skills.storage import ...` 放函数体内 lazy，别提模块顶层。改完裸导入两入口。
- **受保护文件绝不整文件覆盖**（债 B 的 executor.py、债 D 的 feishu/manager/services）：先 `diff <(git show deerflow/main:<path>) <local>`，手工搬上游修复点保留 Noldus 定制。
- **债 C 不要碰 5 个回退测试文件**（test_lead_agent_prompt / test_gateway_services / test_subagent_executor / test_uploads_router / test_feishu_parser）——它们归 B/D。
- **worktree 跑测试必带 `DEER_FLOW_CONFIG_PATH` + `PYTHONPATH=packages/harness:.`**，否则假红。
- **别凭"全量全绿"判断成功**：用红集对比（wt-only 新增 = 0）。

---

## 8. 下一位 Agent 的第一步建议

**最高价值动作**：进 `worktree-debt-b-executor-async` worktree，`git pull --rebase origin dev`，读 [债 B spec](../../specs/2026-06-24-debt-b-executor-async-spec.md) §〇+§二，记 baseline，然后按 §三 开干（先 B-min）。

并行可同时开债 D worktree（完全独立，零冲突）。

债 C 等 B/D 都合 dev 后再开，别抢跑（顺序硬约束，抢跑会污染全量对账）。

---

## milestone 建议

本次会话推进了 **DeerFlow 同步历史债清理** track：债 A 已合 dev（PR #194），债 B/D/C spec 已就绪可并行。建议 milestone 记录：sync-state 仍 = 11415875（#191），债 A/B/C/D 是 #191 标注的本地 harness 历史债清理，目标是"本地接口面追平上游 tip，后续 sync 不再因预存偏离冒红"。债 A ✅ / B-D 进行中 / C 待最后。
