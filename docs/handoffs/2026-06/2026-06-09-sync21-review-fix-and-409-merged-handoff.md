# Handoff — Spec ① sync-21 review+修复完成（待 PR）+ Spec ② 409 已合 main + 两 spec 已写

> 日期：2026-06-09 ｜ 写给下一位接手的 AI Agent（不是给用户的总结）
> 上一会话做了三件事：① 写两份 spec（sync 21 commit + 409 修复）② review 用户的 409 实施并验收 ③ review 用户的 sync-21 实施、**发现并修复 5 处回归**、commit+push。

---

## 0. 一句话现状

- **Spec ②（409 多 worker 切 thread）**：用户已实施 + 你上一会话 review 通过 + 用户已 **merge 到 main**。✅ 完结。
- **Spec ①（sync 上游 21 commit）**：用户已实施（`be2c263e`），你 review 发现 sync 洗掉 2 处 Noldus 定制 + 测试套未对齐，**已修复并 push**（`737ce855`）。**待用户建 PR review 后合 dev**。⏳
- 两份 spec 文档已写在 `docs/superpowers/specs/2026-06-09-*.md`。

---

## 1. 仓库/分支坐标（已核实）

- 主仓库 `dev` HEAD：**`1a819e7e`**（**409 和 sync-21 都还没合进 dev**，都在各自分支待 PR）。
- 上游 `deerflow/main` HEAD：`f92a26d5`（21 commit 已被 sync-21 吃下）。
- **关键 worktree**：
  | worktree | 分支 | HEAD | 状态 |
  |---|---|---|---|
  | `/home/wangqiuyang/noldus-insight` | dev | `1a819e7e` | 主仓库 |
  | `/home/wangqiuyang/noldus-insight-sync-21` | `sync-deerflow-21-commits` | **`737ce855`** | **已 push，待 PR** |
  | `/home/wangqiuyang/noldus-insight-fix-409` | `fix-gateway-multiworker-409` | `35455dfe` | 409 修复（已合 main） |
- sync-21 分支已 push 且与 `origin/sync-deerflow-21-commits` 同步。PR 链接：`https://github.com/noldus-cn-beijing/noldus-insight/pull/new/sync-deerflow-21-commits`

---

## 2. Spec ① sync-21 —— 已完成的工作（✅）

### 2.1 用户的 sync commit `be2c263e`（21 commit，74e3e80c→f92a26d5）
- 45 文件改动（42 harness + `.deerflow-sync-state` 推进到 f92a26d5 + `contracts/subagent_status_contract.json` + 新 test）。
- **源码层 surgical 做得 A 级**（我逐个核实）：9 受保护文件双向都对（Noldus 保住 + 上游 fix 落地）；executor.py 循环导入红线（`seal_handoff_tools` 惰性 import 在函数体）+ auto-seal 全保；deferred-tool 重构（`2bbc7879`/`3b6dd0a4`/`d9f47249`）一致（旧 ContextVar API 在 tool_search.py 外清零）；两生产入口裸导入 0 退出。

### 2.2 我（上一会话）修复的 5 处回归 → commit `737ce855`（已 push）
sync **源码对**，但栽在 **full-follow 洗掉非受保护但有 Noldus 定制的文件** + **backend/tests/ 在 sync subtree 外没对齐**：

1. **`memory/prompt.py`（P0 生产回归）**：上游 `51920072` 重写覆盖了 2026-05-13 topOfMind/history 隔离定制（`format_memory_for_injection` 砍会话级字段防"已上传 X 文件"跨 thread 泄漏致文件幻觉）。**surgical 重打回隔离逻辑 + 保上游 tiktoken offload**。
2. **`config/app_config.py`（P0）**：`handoff_strict_mode` 字段被覆盖丢失（`experiment_context.py:54` 依赖）。**合回字段 + `Literal` import**。
3. **上述两文件加进 `scripts/sync-deerflow.sh` 的 `PROTECTED_FILES`**（防下次再洗）。
4. **deferred-tool 重构测试对齐**：删 `test_deferred_tool_registry_promotion.py`（上游重构已删）；`test_tool_search.py`/`test_checkpointer.py`/`test_deferred_tool_promotion_real_llm.py` 换上游当前版；搬上游 4 新测试（`test_deferred_{catalog,setup,filter_middleware}`/`test_thread_state_promoted`）；修 `test_lead_agent_{model_resolution,prompt}` mock lambda 签名。
5. **upload size 契约跨层漂移（真实 bug）**：上游 `1aac408d` 把 size 改 `int`，横跨 harness(已同步)+`app/`(未同步)。`app/gateway/routers/uploads.py` surgical 合契约（`UploadedFileInfo`/`UploadListResponse` + size int + list response_model，**保 41 行 Noldus 沙箱可读/去重定制**）；`test_uploads_router.py` 改 Pydantic 属性访问。

### 2.3 验收（我亲自重跑，已坐实）
- 全量 **3850 passed / 3 known baseline red / 21 skipped**（135s）。
- 3 baseline red = `test_chart_maker_config_basic_fields` + `inspect_gate_guardrail`/`paradigm_identification_gate` 的 `test_async_delegates_to_sync`。**原 spec 说 5 红，其中 2 个 `deferred_tool_registry_promotion` 因上游删测试而合法消失 → 现在 3 红是对的**。
- 两生产入口裸导入 0 退出；`test_gateway_import_no_cycle` 绿；ruff 干净。

---

## 3. ⚠️ 验证 sync-21 的命门（接手必读，否则验错代码库）

**venv 的 `.pth` 陷阱**：`_editable_impl_deerflow_harness.pth` 把 `deerflow` 包**硬指向主仓** `/home/wangqiuyang/noldus-insight/...`，**不是 worktree**。所以从 worktree 跑 `PYTHONPATH=. python ...` 会**测主仓 dev 的代码，不是 sync-21 的代码**——"通过"是假的。

**正确姿势**（验证 sync-21 任何东西都要这样）：
```bash
cd /home/wangqiuyang/noldus-insight-sync-21/packages/agent/backend
VENV=/home/wangqiuyang/noldus-insight/packages/agent/backend/.venv
WTHARNESS=/home/wangqiuyang/noldus-insight-sync-21/packages/agent/backend/packages/harness
# 强制 worktree deerflow 在 .pth 之前
PYTHONPATH="$WTHARNESS:." "$VENV/bin/python" -c "import deerflow; assert 'sync-21' in deerflow.__file__; import app.gateway; print('OK')"
PYTHONPATH="$WTHARNESS:." "$VENV/bin/python" -m pytest tests/ -q -p no:cacheprovider
```
worktree 已有 `config.yaml` 软链（`packages/agent/config.yaml -> 主仓`），所以 config 依赖测试不假红。worktree **没有自己的 venv**，复用主仓 venv。

---

## 4. 未完成事项（按优先级）

### P0 — 用户操作（不是 agent 自动做）
- **建 sync-21 的 PR**：`sync-deerflow-21-commits` → 先进 `dev`，review 后再 PR 到 `main`。**别让 agent 自动 merge**（仓库约定：先进 dev、commit message 中文、用户发起合并）。两 commit：`be2c263e`(sync) + `737ce855`(回归修复)。
- **合 409**：用户说已 merge 到 main，但**主仓 `dev` 还是 `1a819e7e`**——确认 409 是否也要回灌 dev（main 与 dev 可能分叉）。接手时先 `git log --oneline origin/main -5` 核实 409 在不在 main，再决定 dev 是否需要 cherry-pick/merge。

### P1 — 接手可做的核查（如果用户要你继续 sync-21）
- **dogfood sync-21**（合 dev 前最好做）：`make dev`（Gateway-embedded）实起，发一条 EPM 分析跑通 lead→subagent，确认 deferred-tool 重构 + memory 隔离 + handoff 在真实链路 OK（测试绿 ≠ E2E 绿）。
- **前端 deferred-tool / status_contract 配套**：`8d2e55a0`（结构化 subagent_status）的**前端**消费在 deerflow 前端 subtree，本次只同步了 backend harness。若前端切走/卡片状态有问题，查前端 `subagent_status` 读取是否也要同步（记为 follow-up，backend 向后兼容不阻塞）。

### P2 — 技术债（独立，不阻塞 sync-21）
- 3 个 baseline red（`async_delegates_to_sync`×2 + `chart_maker_config`）是 test isolation 污染/已知债，**别归因 sync-21**（合 dev 前在父 commit 建 detached worktree 全量跑可坐实同样红）。见 memory `feedback_known_full_suite_test_pollution_4_tests`。

---

## 5. 关键发现 / 教训（已写进 memory，但接手要知道）

1. **PROTECTED_FILES 清单滞后于真实定制面**（新写的 memory `feedback_sync_nonprotected_files_with_noldus_customization_overwritten`）：sync「默认全量跟随非受保护文件」是对的，但「非受保护 = 可盲目覆盖」是**错的**。这次洗掉 2 个有 Noldus 定制的非清单文件。**sync 前应对所有触及文件跑 `diff <(git show <last_sync_commit>:..) <(git show dev:..)`，非空即 surgical，不只查静态清单**。
2. **backend/tests/ 在 sync subtree 外**（`backend/packages/harness/deerflow/`），sync 脚本不碰测试。上游源码重构删/改 API 时，配套测试要 head-to-head 取上游版本（spec① §8 已预警）。
3. **契约可能跨 harness/app 两层**（upload size 例）：`app/` 永不被 sync，harness 改了契约要手工同步 app 侧，否则 Pydantic ValidationError。
4. **.pth 陷阱**（§3）：这是上一会话最容易踩错、差点验错代码库的点。

相关 memory（接手必读）：
- `feedback_sync_nonprotected_files_with_noldus_customization_overwritten`（本次新写，核心教训）
- `feedback_conftest_mock_hides_circular_import_verify_bare_prod_import`（executor 红线 + 裸导入）
- `feedback_sync_full_follow_upstream_infra` / `feedback_sync_protected_files_registry_loss` / `feedback_head_to_head_before_claiming_no_merge`（sync 原则）
- `feedback_grill_handoff_must_be_verified`（别信文字、现场核实——上一会话正是靠它抓出 5 处回归）
- `feedback_known_full_suite_test_pollution_4_tests`（baseline red 别归因自己）

---

## 6. 风险与注意事项（容易混淆/不建议的方向）

- ❌ **别用裸 `PYTHONPATH=.` 验 sync-21**（会测主仓 dev，假绿）。必须 `PYTHONPATH="$WTHARNESS:."`（§3）。
- ❌ **别整文件接受上游受保护文件**（红线，CLAUDE.md「同步核心规则」）。9 受保护文件 + 现补的 2 个（memory/prompt.py、app_config.py）都要 surgical。
- ❌ **executor.py 的 `seal_handoff_tools` import 别挪回模块顶层**（循环导入，生产启动崩）。
- ❌ **别让 agent 自动 merge 到 main**——仓库约定先进 dev、用户发起 PR。
- ⚠️ **dev vs main 可能分叉**：409 合了 main 但 dev 还是 `1a819e7e`。接手第一件事核实 main/dev 关系，别假设 dev 是最新。
- ⚠️ sync-21 的 `.deerflow-sync-state` 已推进到 `f92a26d5`——若 PR 被拒/回退，记得 state 文件也要回退，否则下次 sync 基准错。

---

## 7. 下一位 Agent 的第一步建议

**如果用户说"继续 sync-21 / 帮我 dogfood / 准备合 dev"：**
1. 读本文档 §3（.pth 陷阱）+ §5 memory。
2. 核实 main/dev 关系：`cd /home/wangqiuyang/noldus-insight && git log --oneline origin/main -8 && git log --oneline dev -3`，确认 409 在 main、dev 是否落后。
3. 进 sync-21 worktree，按 §3 强制 worktree deerflow 重跑全量（应 3850 passed / 3 baseline red）+ 两入口裸导入。
4. （可选 dogfood）`cd /home/wangqiuyang/noldus-insight-sync-21/packages/agent && make dev` 实起，发 EPM 分析跑通。
5. 把结果给用户，**让用户建 PR**，不自动 merge。

**如果用户说"两份 spec 还要改 / 有新需求"：**
- spec 在 `docs/superpowers/specs/2026-06-09-deerflow-sync-21-commits-spec.md` 和 `2026-06-09-gateway-multiworker-stream-409-fix-spec.md`。直接读+改。

**如果用户提全新任务**：忽略 sync-21 收尾，按新任务走（但提醒用户 sync-21 PR 还没建）。
