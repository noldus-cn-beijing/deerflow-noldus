# Handoff: DeerFlow 上游同步 2d5f0787→d2cc991d（14 commit）+ 路径族根治 review

> 日期：2026-06-16
> 本会话两件事：① review 并修缮 `feat/virtual-path-resolution-family`（路径族根治，已 push）② DeerFlow 上游 sync 一批（已 push，**用户自行 PR merge**）。
> 交接对象：下一个接手的 AI Agent
> 关联：`docs/handoffs/2026-06/2026-06-16-virtual-path-resolution-family-handoff.md`（路径族 spec 实施）、`docs/handoffs/2026-06/2026-06-12-sync-2d5f0787-review-merged-handoff.md`（上次 sync 基准）

---

## 0. 当前状态速览

| 分支 | commit | 状态 |
|---|---|---|
| `feat/virtual-path-resolution-family` | `134a0343` | review 通过 + 修缮（修 dev 既有 `import sys` 债 + rebase dev）已 push，**待用户建 PR** |
| `chore/sync-deerflow-d2cc991d` | `66f37677` | sync 完成已 push，**待用户 PR merge** |

两分支正交、互不依赖（一个改 ethoinsight，一个改 deerflow harness），可分别合。

**`.deerflow-sync-state` 已推进**：`last_sync_commit: 2d5f0787` → `d2cc991d`（在 sync 分支的 commit 内，merge 后生效）。

---

## 1. 路径族根治 review（`feat/virtual-path-resolution-family`）

实施分支 review 结论：**忠实通过**。三项改动逐一核实 + 红→绿亲手回退坐实：
- 改动① `resolve_sandbox_path` 加 `workspace_base`（`_cli.py`）：env 优先路径字节不变（兜底放 env 循环全退出后），workspace_base 只对 workspace 前缀兜底，fail-safe passthrough 保留，debug 信号到位。
- 改动② `_derive_group_counts` 收口到机制 B（`resolve.py`）：行为等价（对 prep 真实虚拟路径入参实测 `(7,2)`）。
- 测试 9 新 + 13 #6a 等价全绿；ethoinsight 全量 844 passed/70 skipped。

**review 时顺手修的两个非阻塞项**（用户授权）：
1. `_cli.py` 顶部补 `import sys`——dev 既有 F821 潜伏 bug（两处 `sys.stderr` 冷路径会 NameError），非本分支引入；与已有 `import logging` 同处补齐。
2. `git rebase dev`——消解 diff-vs-dev 的 ~12,600 行"删除"假象（branch merge-base 早于 dev 那次 docs 批量入库 `73ead9c9`）。rebase 后 diff 干净=只 4 个真实文件。force-with-lease push。

详细见 memory `review_2026-06-16_virtual_path_resolution_family_passed_and_fixed`（本会话新增）。

---

## 2. DeerFlow sync（`chore/sync-deerflow-d2cc991d`，commit `66f37677`，38 文件）

### 2.1 基准 + delta

- 基准 `2d5f0787`（上次 sync，2026-06-11）→ 上游 HEAD `d2cc991d`。
- harness delta = **14 commit**，触及 34 文件 → 分类：**16 safe 全量跟随 + 13 new 全量跟随 + 5 protected surgical**。
- 策略=CLAUDE.md「默认全量跟随、受保护文件 surgical 守 Noldus 定制」。

### 2.2 这次 sync 主要修了什么（按影响排序）

**最值得拉的两条（我们重度路径上的隔离/并发 bug）**：
- **#3559 subagent 与父 run checkpointer 隔离**（`executor.py`，**1 行** `checkpointer=False`）：subagent 继承父 run checkpointer → 状态写回污染父 run 检查点。我们 4 个 ethoinsight subagent 全走这条路，迟早踩。
- **#3518 sandbox 并发状态 fail-closed**（`thread_state.py`，+`merge_sandbox` reducer）：同一 graph step 多 sandbox 工具各发 `sandbox_id` 更新、LangGraph 无 reducer 报错；修=相同 id 幂等、**不同 id 抛错**（不同 id=隔离 bug 宁可响亮）。

**其余 fix/perf**：
- #3549 空 SOUL.md 产废 agent → 加守卫（`setup_agent_tool.py`）
- #3494 provisioner 健康检查网络异常语义：return False → **raise**（区分"探测失败"vs"确认死"，修 stale sandbox 缓存复用，`remote_backend.py`）
- #3499/#3531 RunManager/RunEventStore O(n)→索引（多用户+长会话受益）
- #3505 skill 归档安装移出 event loop（同我们踩过的 read_bytes 阻塞 event loop 同类病）
- #3535 REST 历史接口剥 base64 图片

**新功能（非 bug，顺带）**：IM channel connections 子系统(#3487) + Brave/SearXNG/Browserless 搜索工具(#3528/#3451) + follow-up suggestions 可选化(#3591)。

### 2.3 5 个受保护文件 surgical（保留全部 Noldus 定制，已核实）

| 文件 | 上游 graft | Noldus 定制保住 |
|---|---|---|
| `agents/thread_state.py` | +`merge_sandbox` reducer + sandbox 字段 `Annotated` | `shared_path` ✅ |
| `subagents/executor.py` | +`checkpointer=False` | auto-seal / recursion_limit / max_turns / seal-resume 全套 ✅ |
| `tools/builtins/setup_agent_tool.py` | +空 SOUL.md 守卫 | — |
| `config/app_config.py` | +`suggestions` + `channel_connections` 字段 | `handoff_strict_mode` ✅ |
| `config/paths.py` | `make_safe_user_id` sha1→sha256 + 旧桶迁移 helper | `/mnt/shared`(`SHARED_PATH_PREFIX`) + `shared_dir` ✅ |

### 2.4 ⚠️ 三个超出"标准 surgical"的处理（下一个 sync 必读）

1. **IM-channels #3487 是 harness-only sync 下的"半个 feature"**：消费端在 `backend/app/` 层（channels manager / gateway routers / services / auth，**2191 行 23 文件，在 harness subtree 之外，sync 脚本不覆盖**）。按用户决定**全拉 harness 侧**——已核实 inert（新 persistence 模型/config 纯 pydantic/SQLAlchemy 定义、import 无副作用、`cryptography` 已装 46.0.5、不崩启动）。代价=带进一个 app 层没接的子系统表面（悬空表/config）。**将来真接 IM 渠道时，app 层 2191 行要单独拉**（sync 脚本不会帮你，要手动从 `backend/app/` 搬）。

2. **`todo_middleware.py` 被脚本误分类为 safe，但有 Noldus 定制 → full-follow 一度洗掉**：Noldus 的 `_is_awaiting_clarification` + after_model 的 section-2.5 守卫（修 dogfood thread `ca86744f` "重复要不要画图"）被上游版覆盖。**靠 out-of-subtree 测试 import 失败抓到**，已重新 surgical 回灌（`_is_awaiting_clarification` + `ToolMessage` import + section-2.5 守卫），只放弃上游 #3530 删的未用自由函数 `_completion_reminder_count`。**已审计全部 16 个 safe 文件，确认只有这一个有真 washout**（其余 `>` diff 都是 dev-behind-upstream，无 Noldus marker）。
   → **教训**：`PROTECTED_FILES` 清单滞后于真实定制面（`todo_middleware.py` 该补进清单）。见 memory `feedback_sync_nonprotected_files_with_noldus_customization_overwritten`。

3. **`cryptography>=43.0.0` 补进 harness pyproject**：#3487 的 `channel_connections/sql.py` import `cryptography.fernet`，上游同时在 pyproject 声明了它。**sync 脚本不覆盖 pyproject**（已知缺口），手动补齐。否则靠 transitive install 脆弱、clean `uv sync` 可能缺。

### 2.5 out-of-subtree 测试同步（脚本只管 harness，测试在 backend/tests/）

sync 脚本只覆盖 harness subtree；当 safe 文件行为变了，`backend/tests/` 里对应测试会变 stale。本次 2 个：
- `test_todo_middleware.py`：去 `_completion_reminder_count`（#3530 删的自由函数）+ 保 Noldus 的 `_completion_reminder_count_for_runtime` 测试。
- `test_remote_sandbox_backend.py`：跟随 #3494 的 raise 语义（原 `..._returns_false_on_request_exception` → `..._raises_on_request_exception` + 新增 server_error 测试）；本地与上游 pre-change 字节一致、无 Noldus mod，整文件 full-follow。

> **方法学**：判定 stale 测试用**全量跑**——回退/对照证「唯一新失败」是 `test_remote_sandbox`，随测试同步消解。不要凭猜。

---

## 3. 验证（两分支都做了）

**sync 分支**：
- backend 全量 **6 failed / 4071 passed** = origin/dev 同 venv 既有基线债（`chart_maker_config` + `async_delegates_to_sync`×2 + `local_sandbox`×3，测试隔离污染，memory `feedback_known_full_suite_test_pollution_4_tests` 在案）。
- **0 sync-introduced 失败**（clean dev 跑同套也是这 6 个；回退坐实唯一新失败已消解）。
- 裸导入 `app.gateway` + `make_lead_agent` 0 退出（**无 import 环**——改了 executor/middleware/agents 核心必做，CLAUDE.md 铁律）；import-cycle guard + harness boundary test 绿。
- ruff 仅 `executor.py:1185` 既有 E501（dev 基线，离改动区远，非本次）。

---

## 4. 善后 / 注意事项

1. **两分支待用户 PR**：`feat/virtual-path-resolution-family`(134a0343) + `chore/sync-deerflow-d2cc991d`(66f37677)。base 都选 dev。
2. **工作树里有 3 个无关的未提交改动**（**不是本会话产物，是用户之前的行为学 skill 在制品**，我没碰、没纳入任何 commit）：`skills/custom/ethovision-paradigm-knowledge/references/by-experiment/{epm,light_dark_box,open_field}.md` + untracked `docs/review-packages/2026-06-16-skill-catalog-metric-tristate.md`。**别误并进 sync PR**。
3. **`PROTECTED_FILES` 应补 `todo_middleware.py`**（本次 washout 实证）。下次 sync 前，建议把 `scripts/sync-deerflow.sh:51` 的清单更新（小改，独立 commit）。
4. **sync 脚本两个结构性缺口**（已知、本次手动补，未来值得脚本化）：① 不覆盖 pyproject（依赖声明漂移）② 不覆盖 `backend/tests/`（out-of-subtree 测试 stale）。

## 5. 关联 memory（下一个 agent 先读）
- `feedback_sync_nonprotected_files_with_noldus_customization_overwritten`（#2 washout 家族 + PROTECTED 清单滞后）
- `feedback_sync_full_follow_upstream_infra`（默认全量跟随策略）
- `feedback_sync_protected_files_registry_loss` / `feedback_head_to_head_before_claiming_no_merge`（受保护文件 surgical 纪律）
- `feedback_conftest_mock_hides_circular_import_verify_bare_prod_import`（改 executor/agents 必裸导入两入口）
- `feedback_known_full_suite_test_pollution_4_tests`（基线债对照，别归因自己）
- `feedback_ethoinsight_must_be_backend_workspace_dep`（pyproject 依赖漂移同类）

## milestone 建议
本次是基础设施维护（sync）+ 一个 spec review，未让某 feature track 到达新 checkpoint，**无需新建/更新 milestone**。仅在下次 sync 时参考 §2.4 三个处理 + §4.3 PROTECTED 清单补 todo_middleware。
