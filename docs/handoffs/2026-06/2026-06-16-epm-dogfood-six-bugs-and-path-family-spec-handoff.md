# Handoff: EPM dogfood 连环诊断 — 6 个 bug 全修复合入 + 路径族根治 spec 待实施

> 日期：2026-06-16
> 上一会话产出：跨多轮 EPM dogfood，诊断并修复 6 个串联 bug（全部已合 origin/dev），review 实施分支 3 个，最后把"虚拟路径解析依赖 env"故障族升级为一份根治 spec（待实施）。
> 交接对象：下一个接手的 AI Agent
> 关联起点：`docs/handoffs/2026-06/2026-06-15-four-specs-reviewed-merged-handoff.md`（S1–S4 四 spec 合入）

---

## 1. 当前状态速览

**`origin/dev` HEAD = `a6a49ef7`**，本地 dev 与之一致（已 push，无待推）。线性历史，最近的栈：

```
a6a49ef7 handoff:行为学范式书 PDF → skill 重构（用户的本地提交，非本诊断线）
5b61fafd 重构 v0.1 六范式 by-experiment skill（用户的本地提交，非本诊断线）
868ed712 PR#134 feat/stats-gate-and-analyst-seal  ← #6a+#6b+review-fix
3679ff4d   review-fix(#6a) self-derive workspace_dir 兜底
8952646a   #6b data-analyst statistics 空走描述性 partial 立即 seal
1e3505f7   #6a resolve 自派生组计数 + prep 显式派生
fd73b558 PR#133 feat/identify-zone-detection-nonstandard-columns  ← identify 疑似归属列检测
afe70e24   identify 疑似归属列检测
（更早）PR#132 #5 / PR#131 四 bug(#1-#4)
```

**本批 dogfood 修复已全部闭环合入 dev，无遗留待实施的修复。** 唯一待实施的是一份**根治性 spec**（见 §4），其针对的具体缺陷已点修合入（不阻塞，是防未来复发）。

---

## 2. 这条 dogfood 诊断线修了哪 6 个 bug（全在 origin/dev）

EPM 真实数据 `/home/wangqiuyang/DemoData/real_data/Raw data-EPM-Xuhui-28`（28 文件，列含 `open`/`closed`/`result_1`，分组 control7/treatment21）连环跑出 6 个串联 bug，逐个诊断+修复+review+合入：

| # | bug | 修复 | PR |
|---|---|---|---|
| **#1** | lead 把列名非标准(open/closed)误判 NoZones、建议切模板 | lead prompt + column-confirmation skill 加"模板看划区与否不看列名"铁律 | #131 |
| **#2** | run_metric_plan 进程内跑脚本全 FileNotFoundError(`/mnt` 路径不 resolve) | parse_trajectory/save_output_json 在 I/O sink resolve | #131 |
| **#3** | handoff 校验器无视 status，把诚实 failed 判 incomplete → lead 无限重派 | content check 加 `if status==failed: return None` | #131 |
| **#4** | lead 无诚实失败熔断(附带) | 规则#7 加"诚实失败 status=failed 不机械重派" | #131 |
| **#5** | run_metric_plan Step8 validation 父进程无 env → 全指标误报 result_file_unreadable → 毒化 data-analyst | Step8 aggregate 包 `_scoped_path_env` | #132 |
| **#6a** | prep 不传 n_per_group/n_groups → stats gate 用 None 判 skip → statistics={} → data-analyst 手算螺旋忘 seal | resolve 自派生组计数 + prep 显式派生 + review-fix(workspace_dir 兜底) | #134 |
| **#6b** | data-analyst statistics 空时手算螺旋(纵深防御) | prompt fast-fail 1a：statistics 空走描述性 partial + 立即 seal + 禁手算 | #134 |

**因果链是串联的**：#2 修好后指标能算了才暴露 #5；#5 修好后统计能聚合了才暴露 #6a；#6a 修好后 data-analyst 才走到正常路径。每修一个，dogfood 往前走一步暴露下一个。这是"逐层剥洋葱"的典型——下一轮 dogfood 很可能再往前暴露 report-writer 层的新问题（本批从未跑通到 report-writer，见 §5）。

**详细诊断都在 memory**（feedback_* 条目，按 # 索引），每个都带 gateway.log 行号 + 实测证据。

---

## 3. 本会话 review 的 3 个实施分支（结论：全通过，已合）

用户让其他 agent 实施、本会话 review：
- **feat/dogfood-0615-fixes**（#1-#4）→ 通过，红→绿四块独立坐实，实施者还补了 spec 漏的 txt 裸路径。已合 PR#131。
- **feat/identify-zone-detection-nonstandard-columns**（identify 疑似归属列）→ 通过，回退证 5 锚点红、7 守护绿、regex 精度好。已合 PR#133。**这是 #1 的代码层补强**：identify 工具原 `_ZONE_COLUMN_PATTERN` 漏检 open/closed → 产"数据无 zone 列"假陈述污染 lead；新增 `_SUSPECT_ZONE_COLUMN_PATTERNS` 检测疑似归属列剔 NoZones（守"不猜哪列是哪区"，只判"有疑似列→非 NoZones"）。
- **feat/stats-gate-and-analyst-seal**（#6a+#6b）→ 实施忠实，但 **review 时发现一个真缺陷并当场修复**：resolve 自派生(layer①)对主调用方 prep 形同虚设——prep 传 `/mnt` 虚拟 groups_file 且不设 env，`resolve_sandbox_path` fail-safe 原样返回读不到 → (None,None)。原 layer-A 测试喂真实路径假性证明覆盖。review-fix=`_derive_group_counts` 加 workspace_dir 兜底(commit `3679ff4d`)。已合 PR#134。

**review 方法学**（值得下一个 reviewer 复用）：每个分支都**回退生产改动证 red→green**（不信空绿，守 PR#115 hollow-baseline 教训）；lint/基线债都用 **origin/dev 同 venv 对照**坐实非本次引入；测试是否**喂主调用方真实入参形态**是关键审查点（#6a 就栽在"喂真实路径而非 prep 的虚拟路径"）。

---

## 4. 唯一待实施项：路径族根治 spec

**spec：`docs/superpowers/specs/2026-06-16-virtual-path-resolution-env-family-spec.md`**（⚠️ untracked，见 §6）

**为什么写它**：#6a 的具体缺陷已点修合入，但用户判断"这缺陷会致后续 agent handoff 出问题"——正确。根因是一个**故障族**，不是孤例：

EthoInsight 有两套"虚拟路径 `/mnt`→真实路径"机制：
- **机制 A** `replace_virtual_path(path, thread_data)`（harness 侧，无 env 依赖，可靠）
- **机制 B** `resolve_sandbox_path(path)`（ethoinsight 侧，**依赖 `DEERFLOW_PATH_*` env**）

机制 B 的前提是"脚本跑在沙箱/进程池子进程内 env 已设"。**故障族 = 机制 B 代码被"进程内直接调用"而调用方没设 env → 静默原样返回 `/mnt` → 读不到 → 下游哑退化**。已发生 #5、#6a 两个实例（都点修了）。复发原因：机制 B 函数签名看不出依赖 env，每个新"harness 进程内调 ethoinsight 读 /mnt"调用点都是潜伏雷，喂真实路径的测试测不出。

**spec 根治策略**：① `resolve_sandbox_path` 加 `workspace_base` 参数（无 env 时对 workspace 前缀兜底）② 把 #6a 的 workspace_dir 兜底收口到机制 B 去重 ③ 无 env 静默退化加 debug 信号。**划清边界**：不动机制 A、不强行合 A/B（违分层）、不删 #5/#6a 点修（双保险）、env 优先路径字节不变。spec 自含三层 red 锚点 + 回退验证。

**基线 origin/dev `a6a49ef7`**（spec 写时是 868ed712，现多了用户两个无关提交，不影响——路径族代码未被那两个提交触碰）。

---

## 5. 下一步建议

**若用户继续 dogfood**：本批 6 bug 全闭环，identify→prep→code-executor→data-analyst 这段已修通。**从未跑通到 report-writer**——下一轮 dogfood 很可能在 report-writer 层（或 data-analyst 的描述性 partial 路径）暴露新问题。复跑 EPM 验证：① statistics 非空（#6a）② data-analyst 正常 seal（不再 terminated without emitting）③ 走到 report-writer。

**若用户要推进路径族根治**：把 §4 的 spec 交给 agent 实施（独立 worktree `feat/virtual-path-resolution-family` 基于 origin/dev）。与任何 feature 正交，可独立。

**关于本批 spec 文件未入库**（§6）：值得顺手把诊断/根治 spec 提交进 git（它们是知识资产，现在只在工作区）。

---

## 6. 环境/善后注意事项（本批踩过/留下的）

1. **本批所有 dogfood spec 文件都是 untracked（未进 git）**：`docs/superpowers/specs/` 下 `2026-06-15-epm-dogfood-four-bug-fixes`、`2026-06-15-run-metric-plan-step8-validation-env`、`2026-06-16-stats-gate-none-counts-and-data-analyst-seal`、`2026-06-16-virtual-path-resolution-env-family` 四份 spec 都没 commit。实施分支引用了它们但没把 spec 本身入库。**建议 commit 进 dev**（知识资产，别只留工作区）。
2. **4 个已合并 PR 的 worktree 遗留未清**：`feat/dogfood-0615-fixes`(2056234b)、`feat/identify-zone-detection-nonstandard-columns`(afe70e24)、`feat/run-metric-plan-step8-env`(3818c498)、`feat/stats-gate-and-analyst-seal`(3679ff4d)——**分支都已合进 origin/dev，worktree 可安全 `git worktree remove`**（确认无未提交改动后）。位置：`~/.claude/worktrees/` 和 `~/noldus-insight-*`。
3. **worktree 借主仓 venv 的 editable 陷阱（本批反复踩，已沉淀 memory）**：worktree 无独立 venv，借主仓 backend venv 时 editable deerflow/ethoinsight **指向主仓源码**——测试 `from deerflow/ethoinsight import` 测的是主仓代码、worktree 改动假绿。**测试必须 `importlib.spec_from_file_location(__file__ 相对路径)` 加载 worktree 源**。生产 import 链同理：prep 工具 `from ethoinsight.catalog.resolve import` 走主仓 → backend 测试结构上测不到 worktree 的 ethoinsight 改动，必须在 ethoinsight 层 importlib 验证。见 memory `feedback_worktree_shares_main_venv_editable_link_tests_must_use_importlib`。
4. **基线债**：backend 全量 ~6 failed（chart_maker_config / local_sandbox×3 + 2 测试隔离污染 async_delegates_to_sync），origin/dev 同 venv detached worktree 对照坐实=非本批引入。别归因自己。见 memory `feedback_known_full_suite_test_pollution_4_tests`。
5. **lint 误信号**：standalone ruff（查 `/tmp/*.py` 无项目 config）会报 I001/UP017 等 isort 假阳性；**必须在项目内（`cd packages/xxx && ruff check`）**才是权威。本批多次差点误判"引入 lint 债"，in-project 一查都是 dev 基线既有。

---

## 7. 关键 memory（下一个 agent 先读）
- `feedback_lead_inverts_fewzones_vs_nozones_by_column_name`（#1，含 identify 补强）
- `feedback_run_metric_plan_inprocess_scripts_dont_resolve_sandbox_path`（#2）
- `feedback_handoff_content_validator_rejects_failed_status_infinite_redispatch`（#3）
- `feedback_run_metric_plan_step8_validation_not_scoped_path_env`（#5）
- `feedback_prep_metric_plan_stats_skip_none_counts_poisons_data_analyst_seal`（#6a+#6b+review-fix+**路径族根治 spec 指针**）
- `feedback_subagent_seal_deadlock_is_prompt_not_budget`（#6b 同款 seal 死锁的历史根因）
- `feedback_worktree_shares_main_venv_editable_link_tests_must_use_importlib`（测试加载铁律）
- `feedback_known_full_suite_test_pollution_4_tests`（基线债对照）
- `feedback_isolate_root_cause_before_stacking_fallback_mechanisms`（根因未隔离前别叠兜底——本批反复用于"先点修治本再考虑纵深防御"）
