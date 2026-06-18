# Handoff: 2026-06-18 — 第二轮 EPM dogfood 复跑：4 根因诊断 + 写 spec + review 4 实施分支全合 dev

> 交接对象：下一个接手的 AI Agent。
> 会话主线：用户本地复跑 EPM dogfood（thread `3bcbee10`，28-subject EPM/PlusMaze-FewZones），把跑出来的**前端全量输出 + 本地 log** 交给本 agent 诊断。本 agent **诊断出 4 个结构性根因 → 各写一份 spec → 用户派 4 个 agent 分别实施 → 本 agent 监控 origin 新分支并逐个 review（有问题在 worktree 改+push）**。
> 当前位置：主仓 `/home/wangqiuyang/noldus-insight`，分支 `dev`，**origin/dev HEAD = `6a4c382d`（已同步）**。
> **核心结论：4 个 dogfood 根因 spec 全部实施完毕且全合入 dev，0 个 open PR。该 track 闭环。**

---

## 1. 当前状态速览 — 这批活已干完

### 4 个根因 spec 全部已合 dev（origin/dev `6a4c382d`）

| Bug | spec 文件（`docs/superpowers/specs/`） | PR | review 结论 | 状态 |
|---|---|---|---|---|
| **Bug1** chart-maker「open arm 不存在」 | `2026-06-18-plot-scripts-zone-param-alignment-spec.md` | #157 | ✅ 最干净，无需改 | ✅ 合 |
| **Bug2** 生图不能 batch（串行慢）| `2026-06-18-chart-maker-parallel-plotting-spec.md` | #154 | ⚠️ 与 #155 冲突 → 本 agent rebase 解冲突 | ✅ 合 |
| **Bug3** report 图片 dev↔deploy 路径横跳 | `2026-06-18-report-image-path-ssot-spec.md` | #156 | ⚠️ 基于过时 dev → 本 agent rebase + 补 review 注释 | ✅ 合 |
| **Bug4** data-analyst 撞 50K read_file 截断 | `2026-06-18-code-executor-handoff-slimming-spec.md` | #155 | ✅ 全面，无需改 | ✅ 合 |

**4 个 spec 文档已入库 dev**（本会话末 commit `6a4c382d`，**只提 3 个 spec，不带 uv.lock**；handoff-slimming spec 随 PR #155 已先入库）。**open PR：0。**

### 4 个根因（逐字节实证，全部来自真实 dogfood thread `3bcbee10`）

1. **Bug1 — 列对齐没传到 plot 脚本**：compute 脚本 `compute_open_arm_time_ratio(df, **parameters)` 传了 HITL 对齐的 `open_arm_zones=['open']`，但 per-subject **plot** 脚本 `compute_*(df)` **裸调**→ 走列名模式 fallback `_get_open_zone_cols`（找 `in_zone.*open_arm`）→ 用户列名是 `open` 匹配不上 → 返回 None → 报「no open-arm zone columns」。**数据里开臂存在，只是列名经对齐叫 `open`**。P3（charts-column-alignment-self-read）只修了 prep_chart_plan 自读，没覆盖 plot 脚本重算。
2. **Bug2 — 半 SKILL 缺失半契约断裂**：guardrail `script_invocation_only_provider` **本就放行** code-executor 同款 `bash -c "...& ...& wait"` 并行，但 chart-maker SKILL 没教它（串行 N turn）；且 chart-maker/report-writer SKILL 的 `bash cat>bundle` 指引撞 guardrail（`>` 是禁字符）+ report-writer 根本没 bash 工具。
3. **Bug3 — 图片路径 SSOT 漂移**：「artifact 路径规范形态」分散在**前端 2 处 + 后端 3 处共 5 个改写/判断点**，各自漂移。后端 seal 产 `mnt/...`（无前导斜杠），前端判 `startsWith("/mnt/")`（要斜杠）→ 不命中 → 落 `normalizeArtifactImageSrc` 5-case 兜底 → case 4 砍掉 `mnt/user-data` 前缀 → 后端 `resolve_virtual_path` 拒绝 → 404。**这就是「本地改好部署崩、部署改好本地崩」反复横跳的根**。
4. **Bug4 — handoff 体积越 50K 截断线**：85K 的 `handoff_code_executor.json` 里 44K 是死重量——`task_context`(22K，无消费方，代码注释自承)+ `output_files`(8.9K，聚合器靠 glob 不读它)+ `outlier_diagnostics`(28K，有用但放错层)。data-analyst fast-fail 必读的 `gate_signals`/`data_quality_warnings` 排在尾部，被 sandbox read_file 的 50K 硬截断斩掉 → data-analyst 反复盲读行号、烧光 turn、最后「蒙」一个 `quality_warnings_critical_count:0` 过去。

---

## 2. 本会话 review 修的真问题（便于回归溯源）

逐个记录（review 4 个实施分支，2 个有真问题）：

1. **Bug3 #156（图片路径 SSOT）发现两问题，本 agent 修**：
   - **分支基于过时 dev**（merge-base `0696ef79`，缺整个 9-spec 批次 P2/P3/P4/P5，直接合会 **revert** 那批，含 P5 改的同文件 `seal_handoff_tools.py`）→ 在 worktree `rebase origin/dev`（**0 冲突**，因 dev 改的是 `seal_chart_maker_handoff` L523-549、本分支改 `_resolve_report_image_placeholders` L137-265，不同函数）。
   - **spec §六.3 风险落地**：`normalizeArtifactImageSrc` 有**第二调用方** `artifact-file-detail.tsx:411`（markdown 文件预览），实施 agent 只改了 `markdown-content.tsx`、没考虑它。核实：对 report.md canonical 路径两调用方都正常；对 legacy 非规范 markdown 行为收紧为返回 null（按 spec §2.3「暴露不兜底」语义本就一致），**加注释明示这是有意行为**（commit `5428e946`）。force-push（用户授权）。
2. **Bug2 #154（并行绘图）合并冲突，本 agent rebase 解**：
   - **根因**：#155（Bug4）先合 dev，它和 #154 都删 `chart_maker.py`/`report_writer.py` 同一段 bash bundle 死指引（我在两份 spec §3.2 都标了这个协调点）→ #154 rebase 时 step4/step2 冲突。
   - **解法**：保留 dev(Bug4) 的 bundle 删除文本 + **完整保留 Bug2 独有的并行绘图新增**（step 8 `bash -c "...& wait"`）。核实：bundle 0 处、并行绘图 3 处。force-push → #154 MERGEABLE → 合。
3. **Bug1 #157 / Bug4 #155：无实质问题**。Bug1 最干净——实施 agent 还**扫了全部范式的 per-subject plot 脚本**（epm/ldb/oft/zero_maze 共 13 个），并发现我 spec underspecify 的点：不同 compute 函数签名不同，`**parameters` 全量透传会 TypeError → 加了 `select_zone_kwargs(parameters, keys)` 按函数挑 key（仿 dispatcher `_zone_kwargs`）。Bug4 含 502 行新测试，红→绿用真实 28-subject×5×65outlier payload 坐实 <50K。

### review 时跑的验证（每个分支都过）

- **基线漂移检查**：`git merge-base --is-ancestor origin/dev origin/<branch>`（Bug3 唯一漂移，已 rebase）。
- **测试用 worktree 源**：harness 用 `PYTHONPATH=<worktree>/packages/harness:.`、ethoinsight 用 `PYTHONPATH=<worktree>/packages/ethoinsight`，config 用 `DEER_FLOW_CONFIG_PATH=主仓/packages/agent/config.yaml`（worktree 无 config.yaml）。
- **裸导入两入口**（Bug2/Bug4 改 subagents/tools/builtins 核心）：`import app.gateway` + `from deerflow.agents import make_lead_agent` 均 0 退出。
- **全量回归**：ethoinsight 全量 `916 passed, 70 skipped`（Bug1 改共享 `_cli.py`/`resolve.py`）。
- **已知 pre-existing 污染**：`test_chart_maker_config_basic_fields`（`cfg.model=="inherit"` 断言 vs dev 现状 `deepseek-v4-pro-summary`）在干净 dev 上就红，与本批无关——别归因自己改坏。

---

## 3. 未完成事项 / 注意

### 🟢 这批 track 已无代码待办，但有验证 + 卫生项

1. **🔴 重要：Bug3 图片路径必须 dev + deploy 两环境都验**。这个 bug 的本质就是「只验一个环境就以为修好」。PR #156 body 已强调：合并后**先本地 `make dev` 验 report 图能显示，再 `make deploy-tar` 到 ECS 验同一流程**——任一环境图片 404 都算没修完。**本会话只做了单测，没做端到端两环境验证**——这是最该接着做的。
2. **dogfood 第三轮复跑验证**：4 spec 全合后，用真实数据端到端复跑 EPM（列名 `open`/`closed` 那份），确认：① chart-maker `plot_open_arm_time_ratio_bar` 所有 subject 出图（Bug1）② 图并行跑（Bug2）③ report 图能显示（Bug3）④ data-analyst 单次 read 读全 handoff、不再盲读尾部（Bug4）。
3. **本会话 worktree 未清理**（5 个，磁盘占用）：`chart-maker-parallel-plotting`(137f0c26) / `code-executor-handoff-slimming`(1b5c2090) / `plot-zone-param-alignment`(67dcf9d8) / `report-image-path-ssot`(5428e946) — 都已合 dev，**可安全 `git worktree remove`**。另有 `noldus-insight-worktree`(fix/seal-list-zone-params) 是更早的、与本批无关。

### 🟡 卫生 / 单独问用户

4. **`uv.lock` 的 ` M` 始终未提交**（`packages/agent/backend/uv.lock`）：**从本会话开始前就存在、来历不明**（上一 handoff §2.2 也提醒过）。本会话 commit 3 个 spec 时**显式排除了它**。下一 agent 别顺手提交——单独问用户是 restore 还是保留。
5. **`reports/report for june/`** untracked 目录：与本批无关，没动。

---

## 4. 关键文件指针

- **4 个根因 spec**：`docs/superpowers/specs/2026-06-18-{plot-scripts-zone-param-alignment,chart-maker-parallel-plotting,report-image-path-ssot,code-executor-handoff-slimming}-spec.md`（全已实施全合）
- **本会话起点 handoff（上一会话的 9-spec 批次）**：`docs/handoffs/2026-06/2026-06-18-9spec-dogfood-rootcause-batch-complete-handoff.md`
- **各 Bug 改动的关键文件**：
  - Bug1：`packages/ethoinsight/ethoinsight/scripts/_cli.py`（`make_plot_parser` 加 `--parameters-json` + `select_zone_kwargs`）、`catalog/resolve.py`（`resolve_charts` 复用 `_build_zone_aliases_overrides` 注入）、13 个 `scripts/*/plot_*.py`
  - Bug2：`subagents/builtins/chart_maker.py`（并行绘图 step8 + 删 bundle）、`report_writer.py`（删 bundle）；guardrail `guardrails/script_invocation_only_provider.py`（`_PARALLEL_BASH_PATTERN` 本就支持，未改）
  - Bug3：`tools/builtins/seal_handoff_tools.py`（`_to_canonical_artifact_path` SSOT 函数）、前端 `core/artifacts/utils.ts`（删 case 2-5）、`messages/markdown-content.tsx`（单分支）、`artifacts/artifact-file-detail.tsx`（review 注释）
  - Bug4：`subagents/handoff_schemas.py`（删 task_context + 加 `outlier_diagnostics_ref/count`、`output_files_ref/count`）、`tools/builtins/run_metric_plan_tool.py`（`_slim_payload_into_sidecars` 拆旁路）、`seal_handoff_tools.py`（task_context 注入条件化）、`subagents/builtins/data_analyst.py`（旁路读 outlier + 删 bundle）

---

## 5. 下一位 Agent 的第一步建议

```bash
cd /home/wangqiuyang/noldus-insight
git fetch origin dev && git rev-parse --short origin/dev   # 应 = 6a4c382d 或更新
gh pr list --state open                                     # 预期：本批无 open PR

# 清理本会话已合的 worktree（可选，省磁盘）
for w in chart-maker-parallel-plotting code-executor-handoff-slimming plot-zone-param-alignment report-image-path-ssot; do
  git worktree remove ".claude/worktrees/$w" 2>/dev/null || echo "skip $w"
done
```

**之后按用户意图**：① **dogfood 第三轮端到端复跑验证 4 spec**（推荐，§3.2，尤其 Bug3 的 dev+deploy 两环境 §3.1）② 若图片仍 404，先回 spec 看是不是某层没覆盖，别写 prompt 补丁 ③ memory 沉淀（见下）。

---

## milestone 建议

本会话让「harness 鲁棒性 / dogfood 根因治理」track 再到一个 checkpoint：上一会话 9-spec 批次之后，**真实第二轮 dogfood 复跑暴露第 5-8 个结构根因，4 spec 逐个根治全合 dev**。建议更新该 milestone（track 名同 9-spec 批次）：记录 4 根因 + 4 PR（#154/#155/#156/#157）+ 本会话 review 修的真问题（Bug3 过时 dev rebase + 第二调用方 / Bug2 与 Bug4 bundle 删除冲突 rebase）。

### 值得沉淀的 memory（可复用教训）

1. **「列对齐对 compute 生效但对 plot/重算路径不生效」是一类反复缺口**——凡是「另一条路重算指标」的脚本（plot 脚本、未来任何 recompute）都要确认 zone 参数有没有透传进去。判别法：`grep compute_.*\(df\)` 找裸调。
2. **多 compute 函数签名不同时 `**parameters` 全量透传会 TypeError**——按函数声明的 key 投影（`select_zone_kwargs`），别整份 dict 灌。
3. **「同一约定多处定义」是 dev↔deploy 横跳的通用根**（Bug3 = 5 个改写点）——修法是定唯一规范形态 + 前后端各只解析一次 + 删全部兜底猜测，而不是「再加一层 normalize 迁就」。且**必须两环境都验**，否则只是把横跳延后。
4. **多 agent 并行实施同批 spec 时，改同一文件的会冲突**——先合的赢，后合的 rebase。本批 Bug2/Bug4 都删同段 bundle（spec §3.2 已预判）。review 时若发现分支基于过时 dev（`merge-base != origin/dev`），先 rebase 再 review/合，否则会 revert 已合的批次。
