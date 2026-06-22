# Handoff: 2026-06-18 — 第三轮 EPM dogfood：data-analyst 原地打转诊断 → 2 spec → 2 PR review 闭环

> 交接对象：下一个接手的 AI Agent。
> 会话主线：用户本地复跑 EPM dogfood（thread `2e40514c`，27-subject EPM/PlusMaze-FewZones），data-analyst **原地打转**。本 agent 诊断出**第 9 个结构根因**（thinking 过载撞 turn 内超时）+ 顺带深挖「参数审计」feature 与行为学书增量 → **写 2 份 spec** → 用户派 2 个 agent 实施 → 本 agent 监控 origin、逐个 review（其中一个在 worktree 改+push）。
> 当前位置：主仓 `/home/wangqiuyang/noldus-insight`，分支 `dev`，**origin/dev HEAD = `08f3b2da`（PR#160 已合）**。

---

## 1. 当前状态速览

| spec | PR | review 结论 | 状态 |
|---|---|---|---|
| **book-paradigm 判读判据补 skill**（`2026-06-18-book-paradigm-judgment-criteria-to-skill-spec.md`）| #160 | ✅ 无需改，18 条增量全到、铁律全过 | ✅ **已合 dev** |
| **data-analyst thinking 过载**（`2026-06-18-data-analyst-thinking-overload-spec.md`）| #161 | ⚠️ 发现 1 处降级瑕疵 → 本 agent worktree 改+push（`8e41157d`）| 🟡 **OPEN，待用户合**（CI `backend-blocking-io` pending） |

### 🔴 必须接着做的事（按优先级）

1. **PR#161 等 CI 绿后合并**：`mergeStateStatus=UNSTABLE` 仅因 `backend-blocking-io` check **pending**（非 fail），MERGEABLE。盯 CI 转绿即可合。
2. **两份 spec 文件仍 untracked，未进 git**（`git status` 显示 `??`）：
   - `docs/superpowers/specs/2026-06-18-data-analyst-thinking-overload-spec.md`
   - `docs/superpowers/specs/2026-06-18-book-paradigm-judgment-criteria-to-skill-spec.md`
   - 实施分支按**路径**引用了它们，但 spec 本身没入库。**本 agent 没擅自 commit**（守用户控制 commit 的约定）。建议：单独 commit 这 2 个 spec 到 dev（同上一批 9-spec/4-spec 的做法）。**别顺手带上 `uv.lock`**（历史遗留 ` M`，来历不明，见往期 handoff）。
3. **dogfood 第四轮复跑验证**：PR#161 合后，用同一份 27-subject EPM 数据端到端复跑，确认 data-analyst **单 turn 完成判读、调 seal、不超时、不打转**，且 outlier_findings 只含精选 2-3 条、subject 为真实 Trial 号。

---

## 2. 根因诊断（逐字节实证，dogfood thread `2e40514c`）

### 2.1 真根因 = 第 9 个结构根因：thinking 当产出撞 turn 内超时

- **现象**：code-executor 成功（135/135）→ data-analyst **Task timed out after 900 seconds** → lead 重派 → 又超时 → 原地打转。`handoff_data_analyst.json` 从未产出。
- **排除 Bug4**：`handoff_code_executor.json` 实测 **18260B**（<50K），Bug4 瘦身生效；fast-fail 必读字段全在主文件。**数据 100% 可分析，链路无毛病。**
- **真因**：data-analyst 在**单个 model turn 的 thinking 里**完整且反复地重算+逐条搬运 63 条 outlier（subject #i→Trial 映射、重算 counterfactual、草拟 seal JSON），thinking 撑爆 50K（用户贴出的内部独白末尾 `[Message truncated - exceeded 50,000 character limit]`），生成耗时撞 900s wall-clock，**turn 永远没结束** → `after_model` 的 SealGate **永不触发** → seal 漏调。
- **关键边界**：这是 **turn 内超时**，与之前 4 次 seal 漏调（收尾时漏 call，SealGate 能拦）**不同类**。**SealGate（after_model 门）结构上救不了 turn 内超时**——turn 没结束 after_model 就不执行。诱因是 prompt 自相矛盾（处处写「只读不算」却命令「遍历映射 63 条」「走 a–f 决策树」）。

### 2.2 顺带搞清的两件事（决定 spec 走向）

- **参数审计（step 2.8）的判据事实上造不出来**：行为学同事 680 页书 + 6 个 `reward-criteria/*.yaml` **无一条「参数值 vs 数据分布」数值判据**（design/metric/interpretation/confound/forbidden 五类，无数值参数判据）。因为「velocity_threshold 该设多少」依赖装置/帧率/品系，是**现场标定的工程问题**，非普适领域判据。消费端仅 `present_assumptions.py`（Sprint 7 可选卡片）。→ **从 data-analyst 移除**（恒空数组、保留 schema 向前兼容）。
- **行为学书对 6 范式判读有真增量**：6 个并行 agent 做「书原文 vs 已合 skill」差距分析，经四重筛选（判读相关/skill 未覆盖/守铁律/EV19 可算）得增量：假阳性排查流程、品系基线量化预期、运动抑制三步分离、爬尾混杂、季节混杂、潜伏期二重性等。判据已在 `by-experiment/*.md`（repo 内），缺的是 prompt 强制查表。→ **补进 6 个 by-experiment skill**（PR#160）。

---

## 3. 本会话 review 修的真问题（便于回归溯源）

### PR#161 outlier 真名下沉块发现 1 处降级瑕疵，本 agent 修（`8e41157d`）

- **问题**：spec §6.3 降级路径原仅处理 `subject #i` index 越界，**未处理 EV19 对象名常为空串**——label_map 漏翻译时原样保留空串 → `compute_outlier_diagnostics` 的 counterfactual 出现 `if  excluded` 空洞（实测复现：`subject=''  cf='... if  excluded'`）。
- **修法**（最小、低风险，非 sub-agent 建议的重型 `is_fallback` 参数）：`statistics.py` 的 subject 兜底改为「空串/纯空白也回退 `subject #i`」（1 行 + 注释），保证 subject 永远可读、counterfactual 不空洞。补回归测试 `test_subject_identifier_empty_string_falls_back_to_index`。
- **验证**：修复前后实测对比坐实（`subject=''` → `subject #2`，空洞消失）；ethoinsight outlier 25 passed。

### PR#160 无实质问题

实施质量高：18 条增量全到、组间口径改写到位、量化预期均标注「非判读阈值」、TST「全程评分」与实现一致（核实 catalog/scripts 无时间窗口截断，原「记录后 4 分钟」反而与实现不符）。

### review 时跑的验证（PR#161，每项过）

- **裸导入两生产入口**（改 subagents/tools/builtins 核心，铁律）：`import app.gateway` + `make_lead_agent` 均 0 退出。
- **worktree 测试用主仓 venv + worktree PYTHONPATH 覆盖**（worktree 无独立 venv，借主仓，须 `PYTHONPATH=<wt>/packages/harness:<wt>/packages/ethoinsight` 否则假绿）。
- **shared-logic 全量回归**（改了共享 `statistics.py`/`dispatcher.py`/`_cli.py`）：ethoinsight 全量 **934 passed, 65 skipped**。
- **基线 revert 检查**：两分支都基于 `4297b155`（落后 dev），但与 merge-base 后 dev 改动**零文件交集**（#160 改 md+workflows、#161 改 py+测试，互不重叠）→ **无 revert 风险、无需 rebase**。

---

## 4. 关键文件指针

- **2 份 spec**（**untracked，待入库**）：`docs/superpowers/specs/2026-06-18-{data-analyst-thinking-overload,book-paradigm-judgment-criteria-to-skill}-spec.md`
- **PR#161 改动**：
  - `subagents/builtins/data_analyst.py`（删 step 2.8 整段 171 行、step 2.7b 改「不搬运/不重算/不重映射」、step 2.6 强化必 read by-experiment）
  - `subagents/handoff_schemas.py`（parameter_audit_findings 字段保留默认空、docstring 更新）
  - `tools/builtins/seal_handoff_tools.py`（docstring 更新，7 行）
  - `ethoinsight/statistics.py`（outlier 真名下沉 + 定性 deviation 合成 + **本 agent 修的空串兜底**）、`metrics/dispatcher.py`（group_summary 加 subjects 字段）、`scripts/_cli.py`（`build_subject_label_map`）、6 个 `scripts/*/run_groupwise_stats.py`
  - 新测试：`test_data_analyst_thinking_diet.py`、`test_outlier_sidecar_subject_resolution.py`、`test_dispatcher_group_summary_subjects.py`、`test_outlier_diagnostics.py`
- **PR#160 改动**：6 个 `skills/custom/ethovision-paradigm-knowledge/references/by-experiment/{epm,zero_maze,open_field,forced_swim,tail_suspension,light_dark_box}.md`
- **行为学书**：`~/behavioral-book/by-paradigm/焦虑抑郁行为/`（书原文）+ `~/behavioral-book/reward-criteria/*.yaml`（给 RL 的判据，**非**参数审计判据）

---

## 5. 已沉淀的 memory（可复用教训）

- `feedback_seal_fourth_root_cause_thinking_overload_turn_timeout` — seal 漏调第 4 类：thinking 当产出撞 turn 内超时，after_model 门救不了；判别铁律「收尾漏 call / turn 内超时」二分。
- `feedback_param_audit_value_vs_distribution_criterion_uncreatable` — 参数审计「参数值 vs 分布」数值判据行为学上造不出来，应等确定性标定代码非领域判据。

---

## 6. milestone 建议

「harness 鲁棒性 / dogfood 根因治理」track 再到 checkpoint：第三轮复跑暴露第 9 个结构根因（thinking 过载撞超时），2 spec 根治，PR#160 已合、PR#161 待合。另「行为学领域知识注入」track 推进一步：680 页书经四重筛选提炼 6 范式判读增量补进 by-experiment skill（PR#160）。建议记录：① 第 9 根因 + after_model 门覆盖边界；② 书→skill 价值是「混杂排查可执行检查项」非教科书堆砌 + 必经铁律改写（绝对阈值→组间口径）+ EV19 可算性过滤；③ 两 spec 互补（一个砍机械工作量、一个充实判读判据）。
