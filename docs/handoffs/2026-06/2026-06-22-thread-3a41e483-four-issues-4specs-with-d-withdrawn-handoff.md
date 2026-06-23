# 交接：EPM dogfood thread `3a41e483` 四问题根因核实 + 四份修复 spec（含一次撤回）

> **会话日期**：2026-06-22
> **来源**：`docs/problems/2026-06-22-thread-3a41e483-four-issues-investigation.md`（调研文档，A/B/C/D 四问题）
> **任务**：逐一核实四问题根因（不轻信调研文档叙述），写修复 spec 治本，C 顺带全量扫 prompt 硬编码。
> **本会话性质**：纯 spec 撰写 + 根因核实，**未动任何生产代码**。四份 spec 全部待实施，无一阻塞同事。

---

## 一、当前任务目标

调研文档把 thread `3a41e483`（EPM dogfood，28×4×7，端到端跑通但暴露 4 问题）的 A/B/C/D 四个问题摆出来，让接手 agent「确认根因 → 判断该不该立 spec → 给修复方向」。

本会话做的事：
1. **对每条根因逐字节核实代码**（不只信文档），发现 C 的根因文档判断错了、D 的初版方向错了。
2. **写/撤/改 4 份 spec**，全部治本（不叠加兜底，守 memory `feedback_isolate_root_cause_before_stacking_fallback_mechanisms`）。
3. **撤回 1 份 spec**（D 初版），重写。记了 1 条新 memory + 索引。

---

## 二、当前进展（✅ 已完成）

### 四份 spec 最终状态（全部在 `docs/superpowers/specs/`，待实施）

| Spec 文件 | 修什么 | 优先级 | 状态 |
|---|---|---|---|
| ✅ `2026-06-22-chart-parameters-json-filter-by-requires-columns-spec.md` | **B**：chart 层 `--parameters-json` 注入全范式 zone param 并集不裁剪，`plot_open_arm_time_ratio_bar` 被塞多余 `closed_arm_zones` → TypeError | 🔴 P0 真 bug | 待实施 |
| ✅ `2026-06-22-metric-metadata-sidecar-spec.md` | **A**：report-writer/data-analyst 啃 133K `plan_metrics.json` 找展示元数据撞 turn 超时（memory 第 4 类 seal 根因复发，对象 report-writer） | 🔴 P0 真 bug | 待实施 |
| ✅ `2026-06-22-identify-zone-info-persist-spec.md` | **D（重写版）**：identify_ev19 检测的 `zone_info` 落盘时被剥掉 → lead 为带依据反问得在 thinking 自己推断烧 turn | 🟡 真 bug（小） | 待实施 |
| ✅ `2026-06-22-prompt-hardcoded-counts-spec.md` | **C**：工具文案裸「5 个范式」漂移（实际 `SUPPORTED_PARADIGMS_V01` 已 6 个）+ code-executor 摘要「2 个指标」嘴瓢 | 🟢 P2 纪律 | 待实施 |
| ❌ `2026-06-22-column-signals-paradigm-candidates-spec.md` | **D 初版**（已撤回删除）：想让工具用 `_facts.json` zone_template 反推范式候选返回 ambiguous | — | **撤回**（违反 HITL L480） |

### Memory 更新（✅ 已落盘）
- 新增 `feedback_identify_zone_info_not_persisted_leads_to_lead_thinking_burn.md`（含撤回教训）
- MEMORY.md 索引补一行（在 seal 漏时段）

### 关键核实结论（✅ 已查清）
- **B 根因坐实**：`resolve_charts`（[resolve.py:444](../../../packages/ethoinsight/ethoinsight/catalog/resolve.py#L444)）一份全局 zone_overrides 传每个 `_chart_to_plan`（L463），`_chart_to_plan`（L1075/L1102）无条件全量注入，不按 chart `requires_columns` 裁剪。
- **A 根因坐实**：`plan_metrics.json` 的 `metrics[]` 按 subject 重复（28×5=140 条），report-writer/data-analyst 要的展示元数据去重后只 5 条。`metrics_summary` 不含展示元数据（只有 mean/std/n）。
- **C 根因纠正文档**：code_executor.py **无**「2 个指标」硬编码（grep 证实），是 deepseek 摘要嘴瓢。但扫到 identify_ev19 L458 / prep_metric_plan L92 有真硬编码。
- **D 价值定位纠正**：「减少一轮交互」不成立（L480 强制范式必须反问不许猜），真价值是「让反问带依据、防 lead 隐性猜测」+「检测产物别丢」。

---

## 三、关键上下文（项目结构 / 数据位置）

### 行为学同事数据落盘在 5 个地方（本会话核实，高频查）
1. `packages/ethoinsight/ethoinsight/_facts.json` — **62 EV19 模板变体结构化事实**（template_id/category/zone_template/inferred_zone_config），`source: demodata/ev19 templates/`，同事从 EV19 demodata 抽取。`ev19_facts.py` 加载，identify/guardrail/范式→模板映射消费。
2. `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-experiment/`（23 个 md）— 范式叙述文档（定位/模型/指标判读/与其他范式区分）。data-analyst/report-writer/lead read。
3. `docs/review-packages/`（10+ 包）— 同事 PR 原始 review 包，SSOT 源头（memory `feedback_ssot_lives_in_review_packages`）。
4. `golden-cases/`（6 个 baseline：epm/oft/ldb/fst/zero_maze/tst）— 黄金标准 case（domain 知识 + 回归 + SFT 种子三重角色）。
5. `packages/ethoinsight/ethoinsight/catalog/*.yaml`（7 个）— 范式指标/图表 SSOT（default_metrics/charts/statistics/zone_concept_params）。
6. （仓库外）`~/behavioral-book/reward-criteria/`（6 YAML）— 同事方法论判据，给 RL reward，不给 agent 直接消费。

### 独有 vs 共享 zone 信号（来自 `_facts.json`，本会话抽出）
- **共享（不能单独识别范式）**：`Closed and open arms` → **EPM + Zero Maze 共享**（thread `3a41e483` 绕圈根因）；`Novel object zones` → 两种旷场；`Start box, center, arms and goal zones` → Cross Maze-Fish + T-Maze。
- **EPM 独有**：`Closed-, open arms, head dip zone`（head-dip 列可收紧到 EPM）。
- 其他范式独有信号见会话记录（Barnes=20 holes、FST=diving zone、Sociability=chamber+cage 等）。
- ⚠️ **重要**：独有信号只是「反问依据」，**不替用户定范式**（lead prompt L480/L483 + guardrail ev19_template_provider.py:86-105 双层强制）。

### 6 范式分析区处理审计（本会话核实，compute 路径全对）
- **EPM**：双 list zone（open_arms/closed_arms），HITL column_aliases 对齐，物理列名透传 ✅（chart 路径有 B 隐患）
- **OFT**：center 单 zone（HITL 对齐）+ periphery（**regex 自动识别，不经对齐** ⚠️ 边界风险）
- **LDB**：双标量 zone（light/dark），HITL 对齐 ✅
- **Zero Maze**：双 list zone（open_zones/closed_zones，参数名与 EPM 不同），HITL 对齐 ✅（**chart 路径可能有同款 B 隐患，需验**）
- **FST/TST**：无 zone，纯 velocity/mobility 活动度 ✅

### 三阶段 zone 产物（别混淆，本会话踩过的坑）
| 产物 | 阶段 | 落盘 | 内容 | 谁修 |
|---|---|---|---|---|
| **column_aliases** | 对齐**后**（用户确认） | experiment-context.json | `{"zone x":"open_arm"}` 用户确认映射 | 已落盘已接好，D 不碰 |
| **zone_info** | 对齐**前**（identify 首跑） | template_candidates.json（D 要加） | `{"suspect_columns":["open","closed"]}` 疑似归属列 | **D spec 修** |
| **zone_template** | 静态数据 | `_facts.json` | `PlusMaze-FewZones: "Closed and open arms"` | 同事维护，撤回的旧 D 才想用 |

---

## 四、关键发现 / 决策

### 1. 撤回 D 初版的教训（记进 memory 了）
初版 D 提议「工具用 `_facts.json` zone_template 反推范式候选返回 ambiguous」。核实 lead prompt 后撤回：
- L480「范式推断失败必须 ask_clarification、不许默认猜测」+ L483「未明确选模板必须重问」+ L456「没指定范式=missing_info 必须反问」。
- 范式识别无论 unknown/ambiguous **都要问用户、都不许猜** → unknown→ambiguous **不减交互轮次**。
- 工具造候选违反「工具不猜、只检测；用户定、不替用户选」职责划分。
- **可复用铁律**：判「减少交互」类价值前先核实 HITL 反问铁律。

### 2. D 真根因不是「该造候选」是「检测了就丢」
identify_ev19 三条落盘路径（L609 unknown / L619 ok / L643 ambiguous）写 `template_candidates.json` 时**都没带 zone_info**。花 parse_header 成本检测出的 suspect_columns 只活在内存返回值。后果：lead 为带依据反问得在 thinking 自己推断 open/closed→EPM（烧 turn）；evidence 被 summarize 截断后要重读文件。

### 3. C 文档判断错了（已纠正）
调研文档 C.1 说「code-executor 硬编码了 2 个指标模板」——grep 证实 `code_executor.py` 无此硬编码，是 deepseek 生成摘要文本嘴瓢。真硬编码在 identify_ev19 L458（`f"v0.1 已实现 5 个范式"`，f-string 里「5」是字面量，`SUPPORTED_PARADIGMS_V01` 已 6 个）+ prep_metric_plan L92（「仅支持以下 5 个」）。**「5 实现 vs 文案 5」已漂移成 6 vs 5，C spec 坐实且紧迫。**

### 4. A 比文档更深一层
文档只说 report-writer prompt 诱导读大文件。核实发现 `plan_metrics.json` 之所以 133K 是因元数据按 subject 重复 28 遍（140 条），去重后只 5 条。且 `data_analyst.py:249` 有同构隐患——06-18 data-analyst spec 只删参数审计，**漏修展示元数据读取**。A spec 用去重旁路文件 `_metric_metadata.json` 一次性覆盖两处。

### 5. B 是 06-18 plot-scripts-zone-param-alignment spec 的收尾补丁
06-18 那份让 per-subject plot 脚本能接收 zone 参数（解决「拿不到」），却埋下「拿到的过宽」（不裁剪）。B 是同一条注入链的另一端。

---

## 五、未完成事项（按优先级）

### 🔴 P0（真 bug，每次 EPM 必触发，可立即实施，互不依赖）
1. **实施 B spec**（`2026-06-22-chart-parameters-json-filter-by-requires-columns-spec.md`）
   - 改 `resolve.py`：新增 `_filter_zone_overrides_for_chart(ch, zone_overrides, cat)`，在 `resolve_charts` L463/L493 调用前按 chart `requires_columns` 裁剪。
   - **顺带验证 Zero Maze chart 路径**（同款双 list zone，可能有同款注入过宽）——B spec 测试段已写「多范式回归」，实施时具体化。
   - 守纪律：concept 关键词匹配复用 `_build_zone_aliases_overrides` 已有归一化，不新写；动前 grep review-packages。
2. **实施 A spec**（`2026-06-22-metric-metadata-sidecar-spec.md`）
   - 新增 `metric_metadata_to_dict`（去重，5 条非 140 条），plan 落盘处写 `_metric_metadata.json`。
   - report_writer.py L176-192 + data_analyst.py L249-262 改读旁路 + 加 `<thinking_discipline>`。
   - 守 SSOT：旁路是 catalog→plan 的去重投影，不是新 SSOT；改前确认 metrics_summary 不含展示三字段（已确认）。

### 🟡 P1（真 bug 但小）
3. **实施 D spec**（`2026-06-22-identify-zone-info-persist-spec.md`）
   - 三条落盘路径 L609/619/643 的 data 各加 `"zone_info": zone_info`。
   - lead_agent/prompt.py L479-484 加「范式反问带列依据、不瞎猜」铁律（open/closed 同时支撑 EPM+ZeroMaze 必须都列）。
   - 工具仍返回 unknown（守 L480 不猜），只沉淀检测产物。
   - **可考虑降级**：D 改动很小（3 处加字段 + 1 句 prompt），可并入 B/A 实施批次顺手做，不单独立 PR。

### 🟢 P2（纪律，可立即实施）
4. **实施 C spec**（`2026-06-22-prompt-hardcoded-counts-spec.md`）
   - identify_ev19 L455-465：`message`/`hint` 改用 `len(SUPPORTED_PARADIGMS_V01)` + 动态 labels。
   - prep_metric_plan L88-92 docstring 去清单去数字（指向 identify_ev19）。
   - code_executor.py 加 `<summary_number_discipline>` 段。
   - 全量扫描契约测试 `test_no_hardcoded_drifting_counts_in_prompts`（白名单无害描述性数字）。

### ⚪ 发现但未立 spec（记此供后续）
5. **OFT periphery 不经对齐**：`_find_periphery_zone_column` regex 自动识别（`in_zone.*(peripher|edge|wall|border|outer)`），用户边缘区列名非标准（「外周」）会漏匹配 → thigmotaxis 漏算。违反 memory `feedback_oft_single_zone_must_ask_not_guess` 的精神（center 问了 periphery 没问）。低优先级（次要指标，center 主指标正确），可单立小 spec。

---

## 六、建议接手路径

### 第一步：读这三份文件建立全貌
1. 本 handoff
2. `docs/problems/2026-06-22-thread-3a41e483-four-issues-investigation.md`（原调研，但 C/D 段已被本会话纠正，以本 handoff 为准）
3. memory `feedback_identify_zone_info_not_persisted_leads_to_lead_thinking_burn.md`（撤回教训）

### 第二步：从 B 开始实施（最独立、改动最小、TDD 路径最清晰）
```
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
# 读 B spec → 改 resolve.py（新增 _filter_zone_overrides_for_chart）
# 跑测试：pytest tests/test_chart_zone_overrides_filtered.py（新建）
# 全量：pytest tests/（守 worktree venv 纪律，见 memory）
```
B 实施时**顺带验证 Zero Maze**（grep zero_maze.yaml 的 chart 段，看有没有只收 open_zones 的图会被塞 closed_zones）。

### 第三步：A → C → D（或 D 并入 A/B 批次）

---

## 七、风险与注意事项

### 容易混淆的点
1. **D 的三阶段产物别混**（column_aliases 已落盘 vs zone_info 待落盘 vs zone_template 静态数据）——见上表。下一个 agent 很可能问「column_aliases 不是已经落盘了对齐结果，D 这不是重复？」，答案是 D 修的是更早阶段（对齐前）的检测信号。
2. **「同事数据是不是都落盘了」**——是，5 个地方全落盘全接好。没有任何同事数据在「待产出」状态（结构聚合 Issue #98 / golden case Issue #90 是「已有 baseline 等扩充」非从零没有）。
3. **撤回的旧 D spec 已删除**——别在 specs 目录找它。它的思路（工具造 ambiguous 候选）被否决，理由记在 memory + 新 D spec §1.5。

### 不建议的方向
1. **别让工具替用户猜范式**（哪怕独有信号几乎确定，如 head dip→EPM）——L480 + guardrail 双层禁止。独有 zone 信号只是反问依据。
2. **别在 plot/compute 函数加 `**kwargs` 兜底吞多余参数**（B spec §2.5）——会让参数注入错误变静默。治本在注入端裁剪。
3. **别调 thinking_enabled/timeout 旋钮**（A spec §6.2）——会把响亮超时变哑故障。A 治本是让 turn 能正常结束（旁路文件），不是调旋钮。
4. **C 别去找「code-executor 里硬编码的 2 个指标模板」**——它不存在（调研文档误判已纠正）。治法是 prompt 加数字纪律 + 静态契约测试。
5. **D 别碰工具返回契约**——仍 unknown，只加落盘字段。别返回 ambiguous 候选。

### 工程纪律（memory 铁律，实施时守）
- 改 harness 核心（subagents/tools/agents）后**必须裸导入两生产入口**：`PYTHONPATH=. python -c "import app.gateway"` + `python -c "from deerflow.agents import make_lead_agent"`（memory `feedback_conftest_mock_hides_circular_import_verify_bare_prod_import`）。
- worktree venv 纪律：有独立 venv 用它跑；借主仓 venv 须 importlib 加载 worktree 源（memory `feedback_worktree_shares_main_venv_editable_link_tests_must_use_importlib`）。
- 动 catalog/SSOT 前 grep review-packages（memory `feedback_ssot_lives_in_review_packages`）。
- 守 SSOT：B 复用已有 concept 归一化、A 旁路是投影非新源、C 引用 SUPPORTED_PARADIGMS_V01 非复制清单、D 只消费不复制 zone_template。

### cwd 陷阱（本会话反复踩）
Bash 工具每次新调用 cwd 会重置回 `/home/wangqiuyang`（不是仓库根 `noldus-insight/`）。**所有 Bash 命令用绝对路径或开头 `cd /home/wangqiuyang/noldus-insight &&`**，别依赖前一次的 cd。

---

## 八、下一位 Agent 的第一步建议

1. **读本 handoff + memory `feedback_identify_zone_info_not_persisted_leads_to_lead_thinking_burn.md`**（5 分钟，建立全貌 + 撤回教训）。
2. **从 B 开始**：读 `docs/superpowers/specs/2026-06-22-chart-parameters-json-filter-by-requires-columns-spec.md`，在 worktree 开分支（命名 `worktree-chart-params-filter-<hash>` 之类），按 spec §三改动清单改 `resolve.py`。
3. **B 测试先行**：spec §四的 `test_open_arm_bar_gets_only_open_zone_param` 红线测试先写，复现 dogfood（open_arm_time_ratio_bar 收到 closed_arm_zones），再改 resolve.py 转绿。
4. **B 实施时顺带核 Zero Maze**：grep `zero_maze.yaml` chart 段 + zero_maze plot 脚本签名，确认同款双 list zone 有无同款注入过宽，有则一并修 + 测。
5. B 出 PR 后接 A，再 C，D 可并入 A/B 批次或单独小 PR。

**四份 spec 全部待实施，无一阻塞同事，互不依赖（B/A/C/D 可任意顺序或并行）。建议顺序 B → A → C → D（按 P0/P0/P2/P1 + 改动大小）。**
