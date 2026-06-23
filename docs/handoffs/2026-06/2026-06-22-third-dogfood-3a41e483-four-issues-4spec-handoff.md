# Handoff：第三轮 EPM dogfood (thread 3a41e483) 四问题定位 + 4 spec（含 D 撤回/重写）

> 来源：用户在生产 ECS 跑的 EPM dogfood thread `3a41e483-aea1-4699-987f-0be2b5fb487f`（28 只 × 4 组 × 7），端到端跑通但暴露 4 个独立问题。
> 调研文档（本会话前置）：`docs/problems/2026-06-22-thread-3a41e483-four-issues-investigation.md`
> 本会话产出：4 份修复 spec（A/B/C/D）+ 1 份 memory + 多处判断纠正。
> 日期：2026-06-22

---

## 当前任务目标

定位 thread `3a41e483` 暴露的 4 个问题（A/B/C/D）的真实根因，**逐个对着代码核实**（不轻信调研文档叙述），为每个写工程化治本 spec。用户全程要求「治本不治标」「守 SSOT/纪律」「核实不假设」。

---

## 当前进展（✅ 已完成）

### 核实 + 写 spec

| 问题 | 真 root cause（代码坐实） | spec | 状态 |
|---|---|---|---|
| **B** | `resolve_charts`（resolve.py:444）给每张图注入**全范式 zone param 并集**，不按 chart 的 `requires_columns` 裁剪。`plot_open_arm_time_ratio_bar` 底层 `compute_open_arm_time_ratio` 只收 `open_arm_zones`，被塞 `closed_arm_zones` → TypeError | `2026-06-22-chart-parameters-json-filter-by-requires-columns-spec.md` | ✅ 写完，🔴 P0 |
| **A** | report-writer / data-analyst 的 prompt 指示 read 整个 133K `plan_metrics.json` 找展示/判读元数据（report_writer.py:176-192、data_analyst.py:249-262）。元数据按 subject 重复 28 遍（140 条），thinking 在文件里扫匹配/估算行号撑爆 turn 超时 → SealGate（after_model 门）结构性救不了 → seal 漏调 → lead 重派。memory 第 4 类根因**复发**（对象从 data-analyst 换 report-writer） | `2026-06-22-metric-metadata-sidecar-spec.md` | ✅ 写完，🔴 P0 |
| **D** | `identify_ev19_template` 检测出的 `zone_info`（含 suspect_columns）**落盘 `template_candidates.json` 时被剥掉**（L609/619/643 三条路径都没存）→ lead 为带依据反问得在 thinking 自己推断 open/closed→EPM 烧 turn + 检测不沉淀 | `2026-06-22-identify-zone-info-persist-spec.md` | ✅ 写完，🟡 P1 |
| **C** | 工具文案裸数字漂移：`identify_ev19_template_tool.py:458` 硬编码「v0.1 已实现 **5** 个范式」但 `SUPPORTED_PARADIGMS_V01` **实际 6 个**（已漂移！）+ `prep_metric_plan_tool.py:92` 同款 + code-executor 摘要「2 个指标」嘴瓢（prompt 无硬编码，是 deepseek 自由生成） | `2026-06-22-prompt-hardcoded-counts-spec.md` | ✅ 写完，🟢 P2 |

### 重大判断纠正（本会话核心价值，必须读）

1. **C 的调研文档判断错了**。文档 C.1 说「code-executor 硬编码了 2 个指标模板」——grep 证实 `code_executor.py` **无此硬编码**，是 LLM 生成摘要嘴瓢。但扫描发现**真硬编码**：identify_ev19 的「5 个范式」f-string（`SUPPORTED_PARADIGMS_V01` 在旁 L455 却没用 len()）+ prep_metric_plan docstring。C spec 覆盖这两类。

2. **D spec 被撤回重写过一次**。初版「工具用列信号造 ambiguous 候选」（`2026-06-22-column-signals-paradigm-candidates-spec.md`，**已删**）——核实 lead prompt L480/L483/L456 后撤回：
   - L480「范式推断失败必须 ask_clarification、**不允许默认猜测**」
   - L483「未明确选模板必须重问、**不许默认推荐项**」
   - L456「没指定范式」=missing_info 必须反问
   - **范式识别无论 unknown/ambiguous 都必须反问用户、都不许猜** → unknown→ambiguous **不减少交互轮次**，旧 D 核心价值主张站不住 + 工具造候选违反「不猜」铁律。
   - 重写为新 D（`identify-zone-info-persist`）：只落盘 zone_info（检测了别丢）+ lead prompt 加「带列依据反问」铁律。**工具仍返回 unknown（守 L480），只是把检测产物沉淀。**

3. **「同事 PR 已处理过」的边界澄清**。用户质疑 D 是不是同事 PR `afe70e24` 做过了。核实：`afe70e24` 只做 `_filter_candidates_by_zone` 的**过滤**（有候选时剔 NoZones），D 做**生成**（候选为空时造候选——旧 D）/ **落盘**（检测产物沉淀——新 D）。两者正交，都用 suspect_columns 但作用点不同。新 D 不造候选，只落盘，与 `afe70e24` 完全不冲突。

4. **zone_info（新 D 落盘）vs column_aliases（已落盘）是两阶段产物，别混**：
   - **zone_info**（identify `_detect_zone_config` 产物）= **对齐前检测**：纯模式匹配「疑似有 open/closed 归属列」，不知含义，可信度低，产生于 Gate 1 前。新 D 让它沉淀进 `template_candidates.json`。
   - **column_aliases**（`set_experiment_paradigm(column_semantics=)` 产物）= **对齐后确认**：用户确认 `zone x = open_arm` 的映射表，可信度高，产生于 Gate 1 后，**已落盘**进 `experiment-context.json`，prep_metric_plan/prep_chart_plan 自读，下游 code/chart 全程接好（物理列名透传，源文件不改）。
   - **新 D 不重复造 column_aliases**，只补更早阶段（对齐前）的检测产物沉淀。

### 其他产出

- ✅ memory：`feedback_identify_zone_info_not_persisted_leads_to_lead_thinking_burn.md`（含撤回教训）+ MEMORY.md 索引一行。
- ✅ 6 范式分析区处理过程审计（用户要求）：EPM/OFT/LDB/ZeroMaze/FST/TST 的 zone 链路全部核实，结论见下「关键发现」。

---

## 关键发现（接手必读）

### 1. 6 范式分析区处理审计结果

| 范式 | zone 模型 | 对齐 | 正确性 | 风险 |
|---|---|---|---|---|
| EPM | 双 list(open_arm_zones/closed_arm_zones) | HITL column_aliases | ✅ compute 正确 | 🔴 chart 注入过宽(B spec) |
| OFT | center 标量 + periphery | center:HITL; **periphery:regex 自动** | ✅ center 正确 | ⚠️ periphery 不经对齐(「外周」等非标名漏匹配) |
| LDB | 双标量(light_zone/dark_zone) | HITL column_aliases | ✅ 正确 | 无 |
| Zero Maze | 双 list(open_zones/closed_zones) | HITL column_aliases | ✅ 正确 | ⚠️ 可能同 EPM 的 B 隐患(B 实施时验证) |
| FST | 无 zone | — | ✅ 正确 | 无 |
| TST | 无 zone | — | ✅ 正确 | 无 |

**核心机制对**：物理列名透传（resolve.py:737 `overrides[param_name] = cols`，cols 是物理列名），源文件不改一个字，列叫 `open`/`zone A`/`中心区` 都能命中。

**真 bug 只有 EPM chart 路径（B）**。compute 路径全对。

### 2. OFT periphery 不经 column_aliases 对齐（潜在新 spec）

`_find_periphery_zone_column`（oft.py:43-50）用 regex `in_zone.*(peripher|edge|wall|border|outer)` 自动找边缘区列，**不经过 HITL 对齐**。用户的边缘区列叫「外周区」「外环」等非标准名 → regex 漏匹配 → thigmotaxis 漏算（graceful 降级，不崩）。违反 memory `feedback_oft_single_zone_must_ask_not_guess`（center 问了，periphery 靠猜）。优先级低（thigmotaxis 是次要指标，center 主指标对），**未立 spec，记为已知边界**。

### 3. 行为学同事数据落盘在 5 个地方（用户问过，高频复用）

| # | 位置 | 是什么 |
|---|---|---|
| 1 | `packages/ethoinsight/ethoinsight/_facts.json` | 62 变体结构化事实（template_id/category/zone_template/inferred_zone_config），source: EV19 demodata，identify/guardrail 用 |
| 2 | `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-experiment/`（23 md） | 范式叙述文档，data-analyst/report-writer/lead read |
| 3 | `docs/review-packages/`（10+ 包） | 同事 PR 原始 review 包，SSOT 源头 |
| 4 | `golden-cases/`（6 baseline） | 黄金标准 case（domain 知识+回归+SFT 种子三重角色） |
| 5 | `packages/ethoinsight/ethoinsight/catalog/*.yaml`（7 个） | 范式指标/图表 SSOT |
| 6 | `~/behavioral-book/reward-criteria/`（仓库外，6 YAML） | 方法论判据（给 RL reward，不给 agent 直接消费） |

**全部已落盘，无一待产出**（除 Issue #98 结构聚合 / Issue #90 golden case 的扩充，是「已有 baseline 等扩」，非「从零没有」）。

### 4. 每范式独有 zone 信号（来自 _facts.json，用户问过）

**独有（可作反问依据，但不替用户定！）**：
- PlusMaze(EPM)：`Closed-, open arms, head dip zone`（head dip EPM 独有）
- BarnesMaze：`20 hole zones`；PorsoltCylinder(FST)：`Diving zone`；Sociability：`Chamber and cage zones`；MWM：`Platform, quadrants`；等

**共享（不能单独识别，必须 ambiguous）**：
- `Closed and open arms` → **PlusMaze + ZeroMaze 共享**（这就是 thread `3a41e483` open/closed 分不开 EPM/ZM 的根因）
- `Novel object zones` → OpenFieldCircle + Rectangle；`Start box, center, arms and goal zones` → Cross Maze-Fish + T-Maze

### 5. 独有 zone 信号只是「反问依据」，绝不替用户决定（安全闸）

两层保险：① lead prompt L480/L483 不许猜；② guardrail `ev19_template_provider.py:86-105` 代码拦「未确认就 set_experiment_paradigm」。工具只检测/落盘，lead 只转述/反问，用户拍板。**D spec 不碰这个边界。**

---

## 未完成事项（按优先级）

### 🔴 P0（真 bug，可立即实施，互不依赖）

1. **B：chart parameters-json 按 requires_columns 裁剪**
   - spec：`docs/superpowers/specs/2026-06-22-chart-parameters-json-filter-by-requires-columns-spec.md`
   - 改动点：`resolve.py` 新增 `_filter_zone_overrides_for_chart(ch, zone_overrides, cat)`，在 `resolve_charts`（L463 主路径 + L493 fallback 路径都要改）调 `_chart_to_plan` 前裁剪。复用 `_extract_concept_keyword` + `cat.resolved_zone_concepts`，**不新写匹配逻辑**（守 SSOT）。
   - **实施时顺带验证 Zero Maze chart 路径**（同款双 list zone，可能有同款注入过宽），写进多范式回归测试。
   - 不加 `**kwargs` 兜底（治标，会让配置错变静默）。

2. **A：展示元数据瘦身旁路文件**
   - spec：`docs/superpowers/specs/2026-06-22-metric-metadata-sidecar-spec.md`
   - 改动点：① plan 落盘处额外写 `_metric_metadata.json`（按 metric id 去重，5 条非 140 条，几 KB）——grep `plan_metrics.json` 写出处定位；② report_writer.py:176-192 + data_analyst.py:249-262 两处 prompt 改读旁路 + 加 `<thinking_discipline>` 段；③ report-writer workflow step 2 删掉 `read_file plan_metrics.json`。
   - **覆盖两处 prompt**（report-writer + data-analyst），别只改 report-writer（data-analyst 同构隐患 06-18 spec 漏修了）。

### 🟡 P1（真 bug，小修）

3. **D：identify zone_info 落盘**
   - spec：`docs/superpowers/specs/2026-06-22-identify-zone-info-persist-spec.md`
   - 改动点极小：identify_ev19_template_tool.py 三条落盘路径（L609/619/643）的 data dict 各加 `"zone_info": zone_info` + lead prompt L479-484 加「带列依据反问」铁律。
   - **可降级**：这么小的改动，也可不单独立 spec，并入 B/A 实施时顺手改。用户尚未拍板（保留独立 spec vs 降级成附带修）。

### 🟢 P2（纪律，可立即实施）

4. **C：prompt 硬编码数字根治**
   - spec：`docs/superpowers/specs/2026-06-22-prompt-hardcoded-counts-spec.md`
   - 改动点：① identify_ev19_template_tool.py:455-465 的 f-string「5 个范式」改 `len(SUPPORTED_PARADIGMS_V01)` + 动态 labels；② prep_metric_plan_tool.py:88-92 docstring 去清单去数字（指向 identify_ev19_template）；③ code_executor.py 加 `<summary_number_discipline>` 段；④ 全量扫描契约测试。
   - **已坐实漂移活样本**：`SUPPORTED_PARADIGMS_V01` 已 6 个，文案还写 5 个。

### 未立 spec（已知边界）

- **OFT periphery 不经对齐**：见关键发现 #2。优先级低，记为已知。若要立项：「OFT periphery/边缘区应经 column_aliases 对齐，而非 regex 猜」。

---

## 建议接手路径

### 第一步：从 B 开始（最独立、改动最小、TDD 路径最清晰）

1. 读 `docs/superpowers/specs/2026-06-22-chart-parameters-json-filter-by-requires-columns-spec.md`
2. 读前序 `docs/superpowers/specs/2026-06-18-plot-scripts-zone-param-alignment-spec.md`（B 是它的收尾补丁——那份解决了「plot 拿不到参数」，B 解决「拿到的参数过宽」）
3. 核实 `resolve.py:444`（`_build_zone_aliases_overrides` 调用点）+ `resolve.py:463/493`（`_chart_to_plan` 主/fallback 路径）+ `resolve.py:644-739`（`_build_zone_aliases_overrides` 内部 concept→param 映射，复用它）
4. 写 `_filter_zone_overrides_for_chart`，红→绿 TDD（spec §4 已给测试清单）
5. **多范式回归含 Zero Maze**（验证同款双 list zone chart）

### 工程纪律（违反必踩坑，见 memory）

- 改 harness 核心（subagents/tools/agents）后**必须裸导入两生产入口**：`PYTHONPATH=. python -c "import app.gateway"` + `python -c "from deerflow.agents import make_lead_agent"`（conftest mock 致 pytest 假绿）。
- worktree 共享主仓 venv → 测试用 importlib 加载 worktree 源（别裸 import 测主仓）；有独立 venv 用它跑。
- 改 catalog/SSOT 前 grep `docs/review-packages/`（同事权威源）。
- 别叠加兜底（根因未隔离前）；兜底触发率做成可观测验收项。
- prompt 用**正面指令**不用「禁止/不要」（deepseek 反向激活）。

---

## 风险与注意事项

1. **别再提议「工具造 ambiguous 候选」**（D 撤回教训）：范式识别必须反问用户（L480/L483），工具不猜。判「减少交互」类价值前先核 lead prompt HITL 反问铁律。

2. **C 调研文档判断错了**：别去找「code-executor 硬编码 2 个指标模板」——它不存在。真硬编码在 identify_ev19/prep_metric_plan。

3. **D 别和 column_aliases 混**：新 D 落盘的 zone_info 是「对齐前检测」，column_aliases 是「对齐后确认」（已落盘 experiment-context.json，已接好 code/chart）。新 D 不重复造后者。

4. **B 别加 `**kwargs` 兜底**：保持 compute 函数严格签名，配置错要响亮炸，裁剪在注入端做。

5. **A 别只改 report-writer**：data-analyst 同构隐患（data_analyst.py:249），06-18 spec 漏修，A 要一并覆盖。

6. **cwd 坑**：本会话 Bash 工具工作目录频繁重置回 `/home/wangqiuyang`（仓库在 `noldus-insight/`），所有相对路径 grep/find 会失败。**接手时所有 Bash 命令用绝对路径或开头 `cd /home/wangqiuyang/noldus-insight &&`**。

7. **`SUPPORTED_PARADIGMS_V01` 已 6 个范式**（epm/open_field/zero_maze/light_dark_box/forced_swim/tail_suspension），不是 5 个。文案漂移是 C 的活样本。

---

## 下一位 Agent 的第一步建议

**直接进入 B 的实施**（最独立、价值最高、真 bug 每次必触发）：

```bash
cd /home/wangqiuyang/noldus-insight
# 1. 读 B spec
cat docs/superpowers/specs/2026-06-22-chart-parameters-json-filter-by-requires-columns-spec.md
# 2. 核实关键代码（绝对路径）
sed -n '440,500p' packages/ethoinsight/ethoinsight/catalog/resolve.py   # resolve_charts 调用点
sed -n '644,740p' packages/ethoinsight/ethoinsight/catalog/resolve.py   # _build_zone_aliases_overrides (复用其 concept→param 映射)
# 3. 建 worktree 开干
```

若用户要先做别的：A（report-writer thinking 过载，P0，稍复杂涉及新增旁路文件）、C（硬编码，P2，最简单）、D（极小，可并入 B/A 顺手改）。

---

## 相关文件索引

- 调研来源：`docs/problems/2026-06-22-thread-3a41e483-four-issues-investigation.md`
- 4 spec：`docs/superpowers/specs/2026-06-22-{chart-parameters-json-filter-by-requires-columns,metric-metadata-sidecar,identify-zone-info-persist,prompt-hardcoded-counts}-spec.md`
- 已删（撤回）：`2026-06-22-column-signals-paradigm-candidates-spec.md`（旧 D，误判「工具造 ambiguous」）
- memory：`~/.claude/projects/-home-wangqiuyang/memory/feedback_identify_zone_info_not_persisted_leads_to_lead_thinking_burn.md`
- 前序 spec（B 依据）：`docs/superpowers/specs/2026-06-18-plot-scripts-zone-param-alignment-spec.md`
- 前序 spec（A 依据）：`docs/superpowers/specs/2026-06-18-data-analyst-thinking-overload-spec.md`
- guardrail（D/HITL 依据）：`packages/agent/backend/packages/harness/deerflow/guardrails/ev19_template_provider.py:86-105`
- lead prompt HITL 铁律：`packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py:453-484`
