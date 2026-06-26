# 2026-06-24 EPM 真实数据 dogfood：双 thread 对照，全链路健康 + chart-maker 读盘竞态误判（良性）

> **目的**：把 2026-06-24 对真实 EPM 数据（`Raw data-EPM-Xuhui-28`，28 trials，Treatment 分 XX/XY/YY/YZ 四组）的**两次端到端 dogfood** 沉淀。两次 run 用同一份数据但不同 Gate 3 回答，结果天差地别——正好把「E2E_FULL_ASKVIZ 可视化门」与「chart-maker 读盘竞态」两个点一次性讲透。
>
> **关联 thread**：
> - `2c95aada-e9db-4e12-89da-eceafa691e1f`（用户 `e281f251`，generic `确认` 驱动，**不画图**）
> - `a6e3775c-5a4e-4641-877b-ab6c5cd469e2`（用户 `cd95effa`，手回 "A. 是，画图"，**112/112 画图成功**）
>
> **关联 run dir**：`/tmp/noldus-e2e-runs/20260624-173737/`（thread `2c95aada` 的完整取证：`clarifications.json` / `sse-0..8.txt` / `analyze.txt` / `gateway-excerpt.log` / `final.png`）
>
> **与其它 chart-maker problem doc 的关系**：本文记录的是**与 [silent-drop-completed](../problems/2026-06-24-chart-maker-run-chart-plan-silent-drop-completed-handoff.md) / [permissionerror](../problems/2026-06-24-chart-maker-run-chart-plan-permissionerror.md) 不同的形态**——那两份是 chart **真丢/真失败**；本次 `a6e3775c` 是 chart **全成功**，只是 chart-maker **过程中自述**「n_rendered=4 / 108 failed」的**读盘竞态误判**（最终 handoff 正确）。属良性，但暴露了 `run_chart_plan` 核盘时机的健壮性缺口，见末尾问题记录。

---

## 0. 一句话结论

EPM 真实数据全链路（code-executor → data-analyst → report-writer，可选 chart-maker）**完全健康**：统计决策树正确（正态→ANOVA / 非正态→Kruskal-Wallis），data-analyst `sealed_by=finalize`（#187 修复持续生效），report.md 含扎实的组间比较 + 离群警示。两次 run 的**唯一差异是 Gate 3 可视化门的回答**：

- generic `确认` → lead 主动判 `viz_choice=no` → **不派 chart-maker**（非 bug，是 driver 限制 + lead 对模糊回答的合理推理）
- 手回 "A. 是，画图" → `viz_choice=yes` → chart-maker 全跑 **112/112 成功**，report.md 嵌入代表图

**没有静默丢图、没有 PermissionError、没有伪装 completed**——这与同日早些时候的两次 chart-maker 退化（thread `89be7344` / `590fbbd3`）形成对照，说明 `run_chart_plan` 的渲染层在本次数据/环境下**工作正常**。唯一新观察：chart-maker **过程中**会短暂自述「只渲染了 4 张」，是进程池 flush 前的读盘竞态，最终 seal 是真相（112），**无需修，但值得记**。

---

## 1. 上下文（30 秒）

- **EthoInsight**：行为学 AI 分析助手，Lead → code-executor → data-analyst → [chart-maker] → report-writer 流水线。
- **数据**：`~/DemoData/real_data/Raw data-EPM-Xuhui-28`，28 个 xlsx（每 Trial 1 只），EV19 模板 `PlusMaze-FewZones`，含 `open`/`closed` 区域归属列（划区变体），按 `Treatment` 分 XX/XY/YY/YZ 四组各 n=7。
- **E2E_FULL_ASKVIZ 意图**：用户说「分析」（模糊总称，未明说要图）→ 走 ASKVIZ 路径：跑完解读后**反问**「需要我把结果可视化成图吗? A. 是 / B. 不用」，用户答后 lead 调 `set_viz_choice(choice='yes'|'no')` 落盘 gate3，`yes`→派 chart-maker，`no`→直接 report-writer。Gate 3 反问模板 + set_viz_choice 规则在 [`agents/lead_agent/prompt.py:431-451`](../../packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py)。
- **本次为什么跑两次**：第一次用 `/noldus-insight-e2e` skill 的 generic `确认` 驱动（thread `2c95aada`），目的是自动化回归；generic `确认` 满足不了 Gate 3 的 A/B 选择，于是 lead 判 `no` 没画图。第二次用户在浏览器手动回 "A"（thread `a6e3775c`），验证画图路径。

---

## 2. 两次 run 对照表

| 维度 | thread `2c95aada`（generic 驱动） | thread `a6e3775c`（手回 A） |
|---|---|---|
| Gate 3 回答 | generic `确认`（×3，lead 判歧义） | "A. 是，把刚才的结论画成图" |
| `viz_choice` | **`no`** | **`yes`** |
| chart-maker 是否派遣 | ❌ 未派 | ✅ 派遣 |
| `outputs/*.png` | 0 | **112**（28×4 图种） |
| chart handoff | 不存在 | `chart_files=112, failed=0, sealed_by=model` |
| report.md 图表引用 | "(无可视化输出——用户未选择生成图表)" | 嵌入 `{{img:plot_trajectory_s*.png}}` + bar 图（4 个代表/离群 subject） |
| code-executor | ✅ sealed `run_plan`，140/140 指标 | ✅ 同 |
| data-analyst | ✅ sealed `finalize`，0 errors | ✅ sealed `finalize` |
| report-writer | ✅ sealed `model`，9271B 6 章节 | ✅ sealed `model` |
| 总耗时 | ~19 min（1153s），8 轮 HITL | —（手动，未计时） |

**核心结论**：分析核心三段（code/data/report）两次都健康；**画不画图完全由 Gate 3 回答决定**，与渲染层无关。

---

## 3. 真实分析产物（dogfood 价值，两次一致）

报告质量高，统计决策树 + 判读哲学都对：

- **组间比较主结论**：{XX,XY} vs {YY,YZ} 在 `open_arm_time_ratio` 上 **Kruskal-Wallis H=19.69, p=0.0002, η²=0.20 大效应量**；跨簇事后比较全显著（Cohen's d>0.85），簇内不显著——典型两簇分离。（注：`2c95aada` 走 4 组 ANOVA/KW；`a6e3775c` lead 据用户「XX 对照、其余实验」合并成 control(7) vs treatment(21) 两两比较，Mann-Whitney p=0.006~0.009。两种聚合都对，取决于用户给的对照语义。）
- **自动参数/非参数分支**：正态+齐性 → ANOVA / t-test；非正态 → Kruskal-Wallis / Mann-Whitney U。决策树按预期。
- **运动性混杂已排除**：`total_entry_count` 组间无差异（p=0.186 / 0.593），结论站得住——回避是焦虑样特异性，非活动力下降。
- **离群个体识别**：Trial 3（XX/Control）、Trial 12（XY/Treatment）总进入仅 1 次 + 开臂时间比例 1.0（全测试期滞留开臂），极不典型，建议核查视频轨迹（疑似掉落/跟踪丢失/分区错配）。排除后组均值大幅下降。Trial 19 开臂进入 0 次。
- **判读语言合规**：用「焦虑样回避行为差异」「趋近-回避冲突」相对表述，未踩绝对阈值禁令（守 CLAUDE.md 第 9 条）。
- **统计功效诚实标注**：n=7 偏小，`open_arm_entry_count` 大效应量（η²=0.27 / d=0.90）但 p=0.054 未过 Bonferroni α=0.01，明确提示功效不足而非否定。

---

## 4. 为什么第一次（`2c95aada`）没画图——根因（已用 SSE 逐字取证）

**不是 bug，是 lead 对模糊回答的合理推理 + driver 限制。** Gate 3 问「需要可视化吗? A.是 / B.不用」，generic 驱动连回 3 次 `确认`。lead 的真实推理（`sse-8.txt` 逐字）：

> *"I think the most pragmatic interpretation is that '确认' 3 times means they accept the results as-is and want to proceed to the final deliverable they originally asked for: a report. I'll go with B... The user said '确认' which could mean 'yes, draw the charts' or 'yes, give me the report.' Given the original request was for a report, I'll go with B."*

随后 `set_viz_choice(choice='no')` 被显式调用 5 次（第一次空 args，后 4 次 `{"choice":"no"}`），**lead 主动选了 no，不是工具默认值兜底**。lead 的依据：(1) prompt 规则 `prompt.py:451`「歧义剩余偏 E2E_FULL_ASKVIZ（让用户选，代价小）」，但 lead 认为「已问 3 次」后实用主义压过该规则；(2) 我发的 `E2E_REQUEST_TEXT` 字面只说「生成报告」没提图，lead 据此推断用户要报告。

**判据**：这是 `/noldus-insight-e2e` skill 明确警告的 **generic-driver 限制**——Gate 3 需要真实 A/B 答案，`确认` 满足不了。**不是 agent 回归**。要让自动驱动也画图，需在 driver 里对 Gate 3 可视化问题专门回 "A"（但 skill 故意保持 paradigm-agnostic，不 bake 具体答案）。

> 旁注：第二次 `a6e3775c` 证明——同一份数据、同一套代码，只要 Gate 3 回 "A"，chart-maker 就全跑成功。**反证渲染层无问题。**

---

## 5. chart-maker 读盘竞态误判（良性，新观察，thread `a6e3775c`）

chart-maker 在 `run_chart_plan` 执行**过程中**，于 trace 里自述：

> *"status=partial, n_rendered=4, n_failed=108, failures=[]... ls shows all 112 png files exist on disk... The tool miscounted."*

并一度想写 `/tmp/fix_handoff.py` 去「修正」handoff。但**最终它没改盘**，而是走了正确的 `seal_chart_maker_handoff`（`sealed_by=model`），最终 handoff 是真相：

```
status=completed, sealed_by=model
chart_files=112（全列出）, failed_charts=0
summary="112/112 张图表生成完成"
gate_signals.charts_generated=112, failed_charts=0
```

**根因推断**：`run_chart_plan` 的 `ProcessPoolExecutor` 进程内并行渲染 112 张图，chart-maker 在**进程池尚未全部 flush 到磁盘**的窗口期核盘（`ls outputs/*.png`），只看到最后一批（s27 的 4 张图种）已落盘，于是误判「4 渲染 / 108 失败」。等 seal 时再核，112 已全在盘，handoff 正确。

**为什么是良性**：误判只在 chart-maker **自己的思考过程**里短暂出现，**没有污染最终 handoff、没有丢图、没有伪装**。chart-maker 甚至正确地克制住了「写脚本改盘」的冲动，改走 seal。

**为什么仍值得记（+ 单独 problem 记录）**：`run_chart_plan` 号称「逐个核 output png 真落盘（磁盘真相）」，但核盘时机存在竞态——若 chart-maker 下次没那么克制（真去 `fix_handoff.py` 改盘，或据误判的 `n_failed=108` 向 lead 谎报 partial），就会从良性退化为 [silent-drop-completed](../problems/2026-06-24-chart-maker-run-chart-plan-silent-drop-completed-handoff.md) 那种自洽性破坏。**建议**：`run_chart_plan` 在核盘前 `ProcessPoolExecutor.shutdown(wait=True)` 等所有进程退出，或核盘后对 `n_rendered + n_failed != n_total` 的情况重核一次。详见 [problem 记录](../problems/2026-06-24-chart-maker-run-chart-plan-disk-race-false-partial.md)。

---

## 6. 取证清单

| 证据 | 位置 |
|---|---|
| `2c95aada` 全量取证 | `/tmp/noldus-e2e-runs/20260624-173737/`（`clarifications.json` / `sse-0..8.txt` / `analyze.txt` / `gateway-excerpt.log` / `final.png` / `api-log.json`） |
| `2c95aada` report.md | `<workspace>/user-data/outputs/report.md`（9271B，无图） |
| `a6e3775c` report.md | `<workspace>/user-data/outputs/report.md`（5158B，含 `{{img:plot_*}}`） |
| `a6e3775c` 112 png | `<workspace>/user-data/outputs/plot_{open_arm_time_ratio_bar,zone_entry_distribution,trajectory,heatmap}_s*.png` |
| lead 推理逐字 | `2c95aada` 的 `sse-8.txt`（set_viz_choice 上下文） |
| forensic panel 输出 | `/tmp/noldus-e2e-runs/20260624-173737/analyze.txt` |

`<workspace>` = `packages/agent/backend/.deer-flow/users/<uid>/threads/<tid>/user-data/`。

---

## 7. milestone 建议

本次 dogfood 让 **EPM 范式 dogfood** 从「待跑」变为「已跑·全链路健康」——应更新 [`docs/milestone/fst-e2e-7fixes-askviz-intent.md`](../milestone/fst-e2e-7fixes-askviz-intent.md) 的「遗留项：EPM/OFT/LDB/Zero Maze 其余 4 个范式 dogfood 待跑」：**EPM ✅ 完成（2026-06-24）**，剩 OFT/LDB/Zero Maze。已在末尾执行。

同时本次暴露的 `run_chart_plan` 读盘竞态，应记入 chart-maker 健壮性待办（与 [silent-drop-completed](../problems/2026-06-24-chart-maker-run-chart-plan-silent-drop-completed-handoff.md) 的封存对账洞并列，但本项是**核盘时机**问题，那项是**封存对账**问题，正交）。

---

## 8. 下一步（给下一 agent）

1. **OFT/LDB/Zero Maze dogfood**：EPM 已绿，按同套 `/noldus-insight-e2e` 流程跑剩 3 个范式，验证 chart catalog 覆盖。
2. **（可选）`run_chart_plan` 核盘时机硬化**：见 problem 记录的修复方向，TDD 加一个「进程池未 flush 时核盘」的回归测试。**需用户授权再改代码。**
3. **Gate 3 自动驱动**：若想让 e2e skill 自动也画图，可在 driver 识别到「可视化」关键词的问题时回 "A"——但需权衡是否破坏 paradigm-agnostic 原则。**当前不改**，手动回 A 已验证可用。
