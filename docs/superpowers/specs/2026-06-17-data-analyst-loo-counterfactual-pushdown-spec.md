# Spec: data-analyst 手算 leave-one-out 反事实螺旋 —— LOO 下沉到统计层预算

> 日期：2026-06-17
> 类型：bug 修复（结构病根治 — 手算螺旋家族新变体）
> 触发源：2026-06-17 本地 EPM dogfood（28 真实 FewZones，control=7/treatment=21）。用户："code 的 subagent 执行完成后，为什么 data analyst 的 subagent 运行这么慢？？？"
> 状态：待 review → 批准后新 worktree 实施。
> 关联 memory：`feedback_prep_metric_plan_stats_skip_none_counts_poisons_data_analyst_seal`（手算补偿螺旋耗尽预算）、`feedback_parameters_used_must_reflect_actual_resolution_path`（step 2.8 审计烧 12 turn）、`feedback_subagent_seal_deadlock_is_prompt_not_budget`、`feedback_single_source_of_truth`。

---

## 0. 背景与症状

2026-06-17 dogfood：code-executor 正常完成（28 subject、5 EPM 指标全算成功、statistics **非空**、数据自洽），lead 并行派 data-analyst + chart-maker。data-analyst **跑得异常慢**（前端"预计 1-2 分钟"远超）。

抓 data-analyst 的思考 trace，慢的原因一目了然：它在思考阶段**反复手算 leave-one-out（LOO）反事实**——

- 识别离群个体 Trial 3（control，open_arm_time_ratio=1.0、total_entry_count=1）、Trial 12（treatment，同型）；
- 手动列出 control 6 个剩余 subject 的值、求和、求方差、开方算 SD → 得"排除 Trial 3 后 control 均值 0.253→0.129"；
- 手动列出 treatment 20 个剩余 subject 的值、求和 → 得"排除 Trial 12 后 0.096→0.051"；
- **同一组 LOO 数字至少重算 4 遍**（trace 里 "Fast-Fail Check" 段落重复出现 4 次，每次都把 28 个 subject 的 per_subject 值重新誊一遍手算）。

它没卡死、没触发 loop-detection、数据也没矛盾——它是**老老实实在做 prompt 要求的手算，而且因为自己不确定手算对不对，反复验算**。这是纯粹的"用 LLM 的推理预算做小学算术"，token 烧在重复验算上。

**这不是性能问题，是结构病**：prompt 把"必须做反事实"压给模型，却只给了"手算"这一条路。

---

## 1. 根因：prompt 要求手算 LOO，但"禁手算"只覆盖了组间检验

`packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py`：

**要求手算 LOO**（line 174-177，step 2.7 b）：
```
b. **按受试者 + 反事实**（核心价值，必须做）：
   - 从 per_subject 识别偏离组均值 ≥ 1.5 SD 或偏离组中位数 ≥ 2 倍的受试者
   - 对每个离群个体计算 leave-one-out 统计（排除后组 mean/std 变化）
   - 每个发现写入 outlier_findings 数组
```

**禁手算，但只针对组间检验**（line 85-89，statistics 为空时的降级路径）：
```
走描述性判读路径：给出各组均值/中位数/SD/离群观察（仅描述，不做组间检验），
... 立即调用 seal ... 手算既不可靠又会耗尽推理预算。
```

→ line 89 的"手算既不可靠又会耗尽预算"是在 **statistics 为空** 的语境里说的，针对的是"别手算组间 p 值"。它**没覆盖 LOO 反事实**——LOO 在 statistics **非空**时仍是 step 2.7 b "必须做"的核心动作。于是：

- statistics 非空（正常路径）→ 不走 line 85-89 降级 → 进 step 2.7 b → **被要求手算 LOO** → 螺旋。

**结构断点**：outlier 识别（≥1.5 SD / ≥2× median）和 LOO 反事实（排除某 subject 后重算组 mean/std）**都是纯确定性数值计算**，输入（per_subject 各组标量值）在 code-executor 阶段就已全部产出，却被推到 data-analyst 的**自然语言推理**里手算。这与 memory 记录的"聚合器恒等手算螺旋"、"step 2.8 审计烧 turn"同属一个家族：**确定性计算被放在了 LLM 思考里，而不是代码里**。

---

## 2. 为什么之前没这么明显 / 这次才暴露

- 之前多次 dogfood data-analyst 失败是 **statistics 为空**（走 line 85-89 描述性 partial 路径，那条路明确"仅描述、不手算"，所以快）。
- 这次是**第一次走到 statistics 非空的正常路径 + 真实 28 subject**：subject 多（28 个值手算）、有两个 1.0 极端离群（强烈触发 step 2.7 b 的"必须做 LOO"）→ 手算量陡增 → 螺旋第一次清晰暴露。
- 也就是说：**正常路径的 LOO 手算一直存在，只是 subject 少时不显眼**。这次是 forcing function。

---

## 3. 修法（治本：LOO + outlier 识别下沉到统计层，data-analyst 只读不算）

与"聚合在 ethoinsight 库里算、不在 LLM 里算"同一处方（守 memory `feedback_2026-06-16_io_boundary_asymmetry_and_aggregator_half_built` 的工程纪律：确定性计算定为代码不变量，不交给 LLM）。

### 3.1 在 `statistics.compare_groups` 里产出 per-metric outlier + LOO（核心）

**位置**：`packages/ethoinsight/ethoinsight/statistics.py:compare_groups`（line 358-518）。

该函数已经在 line 415-421 对每个 metric 收集了**每组的 per-subject 值列表** `group_values[grp_name] = [...]`。LOO 和 outlier 识别所需的全部输入就在这里，零新数据管道。

新增一个纯函数 `compute_outlier_diagnostics(group_values: dict[str, list[float]]) -> list[dict]`（建议放 `statistics.py` 或 `metrics/_common.py`，无既有 helper 可复用，确认是新函数——`_common.py` 只有 distance/velocity/zone，无 outlier/LOO），对每个 metric：

```python
def compute_outlier_diagnostics(group_values: dict[str, list[float]]) -> list[dict]:
    """对每组识别离群 subject 并算 leave-one-out 反事实。纯确定性，无 LLM。

    判据与 prompt step 2.7 b 完全一致：
      - 偏离组均值 ≥ 1.5 SD，或偏离组中位数 ≥ 2×（取并集）
    每个离群 subject 产出一条：
      {group, subject_index, value, deviation_sd, deviation_median_ratio,
       group_mean, group_std, loo_mean, loo_std,  # 排除该 subject 后
       counterfactual}  # 预格式化字符串，data-analyst 可直接引用
    """
```

**字段对齐 prompt 的 outlier_findings schema**（line 339-344）：`subject`/`metric`/`value`/`deviation`/`counterfactual`。`counterfactual` 直接预格式化成 prompt 期望的串（如 `"control open_arm_time_ratio mean 0.253 → 0.129 if subject 3 excluded"`），data-analyst **原样引用，不重算**。

把结果挂到 `comparisons[metric_name]` 每条里（新增 `outlier_diagnostics` 键），或挂到一个并列的 `statistics["outlier_diagnostics"][metric_name]`。**推荐后者**（与 comparisons 并列，避免改 comparisons 既有 schema 影响 charts.py 的 significance 消费——见风险 6.2）。

> ⚠️ subject 标识：`group_values` 当前只有标量值列表，没带 subject 名/index。需要在收集时同时保留 subject 标识（从 `group_summary[grp][metric]` 里取，确认 group_summary 是否含 subject 顺序/名——实施第一步先核实 group_summary 结构，若只有 values 无 subject 名，则用组内 index 兜底并在 counterfactual 里标 `subject #i`）。

### 3.2 statistics.json 落盘带上 outlier_diagnostics

`compare_groups` 的 caller（`run_metric_plan` Step8 / `run_groupwise_stats` / `_cli`）把 3.1 的产出一并写进 `statistics.json`。data-analyst 读 statistics.json 时直接拿到 outlier + LOO，**不再从 per_subject 手算**。

> 实施需确认 statistics 的写盘路径（`run_metric_plan` Step8 aggregate / `scripts/_cli.py`），把新字段透传到落盘 JSON。这一步纯接线，复用 3.1 的纯函数产出。

### 3.3 data-analyst prompt：LOO 改"读不算" + 禁手算扩到 LOO

`data_analyst.py` 三处：

**(a) line 174-177 step 2.7 b 改为"读 statistics.outlier_diagnostics"**：
```
b. **按受试者 + 反事实**（核心价值）：
   - 直接读 statistics.json 的 outlier_diagnostics（统计层已按 ≥1.5 SD / ≥2× median
     识别离群 subject 并预算好 leave-one-out 反事实）。
   - 把每条 outlier_diagnostics 映射进 outlier_findings 数组（counterfactual 字段
     原样引用，不要重算）。
   - **严禁手算 mean/std/LOO**：所有数值已由统计层确定性产出，你的职责是【解读】
     这些离群对结论稳健性的影响，不是【重算】它们。
```

**(b) line 89 的禁手算从"仅组间检验"扩到"包括 LOO/任何组级数值"**：
```
... 手算组间检验、leave-one-out、组 mean/std 既不可靠又会耗尽推理预算——
这些都已由统计层（statistics.json）产出，你只读不算。
```

**(c) 兜底**：若 statistics.json 缺 outlier_diagnostics（老数据/降级），改为"**只定性指出哪些 subject 看起来离群 + 方向，不给精确 LOO 数字**"（轻量定性，不退回手算精确值）。

---

## 4. 测试（TDD，红→绿）

### 4.1 ethoinsight 层（纯函数，`packages/ethoinsight/tests/`）

- `test_compute_outlier_diagnostics_flags_high_sd`：构造一组含 1 个 ≥1.5 SD 离群 + 正常值 → 断言该 subject 被标、`loo_mean`/`loo_std` 等于手工核验值、`counterfactual` 串格式正确。
- `test_compute_outlier_diagnostics_median_ratio_rule`：构造 ≥2× median 但 <1.5 SD 的离群（并集判据的另一支）→ 断言被标。
- `test_compute_outlier_diagnostics_no_outlier`：均匀分布 → 空列表。
- `test_compute_outlier_diagnostics_loo_matches_manual`：用本次 dogfood 真实值（control [0.099,0.084,1.0,0.258,0.133,0.138,0.059]）→ 断言排除 1.0 那个后 mean=0.1285（±1e-3），即 data-analyst trace 里手算的那个数，**证明库算 == 手算结果**（等价性，避免行为漂移）。
- `test_compare_groups_attaches_outlier_diagnostics`：`compare_groups` 输出含 `outlier_diagnostics`，且不破坏既有 `comparisons` schema（charts.py 消费不受影响）。

### 4.2 接线层

- `test_statistics_json_includes_outlier_diagnostics`：跑统计落盘路径（或其纯函数变体）→ statistics.json 含 outlier_diagnostics 字段。

### 4.3 红→绿

拆 red/green commit：先加 4.1 测试（`compute_outlier_diagnostics` 尚不存在 → import 红）→ 实施 → 绿。`test_compute_outlier_diagnostics_loo_matches_manual` 用真实 dogfood 数手工核验，咬住"库算结果 == 之前手算结果"。

> prompt 改动（3.3）无单测可咬（自然语言），靠 dogfood 复跑验证 data-analyst 不再手算螺旋（见 §7）。这是纵深防御的辅助层，主防线是 3.1/3.2 让手算"无必要"。

---

## 5. 为什么这样分层（设计依据）

- **主防线 = 让手算无必要**（3.1/3.2 统计层产出 LOO）：data-analyst 读现成数字，结构上无可手算。这比"prompt 求它别手算"可靠——和 seal-gate 的"gate 是主防线非兜底"同一哲学（结构性消除 > 概率性劝阻）。
- **辅助 = prompt 收紧**（3.3）：禁手算 + 改读路径，让模型少绕路、省 turn。但不依赖它兜底。
- **SSOT 守恒**：outlier/LOO 的 SSOT 是 statistics 层（确定性计算的唯一权威），data-analyst 是消费者不是第二计算源。彻底消除"两处算同一个数可能不一致"（手算 vs 库算漂移）。

---

## 6. 风险与边界

1. **group_summary 是否带 subject 标识**（最先核实）：3.1 需要给每个离群 subject 一个可引用的标识。先核实 `group_summary[grp][metric]` 是否含 subject 名/顺序；没有则用组内 index 兜底，counterfactual 写 `subject #i`，并在实施 commit 注明该限制（subject 名映射是 Issue #98 列对齐家族的另一轴，本 spec 不引入）。
2. **不改 comparisons schema**：outlier_diagnostics 挂并列键，不塞进 comparisons 每条，避免 charts.py 的 significance 消费 `comparisons[metric]` 时被新字段干扰。4.1 末条测试咬住。
3. **判据与 prompt 完全一致**（≥1.5 SD ∪ ≥2× median）：库函数判据必须和 prompt step 2.7 b 字面一致，否则 data-analyst 读到的离群集与它"以为该有"的不符会重新手算。判据是这次下沉的 SSOT。
4. **空组/单值组**：某组 <2 值时 LOO 无意义，跳过该组（compare_groups line 423-425 已有 `len>=2` 门，复用同门）。
5. **当前正在跑的 run 不受影响**：用户选"让它跑完"，seal-gate（PR#142）会兜底强制 seal。本 spec 修的是下一轮起的结构。
6. **不碰 statistics 检验逻辑本身**（compare_two_groups / 正态/方差检验）：只在 compare_groups 末尾**附加** outlier_diagnostics，不动既有 p 值/effect size 计算路径。正交。

---

## 7. 实施清单（给下一个 agent）

1. 读本 spec + memory `feedback_2026-06-16_io_boundary_asymmetry_and_aggregator_half_built`（同处方）+ `feedback_single_source_of_truth`。
2. 新 worktree（基线 dev `0b2be473` 或更新）。
3. **第一步核实 §6.1**：`group_summary[grp][metric]` 结构里有没有 subject 标识。决定 counterfactual 用真名还是 index。
4. 先写 4.1 必红测试（含 `_loo_matches_manual` 用真实 dogfood 数）→ 跑红。
5. 实施 3.1（`compute_outlier_diagnostics` 纯函数）→ 3.2（落盘接线）→ 3.3（prompt 三处）。
6. 跑绿 + ethoinsight `pytest tests/`。改了 data_analyst.py（subagent prompt，非 harness 核心 import）一般不触发导入环，但稳妥起见仍裸导入 `from deerflow.agents import make_lead_agent` 验证 exit 0。
7. push，提醒用户建 PR。
8. 合并后用本次 dogfood 同数据复跑（`/home/wangqiuyang/DemoData/real_data/Raw data-EPM-Xuhui-28`），确认 data-analyst：① 读 outlier_diagnostics 不再手算螺旋 ② 耗时回落（无反复验算）③ outlier_findings 的 counterfactual 数字与统计层一致 ④ 正常一次 seal。观测 gateway.log 看 data-analyst 思考轮数是否大幅下降（手算螺旋消失探针）。
