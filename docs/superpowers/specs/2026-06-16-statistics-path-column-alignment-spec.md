# Spec: statistics 路径列对齐缺口 —— compute_paradigm_metrics zone 参数透传

> 日期：2026-06-16
> 类型：bug 修复（架构不对称收口）
> 前置：第三层 file→subject 桥接（`feat/ethoinsight-io-boundary-and-aggregator` commit `9d84d6f4`，已 push）。本 spec 是其揭示的**第四轴**，独立根因。
> 状态：待 review → 批准后新 worktree 实施。

---

## 0. 背景与症状

2026-06-16 EPM dogfood（thread `158187ef`，28 个真实 FewZones 文件）+ 第三层桥接修复后实测坐实：

桥接修好后 `statistics.group_summary` 能正确产出（matched 不再空），但 **EPM 专属指标（`open_arm_time_ratio` 等）在真实 FewZones 数据上全为 `None`** → `comparisons` 只含 `distance_moved`/`velocity` 这类范式无关指标，缺 EPM 核心读数（开放臂时间比、开放臂进入次数等）。data-analyst 拿到的 statistics 判读价值大打折扣。

**这不是"等行为学同事方法论"的阻塞**（此前误判）。列对齐方法论已在 PR #103/#104 完整 merge，且 **compute 路径在用**（实测 `plan_metrics.json` 每个 EPM 指标项带 `parameters_in_use: {"open_arm_zones": ["open"]}`）。缺的纯粹是 **statistics 路径从未接入这套已有机制**的工程接线。

---

## 1. 根因：两条路径架构不对称（statistics 是 compute 的"穷亲戚"）

真实 FewZones EPM 原始列名是 `open`/`closed`（用户自定义归属列），EPM 指标函数默认查 `in_zone.*open_arm` 正则 → 不命中 → 返 None。需要 HITL `column_aliases` 提供的对齐参数 `open_arm_zones=["open"]` 才能算对。

| 路径 | 列对齐怎么传 | 结果 |
|---|---|---|
| **compute**（code-executor 逐指标跑独立脚本） | resolve 把 `column_aliases` 经 `_build_zone_aliases_overrides` → `_compute_parameters_in_use` 注入每个指标项的 `parameters_in_use`；compute 脚本 `parse_parameters(args)` → `compute_open_arm_time_ratio(df, **parameters)` 解包传底层函数 | ✅ 真值 |
| **statistics**（`run_groupwise_stats` 调 dispatcher 批量算） | plan `statistics` 段只有 `{id, script, input, output, skip_reason}` **无 parameters**；dispatcher `compute_paradigm_metrics` 内部 `compute_open_arm_time_ratio(df)` **写死无参数**，签名无 zone 通道 | ❌ 全 None |

**两处架构断点**（缺口的精确位置）：

- **断点 1 — plan 生成（`catalog/resolve.py` Step 3）**：`_stats_to_plan` 生成 statistics 段时不带任何对齐参数。而 `zone_aliases_overrides` 在同一个 `resolve_metrics` 函数体 L216 已算好（给 metrics 用），statistics 段没复用它。
- **断点 2 — dispatcher（`metrics/dispatcher.py:compute_paradigm_metrics`）**：批量算入口，对每个指标硬编码 `compute_X(df)` 无参数调用，函数签名 `(parsed_data, paradigm, groups, metrics)` 根本没有 zone 参数透传通道。**这是更深的结构断点**：即使 statistics 段带上参数，dispatcher 也没法喂给底层函数。

**为什么 compute 对、statistics 错**：compute 是"一指标一脚本"，每个脚本 `**parameters` 自然解包；statistics 是 dispatcher **批量**算，这个批量入口从设计起就没有 per-metric 参数透传。这与已修的第三层 file→subject 不对称、memory 记录的 I/O 边界不对称（`feedback_2026-06-16_io_boundary_asymmetry_and_aggregator_half_built`）同属一个家族：**每次新机制（resolve 路径解析 / 聚合 / 列对齐）落地都漏接 statistics 路径**。本 spec 收口列对齐这一处。

---

## 2. 同事已备的方法论（直接复用，不另造）

PR #103/#104 已提供完整列对齐链路，本 spec 全部复用，零新方法论：

- `catalog/resolve.py:_build_zone_aliases_overrides(column_aliases, cat, overrides)` → 从 `column_aliases` 算出 `{open_arm_zones: ["open"], ...}`（多 concept 路由、anonymous zone、wrap_list 全处理）。**statistics 段直接复用 L216 已算好的 `zone_aliases_overrides`**。
- 底层指标函数**已全部支持** zone kwarg（实测）：EPM `open_arm_zones`/`closed_arm_zones`、OFT `center_zone`、zero_maze `open_zones`、ldb `light_zone`/`dark_zone` 等。
- catalog YAML `parameters` 段已声明每个指标的 zone 参数（如 epm.yaml `open_arm_zones: "由 HITL column_aliases 自动注入"`）。

**关键设计约束**：各范式 zone 参数名不统一（EPM=`open_arm_zones`、OFT=`center_zone`、zero_maze=`open_zones`）。dispatcher 透传必须按范式把对应 override 分发到各指标函数的对应 kwarg，不能假设统一参数名。

---

## 3. 修法（治本，跨四层；全部复用同事机制）

### 3.1 dispatcher 加 zone 参数透传通道（断点 2，核心）

`metrics/dispatcher.py:compute_paradigm_metrics` 签名加可选入参：

```python
def compute_paradigm_metrics(
    parsed_data: dict,
    paradigm: str,
    groups: dict[str, list[str]] | None = None,
    metrics: list[str] | None = None,
    zone_overrides: dict[str, list[str]] | None = None,  # 新增，默认 None 向后兼容
) -> dict:
```

内部按范式把 `zone_overrides` 分发到各指标函数的对应 kwarg。例如 EPM 分支：

```python
zo = zone_overrides or {}
elif paradigm == "epm":
    m["open_arm_time_ratio"] = compute_open_arm_time_ratio(df, open_arm_zones=zo.get("open_arm_zones"))
    m["open_arm_entry_ratio"] = compute_open_arm_entry_ratio(df, open_arm_zones=zo.get("open_arm_zones"), closed_arm_zones=zo.get("closed_arm_zones"))
    ...
```

**设计选择（避免脆弱的硬编码分发）**：dispatcher 对每个指标函数用 `zo.get(<kwarg>)`，None 时底层函数走原有自动检测（完全向后兼容）。zone_overrides 的 key 用底层函数的 kwarg 名（`open_arm_zones` 等），与 resolve 的 `zone_aliases_overrides` 输出 key **同源**（resolve 输出的就是 target_param 名）。

**影响面**：~20 个调用者，全部位置参数不变、新增 kwarg 默认 None → 零破坏。plot 脚本不传 zone_overrides → 行为不变（它们本就只画图不依赖精确指标值，或已有自己的路径）。

### 3.2 catalog schema + resolve 让 statistics 段带 zone 参数（断点 1）

- `catalog/schema.py:PlanStatistics` 加字段 `parameters: dict[str, list[str]] = field(default_factory=dict)`（默认空，向后兼容）。
- `catalog/resolve.py:_stats_to_plan` 加入参 `zone_overrides`，写进 `PlanStatistics.parameters`。
- `resolve.py` Step 3 两处 `_stats_to_plan` 调用传入 L216 已算好的 `zone_aliases_overrides`（同源，不重算）。

> SSOT 守则：statistics 段的 parameters 是 metrics 段同一份 `zone_aliases_overrides` 的投影，不是第二份知识源。两者从同一个 `_build_zone_aliases_overrides` 结果派生。

### 3.3 run_groupwise_stats 读参数传 dispatcher（6 范式）

6 个 `scripts/<paradigm>/run_groupwise_stats.py`：
- 读 plan statistics 段的 `parameters`（经 stats parser 新增 `--parameters-json`，与 compute 脚本 `parse_parameters` 对称复用 `_cli.parse_parameters`）。
- 传给 `compute_paradigm_metrics(..., zone_overrides=parameters)`。

> 接线对称：compute 脚本已有 `--parameters-json` + `parse_parameters`，statistics 脚本复用同一 helper，不另造解析。

### 3.4 metric_aggregation / run_metric_plan 把 statistics 段 parameters 传进子进程

`run_metric_plan` 执行 statistics step 时，把 plan statistics 段的 `parameters` 序列化成 `--parameters-json` 传给 `run_groupwise_stats` 子进程（与它已经为 compute step 传 parameters 的方式对称）。**待实施时核实**：run_metric_plan 当前怎么为 compute step 传 parameters，statistics step 照搬同一机制。

---

## 4. 红→绿验收（独立立证，与第三层桥接的测试分开）

红锚点必须是**列对齐缺口本身**，不是第三层 file→subject（那是已修的不同根因）：

1. **dispatcher 单元**（`tests/test_metrics_epm.py` 或新文件）：构造列名为 `open`/`closed` 的 EPM df + `zone_overrides={"open_arm_zones":["open"], "closed_arm_zones":["closed"]}`：
   - 红：不传 zone_overrides → `open_arm_time_ratio` 为 None。
   - 绿：传 zone_overrides → 非 None 且在 [0,1]。
2. **resolve 单元**（`tests/test_column_semantics.py` 同族）：`column_aliases={"open":"in_zone_open_arm", "closed":"in_zone_closed_arm"}` 的 EPM resolve → plan statistics 段 `parameters` 含 `open_arm_zones=["open"]`（红：当前 statistics 段无 parameters 字段）。
3. **端到端**（`tests/scripts/test_epm_scripts.py` 或第三层桥接测试文件同域）：真实/合成 FewZones 列名（`open`/`closed`）跑 `run_groupwise_stats` 全链：
   - 红：comparisons 里 EPM 专属指标缺失（None 被过滤）。
   - 绿：comparisons 含 `open_arm_time_ratio` 等 EPM 指标的组间比较。
4. **revert-to-prove-red**：注释掉 dispatcher 的 zone_overrides 分发 → 端到端测试红 → 恢复绿（证红锚点真依赖生产改动）。
5. **真实数据 smoke**（非自动化，实施者手动）：thread `158187ef` 8 文件跑 EPM stats，确认 `open_arm_time_ratio` 等从 None → 有值。

---

## 5. 回归 + 闭环纪律

- ethoinsight 全量回归（基线 863 passed）+ backend statistics 消费者（基线 33 passed）。
- dispatcher 改签名后，**20 个调用者全跑一遍**（plot 脚本不传 zone_overrides 应行为不变）。
- 裸导入两生产入口 `import app.gateway` + `make_lead_agent` exit 0（dispatcher 在 ethoinsight，不直接触 harness 导入环，但仍按铁律验证）。
- worktree 用独立 venv（editable 指 worktree 源），dispatcher 测试直接 import 即测改动（实测 io-boundary worktree 已是此形态，非假绿）。

---

## 6. 不做 / 范围外

- **不动第三层 file→subject 桥接**（已修，commit `9d84d6f4`）。
- **不改 catalog 列对齐方法论本身**（同事已备，纯复用）。
- **不碰 Issue #98 结构聚合**（自定义分区按范式聚合的语义，那才是真·等同事方法论；本 spec 只接已有的 per-file zone 列对齐）。
- **不引入新 statistics 参数概念**：只透传已有 zone overrides，不顺手加别的参数（守"根因未隔离前别叠加"纪律）。

---

## 7. 改动文件清单（预估）

```
metrics/dispatcher.py                       (compute_paradigm_metrics 加 zone_overrides 分发)
catalog/schema.py                           (PlanStatistics 加 parameters 字段)
catalog/resolve.py                          (_stats_to_plan 加 zone_overrides + Step 3 传入)
scripts/_cli.py                             (make_stats_parser 加 --parameters-json，复用 parse_parameters)
scripts/{epm,oft,ldb,zero_maze,fst,tst}/run_groupwise_stats.py  (读参数传 dispatcher)
<harness>/subagents/metric_aggregation.py 或 run_metric_plan     (statistics step 传 --parameters-json，待核实精确位置)
tests/test_metrics_epm.py / test_column_semantics.py / scripts/test_epm_scripts.py  (红绿测试)
```

---

## 8. 一句话总结

statistics 路径从未接入同事已 merge 的列对齐机制（两处架构断点：plan statistics 段无 parameters + dispatcher 无 zone 透传通道），导致真实 FewZones 数据上 EPM 专属指标在 statistics 链全 None。修法是把 metrics 路径已有的 `zone_aliases_overrides`（同源）透传到 statistics：dispatcher 加 zone 参数通道 + plan statistics 段带参数 + run_groupwise_stats 接线，纯复用零新方法论。
