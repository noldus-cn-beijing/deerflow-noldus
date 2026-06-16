# Dogfood 失败核实文档（给 Fable 核实）

> thread `38be2753`（EPM，28 文件 4 组）。请核实：根因判断是否成立、修复方案是否正确。
> 完整 spec：`docs/superpowers/specs/2026-06-12-seal-list-zone-params-and-excel-calamine-spec.md`

---

## 一、dogfood 失败的原因

**现象**：code-executor 把 **140 个指标全算完、`validate_catalog` 通过**，前端却显示「子任务失败」。lead 反复重派 code-executor，同一 thread 内**至少 3 次**重跑（18:05 / 18:15 / 更早），每次都失败。

**真根因**（实证，gateway.log `trace=087f87a7` L956-1225，非推测）：

1. subagent 漏调 `seal` 工具 → harness 的 auto-seal 兜底**正常触发了**（这步没坏）。
2. auto-seal 从磁盘上的 `m_*.json` 重建 `CodeExecutorHandoff` 时，Pydantic 抛 **84 个 ValidationError**：

```
metrics_summary.XX.open_arm_entry_count.parameters_used.open_arm_zones.str
  Input should be a valid string [input_value=['open'], input_type=list]
```

3. seal 失败 → 整个 run 被判 FAILED。

**为什么是 84 个**：`MetricStat.parameters_used` 的类型是**标量** `dict[str, float|int|str|None]`（[handoff_schemas.py:109](../packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py)）。但 EPM 的 zone-aware 指标如实产出 **list 值**：`open_arm_zones=['open']`、`closed_arm_zones=['closed']`。每个 list 值对 union 的 3 个标量分支（float/int/str）各报 1 个错，28 个 list 参数 × 3 ≈ 84。

**这是「过窄 schema 拒绝如实完整数据」**：schema 是给旧标量参数（`velocity_threshold=30.0`）设计的，**列语义对齐工作引入 list 型 zone 参数后没同步更新 schema**。数据是对的，schema 装不下。

**一个关键陷阱**：日志里那句 WARNING「LLM finished but forgot to call the seal tool」（忘了调 seal 工具）是**误导**——真因是它**上方**的 ValidationError。如果只看 WARNING 会误判成 prompt 层问题。

**不是 EPM 独有**（逐范式核实）：

| 范式 | zone 参数 | 会炸吗 |
|------|----------|-------|
| EPM | `open_arm_zones` / `closed_arm_zones` → `list[str]` | ✅ 本次实证 |
| Zero Maze | `open_zones` / `closed_zones` → `list[str]`（同款透传）| ✅ 会同样炸 |
| OFT / LDB | 单列 `str` | ❌ |
| FST / TST | float | ❌ |

→ 是**所有多列 zone 聚合范式的通病**，今天 EPM + Zero Maze。

---

## 二、速度慢的原因

慢分两块，叠加在一起：

**1. 被上面的 bug 放大的重试**：每次 seal 失败 lead 都重派 code-executor 重算 140 指标。失败 ≥3 次 = 浪费了好几个完整重算周期。**修了 bug 这部分直接消失。**

**2. Excel 解析引擎慢**（实测，real file 9301×12，742KB，best of 3）：

| 操作 | 现状 openpyxl | calamine | 加速 |
|------|--------------|----------|------|
| `pd.read_excel` 全量 | 0.651s | **0.081s** | **8.1×** |
| 单次 `inspect_uploaded_file` | 2.03s → 28 文件 ≈ **57s** | 预估 ≈8s | — |
| code-executor 单 compute 读盘 | 1.35s ×140 次 ≈ 189s | ≈0.1s ×140 ≈ 14s | 省 ~170s |

- pandas 对 `.xlsx` 默认用 **openpyxl**（纯 Python 解 XML，慢）。换 **calamine**（Rust）快 8×，输出**逐格一致**（已验证 `astype(str).equals`）。
- 另：每个 compute 是独立 `python -m ...` 进程，有 ~0.83s 冷启 import 开销，140 次叠加。**这部分本次不修**（属编排层，记 TODO）。

> 注：13 分钟里**真正计算只 ~2 分钟**，大头是 LLM 逐 token 吐 28 行 bash 脚本 + thinking + 重试。换引擎治读盘那块，重试由修 bug 消除。

---

## 三、修复方案

### A. 修 seal bug（红线）

**A1 — 抽共享类型别名，两个字段共用**（治本：消除「同一值域两份声明各自漂移」）：

```python
# handoff_schemas.py 顶部
ParamValue = float | int | str | list[str] | None

# :109  parameters_used: dict[str, ParamValue]              # 原 dict[str, float|int|str|None]
# :239  used_value: ParamValue                              # 原 float|int|str
```

- `used_value`（data-analyst 的审计字段）也放宽：prompt 指示 LLM 把 `parameters_used[参数名]` 真实值逐字拷进去，list 参数落地后它就是确定触发的潜伏 bug（下一站 data-analyst seal 会同样炸）。不是投机。
- Pydantic v2 smart-union 中 `str` 与 `list[str]` **互不坍缩**（`"open"` 精确匹配 str、`["open"]` 精确匹配 list[str]，v2 绝不 list↔str 互转）。**禁设** `union_mode=left_to_right`。不放宽到 `list[int]`/`bool`。
- `_normalize_audit_finding` 的 `None→""` 映射**保守保留 + 加注释**（删它=改未坏路径行为，违项目纪律）。

**A2 — 红测试**：单元（`MetricStat(parameters_used={"open_arm_zones":["open"]})` 先红后绿）+ 端到端（仿真实 auto-seal 路径写 `m_*.json` 含 list 参数，复现 84-error 再转绿）。

### B. Excel 换 calamine（性能，同 PR）

- 加**硬依赖** `python-calamine` 进 `pyproject.toml` + pin 版本 + `uv sync`。
- 单一常量 `EXCEL_ENGINE = "calamine"`，三处 `pd.read_excel` 显式传 `engine=`。
- **无运行时 fallback**（关键，已采纳 Fable 上轮论证：`try calamine except openpyxl` 对「calamine 成功但值不同」永不触发，只在异常时触发，纯负收益 + 掩盖 engine bug + 环境相关行为）。失败响亮 raise。
- openpyxl 留作 test/dev 依赖（calamine 只读不能写 fixture；公共 skill `analyze.py` 也用），**不删**。
- 测试：语料级 `pd.testing.assert_frame_equal(check_dtype=True)` + 合成边界探针（日期/int-float/空 cell/error cell）。
  - 已核实 EV19 trajectory 数据区**全 float64/int64、零 datetime 列** → Fable 上轮最担心的「日期轴」对真实数据不咬，但仍按语料差分防未来范式。

### 不改动（已核实下游容忍 list）
- `executor.py:493` auto-seal 逐字透传 params，无标量假设。
- `validate_catalog` 从不读 `parameters_used`（故 trace 里它通过）。
- ChartMaker/DataAnalyst/ReportWriter handoff 不内嵌 `MetricStat`。

---

## 四、想请 Fable 核实的点

1. **A1 的 `ParamValue` 别名方案**是否正确收口了 bug 类？两个字段共用别名 vs 改两处裸 union，前者更对吗？
2. **`used_value` 放宽**是否过度（它还没在生产炸过）？还是确属同一 bug 的下一跳、该一起修？
3. **B 的「无 fallback」**是否正确落实了你上轮的论证？有没有遗漏的引擎切换可观测性手段？
4. **calamine 等价性测试**用语料级 `assert_frame_equal` + 合成探针，够不够？还有哪个分歧轴该测？
5. 有没有**我判断错或漏掉**的地方？
