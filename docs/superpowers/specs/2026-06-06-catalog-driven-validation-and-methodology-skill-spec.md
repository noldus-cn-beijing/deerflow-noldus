# Spec: Catalog-Driven 指标验证（两层分工）+ 数据处理方法论下沉 Skill

**日期**：2026-06-06
**状态**：可执行 spec，供 agent 在新 worktree 中实施
**来源**：[数据处理方法论设计 §6/§7](../../design/2026-06-06-data-processing-methodology-design.md) + 2026-06-06 调研 + grill 结论
**关联 Issue**：[#98](https://github.com/noldus-cn-beijing/noldus-insight/issues/98)（Layer 3 聚合，**本 spec 不涉及**，等行为学专家）

---

## 0. 前提与边界（执行前必读）

### 前提
- **① S1-S4（`feature/s1-s4-implementation`）已由用户手动合入 `dev`**。本 spec 假设 `dev` 上已存在：
  - `packages/ethoinsight/ethoinsight/validate.py`（74 行，suffix 匹配版）
  - `packages/ethoinsight/tests/test_validate.py`（181 行）
  - `packages/ethoinsight/ethoinsight/scripts/_cli.py` 的 `emit_result` 已注入 `validate_metrics`
  - code-executor SKILL 已抓 `VALIDATION_ERROR` 行汇总进 `data_quality_warnings`（`code=METRIC_VALIDATION`）
- **开工第一步**：`git log dev --oneline | grep -iE 'S1-S4|validate'` 确认 ① 已在 dev。**若不在，停止并报告用户**——本 spec 的所有改动以 dev 上的 validate.py 为基线。

### 边界（绝对不做）
- ❌ **不实现任何 Layer 3 / zone 聚合逻辑**。聚合语义（OR vs sum）取决于行为学专家对"EV19 zone 是否空间重叠"的回答（Issue #98），未回答前任何聚合实现都是猜测。
- ❌ **不碰 catalog YAML 的 metric 定义**（SSOT，只读）。本 spec 只**读** `output_unit`，不改任何 metric。
- ❌ **不改 compute 脚本的计算逻辑**。只在验证层做文章。

### 工作方式
- 在新 worktree 从 `dev` 拉分支：`feature/catalog-driven-validation`
- **TDD 强制**（CLAUDE.md）：每个改动先写红锚点测试，再实现。
- **改共享 validator 必跑全量**（[教训](../../../.claude/...)：PR #66 只跑新测试致 3 个共享 fixture 旧测试静默 fail）。改 `validate.py` / `_cli.py` 后，跑 `cd packages/ethoinsight && pytest tests/` **全量**，并 `grep -rn 'validate_metrics(' packages/` 检查所有调用方。
- commit message 用中文。

---

## 1. 架构决策：两层分工（已与用户确认）

catalog-driven 验证需要 `paradigm` context 才能 `load_catalog(paradigm)` 查 `output_unit`。但 **compute 脚本进程内（`emit_result`）拿不到 paradigm**——它只有 `--input/--output/--parameters-json`。`paradigm` 只在 **code-executor 层**可得（`plan_metrics.json` 含 `paradigm` 字段）。

因此验证分两层，职责正交：

| 层 | 位置 | 验证什么 | 需要 catalog? |
|----|------|---------|--------------|
| **L-A 进程内安全网** | `_cli.py:emit_result`（保留现状，仅收窄） | **NaN / Inf**（name-agnostic 确定性安全网） | 否 |
| **L-B 语义范围校验** | **新增** code-executor 收集结果后调用的 validate CLI | **output_unit 范围**（ratio→[0,1] / count→≥0整数 / 物理单位→≥0+上限 / 复合_stats / 孤儿判 UNKNOWN） | 是（按 metric_id 查 output_unit） |

**关键**：L-A 不再做 suffix 范围校验（`_ratio`/`_count` 那些）——那些**全部移到 L-B 用 output_unit 驱动**。L-A 只留 NaN/Inf。理由：suffix 匹配是 L-A 在"拿不到 catalog"约束下的妥协，L-B 有了 catalog 就不需要妥协了；两处都做 ratio 校验会重复且可能不一致（违反 SSOT）。

---

## 2. Task A — 收窄 L-A（emit_result 只留 NaN/Inf）

### A.1 红锚点先行
在 `test_validate.py` 新增（或改造）测试，锚定**新契约**：
- `validate_metrics({"open_arm_time_ratio": 1.5})` → **空列表**（ratio 范围校验已移走，L-A 不再管）
- `validate_metrics({"x": float("nan")})` → 仍返回 NaN violation（L-A 保留）
- `validate_metrics({"x": float("inf")})` → 仍返回 Inf violation
- 现有的 ratio/pct/suffix 测试：**改为预期空列表**或迁移到 Task B 的新测试文件（见 B.5）

### A.2 实现
改 `packages/ethoinsight/ethoinsight/validate.py`：
- `validate_metrics(metrics: dict)` **只保留 NaN/Inf 检查**，删除所有 suffix 范围逻辑（`_NON_NEGATIVE_SUFFIXES`、`_ratio`/`_pct` 范围块）。
- 函数 docstring 更新：说明这是 L-A 安全网，范围校验见 L-B（catalog-driven）。
- `_cli.py:emit_result` **不改签名**（仍 `emit_result(payload)`），它继续调收窄后的 `validate_metrics`，只会产 NaN/Inf 的 VALIDATION_ERROR。

### A.3 验证
- `pytest tests/test_validate.py` 全绿
- `grep -rn 'validate_metrics(' packages/` 确认无其它调用方依赖被删的 suffix 行为

---

## 3. Task B — 新增 L-B（catalog-driven 范围校验 CLI）

### B.1 新增模块 `packages/ethoinsight/ethoinsight/validate_catalog.py`

核心函数：
```python
def validate_metrics_against_catalog(
    results: dict[str, Any],      # {metric_id: value}, value 可能是标量或 dict(复合 _stats)
    paradigm: str,
    catalog_dir: str | Path | None = None,
) -> list[dict[str, str]]:
    """按 catalog output_unit 校验指标范围。

    Returns: 违规列表，每条 {"metric", "issue", "value"}（与 L-A 同 schema，下游统一处理）。
    """
```

逻辑：
1. `cat = load_catalog(paradigm, catalog_dir)` —— 复用现成 loader（`catalog/loader.py:71`）。
2. 建 `metric_id → output_unit` 映射：遍历 `cat.default_metrics + cat.optional_metrics`，每个 `MetricEntry` 有 `.id` 和 `.output_unit`（`schema.py:62-66`）。
3. 对每个 `(metric_id, value)`：
   - **孤儿指标**（metric_id 不在 catalog）→ 产 violation `{"metric": id, "issue": "catalog_unknown", "value": str(value)}`。**不静默放过**（缺口 2）。
   - **复合 _stats**（`isinstance(value, dict)`）→ 见 B.2。
   - **标量** → 按 output_unit 查规则表（B.3）校验。
   - **None / 非数值** → 跳过（不是错误，是缺数据）。

### B.2 复合 `_stats` 指标的处理（缺口 3）
已核实 `body_elongation_stats` / `turn_angle_stats` / `head_direction_stats` / `acceleration_stats` / `velocity_stats` 的 `value` 是 `dict`（如 `{mean, std, min, max, ...}`，见 `_common.py:592 compute_body_elongation_stats -> dict`）。

处理规则：
- 对复合值，**对其中的数值字段逐个套 output_unit 范围**（如 output_unit=radians 的 turn_angle_stats，其 mean/min/max 都应 ≥0 且 ≤上限）。
- **但 std 字段例外**：标准差恒 ≥0，不套 output_unit 的范围上下限（一个 ratio 的 std 不必 ≤1）。spec 实现时对字段名做白名单：`{mean, median, min, max, p25, p75}` 套范围，`{std, sem, var, n, count}` 只验 ≥0/非 NaN。
- **执行 agent 注意**：实施前先 `python -c "from ethoinsight.metrics._common import compute_body_elongation_stats"` 跑一个真实样例（或读 `_common.py:592` 起的函数体）确认 dict 的**实际字段名**，不要照搬本 spec 假设的字段名。字段名以代码为准。

### B.3 output_unit → 规则表（缺口 1 + 4）
catalog 实际出现 6 种 output_unit（已核实，100% 覆盖）：`ratio` / `seconds` / `count` / `radians` / `cm` / `mm_s2`。

| output_unit | 下限 | 上限 | 类型 |
|-------------|------|------|------|
| ratio | 0 | 1 | float |
| count | 0 | 无 | int（非整数也报） |
| seconds | 0 | `plausible_max`（见 B.4，缺省无上限） | float |
| cm | 0 | `plausible_max` | float |
| radians | 0 | 见 B.4（**专家问题**，先只验 ≥0） | float |
| mm_s2 | 0 | `plausible_max` | float |

- **枚举强制**（缺口 1）：规则表用 dict 实现。若遇到**不在这 6 种内的 output_unit**，产 violation `{"issue": "unknown_output_unit", ...}`——新单位未登记即报错，不静默。这等价于"loader 加枚举校验"的运行时兜底。
- **radians 上限是开放问题**：[-π,π] vs [0,2π] vs 仅非负，取决于 turn_angle 的定义（带符号还是绝对值）。**本 spec 先只验 ≥0**，在 B.1 模块顶部 TODO 注释标注"radians 上限待确认（关联指标定义）"，不猜。

### B.4 物理单位上限 `plausible_max`（缺口 4，可选增强）
- **本 spec 不强制实现上限**，因为 plausible_max 是领域知识（一只老鼠 5 分钟最多移动多少 cm？），属行为学专家范畴。
- **预留机制**：在规则表里给 seconds/cm/mm_s2 留 `plausible_max=None` 字段位。若未来 catalog 的 MetricEntry 增加可选 `plausible_max` 字段，验证器读它加上限。**当前 catalog 无此字段，故现在等价于"仅 ≥0"**。
- **不要**为了上限去 catalog YAML 加字段（那是改 SSOT，超出本 spec 边界）。只留代码侧的读取位。

### B.5 测试 `test_validate_catalog.py`
新建测试文件，覆盖：
- ratio 越界（>1, <0）→ violation；边界 0.0/1.0 → pass
- count 负数 / 非整数 → violation
- 物理单位（cm/seconds/radians/mm_s2）负数 → violation；正常值 → pass
- **孤儿指标** `velocity_stats`/`thigmotaxis_index`/`distance_moved` → `catalog_unknown` violation
- **复合 _stats** dict 值：mean 越界 → violation；std 为正 → pass
- **未知 output_unit**（构造一个假 catalog 或 mock）→ `unknown_output_unit` violation
- NaN/Inf 在复合 dict 字段里 → 也被抓

### B.6 CLI 入口
在 `validate_catalog.py` 加 `main(argv)`，支持 code-executor 调用：
```
python -m ethoinsight.validate_catalog --plan <plan_metrics.json> --results-dir <dir>
```
- 读 plan 的 `paradigm` + 各 metric 的 output JSON（`{metric, value}`）
- 输出 `VALIDATION_ERROR: <metric>: <issue> (value=<value>)` 行到 stdout（**与 L-A 同格式**，这样 code-executor 现有的抓取逻辑零改动就能复用）
- 退出码：有 violation 也 exit 0（informational，不阻断；下游 data-analyst 决定怎么处理）

---

## 4. Task C — code-executor 接入 L-B

### C.1 改 code-executor SKILL（`subagents/builtins/code_executor.py` 的 prompt 文本）
在"跑完所有 compute 脚本之后、封存 handoff 之前"加一步：
- 调 `python -m ethoinsight.validate_catalog --plan /mnt/user-data/workspace/plan_metrics.json --results-dir <metric 输出目录>`
- 它产出的 `VALIDATION_ERROR` 行**按现有逻辑**（SKILL 已有，见 `code_executor.py:45-50`）汇总进 `data_quality_warnings`（`code=METRIC_VALIDATION`）。
- **复用现有抓取逻辑**：因为 L-B 的输出行格式与 L-A 完全一致，code-executor 不需要新的解析分支，只是多一个产 VALIDATION_ERROR 行的来源。

### C.2 必须扩 guardrail 白名单（已核实，确切坐标）
**code-executor 的 bash 受 `ScriptInvocationOnlyProvider` 管控**（`subagents/executor.py:639` 把它挂在 code-executor + chart-maker 上）。其白名单正则在 `guardrails/script_invocation_only_provider.py:52-53`：
```python
_ALLOWED_PYTHON_PATTERN = re.compile(
    r"^\s*python\s+-m\s+ethoinsight\.(scripts\.\w+\.\w+|catalog\.resolve|parse\.dump_headers)(\s|$)"
```
**`ethoinsight.validate_catalog` 不在内 → code-executor 跑它会被 deny**（这正是 [chart-maker dump_headers 教训]同构：新 CLI 不进白名单，落地即跑不通，潜伏到 dogfood 才暴露）。

**必做**：
- 把正则精准扩成 `(scripts\.\w+\.\w+|catalog\.resolve|parse\.dump_headers|validate_catalog)`——**只加这一项，不放开 `ethoinsight\.\*`**。
- 同步更新该 provider 顶部注释（line 42-49 的"Allowed invocations"清单，加第 4 项 validate_catalog 说明：只读 results + plan，写 stdout，安全）。
- **加 red 锚点测**（`tests/test_script_invocation_only_provider.py` 已存在）：断言 `python -m ethoinsight.validate_catalog ...` 被 allow，且一个 lookalike（如 `ethoinsight.validate_catalogX` 或 `ethoinsight.validate`）仍被 deny。
- deny 消息（line 88 起的中文清单）若枚举了允许的调用，也加上 validate_catalog。

### C.3 测试
- 后端测试：构造一个含越界 metric 的 plan + results，跑 code-executor 流程，断言 `data_quality_warnings` 含 `METRIC_VALIDATION` 条目。
- 若 code-executor 有现成的集成测试 fixture，复用它加一个越界 case。

---

## 5. Task D — 数据处理方法论下沉成 Skill（§7 program.md 非对称的修复）

**问题**（grill 结论）：方法论文档在 `docs/design/` 下，**agent 运行时读不到**。AutoResearch 的 program.md 力量全在"agent 实时读它"。要兑现这个价值，三层列处理框架 + 工具分工必须下沉成 agent 主动 read 的 skill references。

### D.1 落点
挂到 **`ethoinsight-column-confirmation`** skill（它已负责 Sprint 1 列对齐，是三层框架的天然宿主）：
- 新建 `packages/agent/skills/custom/ethoinsight-column-confirmation/references/column-processing-methodology.md`
- （该 skill 的 `references/` 目录当前为空，需创建）

### D.2 内容（**只下沉已确定的部分，不含聚合规则**）
- **三层框架**：Layer 1 固定列（白名单匹配，0 HITL）/ Layer 2 自定义分析区 1:1 映射（Sprint 1 已实现）/ Layer 3 子区聚合（**标注"聚合语义待 Issue #98 专家确认，本 skill 不含聚合规则"**）。
- **工具分工**：identify（模板识别，列名截断前20不可用）→ inspect（完整列名+几何证据，须带 paradigm）。**明确禁止 `bash head -1`**（UTF-16/分号/标题行数，三处全错）。
- **列名来源链**：完整列名只来自 `inspect_uploaded_file` 或 `python -m ethoinsight.parse.dump_headers`。
- **不内嵌任何结构化知识**（SSOT 铁律）：固定列清单、逻辑区清单都**指向 catalog / Issue #98**，不在 skill 里抄一份。

### D.3 SKILL.md 加 read 指引
在 `ethoinsight-column-confirmation/SKILL.md` 加一行：处理用户自定义列前，先 `read_file references/column-processing-methodology.md` 了解三层框架与工具分工。

### D.4 注意
- **不重复 catalog 的内容**（违反 [[SSOT]]）。方法论是"决策框架"，不是"知识清单"。
- Layer 3 部分**只写"如何检测 + 如何 HITL 复述"骨架**，聚合执行明确留白指向 #98。

---

## 6. 交付物清单

| 文件 | 动作 | Task |
|------|------|------|
| `ethoinsight/validate.py` | 收窄为只 NaN/Inf | A |
| `tests/test_validate.py` | 改锚点：suffix 范围测试迁出/改空 | A |
| `ethoinsight/validate_catalog.py` | **新建** catalog-driven 范围校验 + CLI | B |
| `tests/test_validate_catalog.py` | **新建** 覆盖 6 单位+孤儿+复合+未知单位 | B |
| `subagents/builtins/code_executor.py` | prompt 加调 validate_catalog 一步 | C |
| `guardrails/script_invocation_only_provider.py:53` | 正则扩 `validate_catalog`（精准一项，不放开 `ethoinsight.*`）+ 注释 + deny 消息 | C |
| `tests/test_script_invocation_only_provider.py` | red 锚点：validate_catalog allow / lookalike deny | C |
| 后端 code-executor 测试 | 加越界→METRIC_VALIDATION warning 集成测试 | C |
| `ethoinsight-column-confirmation/references/column-processing-methodology.md` | **新建** 三层框架+工具分工方法论 | D |
| `ethoinsight-column-confirmation/SKILL.md` | 加 read 指引 | D |
| `docs/design/2026-06-06-data-processing-methodology-design.md` | §6/§8 标注"已实施" | 收尾 |

## 7. 完成判据
- `cd packages/ethoinsight && pytest tests/` 全量绿（不只新测试）
- `cd packages/agent/backend && make test` 全量绿（注意已知 4 个污染测试，见 CLAUDE.md memory，非本次引入的红可忽略但要点名）
- `grep -rn 'validate_metrics(' packages/` 所有调用方行为正确
- L-A 只产 NaN/Inf；L-B 产 output_unit 范围 + 孤儿 + 复合 + 未知单位 violation
- 3 个孤儿指标（velocity_stats/thigmotaxis_index/distance_moved）跑出 `catalog_unknown`
- code-executor guardrail 放行 validate_catalog，lookalike 仍 deny
- skill references 下沉，SKILL.md 有 read 指引，**不含聚合规则、不内嵌 catalog 知识**

## 8. 给执行 agent 的开工检查清单
1. `git log dev --oneline | grep -iE 'S1-S4|validate'` 确认 ① 已合 dev（否则停止报告）
2. 从 dev 拉 `feature/catalog-driven-validation`，在新 worktree 工作
3. 读 `validate.py`（基线）+ `catalog/schema.py:62`（MetricEntry）+ `catalog/loader.py:71`（load_catalog）+ `_common.py:592`（复合 _stats 真实字段名）
4. 按 A→B→C→D 顺序，每步 TDD（红→绿）
5. 改 validator 后**跑全量** + grep 调用方
6. 不实现任何 zone 聚合（Issue #98 边界）
