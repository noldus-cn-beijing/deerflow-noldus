# Spec S3: handoff schema fail-open 防护 + calamine 启动期断言

> 日期：2026-06-12
> 顺序：第 3 份（共 4 份，按序实施）。独立小项，清理 Fable 核实时记的两个 TODO（A6 + B5）。与 S1/S2/S4 正交。
> 来源：2026-06-12 Fable-5 代码评审实测 + 本次调研坐实
> 实施方式：新开 worktree 基于**已合 S1+S2 的最新 dev**，单 PR
> 前置：calamine 引擎已合 dev（PR #125）；ParamValue 别名已合 dev（PR #125）

---

## 0. 背景

两个正交的健壮性 TODO，都是「哑故障换响亮故障」：

- **TODO 1（schema fail-open）**：handoff 的 `extra="allow"` + `default_factory=dict/list` 组合下，producer 把字段名打错（`parms` 代替 `parameters_used`）→ 模型零报错通过，正确字段拿空默认值，真实数据静默进 `model_extra` → 下游读到「合法的空值」而非异常。Fable 实测演示（pydantic 2.13.4）。
- **TODO 2（calamine 启动断言）**：PR #125 把 Excel 引擎换 calamine（硬依赖），但缺依赖只在第一次 `pd.read_excel(engine="calamine")` 才炸，且 `_detect_ethovision_xlsx` 的 `try/except: return False` 会把 calamine 缺失**误判成「非 EthoVision 文件」**（哑故障）。

两者独立，合一个 PR（都是小型健壮性加固）。

---

## Part A — handoff schema 完成度断言（TODO 1）

### A0 范围裁剪（关键，避免过度）
**不改 `extra="allow"` → `forbid`**。理由：
- `MetricStat`（[handoff_schemas.py:102](packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py)）的 `extra="allow"` 是**有意设计**（docstring :95-100 明说「容纳 ethoinsight 新增 metric-level 统计字段如 median/IQR 不破旧 handoff」）。
- 其他 10 处 `extra="allow"` 的意图虽不全有文档，但横切改 `forbid` 风险大、可能破坏向前兼容，属独立大改。
- **本 spec 只加「status=completed → 关键字段非空」的 model_validator**，把「字段名打错→空值→仍标 completed」的哑故障换成响亮 ValidationError。这是 Fable 推荐的精准处方，不动 `extra` 配置。

### A1 给四个承重 handoff 加完成度 validator
四个承重 schema（被 seal/校验流程当真相源）目前**都没有** model_validator 关联 status 与字段非空：

| schema | 行号 | status=completed 时该非空的关键字段 |
|---|---|---|
| `CodeExecutorHandoff` | [:409](packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py) | `metrics_summary` 非空（核心产物） |
| `DataAnalystHandoff` | :575 | `key_findings` 非空 |
| `ChartMakerHandoff` | :503 | `chart_files` 非空 |
| `ReportWriterHandoff` | :632 | `sections_written` 非空（或 report_path 有值） |

给每个加 `@model_validator(mode="after")`：
```python
@model_validator(mode="after")
def _completed_requires_core_output(self):
    """status=completed 但核心产物为空 = 字段名打错/聚合漏填的哑故障，响亮拒绝。

    Fable 实测（2026-06-12）：extra="allow" + default_factory=dict 下，producer
    字段名打错（parms vs <field>）→ 正确字段拿空默认 → 仍可标 completed →
    下游读到"合法空值"。本校验把这类哑故障换成响亮 ValidationError。
    partial/failed 允许空（部分/失败本就可能无产物）。
    """
    if self.status == "completed" and not self.<core_field>:
        raise ValueError(
            f"status='completed' but <core_field> is empty — "
            f"likely a field-name typo (data landed in model_extra) or aggregation gap. "
            f"Use status='partial'/'failed' if output is genuinely incomplete."
        )
    return self
```
- 每个 schema 的 `<core_field>` 按上表填。
- **只对 `completed` 卡**，`partial`/`failed` 放行（它们本就可能无产物，见 n=1 partial 路径）。
- 注意 import：`model_validator` 已在 handoff_schemas.py 顶部 import（DataQualityWarning/ParameterAuditFinding 已用）。

### A2（可选，低成本）seal 错误信息暴露 model_extra
[seal_handoff_tools.py:376](packages/agent/backend/packages/harness/deerflow/tools/builtins/seal_handoff_tools.py) `_seal_handoff_to_workspace` 捕获 ValidationError 时，若构造成功的 handoff 对象有非空 `model_extra`，在返回/日志里提示「检测到额外字段 X，疑似字段名打错」。**列可选**，不阻塞 A1。

### A 部分测试
- **红→绿**：构造 `CodeExecutorHandoff(status="completed", metrics_summary={})`（空核心产物）→ 改前通过（哑故障）、改后抛 ValidationError。四个 schema 各一例。
- **partial/failed 放行**：`status="partial"` + 空产物 → 不抛（回归保护，守 n=1 partial 路径）。
- **正常 completed 通过**：非空产物 + completed → 通过。
- **字段名打错复现**：用 `CodeExecutorHandoff(status="completed", parms={...})`（错字段名）→ `metrics_summary` 拿空默认 → 触发新 validator 响亮失败（这是 Fable 演示的核心哑故障场景）。

---

## Part B — calamine 启动期断言（TODO 2）

### B0 现状（调研坐实）
- `EXCEL_ENGINE = "calamine"` 在 [parse/_core.py:26](packages/ethoinsight/ethoinsight/parse/_core.py)，三处 read_excel（:97/:201/:326）都用它。
- `python-calamine>=0.6,<0.7` 是**硬依赖**（[pyproject.toml:12](packages/ethoinsight/pyproject.toml)，非 optional）。
- **缺依赖不在 import 期炸**：`import ethoinsight.parse` 不隐式 import calamine；只有第一次 `pd.read_excel(engine="calamine")` 才 ImportError。
- **哑故障点**：`_detect_ethovision_xlsx`（:96-99）`try: pd.read_excel(...) except Exception: return False` → calamine 缺失被误判成「非 EthoVision 文件」。

### B1 在 parse 模块 import 期断言 calamine 可用
在 [ethoinsight/parse/__init__.py](packages/ethoinsight/ethoinsight/parse/__init__.py) 顶部（现有 import 之前）加：
```python
# Excel 解析依赖 python-calamine（PR #125 起，硬依赖）。在 parse 模块 import 期
# 响亮断言其可用——缺依赖应在部署/启动当场失败，而非运行到第一次读 xlsx 才炸，
# 更不该被 _detect_ethovision_xlsx 的 try/except 误判成"非 EthoVision 文件"（哑故障）。
try:
    import python_calamine  # noqa: F401  # ⚠️ import 名是 python_calamine 不是 calamine
except ImportError as e:
    raise ImportError(
        "python-calamine is required for ethoinsight.parse (Excel XLSX/XLS reading via "
        "pandas engine='calamine', PR #125). Install: uv add 'python-calamine>=0.6,<0.7' "
        "or pip install 'python-calamine>=0.6,<0.7'."
    ) from e
```
> **⚠️ import 名核实**：calamine 的 import 名是 **`python_calamine`**（不是 `calamine`）——本次会话已实测确认 `import python_calamine` 可用、`import calamine` 失败。spec 调研 agent 误写成 `import calamine`，实施时务必用 `python_calamine`。

放 `parse/__init__.py` 而非 `ethoinsight/__init__.py`：只在用 parse（读 Excel）时强制，不波及只用 metrics/statistics 的纯计算用户。

### B2（可选）`_detect_ethovision_xlsx` 不吞 ImportError
[parse/_core.py:96-99](packages/ethoinsight/ethoinsight/parse/_core.py) 的 `except Exception: return False` 现在吞一切。B1 落地后 calamine 缺失会在 import 期就炸（到不了这里），故 B2 **优先级降低**。若要做：把 `except Exception` 收窄，让 `ImportError`/`ModuleNotFoundError` 透传（区分「真不是 EV 文件」vs「引擎缺失」）。**列可选**，B1 已根治主路径。

### B 部分测试
- **import 期断言**：mock `python_calamine` 不可导入（`sys.modules` 注入 None 或 monkeypatch `builtins.__import__`），断言 `import ethoinsight.parse` 抛带指引的 ImportError。calamine 实际可用时此测试 skip 或用 mock 隔离。
- **正常路径**：calamine 可用时 `import ethoinsight.parse` 成功（回归）。

---

## 实施顺序
1. **Part A**（schema）：四个 validator 先写红测试 → 加 validator → 转绿。
2. **Part B**（calamine 断言）：B1 加 import 断言 + 测试。
3. A/B 同 PR，commit 可分两笔。

## 验证（端到端）
```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
# A: schema validator
PYTHONPATH=. .venv/bin/python -m pytest tests/test_handoff_schemas.py tests/test_metric_stat_and_executor_schema.py -q
# B: calamine 断言（在 ethoinsight）
cd ../../ethoinsight && pytest tests/ -k "calamine or parse" -q
# 全量回归（基线含 test_subagent_executor 的 ModuleNotFoundError 环境债，勿归因本次）
cd ../agent/backend && PYTHONPATH=. .venv/bin/python -m pytest -q
# 改了 schema → 裸导入两生产入口
PYTHONPATH=. .venv/bin/python -c "import app.gateway"
PYTHONPATH=. .venv/bin/python -c "from deerflow.agents import make_lead_agent"
```

## 关键文件
- `packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py`（四个 handoff 加 completed validator）
- `packages/agent/backend/tests/test_handoff_schemas.py`（红→绿）
- `packages/ethoinsight/ethoinsight/parse/__init__.py`（calamine import 断言）
- `packages/ethoinsight/tests/`（calamine 断言测试）

## 红线（勿违）
- **不改 `extra="allow"` → `forbid`**（MetricStat 是有意向前兼容；横切改属独立大改）。只加 completed validator。
- validator **只对 `completed` 卡**，partial/failed 放行（守 n=1 partial 路径，见 memory `feedback_dataanalyst_reportwriter_handoff_status_missing_partial`）。
- calamine import 名是 **`python_calamine`** 不是 `calamine`（实测坐实）。
- 断言放 `parse/__init__.py` 不放 `ethoinsight/__init__.py`（不波及纯计算用户）。
- 改 schema 后跑裸导入两生产入口。
- `test_subagent_executor` 的 ModuleNotFoundError 失败是纯 dev 环境债，勿归因本次。
