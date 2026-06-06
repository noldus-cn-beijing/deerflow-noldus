# S1-S4 实施 Spec

**日期**：2026-06-06
**状态**：可实施

---

## S1: Subagent 级 LoopDetectionMiddleware

### 现状

- `LoopDetectionMiddleware` 已存在且对 subagent 适用（基于 `AgentState`，不依赖 lead-only 字段）
- 当前 `executor.py:_build_middlewares` 没有包含它
- Lead agent 已有（`agent.py:285`）

### 关键风险：thread_id 共享导致计数污染

`LoopDetectionMiddleware` 按 `thread_id` 跟踪历史（`loop_detection_middleware.py:264-280`，`self._history[thread_id]`）。Subagent 与 lead **共享同一 `thread_id`**（`executor.py:877-879`）。如果共享 middleware 实例，计数互污染。

**解法**：每次 subagent 运行实例化**新的** `LoopDetectionMiddleware()`。`_build_middlewares` 每次 subagent run 调用一次（`executor.py:613`），所以新实例天然隔离。Lead 保有自己的长生命周期实例。

### 实施

**文件**：`packages/agent/backend/packages/harness/deerflow/subagents/executor.py`

在 `_build_middlewares` 方法中（约 line 651，现有 `GuardrailMiddleware` 之后、`return` 之前）：

```python
from deerflow.agents.middlewares.loop_detection_middleware import LoopDetectionMiddleware

# 在 _build_middlewares 的 return 之前添加：
middlewares.append(LoopDetectionMiddleware())
```

### 检测行为

双层检测（`loop_detection_middleware.py:64-69`）：
1. `(tool_name + args)` 哈希滑动窗口 → warn=3 次重复，hard_limit=5 次
2. 按 tool type 频率计数 → 防"微调 bash 参数绕过哈希"变体

Warn → 注入 `HumanMessage` 提示"你正在重复自己"。Hard limit → 剥离所有 `tool_calls`，强制产出文本。

`recursion_limit` 自动调整（`executor.py:47-72`，`calculate_subagent_recursion_limit` 统计 middleware hooks）。

### 测试

```python
# tests/test_subagent_loop_detection.py

async def test_subagent_repeated_tool_call_is_detected():
    """Subagent 重复相同 tool call → warn → hard limit"""

async def test_lead_and_subagent_loop_history_are_isolated():
    """Lead 的 tool call 历史不污染 subagent，反之亦然"""
```

---

## S2: 代码级 Metric Validator

### 设计

AutoResearch 的 `if math.isnan(loss): exit(1)` 是代码级检查——确定性逻辑不交给 LLM。

在 `ethoinsight/` 库中新增 `validate.py`，compute scripts 完成指标计算后调用。不阻止输出（violation 时仍输出值），由 data-analyst 在 Checkpoint 2 判断"是否继续派遣"。

### 实施

**文件**：`packages/ethoinsight/ethoinsight/validate.py`（新建）

```python
"""Metric validation — deterministic range/NaN checks.

AutoResearch-inspired: code-enforced checks that shouldn't be left to LLM judgment.
"""

import math
from typing import Any


def validate_metrics(metrics: dict[str, Any]) -> list[dict[str, str]]:
    """Validate computed metrics are in plausible ranges.

    Args:
        metrics: {metric_name: value} dict from compute script output.

    Returns:
        List of violations (empty list = all clear).
        Each violation: {"metric": str, "issue": str, "value": str}
    """
    violations: list[dict[str, str]] = []

    for name, value in metrics.items():
        if not isinstance(value, (int, float)):
            continue

        # NaN/Inf check
        if math.isnan(value):
            violations.append({"metric": name, "issue": "NaN", "value": "NaN"})
            continue
        if math.isinf(value):
            violations.append({"metric": name, "issue": "Inf", "value": str(value)})
            continue

        # Percentage range check (naming convention: *_pct)
        if name.endswith("_pct") and not (0.0 <= value <= 100.0):
            violations.append({
                "metric": name,
                "issue": "percentage_out_of_range",
                "value": str(value),
            })

        # Non-negative check for duration/distance/velocity/count
        if any(
            name.startswith(prefix)
            for prefix in ("distance_", "duration_", "velocity_", "count_")
        ):
            if value < 0:
                violations.append({
                    "metric": name,
                    "issue": "negative_value",
                    "value": str(value),
                })

    return violations
```

**在 compute scripts 中调用**（以 `epm.py` 为例）：

```python
# 在 compute script 末尾，输出指标之后
from ethoinsight.validate import validate_metrics

violations = validate_metrics(results)
if violations:
    for v in violations:
        print(f"VALIDATION_ERROR: {v['metric']}: {v['issue']} (value={v['value']})")
    # 不阻止输出 — data-analyst 看到 VALIDATION_ERROR 后标记 partial
```

**需要修改的文件**：每个活跃范式对应的 compute script（EPM/OFT/LDB/FST/Zero Maze/TST），在输出指标后加 4 行调用。

### 边界

- 百分比后缀约定（`_pct`）需与 catalog 的指标命名对齐
- 不做跨指标逻辑检查（如"开臂时间 + 闭臂时间 = 100%"归 data-analyst）
- violation 不阻止输出，由 data-analyst 在 fast-fail 规则中处理

### 测试

```python
# packages/ethoinsight/tests/test_validate.py

class TestValidateMetrics:
    def test_normal_values_pass(self):
        assert validate_metrics({"pct_open_arm_time": 45.2, "distance_total": 1500.0}) == []

    def test_nan_is_detected(self):
        violations = validate_metrics({"pct_open_arm_time": float("nan")})
        assert len(violations) == 1
        assert violations[0]["issue"] == "NaN"

    def test_inf_is_detected(self):
        violations = validate_metrics({"pct_open_arm_time": float("inf")})
        assert len(violations) == 1
        assert violations[0]["issue"] == "Inf"

    def test_percentage_out_of_range(self):
        violations = validate_metrics({"pct_open_arm_time": 150.0})
        assert len(violations) == 1
        assert violations[0]["issue"] == "percentage_out_of_range"

    def test_negative_distance(self):
        violations = validate_metrics({"distance_total": -50.0})
        assert len(violations) == 1
        assert violations[0]["issue"] == "negative_value"

    def test_percentage_at_boundary_passes(self):
        assert validate_metrics({"pct_open_arm_time": 0.0}) == []
        assert validate_metrics({"pct_open_arm_time": 100.0}) == []

    def test_non_numeric_values_are_skipped(self):
        assert validate_metrics({"note": "no data"}) == []
```

---

## S3: data-analyst SKILL.md Fast-Fail 规则

### 设计

在 data-analyst 的 SKILL.md 中增加"快速失败"段。只含 LLM 擅长的判断，不含代码级检查（NaN/范围由 S2 validator 处理）。

### 实施

**文件**：`packages/agent/skills/custom/ethoinsight-data-analyst/SKILL.md`（或实际路径）

在现有内容之前（或作为独立段落）插入：

```markdown
## Fast-Fail Rules

Before proceeding to full interpretation, check for these conditions.
If any hard-fail triggers, emit handoff immediately with status="failed" 
or status="partial" — do NOT produce a full analysis.

### Hard Fail (abort interpretation)

1. **Insufficient sample size**: any group has n < 3.
   → Emit handoff status="partial". Report descriptive statistics only.
   Do NOT perform inferential tests. Note "n<3: descriptive only".

2. **All metrics failed**: code-executor output contains VALIDATION_ERROR 
   for every metric, or all metrics are null/missing.
   → Emit handoff status="failed". Note "all metrics invalid".

3. **Data quality gate not passed**: Gate 2 gate_signals indicate 
   unrecoverable issues (wrong file format, completely mismatched columns).
   → Do not proceed. Emit handoff status="failed". Report the gate failure.

### Soft Fail (continue with limitation note)

4. **Normality-test mismatch**: Shapiro-Wilk p < 0.05 but parametric test 
   was chosen by the statistical decision tree (statistics.py).
   → Flag as potential issue but do NOT override. The decision tree is 
   deterministic. Note the mismatch for expert review.

5. **Small effect with non-significant p**: Cliff's δ < 0.3 AND p > 0.05.
   → Report findings but mark confidence="low".

### Warning (continue with caveat)

6. **Conclusion-statistics inconsistency**: finding text claims "significant" 
   but p ≥ 0.05 for the relevant metric.
   → Flag as report-writer error if detected in final report.

7. **Forbidden claims**: absolute threshold judgment ("X% is normal"), 
   hallucinated mechanisms ("acts on GABA receptors"), or claims not 
   supported by statistical output.
   → Flag and request revision. Never include in final findings.
```

### 测试

手动 dogfood：给 data-analyst 一个 n=2 的 handoff → 产出 `status="partial"`；给一个全部 VALIDATION_ERROR 的 code-executor 输出 → 产出 `status="failed"`；正常 handoff → 不走 fast-fail。

---

## S4: ethoinsight/ 写保护 Red Anchor Test

### 背景

Audit 确认 sandbox 两层防护已拒绝 `write_file` 到 `.venv/site-packages/ethoinsight/`。但无回归测试。万一未来 sandbox 配置变更打破保护，会静默失效。

### 实施

**文件**：`packages/agent/backend/tests/test_ethoinsight_write_protection.py`（新建）

```python
"""Red anchor tests: ethoinsight/ library code is immutable.

These tests verify that agents CANNOT modify ethoinsight/ library code
via write_file, str_replace, or bash operations. All tests expect DENY.

If any test unexpectedly passes (= write is allowed), CI fails — 
the protection has been broken by a configuration change.
"""

import pytest


class TestEthoInsightWriteProtection:
    """Verify ethoinsight/ library code is write-protected at the harness level."""

    async def test_write_file_to_ethoinsight_package_is_denied(self):
        """Agent cannot write_file to ethoinsight/ package directory."""
        # Attempt to write to a path under .venv/.../ethoinsight/
        # Expected: PermissionError or deny message from validate_local_tool_path
        ...

    async def test_str_replace_to_ethoinsight_package_is_denied(self):
        """Agent cannot str_replace ethoinsight/ library code."""
        ...

    async def test_bash_cp_into_venv_is_denied_for_code_executor(self):
        """code-executor bash cp/mv/tee into .venv/site-packages is denied
        by ScriptInvocationOnlyProvider._is_path_safe (line 161)."""
        ...
```

### 防护链验证

测试需覆盖两条防护路径：

| 防护层 | 文件 | 关键行 | 保护范围 |
|--------|------|--------|---------|
| Sandbox path validation | `sandbox/tools.py` | `validate_local_tool_path:645-700` | 所有 agent 的 `write_file`/`str_replace` |
| Script invocation guard | `guardrails/script_invocation_only_provider.py` | `_is_path_safe:161` (`.venv`/`site-packages` 拒绝) | code-executor/chart-maker 的 bash 操作 |

---

## S1-S4 实施顺序

无依赖关系，可并行实施：

```
S1 ─┐
S2 ─┼── 并行实施 ──→ PR
S3 ─┤
S4 ─┘
```

**预估工作量**：总计 < 半天。
- S1: 1 行代码 + 2 个测试
- S2: ~40 行 Python + 6 行 × 6 个 compute scripts + 7 个测试
- S3: ~40 行 markdown
- S4: 3 个测试函数
