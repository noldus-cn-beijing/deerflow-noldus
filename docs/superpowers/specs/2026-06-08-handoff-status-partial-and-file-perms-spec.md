# Spec 1 — handoff status 补 `partial` + 指标输出文件权限修复

> 日期：2026-06-08 ｜ 目标分支：从 `dev` 新建 worktree
> 来源：EPM n=1 dogfood（thread `1bda1847`）发现的 3.2 + 3.3 两个问题
> 性质：schema/契约层小修，**根因明确、改动面小、可直接 TDD**。两个问题独立但都属"输出契约 bug"，合并为一个 spec。
> 这是给执行 agent 的施工单，不是给用户的总结。

---

## 0. 背景与证据（已现场核实，不要重新质疑）

本次 EPM dogfood（PR #104 合入 dev 后，单文件 n=1）暴露两个**独立**的输出契约 bug。两者都不是 prompt 问题，是 schema/文件 I/O 层的 bug。

### 问题 3.2：data-analyst 在 n=1 卡死 —— 真根因是 `DataAnalystHandoff.status` 缺 `partial`

**实锤证据**（gateway.log，thread 1bda1847，trace=acdfb7e5）：
```
2026-06-08 11:47:10 - deerflow.agents.middlewares.tool_error_handling_middleware - ERROR -
  Tool execution failed (async): name=seal_data_analyst_handoff
  File ".../seal_handoff_tools.py", line 375, in _seal_handoff
    handoff = model_cls(**payload)
pydantic_core.ValidationError: 1 validation error for DataAnalystHandoff
status
  Input should be 'completed' or 'failed' [type=literal_error, input_value='partial', input_type=str]
```

**真相**（与 dogfood 报告 §3.2 的"prompt 矛盾→忘记调 seal/叙述黑洞"判断**相反**）：
- data-analyst **记得调 seal、也真的调了** `seal_data_analyst_handoff`，传 `status="partial"`。
- `partial` 在 n=1（数据不完整、只能描述性统计、无推断检验）下是**语义最正确**的状态。
- 但 `DataAnalystHandoff` schema 的 `status` 是 `Literal["completed", "failed"]`，**不接受 `partial`** → Pydantic 抛 ValidationError → seal 失败 → subagent 反复试 → 最后被迫改 `failed` 才落盘（磁盘 handoff_data_analyst.json 实测 `status=failed`）。
- 所谓"反复辩论 5 分钟"是它调 `partial` 被拒、收到 ValidationError ToolMessage、困惑重试的过程，**不是忘记 seal**。

**契约自相矛盾的双证**：
- `data_analyst.py`（prompt）在 **4 处**明确教传 `partial`：line 75 `or status="partial"`、line 80 `→ Emit handoff status="partial"`、line 127 `任一组 n < 3 → emit handoff status="partial"`、line 131 `直接调 seal_data_analyst_handoff 写入 status="partial"/"failed"`。
- 但 schema 拒收 `partial`。**prompt 教传 partial，schema 拒收 partial。**

**四个 handoff schema 的 status 枚举不一致**（`packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py`）：

| Schema | 行号 | status Literal | partial? |
|--------|------|---------------|----------|
| `CodeExecutorHandoff` | 394 | `completed / partial / failed` | ✅ |
| `ChartMakerHandoff` | 480 | `completed / partial / failed` (默认 completed) | ✅ |
| **`DataAnalystHandoff`** | 552 | `completed / failed` | ❌ **缺** |
| **`ReportWriterHandoff`** | 609 | `completed / failed` | ❌ **缺** |

`partial` 是项目里**已确立的通用约定**（两个 handoff 有、prompt 处处用、两个 seal 工具 docstring `seal_handoff_tools.py:423/521` 都写三态），DataAnalyst/ReportWriter 缺它是**疏漏**，不是有意设计。

### 问题 3.3：指标输出 JSON 文件权限 600，导致 L-B catalog 校验被跳过

**实锤证据**：
- `m_open_arm_time.json` 等指标文件实测 `-rw-------`（600）。
- code-executor seal 摘要含 warning：`validate_catalog 因文件权限 mode 600 报不可读（不影响数据有效性）`。
- L-B 层 `validate_catalog`（catalog-driven 范围校验）读不了 600 文件 → 校验被静默跳过。不影响指标值正确性（值已在 compute stdout 的 `[result]` 行），但**两层指标验证的第二层失效**。

**真根因**（与 dogfood 报告 §3.3 建议的修复位置**不同**）：
- 报告说改 `emit_result`，**错了** —— `emit_result`（`packages/ethoinsight/ethoinsight/scripts/_cli.py:25`）只 `print` 那个 `[result]` 行到 stdout，**不写文件**。
- 真正写指标 JSON 的是 `save_output_json`（`_cli.py:53`）。它用 `tempfile.mkstemp()`（line 57）建临时文件——**`mkstemp` 默认权限就是 `0o600`**（安全默认，仅 owner 可读写），`os.replace` 保留该权限 → 最终文件 600。
- 所有 compute 脚本都走 `save_output_json` 写输出 → **改这一处，全范式全指标覆盖**。

---

## 1. 改动清单（两处源码 + 两处测试，TDD）

### 改动 A（3.2）：补 `partial` 到两个 schema

**文件**：`packages/agent/backend/packages/harness/deerflow/subagents/handoff_schemas.py`

1. **line 552** `DataAnalystHandoff.status`：
   ```python
   # 改前
   status: Literal["completed", "failed"]
   # 改后
   status: Literal["completed", "partial", "failed"]
   ```

2. **line 609** `ReportWriterHandoff.status`：
   ```python
   # 改前
   status: Literal["completed", "failed"]
   # 改后
   status: Literal["completed", "partial", "failed"]
   ```
   > ReportWriterHandoff 一并补，理由：report-writer 同样可能遇到上游 partial/数据不全只能出部分报告的场景，`partial` 同样是合法语义；且四个 handoff 统一三态消除未来同类坑。这是**主动对齐**，不是过度设计——三态本就是既定约定。

**不要改** `CodeExecutorHandoff`(394)/`ChartMakerHandoff`(480)，它们已有 partial。

### 改动 B（3.3）：`save_output_json` 写完后 chmod 0o644

**文件**：`packages/ethoinsight/ethoinsight/scripts/_cli.py`，函数 `save_output_json`（line 53-64）

当前：
```python
def save_output_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write `data` to `path` atomically (temp file + rename), creating parent dirs."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, p)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise
```

改为（在 `os.replace` 之后补一行 chmod，让最终文件 0o644）：
```python
def save_output_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write `data` to `path` atomically (temp file + rename), creating parent dirs.

    Note: tempfile.mkstemp() creates the temp file with 0o600 (owner-only). The
    metric JSONs must be world-readable so the L-B catalog validator
    (`python -m ethoinsight.validate_catalog`) — which may run under a different
    sandbox uid — can read them. So we relax to 0o644 after the atomic rename.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, p)
        os.chmod(p, 0o644)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise
```

> 实现注意：
> - chmod 放在 `os.replace` **之后**（对最终路径 `p` chmod，不是对 tmp）。`os.replace` 是原子的，chmod 紧随其后；即便 chmod 失败也已落盘正确数据（chmod 失败极罕见，可让它进 except 清理——但更稳妥是 chmod 单独不回滚已 replace 的文件。执行 agent 按上面写法即可：chmod 在 try 内，失败会 unlink tmp_path——但此时 tmp 已被 replace 消费、unlink missing_ok 无害，已落盘的 p 不受影响）。
> - **不要**用 umask 方案（进程级副作用，影响其他文件）。
> - **不要**去改 `mkstemp` 的 mode 参数后再 replace——chmod 最终文件更直观、更易测。

---

## 2. 测试（TDD，先写 red）

### 测试 A（3.2）：放 `packages/agent/backend/tests/`

新增或追加到 handoff schema 测试（找现有 `test_*handoff*schema*.py` 或 `test_metric_stat_and_executor_schema.py`；若无合适文件新建 `test_handoff_status_partial.py`）：

```python
import pytest
from pydantic import ValidationError
from deerflow.subagents.handoff_schemas import (
    DataAnalystHandoff, ReportWriterHandoff, CodeExecutorHandoff, ChartMakerHandoff,
)

class TestHandoffStatusPartialConsistency:
    def test_data_analyst_accepts_partial(self):
        # red 锚点：修复前这里抛 ValidationError（dogfood 1bda1847 trace=acdfb7e5 实证）
        h = DataAnalystHandoff(status="partial")
        assert h.status == "partial"

    def test_report_writer_accepts_partial(self):
        h = ReportWriterHandoff(status="partial", report_path="/x/r.md")
        assert h.status == "partial"

    def test_all_four_handoffs_share_same_status_enum(self):
        # 防回归：四个 handoff 的 status 三态一致
        for cls, extra in [
            (CodeExecutorHandoff, {"summary": "s", "paradigm": "epm"}),
            (ChartMakerHandoff, {}),
            (DataAnalystHandoff, {}),
            (ReportWriterHandoff, {"report_path": "/x/r.md"}),
        ]:
            for st in ("completed", "partial", "failed"):
                obj = cls(status=st, **extra)
                assert obj.status == st

    def test_invalid_status_still_rejected(self):
        with pytest.raises(ValidationError):
            DataAnalystHandoff(status="garbage")
```
> 注意各 handoff 的必填字段：`CodeExecutorHandoff` 需 `summary`+`paradigm`（status 之外）；`ReportWriterHandoff` 需 `report_path`。执行 agent 跑测试时按实际 required 字段补全，别让测试因缺别的字段而误红。

### 测试 B（3.3）：放 `packages/ethoinsight/tests/`

新建或追加 `test_cli_output.py`：
```python
import json, os, stat
from ethoinsight.scripts._cli import save_output_json

def test_save_output_json_is_world_readable(tmp_path):
    out = tmp_path / "m_test.json"
    save_output_json(out, {"value": 1.23})
    mode = stat.S_IMODE(os.stat(out).st_mode)
    # red 锚点：修复前是 0o600（mkstemp 默认），validate_catalog 读不了
    assert mode == 0o644, f"expected 0o644, got {oct(mode)}"
    # 数据完整性不受影响
    assert json.loads(out.read_text())["value"] == 1.23
```

---

## 3. 验收标准

1. 改动前：测试 A 的 `test_data_analyst_accepts_partial` + 测试 B 的 `test_save_output_json_is_world_readable` 都 **red**（坐实根因）。
2. 改动后：
   - `cd packages/agent/backend && PYTHONPATH=. uv run pytest tests/test_handoff_status_partial.py -v` 全绿。
   - `cd packages/ethoinsight && uv run pytest tests/test_cli_output.py -v` 全绿。
3. **回归**：因为改了共享 schema + 共享 I/O helper，**必须跑全量**（memory 教训：改共享判定逻辑只跑新测试会漏静默 fail）：
   - `cd packages/agent/backend && make test`
   - `cd packages/ethoinsight && uv run pytest tests/ -q`
   - 已知全量有 4 个确定性污染失败（deferred_tool_registry_promotion×2 + inspect_gate_guardrail/paradigm_identification_gate 的 async test），与本改动无关，不阻塞。
4. **不改 prompt**：data_analyst.py / report_writer.py 的 prompt **不动**——它们教传 partial 本来就对，错的是 schema。改完 schema 后 prompt 与 schema 自洽。

---

## 4. 影响面核实（已查，确认安全）

- **没有下游对 `DataAnalystHandoff.status` / `ReportWriterHandoff.status` 做"只认 completed/failed"的硬比较**。已 grep：`manager.py`/`worker.py`/`task_tool.py` 比较的是 RunStatus/SubagentStatus（runtime 枚举），不是 handoff JSON 的 status。
- report-writer 消费的是 data-analyst handoff 的 `key_findings`/`recommendations`（**内容**，report_writer.py:114/134），从不读其 status → 补 partial 不破坏下游渲染。
- 补 `partial` 是**放宽** Literal（新增允许值），对已有 `completed`/`failed` 调用零影响。

---

## 5. 提交

- worktree 名建议：`worktree-handoff-status-partial-and-file-perms`
- commit message（中文）：
  - `fix(schema): DataAnalyst/ReportWriter handoff status 补 partial，四 handoff 三态对齐`
  - `fix(ethoinsight): save_output_json chmod 0o644，修 validate_catalog 读不了 600 文件`
- 跑完全量测试确认绿（除已知 4 污染）后，建 PR 合入 dev。

---

## 6. 关联

- dogfood 报告（根因被本 spec 纠正）：`docs/handoffs/2026-06/2026-06-08-epm-dogfood-findings.md` §3.2/§3.3
- memory：`feedback_dataanalyst_reportwriter_handoff_status_missing_partial.md`（本 spec 的根因记录）
- 同类"handoff 契约层 bug 伪装成 prompt 问题"：`feedback_handoff_metrics_field_divergence_mislabels_failed.md`
- 3.1（guardrail n=1 路径）单独走 Spec 2：`docs/superpowers/specs/2026-06-08-guardrail-n1-path-awareness-spec.md`。**注意**：3.1 修好后 n=1 不再强制 data-analyst，会从源头减少触发本 spec 3.2 的路径——但 3.2 仍须独立修，因为 n≥2 时 data-analyst 若想报 partial（如某组数据质量差）一样会撞 schema。两个 spec 正交，不互相阻塞。
