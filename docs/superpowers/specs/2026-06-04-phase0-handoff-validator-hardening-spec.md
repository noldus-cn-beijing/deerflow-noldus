# Spec：Phase 0 — code-executor handoff 校验器加固（字段唯一真相）

> 状态：实施 spec，可直接交付 agent 执行
> 日期：2026-06-04
> 范围：**仅** code-executor handoff 的校验器加固。**不含** task_context、evaluator、契约形态决策。
> 关联：
> - 愿景背景：[`2026-06-04-handoff-contract-vision-design.md`](2026-06-04-handoff-contract-vision-design.md)（这是契约不变量 1「唯一字段真相」的第一块砖）
> - 问题来源：[`2026-06-04-handoff-task-package-optimization-design-review.md`](2026-06-04-handoff-task-package-optimization-design-review.md) §4.5
> - 铁律：`feedback_handoff_metrics_field_divergence_mislabels_failed`、`feedback_pr_merge_must_run_full_suite_on_shared_logic`

---

## 〇、给实施 agent 的一句话

校验器 `_check_code_executor_content` 只认 `metrics_summary` 一个字段名，导致历史上用 `metrics`/`metrics_results` 字段的 handoff 被误判残缺。**本任务把校验器加固成「字段名唯一真相 + 未知格式显式可见」，并解锁已就位的 red 测试。不改产出端、不改消费者、不动 schema 结构。**

---

## 一、现状事实（已核实，实施前不需重新验证，但要理解）

### 1.1 字段曾经三分裂，但当前产出/消费已统一

56 个真实落盘 `handoff_code_executor.json` 顶层指标字段名：`metrics`（27，历史）/ `metrics_summary`（24，当前）/ `metrics_results`（1，过渡）。

**追落盘时间线 + 写入指纹已证实**：`metrics` 是 5-13~5-27 的历史产物（当时 LLM 直接 `write_file`、seal tool 还不存在，34 个旧样本 100% 缺 `analysis_config_id` + `.lineage/manifest.json`）。5-28 Sprint 0 引入 seal tool 后，**产出端、消费端全部统一到 `metrics_summary`**：

| 角色 | 文件:行 | 字段 |
|------|---------|------|
| 产出（seal tool 参数） | `tools/builtins/seal_handoff_tools.py:235` | `metrics_summary` |
| 消费（data-analyst） | `subagents/builtins/data_analyst.py:115` | `metrics_summary[*].parameters_used` |
| 消费（report-writer） | `subagents/builtins/report_writer.py:30` | `metrics_summary` |
| seal helper 内部 | `tools/builtins/seal_handoff_tools.py:57-58` | `metrics_summary` |

**没有任何当前代码产出或读取 `metrics`/`metrics_results`。** 那 27 个旧样本是孤儿，无管道触及（训练数据管道也不读 handoff 落盘文件，污染风险为零）。

### 1.2 唯一的残留缺陷：校验器硬编码单字段名

`subagents/executor.py:104-107`：

```python
def _check_code_executor_content(d: dict) -> str | None:
    if not d.get("metrics_summary"):
        return "metrics_summary is empty"
    return None
```

它在 `_validate_handoff_emitted` 里被调用（`executor.py:1002,1019`），返回非 None → seal-resume 补一轮 → 仍非 None → `try_set_terminal(FAILED)`（`executor.py:1032`）。对当前 `metrics_summary` 格式工作正常；但对任何写 `metrics`/`metrics_results` 的 handoff（历史样本、或未来新范式若走了不同字段名）会误判残缺 → 误标 FAILED。

### 1.3 red 锚点已就位

`tests/test_handoff_content_validation.py::TestCodeExecutorFieldNameDivergence`（2026-06-04 落盘）：

- `test_real_metrics_field_handoff_should_pass` — `xfail(strict=True)`：`metrics` 字段的完整 handoff 不应被判残缺
- `test_real_metrics_results_handoff_should_pass` — `xfail(strict=True)`：同上 `metrics_results`
- `test_documents_current_buggy_behavior` — pass：固化当前误判行为

当前 `14 passed, 2 xfailed`。本任务完成后，前两个 xfail 会转为 pass（strict 模式会让它们「意外 pass」报错，需手动摘 xfail），第三个需删除/改写。

---

## 二、本任务做什么（精确改动点）

### 2.1 改动 1：校验器认「字段唯一真相」的等价集，并对未知格式显式记录

**文件**：`packages/agent/backend/packages/harness/deerflow/subagents/executor.py`
**函数**：`_check_code_executor_content`（行 104-107）

**改成**：按优先级检查指标数据是否存在，承认历史等价字段名，且**当数据只存在于非规范字段时记 warning 日志（可见但不阻断）**：

```python
# 指标数据的规范字段名（single source of truth）。
# metrics_summary = 当前规范（Sprint 0 起）。
# metrics / metrics_results = 历史等价字段（Sprint 0 前的旁路写入产物），
#   承认其数据有效性以免误判残缺，但记 warning 暴露「非规范格式」。
_CODE_EXECUTOR_METRICS_FIELDS = ("metrics_summary", "metrics", "metrics_results")


def _check_code_executor_content(d: dict) -> str | None:
    present = [f for f in _CODE_EXECUTOR_METRICS_FIELDS if d.get(f)]
    if not present:
        # 三个字段全空 → 真残缺
        return "metrics data is empty (none of metrics_summary/metrics/metrics_results populated)"
    if "metrics_summary" not in present:
        # 数据在历史等价字段，非当前规范 → 放行但暴露，便于发现新的格式漂移
        logger.warning(
            "[handoff_content] code-executor handoff uses non-canonical metrics field(s) %s "
            "instead of 'metrics_summary'. Data accepted, but this is a format drift — "
            "current pipeline should emit metrics_summary.",
            present,
        )
    return None
```

**设计理由**：
- **不再误判**：只要任一等价字段有数据，就不判残缺（解决误标 FAILED）。
- **不静默**：数据走非规范字段时记 warning——这是契约不变量「字段唯一真相」的可观测性兜底。未来某新范式若意外走了 `metrics`，日志立刻暴露，不会像 5-13 那次无声漂移。
- **不放宽真残缺检查**：三个字段全空仍判残缺（保留 Sprint 5.5 的核心保护）。

### 2.2 改动 2：同步注释

`executor.py:98` 的注释（`code-executor : metrics_summary 非空 dict`）改为反映「metrics_summary 规范 + metrics/metrics_results 历史等价」。

### 2.3 改动 3：解锁 red 测试

**文件**：`tests/test_handoff_content_validation.py::TestCodeExecutorFieldNameDivergence`

- `test_real_metrics_field_handoff_should_pass`：**删除 `@pytest.mark.xfail`**（它现在应真 pass）。
- `test_real_metrics_results_handoff_should_pass`：**删除 `@pytest.mark.xfail`**（同上）。
- `test_documents_current_buggy_behavior`：**删除整个测试**（它固化的是已修复的错误行为）。
- 类的 docstring 更新：从「活 bug 回归」改为「字段唯一真相 + 历史等价字段兼容」的契约说明。

### 2.4 新增测试：覆盖加固后的三条路径

在 `TestCodeExecutorContentValidation` 或新类中补：

1. **三字段全空 → 判残缺**：`{"status":"completed"}`（无任何 metrics 字段）→ `_validate` 返回非 None，含 "metrics data is empty"。
2. **metrics_summary 有数据 → pass 且不记 warning**：（已有 `test_nonempty_metrics_summary_passes` 覆盖，确认仍绿）。
3. **metrics/metrics_results 有数据 → pass**：（即解锁的两个测试）。
4. **（可选，验证可观测性）** 用 `caplog` 断言：`metrics` 字段的 handoff 经校验时记了「non-canonical metrics field」warning。

---

## 三、本任务**不做什么**（硬边界）

实施 agent 必须严格遵守，越界即返工：

- ❌ **不改产出端**：`code_executor.py` workflow、`seal_handoff_tools.py` 的 seal 参数——它们已经产 `metrics_summary`，正确，不动。
- ❌ **不迁移旧样本**：那 27 个 `metrics` 孤儿样本不读不动（无管道触及，污染风险零）。
- ❌ **不改 schema 结构**：`handoff_schemas.py` 的 `CodeExecutorHandoff` 不动（去 `extra="allow"`、改 strict 是契约 sprint 的事，不在本任务）。
- ❌ **不做 task_context**：不加 TaskContext 模型、不改 seal 签名、不改 lead prompt。
- ❌ **不做 evaluator**：不新建 `evaluator.py`、不加 evaluator 调用。
- ❌ **不碰其他三个 subagent 的校验**（data-analyst/chart-maker/report-writer 的 content check 不动）。
- ❌ **不决定契约形态**（① Pydantic vs ② SSOT）——那是后续 review 的事。

本任务是「契约第一块砖」的最小切片：**只让 code-executor 校验器不再因字段名误判，并让格式漂移可见。**

---

## 四、验收标准

1. `TestCodeExecutorFieldNameDivergence` 的两个原 xfail 测试**转为 pass**，xfail 标记已删除，无「意外 pass」报错。
2. 新增的「三字段全空判残缺」测试通过。
3. **全量回归绿**：`cd packages/agent/backend && make test`（不是只跑改动的测试文件——`_check_code_executor_content` 是共享逻辑，按 `feedback_pr_merge_must_run_full_suite_on_shared_logic` 铁律必须跑全量，且 grep 所有调用 `_validate_handoff_emitted` / `_HANDOFF_CONTENT_CHECKS` 的测试确认未回归）。
4. `make lint` 通过。
5. 改动后 grep 确认没有其他地方还假设「code-executor handoff 必有 metrics_summary」而会被历史等价字段破坏（重点复查 `executor.py:1016` 注释、data-analyst 消费逻辑——data-analyst 读 `metrics_summary[*].parameters_used`，若喂它 `metrics` 字段样本会取空，但**当前管道不产 metrics，此为历史样本独有，不在本任务修复范围**；实施 agent 只需在 spec 或代码注释中记录此已知边界，不扩大改动）。

---

## 五、提交

- 分支：从 `dev` 切（项目规范：commit 先进 dev）。
- commit message 中文，说明「加固 code-executor handoff 校验器：字段唯一真相 + 历史等价字段兼容 + 格式漂移可见」。
- 不要 push 到 main、不自动建 PR，除非用户明确要求。

---

## 附录：为什么这是「契约第一块砖」而非「修 bug」

contract-vision 文档把 handoff 升级为「节点间声明式契约」，不变量 1 是「唯一字段真相」。本任务实现它的最小可落地形态：

- **唯一真相**：`metrics_summary` 是规范字段（校验器以它为准）。
- **可观测性兜底**：非规范字段被记录而非静默——这正是 contract-vision 强调的「约束在节点间，且可观测」。

它不依赖契约形态（①/②）决策，所以可以现在做；它做完，契约方向得到一次低风险的落地验证。
