# Code-executor 处理 groups

**适用**: code-executor read plan_metrics.json 之后、跑 compute_* 之前。

## 强制规则

读到 plan 后**立刻检查** `plan.inputs.groups_file`：

```python
groups_file = plan["inputs"]["groups_file"]  # str | null
```

| `groups_file` 状态 | code-executor 行为 |
|---|---|
| 非空字符串（如 `/mnt/user-data/workspace/groups.json`）| **必须** read_file 读它，按 `{subject: group_name}` dict 分组聚合 |
| `null` / 缺失 | 按"无分组（per-subject only）"处理，metrics_summary 用 `"all"` 作为 group key 或省略分组层 |

## 标准聚合流程（有 groups 时）

1. **读 groups.json**：
   ```bash
   read_file /mnt/user-data/workspace/groups.json
   ```
   返回 dict 形如：
   ```json
   {
     "/mnt/.../arena1.txt": "treatment",
     "/mnt/.../arena2.txt": "control"
   }
   ```
2. **按 group_name 反向索引**：
   ```
   group_to_subjects = {
     "treatment": ["arena1"],
     "control": ["arena2"],
   }
   ```
3. **跑完所有 compute_\* 之后**，收集每个 subject 的 `[result] {json}` 行，按 group 聚合：
   - 对每个 metric_id：
     - 对每个 group：算 `mean / std / n` over subjects-in-group
4. **写 metrics_summary**：
   ```python
   metrics_summary = {
     "treatment": {
       "immobility_time": {"mean": 120.5, "std": null, "n": 1},
       ...
     },
     "control": {
       "immobility_time": {"mean": 95.2, "std": null, "n": 1},
       ...
     },
   }
   ```
5. **per_subject 也保留**：
   ```python
   per_subject = {
     "Subject 1 (arena1)": {"immobility_time": 120.5, ...},
     "Subject 2 (arena2)": {"immobility_time": 95.2, ...},
   }
   ```

## 标准聚合流程（无 groups 时）

不要尝试探测分组。直接：

```python
metrics_summary = {
  "all": {  # 或省略,空字典也合法
    "immobility_time": {"mean": ..., "std": ..., "n": N},
    ...
  }
}
per_subject = { ... }  # 同上
```

## 不存在脚本立即停手

如果发现"自己应该探测分组"的冲动 → **看一眼 plan.inputs.groups_file**：

- **非空** → 你应该读这个 JSON 而不是探测原始数据
- **null** → 不要探测；按无分组处理；write `errors` 提示 lead

**禁止调用以下幻觉脚本**（实证 2026-05-28 thread 485a899d 踩过）：

```
python -m ethoinsight.scripts.<paradigm>.inspect_drug_column      # ❌ 不存在
python -m ethoinsight.scripts.<paradigm>.dump_subject_metadata    # ❌ 不存在
python -m ethoinsight.scripts.<paradigm>.view_drug_values         # ❌ 不存在
python -c "import pandas as pd; ..."                              # ❌ 被 sandbox 拦
```

详见 [missing-script-recovery.md](missing-script-recovery.md)。

## 派遣 prompt 与 plan 冲突时

如果 lead 的派遣 prompt 用自然语言说"按 drug 列分组"，但 `plan.inputs.groups_file` 是 null：

**正确做法**：
1. 不按 lead 的自然语言指令执行（无法可靠翻译成代码）
2. 按 plan（无分组）跑完 metrics
3. seal handoff 时在 `errors` 数组加一条：
   ```
   "lead 派遣 prompt 提到分组但 plan.inputs.groups_file 为 null;
    需 lead 通过 prep_metric_plan(groups={path: group_name, ...}) 重新生成 plan"
   ```
4. lead 看到这条 error 会知道下次该传 groups dict

**错误做法**：自己用 bash 试图读 xlsx / 调不存在的脚本探测 drug 列 → 浪费 turn 预算 → 最终还是失败。

## 校验

读完 groups.json 后建议做一次 sanity check：

- groups dict 的 keys 是否全部在 plan.metrics[].input 中出现？
- 是否有 plan.metrics[].input 不在 groups dict 中？（部分 subject 无分组归属）

任一不一致 → 写入 data_quality_warnings 一条 `SAMPLE.GROUP_MISMATCH` warning：
```python
{
  "severity": "warning",
  "code": "SAMPLE.GROUP_MISMATCH",
  "metric": "all",
  "evidence": {
    "subjects_without_group": [...],
    "extra_groups_unused": [...],
  },
  "message": "Some subjects not in groups.json or groups.json contains unused entries.",
  "blocks_downstream": False,
}
```

注意 code 必须用 **点分隔**（`SAMPLE.GROUP_MISMATCH` 不是 `SAMPLE_GROUP_MISMATCH`）。
