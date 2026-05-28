# 不存在脚本的应对

**适用**: code-executor 跑 `python -m ethoinsight.scripts.<paradigm>.<name>` 报错时。

## 触发现象

bash 调用返回任一以下错误：

```
/usr/bin/python: No module named ethoinsight.scripts.<paradigm>.<name>
```

```
return code != 0
stderr: <traceback> 顶层 ModuleNotFoundError / ImportError
```

```
[script not found] / [unknown script]  （sandbox 拦截消息）
```

## 黄金规则：一次失败立即停手

**不允许**：
- ❌ 改换名字重试（`inspect_drug_column` 不行 → 试 `dump_drug_column` → 还是不行 → 试 `view_drug` ...）
- ❌ `ls /mnt/skills/custom/ethoinsight-code/references/` 找替代脚本
- ❌ `python -c "import pandas as pd; ..."`（被 sandbox 拦）
- ❌ 假设"应该存在某个工具脚本"

**必须**：
1. 立即停止重试
2. 把错误写入 `errors` 数组
3. 按现有 plan 已能完成的部分 seal handoff
4. 让 lead 看到 errors 决定下一步

## 实战案例（2026-05-28 thread 485a899d）

LLM 看到 plan.inputs.groups_file=null 但派遣 prompt 说"按 drug 列分组"，于是连续试了：

```
1. inspect_drug_column           # No module
2. dump_subject_metadata          # No module
3. ls -la /mnt/skills/...        # 找替代
4. ls /home/.../site-packages/   # 试图找源码
5. python -c "..."                # sandbox 拦截
6. find /mnt -name "*.py" ...    # sandbox 拦截
7. grep -r "drug" /mnt/skills/   # 没用
```

每次 LLM 调用 ~30s，叠加 6 次 = 3 分钟纯浪费 + 用户卡 loading。**正确做法**：第 1 次 `inspect_drug_column` 失败时立刻停。

## 失败后的 seal 模板

```python
seal_code_executor_handoff(
  status="partial",  # 或 "failed" 若主流程也挂
  summary="跑完 6 个 metric, 但无法按 drug 分组(plan.inputs.groups_file 为 null), 已按无分组聚合",
  paradigm="forced_swim",
  metrics_summary={
    "all": {  # 无分组兜底
      "immobility_time": {"mean": ..., "std": ..., "n": 2},
      ...
    }
  },
  per_subject={...},  # 完整保留
  data_quality_warnings=[
    {
      "severity": "warning",
      "code": "SAMPLE.GROUP_MISMATCH",
      "metric": "all",
      "evidence": {
        "groups_file": None,
        "expected_groups": ["treatment", "control"],
      },
      "message": "lead 派遣 prompt 提到 drug 分组,但 plan.inputs.groups_file 为 null,无法聚合分组统计。已按无分组兜底。",
      "blocks_downstream": False,
    }
  ],
  errors=[
    "groups_file is null but dispatcher prompt mentions drug grouping; need lead to call prep_metric_plan(groups={...}) explicitly"
  ],
  ...
)
```

注意：

- code 必须 **点分隔**（`SAMPLE.GROUP_MISMATCH` 不是 `_`），见 DataQualityWarning schema 报错时的明确指引
- `errors` 数组的消息要包含具体下一步指引，方便 lead 看到立即知道该 `prep_metric_plan(groups=...)` 重做

## 为什么不重试

| 重试动机 | 反驳 |
|---|---|
| "也许我记错名字了" | catalog YAML + plan_metrics.json 是脚本注册唯一权威。plan 列出哪些 script 就是哪些。 |
| "也许有 helper 脚本" | 没有。所有 ethoinsight.scripts.* 都是按指标命名的 compute_*/plot_*/run_*。没有 inspect_* / dump_* 这种通用工具。 |
| "也许能用 python -c 临时读" | sandbox 拦。这是设计选择（防 LLM 写脏代码污染管线），不是 bug。 |
| "应该自己探测一下原始数据" | 这正是 plan.inputs.groups_file 设计要解决的问题。lead 没传 = lead 没问清楚 = 你应该让 lead 看到 error 再问用户。 |

## 错误处理速查

| 错误现象 | 立即应对 |
|---|---|
| `No module named ethoinsight.scripts.<paradigm>.<name>` | seal failed/partial + errors |
| compute_* 报 stderr 但有产物 | 视为部分成功，按 plan 走，写 warning |
| compute_* 反复挂 | LoopDetectionMiddleware 会兜底；你不要主动重试超 1 次 |
| sandbox 拦截 `python -c` | seal failed/partial + errors（**不要**改用 python -m 调不存在脚本绕过） |

## 与 ethoinsight-code skill 的关系

ethoinsight-code skill 的 [error-recovery.md](../../ethoinsight-code/references/error-recovery.md) 讲的是 **脚本存在但运行失败**（脚本 bug / 数据格式不对 / column missing）的应对——可以重试 / 跳过 / 反问 lead。

本文件讲的是 **脚本不存在**（LLM 幻觉名字 / 试图调通用工具）的应对——**立刻停手，不重试**。

两者不冲突，按错误类型走不同路径。
