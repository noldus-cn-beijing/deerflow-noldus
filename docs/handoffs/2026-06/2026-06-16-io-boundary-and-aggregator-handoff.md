# 2026-06-16 统一 ethoinsight 脚本 I/O 边界 + 修聚合累加器 — 实施 handoff

> Spec：[docs/superpowers/specs/2026-06-16-io-boundary-symmetry-and-aggregator-spec.md](../../superpowers/specs/2026-06-16-io-boundary-symmetry-and-aggregator-spec.md)
> 分支：`feat/ethoinsight-io-boundary-and-aggregator`（worktree `/home/wangqiuyang/noldus-insight-io-boundary`，基线 origin/dev `4441ddda`）
> 实施者：Claude（本会话）

## 背景（spec §0）

2026-06-16 EPM dogfood 复跑（thread `158187ef`）：前序六 bug（#1–#6）全合后**首次走到 data-analyst 正常路径**，却暴露两个潜伏 bug，外加它们触发的 subagent 连锁失败（data-analyst / report-writer 各三连 `terminated without emitting handoff`）：

- **缺陷 1**（`read_inputs_json`/`read_groups_json` 不 resolve 自身 path 参数）→ statistics 读 `/mnt/.../inputs.json` FileNotFoundError → `statistics: {}`。
- **缺陷 2**（`metric_aggregation` 累加器半成品：n 对、mean 恒=首个 subject、std 恒=null）→ metrics_summary 与 per_subject 矛盾 → data-analyst 手算螺旋 → 漏调 seal。

治本判断：data-analyst 的 seal 失败**不是** prompt 不可靠——LLM 的怀疑**有道理**（上游数据真矛盾）。治本在 ethoinsight 层让数据自洽，不在 prompt 层教 LLM 忍耐矛盾。

## 实施内容（按 spec §5 顺序）

### 缺陷 2（聚合累加器）—— 先做

`packages/agent/backend/packages/harness/deerflow/subagents/metric_aggregation.py`：

1. 顶层加 `import statistics as _stats`。
2. 抽纯函数 `_compute_stat(values)`：忽略 None；mean=算术平均；std=样本标准差（`statistics.stdev`，n−1）；n<2 时 std=None；n=非 None 值个数。spec §2 末允许抽纯函数便于单测。
3. 循环内累加器改成临时 `_values` 列表（首个赋 `"_values":[value]`、后续 `append`），不再当场算 mean。
4. 循环结束后新增 finalize 步骤：遍历 `metrics_summary`，pop `_values`、用 `_compute_stat` 回填 mean/std/n。

语义锁定（spec §2 显式）：mean=算术平均忽略 None；std=样本标准差 n−1（与 scipy/pandas ddof=1 一致，行为学惯例）；n=applicable subject 数（None 不计）；保留 `parameters_used`。

### 缺陷 1a（read 函数 resolve path 参数）

`packages/ethoinsight/ethoinsight/scripts/_cli.py`：`read_inputs_json` 与 `read_groups_json` 入口 `Path(path)` → `Path(resolve_sandbox_path(path))`。与 `save_output_json`（#2 已修）在同一 I/O 边界对称。fail-safe 幂等，对真实路径调用方零变化。

### 缺陷 1b（groups 格式反转）

同文件 `read_groups_json`：旧 docstring 认为文件是 `{group:[files]}`，但 SSOT（`prep_metric_plan` 写、`metric_aggregation` 主读）是 `{file:group}`（flat map）。旧透传会把字符串组名当可迭代拆字符。修法：函数内反转 flat map 成 `{group:[files]}`（**派生视图，非双存**——守 SSOT 铁律，格式只存一份，写方/主读方都不动），同时兼容遗留 `{group:[files]}` 直通（用首个 value 类型判别）。

## 测试（全 importlib 加载 worktree 源，守 worktree 共享主仓 venv 假绿铁律）

| 层 | 文件 | 覆盖 | 结果 |
|---|---|---|---|
| C | `backend/tests/test_metric_aggregation_stats.py`（新） | `_compute_stat` 纯函数 4 + 聚合 5（mean/std/n 正确、多组不污染、跳 None、单值 std=None、per_subject 不破坏） | 9 passed |
| A | `ethoinsight/tests/test_cli_read_helpers_resolve.py`（新） | read 函数 resolve /mnt path（3） | 含在 B 文件 |
| B | 同上 | SSOT flat map 反转 + 遗留直通 + 空 dict + 非 dict 报错（5） | 8 passed（A+B 合计） |
| D | `ethoinsight/tests/scripts/test_epm_scripts.py::TestRunGroupwiseStats`（+2） | SSOT 格式端到端、/mnt 虚拟路径端到端 | 3 passed |
| E | ethoinsight 全量 | 854 passed, 70 skipped（spec 预期 844+，无回归） | ✅ |
| E | backend `test_run_metric_plan*` + 聚合 + import-cycle | 34 passed | ✅ |
| E | 裸导入 `import app.gateway` + `make_lead_agent` | 无环 | ✅ |

### red 锚点坐实（回退 origin/dev 状态验证，非假绿）

- **缺陷 2**：旧聚合器 `[2,4,6]` → `mean=2.0`（首个值）、`std=None`、`n=3` —— 与 spec §2 实测一致。
- **缺陷 1a**：`read_inputs_json('/mnt/user-data/workspace/inputs.json')` → `FileNotFoundError: '/mnt/user-data/workspace/inputs.json'`（虚拟路径直接读）—— 与 spec §1a 实测一致。
- **缺陷 1b**：同上根因（path 未 resolve）。

## 边界（spec §3，守根因隔离纪律——全部遵守）

- 不动 PR#135 `workspace_base` 兜底（缺陷 1 是"没调 resolve"，修法是补调用）。
- 不改 groups.json 文件格式（SSOT 是 `{file:group}`，只函数内反转）。
- 不动 data-analyst / report-writer prompt（seal 失败是上游矛盾连锁后果，上游修好即消）。
- 不删 `_scoped_path_env`、不改 plan 遗留 input 字段、不碰 compute 脚本。

## ⚠️ 范围外发现（供下 sprint 评估，本次不动）

实施层 D 时发现 **statistics 链存在 file→subject 标识鸿沟**（既有问题，非本 spec 引入、非本 spec 修复范围）：

- `prep_metric_plan` 写的 groups.json SSOT key 是**文件路径**（`{"/mnt/.../Trial 1.xlsx": "control"}`，prep_metric_plan_tool.py:98 注释铁律）。
- 但 `compute_paradigm_metrics`（dispatcher.py:168 `matched = [s for s in grp_subjects if s in per_subject]`）期望 groups value 是 **subject_name**（`parse_batch` 的 `subjects` key，来自 EV19 元数据"对象名称"，如 `"Subject 1"`）。
- 文件路径 ≠ subject_name → dispatcher `matched` 全空 → `comparisons` 空（即便缺陷 1a 修好后 statistics payload 能产出，`comparisons` 仍可能空）。

这解释了为何我的合成 fixture（subject 名 `"Subject N"` vs 文件路径 key）跑出 `comparisons={}`。**spec §5 验收期望 `comparisons` 含 EPM 5 指标**——这要求生产数据里 file→subject 能匹配，或在 read_groups_json 与 dispatcher 之间有未实现的映射。本次按 spec §3 边界**不处理**（不动 dispatcher / statistics 脚本 / prep）。层 D 测试断言精确到 spec red 锚点（rc=0 + payload 产出 `paradigm`/`comparisons` key 存在，不再 FileNotFoundError），未强求 `comparisons` 非空。**建议下 sprint 评估此鸿沟**（可能修法：statistics 脚本 read_groups_json 后把 file 路径转 stem/subject，或 prep 写 subject 名而非路径——需行为学同事确认 SSOT 语义）。

## 验收（spec §5，merge 后 dogfood 复跑 EPM）

- ✅（代码侧已就位）`run_metric_plan` 的 statistics payload 非空（不再 FileNotFoundError）。
- ✅（代码侧已就位）`metrics_summary[group][metric]` 的 mean/std 与 per_subject 重算一致。
- ⏳（待 dogfood）data-analyst 正常 seal、走到 report-writer 产出报告——需 merge 后复跑确认（注意上文 file→subject 鸿沟可能让 `comparisons` 仍空，需一并观察）。

## 改动清单

- `packages/agent/backend/packages/harness/deerflow/subagents/metric_aggregation.py`（+45：`_compute_stat` 纯函数 + 累加器 `_values` + finalize）
- `packages/ethoinsight/ethoinsight/scripts/_cli.py`（+53/-16：两 read 函数 path resolve + groups SSOT 反转）
- `packages/ethoinsight/tests/scripts/test_epm_scripts.py`（+90：层 D 两端到端测试）
- `packages/agent/backend/tests/test_metric_aggregation_stats.py`（新，层 C）
- `packages/ethoinsight/tests/test_cli_read_helpers_resolve.py`（新，层 A+B）

> 注：worktree 新建 venv 时 uv 自动改了 `packages/agent/backend/uv.lock`（环境副作用），已 `git checkout` 丢弃，不进 PR。
