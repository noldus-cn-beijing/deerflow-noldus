# 6 范式 SOTA 架构切换 实施 Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **🎯 本文件正式路径**: `docs/superpowers/plans/2026-05-11-paradigm-sota-architecture-plan.md`
> 当前临时位于 `~/.claude/plans/` 仅因 plan-mode 限制；执行 agent 接手第一步把本文件 `mv` 到正式路径。

**Goal:** 把 ethoinsight 从「5 个 langchain 工具串行 + shoaling.py 脚本模板复制」的旧架构，切换到「指标级 Python 函数 + LLM 写胶水代码 + skill 渐进披露」的 SOTA 架构，先用 EPM 跑通端到端，再依次完成 OFT / Zero Maze / LDB / TST / FST 5 个范式。

**Architecture:** code-executor 通过 `bash` 写胶水脚本 `import` `ethoinsight.metrics.<范式>` 模块里的纯函数 → 跑统计 → 出图 → 写 handoff JSON。代码不再用 langchain @tool 包装。reliability 由 deerflow 现成 middleware 兜底（`tool_error_handling` / `loop_detection` / `summarization` / `view_image` / `subagent_limit`）。

**Tech Stack:** Python 3.10+ (ethoinsight) / 3.12+ (agent backend), pytest, pandas, numpy, LangGraph + deerflow harness, skill markdown frontmatter + references 渐进披露。

---

## 当前阶段

> **多 agent 接力主进度板。** 每个 task 完成后由执行 agent 把对应 checkbox 勾上 + 在「当前 Phase」字段更新到下一个 task。

```yaml
current_phase: "Phase 5 - 同事 review checklist（异步并行）"
current_task: "T1 - 建 review 包结构（待启动）"
last_completed: "Phase 4 完成（commit a25cf3ba）：10 文件删除（7 langchain 工具 + shoaling/_gate/tool 模板 + 4 旧测试），data_analyst/report_writer/knowledge_assistant 清理 disallowed_tools"
blockers: []
notes: "SOTA 架构迁移主体工程完成（Phase 1-4）。Phase 5 是异步 review，不阻塞工程。ethoinsight: 182 passed/9 pre-existing fail；agent backend: 2168 passed/14 pre-existing fail。"
```

---

## 全景图（5 Phase）

```
Phase 1: EPM 端到端打通（最小通路）        ← ✅ 完成（2026-05-11）
  ↓ 验收：EPM 单测全过 + code-executor 用胶水脚本跑通 demo EPM 数据 + handoff JSON 落地
Phase 2: OFT 端到端验证 SOTA 架构        ← ✅ 完成（2026-05-11）
  ↓ 验收：OFT 单测全过 + 第二个范式用同套机制跑通（证明架构通用，不是 EPM hack）
Phase 3: Zero Maze / LDB / TST / FST 批量        ← ✅ 完成（2026-05-11）
  ↓ 验收：4 个范式各自单测全过 + Zero Maze e2e 通过
Phase 4: 旧工具清理        ← 我们在这里
  ↓ 验收：删除 7 个废弃 langchain 工具 + templates/shoaling.py + templates/_gate.py + 测试全绿
Phase 5: 同事 review checklist（异步并行，不阻塞 1-4）
  ↓ 验收：review 包结构落地 + 行为学同事 PR 通过 + 工程衍生 by-paradigm/<范式>.md 与同事 by-experiment/<范式>.md 校对一致
```

**关键并行关系：** Phase 5 在 Phase 1 后就可以异步启动（同事 review 不阻塞工程），但同事 PR 回来时如果 Phase 1-3 已 freeze 了 SOTA 架构，PR 合并就只是补齐文档。Phase 4 必须等 Phase 1-3 完成（否则会删掉还在用的工具）。

---

## 核心架构决策记录（grill-me 沉淀，不可违反）

### 11 条架构决策

1. **指标实现路径** = 「指标级函数 + LLM 写胶水代码」（A 路径）。否决 B 路径（按整范式包成 1 个 langchain 工具）和 C 路径（dispatcher 模板 templates/<范式>.py 脚本）。
2. **指标颗粒度** = 按学术「指标公式」切。每个指标 = 1 个有具名的纯函数（如 `compute_open_arm_time(df, open_arm_zones=None) -> float | None`）。不是按「数据流」切（不是 1 个大的 `analyze_epm` 函数）。
3. **指标物理位置** = `packages/ethoinsight/ethoinsight/metrics/<学术范式>.py`。**按学术范式分文件**（不按 EV19 模板大类）。原因：内部分析路径走学术范式（EPM / OFT / Zero Maze / LDB / TST / FST / Shoaling）；EV19 模板是用户语言层。
4. **code-executor 调用方式** = 通过 `bash` 写胶水脚本 `import` 指标函数 + 调用。**不**调 langchain 工具，**不**调 dispatcher。胶水脚本由 LLM 现场组织（read skill 了解函数清单 → write_file 写脚本 → bash 跑）。
5. **废弃的 langchain 工具**（7 个，Phase 4 删）：`parse_trajectories`、`compute_metrics`、`run_statistics`、`generate_charts`、`assess_and_handoff`、`run_paradigm_analysis`、`get_analysis_template`。
6. **图表生成职责** = code-executor 出图。**不**单独建 chart-maker subagent。`ethoinsight-charts` skill 移入 code-executor 白名单。
7. **paradigm-knowledge skill** = 不进 code-executor 白名单。code-executor **自包含**所有「函数清单 + 调用范例 + handoff schema」信息（在新 `ethoinsight-code` skill 的 `by-paradigm/<范式>.md`）。`ethovision-paradigm-knowledge` 给 lead / data-analyst / report-writer 用。
8. **ethoinsight-code skill** = 工程师维护的执行手册。SKILL.md 顶层是「按 paradigm 渐进披露」入口；`by-paradigm/<范式>.md` 是某范式的完整 API 视角。
9. **行为学决策点处理** = raw data 已经在导出阶段固化阈值（Activity_State / In zone 列），工程师写函数时只需 read 前几行确认列名，**不**要求行为学同事补阈值。
10. **穿梭次数定义** = 单向（明 → 暗 / 暗 → 明 任一跨越算 1 次），不是双向。
11. **数据可靠性兜底** = 利用 deerflow 5 个现成 middleware（`tool_error_handling` / `loop_detection` / `summarization` / `view_image` / `subagent_limit`），**不**新建中间件。

### 6 个 custom skill 命运

| Skill | 命运 | 主要操作 |
|---|---|---|
| `compaction-recovery` | 保留不动 | — |
| `ethoinsight` | 保留 + 重定位 | 改 description「给 data-analyst + report-writer 看判读方法论」；`paradigm-interpretation.md` 改指针文件指向 `ethovision-paradigm-knowledge` |
| `ethoinsight-code` | **重命名为 `ethoinsight-code` + 大改** | 目录 mv；删 5 工具流程描述；改写为「按 paradigm 渐进披露 + 胶水脚本范例 + handoff schema」 |
| `ethoinsight-charts` | 保留 + 移入 code-executor 白名单 + description 改为给 code-executor 用 | 调 description（不动 references） |
| `ethoinsight-planning` | 保留不动 | — |
| `ethovision-paradigm-knowledge` | 保留不动（5 月 9 日刚改过） | — |

### subagent 白名单矩阵

```python
# code_executor.py: 切换前 → 切换后
skills=["ethoinsight-code"]   →   skills=["ethoinsight-code", "ethoinsight-charts"]

# data-analyst / report-writer / knowledge-assistant:
skills 参数不存在  →  保持不存在（None = 默认全继承）
```

### code-executor 真实职责（SOTA 修正后）

```
输入: lead 派遣 task() prompt（paradigm + ev19_template + groups + workspace_path + 用户特殊需求）
工作:
  1. read ethoinsight-code/by-paradigm/<范式>.md → 知道该范式可用的指标函数清单 + 调用范例 + handoff schema
  2. read ethoinsight-charts skill → 按数据特性选图
  3. write_file 写胶水脚本（import ethoinsight.metrics.<范式> + 算指标 + 跑统计 + 出图 + 写 handoff_code_executor.json）
  4. bash 执行胶水脚本
  5. tool_error_handling middleware 自动给 traceback → LLM 改代码重跑（用户接受 1-2 次重试）
输出: handoff_code_executor.json + chart PNG 文件（落 workspace + outputs 目录）
```

### 反模式（永远禁止）

1. ❌ 复制 `templates/shoaling.py` 脚本模板模式写新范式
2. ❌ 给 code-executor 加 langchain @tool 工具
3. ❌ 让 lead 在 task() prompt 里抽 by-experiment 段拷进给 code-executor（退化成单 agent）
4. ❌ 让 LLM 现场写指标算法（如 `compute_immobility_latency` 的 run-length encoding 必须工程师预制）
5. ❌ 强制行为学同事补阈值（已在 raw data 列里固化）
6. ❌ 把 by-experiment / by-template 物理切分到两个 skill（同一份内容走两份引用）

---

## 前置条件（执行 plan 前确认）

- [x] **0.1** 把本文件从 `~/.claude/plans/skill-agent-plan-home-wangqiuyang-noldu-adaptive-cook.md` mv 到正式路径 `docs/superpowers/plans/2026-05-11-paradigm-sota-architecture-plan.md`

```bash
mv ~/.claude/plans/skill-agent-plan-home-wangqiuyang-noldu-adaptive-cook.md \
   /home/wangqiuyang/noldus-insight/docs/superpowers/plans/2026-05-11-paradigm-sota-architecture-plan.md
cd /home/wangqiuyang/noldus-insight
git add docs/superpowers/plans/2026-05-11-paradigm-sota-architecture-plan.md
git commit -m "plan: 6 范式 SOTA 架构切换实施 plan"
```

- [x] **0.2** 确认工作目录干净 + 在 `dev` 分支

```bash
cd /home/wangqiuyang/noldus-insight
git status --short
# 期望: 只有 docs/sop/aliyun-deployment-guide.md untracked（无关），其它都干净
git branch --show-current
# 期望: dev
```

- [x] **0.3** 读完前置文档（不读不动手）

```bash
cat docs/handoffs/2026-05-11-paradigm-sota-architecture-grill-handoff.md  # 本 plan 的上下文
cat docs/handoffs/2026-05-09-mvp-paradigm-knowledge-into-skill-handoff-v2.md  # grill-me 前的状态
cat packages/agent/skills/custom/ethovision-paradigm-knowledge/SKILL.md  # 5 月 9 日修订后的最终形态
grep -n "^def" packages/ethoinsight/ethoinsight/metrics.py  # 当前 14 个函数 + 5 helper
```

- [x] **0.4** 启动 superpowers:subagent-driven-development 或 superpowers:executing-plans 跑这个 plan

---

## Phase 1: EPM 端到端打通（最小通路）

**目标:** 用 EPM 当试金石跑通整套 SOTA 流程。完成时：metrics.py 已拆为 metrics/ 子模块、ethoinsight-code 已改名为 ethoinsight-code、code-executor 已切换白名单、demo EPM 数据用胶水脚本能跑出 handoff JSON + 至少 1 张 PNG 图。

**关键资产（已就位，不重写）:**
- `packages/ethoinsight/ethoinsight/metrics.py:301-444` — EPM 5 个函数 + 5 helper（commit `16eeac9b`）
- `packages/ethoinsight/tests/test_metrics_epm.py` — 20 个 EPM 测试（5 个 TestCompute* 类 + 1 个 TestComputeParadigmMetricsEpm）
- `docs/review-packages/2026-04-29-ev19-templates/by-experiment/epm.md` — 行为学同事维护的 47 行 EPM 判读文档（不动）

### Task 1: 建 metrics/ 子模块目录骨架 + 迁 3 范式

**Files:**
- Create: `packages/ethoinsight/ethoinsight/metrics/__init__.py`
- Create: `packages/ethoinsight/ethoinsight/metrics/_common.py`
- Create: `packages/ethoinsight/ethoinsight/metrics/shoaling.py`
- Create: `packages/ethoinsight/ethoinsight/metrics/oft.py`
- Create: `packages/ethoinsight/ethoinsight/metrics/epm.py`
- Create: `packages/ethoinsight/ethoinsight/metrics/dispatcher.py`
- Modify: `packages/ethoinsight/ethoinsight/metrics.py:1-670` — **由 670 行模块文件变为 ≤15 行 shim**（仅 `from .metrics.* import *` 转发，保 `from ethoinsight.metrics import compute_open_arm_time` 等历史导入路径继续工作）

> **拆分映射表（搬运不改算法）:**
>
> | 来源 metrics.py 行号 | 函数 | 去向文件 |
> |---|---|---|
> | 23-94 | `compute_distance_moved`, `compute_velocity_stats` | `_common.py` |
> | 54 | `_align_subjects_xy` (helper) | `_common.py` |
> | 96-225 | `compute_inter_individual_distance`, `compute_nearest_neighbor_distance`, `compute_group_polarity` | `shoaling.py` |
> | 227 | `_find_zone_column` (helper) | `_common.py`（被 oft / epm 共用） |
> | 236-300 | `compute_center_time_ratio`, `compute_thigmotaxis_index` | `oft.py` |
> | 301-444 | `compute_open_arm_time_ratio`, `compute_open_arm_entry_count`, `_entry_ratio`, `_open_arm_time`, `compute_total_entry_count` | `epm.py` |
> | 325-373 | `_count_zone_entries`, `_find_arm_zone_columns`, `_get_frame_duration` (helpers) | `epm.py`（EPM 专用） |
> | 446-638 | `compute_paradigm_metrics(parsed_data, paradigm, ...)` dispatcher | `dispatcher.py` |
> | 646-670 | `save_to_csv` | `_common.py`（导出工具，范式无关） |

- [ ] **Step 1.1: 看现状确认拆分映射**

```bash
cd /home/wangqiuyang/noldus-insight
grep -n "^def\|^_" packages/ethoinsight/ethoinsight/metrics.py | head -40
wc -l packages/ethoinsight/ethoinsight/metrics.py
# 预期: 670 行；14 个 def + 5 个 _helper（与映射表对齐）
```

- [ ] **Step 1.2: 建目录 + `__init__.py`**

```bash
mkdir -p packages/ethoinsight/ethoinsight/metrics
```

写 `packages/ethoinsight/ethoinsight/metrics/__init__.py`：

```python
"""按学术范式组织的指标函数子模块。

物理路径:
  - _common.py: 范式无关（distance / velocity / zone 列查找 / csv 导出）
  - shoaling.py: 群体游动指标
  - oft.py: Open Field 指标
  - epm.py: 高架十字迷宫指标
  - dispatcher.py: compute_paradigm_metrics() 派发入口

代码外部仍可 `from ethoinsight.metrics import compute_open_arm_time` 等历史导入路径
（通过顶层 ethoinsight/metrics.py shim 转发）。新代码推荐显式导入：
  from ethoinsight.metrics.epm import compute_open_arm_time
"""

from ethoinsight.metrics._common import (
    compute_distance_moved,
    compute_velocity_stats,
    save_to_csv,
)
from ethoinsight.metrics.shoaling import (
    compute_inter_individual_distance,
    compute_nearest_neighbor_distance,
    compute_group_polarity,
)
from ethoinsight.metrics.oft import (
    compute_center_time_ratio,
    compute_thigmotaxis_index,
)
from ethoinsight.metrics.epm import (
    compute_open_arm_time_ratio,
    compute_open_arm_entry_count,
    compute_open_arm_entry_ratio,
    compute_open_arm_time,
    compute_total_entry_count,
)
from ethoinsight.metrics.dispatcher import compute_paradigm_metrics

__all__ = [
    "compute_distance_moved",
    "compute_velocity_stats",
    "save_to_csv",
    "compute_inter_individual_distance",
    "compute_nearest_neighbor_distance",
    "compute_group_polarity",
    "compute_center_time_ratio",
    "compute_thigmotaxis_index",
    "compute_open_arm_time_ratio",
    "compute_open_arm_entry_count",
    "compute_open_arm_entry_ratio",
    "compute_open_arm_time",
    "compute_total_entry_count",
    "compute_paradigm_metrics",
]
```

- [ ] **Step 1.3: 写 `_common.py`（把 metrics.py 行 23-94 + 227 + 646-670 搬过来）**

把原 `metrics.py` 里这些范围的代码**整段复制**到 `packages/ethoinsight/ethoinsight/metrics/_common.py`：
- 行 23-30: `compute_distance_moved`
- 行 30-53: `compute_velocity_stats`
- 行 54-94: `_align_subjects_xy` (helper)
- 行 227-235: `_find_zone_column` (helper)
- 行 646-670: `save_to_csv`

文件头加：

```python
"""范式无关的指标函数 + 共享 helper。"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd
```

不改算法实现，只是搬位置。如果原代码导入了模块顶部的 import，需要把对应 import 也复制过来。

- [ ] **Step 1.4: 写 `shoaling.py`（把 metrics.py 行 96-225 搬过来）**

```python
"""Shoaling 范式指标：群体游动相关。

涉及指标：mean_iid、mean_nnd、polarity（群体方向一致性）。
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from ethoinsight.metrics._common import _align_subjects_xy

# 把 metrics.py:96-225 的 3 个 compute_* 函数原样搬过来
```

- [ ] **Step 1.5: 写 `oft.py`（把 metrics.py 行 236-300 搬过来）**

```python
"""Open Field Test 范式指标：中心区滞留 + 趋触性。"""

from __future__ import annotations
import re
import numpy as np
import pandas as pd
from ethoinsight.metrics._common import _find_zone_column

# 搬 compute_center_time_ratio + compute_thigmotaxis_index
```

- [ ] **Step 1.6: 写 `epm.py`（把 metrics.py 行 301-444 + 5 helper 搬过来）**

```python
"""Elevated Plus Maze 范式指标：开臂滞留 + 开臂进入 + 总进臂次数。

数据质量风险（详见 docs/review-packages/2026-04-29-ev19-templates/by-experiment/epm.md）:
  - n < 5/组: 统计功效不足
  - total_entry_count < 8: 可能为运动抑制（混杂因素）
"""

from __future__ import annotations
import re
import numpy as np
import pandas as pd

# 搬 5 个 compute_* + 3 个 _helper (_count_zone_entries, _find_arm_zone_columns, _get_frame_duration)
```

- [ ] **Step 1.7: 写 `dispatcher.py`（把 metrics.py 行 446-638 搬过来）**

```python
"""compute_paradigm_metrics 派发器：按 paradigm 路由到 metrics/<范式>.py 的函数。

新增 paradigm 时（如 Zero Maze），加一个 elif 分支。
"""

from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd

from ethoinsight.metrics._common import compute_distance_moved, compute_velocity_stats
from ethoinsight.metrics.shoaling import (
    compute_inter_individual_distance,
    compute_nearest_neighbor_distance,
    compute_group_polarity,
)
from ethoinsight.metrics.oft import compute_center_time_ratio, compute_thigmotaxis_index
from ethoinsight.metrics.epm import (
    compute_open_arm_time_ratio,
    compute_open_arm_entry_count,
    compute_open_arm_entry_ratio,
    compute_open_arm_time,
    compute_total_entry_count,
)

# 整段搬 metrics.py:446-638 的 compute_paradigm_metrics 实现
```

- [ ] **Step 1.8: 把原 `metrics.py` 改写为 shim**

```python
"""向后兼容 shim：保旧导入路径 `from ethoinsight.metrics import compute_*` 工作。

实际实现在 `ethoinsight/metrics/<范式>.py`，通过 `metrics/__init__.py` 重导出。

⚠️ 新代码推荐: from ethoinsight.metrics.epm import compute_open_arm_time
"""

# Python 解析时优先 metrics/ 包 over metrics.py（因为同名 package 比 module 优先）
# 所以这个 shim 其实不会被加载，但保留作为文档锚点。
# 真正的转发由 metrics/__init__.py 完成。
```

> **重要技术细节:** Python 中 `metrics/` 子包优先级高于 `metrics.py` 同名模块。当 `metrics/` 目录存在且有 `__init__.py` 时，`import ethoinsight.metrics` 加载的是子包，旧 `metrics.py` 被忽略。所以 Step 1.8 只是保留一个文档锚点；真正的导入兼容由 Step 1.2 的 `metrics/__init__.py` 重导出实现。**Step 1.8 完成后立刻 Step 1.9 跑测试验证导入路径还工作。**

- [ ] **Step 1.9: 跑现有测试，验证拆分没破任何东西**

```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
python -m pytest tests/test_metrics.py tests/test_metrics_epm.py -v
# 预期: 31 + 20 = 51 个测试全过
# 如果 test_parse.py 有 pre-existing 失败（9 个），不影响。
```

- [ ] **Step 1.10: 跑 ethoinsight 全套测试 + agent backend 测试，确认无新增 fail**

```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
python -m pytest tests/ -v 2>&1 | tail -40
# 预期: 只能看到 pre-existing 9 个 test_parse 失败，无新增。

cd /home/wangqiuyang/noldus-insight/packages/agent/backend
source .venv/bin/activate
make test 2>&1 | tail -20
# 预期: 全绿（不该被 ethoinsight 拆分影响）
```

- [ ] **Step 1.11: commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/ethoinsight/ethoinsight/metrics/ packages/ethoinsight/ethoinsight/metrics.py
git commit -m "refactor(ethoinsight): metrics.py 拆为 metrics/<范式>.py 子模块（SOTA 架构 Phase 1 T1）

- 新建 metrics/_common.py: 范式无关函数 + 共享 helper
- 新建 metrics/shoaling.py / oft.py / epm.py: 学术范式按指标公式切函数
- 新建 metrics/dispatcher.py: compute_paradigm_metrics() 派发入口
- metrics.py 留作 shim（实际由 metrics/ 子包优先加载）
- 历史导入 from ethoinsight.metrics import compute_* 继续工作"
```

### Task 2: 检查 EPM 指标完整性

**目的:** 跟同事 review 包 `docs/review-packages/2026-04-29-ev19-templates/by-experiment/epm.md` 对照，确认行为学需要的所有 EPM 指标都已经在 `metrics/epm.py` 里。

**Files:**
- Read: `docs/review-packages/2026-04-29-ev19-templates/by-experiment/epm.md`
- Read: `packages/ethoinsight/ethoinsight/metrics/epm.py`
- 可能 Modify: `packages/ethoinsight/ethoinsight/metrics/epm.py`（如发现缺指标）
- 可能 Modify: `packages/ethoinsight/tests/test_metrics_epm.py`（如新增函数要新增测试）

- [ ] **Step 2.1: 读同事的 EPM 行为学需求**

```bash
cat docs/review-packages/2026-04-29-ev19-templates/by-experiment/epm.md
```

把里面所有「核心指标」清单列出来。

- [ ] **Step 2.2: 与 `metrics/epm.py` 现有 5 函数对比**

现有 5 个: `compute_open_arm_time_ratio`、`compute_open_arm_entry_count`、`compute_open_arm_entry_ratio`、`compute_open_arm_time`、`compute_total_entry_count`。

判断同事提到的每个指标是否已有对应函数。

- [ ] **Step 2.3: 如果缺指标，新增函数 + 测试（TDD）**

对每个缺的指标：
1. 先在 `tests/test_metrics_epm.py` 加 `test_compute_<指标>_*` 测试（合成 DataFrame + 已知答案）
2. 跑测试确认 FAIL
3. 在 `metrics/epm.py` 实现函数
4. 跑测试确认 PASS
5. 在 `metrics/__init__.py` + `dispatcher.py` 加导出 + 派发分支

> **如果第 2.2 步发现 5 函数已经覆盖所有需求，跳过 2.3 + 2.4 直接 Step 2.5。**

- [ ] **Step 2.4: 跑测试**

```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
python -m pytest tests/test_metrics_epm.py -v
```

- [ ] **Step 2.5: commit（仅当 2.3 改了代码）**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/ethoinsight/ethoinsight/metrics/epm.py packages/ethoinsight/tests/test_metrics_epm.py
git commit -m "feat(ethoinsight): 补 EPM 缺失指标 <名称> + 单测（Phase 1 T2）"
```

### Task 3: 重命名 skill 目录 ethoinsight-code → ethoinsight-code

**Files:**
- Move: `packages/agent/skills/custom/ethoinsight-code/` → `packages/agent/skills/custom/ethoinsight-code/`
- Modify: `packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py:91` — `skills=["ethoinsight-code"]` → `skills=["ethoinsight-code", "ethoinsight-charts"]`

> **注意:** 这一步只是改名 + 改引用。SKILL.md 内容大改放到 Task 4。

- [ ] **Step 3.1: 看现有 skill 引用点**

```bash
cd /home/wangqiuyang/noldus-insight
grep -rn "ethoinsight-code" --include="*.py" --include="*.md" --include="*.json" --include="*.yaml" \
  packages/agent/ docs/ 2>/dev/null
```

记下所有结果，下面要全部改。

- [ ] **Step 3.2: git mv 目录**

```bash
cd /home/wangqiuyang/noldus-insight
git mv packages/agent/skills/custom/ethoinsight-code \
       packages/agent/skills/custom/ethoinsight-code
```

- [ ] **Step 3.3: 改 code_executor.py 白名单**

文件 `packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py` 第 91 行：

```python
# 旧:
skills=["ethoinsight-code"],

# 新:
skills=["ethoinsight-code", "ethoinsight-charts"],
```

- [ ] **Step 3.4: 把 Step 3.1 grep 结果里其它所有 `ethoinsight-code` 引用全改为 `ethoinsight-code`**

逐个文件 sed 或手动改。重点检查的位置：
- `packages/agent/extensions_config.json`（如果有）
- `docs/` 下任何提到 `ethoinsight-code` 的文档
- `packages/agent/skills/custom/ethoinsight-code/SKILL.md` 自己（如果有自引用）

- [ ] **Step 3.5: 跑测试 + 启服务冒烟**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
source .venv/bin/activate
make test 2>&1 | tail -10
# 预期: 全绿

# 启服务确认 code-executor subagent 能正常注册
make dev &
sleep 10
curl -s http://localhost:2024/ok && echo "alive"
make stop
```

- [ ] **Step 3.6: commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add -A
git commit -m "refactor: skill ethoinsight-code → ethoinsight-code + code-executor 白名单加 ethoinsight-charts（Phase 1 T3）"
```

### Task 4: 大改 ethoinsight-code SKILL.md + 写 by-paradigm/epm.md

**目的:** 把旧的「5 工具串行流程」描述彻底删掉，改写成 SOTA 架构的「按 paradigm 渐进披露 + 胶水脚本范例 + handoff schema」。

**Files:**
- Rewrite: `packages/agent/skills/custom/ethoinsight-code/SKILL.md`（从 87 行重写）
- Create: `packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/epm.md`
- Delete: `packages/agent/skills/custom/ethoinsight-code/references/run-paradigm-analysis-api.md`（API 已废弃）
- Delete: `packages/agent/skills/custom/ethoinsight-code/references/tool-reference.md`（5 工具描述已废弃）
- Delete: `packages/agent/skills/custom/ethoinsight-code/references/fallback-workflow.md`（旧 fallback 已废弃）
- Delete: `packages/agent/skills/custom/ethoinsight-code/references/shoaling-paradigm.md`（被 `by-paradigm/shoaling.md` 替代，Phase 2 写）
- 保留: `packages/agent/skills/custom/ethoinsight-code/references/error-recovery.md`、`quality-checks.md`（仍适用）
- 保留: `packages/agent/skills/custom/ethoinsight-code/templates/output-contract.md`（handoff JSON schema 仍适用）

- [ ] **Step 4.1: 看现有 SKILL.md 结构（确认要删 / 改 / 保留的部分）**

```bash
cat packages/agent/skills/custom/ethoinsight-code/SKILL.md
ls packages/agent/skills/custom/ethoinsight-code/references/
```

- [ ] **Step 4.2: 写新 SKILL.md（全文替换）**

文件 `packages/agent/skills/custom/ethoinsight-code/SKILL.md`：

```markdown
---
name: ethoinsight-code
description: >
  EthoInsight 数据分析执行手册（给 code-executor subagent 用）。
  按 paradigm 渐进披露指标函数清单、胶水脚本范例、handoff JSON schema。
  Use when code-executor receives a paradigm-specific analysis task.
type: workflow
---

# EthoInsight 代码执行指南

## 工作模式

你（code-executor）拿到 lead 派遣的任务后:

1. **read** `references/by-paradigm/<paradigm>.md` — 看本范式可用的指标函数清单 + 调用范例 + handoff schema
2. **read** ethoinsight-charts skill（已在你的白名单） — 按数据特性选图
3. **write_file** 写胶水脚本 `analysis.py`（在 `${workspace_path}/` 下）— `import ethoinsight.metrics.<范式>` + 算指标 + 跑统计 + 出图 + 写 `handoff_code_executor.json`
4. **bash** `python ${workspace_path}/analysis.py`
5. 出错时按 `references/error-recovery.md` 处理；脚本崩溃 traceback 会自动回到你的 context，改代码重跑（最多 2 次）

## 范式渐进披露入口

- **EPM** (高架十字迷宫): `references/by-paradigm/epm.md`
- **OFT** (Open Field): `references/by-paradigm/oft.md` *(Phase 2 时撰写)*
- **Shoaling** (群体游动): `references/by-paradigm/shoaling.md` *(Phase 2 时撰写)*
- **Zero Maze**: `references/by-paradigm/zero-maze.md` *(Phase 3 时撰写)*
- **LDB** (Light-Dark Box): `references/by-paradigm/ldb.md` *(Phase 3 时撰写)*
- **TST** (Tail Suspension): `references/by-paradigm/tst.md` *(Phase 3 时撰写)*
- **FST** (Forced Swim): `references/by-paradigm/fst.md` *(Phase 3 时撰写)*

## 通用资源

- `references/error-recovery.md` — 常见错误诊断 + 重试策略
- `references/quality-checks.md` — 数据质量自检清单（NaN / 单位 / 列名缺失）
- `templates/output-contract.md` — handoff JSON 字段规范

## 反模式（永远禁止）

1. ❌ 跑 `parse_trajectories` / `compute_metrics` / `run_statistics` 等老 langchain 工具（已废弃）
2. ❌ 把整段范式分析包装成 1 个 `analyze_<paradigm>()` 函数（颗粒度错）
3. ❌ 现场实现指标算法（如 run-length encoding）— 工程师已预制在 `ethoinsight.metrics.<范式>`
4. ❌ 读 EthoVision raw txt 前几行确认列名 — 函数内部已固化列名识别（regex 匹配）
```

- [ ] **Step 4.3: 删过时 references**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/skills/custom/ethoinsight-code/references
git rm run-paradigm-analysis-api.md tool-reference.md fallback-workflow.md shoaling-paradigm.md
```

- [ ] **Step 4.4: 建 `by-paradigm/` 目录 + 写 epm.md**

```bash
mkdir -p packages/agent/skills/custom/ethoinsight-code/references/by-paradigm
```

文件 `packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/epm.md`：

```markdown
# EPM (Elevated Plus Maze) 代码执行参考

> 学术范式 key: `epm`
> EV19 模板映射: `Elevated Plus Maze` 大类下所有变体
> 行为学同事维护的领域知识: `docs/review-packages/2026-04-29-ev19-templates/by-experiment/epm.md`

## 可用指标函数（在 `ethoinsight.metrics.epm`）

| 函数 | 输入 | 输出 | 含义 |
|---|---|---|---|
| `compute_open_arm_time_ratio(df, open_arm_zones=None, closed_arm_zones=None)` | DataFrame | `float \| None` | 开臂时间占比（开臂帧数 / (开臂+闭臂)帧数） |
| `compute_open_arm_entry_count(df, open_arm_zones=None)` | DataFrame | `int \| None` | 开臂进入次数（0→1 跨越，首帧为 1 也算 1 次） |
| `compute_open_arm_entry_ratio(df, open_arm_zones=None)` | DataFrame | `float \| None` | 开臂进入次数 / 总进臂次数 |
| `compute_open_arm_time(df, open_arm_zones=None)` | DataFrame | `float \| None` | 开臂总停留时间（秒；从 trial_time 列估算 dt） |
| `compute_total_entry_count(df)` | DataFrame | `int \| None` | 总进臂次数（开臂进入 + 闭臂进入，各组内 OR 合并） |

通用指标（任范式可用）:
- `ethoinsight.metrics._common.compute_distance_moved(df) -> float | None`
- `ethoinsight.metrics._common.compute_velocity_stats(df) -> dict`

## 派发器（一次性算全套）

```python
from ethoinsight.metrics import compute_paradigm_metrics

result = compute_paradigm_metrics(parsed_data, paradigm="epm", groups=groups)
# result = {
#   "paradigm": "epm",
#   "per_subject": {subject_name: {open_arm_time_ratio, open_arm_entry_count, ...}},
#   "group_summary": {group_name: {metric: {mean, std, n, values}}},
#   "data_quality_warnings": [...],  # n < 5/组 或 total_entry_count < 8 自动警告
#   ...
# }
```

## 胶水脚本范例（end-to-end）

文件名建议: `${workspace_path}/analysis.py`

```python
"""EPM 分析胶水脚本。

由 code-executor 写，bash 执行。
"""
import json
from pathlib import Path

from ethoinsight import parse, statistics, charts
from ethoinsight.metrics import compute_paradigm_metrics

WORKSPACE = Path("/mnt/user-data/workspace")
RAW_FILES = list(WORKSPACE.glob("inputs/*.txt"))  # 用户上传的 EthoVision 导出
GROUPS = {  # 由 lead 在 task() prompt 里给
    "control": ["subject_1", "subject_2", "subject_3", "subject_4", "subject_5"],
    "treatment": ["subject_6", "subject_7", "subject_8", "subject_9", "subject_10"],
}

# 1. 解析（仍用现成 parse 模块）
parsed_data = parse.parse_trajectories([str(f) for f in RAW_FILES])

# 2. 算指标
metrics_result = compute_paradigm_metrics(parsed_data, paradigm="epm", groups=GROUPS)

# 3. 统计（统计模块仍是 Phase 1 范围外，沿用 ethoinsight.statistics）
stats_result = statistics.run_groupwise(
    metrics_result["per_subject"],
    groups=GROUPS,
    metrics=["open_arm_time_ratio", "open_arm_entry_count", "open_arm_time", "total_entry_count"],
)

# 4. 出图（read ethoinsight-charts skill 后选 box_plot / raincloud_plot 等）
chart_files = []
for metric_name in ["open_arm_time_ratio", "open_arm_entry_count"]:
    fig_path = WORKSPACE / "outputs" / f"epm_{metric_name}_boxplot.png"
    charts.box_plot(
        metrics_result["per_subject"],
        groups=GROUPS,
        metric=metric_name,
        output_path=str(fig_path),
    )
    chart_files.append(str(fig_path))

# 5. 写 handoff
handoff = {
    "paradigm": "epm",
    "metrics": metrics_result,
    "statistics": stats_result,
    "charts": chart_files,
    "data_quality_warnings": metrics_result.get("data_quality_warnings", []),
}
(WORKSPACE / "handoff_code_executor.json").write_text(
    json.dumps(handoff, ensure_ascii=False, indent=2)
)
print(f"OK: handoff written to {WORKSPACE / 'handoff_code_executor.json'}")
```

## 数据质量自动警告（dispatcher 内已实现）

- `n < 5/组` → `warning` 级警告，提示统计功效不足
- `total_entry_count < 8` → `warning` 级警告，提示可能为运动抑制（混杂因素）

这些警告会自动进 `metrics_result["data_quality_warnings"]`，不需要胶水脚本额外加。

## 出图建议（详见 ethoinsight-charts skill）

- `open_arm_time_ratio` / `open_arm_entry_ratio`: box_plot 或 raincloud_plot（连续 + 组比较）
- `open_arm_entry_count` / `total_entry_count`: bar_chart 或 box_plot
- 时间序列（如分钟 bin 化的开臂滞留率）: 留给后续扩展

## handoff JSON 必须字段

见 `../templates/output-contract.md`。EPM 特别需要 `data_quality_warnings` 字段（dispatcher 已自动填）。
```

- [ ] **Step 4.5: 跑测试 + 启服务冒烟（确认 skill 加载没破）**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
source .venv/bin/activate
make test 2>&1 | tail -10
```

启服务确认 SKILL.md 注入到 code-executor system prompt 时没语法错误：

```bash
make dev &
sleep 15
# 检查 langgraph 日志没 skill 加载报错
grep -i "skill\|error" packages/agent/backend/.langgraph_api/*.log 2>/dev/null | tail -10
make stop
```

- [ ] **Step 4.6: commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/skills/custom/ethoinsight-code/
git commit -m "feat(skill): ethoinsight-code SKILL.md 改 SOTA 架构 + by-paradigm/epm.md（Phase 1 T4）

- 删旧 references: run-paradigm-analysis-api / tool-reference / fallback-workflow / shoaling-paradigm
- 改 SKILL.md: 改为「按 paradigm 渐进披露 + 胶水脚本范例」入口
- 加 by-paradigm/epm.md: EPM 完整 API 视角（指标函数 / 胶水脚本 / handoff schema）"
```

### Task 5: 改写 code_executor.py system prompt + tools

**目的:** 把 system prompt 里「依次调用 5 个 langchain 工具」的描述改成「写胶水脚本」；把 `tools=[...]` 里的 5 个 langchain 工具去掉。

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py`（全文 92 行重排）

- [ ] **Step 5.1: 读现有 code_executor.py**

```bash
cat packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py
```

- [ ] **Step 5.2: 改 system prompt**

把 `system_prompt="""..."""` 里旧 6 步工作流（parse → compute → statistics → charts → assess_and_handoff → return）改成：

```python
system_prompt = """\
<语言>
中文优先，确保你输出的语言一致。
</语言>

<environment>
ethoinsight 是 pre-installed Python 库（无需 pip install）。
工作目录由 lead 提供（workspace_path）。
不用 venv，直接 python ${workspace_path}/analysis.py。
</environment>

<workflow>
1. read `ethoinsight-code/references/by-paradigm/<paradigm>.md` — 看本范式可用的指标函数 + 胶水脚本范例 + handoff schema
2. read `ethoinsight-charts` skill — 按数据特性选图
3. write_file 写胶水脚本（在 `${workspace_path}/analysis.py`） — import ethoinsight.metrics.<范式> + 算指标 + 跑统计 + 出图 + 写 handoff_code_executor.json
4. bash `python ${workspace_path}/analysis.py`
5. 如果失败，traceback 自动回来，改代码重跑（最多 2 次）
</workflow>

<output>
工作完成后输出 1 行确认（如 "OK: handoff written"），handoff JSON 已写盘 ${workspace_path}/handoff_code_executor.json。
lead 用 read_file 读 handoff 继续派遣后续 subagent。
</output>

<failure>
- 脚本崩溃: traceback 会被 tool_error_handling middleware 自动给到你，改代码重跑
- 同一脚本反复失败: loop_detection middleware 会自动中断，向 lead 报错
- 列名识别不到: 不要硬编码列名，所有列查找已封装在 metrics/<范式>.py 的 helper 函数里
</failure>
"""
```

- [ ] **Step 5.3: 删 `tools=[...]` 里的 5 个 langchain 工具 + `get_analysis_template`**

旧 `tools=[...]` 包含 11 个工具，新 `tools=[...]` 只保留：

```python
tools=[
    "bash",
    "read_file",
    "write_file",
    "ls",
    "str_replace",
],
```

> 删除的: `parse_trajectories`, `compute_metrics`, `run_statistics`, `generate_charts`, `assess_and_handoff`, `get_analysis_template`
> 这 6 个工具的删除（从 deerflow tool registry 里）放到 Phase 4 做（先解除引用即可）。

- [ ] **Step 5.4: 跑测试**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
source .venv/bin/activate
make test 2>&1 | tail -10
```

- [ ] **Step 5.5: commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py
git commit -m "refactor(code-executor): system prompt + tools 切换到 SOTA 架构（Phase 1 T5）

- system prompt: 5 工具串行流程 → 胶水脚本工作流
- tools: 移除 5 个 langchain 工具 + get_analysis_template，保留 bash/read_file/write_file/ls/str_replace
- skills 白名单已在 T3 改成 [ethoinsight-code, ethoinsight-charts]"
```

### Task 6: 改 ethoinsight skill paradigm-interpretation.md 为指针

**Files:**
- Rewrite: `packages/agent/skills/custom/ethoinsight/references/paradigm-interpretation.md`（从 77 行重写）
- Modify: `packages/agent/skills/custom/ethoinsight/SKILL.md` 的描述（重定位给 data-analyst + report-writer）

- [ ] **Step 6.1: 读现有 paradigm-interpretation.md 和 ethovision-paradigm-knowledge SKILL.md**

```bash
cat packages/agent/skills/custom/ethoinsight/references/paradigm-interpretation.md
cat packages/agent/skills/custom/ethovision-paradigm-knowledge/SKILL.md
```

- [ ] **Step 6.2: 写新 paradigm-interpretation.md 作指针**

```markdown
# 范式判读方法论入口

⚠️ 本文件已迁移。范式专属判读知识（焦虑度阈值、混杂因素、组比较规则等）由独立 skill 维护:

→ **`ethovision-paradigm-knowledge`** skill (5 月 9 日补完)
  - SKILL.md 总览
  - `references/by-experiment/<范式>.md` 每范式详细判读
  - `references/by-template/<EV19 大类>.md` EV19 → 学术范式映射

本 skill (`ethoinsight`) 现在专注于:
- **统计方法论** (`statistics-decision-tree.md`)
- **效应量判读** (`effect-size-guide.md`)
- **混杂因素清单** (`confound-checklist.md`)
- **报告写作** (`report.md`)

→ data-analyst / report-writer 收到 code-executor handoff 后:
  1. read 本 skill 的 `confound-checklist.md` / `statistics-decision-tree.md`
  2. 如需特定范式深度判读，read `ethovision-paradigm-knowledge/references/by-experiment/<范式>.md`
```

- [ ] **Step 6.3: 改 ethoinsight SKILL.md 描述**

把 frontmatter 的 `description:` 改成：

```yaml
description: >
  Behavioral neuroscience data interpretation methodology for data-analyst &
  report-writer subagents. Covers confound checking, statistical method selection,
  effect size assessment, and APA-format scientific reporting. Use AFTER
  code-executor handoff is received. For paradigm-specific judgment knowledge,
  see `ethovision-paradigm-knowledge` skill.
```

- [ ] **Step 6.4: 启服务冒烟 + commit**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
source .venv/bin/activate
make dev &
sleep 15
make stop

cd /home/wangqiuyang/noldus-insight
git add packages/agent/skills/custom/ethoinsight/
git commit -m "refactor(skill): ethoinsight 重定位 + paradigm-interpretation 改指针（Phase 1 T6）

- SKILL.md description: 改为「给 data-analyst + report-writer 看判读方法论」
- paradigm-interpretation.md: 77 行方法论改为 12 行指针文件指向 ethovision-paradigm-knowledge
- 不删 confound-checklist / effect-size-guide / statistics-decision-tree / report.md（这些仍归 ethoinsight）"
```

### Task 7: EPM 端到端打通验证

**目的:** 用真 EPM raw 数据跑完整流程，确认 lead → code-executor → handoff → data-analyst → report-writer 链路通畅。

**前置:** 用户在服务器上放好 EPM raw 数据（用户已确认会放）。本 task 假设数据在 `demo-data/DemoData/EPM/` 路径下；如果路径不同，调整 raw_dir。

**Files:**
- Use existing: `packages/agent/Makefile`（`make dev`）
- Possibly create: `docs/handoffs/2026-05-11-epm-e2e-verification-<status>.md` 记录验证结果

- [ ] **Step 7.1: 确认服务器上 EPM 数据位置**

```bash
find /home/wangqiuyang/noldus-insight -maxdepth 5 -iname "*epm*" -type d 2>/dev/null
# 或者用户指定的其它路径
ls demo-data/DemoData/ 2>/dev/null
```

把找到的 EPM raw 数据路径记下来。

- [ ] **Step 7.2: 启服务**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent
make dev
# 等几秒，确认 localhost:2026 起来
```

- [ ] **Step 7.3: 浏览器/curl 触发一个 EPM 分析任务**

```bash
# 方式 A: 通过 frontend (localhost:2026) 上传 EPM raw 数据 + 文字描述「分析这个 EPM 实验数据」
# 方式 B: 直接调 gateway API
curl -X POST http://localhost:8001/api/threads \
  -H 'Content-Type: application/json' \
  -d '{"message": "请分析这个 EPM 高架十字迷宫数据，对照组 5 只，治疗组 5 只", "files": [<raw_paths>]}'
```

- [ ] **Step 7.4: 观察 thread 状态机日志**

```bash
# 实时看 langgraph + gateway 日志
tail -f packages/agent/backend/.langgraph_api/*.log
```

期望路径（每步等几十秒 - 几分钟，按 model 速度）:

| 顺序 | 期望事件 | 在日志中找的关键字 |
|---|---|---|
| 1 | lead 收到任务，识别 paradigm=epm | `paradigm.*epm`、`ev19_template` 字段填好 |
| 2 | lead 派遣 code-executor | `subagent.*code-executor.*invoked` 或 `task tool` 调用 |
| 3 | code-executor read by-paradigm/epm.md | `read_file.*by-paradigm/epm.md` |
| 4 | code-executor write_file analysis.py | `write_file.*analysis.py` |
| 5 | code-executor bash 跑脚本 | `bash.*python.*analysis.py` |
| 6 | handoff JSON 落地 | `handoff_code_executor.json` 出现在 workspace |
| 7 | lead 派遣 data-analyst | `subagent.*data-analyst.*invoked` |
| 8 | data-analyst read ethoinsight skill | `read_file.*ethoinsight/references/.*.md` |
| 9 | data-analyst 出 handoff | `handoff_data_analyst.json` 出现 |
| 10 | lead 派遣 report-writer | `subagent.*report-writer.*invoked` |
| 11 | report-writer 出报告 | 最终消息在 thread 出现 |

如果某一步卡 > 10 分钟，看 Appendix A 错误诊断反查表。

- [ ] **Step 7.5: 验证 handoff JSON 内容**

```bash
WORKSPACE=$(ls -td /mnt/user-data/workspace/*/ | head -1)  # 最新 thread workspace
cat ${WORKSPACE}/handoff_code_executor.json | python -m json.tool
```

检查清单:
- `paradigm` == `"epm"`
- `metrics.per_subject` 含每个 subject 的 4 个 EPM 指标
- `metrics.group_summary` 含 control / treatment 组 mean / std / n
- `statistics` 含 p-value / 检验方法
- `charts` 含 ≥1 个 .png 文件路径
- `data_quality_warnings` 字段存在（可能为空数组）

- [ ] **Step 7.6: 验证 chart PNG 文件存在 + 可视**

```bash
ls ${WORKSPACE}/outputs/*.png
file ${WORKSPACE}/outputs/*.png  # 确认是 valid PNG
```

- [ ] **Step 7.7: 验证最终报告**

通过 frontend 看 report-writer 输出。检查:
- 报告语言: 中文
- 包含统计结论（p-value + 效应量）
- 包含混杂因素提示（如 total_entry_count 异常）
- 包含至少 1 张图嵌入

- [ ] **Step 7.8: 停服务 + 写验证 handoff**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent
make stop
```

写 `docs/handoffs/2026-05-11-epm-e2e-verification-<pass|fail>-handoff.md` 记录:
- 验证时间
- 数据路径
- thread_id
- 关键中间产物路径（handoff JSON、PNG）
- pass / fail 结论
- 如果 fail，详细 traceback + 建议修复方向

- [ ] **Step 7.9: commit 验证 handoff**

```bash
cd /home/wangqiuyang/noldus-insight
git add docs/handoffs/2026-05-11-epm-e2e-verification-*.md
git commit -m "docs(handoff): EPM 端到端验证 <pass|fail>（Phase 1 T7）"
```

### Phase 1 验收清单

完成所有以下条件才能进入 Phase 2:

- [x] `metrics/__init__.py` + 4 个范式文件 + dispatcher 全部建立，pytest tests/ 无新增失败
- [x] `tests/test_metrics_epm.py` 20+ 个测试全过
- [x] `tests/test_metrics.py` 31 个测试全过（拆分没破 shoaling/oft）
- [x] skill 目录已重命名 `ethoinsight-analysis` → `ethoinsight-code`，所有引用同步
- [x] `code_executor.py` 白名单 = `["ethoinsight-code", "ethoinsight-charts"]`，system prompt + tools 切换完
- [x] `ethoinsight-code/by-paradigm/epm.md` 存在，含函数清单 + 胶水脚本范例 + handoff schema
- [x] `ethoinsight/SKILL.md` description 重定位，`paradigm-interpretation.md` 改为指针
- [x] EPM e2e: 胶水脚本跑通 demo EPM 数据 → handoff JSON 落地（Script 级；完整 agent 服务 e2e 待后续补）
- [x] 至少 6 个 commit 入 dev 分支（共 8 个）

### Phase 1 完成后接手点

> 下一位 agent 接手时看这里：
> - Phase 1 已完成: ✅ EPM 跑通整条 SOTA 链路
> - 仓库状态: metrics/ 子包已建，3 范式（shoaling/oft/epm）已迁出 metrics.py；ethoinsight-code skill 已生效；code-executor 切换完
> - 「当前阶段」字段应更新为 `current_phase: "Phase 2 - OFT 端到端验证"`, `current_task: "T1 - 检查 OFT 指标完整性"`
> - Phase 2 起步: read `docs/review-packages/2026-04-29-ev19-templates/by-experiment/oft.md`

---

## Phase 2: OFT 端到端验证 SOTA 架构

**目标:** 用 OFT 范式重复 EPM 流程，证明 SOTA 架构通用，不是 EPM 专属 hack。Phase 2 比 Phase 1 简单很多（基础设施都建好了），主要是补 OFT 指标缺口 + 写 by-paradigm/oft.md + e2e 验证。

**关键资产（已就位）:**
- `packages/ethoinsight/ethoinsight/metrics/oft.py` — Phase 1 T1 时已建（含 `compute_center_time_ratio`, `compute_thigmotaxis_index`）
- `packages/ethoinsight/tests/test_metrics.py` 里 OFT 相关测试已绿
- `packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/epm.md` — Phase 1 T4 写的「样板」，本 Phase 照着写 oft.md

### Task 1: 检查 OFT 指标完整性 + 补缺

**Files:**
- Read: `docs/review-packages/2026-04-29-ev19-templates/by-experiment/oft.md`
- Read: `packages/ethoinsight/ethoinsight/metrics/oft.py`
- Possibly modify: `packages/ethoinsight/ethoinsight/metrics/oft.py`
- Possibly modify: `packages/ethoinsight/tests/test_metrics.py` 或新建 `tests/test_metrics_oft.py`
- Possibly modify: `packages/ethoinsight/ethoinsight/metrics/dispatcher.py` + `metrics/__init__.py`

- [ ] **Step 1.1: 读同事 OFT 行为学需求**

```bash
cd /home/wangqiuyang/noldus-insight
cat docs/review-packages/2026-04-29-ev19-templates/by-experiment/oft.md
```

逐字摘录里面的「核心指标 / 必算指标 / 推荐指标」清单。常见包括（仅参考，**以同事文档为准**）：
- 中心区滞留时间 / 占比
- 趋触性指数（thigmotaxis index）
- 总移动距离
- 平均速度 / 最大速度
- 静止时间
- 中心区进入次数 / 边缘区滞留时间

- [ ] **Step 1.2: 看 metrics/oft.py 现有 2 函数**

```bash
grep -n "^def" packages/ethoinsight/ethoinsight/metrics/oft.py
# 预期: compute_center_time_ratio + compute_thigmotaxis_index 共 2 个
```

- [ ] **Step 1.3: 列缺失指标清单，按是否「单指标公式可独立写函数」分流**

把 Step 1.1 摘录的指标对照 Step 1.2 现有函数：
- 已有: ✅ 跳过
- 缺失但可调用 `_common`: 比如「总移动距离」直接用 `compute_distance_moved`，「平均速度」直接用 `compute_velocity_stats` —— 不需要新写函数，只需在 Step 4 by-paradigm/oft.md 里说明
- 缺失且需新函数: 加到本 Task 范围

**例子:** 假设缺「中心区进入次数」`compute_center_entry_count`。

- [ ] **Step 1.4: 对每个缺失函数走完整 TDD**

每个缺失函数重复 5 步（红/绿/重构/导出/commit）：

#### 1.4.A 写 failing test

文件 `packages/ethoinsight/tests/test_metrics_oft.py`（如不存在则新建）：

```python
"""Tests for OFT (Open Field Test) metric functions.

Mirrors test_metrics_epm.py structure.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest

from ethoinsight.metrics.oft import (
    compute_center_time_ratio,
    compute_thigmotaxis_index,
    # 新增导入: compute_center_entry_count, ...
)


def _make_oft_df(
    n_frames: int = 100,
    *,
    center_cols: list[str] | None = None,
    border_cols: list[str] | None = None,
    center_pattern: list[int] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Helper: synthetic OFT DataFrame with controllable zone presence."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "trial_time": np.arange(n_frames) * 0.04,  # 25 fps
        "x_center": rng.uniform(-10, 10, n_frames),
        "y_center": rng.uniform(-10, 10, n_frames),
    })
    center_cols = center_cols or ["in_zone_center"]
    border_cols = border_cols or ["in_zone_border"]
    if center_pattern is None:
        center_pattern = [1] * 20 + [0] * 80  # 20% center
    for col in center_cols:
        df[col] = center_pattern[:n_frames] + [0] * max(0, n_frames - len(center_pattern))
    for col in border_cols:
        df[col] = [1 - v for v in df[center_cols[0]]]
    return df


class TestComputeCenterEntryCount:
    def test_no_center_presence_returns_zero(self):
        df = _make_oft_df(center_pattern=[0] * 100)
        # assert compute_center_entry_count(df) == 0
        pass  # 实现后取消注释

    def test_single_entry(self):
        df = _make_oft_df(center_pattern=[0] * 20 + [1] * 30 + [0] * 50)
        # assert compute_center_entry_count(df) == 1
        pass

    def test_multiple_entries(self):
        df = _make_oft_df(center_pattern=[1, 0, 1, 0, 1, 0] * 16 + [0] * 4)
        # assert compute_center_entry_count(df) == 16
        pass
```

跑测试看 FAIL（因 import 不存在）：

```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
python -m pytest tests/test_metrics_oft.py::TestComputeCenterEntryCount -v 2>&1 | tail -10
# 预期: ImportError 或 NameError
```

#### 1.4.B 实现函数

文件 `packages/ethoinsight/ethoinsight/metrics/oft.py`，加：

```python
def compute_center_entry_count(
    df: pd.DataFrame,
    center_zones: list[str] | None = None,
) -> int | None:
    """Number of entries into the center zone (0→1 transitions)."""
    if center_zones:
        cols = [c for c in center_zones if c in df.columns]
    else:
        cols = [c for c in df.columns if "center" in c.lower() and c.startswith("in_zone_")]
    if not cols:
        return None
    combined = df[cols].max(axis=1).dropna()
    if combined.empty:
        return 0
    vals = combined.to_numpy(dtype=int)
    entries = 1 if vals[0] == 1 else 0
    transitions = (vals[1:] == 1) & (vals[:-1] == 0)
    return entries + int(transitions.sum())
```

#### 1.4.C 取消 test 注释（去掉 `pass`）+ 跑测试

```bash
python -m pytest tests/test_metrics_oft.py::TestComputeCenterEntryCount -v 2>&1 | tail -10
# 预期: 3 passed
```

#### 1.4.D 在 `__init__.py` + `dispatcher.py` 加导出 + 派发

文件 `packages/ethoinsight/ethoinsight/metrics/__init__.py`：

```python
# 在 OFT 那一段加:
from ethoinsight.metrics.oft import (
    compute_center_time_ratio,
    compute_thigmotaxis_index,
    compute_center_entry_count,   # 新增
)

# __all__ 也加:
"compute_center_entry_count",
```

文件 `packages/ethoinsight/ethoinsight/metrics/dispatcher.py`，在 `elif paradigm == "open_field"` 分支：

```python
elif paradigm == "open_field":
    m["center_time_ratio"] = compute_center_time_ratio(df)
    m["thigmotaxis_index"] = compute_thigmotaxis_index(df)
    m["center_entry_count"] = compute_center_entry_count(df)  # 新增
```

跑 dispatcher 测试：

```bash
python -m pytest tests/test_metrics.py -k "ParadigmMetrics" -v 2>&1 | tail -10
```

如果 dispatcher 测试 fixture 没覆盖新指标，加一个 assert 进 `TestComputeParadigmMetrics::test_open_field_returns_*`。

#### 1.4.E commit

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/ethoinsight/ethoinsight/metrics/oft.py \
        packages/ethoinsight/ethoinsight/metrics/__init__.py \
        packages/ethoinsight/ethoinsight/metrics/dispatcher.py \
        packages/ethoinsight/tests/test_metrics_oft.py
git commit -m "feat(ethoinsight): 补 OFT 指标 compute_center_entry_count + 单测（Phase 2 T1）"
```

> **重复 1.4 直到 Step 1.3 所列的所有缺失函数都加完。** 如 Step 1.3 显示 0 个缺失，跳过整个 Step 1.4。

- [ ] **Step 1.5: 跑全套 OFT 测试**

```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
python -m pytest tests/test_metrics_oft.py tests/test_metrics.py -k "oft or open_field" -v
# 预期: 所有 OFT 测试全过
```

### Task 2: 写 by-paradigm/oft.md

**Files:**
- Create: `packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/oft.md`
- Modify: `packages/agent/skills/custom/ethoinsight-code/SKILL.md`

- [ ] **Step 2.1: 拷 epm.md 当骨架**

```bash
cd /home/wangqiuyang/noldus-insight
cp packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/epm.md \
   packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/oft.md
```

- [ ] **Step 2.2: 改 oft.md 内容**

文件 `packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/oft.md` — 改这几处：

1. **顶部标题**：`# OFT (Open Field Test) 代码执行参考`
2. **slug**：`open_field`（不是 `oft`！dispatcher 里用的 paradigm key 是 `open_field`，与历史一致）
3. **EV19 模板映射**：`Open Field` 大类下所有变体
4. **行为学同事文档**：`docs/review-packages/2026-04-29-ev19-templates/by-experiment/oft.md`
5. **可用指标函数表**（按 Task 1 实际函数清单填）：

| 函数 | 输入 | 输出 | 含义 |
|---|---|---|---|
| `compute_center_time_ratio(df, center_zones=None, border_zones=None)` | DataFrame | `float \| None` | 中心区时间占比（中心帧 / (中心+边缘)帧） |
| `compute_thigmotaxis_index(df, ...)` | DataFrame | `float \| None` | 趋触性指数（边缘滞留偏好） |
| `compute_center_entry_count(df, center_zones=None)` | DataFrame | `int \| None` | 中心区进入次数（如 Task 1 新增） |
| ... 其余新增函数 ... |

6. **派发器示例**：把 `paradigm="epm"` 改为 `paradigm="open_field"`；指标 keys 改 `center_time_ratio` / `thigmotaxis_index` / ...
7. **胶水脚本范例**：改 paradigm 名、GROUPS（如 OFT 通常 8-10 只/组）、metrics 列表、图表 metric_name
8. **数据质量警告**：根据同事 by-experiment/oft.md 写（OFT 常见: 总移动距离 < 阈值 → 可能为运动抑制；中心区帧数 == 0 → 极度焦虑或 zone 标定错）
9. **出图建议**：`center_time_ratio` 用 box_plot；`thigmotaxis_index` 用 box_plot 或 violin_plot；轨迹图 (`trajectory_plot`) 是 OFT 经典图，加进推荐
10. **反模式提醒**：保留 epm.md 同款 + 加 OFT 特异（如 ❌ 不要把「中心区 < 15% 算高焦虑」绝对阈值写进解读，只做组比较）

- [ ] **Step 2.3: 改 SKILL.md 入口标记**

文件 `packages/agent/skills/custom/ethoinsight-code/SKILL.md`，「范式渐进披露入口」section：

```diff
- - **OFT** (Open Field): `references/by-paradigm/oft.md` *(Phase 2 时撰写)*
+ - **OFT** (Open Field): `references/by-paradigm/oft.md`
```

- [ ] **Step 2.4: 启服务冒烟（确认 skill 加载没报 yaml/markdown 错）**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
source .venv/bin/activate
make dev &
sleep 15
# 看 langgraph 日志
grep -iE "skill|error|fail" .langgraph_api/*.log 2>/dev/null | tail -10
make stop
```

- [ ] **Step 2.5: commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/oft.md \
        packages/agent/skills/custom/ethoinsight-code/SKILL.md
git commit -m "feat(skill): ethoinsight-code by-paradigm/oft.md + SKILL 入口去 placeholder（Phase 2 T2）"
```

### Task 3: OFT 端到端验证

**Files:**
- No code changes expected
- Possibly create: `docs/handoffs/2026-05-XX-oft-e2e-verification-<pass|fail>-handoff.md`

> **前置:** 用户在服务器上有 OFT raw 数据；与 EPM e2e 同样位置查找。

- [ ] **Step 3.1: 找 OFT raw 数据位置**

```bash
cd /home/wangqiuyang/noldus-insight
find . -maxdepth 6 -iname "*open*field*" -type d 2>/dev/null
find . -maxdepth 6 -iname "*oft*" 2>/dev/null | head -10
ls demo-data/ 2>/dev/null
```

记下 raw 数据路径。如果找不到，先 ask user：「OFT raw 数据放在哪？」

- [ ] **Step 3.2: 启服务**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent
make dev
sleep 10
curl -s http://localhost:2026/ | head -5
```

- [ ] **Step 3.3: 触发 OFT 分析**

通过 frontend (`http://localhost:2026/`) 上传 OFT raw 数据 + 输入 prompt:

```
请分析这个 Open Field 测试数据。对照组 N 只，治疗组 M 只。
```

或通过 API:

```bash
THREAD_ID=$(curl -s -X POST http://localhost:8001/api/threads \
  -H 'Content-Type: application/json' \
  -d '{}' | python -c "import sys,json;print(json.load(sys.stdin)['thread_id'])")
echo "thread: $THREAD_ID"

# upload files + send message: 参考 frontend 调用代码
```

- [ ] **Step 3.4: 实时观察 thread 状态**

```bash
tail -f packages/agent/backend/.langgraph_api/*.log &
TAIL_PID=$!

# 等 3-5 分钟看流程
# 期望路径:
#   1. lead 识别 paradigm=open_field, ev19_template=<Open Field 变体>
#   2. lead 派遣 code-executor (含 paradigm + groups + workspace_path)
#   3. code-executor read by-paradigm/oft.md
#   4. code-executor write_file ${workspace}/analysis.py
#   5. code-executor bash python analysis.py
#   6. handoff JSON + 至少 1 张 PNG 落 ${workspace}/outputs/
#   7. lead 派遣 data-analyst (附 handoff JSON 路径)
#   8. data-analyst read handoff + ethoinsight skill 的 confound-checklist + statistics-decision-tree
#   9. data-analyst 出 handoff
#   10. lead 派遣 report-writer
#   11. report-writer 出最终报告

kill $TAIL_PID
```

- [ ] **Step 3.5: 验证 handoff JSON**

```bash
WORKSPACE=$(ls -td /mnt/user-data/workspace/*/ 2>/dev/null | head -1)
ls "${WORKSPACE}"
cat "${WORKSPACE}/handoff_code_executor.json" | python -m json.tool | head -50
```

检查清单（任一不满足都是 fail）：
- `paradigm` == `"open_field"`
- `metrics.per_subject` 每个 subject 含 `center_time_ratio` / `thigmotaxis_index` / ...
- `metrics.group_summary.<group>.<metric>.{mean, std, n, values}` 都有
- `statistics` 含 p-value
- `charts` 至少 1 项是真实存在的 .png 文件路径

- [ ] **Step 3.6: 验证 PNG**

```bash
ls "${WORKSPACE}/outputs/"*.png
for f in "${WORKSPACE}/outputs/"*.png; do
  file "$f"
  python -c "from PIL import Image; img = Image.open('$f'); print('$f', img.size)"
done
```

- [ ] **Step 3.7: 验证 frontend 报告**

通过 `http://localhost:2026/threads/${THREAD_ID}` 看 report-writer 输出，对照清单：
- [ ] 中文
- [ ] 含 p-value + 效应量
- [ ] 含统计方法名（如 Welch's t-test / Mann-Whitney U）
- [ ] 含组内 mean ± SD 数据
- [ ] 嵌入至少 1 张图
- [ ] 含 data_quality_warnings 内容（如果有）

- [ ] **Step 3.8: 停服务**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent
make stop
```

- [ ] **Step 3.9: 写验证 handoff**

按 `<status>` ∈ {pass, fail} 写 `docs/handoffs/2026-05-XX-oft-e2e-verification-<status>-handoff.md`：

```markdown
# OFT 端到端验证 <pass|fail> 交接

**日期:** 2026-05-XX
**Phase:** Phase 2 Task 3
**数据路径:** <oft_raw_data_path>
**Thread ID:** <thread_id>
**Workspace:** <workspace_path>

## 关键产物

- handoff JSON: `<workspace>/handoff_code_executor.json`
- 图表: `<workspace>/outputs/*.png` (N 张)
- 最终报告: frontend thread view

## 验证结果

[列出 Step 3.5-3.7 的每个 checklist 的 ✅/❌]

## <如果 fail> 详细错误

[traceback / 日志摘录 / 失败点分析]

## 下一步建议

[修复方向 / 是否阻塞 Phase 3]
```

- [ ] **Step 3.10: commit + push**

```bash
cd /home/wangqiuyang/noldus-insight
git add docs/handoffs/2026-05-XX-oft-e2e-verification-*.md
git commit -m "docs(handoff): OFT 端到端验证 <pass|fail>（Phase 2 T3）"
git push origin dev
```

### Phase 2 完成后接手点

> 下一位 agent 接手时看这里：
> - Phase 2 已完成: ✅ `metrics/oft.py` 函数齐 + `by-paradigm/oft.md` 写完 + OFT e2e pass
> - SKILL.md 入口只剩 zero-maze / ldb / tst / fst 4 个 `*(Phase 3 时撰写)*` 标记
> - Phase 3 起步: read `docs/review-packages/2026-04-29-ev19-templates/by-experiment/zero-maze.md` 开始第一个范式

### Phase 2 验收清单

- [ ] `metrics/oft.py` 指标函数清单与同事 by-experiment/oft.md 完全对应
- [ ] `tests/test_metrics_oft.py` 或 `tests/test_metrics.py` OFT 段全过
- [ ] `by-paradigm/oft.md` 完整（≥ epm.md 80% 长度）
- [ ] SKILL.md 入口的 OFT 那行无 `(Phase 2 时撰写)` 标记
- [ ] OFT e2e pass: handoff JSON ✅ + ≥1 PNG ✅ + 最终报告中文 + 含 p-value
- [ ] handoff `2026-05-XX-oft-e2e-verification-pass-handoff.md` 已 commit

---

## Phase 3: Zero Maze / LDB / TST / FST 批量

**目标:** 4 个范式全部加入 SOTA 架构。这 4 个范式在 `metrics.py` 里**完全没写**，是真从零起。

**做法:** 每个范式独立一个 Task，4 个 Task 流程**完全一致**（参考 Phase 1 T2 + Phase 2 T1+T2 做法）。本 Phase 用「检查点表」推进，不重复写 step（一处错 4 倍工作量）。

### 4 个范式的范围声明

| 范式 | slug (paradigm key) | metrics 文件 | 测试文件 | by-paradigm doc | 同事 review |
|---|---|---|---|---|---|
| Zero Maze | `zero_maze` | `metrics/zero_maze.py` | `tests/test_metrics_zero_maze.py` | `by-paradigm/zero-maze.md` | `by-experiment/zero-maze.md` |
| Light-Dark Box | `ldb` | `metrics/ldb.py` | `tests/test_metrics_ldb.py` | `by-paradigm/ldb.md` | `by-experiment/ldb.md` |
| Tail Suspension Test | `tst` | `metrics/tst.py` | `tests/test_metrics_tst.py` | `by-paradigm/tst.md` | `by-experiment/tst.md` |
| Forced Swim Test | `fst` | `metrics/fst.py` | `tests/test_metrics_fst.py` | `by-paradigm/fst.md` | `by-experiment/fst.md` |

> **slug 注意:** dispatcher 里 paradigm key 用下划线，文件名用连字符（已与现有 `open_field` paradigm vs 历史习惯一致）。

### 单 paradigm 工作流（4 倍重复）

对每个 paradigm，按 Phase 1 T2 + Phase 2 T1 模式走以下 9 步（参考 Phase 2 T1 Step 1.4 完整代码模板）：

1. **read** `docs/review-packages/2026-04-29-ev19-templates/by-experiment/<paradigm>.md` 列必算指标
2. **write** failing tests 在 `tests/test_metrics_<paradigm>.py`（参考 `test_metrics_epm.py` 结构）
3. **verify** 测试 FAIL: `python -m pytest tests/test_metrics_<paradigm>.py -v` 期望 ImportError
4. **implement** `packages/ethoinsight/ethoinsight/metrics/<paradigm>.py`（含必要 helper + 算法）
5. **export** 在 `metrics/__init__.py` 和 `dispatcher.py` 加导出 + 派发分支
6. **verify** 测试 PASS: `python -m pytest tests/test_metrics_<paradigm>.py tests/test_metrics.py -v`
7. **write** `packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/<paradigm>.md`（拷 epm.md 当骨架，改 5 处）
8. **update** SKILL.md 入口去 `(Phase 3 时撰写)` 标记
9. **commit**：`feat(ethoinsight): <paradigm> 指标 + skill 文档（Phase 3 T<N>）`

### 各范式算法特异点（事先警告）

**Zero Maze** (`zero_maze`)：
- 几何与 EPM 类似（开/闭区段，但是环形）
- 必算: 开放区段时间占比、开放区段进入次数、总进入次数、对侧穿越次数
- 列名: `in_zone_open_<n>` / `in_zone_closed_<n>` 通常 4 段
- ⚠️ helper 思路可大量复用 EPM `_count_zone_entries` / `_find_arm_zone_columns`，但不要 import EPM 函数；在 `metrics/_common.py` 抽取一个 `_count_zone_entries_generic(df, zone_cols)` 共用

**Light-Dark Box** (`ldb`)：
- 必算: 明箱时间占比、明箱进入次数、首次进入明箱潜伏期（关键指标，需新 helper）、穿梭次数（grill-me 决策 10: 单向）
- 列名: `in_zone_light` / `in_zone_dark`
- ⚠️ 「首次进入明箱潜伏期」是新算法：找第一个 `in_zone_light == 1` 的 frame，取其 trial_time。如果整段为 0，返回 None 或 trial_total。

**Tail Suspension Test** (`tst`)：
- 必算: 静止时间总长、静止时间占比、静止次数（run-length encoding）、平均静止时长、运动潜伏期
- 列名: `Activity_State` (1=motile / 0=immobile) 或类似 EthoVision 提供的 motion-state 列
- ⚠️ run-length encoding 关键：连续 0 算 1 次 immobility bout；不要把 0-1-0 当 1 次 bout。建议 helper `_runs(arr, value)` 返回 list of (start, end) 对。

**Forced Swim Test** (`fst`)：
- 与 TST 算法**几乎完全相同**（都是不动检测）
- 必算: 与 TST 同集
- 列名: 同 TST
- ⚠️ TST 和 FST 的差异是阈值（行为学上 FST 静止判定更严格），但 grill-me 决策 9 已校准：阈值在 EthoVision 导出时已固化在 `Activity_State` 列里，工程师不需要管。所以代码上 FST 可以 import TST 的 helper 函数，或者干脆 dispatcher 把两个范式 route 到同一组指标函数。**推荐方案:** 在 `metrics/_common.py` 写 `compute_immobility_*` 系列函数；`metrics/tst.py` 和 `metrics/fst.py` 各自 thin re-export + 一两个范式特异指标（如果有）。

### 单 paradigm 检查点表

每个 paradigm 完成后填这个表（plan agent 把表附进 commit message 或交接 handoff）：

| Paradigm | 必算指标数 | 实现函数数 | 测试函数数 | dispatcher 分支 | by-paradigm doc 行数 | SKILL 入口 ✅ | commit hash |
|---|---|---|---|---|---|---|---|
| zero_maze | <N> | <N> | <N> | ✅ | <N> | ✅ | `<hash>` |
| ldb | <N> | <N> | <N> | ✅ | <N> | ✅ | `<hash>` |
| tst | <N> | <N> | <N> | ✅ | <N> | ✅ | `<hash>` |
| fst | <N> | <N> | <N> | ✅ | <N> | ✅ | `<hash>` |

### Task 5 (Phase 3 收尾): 至少 2 个范式 e2e

**Files:**
- Create: 2 个 `docs/handoffs/2026-05-XX-<paradigm>-e2e-verification-<pass|fail>-handoff.md`

**推荐 e2e 范式选择:** Zero Maze + TST（算法差异覆盖最广：Zero Maze 是 zone 类，TST 是 immobility 类）。

- [ ] **Step 5.1: Zero Maze e2e**（重复 Phase 2 T3 Step 3.1-3.10，仅 paradigm 改为 `zero_maze`）
- [ ] **Step 5.2: TST e2e**（重复 Phase 2 T3，paradigm = `tst`）
- [ ] **Step 5.3: commit 2 个验证 handoff**

### Phase 3 完成后接手点

> 下一位 agent 接手时看这里：
> - Phase 3 完成: 4 个 paradigm 在 metrics/、tests/、by-paradigm/ 全部齐全 + 2 个 e2e pass
> - SKILL.md 入口完全无 `(Phase X 时撰写)` 标记
> - Phase 4 起步: 旧工具清理（先 grep 确认无引用再 rm）

### Phase 3 验收清单

- [ ] 4 个 paradigm 检查点表全填，所有项 ✅
- [ ] `pytest packages/ethoinsight/tests/ -v` 无新增失败（除 pre-existing test_parse.py 9 个）
- [ ] `packages/agent/backend && make test` 全绿
- [ ] `packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/` 下 6 个 .md（epm/oft/zero-maze/ldb/tst/fst）齐全
- [ ] SKILL.md 范式入口段无 `(Phase 3 时撰写)` 标记
- [ ] 至少 2 个 e2e pass handoff 已 commit

---

## Phase 4: 旧工具清理

**目标:** 解除引用后真删旧代码，让仓库干净。在 Phase 1-3 都跑通后做。

### Task 1: 删 7 个废弃 langchain 工具

**Files:**
- Identify: 工具函数定义文件（grep 找）
- Delete or rewrite: deerflow tool registry
- Possibly delete: 工具实现 .py 文件
- Possibly delete: 对应 tests

**目标 7 个工具:**
- `parse_trajectories`
- `compute_metrics`
- `run_statistics`
- `generate_charts`
- `assess_and_handoff`
- `run_paradigm_analysis`
- `get_analysis_template`

- [ ] **Step 1.1: grep 定位每个工具的定义文件**

```bash
cd /home/wangqiuyang/noldus-insight
for tool in parse_trajectories compute_metrics run_statistics generate_charts assess_and_handoff run_paradigm_analysis get_analysis_template; do
  echo "=== $tool ==="
  grep -rln "def ${tool}\\b\\|@tool.*name=\"${tool}\"\\|@tool.*\"${tool}\"\\|register.*\"${tool}\"" \
    packages/agent/backend/packages/harness/deerflow/ packages/ethoinsight/ 2>/dev/null
done
```

把每个工具的定义文件路径记下来，整理成清单。

- [ ] **Step 1.2: 找 deerflow tool registry**

```bash
find packages/agent/backend/packages/harness/deerflow -name "tools" -type d
ls packages/agent/backend/packages/harness/deerflow/tools/builtins/ 2>/dev/null
grep -rn "BUILTIN_TOOLS\\|TOOL_REGISTRY\\|register_tool" \
  packages/agent/backend/packages/harness/deerflow/tools/ 2>/dev/null | head -10
```

记下 registry 文件路径（通常是 `tools/builtins/__init__.py` 或 `tools/registry.py`）。

- [ ] **Step 1.3: 验证 7 个工具在所有 subagent 配置里无引用**

```bash
cd /home/wangqiuyang/noldus-insight
grep -rln "parse_trajectories\\|compute_metrics\\|run_statistics\\|generate_charts\\|assess_and_handoff\\|run_paradigm_analysis\\|get_analysis_template" \
  packages/agent/backend/packages/harness/deerflow/subagents/ 2>/dev/null
```

**期望: 无结果**（Phase 1 T5 已经从 code_executor.py 移除；其他 subagent 历史上不应该用这些）。

如有结果：先回到对应 subagent 文件移除引用，再继续。

- [ ] **Step 1.4: 删工具函数定义**

对每个工具：
1. 打开 Step 1.1 定位的文件
2. 删 `def <tool>(...)` 函数体 + 任何 `@tool` decorator
3. 删对应的 unit test（如果在 `tools/builtins/tests/test_<tool>.py`）

如果一个文件里只有一个工具，整个文件 `git rm`。如果一个文件里多个工具，逐个删函数。

- [ ] **Step 1.5: 从 registry 注销**

打开 Step 1.2 定位的 registry 文件，删除：
- 7 个工具的 `from ... import <tool>` 语句
- 注册到 `BUILTIN_TOOLS` 或 `TOOL_REGISTRY` 字典的对应条目

- [ ] **Step 1.6: 跑全套测试**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
source .venv/bin/activate
make lint 2>&1 | tail -20
make test 2>&1 | tail -20

cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
python -m pytest tests/ -v 2>&1 | tail -20
```

期望: 测试全绿。如果出现 ImportError，说明还有未解除的引用，回 Step 1.3 重 grep。

- [ ] **Step 1.7: 再次 grep 确认彻底干净**

```bash
cd /home/wangqiuyang/noldus-insight
grep -rln "parse_trajectories\\|compute_metrics\\|run_statistics\\|generate_charts\\|assess_and_handoff\\|run_paradigm_analysis\\|get_analysis_template" \
  packages/ 2>/dev/null
# 期望: 无结果（docs/ 历史描述允许保留）
```

- [ ] **Step 1.8: commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add -A
git commit -m "refactor(deerflow): 删除 7 个废弃 langchain 工具（Phase 4 T1）

被删工具:
- parse_trajectories / compute_metrics / run_statistics
- generate_charts / assess_and_handoff
- run_paradigm_analysis / get_analysis_template

SOTA 架构下 code-executor 通过 bash 写胶水脚本调用 ethoinsight.metrics.<范式>.<函数>，
不再需要这些 langchain @tool 包装。"
```

### Task 2: 删 templates/shoaling.py + templates/_gate.py + templates/tool.py

**前置确认:** shoaling 范式的真实数据流走 SOTA 架构（code-executor 写胶水脚本调用 `ethoinsight.metrics.shoaling`），不再走 `templates/shoaling.py:main()`。

**Files:**
- Delete: `packages/ethoinsight/ethoinsight/templates/shoaling.py`
- Delete: `packages/ethoinsight/ethoinsight/templates/_gate.py`
- Delete: `packages/ethoinsight/ethoinsight/templates/tool.py`
- Possibly delete: `packages/ethoinsight/tests/test_template_soft_gate.py`
- Possibly delete: `packages/ethoinsight/tests/test_templates.py`
- Possibly delete: `packages/ethoinsight/tests/test_template_tool.py`
- Possibly empty/delete: `packages/ethoinsight/ethoinsight/templates/__init__.py`

- [ ] **Step 2.1: grep 确认无 production 引用**

```bash
cd /home/wangqiuyang/noldus-insight
grep -rln "ethoinsight\\.templates\\.shoaling\\|from ethoinsight import templates\\|templates\\.shoaling\\.main\\|require_ev19_template" \
  packages/agent/ packages/ethoinsight/ethoinsight/ 2>/dev/null

# 测试代码可能引用：
grep -rln "templates.shoaling\\|require_ev19_template" packages/ethoinsight/tests/ 2>/dev/null
```

**期望 production:** 无结果（Phase 1 T5 已切；Phase 4 T1 已删 `get_analysis_template`）。
**测试 expected:** `test_templates.py` / `test_template_soft_gate.py` / `test_template_tool.py` 仍引用 → 一并删除（它们只测旧模板）。

- [ ] **Step 2.2: 删 templates/ 文件**

```bash
cd /home/wangqiuyang/noldus-insight
git rm packages/ethoinsight/ethoinsight/templates/shoaling.py
git rm packages/ethoinsight/ethoinsight/templates/_gate.py
git rm packages/ethoinsight/ethoinsight/templates/tool.py
```

如果 `templates/__init__.py` 变空（无 import），看一下：

```bash
cat packages/ethoinsight/ethoinsight/templates/__init__.py
```

如果是空文件或只有 `from .shoaling import *` 一行：

```bash
git rm packages/ethoinsight/ethoinsight/templates/__init__.py
rmdir packages/ethoinsight/ethoinsight/templates 2>/dev/null
```

- [ ] **Step 2.3: 删依赖旧模板的测试**

```bash
cd /home/wangqiuyang/noldus-insight
git rm packages/ethoinsight/tests/test_template_soft_gate.py
git rm packages/ethoinsight/tests/test_templates.py
git rm packages/ethoinsight/tests/test_template_tool.py
```

如果 `test_template_tool.py` 测的是 `templates/tool.py` 里现在已删的 `get_analysis_template` 工具，它已没意义。

- [ ] **Step 2.4: 跑测试**

```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
python -m pytest tests/ -v 2>&1 | tail -20
# 期望: 无新增 failure。test_parse.py 的 9 个 pre-existing 失败仍在（不是本任务）

cd /home/wangqiuyang/noldus-insight/packages/agent/backend
source .venv/bin/activate
make test 2>&1 | tail -10
# 期望: 全绿
```

- [ ] **Step 2.5: 再次 grep 全局确认无残留**

```bash
cd /home/wangqiuyang/noldus-insight
grep -rln "ethoinsight\\.templates\\|require_ev19_template\\|templates/shoaling" \
  packages/ 2>/dev/null
# 期望: 无结果（docs/ 历史描述允许保留）
```

- [ ] **Step 2.6: commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add -A
git commit -m "refactor(ethoinsight): 删除旧脚本模板 templates/（Phase 4 T2）

- 删 templates/shoaling.py（脚本模板模式，被 SOTA 否决）
- 删 templates/_gate.py（require_ev19_template 已无引用）
- 删 templates/tool.py（get_analysis_template 工具已在 Phase 4 T1 注销）
- 删依赖旧模板的测试: test_templates / test_template_soft_gate / test_template_tool

SOTA 架构下 code-executor 用 ethoinsight-code skill 的 by-paradigm/<范式>.md 自包含
所有范式信息，不再需要 dispatcher 模板。"
```

### Phase 4 完成后接手点

> 下一位 agent 接手时看这里：
> - Phase 4 完成: 7 个 langchain 工具 + 3 个 templates/ 文件 + 3 个旧测试全部删除
> - 仓库进入「干净 SOTA 架构」状态
> - Phase 5 异步在跑（同事 review）

### Phase 4 验收清单

- [ ] 7 个 langchain 工具的函数定义 + tool registry 注册全部消失
- [ ] `packages/ethoinsight/ethoinsight/templates/shoaling.py / _gate.py / tool.py` 全部消失
- [ ] `packages/ethoinsight/tests/test_templates.py / test_template_soft_gate.py / test_template_tool.py` 全部消失
- [ ] 测试全绿（含 ethoinsight + agent backend）
- [ ] grep `parse_trajectories\\|compute_metrics\\|run_statistics\\|generate_charts\\|assess_and_handoff\\|run_paradigm_analysis\\|get_analysis_template\\|require_ev19_template\\|templates\\.shoaling` 在 `packages/` 下无结果

---

## Phase 5: 同事 review checklist（异步并行）

**目标:** 行为学同事 review 工程衍生的 by-paradigm/<范式>.md 与他们维护的 by-experiment/<范式>.md 是否对齐。这是异步工作，不阻塞 Phase 1-4。

> 启动时机: Phase 1 完成后即可启动；不必等 Phase 2-4 完成。

### Task 1: 建 review 包结构

**Files:**
- Create: `docs/review-packages/2026-05-11-sota-paradigm-code/README.md`
- Create: `docs/review-packages/2026-05-11-sota-paradigm-code/by-paradigm/<6 范式>.md`（从 `packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/` 同步复制）
- Create: `docs/review-packages/2026-05-11-sota-paradigm-code/CHECKLIST.md` 同事 review 清单

- [ ] **Step 1.1: 建目录**

```bash
cd /home/wangqiuyang/noldus-insight
mkdir -p docs/review-packages/2026-05-11-sota-paradigm-code/by-paradigm
```

- [ ] **Step 1.2: 写 README.md**

文件 `docs/review-packages/2026-05-11-sota-paradigm-code/README.md`：

```markdown
# 工程衍生 SOTA 范式代码 review 包（2026-05-11）

## Review 目的

工程师按 SOTA 架构把 6 范式（EPM / OFT / Zero Maze / LDB / TST / FST）的指标算法实现 + 文档化。
本 review 包让行为学同事确认:

1. **指标列表对齐**: 工程实现的指标函数清单与同事维护的 `by-experiment/<范式>.md` 必算指标完全对应
2. **数据质量警告阈值合理**: 如 EPM 「n < 5/组」或「total_entry_count < 8」是否合行为学惯例
3. **派发器逻辑正确**: `compute_paradigm_metrics(paradigm="<X>")` 返回的字段完整

## Review 工作流

1. 打开 `CHECKLIST.md`
2. 逐范式对照 `by-paradigm/<范式>.md` 与你之前写的 `docs/review-packages/2026-04-29-ev19-templates/by-experiment/<范式>.md`
3. 在 `CHECKLIST.md` 对应项 `[ ]` 改 `[x]` 或写反馈
4. 提 PR 把 CHECKLIST 改动合入 dev

## 关联

- 同事领域知识源（你之前写的）: `docs/review-packages/2026-04-29-ev19-templates/by-experiment/`
- 工程师衍生（本 review 对象）: `docs/review-packages/2026-05-11-sota-paradigm-code/by-paradigm/`
- 实际生产用 skill 文档（与本 review by-paradigm/ 内容应一致）: `packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/`
```

- [ ] **Step 1.3: 同步复制 6 范式 by-paradigm 文档**

```bash
cd /home/wangqiuyang/noldus-insight
for p in epm oft zero-maze ldb tst fst; do
  SRC="packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/${p}.md"
  DST="docs/review-packages/2026-05-11-sota-paradigm-code/by-paradigm/${p}.md"
  if [ -f "$SRC" ]; then
    cp "$SRC" "$DST"
    echo "✅ copied $p"
  else
    echo "⚠️  source missing for $p (Phase 1-3 might not be done yet)"
  fi
done
```

> Phase 5 启动时 Phase 1 已完成 → EPM 有，OFT/其他可能缺。Phase 1-3 持续做完后会陆续补齐。

- [ ] **Step 1.4: 写 CHECKLIST.md**

文件 `docs/review-packages/2026-05-11-sota-paradigm-code/CHECKLIST.md`：

```markdown
# 工程衍生 SOTA 范式代码 Review CHECKLIST

> 行为学同事填。每范式一节，节内逐项打勾或填反馈。

## 通用 review 项（每范式都要看）

- [ ] 「可用指标函数」表里的所有函数名与 `by-experiment/<范式>.md` 「必算指标」对齐
- [ ] 「数据质量警告」阈值合行为学惯例（如 EPM `n < 5/组` 是否合理）
- [ ] 「胶水脚本范例」的 GROUPS 结构合实验设计（如对照组/治疗组的 n 数典型值）
- [ ] 「出图建议」推荐的图表类型适合本范式（不要 box_plot 用在不该 box 的指标上）
- [ ] 「反模式提醒」段没有遗漏本范式特有的判读陷阱

## EPM 范式 review

- [ ] `compute_open_arm_time_ratio` / `_entry_count` / `_entry_ratio` / `_open_arm_time` / `compute_total_entry_count` 是 EPM 必算 5 项？
- [ ] 「单向穿越定义」是 EPM 适用还是 LDB 适用？（grill-me 决策 10 是 LDB 的）
- [ ] `total_entry_count < 8 → 运动抑制混杂` 阈值是否合 mouse/rat 行业经验？
- [ ] 是否需要补「开臂头部探究次数」「滞留区段持续时间分布」等次级指标？

[填反馈或建议补的指标:]

## OFT 范式 review

- [ ] `compute_center_time_ratio` / `compute_thigmotaxis_index` 是 OFT 核心 2 项？
- [ ] 工程师补的额外函数（如 `compute_center_entry_count`）是否必要？
- [ ] 「中心区进入次数」/「中心区滞留段平均时长」哪个更重要？
- [ ] 「总移动距离」「平均速度」用通用 `compute_distance_moved / _velocity_stats` 是否够？

[填反馈:]

## Zero Maze / LDB / TST / FST 4 范式 review

每范式同上结构。Phase 3 完成后补上。

[填反馈:]

## 整体结论

- [ ] 6 范式全部 review 完
- [ ] 已与工程师沟通修改点
- [ ] 同意 commit 进 dev 分支

签名: __________  日期: __________
```

- [ ] **Step 1.5: commit + 通知同事**

```bash
cd /home/wangqiuyang/noldus-insight
git add docs/review-packages/2026-05-11-sota-paradigm-code/
git commit -m "docs(review): 工程衍生 SOTA 范式 review 包（Phase 5 T1）"
git push origin dev
```

通过项目沟通渠道（Slack/邮件/PR）@同事 review。

### Task 2: 同事 PR 反馈合入

> 同事 review 完后会提 PR 改 CHECKLIST.md。merge 后:

- [ ] **Step 2.1: 看 PR diff 看反馈**

```bash
gh pr list --search "review-packages/2026-05-11-sota-paradigm-code"
gh pr view <PR_ID>
```

- [ ] **Step 2.2: 把反馈合入 `packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/<范式>.md`**

对每条同事反馈：
- 「指标补」→ 走 Phase 1/2/3 同款 TDD 流程加函数 + 改 by-paradigm 文档
- 「阈值改」→ 改 `metrics/dispatcher.py` 里 data_quality_warnings 的判断条件 + 改 by-paradigm 文档
- 「文字改」→ 直接改 skill 文档

- [ ] **Step 2.3: 同步 review 包**

把 skill 文档改完同步回 `docs/review-packages/2026-05-11-sota-paradigm-code/by-paradigm/<范式>.md`。

- [ ] **Step 2.4: commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add -A
git commit -m "feat(skill): 合入同事 EPM review 反馈（Phase 5 T2）"
```

### Phase 5 完成后接手点

> 下一位 agent 接手时看这里：
> - Phase 5 完成: 6 范式至少 review 完 EPM + OFT；其余可滚动
> - 同事 PR 合入或在 review 队列中
> - 工程衍生 by-paradigm 与同事 by-experiment 内容对齐

### Phase 5 验收清单

- [ ] review 包结构完整（README + 6 个 by-paradigm 副本 + CHECKLIST）
- [ ] 行为学同事至少 review 完 EPM + OFT 两个范式（其余可滚动）
- [ ] review 反馈合入 `packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/`

---

## 跨 Phase 注意事项

### 每个 Phase 完成时

- 跑全套测试 `make test`（agent backend）+ `pytest tests/`（ethoinsight）
- 更新本 plan 顶部「当前阶段」字段
- 写 handoff（`docs/handoffs/2026-05-XX-<phase>-completed-handoff.md`）记录验证证据
- commit

### 风险提示

1. **Python 子包优先级:** Phase 1 T1 之后 `metrics/` 子包覆盖 `metrics.py`。如果出现 `ImportError` 或循环导入，先 `python -c "import ethoinsight.metrics; print(ethoinsight.metrics.__file__)"` 确认加载的是子包而非旧 module。
2. **skill 注入时机:** SKILL.md 改动需要 deerflow 重启才生效（skill 全文在 system prompt 启动时注入）。每次改 SKILL.md 后 `make stop && make dev` 重启。
3. **chart 路径:** code-executor 胶水脚本写 PNG 时落 `${workspace_path}/outputs/`，必须用绝对路径（lead 给的 workspace_path 已是绝对路径）。
4. **e2e 失败 ≠ 架构错:** Phase 1 T7 / Phase 2 T3 e2e 如果失败，先看 traceback 是 EPM 算法错（fix metrics/epm.py）还是流程错（fix code_executor.py / skill）。算法错占多数。

### 反模式提醒

- ❌ 不要为了让 e2e 过去改 `metrics.py` 行为（应该改 fixture 或胶水脚本）
- ❌ 不要在 SKILL.md 里写实际算法（算法在 .py，SKILL.md 给 API 视角）
- ❌ 不要把 deerflow 中间件改了（用现成的，grill-me 已校准）
- ❌ 不要让 lead 在 task() prompt 里抽 by-experiment 内容拷给 code-executor（直接告诉 paradigm name 即可，code-executor 自己 read by-paradigm）

---

## Appendix A: 错误诊断反查表

执行过程出错时按现象查这张表。多数情况已被 deerflow 现成 middleware 兜底。

| 现象 | 可能原因 | 诊断命令 | 修复方向 |
|---|---|---|---|
| `ImportError: cannot import name 'compute_X' from 'ethoinsight.metrics'` | metrics 子包 `__init__.py` 没导出 / metrics.py shim 仍在抢占 | `python -c "import ethoinsight.metrics as m; print(m.__file__); print(dir(m))"` | 检查 `metrics/__init__.py` 的 `from .X import *` + `__all__` 是否包含此函数 |
| `ModuleNotFoundError: No module named 'ethoinsight.metrics.epm'` | 子包目录或文件没建 / `__init__.py` 缺失 | `ls packages/ethoinsight/ethoinsight/metrics/` | 补建文件，再 `pip install -e packages/ethoinsight` 重装 |
| `pytest tests/test_metrics_epm.py` 全跳过 | demo-data 路径没找到 → `pytest.skip` | 看测试源码 fixture | 测试已用 `_make_epm_df` 合成数据，不应跳过；如真跳过看是哪个 fixture |
| `make test` 出 `RuntimeError: Event loop is closed` | 上游 deerflow 已修（2026-04-29 sync 82731aeb） | 看 `docs/handoffs/2026-04-29-event-loop-fix-v2-completed-handoff.md` | 如复现，回退到 4-29 修复后的 deerflow 同步状态 |
| code-executor 报 `tool 'parse_trajectories' not found` | Phase 1 T5 移除了 tools，但 lead 派遣 prompt 仍引用 | grep lead_agent prompt | 在 lead_agent/prompt.py 同步删除对老工具的描述 |
| code-executor 写胶水脚本但 bash 跑不起 | venv 路径 / PATH 没设 | 看 sandbox 日志 `make dev` 输出 | `local_sandbox.py` 的 `DEERFLOW_PATH_*` 环境变量应该自动配；如失败，看 sandbox 是否在 venv 下启动 |
| code-executor 反复跑同一个失败脚本 | loop_detection middleware 应该≥5 次后 strip | 看 langgraph 日志 `loop_detection` warning | 已自动兜底；如仍循环，看 middleware 是否 enabled |
| 图片不出现在最终报告 | view_image_middleware 没 enabled / chart path 不可达 | 看 lead/data-analyst handoff JSON 里 charts 字段 | 检查 `view_image_middleware` 在中间件链；检查 chart_files 是绝对路径 |
| skill SKILL.md 改完没生效 | deerflow 启动时注入 system prompt，需重启 | `make stop && make dev` | 重启服务 |
| skill 加载报 YAML frontmatter 错 | description 多行用 `>` 折叠语法错误 | `python -c "import yaml; yaml.safe_load(open('SKILL.md').read().split('---')[1])"` | 改 frontmatter 缩进或加 `\|` 替代 `>` |
| `pytest` 报 `pytest: command not found` | venv 没 activate | `which python` 看是否在 venv 路径 | `source packages/agent/backend/.venv/bin/activate` 或 `cd packages/ethoinsight && python -m pytest tests/` |
| dispatcher 测试报「key 'X' not in result」 | dispatcher.py 新加的 paradigm 分支漏写 metric_key | 看 dispatcher.py 的 `elif paradigm == "<X>":` 段 | 补 `m["X"] = compute_X(df)` |
| skill 文件 mv 后 import 仍找 `ethoinsight-code` | 漏 grep 漏改的引用 | `grep -rln "ethoinsight-code" packages/` | 全文替换为 `ethoinsight-code` |
| EPM e2e 跑出 `data_quality_warnings` 全是 critical | demo 数据组 n < 3 触发 critical 警告 | 看 dispatcher.py 数据质量阈值段 | 这是预期行为（demo 小样本）；用更大 n 数据验证 |
| code-executor write_file 报权限 | sandbox workspace_path 没创建 / 没写权限 | `ls -ld $WORKSPACE` | 看 sandbox 启动时 mkdir + chmod 是否正常 |
| EPM 函数返回 None 全部 | raw data 列名识别不到（不是 `in_zone_open_arm_*`） | `parse.parse_trajectories(...)` 后 print column names | 改 `_find_arm_zone_columns` 的 regex 适配新列名 |
| `git push origin dev` 被拒 | 上游有新 commit | `git fetch && git rebase origin/dev` | 同步上游再 push |

### 常见误诊（不要这样）

- ❌ 看到 pytest 9 个失败立刻怀疑自己的代码 → 这 9 个是 pre-existing test_parse.py 失败，与本任务无关
- ❌ skill 加载没反应就改中间件 → 99% 是没重启 deerflow
- ❌ ImportError 直接改回老 metrics.py → 应该修 `metrics/__init__.py` 导出
- ❌ e2e 失败立刻怀疑架构 → 看 traceback 在哪一层（多数是算法层 / 数据列名层 / fixture 层）

---

## 关键文件路径速查

| 用途 | 路径 |
|---|---|
| 本 plan 正式位置 | `docs/superpowers/plans/2026-05-11-paradigm-sota-architecture-plan.md` |
| 前置交接 | `docs/handoffs/2026-05-11-paradigm-sota-architecture-grill-handoff.md` |
| 同事维护的 EPM 知识 | `docs/review-packages/2026-04-29-ev19-templates/by-experiment/epm.md` |
| EPM 函数实现（Phase 1 T1 后） | `packages/ethoinsight/ethoinsight/metrics/epm.py` |
| EPM 测试 | `packages/ethoinsight/tests/test_metrics_epm.py` |
| code-executor 配置 | `packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py` |
| 工程师 EPM 衍生文档（Phase 1 T4 后） | `packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/epm.md` |
| EV19 模板渐进披露 skill | `packages/agent/skills/custom/ethovision-paradigm-knowledge/SKILL.md` |
| 5 月 8 日地基 spec | `docs/superpowers/specs/2026-05-08-ev19-template-skill-foundation-design.md` |
| 5 月 8 日地基 plan（已完成） | `docs/superpowers/plans/2026-05-08-ev19-template-skill-foundation-plan.md` |
| deerflow middleware 目录 | `packages/agent/backend/packages/harness/deerflow/agents/middlewares/` |

## 关联 commit / PR

- `16eeac9b` EPM 4 个新指标算法 + 数据质量警告 + 单元测试（前置）
- `aed045ac` v2 交接文档：MVP 范式知识搬入 skill 全程（含 grill-me 修订）
- `8e6a6c1c` 按 grill-me 讨论修订 ethovision-paradigm-knowledge SKILL.md

---

## 执行交接

Plan 写完，保存到 `docs/superpowers/plans/2026-05-11-paradigm-sota-architecture-plan.md`（待 Step 0.1 mv）。

**两种执行方式（用户选一）:**

1. **Subagent-Driven（推荐）** - 每 task 派一个 fresh subagent，主 agent 做 task 间 review，迭代快
2. **Inline Execution** - 当前 session 用 `superpowers:executing-plans` 批量执行，checkpoint review

无论选哪种，都走 `superpowers:subagent-driven-development` 或 `superpowers:executing-plans` skill。
