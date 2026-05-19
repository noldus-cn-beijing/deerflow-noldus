# 2026-05-11 SOTA 架构迁移完成 + 真数据适配待办 交接

## 已完成

Phase 1-4 SOTA 架构迁移主体工程完成（16 commits on `dev`，无新增测试失败）：

- `metrics.py` 669行单体 → `metrics/{_common,epm,oft,shoaling,zero_maze,ldb,fst,tst,dispatcher}.py`
- `ethoinsight-analysis` → `ethoinsight-code` + 6 个 `by-paradigm/<范式>.md`
- `code_executor.py`: 5 langchain 工具 → 胶水脚本 + `["ethoinsight-code","ethoinsight-charts"]`
- 7 个废弃 langchain 工具 + 3 个旧模板文件已删除
- `ethoinsight` skill → pointer → `ethovision-paradigm-knowledge`
- 29 个指标函数，170+ 测试，全绿

## 阻塞：DemoData 不是真 raw data

Phase 1-4 用的 `/home/wangqiuyang/DemoData/` 的列名与实际 EthoVision 导出不一致（LDB 已验证：数据无 `in_zone_light` 列，函数全部返回 None）。所有代码逻辑和算法已用合成数据验证正确，但**每个范式函数的列名 auto-detect regex 需要根据真数据调校**。同事明天提供真 raw data。

## 明天接手 agent 的工作

### Step 1: 确认真数据位置

同事给的数据放好后，记下每个范式的目录路径。

### Step 2: 跑列名诊断

对每个范式目录，用以下脚本打印所有轨迹文件的列名：

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
source .venv/bin/activate
python3 -c "
from ethoinsight.parse import parse_batch
data = parse_batch(['<真数据文件1.txt>', ...])
for name, df in data['subjects'].items():
    print(f'{name}: {list(df.columns)}')
    break
"
```

### Step 3: 调 regex

对照真数据列名，修改对应 `metrics/<范式>.py` 中的 auto-detect 逻辑。需要修改的位置是每个函数里的正则匹配，例如：

```python
# 改前（基于 DemoData 猜测）
cols = [c for c in df.columns if re.search(r'in_zone.*open.?arm', c, re.I)]

# 改后（基于真数据列名，例如）
cols = [c for c in df.columns if re.search(r'in_zone.*OpenArm', c)]
```

涉及文件：

| 文件 | 当前 regex | 待验证 |
|------|-----------|--------|
| `metrics/epm.py` | `in_zone.*open.?arm` / `in_zone.*closed.?arm` | |
| `metrics/oft.py` | `in_zone.*center` / `in_zone.*(peripher\|edge\|wall\|border)` | |
| `metrics/zero_maze.py` | `in_zone.*open` / `in_zone.*closed` | |
| `metrics/ldb.py` | `in_zone_light` / `in_zone_dark` | **已验证不匹配** |
| `metrics/fst.py` | `Mobility_State` | |
| `metrics/tst.py` | `Activity_State` | |
| `metrics/_common.py` | `_find_mobility_column()` | `Mobility_State`, `Activity_State`, fallback regex |

### Step 4: 跑测试 + e2e

```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
python3 -m pytest tests/ -v -k "not test_parse"   # 确认列名改动不破已有单测

# 对每个范式跑胶水脚本 e2e
python3 -c "
from ethoinsight.parse import parse_batch
from ethoinsight.metrics import compute_paradigm_metrics
data = parse_batch(['<路径1>', ...])
result = compute_paradigm_metrics(data, paradigm='<epm|open_field|zero_maze|light_dark_box|forced_swim|tail_suspension>')
print(f'per_subject keys: {list(result[\"per_subject\"].keys())}')
print(f'first subject metrics: {list(result[\"per_subject\"].values())[0]}')
"
```

### Step 5: 如有需要，调 by-paradigm 文档

如果真数据的列名/结构有特殊之处，同步更新 `packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/<范式>.md` 的说明。

### Step 6: commit

每个范式改完单独 commit：`fix(ethoinsight): <范式> 列名 regex 适配真数据`

## 关键文件速查

| 用途 | 路径 |
|------|------|
| 指标函数 | `packages/ethoinsight/ethoinsight/metrics/{_common,epm,oft,shoaling,zero_maze,ldb,fst,tst,dispatcher}.py` |
| 测试 | `packages/ethoinsight/tests/test_metrics_{epm,oft,zero_maze,ldb,fst,tst}.py` |
| skill 文档 | `packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/{epm,oft,zero-maze,ldb,fst,tst}.md` |
| code-executor | `packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py` |
| Plan | `docs/superpowers/plans/2026-05-11-paradigm-sota-architecture-plan.md` |
| 同事领域知识 | `docs/review-packages/2026-04-29-ev19-templates/by-experiment/<范式>.md` |
