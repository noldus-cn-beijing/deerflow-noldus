# 2026-05-26 shoaling 范式下线 + v0.1 范围公告 (PR 待建)

## 当前任务目标

明示 agent v0.1 仅支持 5 个哺乳动物焦虑/抑郁范式 (EPM/OFT/LDB/FST/Zero Maze),
其他范式 (尤其是用户实际会上传的鱼类数据) 在范式识别阶段就反馈「暂不支持」,
不再走流水线伪装出结果。

**真实驱动**: 用户 2026-05-26 反馈「我们的 agent 对于斑马鱼的范式还是不支持」。
之前 shoaling catalog/scripts/metrics 存在 ≠ "支持"; metric 函数返回 None 但
dispatcher 不报错, agent 端到端跑完得到空结果, 用户体验差。本 PR 把
"代码下线 + agent 显式反馈" 一起做。

**状态**: HEAD 在 worktree `.claude/worktrees/retire-shoaling` 分支
`fix/retire-shoaling-paradigm`. 测试全绿待 push.

## Phase 1 调查

| 类型 | 文件数 | 处理 |
|---|---|---|
| 代码实现 | catalog/scripts/metrics 实体 | 物理删除 (catalog/shoaling.yaml, metrics/shoaling.py, scripts/shoaling/) |
| 引用 (ethoinsight 库) | metrics/__init__/_common/dispatcher + catalog/loader + templates/__init__ + ev19_facts/utils/assess | 去除 shoaling 项, 保留鱼类 EV19_VARIANTS 知识 |
| Agent 行为 | lead_agent prompt + identify_ev19_template_tool + report_writer + memory prompt + prep_metric_plan_tool | 明示 v0.1 范围 + unsupported 反馈分支 |
| Tests | tests/scripts/test_shoaling_scripts.py + 5 个 test_metrics 类 + parametrize 列表 | @pytest.mark.skip (修改不删, v0.2 重启时复用) |
| Golden-case | golden-cases/case-001-shoaling-baseline + validate_golden_case.py allowlist | 物理删除 + allowlist 调整 |
| Docs/Skills 知识库 | review-packages/by-experiment + by-template 4 份鱼类 .md | **保留** (v0.2 重启时复用) |

## Phase 3 实施摘要

### A. 代码实现层删除 (6 文件 + 1 目录 + 1 golden-case 目录)

* `packages/ethoinsight/ethoinsight/catalog/shoaling.yaml`
* `packages/ethoinsight/ethoinsight/metrics/shoaling.py`
* `packages/ethoinsight/ethoinsight/scripts/shoaling/` (6 scripts)
* `golden-cases/case-001-shoaling-baseline/` (3 yaml + 1 md + 5 raw .txt)

### B. Agent 行为公告

* [lead_agent/prompt.py](../../packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py)
  - 派遣硬约束段下方加 "当前支持的范式范围 (v0.1)" 段
  - 明示 5 个已支持 + 鱼类/学习记忆等暂不支持
  - 识别到不支持范式时硬规范 ask_clarification + 禁止伪装成相近范式
  - 反问示例 options 改成 5 个支持范式

* [identify_ev19_template_tool.py](../../packages/agent/backend/packages/harness/deerflow/tools/builtins/identify_ev19_template_tool.py)
  - Step 2.5 加 unsupported 分支
  - paradigm_key ∉ SUPPORTED_PARADIGMS_V01 → 直接返回 status=unsupported
  - 返回 paradigm_label / supported_paradigms / message / hint 4 字段
  - 不进入 zone 解析 / 候选搜索流水线

### C. SUPPORTED_PARADIGMS_V01 白名单 (新增, ev19_facts.py)

```python
SUPPORTED_PARADIGMS_V01: frozenset[str] = frozenset({
    "epm", "open_field", "zero_maze", "light_dark_box", "forced_swim",
})
```

唯一真源, identify tool 用此判断.

### D. 测试同步下线 (修改不删, 14 新 skip)

* `tests/scripts/test_shoaling_scripts.py` 整文件 `pytestmark = pytest.mark.skip(...)`
* `tests/test_metrics.py` 5 个 shoaling 测试类各加 `@pytest.mark.skip` (TestComputeIID,
  TestComputeNND, TestComputeGroupPolarity, TestShoalingGroupMetricsNotFakedPerSubject,
  TestShoalingSingleSubjectBoundary)
* `tests/test_catalog.py::test_all_catalog_scripts_are_importable` parametrize 去 shoaling
* `tests/test_catalog_loader_aliases.py::_ALREADY_ALIGNED` 去 shoaling
* `tests/test_catalog_schema_v11.py` 2 个 fallback 测试换用 fst + 缺列触发
  (而非依赖 shoaling.yaml 的空 charts list)
* `scripts/validate_golden_case.py::VALID_PARADIGMS` 去 shoaling + 加 SUPPORTED / planned 注释

## 关键决策

### 知识保留, 代码删除

鱼类范式 markdown 文档 (4 份 by-experiment + 部分 by-template) 全部保留, 因为:
- 行为学同事维护的 SSOT, 删了无法恢复
- v0.2 重启鱼类范式时仍要用作渐进披露知识源
- EV19_VARIANTS 中 OpenFieldCircle-NoZones-Fish / AquariumTrack3D 等模板条目保留

而代码层 (catalog/metrics/scripts/dispatcher 分支) 全删, 因为:
- 留着会让 metric 函数返回 None 但 agent 端到端跑完仍报"成功" → 用户体验差
- 让 SUPPORTED_PARADIGMS_V01 成为唯一真源, identify tool 拒绝在范式识别阶段

### 测试 skip 而非删

用户 5-26 指令 "测试同步下线 - 修改不删":
- 整文件 module-level `pytestmark = pytest.mark.skip(...)`
- 类级 `@pytest.mark.skip(reason="shoaling paradigm retired in v0.1 (2026-05-26)")`
- v0.2 重启时只需删两行 mark 即可恢复

### Backend tests 中的 "shoaling" 字符串保留

`test_archiving_summarization.py:46` / `test_subagent_contracts.py:105,116,121` /
`test_set_viz_choice.py:68,72,84,86` / `test_training_data_middleware.py` 中
出现的 "shoaling" 字符串只是 mock/fixture 用例, 不依赖 shoaling 范式代码,
对功能无依赖. **保留**, 不算回归.

## 测试结果

| 套件 | 结果 |
|---|---|
| ethoinsight pytest | **374 passed / 64 skipped** (新增 14 个 shoaling skip; 之前 50) |
| backend pytest | **3016 passed / 19 skipped** |
| ruff check (本 PR 改的文件) | clean; 3 个 pre-existing F401/E501/F841 在 identify_tool 不算回归 |

## 提交分支

- 分支: `fix/retire-shoaling-paradigm` (基于 dev `2acae6da`)
- 文件: 36 changed (含删除); +140 / −169 lines
- worktree: `.claude/worktrees/retire-shoaling`
- gh CLI 不可用; push 后请用户去
  https://github.com/noldus-cn-beijing/noldus-insight/pull/new/fix/retire-shoaling-paradigm
  手工建 PR, body 从 /tmp/PR-retire-shoaling-description.md 复制

## 未完成事项

### 🟡 中优先级

1. **OFT/LDB/zero_maze metric 函数 zone 歧义反问机制**
   - 同事 5-13 feedback Q2: bare in_zone 列名歧义时应反问用户而非静默 None
   - metric 函数 (`oft.py::_find_center_zone_column`) 当前 silent None
   - 应抛 AmbiguousZoneError 让 agent 层接 ask_clarification
   - 独立 PR

2. **E2E 验证**
   - 上传 newdemodata 中的鱼类数据 (无, 但可用文件名含 shoaling 的合成测试) → 确认
     agent 在 identify_ev19_template 即反馈 "暂不支持"
   - 上传 5 个支持范式数据各跑一遍 → 确认无回归

### 🟢 低优先级

3. **identify_ev19_template_tool pre-existing ruff F401/E501/F841 清理** (3 处)
4. 鱼类范式 v0.2 重启时:
   - 取消 test_shoaling_scripts.py / 5 个 test_metrics 类的 @pytest.mark.skip
   - 恢复 catalog/shoaling.yaml + metrics/shoaling.py + scripts/shoaling/
   - SUPPORTED_PARADIGMS_V01 加 shoaling

## 关键文件清单

### Plan / Handoff

- 本文件
- [/tmp/PR-retire-shoaling-description.md] — PR 描述

### 改动核心

- [packages/ethoinsight/ethoinsight/ev19_facts.py](../../packages/ethoinsight/ethoinsight/ev19_facts.py) — `SUPPORTED_PARADIGMS_V01` 白名单
- [packages/agent/backend/packages/harness/deerflow/tools/builtins/identify_ev19_template_tool.py](../../packages/agent/backend/packages/harness/deerflow/tools/builtins/identify_ev19_template_tool.py) — Step 2.5 unsupported 分支
- [packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py](../../packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py) — v0.1 范围公告段

### 相关 memory

- `feedback_single_source_of_truth.md` — SUPPORTED_PARADIGMS_V01 单一真源
- `project_2026-05-26_chart_maker_fst_5_root_causes.md` — 上一个 chart 列过滤 PR
- `project_2026-05-26_chart_cols_filter_and_subtask_state.md` — 待写 (PR #46)
