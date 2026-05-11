# EPM 端到端验证 pass 交接

**日期:** 2026-05-11
**Phase:** Phase 1 Task 7
**数据路径:** `/home/wangqiuyang/DemoData/高架十字迷宫/`
**Workspace:** `/tmp/epm-e2e-test/`

## 关键产物

- handoff JSON: `/tmp/epm-e2e-test/handoff_code_executor.json` (2.3KB)
- 胶水脚本: `/tmp/epm-e2e-analysis.py`

## 验证结果

- ✅ `paradigm` == `"epm"`
- ✅ `per_subject` 含 2 个 subject
- ✅ `group_summary` 含 5 个 EPM 指标 (mean/std/n)
- ✅ `data_quality_warnings` 字段存在（2 条: n=1 critical + n<5 warning）
- ✅ handoff JSON 落盘成功

## 数据说明

4 个轨迹文件属于同一只动物（Subject 1）的 4 个 trial，所以 n=1。样本量不足警告是预期行为。

## 指标数值

| 指标 | 值 |
|------|-----|
| open_arm_time_ratio | 0.080 (8.0%) |
| open_arm_entry_count | 6 |
| open_arm_entry_ratio | 0.286 (28.6%) |
| open_arm_time | 23.56s |
| total_entry_count | 21 |

## 架构验证点

- ✅ `metrics/epm.py` 5 个函数通过 dispatcher 正确调用
- ✅ `metrics/_common.py` 通用指标（distance_moved, velocity_stats）可通过 dispatcher 访问
- ✅ 胶水脚本模式（import → compute → output）可工作
- ✅ handoff JSON schema 字段齐全

## 已知限制

- 本次 e2e 仅验证了 Python 脚本层面的胶水脚本模式，未启动完整 agent 服务（make dev）通过 code-executor subagent 跑
- 仅 1 个动物（n=1），未做组间统计检验
