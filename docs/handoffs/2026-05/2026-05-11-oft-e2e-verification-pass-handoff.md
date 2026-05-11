# OFT 端到端验证 pass（部分）交接

**日期:** 2026-05-11
**Phase:** Phase 2 Task 3
**数据路径:** `/home/wangqiuyang/DemoData/旷场实验/`（仅有统计摘要文件，无轨迹 raw txt）

## 关键产物

- OFT 指标模块: `packages/ethoinsight/ethoinsight/metrics/oft.py` (4 函数)
- OFT 测试: `packages/ethoinsight/tests/test_metrics_oft.py` (9 tests)
- OFT 参考文档: `packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/oft.md`

## 验证结果

### 单元测试
- ✅ test_center_distance_ratio_range — PASS
- ✅ test_no_center_column_returns_none — PASS
- ✅ test_all_center_returns_one — PASS
- ✅ test_no_center_frames_returns_zero — PASS
- ✅ test_no_center_presence_returns_zero — PASS
- ✅ test_single_entry — PASS
- ✅ test_multiple_entries — PASS
- ✅ test_starts_in_center_counts_as_entry — PASS
- ✅ test_no_center_column_returns_none — PASS

### 合成数据验证

| 函数 | 输入 | 结果 |
|------|------|------|
| `compute_center_time_ratio` | 100 frames, 25% center | 0.2500 ✅ |
| `compute_center_distance_ratio` | 100 frames, 25% center | 0.2897 ✅ |
| `compute_center_entry_count` | 100 frames, 25% center | 1 ✅ |
| `compute_thigmotaxis_index` | r=10, periphery=0.2 | 0.4000 ✅ |

### 架构验证点

- ✅ `metrics/oft.py` 4 个函数均可正常 import + 调用
- ✅ dispatcher `paradigm="open_field"` 分支含全部 4 个 OFT 指标
- ✅ `by-paradigm/oft.md` 含函数清单 + 胶水脚本范例 + handoff schema
- ✅ SKILL.md 入口 OFT 行已去 placeholder
- ✅ SOTA 架构通用性证明: OFT 走与 EPM 完全相同的 metrics/<范式>.py + dispatcher + by-paradigm 机制

## 已知限制

- DemoData/旷场实验/ 仅有 5 个统计摘要文件（summary level），无轨迹 raw txt 文件，无法跑完整 parse→compute→handoff 胶水脚本
- 如需完整 e2e，需要 EthoVision 导出的轨迹 raw txt（类似 EPM 数据目录里的 "轨迹-*.txt" 文件）
