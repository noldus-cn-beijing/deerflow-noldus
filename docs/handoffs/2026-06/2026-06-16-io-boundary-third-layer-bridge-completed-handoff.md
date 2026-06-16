# 2026-06-16 第三层 bug 修复完成 — file→subject 标识桥接（io-boundary 续）

> 分支：`feat/ethoinsight-io-boundary-and-aggregator`（worktree `/home/wangqiuyang/noldus-insight-io-boundary`，基线 origin/dev `4441ddda`）
> 上一会话：review 核心三缺陷（已 push `9cc2549b`）+ 发现 spec 范围外第三层 bug，留决策。
> 本会话：用户选「先深挖根因再定」→ 实测坐实 → 选修法 A（parse_batch 返回 file→subject 映射，治本）→ 实施完成。

---

## 1. 决策与结论

第三层 bug（`statistics.comparisons` 在真实 EPM 数据上必然空）**已在本 PR 一并修复**，走治本修法（按文件桥接，非 index）。核心三缺陷（`9cc2549b`）+ 本次第三层修复构成完整的 statistics 链修复。

## 2. 根因（实测坐实，比 handoff 原描述更精确）

statistics 链三层标识不一致：
- `read_groups_json` 反转后 groups 成员 = **文件路径**（`{"control": ["/mnt/.../Trial 1.xlsx", ...]}`）。
- `parse_batch()["subjects"]` 的 key = **EV19 "对象名称"**，真实 EPM 数据上**全为空串** → dedup 成 `''` / `'_1'` / `'_2'` …（实测 thread `158187ef` 28 文件全空名）。
- `compute_paradigm_metrics` 用 `matched = [s for s in grp_subjects if s in per_subject]` 匹配 → 文件路径 ∉ subject key → **零交集** → `group_summary` 空 → `comparisons` 空。

**两个实测事实**（决定修法）：
1. subject 名全空串 → **无法用文件名/stem 桥接**（subject key 不含文件身份）。
2. `parse_batch` **保序**（subjects[i] ↔ 输入 paths[i]，shuffled 输入用 row-count 指纹验证），**但只对通过 `trajectory_paths` 过滤的文件**（不存在/非 ethovision 静默丢弃，`_core.py:377-385`）→ **index 桥接有静默错位风险**（handoff 原 §3 修法 A 的隐患）。

## 3. 修法（治本：按文件桥接）

- **`parse/_core.py`**：`parse_batch` 返回新增 `file_subjects: dict[str, str]`（`{输入路径: subject_key}`），在现有 parse loop 顺手记录（零额外 parse），空结果分支也补空 map。**纯增量键，向后兼容**（无测试断言 parse_batch 精确返回键集）。
- **`scripts/_cli.py`**：新增纯函数 `bridge_groups_to_subjects(groups, file_subjects)`，把 `{group: [文件路径]}` 按**文件**翻译成 `{group: [subject_key]}`。文件被过滤（不在 map）→ 只丢该成员、不错位、stderr 留痕（不静默截断）。空串 subject key 正确保留（`'' is None` 为 False）。
- **6 个 `run_groupwise_stats.py`**（epm/oft/ldb/zero_maze/fst/tst）：在 `parse_batch` 与 `compute_paradigm_metrics` 之间插一行 `groups = bridge_groups_to_subjects(groups, parsed["file_subjects"])`，import 加 `bridge_groups_to_subjects`。

## 4. 验证（红→绿 + 回归）

- **新测试** `tests/scripts/test_groups_subject_bridge.py`（9 passed）：parse_batch file_subjects 映射 + 空名现实 + bridge 纯函数（含过滤丢弃 stderr 可观测 + 空串保留）+ 端到端 red（文件路径 groups 不桥接→comparisons 空）→ green（桥接→非空）+ subprocess 真实脚本端到端。
- **revert-to-prove-red**：注释掉 EPM 脚本桥接行 → e2e subprocess 测试 FAILED（`comparisons={}`）；恢复 → PASSED。证明红锚点真依赖生产改动，非套套逻辑。
- **真实数据 smoke**：thread `158187ef` 8 文件 4-vs-4，`distance_moved`（无需列对齐）→ 未桥接 comparisons 空、桥接后产出完整 t-test/p/effect-size。
- **全量回归**：ethoinsight `863 passed, 70 skipped, 0 failed`；backend statistics 消费者 `33 passed`。
- **裸导入**：`import app.gateway` + `make_lead_agent` 均 exit 0（唯一报错是 worktree 无 `config.yaml` 的环境差异，非导入环——无 `partially initialized module`/`ImportError`）。

## 5. ⚠️ 次要发现（必须记录，非本次修复范围）— EPM 指标列对齐缺口

桥接修好后，真实 EPM（FewZones）数据上 `group_summary` 正确产出，但 **EPM 专属指标**（`open_arm_time_ratio` 等）仍全 `None` → 这些指标的 comparisons 仍空（只有 `distance_moved`/`velocity` 这类范式无关指标有值）。

**根因**（与桥接正交）：真实 FewZones EPM 原始列名是 `open`/`closed`（用户自定义归属列），EPM 指标函数期望 `in_zone_open_arm_*` 默认列名 → 不对齐时返 None。生产里 **code-executor 的 compute 脚本经 `--parameters-json` 拿到对齐后的列名**，但 `run_groupwise_stats` 这条 statistics 路径**没接收列对齐参数** → 它直接 `compute_paradigm_metrics` 无 zone-column override。

这是**独立的第四轴问题**（statistics 路径的列对齐），属已知 column-semantics-alignment track（[milestone](../../milestone/column-semantics-alignment.md) Sprint 2 / Issue #98 结构聚合家族），**不该由本 statistics 桥接修**。本次桥接是必要且正确的前置：它让 `matched`/`group_summary` 不再空；列对齐让 EPM 专属指标不再 None。两者叠加后 comparisons 才在 FewZones 数据上完整。

合成 fixture（默认列名）上 EPM 指标正常计算，所以新测试的 green 用合成数据证明桥接闭合了「matched 空」这一缺口；列对齐缺口在真实 FewZones 数据上才显现，已在此记录待单独处理。

## 6. 改动清单（`git diff 9cc2549b..HEAD`）

```
parse/_core.py                          (+file_subjects map)
scripts/_cli.py                         (+bridge_groups_to_subjects)
scripts/{epm,oft,ldb,zero_maze,fst,tst}/run_groupwise_stats.py  (+bridge call)
tests/scripts/test_groups_subject_bridge.py  (新，9 测试)
```

## 7. 下一步

- 用户建 PR（基线 `4441ddda`，与 origin/dev `21114c7e` 零文件重叠，GitHub 自动 merge dev）。
- FewZones 列对齐进 statistics 路径（§5）：单独走 column-semantics-alignment track，等行为学同事聚合方法论（Issue #98 家族）。本次不做。
