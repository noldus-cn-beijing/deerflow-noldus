# 2026-05-12 Plan A 完成 + Plan B/C 待办 交接

## 已完成（Plan A：脚本即指标 + EPM 验证）

- ethoinsight/scripts/ 包骨架 + CLI helper（_cli.py）
- _common/ 通用脚本（compute_distance_moved / compute_velocity_stats / plot_trajectory）
- epm/ 全量脚本（5 compute + 1 plot + 1 stats）
- ScriptInvocationOnlyProvider 白名单 Guardrail + 挂载
- by-paradigm/epm.md 重写为决策手册
- ethoinsight-code SKILL.md + code_executor.py prompt 切换到脚本编排
- 23 个脚本测试（含 2 e2e）+ 20 个 Guardrail 测试 全绿
- agent backend 2223 测试全绿（5 pre-existing failures）

### 关键 commit 链

```
ea379253 skill(ethoinsight-code): 清理胶水脚本残留（Plan A T12）
0bf190bc prompt(code-executor): 切换到脚本即指标工作流（Plan A T11）
5b649db9 skill(ethoinsight-code): epm.md 重写为脚本清单 + 决策手册（Plan A T10）
f08e9c3e feat(executor): 挂载 ScriptInvocationOnlyProvider 到 subagent（Plan A T9）
4a061097 feat(guardrails): ScriptInvocationOnlyProvider 白名单 Guardrail（Plan A T8）
f03dd48d test(scripts): EPM 端到端编排测试（Plan A T7）
6d845cd6 feat(scripts): EPM plot + stats 脚本（Plan A T6）
dad2d1e7 feat(scripts): _common 通用脚本（distance/velocity/trajectory）（Plan A T5）
80dcebc2 feat(scripts): EPM 剩余 4 个 compute 脚本（Plan A T4）
69250b74 feat(scripts): EPM compute_open_arm_time_ratio 脚本（Plan A T3）
0ed57345 test(scripts): conftest fixture + 合成 EthoVision 文件烟测（Plan A T2）
d8ece4d1 feat(scripts): CLI helper + scripts 包骨架（Plan A T1）
```

## 待办（Plan B：剩余 6 个范式按 EPM 模板补齐）

按 EPM 模板复制：
1. oft/ —— compute_center_time_ratio / compute_thigmotaxis_index / compute_center_distance_ratio / compute_center_entry_count + plot_box_center + run_groupwise_stats
2. zero_maze/ —— compute_open_zone_* (4 个) + plot_box + run_groupwise_stats
3. ldb/ —— compute_light_time_ratio / compute_transition_count / compute_light_latency + plot_box + run_groupwise_stats
4. fst/ —— compute_immobility_* (3 个) + plot_box + run_groupwise_stats
5. tst/ —— compute_immobility_* (3 个) + plot_box + run_groupwise_stats
6. shoaling/ —— compute_inter_individual_distance / compute_nearest_neighbor_distance / compute_group_polarity + plot_box + run_groupwise_stats

每个范式按 EPM 模板：
- 写脚本（照 T3-T6）
- 写 by-paradigm/<paradigm>.md（照 T10）
- 加测试（照 T3-T7）

## 待办（Plan C：前端 reasoning 重复修复）

详见 docs/superpowers/specs/2026-05-12-script-per-metric-architecture-design.md §5

## 注意事项

- 真数据列名 regex 调校还没做（接续 2026-05-11-handoff.md），等同事提供真 EthoVision 数据后再做
- Plan A 验证了"agent 走脚本路径"成立，Plan B 是机械复制工作
- conftest.py 的 _df_to_ethovision_file 格式已修正（n_header_lines=6），与 parse_header 的 col_line_idx/unit_line_idx 对齐

## 关键文件速查

- Spec: docs/superpowers/specs/2026-05-12-script-per-metric-architecture-design.md
- Plan A: docs/superpowers/plans/2026-05-12-plan-a-script-per-metric-epm.md
- Plan B（计划中）: docs/superpowers/plans/<待生成>
- Plan C（计划中）: docs/superpowers/plans/<待生成>
