# 2026-04-21 Subtask 可见性 + 语言一致性 + 洞察深度 — 完成记录

> 对应 handoff：[2026-04-21-subtask-visibility-handoff.md](2026-04-21-subtask-visibility-handoff.md)
> 对应执行计划：`docs/plans/2026-04-21-subtask-visibility-and-language.md`（未追踪）
> 完成日期：2026-04-21
> 签名：Claude Opus 4.7 (1M context)

---

## 1. 结果

用户在本地重跑 shoaling 5 文件 E2E 分析，三类问题全部闭环：

| # | 目标 | 结果 |
|---|---|---|
| 1 | SubtaskCard 展开显示完整 CoT（不只是"读取 skill 文件"） | ✅ 通过 |
| 2 | parse_trajectories / compute_metrics 等专家工具可见，read_file/write_file/ls 不可见 | ✅ 通过 |
| 3 | Lead 主时间线仍然简洁 | ✅ 通过 |
| 4 | 全链路语言一致，无 `## Extracted Context` 英文 dump | ✅ 通过 |
| 5 | data-analyst 点名 Subject 3 + 给出反事实（排除后降至 37.23 mm） | ✅ 通过 |
| 6 | 无崩溃 / 无 setState warning | ✅ 通过 |
| 7 | 后端日志干净 | ✅ 通过 |

---

## 2. 合并的 commit（dev 分支）

按时间序，均已 push 到 `origin/dev`，HEAD 打标 `v0.1` (cd2d6aba)：

| Commit | 范围 | 说明 |
|---|---|---|
| `4927baea` C1 | 前端 | `Subtask.messages: AIMessage[]` 累积保留，替代 latestMessage 单条覆盖 |
| `918bb8a2` C2 | 前端 | SubtaskCard 展开态渲染统一 CoT 时间线 |
| `c686729f` C3 | 前端 | 拆分 `LEAD_HIDDEN_TOOL_CALL_NAMES` vs `SUBTASK_HIDDEN_TOOL_CALL_NAMES` |
| `33e273f4` C4 | 后端 | Lead prompt 加 `<用户语言锁定>` + `<回答风格>` |
| `26e585e5` C5 | 后端 | 三个 subagent prompt 各加 `<语言>` 小节 |
| `04b76199` C6 | 后端 + ethoinsight | DataAnalystHandoff.outlier_findings；per_subject 写入 handoff JSON；data-analyst prompt Step 6 按受试者检查 + 反事实 |
| `6bd12282` fix | 前端 | ToolCall bash 分支缺 description 时返回 JSX |
| `3c51a235` fix | 前端 | thread 切换时把 setSubtasks 挪到 useEffect |
| `d735fef3` fix | 前端 | useUpdateSubtask 幂等短路，避免 render 中同步 setState 死循环 |
| `81f67bd7` fix | 前端 | useUpdateSubtask 恢复双路径 — render-time mutate, SSE-time setState |
| `6a6fa91e` fix | 前端 | 导出 markdown/JSON 时过滤 hide_from_ui 消息 |
| `f45705cc` skill | skill | compaction-recovery 明确 post-compaction 行为 |
| `6fa6962a` refactor | 后端 | 统一用 handoff_*.json 在 subagent 之间交接，废除 analysis_summary.md |
| `1b605d35` prompt | 后端 | 给 data-analyst / report-writer prompt 加 JSON 字符串转义指引 |
| `cd2d6aba` tag | — | v0.1 标签 |

测试基线：backend 1660 passed / 14 skipped；ethoinsight 131 passed / 3 skipped；前端 check + build 全绿。

---

## 3. 验证证据

E2E 产物目录：
```
packages/agent/backend/.deer-flow/threads/6f046cc7-775a-4eb9-9027-2022e50781ca/user-data/workspace/
  ├── handoff_code_executor.json   ← per_subject 字段齐全
  ├── handoff_data_analyst.json    ← outlier_findings 含 Subject 3 明确点名
  ├── code_summary.json
  ├── metrics.pkl / metrics.csv
  └── statistics.json
```

关键数据点：Subject 3 `mean_nnd = 70.02223`，排除后组均值降至 37.23 mm — 与 fix3 基线一致。

---

## 4. 保留的 pre-existing issue（不动）

这些在此轮范围外，沿用 handoff §7.3 的判断：

- Pydantic `PydanticSerializationUnexpectedValue` warning 刷屏（langgraph.log）— 上游问题，不影响功能
- `test_granular_tools.py::test_full_pipeline_writes_valid_handoff` skip — demo-data/ 结构变了，已用 `test_code_summary_per_subject.py` 覆盖等价断言
- `docs/ethoinsight-architecture.html` 的 M 标记 — 来自更早会话，未清理

---

## 5. 下一阶段重心

本轮闭环了"流程层"（可见性 / 语言 / 洞察链路）。下一步转入"内容层"：行为学逻辑与判断深度、范式补全。参见：

- [roadmap.md Phase 0 M0.2 / M0.3](../roadmap.md) — EPM + Open Field 范式补全（v0.1 9 月硬指标还差 4 个范式）
- [roadmap.md Phase 0 M0.1 余项](../roadmap.md) — 429 重试策略优化
- 新建：`docs/plans/2026-04-21-behavioral-reasoning-design.md`（行为学判断能力设计，下一步通过 brainstorming 确立）
