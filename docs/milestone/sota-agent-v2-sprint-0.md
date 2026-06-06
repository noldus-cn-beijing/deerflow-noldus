# SOTA Agent v2 — Sprint 0/1/2a 合 dev + seal bug 真根因 + Sprint 5.7 实施中

**状态**：in-progress
**时间跨度**：2026-05-28 ~
**dev HEAD**：`bbff6a33`

## 做了什么

从"橡皮泥 Agent"向"可复现 AI Native Agent"演进的 v2 路线图。用 Opus 对现有代码做了逐文件审计（6 处发现），跑 10 轮 grill 设计决策，锁定 10 个 sprint / 16.5 周的实施路线。Sprint 0 由独立 agent 完成（handoff schema 全面化 + 4 个 seal_*_handoff first-party tool），之后跑 FST dogfood 暴露 4 个真 bug，修复后第二次 dogfood（thread af5bc3a2）全程跑通，Sprint 1 已交给新 agent 执行。

## 关键节点

| 日期 | 事件 | handoff |
|------|------|---------|
| 2026-05-28 | v2 路线图评估 + grill 10 轮 + 5 份 sprint spec 产出 | [本次 handoff](../handoffs/2026-05/2026-05-28-sprint-0-dogfood-fix-and-roadmap-spec-handoff.md) |
| 2026-05-28 | Sprint 0 实施完成 (PR #60, commit aed27729) 合 dev | 同上 |
| 2026-05-28 | FST dogfood 暴露 4 个 bug → 全部修复 (commit db791480) | 同上 |
| 2026-05-28 | 第二次 dogfood (thread af5bc3a2) 端到端跑通 | 同上 |
| 2026-05-28 | Sprint 1 交给新 agent 执行 | — |
| 2026-05-28 | Sprint 1 复核 PASS + Banner 数据通道修复 (QualityWarningBroadcastMiddleware) 合 dev | [5/28 Sprint1+2a handoff](../handoffs/2026-05/2026-05-28-sprint1-2a-deploy-prep-and-seal-bug-handoff.md) |
| 2026-05-28 | EPM n=1 dogfood 端到端跑通 + Banner 红字渲染验证 | 同上 |
| 2026-05-28 | Sprint 2a 复核 PASS (catalog 参数 SSOT 下沉) 合 dev (commit da924145) | 同上 |
| 2026-05-28 | data-analyst seal 漏调 bug 真根因定位 (data_analyst.py 两个 step 2 编号冲突) + prompt 修复 (commit 2df10880) | 同上 |
| 2026-05-28 | Sprint 5.7 spec 产出 (handoff emission validator, harness 层兜底) | 同上 |
| 2026-05-29 | Sprint 5.7 实施文档产出 (代码核验版, 修正 spec 3 处踩空点) + 交新 agent 实施 | [5.7 实施文档](../superpowers/plans/2026-05-29-sprint-5.7-handoff-emission-validator-impl.md) |

## 当前状态

**完成项**：
- Sprint 0：handoff schema 全面化 + 4 个 seal_*_handoff tool + strict mode + guardrail
- 4 个 dogfood bug：parameters_used 允许 None / code 错误消息明确 / groups 链路闭环 / artifacts 504
- 新工具：inspect_uploaded_file（读 EV19 metadata header 自动提取分组）
- 新 skill：ethoinsight-grouping（分组链路单一权威）
- **Sprint 1 合 dev**：dispatcher.py warning 结构化 + data-analyst 透传 + 前端红字 banner + QualityWarningBroadcastMiddleware（补 spec §2.5 数据通道缺口）+ LEGACY 清理
- **Sprint 2a 合 dev**（commit da924145）：catalog 参数 C 混合 SSOT 下沉（_common.yaml 13 个 shared + 6 范式 yaml）+ loader 6 helper + dispatcher 5 处阈值改 catalog 取值
- **seal bug 真根因 + prompt 修复**（commit 2df10880）：data_analyst.py 两个 step 2 编号冲突 → step 2.7
- 全量回归：ethoinsight 504 / agent backend 3111 / frontend tsc 0 错

**进行中**：
- **Sprint 5.7**（handoff emission validator）：harness 层兜底 LLM 漏调 seal tool。实施文档已产（代码核验版，修正了原 spec 3 处会让实施 agent 踩空的点：C1 复用现成 resolve_workspace / C2 连字符 vs 下划线命名 / C3 validation helper 异常安全）。**已交新 agent 实施**。

**下一 milestone**：Sprint 5.7 实施完成 + dogfood 真机验证（临时削弱 step 2.7 → 验证 executor 标 FAILED + lead 自动 retry）；之后 Sprint 2b（参数管线 5 跳封闭，含补 immobility metric parameters_ref）

## Sprint 执行依赖图

```
Sprint 0 ✅ → Sprint 1 ✅ → Sprint 2a ✅ → Sprint 2b 🔜 → Sprint 3 → Sprint 4 → Sprint 4.5
                                                                                ↓
                                                    Sprint 5 → Sprint 5.5 → Sprint 6 → Sprint 7 → Sprint 8

Sprint 5.7 🟡 (独立, 兜底 seal 漏调, 实施中) ── 正交于 5/5.5, 任何时点可启动
```

## v2 核心设计原则（grill 10 轮锁定）

- 4 个 seal_*_handoff first-party tool：LLM 只填参数，Python 负责序列化 + atomic write + manifest
- warning code 4 一级分类：SAMPLE / MOTOR / SIGNAL / METHOD（点分隔）
- catalog 参数 C 混合 SSOT：跨范式 shared 进 _common.yaml，范式独有在各 yaml
- analysis_config_id = deterministic sha256（不 UUID）
- auto/manual 双轨：data quality guardrail 仅 manual 模式启用
- Sprint 6 砍顶层结构：复用 deerflow facts 通道（confidence=1.0）
- Sprint 7 轻量化：只做 present_assumptions tool 不做 GateProvider 强制

## 相关 handoff

- [2026-05-28 Sprint 0 dogfood 修复 + 路线图 spec 全套就绪](../handoffs/2026-05/2026-05-28-sprint-0-dogfood-fix-and-roadmap-spec-handoff.md) — 本次会话完整记录

## 相关 spec

- [roadmap v2](../plans/2026-05-28-sota-agent-7-sprint-roadmap-v2.md)
- [Sprint 0 spec](../superpowers/specs/2026-05-28-sprint-0-handoff-schema-foundation-design.md)
- [Sprint 1 spec](../superpowers/specs/2026-05-28-sprint-1-data-quality-structured-design.md) ← Sprint 1 agent 执行依据
- [Sprint 2a spec](../superpowers/specs/2026-05-28-sprint-2a-catalog-parameters-design.md)
- [Sprint 2b spec](../superpowers/specs/2026-05-28-sprint-2b-parameter-pipeline-design.md)
- [Sprint 5.7 spec](../superpowers/specs/2026-05-28-sprint-5.7-handoff-emission-validator-design.md) — handoff emission validator 设计
- [Sprint 5.7 实施文档](../superpowers/plans/2026-05-29-sprint-5.7-handoff-emission-validator-impl.md) ← **5.7 agent 执行依据**（代码核验版）
