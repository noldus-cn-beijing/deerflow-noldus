# EthoInsight 产品 Roadmap

> **时间跨度**: 2026 Q2 — 2027 Q1（12 个月）
> **关键里程碑**: **2026 年 9 月 — v0.1 可用版本交付**
> **愿景**: 从"数据分析工具"演进为"全生命周期行为学研究助手"
> **读者**: 内部开发团队
> **创建日期**: 2026-04-16
> **相关文档**: `docs/plans/` 下的各设计文档为本 roadmap 的执行细节补充

---

## 愿景

让 EthoInsight 成为一个全能的行为学研究助手——覆盖从实验设计到论文产出的全生命周期：

```
实验前                      实验中                实验后                  持续
──────────────────────────────────────────────────────────────────────────
教用户做实验           →   (用户用 Noldus 产品   →  数据分析 + 洞察   →  追问 / 异常排查
推荐范式 + 产品             做实验, 采集数据)       统计 + 图表 + 报告    知识问答
实验设计指导                                       跨范式证据链整合      文献检索
```

---

## 当前状态

| 能力 | 成熟度 | 说明 |
|------|--------|------|
| 端到端数据分析 | ⬛⬛⬛⬜⬜ | 流水线可用，但仅 shoaling 范式完整；EPM/OFT 等 10+ 范式未补全 |
| Agent 鲁棒性 | ⬛⬛⬛⬜⬜ | 循环问题已修复（`6bf51adc`），尚未 E2E 验证 |
| 知识问答 | ⬛⬛⬜⬜⬜ | 追问分析结果可用；纯知识问答偏弱（noldus-kb 禁用，skill 知识面窄） |
| 实验指导 | ⬜⬜⬜⬜⬜ | 不存在 |
| 跨范式分析 | ⬜⬜⬜⬜⬜ | 不存在 |
| 微调模型 | ⬛⬜⬜⬜⬜ | 方案已确定（Qwen3-8B + Fireworks.ai），数据采集未启动 |

---

## 时间线总览

```
2026 Q2                 2026 Q3                    2026 Q4            2027 Q1
Apr      May      Jun   Jul      Aug      Sep      Oct  Nov  Dec     Jan  Feb  Mar
├────────┼────────┤     ├────────┼────────┼───┤    ├────┴────┴────┤  ├────┴────┴────┤
 Phase 0   Phase 1        Phase 1   Phase 2  │      Phase 3          Phase 4  Phase 5
 稳固根基   微调数据+SFT   集成切换   知识升级  │      实验指导          跨范式   部署
 4周       6周             ←──────→  8周     │      8周               8周      8周
                                             │
                                        ★ v0.1 可用版本
                                        9月初交付
                                        ─────────────
                                        • 5 个范式可分析
                                        • 微调模型替代 GLM
                                        • 知识问答能推理
                                        • 数据异常可诊断
```

---

## Phase 0: 稳固根基

> **时间**: 2026 年 4 月中 — 5 月中（4 周）
> **目标**: 让现有的数据分析流水线真正可靠可用

| 里程碑 | 具体任务 | 验收标准 |
|--------|---------|---------|
| **M0.1** 鲁棒性验证 | E2E 测试 EPM 降级路径；修复 2 个 pre-existing 测试（`test_client.py` 中 `subagent_enabled` 断言）；429 重试策略优化（1s/2s → 5s/15s/30s） | agent 遇到不支持范式时优雅降级，不循环 |
| **M0.2** EPM 范式补全 | 创建 `epm.py` 模板；补全 `metrics.py` 6 个函数（closed_arm_time_ratio、center_time、entries 等）；补全 `assess.py` 阈值 | EPM 数据端到端分析跑通 |
| **M0.3** Open Field 范式 | open_field 模板/指标/阈值 | 累计 3 个范式完整可用（shoaling + epm + open_field） |
| **M0.4** 基础设施 | `read_file` UTF-16 fallback；恢复 noldus-kb（等 `180.184.84.124:7001` 恢复）；提交积压代码 | noldus-kb 可查询；无未提交改动 |

**关键文件**:
- `packages/ethoinsight/ethoinsight/templates/epm.py` — 新建，参照 `shoaling.py`
- `packages/ethoinsight/ethoinsight/templates/open_field.py` — 新建
- `packages/ethoinsight/ethoinsight/metrics.py` — 扩展 EPM/OFT 指标
- `packages/ethoinsight/ethoinsight/assess.py` — 扩展阈值
- `packages/agent/backend/packages/harness/deerflow/agents/middlewares/llm_error_handling_middleware.py` — 429 重试

**依赖**: 无外部依赖，可立即开始。**同步启动产品资料收集**（为 Phase 1 做准备）。

---

## Phase 1: 微调模型上线

> **时间**: 2026 年 5 月中 — 6 月底（6 周）
> **目标**: 用微调 Qwen3-8B 替代 GLM-5.1 API，提升领域推理能力和稳定性
> **设计文档**: `docs/plans/2026-04-13-fine-tuning-small-model-design.md`
> **数据 checklist**: `docs/plans/2026-04-15-fine-tuning-data-checklist.md`

| 里程碑 | 具体任务 | 验收标准 |
|--------|---------|---------|
| **M1.1** 数据采集 Batch A | 产品资料获取（EthoVision Reference Manual）；`statistics.py` / `SKILL.md` 自动转化脚本；LLM 合成数据生成 | ~800 条 SFT 数据（JSONL） |
| **M1.2** 首轮 SFT | Fireworks.ai 上训练 Qwen3-8B LoRA（带 `<think>` CoT traces）；本地评估 | 设计识别准确率 > 80%；tool calling 成功率 > 90% |
| **M1.3** 集成与切换 | 微调模型接入 DeerFlow（`config.yaml` 模型配置）；A/B 对比测试 vs GLM-5.1 | 3 个范式端到端分析质量不低于 GLM-5.1 |

**注意**: 原计划的 M1.4（Batch B + DPO）推迟到 v0.1 之后。先用 SFT 版本上线，DPO 作为 v0.1 后的质量迭代。

**技术决策点**:
- 若 SFT 后 tool calling 成功率 < 85% → 增加 tool calling 专项训练数据后再迭代
- 模型切换策略：**渐进式**（先替换 subagent 模型，验证后再替换 lead agent），不一次全切
- 保留 GLM-5.1 作为 fallback，`config.yaml` 中配置切换开关

**关键文件**:
- `packages/agent/config.yaml` — 模型配置切换
- 新增 `scripts/generate_stats_qa.py`、`scripts/generate_skill_qa.py`、`scripts/generate_synthetic_data.py`、`scripts/convert_to_fireworks_jsonl.py`

**前置提醒**: M1.1 需要产品团队配合提供 EthoVision XT Reference Manual 和各范式 demo 数据。**必须在 Phase 0 期间同步启动资料收集**。

---

## Phase 2: 知识问答升级

> **时间**: 2026 年 7 月 — 8 月底（8 周，与 Phase 1 尾部有 2 周重叠）
> **目标**: 让 knowledge-assistant 从"查表工具"升级为"能推理的领域专家"

| 里程碑 | 具体任务 | 验收标准 |
|--------|---------|---------|
| **M2.1** 知识体系扩充 | 扩展 skill reference 文件（范式方法学深度、实验设计指南、常见陷阱）；noldus-kb 检索质量优化 | 覆盖 5+ 范式的深度知识问答 |
| **M2.2** knowledge-assistant 增强 | 微调模型赋能推理能力（结合实验上下文回答，不是查表）；多轮追问上下文保持 | 用户满意度评估 > 7/10 |
| **M2.3** 数据异常诊断 | 扩展 data-analyst 的异常模式库；新增 skill reference（离群值判断、设备故障特征、行为表型混淆模式） | 能识别并解释 5+ 种常见数据异常 |
| **M2.4** 范式扩展 | forced_swim + morris_water_maze 模板/指标/阈值 | 累计 5 个范式完整可用 |

**关键文件**:
- `packages/agent/skills/custom/ethoinsight/references/` — 扩展知识文件
- `packages/agent/backend/packages/harness/deerflow/subagents/builtins/knowledge_assistant.py` — 增强
- `packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py` — 异常诊断能力
- `packages/ethoinsight/ethoinsight/templates/` — 新范式模板

---

## ★ v0.1 可用版本里程碑（2026 年 9 月初）

> Phase 0 + Phase 1 + Phase 2 完成后的交付物。**这是对外可演示、对内可试用的第一个版本。**

**v0.1 包含的能力**:

| 能力 | 具体表现 |
|------|---------|
| 数据分析 | 5 个范式端到端可用（shoaling, epm, open_field, forced_swim, morris_water_maze） |
| 自有模型 | 微调 Qwen3-8B 替代 GLM-5.1，无外部 API 依赖 |
| 知识问答 | 追问分析结果 + 纯领域知识问答，能推理不只查表 |
| 异常诊断 | 识别 5+ 种常见数据异常（离群值、运动量异常、分组不均等） |
| 鲁棒性 | 不支持的范式优雅降级，不循环 |

**v0.1 不包含的能力**（留给后续 Phase）:
- 实验指导（Phase 3）
- 跨范式证据链整合（Phase 4）
- 多用户 / 本地部署（Phase 5）
- DPO 质量优化（Phase 3 后启动）
- 剩余 6 个范式（Phase 3-4）

**一句话验收**: "上传 EPM 或旷场数据，agent 自动分析出报告；问它'为什么 p 值不显著'，能结合数据上下文给出专家级回答。"

---

> **时间**: 2026 年 9-11 月（8 周，v0.1 之后）
> **目标**: 新增"教用户做实验"能力，结合 Noldus 产品知识；同步启动 DPO 质量优化

| 里程碑 | 具体任务 | 验收标准 |
|--------|---------|---------|
| **M3.1** 产品知识体系 | EthoVision XT 操作指南结构化；设备配置知识库；范式 setup 流程文档化 | 作为 skill reference 或 MCP 服务可查询 |
| **M3.2** experiment-advisor | 新建 subagent：根据研究问题推荐范式、设备、EthoVision 配置；输出结构化实验方案 | "我想研究焦虑" → 完整实验方案（范式选择、设备清单、EthoVision 设置步骤） |
| **M3.3** 路由升级 | Lead agent 多意图识别（实验指导 / 数据分析 / 知识问答 / 产品操作）；微调路由能力 | 混合意图正确拆解率 > 80% |
| **M3.4** 实验→分析闭环 | experiment-advisor 输出的实验方案与分析流水线参数衔接（范式、分组、指标） | 用户按方案做完实验后，上传数据直接进入对应范式分析 |
| **M3.5** DPO + Batch B | 内部试用收集 ~500 条专家标注；DPO 训练 ~300 对；范式扩展：y_maze, novel_object | 报告质量提升；累计 7 个范式可用 |

**技术决策点**:
- **产品知识存储方式**：前期用 skill reference + noldus-kb MCP 组合（灵活、可更新）；后期将高频产品知识微调进模型（减少 context 开销）
- **路由升级方式**：建议在 Phase 1 的 SFT 数据中**提前加入意图分类样本**（~200 条），Phase 3 时路由能力已部分内化在微调模型中

**关键文件**:
- `packages/agent/backend/packages/harness/deerflow/subagents/builtins/experiment_advisor.py` — 新建
- `packages/agent/skills/custom/ethoinsight-products/` — 新建 skill（产品知识）
- `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` — 路由规则升级

---

## Phase 4: 多范式与跨实验分析

> **时间**: 2026 年 11 月 — 2027 年 1 月（8 周）
> **目标**: 实现跨范式证据链整合，覆盖剩余范式
> **产品概览**: `docs/plans/2026-04-02-ethoinsight-product-overview.md` 中描述的"多范式持续分析"场景

| 里程碑 | 具体任务 | 验收标准 |
|--------|---------|---------|
| **M4.1** 跨范式分析 | cross-paradigm 对比分析引擎（多范式方向一致性评估、证据强度打分） | "综合 EPM + OFT + 明暗箱，焦虑表型是否一致？" → 跨范式综合报告 |
| **M4.2** 长会话上下文 | 结构化 session state（当前实验范式、设备、分组、已有结论持久化）；summarization 后核心事实不丢 | 跨多轮对话（实验指导→分析→追问）关键上下文不丢失 |
| **M4.3** 剩余范式补全 | social_interaction, three_chamber, light_dark, o_maze 等 | 累计 11 个范式全部可用 |
| **M4.4** 全生命周期验证 | 端到端场景测试：实验指导→数据分析→追问→跨范式综合 | 3 个完整用户旅程跑通 |

**关键文件**:
- `packages/ethoinsight/ethoinsight/cross_paradigm.py` — 新建
- `packages/agent/backend/packages/harness/deerflow/agents/thread_state.py` — 扩展结构化 session state
- `packages/agent/backend/packages/harness/deerflow/agents/middlewares/archiving_summarization.py` — summarization 优化

---

## Phase 5: 部署与规模化

> **时间**: 2027 年 1-3 月（8 周）
> **目标**: 从内部工具到可交付产品
> **部署方案**: `docs/plans/2026-04-08-multi-user-deployment.md`

| 里程碑 | 具体任务 | 验收标准 |
|--------|---------|---------|
| **M5.1** 多用户支持 | 认证贯通（better-auth）；Thread / Memory 按 user_id 隔离 | 10 人并发内测无数据串扰 |
| **M5.2** 本地部署 | RTX 4090 单卡部署（vLLM + 微调模型）；Docker Compose 一键部署 | 客户环境可独立运行，无外部 API 依赖 |
| **M5.3** 持续学习闭环 | TrainingDataMiddleware 自动录制对话为 JSONL；用户反馈 → SFT/DPO 增量训练流程 | 每月可增量训练一次 |
| **M5.4** 产品化收尾 | 错误处理完善；性能优化；用户文档 | 可交付给第一批外部用户 |

---

## 范式覆盖进度

| Phase | 新增范式 | 累计可用 | 覆盖领域 |
|-------|---------|---------|---------|
| 当前 | shoaling | 1 | 社交行为（鱼） |
| Phase 0 | epm, open_field | 3 | + 焦虑行为、探索行为 |
| Phase 2 | forced_swim, morris_water_maze | 5 | + 抑郁样行为、空间学习记忆 |
| ★ **v0.1** | — | **5** | **9 月交付** |
| Phase 3 | y_maze, novel_object | 7 | + 工作记忆、认知 |
| Phase 4 | social_interaction, three_chamber, light_dark, o_maze | 11 | + 社交行为（鼠）、焦虑补全 |

**优先级原则**: 焦虑范式优先（EPM、OFT、明暗箱 — 用户需求最高频），其次认知范式（MWM、Y-maze、NOR），最后社交范式。

---

## 贯穿全程的工作

以下工作持续进行，不归属于某个 Phase：

| 工作 | 节奏 | 负责方 |
|------|------|--------|
| 微调数据持续采集 | 每个 Phase 产出新训练数据 | 开发 + 行为学专家 |
| noldus-kb 内容扩充 | 配合范式扩展补充论文 | 开发 + 产品团队 |
| Skill reference 维护 | 随领域知识积累持续更新 | 开发 |
| 上游 DeerFlow 同步 | 每 2-4 周 | 开发（按 `docs/sop/deerflow-sync-sop.md`） |
| 微调模型增量训练 | Phase 1 后每月一次 | 开发 |

---

## 风险与缓解

| 风险 | 影响 Phase | 缓解措施 |
|------|-----------|---------|
| 微调效果不达预期 | P1 → 全部延迟 | 保留 GLM-5.1 作为 fallback；SFT 后快速评估，不达标则增加数据再迭代 |
| 产品团队资料配合慢 | P1（数据）、P3（产品知识） | Phase 0 期间提前启动资料收集；优先获取 EthoVision Reference Manual |
| noldus-kb 服务不稳定 | P0、P2 | skill reference 覆盖核心知识；noldus-kb 定位为增强而非必需 |
| 范式扩展工作量大 | P0、P2、P4 | 按优先级分批；模板化结构降低边际成本（后续范式复用 shoaling.py 骨架） |
| GLM-5.1 API 持续不稳定 | P0 期间 | 429 重试优化先行；P1 微调模型上线后彻底脱离依赖 |
| 长会话 context 溢出 | P4 | P4 前先在 P2 积累经验；结构化 session state 比聊天记录压缩更可靠 |

---

## 关键依赖关系

```
Phase 0 ──→ Phase 1 ──→ Phase 2 ──→ ★ v0.1 ──→ Phase 3 ──→ Phase 4 ──→ Phase 5
4周          6周          8周        9月交付      8周          8周          8周
              │                                    ▲
              │                                    │
              └── SFT 数据中提前加入意图分类样本 ───┘
                  （为 P3 路由升级做准备）
```

- Phase 0 → Phase 1：需要 3+ 范式做 A/B 对比测试
- Phase 1 → Phase 2：知识问答升级核心靠微调推理能力
- **v0.1 是 Phase 2 完成后的交付节点**
- Phase 3 可在 v0.1 交付后立即启动，DPO 和实验指导并行
- Phase 3 的路由升级可以在 Phase 1 **提前埋种**（SFT 数据加意图分类样本）
- Phase 5 的多用户支持技术上独立，可以根据业务需要提前

---

## 每个 Phase 的核心交付物

| Phase | 核心交付物 | 一句话验收 |
|-------|-----------|-----------|
| 0 | 3 个可用范式 + 鲁棒 agent | "上传 EPM 数据，能分析或优雅降级" |
| 1 | 微调模型替代 GLM-5.1 | "用自己的模型跑完分析，质量不降" |
| 2 | 能推理的知识助手 + 5 范式 | "问深度方法学问题，给出专家级回答" |
| ★ **v0.1** | **9 月可用版本** | **"上传数据自动分析；追问能答；不依赖外部 API"** |
| 3 | 实验指导 + DPO + 7 范式 | "说'我想研究焦虑'，输出完整实验方案" |
| 4 | 跨范式证据链 + 11 范式 | "综合 3 个范式结果，判断表型一致性" |
| 5 | 可交付产品 | "客户拿到后一键部署，10 人同时用" |
