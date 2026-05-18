# 基座模型决策备忘录：从 Qwen3-8B Dense 升级到 Qwen3-30B-A3B MoE

> **创建日期**: 2026-05-13
> **关系**: 更新 [2026-04-13 微调设计](2026-04-13-fine-tuning-small-model-design.md) §基座模型 + [2026-04-21 微调策略更新](2026-04-21-finetuning-strategy-update.md) + [docs/specs/llm-finetuning-strategy.md](../specs/llm-finetuning-strategy.md) §3
> **作者**: Claude Opus 4.7 + Qiuyang（Noldus）
> **状态**: 决策提案（待团队对齐）

---

## 0. TL;DR

**提议把基座从 Qwen3-8B Dense 切换到 Qwen3-30B-A3B-Instruct-2507 MoE。**

四个支撑论点：

1. **打包卖给客户 + 客户机器统一配 5090 32GB**——硬件约束从"既要又要"收敛到单档
2. **不再保留 5060 Ti 16GB 兜底**——30B-A3B 不再需要被低端硬件否决
3. **后训练在火山引擎 / Fireworks 托管**——MoE 微调的工程复杂度不归我们扛
4. **打包体积在工作站场景不是问题**——17GB 权重 vs 5GB 权重的差异不影响交付

这四个约束变化叠加，前期"Qwen3-8B Dense 是最稳选择"的论证基础已经不存在。

---

## 1. 历史决策回顾

时间线：

| 日期 | 文档 | 决策 |
|---|---|---|
| 2026-04-09 | [llm-finetuning-strategy.md](../specs/llm-finetuning-strategy.md) | 推荐 Qwen2.5-14B |
| 2026-04-13 | [fine-tuning-small-model-design.md](2026-04-13-fine-tuning-small-model-design.md) | 改 Qwen3-8B LoRA |
| 2026-04-21 | [finetuning-strategy-update.md](2026-04-21-finetuning-strategy-update.md) | 继续锁定 Qwen3-8B + Fireworks |
| 2026-05-13 | 本文档 | **提议改 Qwen3-30B-A3B** |

四次决策都是在**当时的约束下**做的最优解，没有任何一次是错的。本次变化的触发点不是模型本身，是**产品形态确认**：

- **打包卖** → 隐私 + 离线 + 成本三大约束确立 → 后训练 + 本地部署是真需求
- **5090 单档** → 显存上限从 32GB-16GB ≈ 16GB headroom 跳到 5090 全显存可用
- **托管平台微调** → MoE 后训练的工程门槛由平台承担
- **打包体积放宽** → 工作站客户不在乎 5GB vs 17GB

---

## 2. 候选对比（约束放开后）

| 维度 | Qwen3-8B Dense | Qwen3-32B Dense | **Qwen3-30B-A3B-Instruct-2507 MoE** |
|---|---|---|---|
| 总参 / 激活 | 8B / 8B | 32B / 32B | 30B / **3B 激活** |
| 5090 部署 | FP16 ~16GB | FP8 ~32GB，紧 | NVFP4 ~24GB |
| 5090 推理速度 | ~100 tok/s | ~30-50 tok/s | **~135 tok/s** |
| KV cache 余量 | ~14GB | ~0GB（卡满） | ~8GB |
| Agent 能力（Tau2-Bench） | 基线 | 高 | **更高** |
| 长 context 表现 | 中（128K） | 紧（KV 受压） | 优（KV 余量足） |
| 后训练平台支持 | 一级（成熟） | 一级 | 一级（火山 verl / Fireworks / Unsloth） |
| 训练成本（单次实验） | $50-200 | $200-500 | $200-800 |
| 训练样本下限 | 几百条 | 1K+ | 1K+ |
| 迭代时间 | 4-8h | 12-24h | 12-24h |
| 许可证 | Apache 2.0 | Apache 2.0 | Apache 2.0 |
| 打包体积（NVFP4 量化） | ~5GB | ~16GB | ~17GB |

**核心权衡**：
- 选 Qwen3-8B：保守、稳定、能力天花板低
- 选 Qwen3-32B Dense：能力强但 5090 上 KV cache 被压垮，长 context 场景受限
- **选 Qwen3-30B-A3B：能力强 + 推理快 + KV cache 余量足，唯一缺点是训练成本高**

---

## 3. 为什么是 30B-A3B 而不是 32B Dense

这是约束放开后最容易混淆的二选一。两者在 5090 上都"能跑"，但**实际产品体验差距大**：

**Qwen3-30B-A3B 的胜负手**：

1. **推理速度 3-4 倍于 32B Dense**——MoE 每 token 只激活 3B 参数，5090 上 NVFP4 量化跑到 ~135 tok/s。Agent 多 turn 场景用户感知延迟显著降低
2. **KV cache 余量决定 context 上限**——你们 skill 渐进披露 + handoff JSON 的实际上下文经常 50K-100K token。32B Dense 把 5090 显存吃光，没空间留给 KV cache；30B-A3B 还有 ~8GB 余量
3. **同硬件并发能力更高**——如果未来要支持多用户同时跑 agent，MoE 的吞吐优势进一步放大

**32B Dense 的唯一优势是"训练简单"**，但这个优势在火山 verl / Fireworks 托管平台下被完全抹平——平台已经处理了 MoE 训练的 DeepSpeed/router/load balancing 复杂度。

---

## 4. 为什么选 Instruct-2507 而不是 base

提议直接基于 `Qwen/Qwen3-30B-A3B-Instruct-2507`（已 RLHF 完成的 instruct 版本），而不是 base 模型。

理由：
- Instruct 版本已经做完通用 SFT + RLHF，agent / tool calling 能力已经在线
- 我们的微调是"在 Instruct 上叠加领域知识 + 任务适配"，不是"从 base 重训"
- **轻触式 LoRA 对 MoE 风险最小**——不重训 router，只微调几个 expert 的领域偏好
- 与现有 deepseek-v3 lead-agent 调度模式无缝对接

---

## 5. 风险清单与缓解

| 风险 | 严重度 | 缓解方案 |
|---|---|---|
| MoE LoRA 训练比 Dense 复杂，平台支持成熟度未实测 | 中 | v0.1 验证期先跑通 1-2 次小规模 SFT 摸底 |
| 训练成本 ~3-5 倍于 8B Dense | 中 | 实验设计转为"假设驱动"，少做盲试；先 SFT 验证 ROI 再决定是否 CPT |
| CPT 在 30B-A3B 上效果可能被通用知识稀释 | 中 | **暂缓 CPT，先做 SFT**。若 SFT 后领域知识仍不足，再评估 CPT |
| NVFP4 量化生态依赖较新 vLLM / TRT-LLM | 低 | 火山 / Fireworks 都已支持；本地部署用 vLLM 0.6+ |
| 打包发行首次安装包 ~20GB | 低 | 工作站客户网络条件好；提供分包 / 离线介质方案 |
| 客户 5090 实际可分配显存因 OS + EthoVision 占用而不足 | 中 | 部署时强制设定 context budget；ArchivingSummarizationMiddleware 主动压缩历史 |
| Qwen 上游模型迭代快（3.5/3.6 已出），基座选型可能再次变 | 中 | **本次决策只锁"模型家族 + 架构"，具体版本号在每次微调启动时再确认最新稳定 release** |

---

## 6. 对现有工作的影响

| 领域 | 影响 | 处理 |
|---|---|---|
| Training data 飞轮（`packages/agent/backend/.deer-flow/training-data/`） | 无 | 数据格式与模型无关，继续积累 |
| Golden-cases 标注 | 无 | 评估基准与模型无关 |
| Agent 架构（lead / code-executor / data-analyst / report-writer） | 无 | tool schema、handoff JSON、skill 系统都不变 |
| Skill 渐进披露 | **有利** | 30B-A3B 更长 KV cache 余量直接受益 |
| GuardrailMiddleware / LoopDetectionMiddleware | 无 | 中间件层与基座无关 |
| EV19 模板地基实施计划 | 无 | spec + plan 都不依赖具体基座 |
| 现有 prompt（中文调度规则） | 验证后微调 | Instruct-2507 中文能力强于 8B，可能需要回退一些"对抗 8B"的保守约束 |
| 微调成本预算 | **上调** | 单次实验 $200-800 vs 之前 $50-200，建议 v0.1 微调预算扩到 $5K |

---

## 7. 执行路径

### 阶段 0：基线验证（2026-05 ~ 2026-06）

**不微调，纯推理摸底**：

1. 在一台 5090 工作站上部署 Qwen3-30B-A3B-Instruct-2507（vLLM + NVFP4 量化）
2. 跑通现有 agent 架构（lead → code-executor → data-analyst → report-writer）
3. 用现有 golden-cases 测试（shoaling 完整 case + 其他在标注的 case）
4. 量化指标：
   - 端到端 case 通过率（vs deepseek-v3 lead-agent 基线）
   - 平均推理速度 tok/s
   - KV cache 在 100K context 下的实际占用
   - tool calling 准确率

**这一步的产出决定后续是否值得投入微调。** 如果原始 Instruct-2507 已能跑通 80%+ golden-cases，后训练就主要做剩下 20% 的领域特化。

### 阶段 1：数据准备（与阶段 0 并行）

- Golden-cases 标注 50 → 100
- 训练数据飞轮持续运转，目标累计 1K+ 真实交互
- **不启动 CPT**，等 SFT 验证 ROI 后再评估

### 阶段 2：首次 SFT（2026-07 ~ 2026-08）

- 选平台：火山引擎 verl 或 Fireworks.ai（建议先各跑一次小规模实验对比）
- LoRA SFT，目标解决阶段 0 暴露的失败模式
- 评估基准：golden-cases 通过率 + agent 编排稳定性

### 阶段 3：v0.1 发布（2026-09）

- 部署 SFT 后的 Qwen3-30B-A3B 到客户 5090 工作站
- 打包方案：vLLM serving + NVFP4 权重 + agent 框架

### 阶段 4：后训练迭代（v0.1 后）

- 根据真实用户反馈做 DPO 或第二轮 SFT
- 评估是否需要 CPT 注入领域知识
- 评估是否需要降级到 8B 服务低端客户（如果客户群拓展到 5060 Ti 档）

---

## 8. 不做这件事的代价

如果维持 Qwen3-8B Dense：

- **产品能力天花板被压低**——8B 在 agent 编排、长 context、tool calling 准确性上明显弱于 30B-A3B
- **5090 显存大量闲置**——客户买了 32GB 显存的卡，我们只用 16GB
- **竞品差异化困难**——8B 微调的天花板，竞争对手抄起来快；30B-A3B + 领域微调的护城河更高
- **后续迁移成本累积**——现在切换需要重做基线评估，半年后再切换需要重做基线 + 已积累的 SFT 数据可能要重训

---

## 9. 决策请求

**请团队就以下问题对齐：**

1. ✅/❌ 同意将基座从 Qwen3-8B Dense 切换到 Qwen3-30B-A3B-Instruct-2507 MoE
2. ✅/❌ 同意客户硬件最低线锁定 RTX 5090 32GB，不保留 5060 Ti 16GB 兜底
3. ✅/❌ 同意 v0.1 微调预算上调到 $5K
4. ✅/❌ 同意 CPT 暂缓，先 SFT 验证 ROI
5. ✅/❌ 同意阶段 0 基线验证（不微调摸底）作为微调启动前置条件

如果以上 5 项有任一项不能对齐，本决策需要重新评估。

---

## 10. 相关文档

- [docs/specs/llm-finetuning-strategy.md](../specs/llm-finetuning-strategy.md) — 原始微调策略（基座部分需更新）
- [docs/plans/2026-04-13-fine-tuning-small-model-design.md](2026-04-13-fine-tuning-small-model-design.md) — Qwen3-8B 决策原始文档
- [docs/plans/2026-04-21-finetuning-strategy-update.md](2026-04-21-finetuning-strategy-update.md) — 上一次微调策略更新
- [docs/plans/2026-04-23-training-data-flywheel.md](2026-04-23-training-data-flywheel.md) — 训练数据飞轮设计
- [docs/sop/training-data-flywheel-sop.md](../sop/training-data-flywheel-sop.md) — 训练数据飞轮 SOP
- [golden-cases/SCHEMA.md](../../golden-cases/SCHEMA.md) — Golden-case 评估基准结构
