# 微调策略更新 + 分步执行计划

> **创建日期**: 2026-04-21
> **关系**: 补充和更新 [2026-04-13-fine-tuning-small-model-design.md](2026-04-13-fine-tuning-small-model-design.md)，不替代
> **关联文档**: [2026-04-21-behavioral-reasoning-design.md](2026-04-21-behavioral-reasoning-design.md) §3.3 §4.5 §9
> **作者**: Claude Opus 4.7 + Qiuyang（Noldus）
> **状态**: 决策版（核心方向已定，执行细节可调）

---

## 0. 这份文档是什么

[2026-04-13 微调设计文档](2026-04-13-fine-tuning-small-model-design.md) 已经定下：
- 架构：两层（规则引擎 + LLM）+ RAG
- 基座：Qwen2.5-7B / Qwen3-8B LoRA
- 策略：SFT → DPO 两阶段
- 数据：~1800 条训练数据、~300-500 对 DPO 偏好对

本文档做两件事：

1. **补上 04-13 文档的"待定事项"第 2、3、5 条**（fallback / 评估 / 部署）和第 4 条的具体化
2. **加三个新维度**：蒸馏作为主数据源、DPO 推迟到 v0.1 后、行为学同事协作的具体分步

阅读方式：**先读 04-13 建立架构全景，再读本文档看最新决策 + 执行节奏**。

---

## 1. 核心概念速查表

为方便读者，先把几个术语的差异整理清楚：

| 术语 | 是什么 | 我们要不要 |
|---|---|---|
| **Pre-training** | 在海量文本上训练 base 能力（懂语言、懂常识） | ❌ 不做（用 Qwen3-8B base） |
| **SFT**（Supervised Fine-Tuning） | 给"输入→理想输出"对，教模型跟指令产出格式化输出 | ✅ Phase 1 核心 |
| **RLHF**（Reinforcement Learning from Human Feedback） | 偏好学习的统称，通过人类对"好 vs 差"的判断提升质量 | ✅ 但用 DPO 实现 |
| **PPO**（Proximal Policy Optimization） | RLHF 的一种具体算法（强化学习），OpenAI 在 ChatGPT 上用的那个 | ❌ 太复杂，我们不做 |
| **DPO**（Direct Preference Optimization） | RLHF 的简化版，不用 RL，直接从偏好对学习。2023 提出，已成主流 | ✅ v0.1 后做 |
| **LoRA**（Low-Rank Adaptation） | 只训练小部分"补丁权重"而不动全模型，省成本、可切换 | ✅ Fireworks 默认 |
| **Knowledge Distillation**（知识蒸馏） | 用大模型（teacher）生成数据，训练小模型（student）模仿 | ✅ 主数据源 |

**一句话总结**：SFT 教模型"怎么做"，DPO 教模型"做得更好"，PPO 是 DPO 的复杂版（我们不用），蒸馏是"让 Sonnet 帮我们生成训练数据"。

---

## 2. 对 04-13 文档的关键更新

### 2.1 更新 #1：蒸馏作为第一批数据的主要来源

04-13 文档 §数据构建提到 "Claude/GPT-4 合成变体 ~500 条"——这一条要提升为**主力策略**，而不是补充。

理由：
- 当前 lead agent 跑的是 claude-sonnet-4-6，已经在生产环境积累真实 E2E 会话
- Sonnet 在我们系统里见过 skill references、handoff JSON schema、中文/英文规范——生成的数据**天然符合系统风格**
- 真实 E2E 会话（shoaling fix5 的那类）是 **on-policy 蒸馏**，分布和部署一致

**新的数据来源优先级（替代 04-13 §数据构建第一批的排序）**：

| 优先级 | 来源 | 目标量 | 成本 | 质量 |
|---|---|---|---|---|
| **1** | 现有 E2E 会话里被人工/专家认可的 (input, output) 对 | 50-150 条 | 低（筛选） | 最高（on-policy） |
| **2** | Claude Sonnet 4.6 离线生成：场景参考表 A-E × 各范式 × 多变体 | 400-500 条 | 中（API 成本） | 高（受控生成） |
| **3** | 脚本合成：statistics.py / SKILL.md / 测试用例自动转 QA | 150-200 条 | 低（一次性写脚本） | 中（结构化正确，措辞待调） |
| **4** | 行为学同事专家标注：关键场景手写（SFT 关键用例、所有异常模式模板） | 50-100 条 | 高（专家时间） | 最高 |

合计 **650-950 条**，落在 04-13 规划的 800 条量级。

### 2.2 更新 #2：DPO 时机推迟到 v0.1 后

04-13 文档 §数据构建写 "第三批 DPO 偏好数据 300 对 周 6-8"，这个时机要**推迟**。

理由：
- **DPO 的前提是有"成对偏好"数据**（chosen / rejected）。v0.1 前没有可靠的偏好信号源
- 最好的偏好数据来源是**quality-reviewer 的互审对话**：
  - `chosen` = data-analyst 修正后的终版（有数据/文献支撑）
  - `rejected` = data-analyst 初版（被 reviewer 指出问题）
- quality-reviewer 要到 v0.1 才上线（[2026-04-21-behavioral-reasoning-design.md](2026-04-21-behavioral-reasoning-design.md) §6）
- v0.1 上线后 6-8 周才能攒够 500+ 对可用数据

**新的节奏**：
- Phase 1（7-8 月）：只做 SFT，目标是能上线替代 claude-sonnet
- Phase 1.5（v0.1 后 10-11 月）：DPO 第一轮
- Phase 2+：DPO 随 quality-reviewer 互审数据持续迭代（DPO 飞轮）

### 2.3 更新 #3：训练数据带 `<think>` CoT traces

04-13 文档没有特别强调这点（roadmap 提了一句）。确认策略：

**所有 SFT 数据的 assistant 输出必须带 `<think>...</think>` 推理过程**。

Qwen3 系列原生支持 thinking，训练时保留推理过程对小模型的推理能力提升显著。生成数据时：
- E2E 会话来源：从 Sonnet 的 thinking step 抽取（DeerFlow 日志里有）
- Sonnet 离线生成：prompt 里要求输出 `<think>...</think>\n<answer>...</answer>` 结构
- 脚本合成：手工补推理骨架（花点时间但质量值得）

### 2.4 更新 #4：Fireworks LoRA 具体化

04-13 文档写 "LLaMA-Factory / Swift" 作为工具选项。**定稿用 Fireworks.ai 平台的 LoRA**（见 roadmap Phase 1），原因：

- 无需自建训练基础设施
- Fireworks 原生支持 Qwen 系列、提供 inference 端点可以直接对接
- LoRA rank/学习率默认值对 8B 模型较为合理（默认 rank=16, lr=2e-4）
- 多 LoRA 切换便于 A/B 测试

**超参数建议**（起步值，后续按 golden-case 结果调）：
- LoRA rank: 16
- LoRA alpha: 32
- Learning rate: 2e-4
- Epochs: 3
- Batch size: 按 Fireworks 默认
- Gradient checkpointing: on（节省显存）

### 2.5 更新 #5：评估方法 = golden-case 体系

04-13 §待定事项第 3 条 "评估方法" 的答案：**用 golden-case 回归套**，详见 [2026-04-21-behavioral-reasoning-design.md §9](2026-04-21-behavioral-reasoning-design.md)。

具体到微调阶段的验收：
- **SFT 基础验收**（roadmap M1.2 原定）：设计识别 > 80%、tool calling > 90%
- **SFT 质量验收**（新增，硬闸门）：golden-case 回归通过率 ≥ 与 claude-sonnet baseline 持平
- **SFT 上线前**：行为学同事 rubric 抽检（§9.7）≥ 4/5 每个维度

**没有 golden-case 就不微调**——这是硬约束（§9 埋下的）。

### 2.6 更新 #6：两个独立 checkpoint（reviewer 独立性）

[behavioral-reasoning-design.md §4.5](2026-04-21-behavioral-reasoning-design.md) 提出：data-analyst 和 quality-reviewer 在 Phase 1 微调后应该是**两个独立 checkpoint**。

v0.1 阶段：两者共享 SFT 基础，用不同 system prompt 激活不同行为（省成本）
Phase 1+：
- data-analyst checkpoint：SFT 数据偏向"生成洞察、使用术语、输出结构"
- quality-reviewer checkpoint：额外 SFT 偏向"批判性审核、引用文献、提替代解释"

各自的 LoRA 权重独立，共享 base model。Fireworks 支持运行时切换，不影响部署复杂度。

---

## 3. 分步执行计划（行为学同事协作版）

这章回答你的问题：**你一步步应该做什么，怎么和行为学同事同步**。

路径分成 **"工程线"** 和 **"行为学线"**，两线并行推进。

### 3.1 Step 1（现在 — 5 月初，2 周）：建立协作机制

**工程线**：

1. ~~完成第 9 章 golden-case 自动化 runner 基础设施（不含 case 数据）~~ 推迟到 Step 3
2. **核心动作**：用 fix5 的那份 shoaling E2E 数据构造**第一个 baseline golden-case**
   - 拷贝 `.deer-flow/threads/6f046cc7.../user-data/workspace/` 到 `golden-cases/case-001-shoaling-baseline/raw-data/`
   - 写 `metadata.yaml`（物种 zebrafish、范式 shoaling、分组）
   - 写 `expected-analysis.yaml` 的框架（先填你能填的部分）
3. 把这份 case 给行为学同事，作为他们理解"expected-analysis.yaml 要长什么样"的样板

**行为学线**（你向同事发的信息内容，可以直接用下面这段）：

> **主题**：EthoInsight v0.1 判断能力 & 微调项目的行为学协作需求
>
> 你好，我们的 EthoInsight 进入了下一阶段——从"流程能跑通"到"判断质量达到专业水准"。这一阶段的成败取决于你的专业标注。
>
> 我需要你帮忙做三件事，按优先级：
>
> **事 1（本周）**：过一遍我发给你的 `case-001-shoaling-baseline`。看看 `expected-analysis.yaml` 是不是能表达你做分析时的"期望结论"？有没有字段缺失？欢迎直接在文件里改。
>
> **事 2（4-5 月）**：提供 1 份 EPM（高架十字迷宫）的真实实验数据 + 你预期的分析结论。数据量 = 5-10 个受试者的 EthoVision 导出文件。结论 = 你会怎么解读这份数据（什么正常、什么异常、什么混杂因素）。
>
> **事 3（5-6 月）**：review 我们写的"异常诊断模式库"（文档 §8 四种经典异常）。看看分类对不对、区分标志在实战里用不用得上、有没有漏的常见异常类型。
>
> 总投入估计：每周 2-3 小时，持续 6-8 周。最终产出：一份可以让 agent 对齐专家判断的标注数据集。
>
> 数据隐私：所有数据只在本地使用，不上传到任何外部服务。

**这一步完成的标志**：
- [ ] case-001 的 raw-data 和 metadata.yaml 就绪
- [ ] 行为学同事口头确认参与、理解工作量
- [ ] 第一次协作会议（30 min），过 expected-analysis.yaml 结构

### 3.2 Step 2（5 月，4 周）：同步推进 EPM 实装 + 第一份专家数据

**工程线 — EPM 范式补全**（roadmap M0.2）：
- 按 §7.2 八件套补齐 EPM：`epm.py` 模板 + metrics 扩展 + assess 阈值 + skill reference
- 抽象 `templates/_base.py`（§7.3 的加速器）
- 单元测试 + E2E demo 跑通
- **不动微调，不动 quality-reviewer**

**行为学线**：
- 行为学同事按你们约定的 YAML schema 标注 **1 份 EPM golden-case**
- 同步 review §8 异常模式库（先不急于改定，记录反馈）

**协作节奏**：每周 1 次 30 min 同步会
- 工程这边汇报 EPM 补全进度
- 行为学这边反馈标注过程中遇到的"schema 不够用"的地方
- 双方约定下周互相交付什么

**这一步完成的标志**：
- [ ] EPM 范式端到端跑通 demo
- [ ] case-002-epm-baseline 完成（行为学同事标注）
- [ ] §8 异常模式库 v0.2（融入同事第一轮反馈）

### 3.3 Step 3（6 月，4 周）：Golden-case 自动化 + OFT + quality-reviewer 原型

**工程线**：
- `tests/test_golden_cases.py` runner：读 `expected-analysis.yaml` → 跑 E2E → 断言
- `make test-golden` 命令
- 把 case-001 和 case-002 跑通第一次自动化回归
- OFT 范式补齐（§7.6 排序 #3，预计 1 周）
- **quality-reviewer 原型实装**（降级路径版，§6.4）：
  - 新 subagent 注册
  - ReviewFinding schema 落代码
  - 集成到 lead agent 的场景参考表（但 noldus-kb 仍禁用，走降级路径）

**行为学线**：
- 标注 1 份 OFT golden-case
- 选 1 个典型 shoaling/EPM 分析，手动写一遍"理想的初版洞察 + 期望的 reviewer 反馈"——为后续微调数据和 DPO 埋种子

**这一步完成的标志**：
- [ ] Golden-case 自动化 runner 可用，3 个 case 每日回归
- [ ] OFT 范式完整
- [ ] quality-reviewer 降级版上线，在 shoaling/EPM 上 E2E 验证
- [ ] 2 份专家手写的"初版 vs 理想版"作为 DPO 种子

### 3.4 Step 4（7 月，4 周）：SFT 数据采集 + 训练（roadmap M1.1-M1.2）

**工程线**：
1. **数据采集脚本**：
   - `scripts/extract_e2e_sessions.py`：从 `.deer-flow/threads/` 和导出的 markdown 里提取高质量 (input, thinking, output) 三元组
   - `scripts/generate_synthetic_data.py`：用 claude-sonnet-4-6 API 按场景参考表 A-E 生成 400+ 条变体数据
   - `scripts/generate_stats_qa.py` + `scripts/generate_skill_qa.py`：从代码和 skill reference 自动转 QA
2. **数据筛选与标注**：
   - 把合成数据交给行为学同事人工抽样审核（抽 10%）
   - 合格率低于 80% 则调整生成 prompt 重跑
3. **格式转换**：`scripts/convert_to_fireworks_jsonl.py`，产出带 `<think>` 的 ChatML 格式
4. **上传 Fireworks 训练**：Qwen3-8B + LoRA rank 16 + 默认超参 → SFT checkpoint
5. **Golden-case 验收**：跑 3 个 case（shoaling/EPM/OFT），看和 claude-sonnet baseline 对比

**行为学线**：
- 合成数据抽样审核（约 5-10 小时工作量）
- 关键场景手写（50-100 条）：场景参考表 A-E 里最难的、异常模式 A-D 的典型解读、reviewer 的批判示范
- 第 4 份 golden-case 标注（FST 或 MWM 选一）

**这一步完成的标志**：
- [ ] SFT 训练数据集 ~800 条，带 `<think>`，合格率 > 80%
- [ ] Qwen3-8B SFT checkpoint 可跑 inference
- [ ] Golden-case 对比：SFT 模型 ≥ claude-sonnet baseline 在 3 个 case 上
- [ ] case-004 就绪（FST 或 MWM）

### 3.5 Step 5（8 月，4 周）：模型切换验证 + FST 或 MWM + v0.1 收尾

**工程线**：
- **渐进模型切换**（roadmap M1.3）：先换 subagent 模型（data-analyst / code-executor），最后换 lead——每一步跑 golden-case
- Rubric 抽检（§9.7）：行为学同事打分
- 5 个范式全部完整交付
- v0.1 上线前 checklist（配置、部署、文档）

**行为学线**：
- Rubric 打分（每范式 1-2 小时）
- 标注第 5 份 golden-case
- 为 v0.1 发布写一份"使用指南"或"测试报告"（可选）

**这一步完成的标志**：
- [ ] 5 个范式 × 5 份 golden-case 全绿
- [ ] Qwen3-8B SFT 模型在生产环境稳定运行
- [ ] Rubric 每个维度 ≥ 4/5
- [ ] **v0.1 交付**

### 3.6 Step 6（9 月后）：持续采集 + DPO 迭代

v0.1 上线后，quality-reviewer 每次互审自动产生 (初版, 反馈, 修正版) 三元组。目标 6-8 周攒 500+ 对 → Phase 1.5 DPO。

---

## 4. 行为学同事的"价值递增"路径

设计这份计划时，有意识地把同事的投入**从浅到深**排列，让他们的时间价值越来越高：

| 阶段 | 工作类型 | 门槛 | 时间投入 |
|---|---|---|---|
| Step 1 | 给一份 YAML 样板提建议 | 低（看看填什么） | 1-2 小时 |
| Step 2 | 标注 1 份 EPM golden-case | 中（按 YAML 填） | 3-5 小时 |
| Step 3 | 标注 OFT + "初版 vs 理想版"对 | 中偏高（需要写"好答案"） | 4-6 小时 |
| Step 4 | 审核合成数据 + 关键场景手写 | 高（最专业投入） | 10-15 小时 |
| Step 5 | Rubric 评分 + 最后一份 case | 中（系统性评估） | 4-6 小时 |

**总计 22-34 小时，分摊到 4 个月**——每周约 2 小时。这个负荷对一个专业身份的人是可持续的。

---

## 5. 关键风险与应对

| 风险 | 可能性 | 影响 | 应对 |
|---|---|---|---|
| 行为学同事时间不稳定 | 中 | 高 | 把他们的每一步产出放进流水线的**非阻塞位置**（比如 Step 1 的审核可以和 Step 2 的工程并行） |
| Golden-case 数据量不够 | 中 | 高 | 早期就用 fix5 的 shoaling 数据起步（Step 1），不等完美 |
| noldus-kb 一直不恢复 | 中 | 中 | quality-reviewer 的降级路径（§6.4）保证能跑 |
| SFT 后效果不如 claude-sonnet | 低 | 高 | roadmap 原定的 fallback 开关（config.yaml 切回 claude-sonnet）保留 |
| Fireworks 平台问题 | 低 | 中 | 模型权重可导出，必要时迁 vLLM 自托管 |

---

## 6. 衡量成功的硬指标（v0.1 交付时）

- 5 个范式 × 5 份 golden-case **全绿**（自动化回归通过）
- Qwen3-8B SFT 模型在所有范式上 **≥ claude-sonnet baseline**（同一套 golden-case 跑出的质量分）
- 行为学同事 rubric **每维度 ≥ 4/5**
- 训练数据集 **≥ 800 条**（带 `<think>`、合格率 > 80%）
- quality-reviewer **降级版稳定运行**，每次互审产生结构化日志（为 DPO 积累数据）

达成以上 5 条 → v0.1 交付 ✅

---

## 7. 接下来你的第一动作

1. **今天**：把 fix5 的 shoaling E2E 数据（thread 6f046cc7...）拷贝到新建的 `golden-cases/case-001-shoaling-baseline/` 目录，按本文档 §3.1 结构组织
2. **本周内**：用 §3.1 的信息模板联系行为学同事，约一次 30 min 启动会
3. **启动会前**：把本文档 + [behavioral-reasoning-design.md](2026-04-21-behavioral-reasoning-design.md) 发给他们，会上过一遍 §3 分步计划和他们的投入时间线
4. **启动会后**：把同事的反馈合入本文档 §3 / §4，作为正式项目启动

祝项目成功。
