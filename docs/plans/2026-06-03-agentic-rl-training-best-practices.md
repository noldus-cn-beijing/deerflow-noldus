# EthoInsight Agentic RL 训练最佳实践

> **创建日期**: 2026-06-03
> **状态**: 决策文档（待团队对齐）
> **来源**: DeepSeek-R1/V3/V4 技术报告 + 2025-2026 年 Agentic RL 文献综述 + 项目实际故障分析
> **关联文档**:
> - [docs/specs/llm-finetuning-strategy.md](../specs/llm-finetuning-strategy.md) — 原始微调策略（本文档替代其中的 CPT + RLHF 部分）
> - [docs/plans/2026-05-13-base-model-decision-memo.md](2026-05-13-base-model-decision-memo.md) — 基座模型选型
> - [docs/plans/2026-04-13-fine-tuning-small-model-design.md](2026-04-13-fine-tuning-small-model-design.md) — 历史决策记录

---

## 0. TL;DR

1. **不做 CPT（继续预训练注入行为学知识）** — 行为学知识在 ethoinsight 库中，模型只需学会操作框架
2. **算法选 GRPO，不选 PPO** — 不需要 Critic 网络，节省显存，适合 binary verifier reward
3. **奖励来自 Verifier（程序化规则），不训练 Reward Model** — ethoinsight + golden-cases 已有足够的客观验证能力
4. **先训范式 specialist，再 OPD 蒸馏** — 不要一次训全范式 agent，DeepSeek V4 的教训
5. **Naive dense reward 有害** — 每个 turn 给手工奖励可能让性能下降 14pp，需要用 IRC 方法论做实证校准
6. **RC-GRPO 解决组内方差为 0** — 用 reward-conditioning token 强制组内多样性
7. **Fission-GRPO 把错误变成训练信号** — 用已有 verifier 自动注入诊断信息到失败轨迹

---

## 1. 核心理念翻转：训操作能力，不训领域知识

### 1.1 为什么不做 CPT

项目此前文档（`llm-finetuning-strategy.md`）假设需要先 CPT 注入行为学知识。经过分析项目全部 9 个生产路径故障记录，**0 个**因为模型不懂行为学：

| 故障 | 根因 |
|---|---|
| subagent seal 卡死 | prompt 矛盾制造叙述黑洞 |
| handoff 内容为空就 seal | guardrail 只校验文件存在不校验内容 |
| lead 不读 handoff 再次复现 | prompt-only fix 不够，需 harness 硬约束 |
| chart-maker 派遣边界错误 | SKILL.md/prompt 模板/输出路径 4 处契约错乱 |
| tool call 同一 AIMessage 发两遍 | 去重机制缺失 |
| FST 卡死 parameters_used 全报 | compute 脚本没按路径裁剪参数 |
| sync 误删 3 个工具注册 | merge 冲突没发现 |

每一个都是 **agent 操作能力**问题——调度、handoff、参数传递、错误恢复、中间件顺序。

### 1.2 模型需要学什么

```
模型不需要知道：
  ✗ EPM 的焦虑指标是开臂时间百分比
  ✗ Shapiro-Wilk 的 α 阈值该设多少
  ✗ EthoVision XT 19 个模板各是什么
  ✗ Noldus 设备的采样率

模型需要知道：
  ✓ 用户说"帮我分析这个 EPM 数据"→ dispatch code-executor，不是 knowledge-assistant
  ✓ handoff JSON 里 empty metric_plan → 不能直接 seal，要报质量警告
  ✓ code-executor 返回了 KeyError → 把错误传回 lead，不自己瞎改参数重跑
  ✓ 上一步用了 Shapiro-Wilk 结果 p>0.05 → 下一步该选 t-test，不是 Mann-Whitney
```

框架是 SSOT，模型只是框架的操作系统。

---

## 2. 算法选型：GRPO，不是 PPO

### 2.1 PPO vs GRPO

| | PPO | GRPO |
|---|---|---|
| **网络数量** | Actor + Critic + Reward Model（3 个） | Actor only（1 个） |
| **显存占用** | 训练时 ~2-3x 推理 | 训练时 ~1.5x 推理 |
| **Advantage 估计** | Reward - Critic 估计值 | 组内均值/标准差归一化 |
| **多 turn agent 稳定性** | Critic 在变长轨迹上发散 | 天然适应变长 |
| **Binary reward 场景** | 需要 reward model 稠密化 | 组内归一化直接处理 |

**选 GRPO。** DeepSeek-R1/R1-Zero/V3.2 全部使用 GRPO。PPO 的 Critic 网络在 agent 变长 trajectory 上极不稳定，且在 5090 32GB 上训练 30B-A3B MoE 时 Critic 显存占用不可接受。

### 2.2 GRPO 的核心机制

```
对同一个 prompt，采样 G 个 rollout（推荐 G=4-8）
用 verifier 给每个 rollout 打分
组内归一化：Advantage_i = (r_i - mean(r_group)) / std(r_group)
用 clipped policy update + KL 惩罚更新策略
```

### 2.3 GRPO 在 Agent 场景的三个已知问题及解法

| 问题 | 描述 | 解法 |
|---|---|---|
| **组内方差为 0** | G 个 rollout 全部成功或全部失败 → advantages 全 0 → 浪费计算 | RC-GRPO（§3.3） |
| **稀疏奖励** | 只有终点奖励，中间 turn 无信号 → 模型学会尽早 seal | Turn-level advantage + IRC 校准（§3.1, §3.2） |
| **犹豫致死** | 模型 thinking 越来越长但不行动 | T²PO 的 token/turn-level uncertainty 截断（§3.4） |

---

## 3. 奖励设计

### 3.1 Naive Dense Reward 有害（IRC 核心发现）

2026 年 4 月 IRC（Iterative Reward Calibration）论文的实验结果：

> Qwen3-30B-A3B 在 Tau-Bench 上，手工设计的 per-turn dense reward 让性能**下降 14pp**

原因：手工 reward 和 GRPO 的 group-relative advantage 计算产生 **advantage misalignment**——你设计 +1，但组内归一化后梯度方向可能和你预期完全相反。

### 3.2 IRC 校准方法论

不手工赋权。用实证校准：

```
Step 1: 用当前模型跑一批 rollout（~200 条），采集所有 trajectory
Step 2: 定义候选 reward component：

  R_structure:    handoff JSON schema 合法 → 0/1
  R_tool_name:    tool call 名称在 BUILTIN_TOOLS 中 → 0/1
  R_nonempty:     handoff 核心字段非空 → 0/1
  R_correct_sub:  派遣了正确的 subagent → 0/1
  R_error_recov:  错误恢复路径正确 → -1/0/+1
  R_outcome:      最终结果 vs golden-case → 0/1

Step 3: 对每个 component 做 empirical discriminative analysis：
  - 该 component 在成功/失败 trajectory 之间的 effect size (Cohen's d)
  - 该 component 和最终 outcome 的 rank correlation (Spearman ρ)
  - 该 component 在不同范式之间的稳定性 (variance across paradigms)

Step 4: 只保留区分度显著的 component（d > 0.5 且 ρ > 0.3）
Step 5: 用 logistic regression 学习每个 component 的权重
        目标变量 = trajectory 最终成功/失败

Step 6: 每 N 步重新校准（reward 权重随模型能力变化而变化）
```

### 3.3 RC-GRPO：解决组内方差为 0

**核心思路**：在 prompt 里注入 reward-conditioning token，强制组内多样性。

```
阶段 1（SFT 预热）：
  训练数据里混入 reward-conditioning tags：
  - <|high_reward|>  + 正确 trajectory
  - <|medium_reward|> + 部分正确 trajectory
  - <|low_reward|>   + 失败 trajectory
  
  让模型学会"被要求高质量时输出高质量"

阶段 2（RC-GRPO 训练）：
  组内 G 个 rollout，每个注入不同的 reward token：
  - rollout 1: <|high_reward|>
  - rollout 2: <|medium_reward|>
  - rollout 3: <|low_reward|>
  - rollout 4: <|high_reward|>
  
  即使模型本身输出高度一致，reward-conditioning token 的差异
  也会产生 trajectory 级别的多样性 → 组内方差 > 0 → advantages 有效
```

Qwen-2.5-7B + RC-GRPO 在 BFCL v4 Multi-Turn 上**超过了所有闭源 API 模型**。

### 3.4 Verifier 设计（基于 PROVE 五组件模型）

不需要 LLM-as-judge，不需要 reward model。程序化评分：

```python
class EthoInsightVerifier:
    """
    五组件程序化奖励，基于 PROVE 框架 (arXiv:2606.03892)
    """
    
    def score(self, trajectory: list[dict], golden_case: dict) -> float:
        components = {}
        
        # 1. Graduated validity scoring（渐进合法性评分）
        components["validity"] = self._score_validity(trajectory)
        # 0 = unparseable, 1 = parseable but schema invalid, 
        # 2 = schema valid but not executable, 3 = fully valid
        
        # 2. Dependency-aware coverage（依赖感知覆盖率）
        components["coverage"] = self._score_coverage(trajectory, golden_case)
        # metric_plan.json 里的指标是否全被执行了
        
        # 3. Adaptive efficiency penalty（自适应效率惩罚）
        components["efficiency"] = self._score_efficiency(trajectory)
        # 根据任务复杂度动态调整 tool call 数量上限
        
        # 4. Tool-name signal（工具名信号）
        components["tool_name"] = self._score_tool_name(trajectory)
        # BUILTIN_TOOLS 白名单校验 + 幻觉工具名惩罚
        
        # 5. Argument-value matching bonus（参数值匹配奖励）
        components["arg_value"] = self._score_arg_value(trajectory, golden_case)
        # stat_method vs ethoinsight 决策树输出
        
        # 最终奖励 = 校准后的加权和（权重通过 IRC 确定）
        return sum(
            self.calibrated_weights[k] * v 
            for k, v in components.items()
        )
```

**已有基础设施直接复用**：

| Verifier 检查项 | 已有基础设施 |
|---|---|
| handoff JSON schema 合法 | handoff schema validator（Sprint 0 已实现） |
| tool call 名称合法 | BUILTIN_TOOLS 注册表 |
| handoff 核心字段非空 | QualityWarningBroadcastMiddleware（Sprint 1 已实现） |
| 统计方法正确 | ethoinsight.statistics 决策树 |
| 分析结果正确 | golden-cases ground truth |
| 错误恢复路径正确 | Error Simulator / 已有错误消息分类 |

---

## 4. 训练策略

### 4.1 不要一次性训全范式 Agent（DeepSeek V4 教训）

DeepSeek V3.2 → V4 的最大方法论变化：从"混合 GRPO RL"切换到**On-Policy Distillation（OPD）**。

```
V3.2：一个模型，多个 RL 奖励信号混在一起优化
  → 学数学能力退化代码能力，学 agent 能力退化推理能力
  → catastrophic forgetting

V4：独立训练 10+ specialist → OPD 蒸馏到一个 student
  → 每个 specialist 在自身领域用 GRPO 训到极致
  → reverse KL divergence 对齐 teacher logit 分布（mode-seeking）
  → 能力不退化
```

**EthoInsight 对应方案**：

```
Phase 1: 按范式独立训练 specialist
  SFT + GRPO → EPM specialist（v0.1 主范式，优先）
  SFT + GRPO → OFT specialist
  SFT + GRPO → FST specialist
  ...

Phase 2: OPD 蒸馏
  用 On-Policy Distillation 把多个 specialist 蒸馏到 Qwen3-30B-A3B
  student 自己产生 rollout，用 reverse KL 对齐每个 teacher

Phase 3: 端到端微调
  蒸馏后的统一模型 + GRPO 微调（小学习率）
```

### 4.2 Fission-GRPO：把执行错误变成训练信号

你记录的 9 个生产故障中，多个是"模型遇到错误后不知道如何恢复"。Fission-GRPO 直接解决这个问题：

```
原始失败轨迹：
  lead → dispatch code-executor → KeyError: 'treatment' → seal（失败，-10 分）
  ↑ 模型只学到"这条轨迹不好"，不知道为什么、怎么改

Fission-GRPO 增强后：
  lead → dispatch code-executor → KeyError: 'treatment'
  → Error Simulator 注入诊断: "缺少 treatment 列，检查文件表头"
  → 重新采样恢复方案:
      A: lead 调 inspect_uploaded_file 查看表头 → 发现实际列名 → 修正 handoff
      B: lead 反问用户 "数据是否包含分组信息？"
      C: lead 直接 seal 错误信息（错误恢复）
  → 选恢复成功的 trajectory → seal（成功，+5 分）

训练数据同时包含：
  - 原始失败轨迹（negative sample）
  - 裂变后的恢复轨迹（positive sample，展示正确的错误处理路径）
```

你的 Error Simulator 不需要训练——已有的基础设施直接提供诊断信号：
- ethoinsight 解析错误 → 表头信息
- handoff schema validator → 缺失字段清单
- 错误消息分类 → 错误类型标签

### 4.3 T²PO：防止训练中"犹豫致死"

你刚修过的 "subagent seal 卡死" 本质是 T²PO 描述的 hesitation 崩溃——模型 thinking 越来越长但不产生新动作。prompt-only fix 治标，T²PO 治本。

**Token 层面（TTI）**：
```
监控滑动窗口内 uncertainty 变化率 Δₜ
当 Δₜ < threshold（thinking 不再产生新信息）
→ 强制终止 thinking，让模型行动
```

**Turn 层面（TDS）**：
```
比较相邻 turn 的 uncertainty 指纹 Φₖ
当 Γₖ = |Φₖ - Φₖ₋₁| < threshold（agent 在兜圈子）
→ 终止当前 rollout，重新采样
→ 节省 ~25% 计算资源
```

---

## 5. 轨迹处理

### 5.1 Mask 掉环境输出

只对模型生成的 token 计算 loss。环境返回内容不计入梯度：

```
[模型]  dispatch code-executor, handoff={...}          ← 计 loss
[环境]  code-executor 返回: {"open_arm_time_pct": ...}  ← 不计 loss
[模型]  dispatch data-analyst, handoff={...}            ← 计 loss
[环境]  data-analyst 返回: {...}                        ← 不计 loss
[模型]  seal, content={...}                             ← 计 loss
```

实现：`role=tool` 和 `role=user` 的 token 全部 mask=0。

### 5.2 保留失败轨迹

SFT 数据不要只保留成功轨迹。保留部分失败轨迹 + reward-conditioning tag（`<|low_reward|>`），让模型学会区分好操作和坏操作。

---

## 6. 执行路径

### Phase 0：基线验证（当前 ~ 2026-06）

不微调，纯推理摸底：

1. 部署 Qwen3-30B-A3B-Instruct-2507（vLLM + NVFP4 量化）到 5090
2. 跑通现有 agent 架构（lead → code-executor → data-analyst → report-writer）
3. 用 golden-cases 测试，量化指标：
   - 端到端通过率（vs deepseek-v4-pro 基线）
   - tool calling 准确率
   - handoff 完整性（核心字段非空率）

### Phase 1：Verifier 开发 + IRC 校准（2026-06 ~ 2026-07）

1. 实现 `EthoInsightVerifier`（五组件程序化评分）
2. 跑 ~200 条 rollout，用 IRC 方法论做 empirical discriminative analysis
3. 确定每个 reward component 的校准权重
4. 同步推进 golden-cases 标注 50 → 100

### Phase 2：SFT 预热（2026-07）

1. 训练数据构建：
   - 从 auto-collected trajectory 提取正确操作序列（~1000 条）
   - 从失败 trajectory 提取 + 注入 Error Simulator 诊断 + 恢复方案（~500 条）
   - 合成变体（~500 条）
   - 全部带上 reward-conditioning tags
2. LoRA SFT 在 Qwen3-30B-A3B-Instruct
3. 评估：golden-cases 通过率 + agent 编排稳定性

### Phase 3：GRPO 训练（2026-07 ~ 2026-08）

1. 先从 Qwen3-8B 开始，验证 GRPO 循环 + reward 设计（5090 完全够用）
2. 验证通过后切 30B-A3B，正式 GRPO 训练
3. 优先训 EPM specialist（v0.1 主范式）
4. 逐个范式扩展：OFT → FST → LDB → Zero Maze → TST
5. 训练基础设施：火山引擎 verl + vLLM

### Phase 4：OPD 蒸馏（2026-08 ~ 2026-09）

1. 各范式 specialist 训完后，OPD 蒸馏到统一模型
2. 端到端 GRPO 微调（小学习率）
3. v0.1 发布（2026-09）

---

## 7. 不要做的事

| 不要 | 原因 |
|---|---|
| 不要做 CPT | 行为学知识在 ethoinsight 库中，訓進模型權重是數據冗餘 |
| 不要訓 Reward Model | 有可驗證的客觀獎勵，RM 只會引入 bias + reward hacking |
| 不要用 deepseek API 的軌跡做 GRPO | On-policy 必須用當前模型自己的 rollout |
| 不要只給終點獎勵 | 你剛修的 subagent seal 卡死就是「只有最後一步有信號」的反例 |
| 不要手工賦權 reward component | 必須用 IRC 做實證校准，否則可能 -14pp |
| 不要一次性訓全範式 agent | DeepSeek V4 教訓：先訓 specialist，再 OPD 蒸餾 |
| 不要讓模型看到 ethoinsight 源碼 | Verifier 的檢查邏輯對模型不可見，否則學的是「迎合 verifier」 |

---

## 8. 基础设施

| 组件 | 推荐 | 备注 |
|---|---|---|
| **RL 框架** | verl (VolcEngine RL) v0.4+ | DeepSeek-R1 训练底座，2026 年 agentic RL 论文最常用 |
| **推理引擎** | vLLM v0.8.5+ | 并行 rollout 采样，NVFP4 量化 |
| **分布式编排** | Ray + PyTorch | verl 内置支持 |
| **基座模型** | Qwen3-30B-A3B-Instruct-2507 | NVFP4 量化，5090 32GB 部署 |
| **训练方法** | LoRA | MoE 模型轻触式微调，不重训 router |
| **验证环境** | ethoinsight sandbox + golden-cases | 已有基础设施 |

---

## 9. 关键论文参考

| 论文 | 核心贡献 | arXiv |
|---|---|---|
| **DeepSeek-R1** | GRPO + 纯规则奖励 + 涌现推理 | Nature 2025 |
| **DeepSeek-V3.2** | 混合 GRPO RL + Specialist 蒸馏 + 四项 RL 稳定技术 | 2412.19437 |
| **DeepSeek-V4** | OPD 替代混合 RL，10+ specialist 独立训练后蒸馏 | 2026-03 |
| **RC-GRPO** | Reward-conditioning token 解决 GRPO 组内方差为 0 | 2602.03025 |
| **IRC** | Naive dense reward -14pp，实证校准方法 | 2604.02869 |
| **Fission-GRPO** | 把执行错误转化为 on-policy 纠正监督 | 2601.15625 |
| **T²PO** | Uncertainty-guided exploration 防止 hesitation 崩溃 | 2605.02178 (ICML 2026 Spotlight) |
| **PROVE** | 程序化五组件奖励 + MCP 环境 + 无 LLM-as-judge | 2606.03892 |
| **GRPO 理论分析** | Binary reward 下 GRPO = adaptive weighted contrastive loss | 2503.06639 |

---

## 10. 决策请求

请团队就以下问题对齐：

1. ✅/❌ 同意不做 CPT，训练目标从"注入领域知识"翻转为"训练 agent 操作能力"
2. ✅/❌ 同意算法选 GRPO（不选 PPO），奖励来自程序化 Verifier（不训练 Reward Model）
3. ✅/❌ 同意按范式独立训练 specialist，最后 OPD 蒸馏（不一次性训全范式 agent）
4. ✅/❌ 同意 Phase 0 基线验证（部署 30B-A3B 不微调摸底）作为微调启动前置条件
5. ✅/❌ 同意 Phase 1 优先开发 Verifier + IRC 校准（先跑 8B 验证，再切 30B）
6. ✅/❌ 同意 v0.1 微调预算上调到 $5K（对应 30B-A3B LoRA GRPO 训练成本）
