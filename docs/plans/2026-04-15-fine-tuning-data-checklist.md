# EthoInsight 微调数据采集清单

> 目标：为 Qwen3-8B 微调准备 ~1800 条 SFT + ~300 对 DPO 数据
> 训练平台：Fireworks.ai（JSONL 格式，支持 thinking traces）
> 最后更新：2026-04-15

---

## 总览

```
数据来源              预估产出        负责人          优先级    状态
─────────────────────────────────────────────────────────────────
A. Noldus 产品资料      ~200 条 SFT    产品/技术支持     P0      [ ]
B. 行为学专家知识       ~350 条 SFT    行为学专家        P0      [ ]
C. 代码/文档自动转化    ~200 条 SFT    开发者(自动化)    P1      [ ]
D. 大模型合成变体       ~1000 条 SFT   开发者(脚本)      P1      [ ]
E. E2E 测试日志录制     持续积累        开发者(钩子)      P1      [ ]
F. 公开文献提取         ~50 条 SFT     开发者(脚本)      P2      [ ]
G. 内部试用反馈         ~500 SFT       行为学专家        P2      [ ]
H. DPO 偏好标注         ~300 对 DPO    行为学专家        P2      [ ]
─────────────────────────────────────────────────────────────────
合计                    ~1800 SFT + ~300 DPO
```

---

## A. Noldus 产品资料（找产品团队 / 技术支持团队）

### A1. EthoVision XT 指标定义手册

**为什么需要**：模型必须精确理解每个指标的含义、单位、计算方式。
"Total Distance Moved" 和 "Distance to Zone" 不是常识，是 Noldus 产品特有概念。

- [ ] 获取 EthoVision XT Reference Manual（PDF）
- [ ] 提取所有可导出指标的定义（预计 200+ 个指标）
- [ ] 结构化为 JSONL：

```jsonl
{"messages": [{"role": "user", "content": "EthoVision XT 中 'Mean Velocity' 这个指标是什么意思？"}, {"role": "assistant", "content": "Mean Velocity 是 EthoVision XT 中的速度指标，定义为采样周期内动物质心的位移除以时间间隔，单位通常为 cm/s。计算公式：V = √((x₂-x₁)² + (y₂-y₁)²) / Δt。注意：这是瞬时速度的均值，不是总距离/总时间。当轨迹有断点（tracking lost）时，该时段不参与均值计算。"}]}
```

**产出**：~100 条指标定义 Q&A

### A2. 各范式 Demo 导出数据

**为什么需要**：教模型识别真实数据的格式、列名、编码规则。

- [ ] 收集 13 个范式的导出样例（.txt 和 .xlsx 各一份）

| 范式 | 优先级 | 已有 |
|------|--------|------|
| Shoaling (群体聚集) | P0 | ✅ E2E 测试用的 |
| EPM (高架十字迷宫) | P0 | [ ] |
| OFT (旷场测试) | P0 | [ ] |
| FST (强迫游泳) | P0 | [ ] |
| MWM (Morris 水迷宫) | P1 | [ ] |
| NOR (新物体识别) | P1 | [ ] |
| Social Interaction | P1 | [ ] |
| Rotarod | P1 | [ ] |
| Y-Maze / T-Maze | P1 | [ ] |
| Fear Conditioning | P2 | [ ] |
| Three-Chamber Social | P2 | [ ] |
| Light-Dark Box | P2 | [ ] |
| Tail Suspension | P2 | [ ] |

**产出**：~30 份数据文件，用于后续生成格式识别训练样本

### A3. 技术支持 FAQ

**为什么需要**：真实用户遇到的真实问题，是最好的 Q&A 训练数据。

- [ ] 导出技术支持知识库（或最常见 50 个问题）
- [ ] 筛选与数据分析相关的问题（排除安装、许可证等）
- [ ] 转化为对话格式 JSONL

常见问题示例：
- "轨迹数据中出现 -1 是什么意思？" → tracking lost 标记
- "为什么导出的速度有负值？" → 可能是角速度或数据错误
- "Multiple Body Points 和 Center Point 导出的距离为什么不一样？"

**产出**：~30 条 Q&A

### A4. Application Notes

**为什么需要**：Noldus 官方的范式操作指南，包含实验设计建议和参考值。

- [ ] 收集与 13 个范式相关的 Application Notes（PDF）
- [ ] 提取关键内容：推荐实验参数、正常参考值范围、常见错误
- [ ] 转化为 Q&A 和解读样本

**产出**：~40 条

---

## B. 行为学专家知识（找行为学研究者 / 应用科学家）

> 预估专家时间：阶段一 2-3 天，阶段二每周 2-3 小时

### B1. 范式解读规则（阶段一，最高优先级）

**请专家为每个范式写一份 1-2 页的"解读指南"**。

模板（以 EPM 为例）：

```markdown
# EPM (Elevated Plus Maze) 解读指南

## 核心指标及正常范围
- Open Arm Time%: C57BL/6 正常 25-40%, < 20% 提示焦虑样行为
- Open Arm Entries: 正常 8-15 次, 显著减少配合开臂时间减少才有意义
- Closed Arm Time%: 正常 50-65%
- Total Arm Entries: 反映运动活性，< 10 次可能提示镇静效应

## 组间差异解读
- 开臂时间↓ + 总进入次数不变 = 焦虑增加
- 开臂时间↓ + 总进入次数↓ = 可能是运动抑制而非焦虑
- 开臂时间↑ = 抗焦虑效应或冲动性增加（需结合其他范式判断）

## 常见混杂因素
- 照明条件影响基线焦虑水平
- 测试时间（光/暗周期）
- 先前测试经验（重复测试效应）

## 统计注意事项
- 开臂时间通常不符合正态分布（地板效应）
- 推荐 Mann-Whitney U 或 Kruskal-Wallis
- 效应量用 rank-biserial correlation
```

- [ ] EPM 解读指南
- [ ] OFT 解读指南
- [ ] FST 解读指南
- [ ] MWM 解读指南
- [ ] Shoaling 解读指南
- [ ] NOR 解读指南
- [ ] Social Interaction 解读指南
- [ ] 其余 6 个范式（P2）

**产出**：13 份解读指南 → 转化为 ~130 条结果解读 SFT 数据

### B2. 统计边界情况判断（阶段一）

**请专家回答 50 个边界场景**。我们提供场景，专家给出判断和理由。

示例场景清单（需要专家填写"你会怎么做"和"为什么"）：

```
场景 1: Shapiro-Wilk p = 0.06, n = 10, 两组独立设计
  → 你选参数检验还是非参数检验？为什么？

场景 2: 三组 ANOVA, Levene p = 0.04 (方差不齐), 但每组 n = 30
  → 还能用 ANOVA 吗？用 Welch 校正还是换 Kruskal-Wallis？

场景 3: 两组比较 p = 0.048, Cohen's d = 0.3
  → 统计显著但效应量很小，你怎么报告？怎么向 PI 解释？

场景 4: 重复测量数据，一只动物的第三天数据缺失
  → 删除这只动物？插值？用混合模型？

场景 5: Open Arm Time 数据中有 5/20 只动物是 0%（从未进入开臂）
  → 零膨胀数据，怎么处理？

...（共 50 个场景）
```

- [ ] 准备 50 个边界场景问卷
- [ ] 专家填写判断 + 理由
- [ ] 转化为 SFT 数据（带 `<think>` 推理过程）

**产出**：~50 条方法推荐 + ~50 条可用于 DPO 的种子

### B3. 模范解读样本（阶段一）

**请专家写 15-20 份完整的"专家级结果解读"**。

每份包含：
1. 输入：完整的统计结果 JSON（我们提供）
2. 输出：专家撰写的解读段落（300-500 字）

覆盖场景：

- [ ] 显著差异 + 大效应量（清晰结论）× 3 个范式
- [ ] 显著差异 + 小效应量（需要谨慎解读）× 3 个范式
- [ ] 不显著差异（如何正确报告阴性结果）× 3 个范式
- [ ] 多组比较（事后检验的解读）× 3 个范式
- [ ] 混杂因素存在时的解读 × 3 个范式
- [ ] 数据质量问题的解读（方差为零、异常值等）× 3 个范式

**产出**：~18 条高质量种子 → 用大模型扩展为 ~200 条变体

### B4. 审核修正（阶段二，内部试用后）

- [ ] 每周审核模型输出 10-20 条
- [ ] 标记：✅ 正确 / ⚠️ 基本正确但措辞不够专业 / ❌ 错误
- [ ] 修正错误和不专业的输出

**产出**：持续积累 SFT 增量数据和 DPO 对

---

## C. 代码 / 文档自动转化（开发者用脚本处理）

### C1. statistics.py → Q&A 对

```python
# 脚本思路：解析 if-else 决策树，生成问答对
# 输入: if n_groups == 2 and normality and equal_var: method = "t-test"
# 输出:
# Q: 数据特征: {n_groups: 2, normality: true, equal_variance: true}，应该用什么统计方法？
# A: <think>两组，正态分布，方差齐性满足，经典的独立样本t检验场景</think>
#    {method: "independent_t_test", reason: "两组独立+正态+方差齐"}
```

- [ ] 编写 `scripts/generate_stats_qa.py`
- [ ] 覆盖 statistics.py 的全部决策路径

**产出**：~80 条

### C2. SKILL.md 决策树 → 样本

- [ ] 解析 `ethoinsight/SKILL.md` 的决策树
- [ ] 解析 `ethoinsight-analysis/SKILL.md` 的工作流
- [ ] 解析 `ethoinsight-charts/SKILL.md` 的图表选择树
- [ ] 每条路径生成一个 Q&A 对

**产出**：~80 条

### C3. pytest 测试用例 → 训练数据

- [ ] 遍历 `tests/` 中行为学相关测试
- [ ] 提取 `(input, expected_output)` 对
- [ ] 转化为 JSONL

**产出**：~32 条

---

## D. 大模型合成变体（用 GLM-5.1 / GPT-4 批量生成）

### D1. 实验描述变体

用 B1 的解读指南 + C1/C2 的种子数据作为 few-shot 示例：

- [ ] 编写合成脚本 `scripts/generate_synthetic_data.py`
- [ ] 变化维度：
  - 物种：斑马鱼 / C57BL/6 小鼠 / Wistar 大鼠 / SD 大鼠 / APP/PS1 小鼠
  - 范式：13 种
  - 设计：独立两组 / 独立多组 / 重复测量 / 析因设计
  - 样本量：5-50
  - 描述风格：口语 / 书面 / 中英混合 / 带错别字
- [ ] 每个组合生成 3-5 个变体
- [ ] 人工抽样检查质量（每 100 条检查 10 条）

**产出**：~500 条设计识别 + ~300 条方法推荐 + ~200 条报告片段

### D2. Tool Calling 变体

- [ ] 基于真实 E2E 对话，生成 tool calling 变体
- [ ] 覆盖：run_paradigm_analysis / read_file / write_file / search_knowledge
- [ ] 包含多轮对话（调用 → 结果 → 追问 → 再调用）

**产出**：~200 条

---

## E. E2E 测试日志录制（开发者加钩子）

### E1. 对话录制中间件

在 agent 运行时自动录制，不需要额外工作。

- [ ] 实现 `TrainingDataMiddleware`，在每次对话结束后：
  - 保存 lead agent 的 (input, plan_output) → 设计识别样本
  - 保存 subagent 的 (task_description, execution_result) → tool calling 样本
  - 保存 data-analyst 的 (input_json, interpretation) → 结果解读样本
  - 保存 report-writer 的 (input, report) → 报告生成样本
- [ ] 输出格式：Fireworks JSONL，直接可用于训练
- [ ] 存储路径：`.deer-flow/training-data/auto-collected/`

### E2. 质量过滤

- [ ] 自动过滤：删除包含错误的对话（subagent 超时、429 失败等）
- [ ] 人工审核：标记低质量输出

**产出**：持续积累，预计内部试用 2 周后积累 100-200 条

---

## F. 公开文献提取

### F1. 范式综述论文

- [ ] 每个范式找 1-2 篇经典综述
- [ ] 提取 Methods 段落中的参考值和方法论建议
- [ ] 转化为 Q&A

关键论文清单：
- EPM: Walf & Frye (2007), Komada et al. (2008)
- OFT: Seibenhener & Wooten (2015)
- FST: Can et al. (2012), Slattery & Cryan (2012)
- MWM: Vorhees & Williams (2006)
- Shoaling: Miller & Gerlai (2012)
- NOR: Lueptow (2017)

**产出**：~30 条

### F2. APA 报告规范

- [ ] APA 第 7 版统计结果报告格式
- [ ] 常见检验方法的标准报告模板
- [ ] 转化为"给定结果 → 写出 APA 格式" 训练对

**产出**：~20 条

---

## G & H. 内部试用反馈 + DPO 标注（阶段二）

在 agent 基本可用、SFT 第一轮完成后进行。

### G1. 专家审核流程

```
专家使用 EthoInsight 分析真实数据
         │
         ▼
    模型产出结果
         │
    ┌────┴────┐
    │ 专家审核 │
    └────┬────┘
         │
    ┌────┼────────┐
    ✅    ⚠️        ❌
  正确  基本正确   错误
    │   需修正     需重写
    │     │         │
    │     ▼         ▼
    │   修正版    正确版
    │   (SFT)    (SFT)
    │     │         │
    │     ▼         ▼
    └──→ DPO 对 ←──┘
         preferred = 修正/正确版
         rejected = 模型原始版
```

### H1. DPO 数据格式

```jsonl
{"prompt": "基于以下统计结果撰写解读: {EPM数据...}", "chosen": "实验组在高架十字迷宫的开臂停留时间显著低于对照组...(专家修正版)", "rejected": "根据数据分析，两组之间存在显著差异...(模型原始版，泛泛而谈)"}
```

**产出目标**：~300 对 DPO 数据

---

## 时间线

```
第 1-2 周: 收集 A（产品资料）+ B1-B3（专家知识）+ C（自动转化）
                                                    ↓
第 2-3 周: D（大模型合成变体）+ E1（加日志钩子）
                                                    ↓
第 3 周:   SFT 第一轮微调 (~1200 条) on Fireworks
                                                    ↓
第 4-6 周: 内部试用 + G（专家审核）+ E2（持续录制）
                                                    ↓
第 6-7 周: SFT 增量微调 (+500 条)
                                                    ↓
第 7-8 周: H（DPO 标注 ~300 对）+ DPO 训练
                                                    ↓
第 8 周:   评估 → 部署 or 继续迭代
```

---

## 交付物 checklist

完成微调所需的全部交付物：

### 从产品团队获取
- [ ] EthoVision XT Reference Manual (PDF)
- [ ] 13 个范式的 demo 导出数据
- [ ] 技术支持 FAQ 导出
- [ ] Application Notes (PDF)

### 从行为学专家获取
- [ ] 13 个范式的解读指南文档
- [ ] 50 个统计边界场景的专家判断
- [ ] 15-20 份模范解读样本
- [ ] (后续) 每周审核反馈

### 开发者产出
- [ ] `scripts/generate_stats_qa.py` — statistics.py 转 Q&A
- [ ] `scripts/generate_skill_qa.py` — SKILL.md 转 Q&A
- [ ] `scripts/generate_synthetic_data.py` — 大模型合成变体
- [ ] `scripts/convert_to_fireworks_jsonl.py` — 统一转 Fireworks 格式
- [ ] `TrainingDataMiddleware` — E2E 日志录制钩子
- [ ] `training_data_v1.jsonl` — 第一轮 SFT 数据集
- [ ] `training_data_v2.jsonl` — 增量 SFT 数据集
- [ ] `dpo_pairs_v1.jsonl` — DPO 偏好数据
