# EthoInsight 知识层架构设计

> **创建日期**: 2026-06-03
> **状态**: 设计文档（待团队对齐）
> **关联文档**:
> - [docs/plans/2026-06-03-agentic-rl-training-best-practices.md](2026-06-03-agentic-rl-training-best-practices.md) — Agentic RL 训练最佳实践
> - [docs/specs/llm-finetuning-strategy.md](../specs/llm-finetuning-strategy.md) — 原始微调策略（其中 CPT 注入行为学知识的部分本文档替代）
> - [docs/plans/2026-05-13-base-model-decision-memo.md](2026-05-13-base-model-decision-memo.md) — 基座模型选型

---

## 0. TL;DR

1. **不做 CPT**——行为学知识不进模型权重，模型只训 agent 操作能力
2. **不同知识类型用不同架构**——不是所有"外部知识"都能用同一种 RAG
3. **四类知识 × 四种架构 × 一个 Router**：

| 知识类型 | 问题形态 | 架构 | 延迟 |
|---|---|---|---|
| 基础知识 / 产品参数 | "C57BL/6 的 EPM 基线是多少？" | CAG prefix（常驻 KV cache） | 0ms |
| 产品操作步骤 | "怎么设置 multiple body points？" | Manual RAG（章节树索引 + 版本过滤） | ~200ms |
| 设备排错 | "distance_moved 全是 0 怎么回事？" | GraphRAG KG（因果图遍历） | ~100ms |
| 文献证据 | "有类似效应的文献报道吗？" | SciRAG（多轮检索-验证-合成） | ~1-3s |

4. **知识层和模型层解耦**——模型通过 Agentic RL 学会"何时调哪个 tool"，tool 内部是确定性的 RAG pipeline，模型不需要知道检索逻辑

---

## 1. 为什么不做 CPT（重申）

CPT（继续预训练注入领域知识）在本项目中的根本问题：

### 1.1 三类知识中，CPT 能处理的只有一类

| 知识类型 | CPT 能解决吗 | 为什么 |
|---|---|---|
| 行为学基础知识 / 产品参数 | 勉强可以 | 但知识会变（产品迭代），CPT 后的模型权重是 frozen 的，更新成本高 |
| 产品操作步骤 | 不能 | 操作步骤是程序性知识，需要精确复现。生成模型天然不适合精确复述长步骤 |
| 设备排错 | 不能 | 排错需要确定性的因果推理，不是概率生成。模型不能"大概知道" calibration 是原因 |
| 文献证据 | 绝对不能 | 论文持续新增，CPT 后的模型不知道 2026 年的论文。而且引用需要精确溯源 |

### 1.2 项目实际故障分析

全部 9 个生产路径故障，**0 个**因为模型缺少行为学知识（详见 Agentic RL 最佳实践文档 §1.1）。

### 1.3 Agentic RL 文档已明确

模型训练目标是**操作能力**（调度、handoff、参数传递、错误恢复），不是领域知识。领域知识在框架层（ethoinsight 库 + skill 文件 + 知识层），模型通过 tool call 访问。

---

## 2. 知识分类学

EthoInsight 需要的外部知识，按查询形态分为四类：

```
知识类型 1: 基础知识 / 产品参数
├── 特点: 有边界、确定性、高频查询、答案唯一
├── 示例: "EPM 开臂时间百分比的典型基线值？"
├── 示例: "EthoVision XT 17 的采样率上限？"
├── 示例: "C57BL/6 和 BALB/c 在旷场中的行为差异？"
└── 规模: ~15-20K token

知识类型 2: 产品操作步骤
├── 特点: 有边界、程序性、版本敏感、需要精确复现
├── 示例: "怎么在 EthoVision 里定义多个 arena？"
├── 示例: "multiple body points 怎么设置？"
├── 示例: "数据导出时怎样选择特定的 time bin？"
└── 规模: 产品手册总量 ~50-100K token（但单次查询只需要 1-2 个小节）

知识类型 3: 设备排错
├── 特点: 半边界、因果推理、症状→原因→修复、版本敏感
├── 示例: "distance_moved 全是 0"
├── 示例: "tracking 总是跟丢"
├── 示例: "导出的 CSV 里某些列为空"
└── 规模: 已知故障模式 ~50-100 条，因果图 ~200-300 节点

知识类型 4: 文献证据
├── 特点: 无边界、持续增长、需要多源综合、需要精确引用
├── 示例: "FST 不动时间增加在其他模型中是否有类似报道？"
├── 示例: "这个效应量的文献背景是什么？"
├── 示例: "有没有文献报告过相反的结果？"
└── 规模: noldus-kb 6200+ 论文 + Semantic Scholar / Crossref 外部 API
```

**核心洞察**：这四类知识在知识边界、查询模式、更新频率、正确性要求上完全不同。用同一种架构处理所有类型是架构失误。

---

## 3. 四层架构

### 3.1 层 1：CAG Prefix — 基础知识常驻（0ms）

**适用**：基础知识、产品核心参数——每次 agent 会话都可能用到的 fact。

**原理**：vLLM prefix cache。把编译好的知识前缀常驻显存 KV cache，每次请求自动共享，零检索延迟。

**内容**（~15-20K token）：

```
[CAG Prefix 内容规划]

§1 行为学基础 Primer
  - 6 个 v0.1 范式：定义、核心指标、典型基线值（按品系区分）
  - 常用行为学术语解释（anxiolytic / anxiogenic / locomotor activity / 
    behavioral despair / thigmotaxis 等）
  - 实验设计基本概念（independent groups / repeated measures / 
    counterbalancing / Latin square）

§2 EthoVision 核心参数
  - 支持的文件格式和版本（TXT / CSV / XLSX / XLS）
  - 关键设置项的默认值和推荐值
  - 已知限制（最大 arena 数、最小物体大小、tracking 精度）

§3 产品架构常识
  - EthoVision 数据导出的列结构（Animal ID / Treatment / Trial / 
    Zone / 时间 bin / 行为指标列）
  - tracking vs detection 的区别
  - trial / arena / zone / subject 的层级关系
  - EthoVision XT 19 模板 × 学术范式映射表
```

**维护方式**：行为学同事维护一个 markdown 文件 → build script 将其编译为 CAG prefix → 部署时随模型一起加载。

**为什么不用 RAG**：这些事实每次会话都可能需要。RAG 每次检索 200ms × 每天几百次会话 = 不必要的延迟和成本。CAG 一次加载，无限复用。

---

### 3.2 层 2：Manual RAG — 产品操作步骤按需检索（~200ms）

**适用**：产品手册的细节——某个菜单的第几层、某个配置的完整步骤。

**为什么不用向量检索**：

```
用户问："怎么设置 multiple body points？"

Vector RAG:
  检索 "multiple body points" → 返回关于 body point detection 算法的论文
  → 返回关于 body elongation 测量的 FAQ
  → 返回关于 multi-animal tracking 的章节
  → 用户要的是设置步骤，返回的是"相关内容"   ← 完全没用

Manual RAG（章节树索引）:
  精确匹配 §5.3.2 "Multiple Body Points"
  → 返回该小节 + 父节（§5.3 Detection Settings）+ 相邻姐妹节（§5.3.1, §5.3.3）
  → 附带版本标注（XT 16 vs XT 17）
  → 返回包含截图引用的完整操作步骤     ← 直接解决问题
```

**架构**：

```
┌──────────────────────────────────────────┐
│ 1. 知识组织：章节树而非 chunk              │
│                                          │
│   EthoVision XT 17 Manual               │
│   ├── 3. Getting Started                │
│   │   ├── 3.1 System Requirements       │
│   │   └── 3.2 Installation              │
│   ├── 5. Arena Settings                 │
│   │   ├── 5.1 Defining Arenas           │
│   │   ├── 5.2 Calibration               │
│   │   └── 5.3 Detection Settings        │
│   │       ├── 5.3.1 Single Body Point   │
│   │       ├── 5.3.2 Multiple Body Points│
│   │       └── 5.3.3 Color Marker        │
│   └── ...                               │
│                                          │
│   检索粒度 = 最小有意义的完整小节           │
│   不切 chunk，不破坏结构                   │
└──────────────┬───────────────────────────┘
               ▼
┌──────────────────────────────────────────┐
│ 2. 检索：精确匹配 + 上下文扩展             │
│                                          │
│   匹配到目标小节                           │
│   → 自动包含父节（知道在哪一章节下）         │
│   → 自动包含相邻姐妹节（知道前后是什么）      │
│   → 自动包含目标小节下的子步骤              │
└──────────────┬───────────────────────────┘
               ▼
┌──────────────────────────────────────────┐
│ 3. 版本感知                               │
│                                          │
│   每个小节标注适用的产品版本：               │
│   §5.3.2: EV XT 14 ✓ | XT 16 ✓ | XT 17 ✓│
│   如果用户版本未知：                        │
│   → 返回所有版本 + 标注差异                │
│   → 反问用户确认版本                       │
│   如果用户版本明确：                        │
│   → 只返回该版本的操作步骤                  │
└──────────────────────────────────────────┘
```

**维护方式**：Noldus 产品文档团队维护手册 → 按章节树结构化导出 → 定期同步到知识库。

---

### 3.3 层 3：GraphRAG KG — 设备排错因果遍历（~100ms）

**适用**：数据异常 → 诊断根因 → 给出修复方案。

**为什么不能用向量检索**：

```
"distance_moved 全是 0" 和 "calibration scale 未设置"
→ 语义上完全不相似
→ cosine similarity ≈ 0
→ 向量检索永远不会把这两者关联起来
→ 但 calibration scale 未设置就是距离全 0 的最常见原因
```

**为什么用知识图谱**：排错是因果推理——从症状节点出发，沿 has_cause 边遍历到原因节点，再沿 has_fix 边遍历到修复方案。这不是检索，是图遍历。

**图谱 Schema**：

```
症状节点（Symptom）         原因节点（Cause）          修复节点（Fix）
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ distance_moved  │     │ calibration     │     │ Arena Settings  │
│ = 0             │────→│ scale 未设置     │────→│ → Calibration   │
│                 │     │                 │     │ → Set Scale     │
│ 伴随: velocity  │     │ 影响: 所有基于   │     │                 │
│ = 0, mobility   │     │ 距离的指标为 0   │     │ EV 14/15/XT     │
│ 分类异常        │     │                 │     │ 所有版本适用    │
└─────────────────┘     └─────────────────┘     └─────────────────┘

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ tracking 频繁   │     │ sample rate     │     │ Acquisition     │
│ 丢失 (ID jumps) │────→│ 低于最低要求     │────→│ → Sample Rate   │
│                 │     │                 │     │ → 提高到 ≥15fps │
│ 伴随: track     │     │ 影响: 快速运动   │     │                 │
│ visualization   │     │ 的动物跟踪不上   │     │ 需根据动物速度   │
│ 出现跳跃线      │     │                 │     │ 调整             │
└─────────────────┘     └─────────────────┘     └─────────────────┘

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ 导出 CSV 某列   │     │ trial/arena     │     │ 检查以下三个    │
│ 全为空          │────→│ 层级下未定义     │────→│ 设置:           │
│                 │     │ 该分析区/行为    │     │ 1. Arena Settings│
│ 伴随: 其他列    │     │                 │     │ 2. Detection    │
│ 正常            │     │ 影响: 仅该 zone  │     │ 3. Trial List中 │
│                 │     │ 的指标缺失       │     │ 的 zone 勾选    │
└─────────────────┘     └─────────────────┘     └─────────────────┘

边属性:
  has_cause: {confidence: 0.95, frequency: "最常见"}
  has_fix: {verification: "重跑 acquisition 检查 track visualization"}
  related_to: {relationship: "区分于 tracking failed（tracks 文件为空）"}
  applies_to_version: ["EV 14", "EV 15", "EV XT 16", "EV XT 17"]
```

**查询过程**：

```
输入: "distance_moved 全是 0"

Step 1: 症状匹配
  实体链接 → Symptom 节点 "distance_moved = 0"（精确匹配）

Step 2: 因果遍历
  (Symptom)-[has_cause]→(Cause) 
  返回: ["calibration scale 未设置", "arena 坐标系未定义", ...]
  每个原因带 confidence score

Step 3: 诊断区分
  提取伴随症状: "velocity 也全是 0 吗？mobility 分类是否异常？"
  → 如果是 → calibration scale 未设置（置信度 0.95）
  → 如果不是 → arena 坐标系未定义（置信度 0.60）

Step 4: 修复遍历
  (Cause)-[has_fix]→(Fix)
  返回操作步骤 + 版本兼容性 + 验证方法

输出:
  "大概率是 calibration scale 没设（置信度 95%）。
   去 Arena Settings → Calibration → Set Scale，
   在画面中画一条已知距离的线。
   EV XT 所有版本适用。
   修复后验证：重跑 acquisition，检查 track visualization 是否有轨迹线。
   
   如果 velocity 正常只有 distance_moved 为 0，也可能是 arena 坐标系没设（置信度 60%）。
   需要确认吗？"
```

**为什么不是大模型直接给答案**：校准设置错误导致距离为 0——这个因果知识不在任何通用模型的训练数据里。它是 Noldus 产品特有的。即使模型通过 CPT 学过，产品版本迭代后 CPT 知识也过时了。KG 是写死的、可审计的、可更新的。

---

### 3.4 层 4：SciRAG — 文献证据多轮检索-验证-合成（~1-3s）

**适用**：跨论文证据合成、效应量对比、文献背景引用。

**2026 年最佳架构**（综合 SciRAG + BRAG + TechGraphRAG）：

```
用户提问: "FST 不动时间增加 40%，有类似效应的文献报道吗？"
    │
    ▼
┌───────────────────────────────────────────┐
│ Phase 1: Query Decomposition              │
│                                           │
│ LLM 分解为多个子查询：                      │
│   子查询1: "forced swim test immobility    │
│            increased chronic stress"       │
│   子查询2: "FST immobility time effect     │
│            size meta-analysis rodent"      │
│   子查询3: "behavioral despair immobility   │
│            pharmacological model FST"       │
└──────────────┬────────────────────────────┘
               ▼
┌───────────────────────────────────────────┐
│ Phase 2: Hybrid Retrieval                 │
│                                           │
│ 三路检索（并行）：                          │
│   A. 本地向量库: noldus-kb 6200+ 论文     │
│      dense retrieval (embedding)          │
│   B. 本地稀疏检索: BM25 关键词匹配         │
│      （补 dense 对术语的漏检）             │
│   C. 外部 API: Semantic Scholar /         │
│      Crossref（论文引用关系）              │
│                                           │
│ 合并去重 → Cross-encoder Reranking        │
└──────────────┬────────────────────────────┘
               ▼
┌───────────────────────────────────────────┐
│ Phase 3: Evidence Sufficiency Scoring     │
│                                           │
│ 对检索结果做五维评分（TechGraphRAG 100分制）：│
│   1. 主题相关度: 论文是否真的在讲同一件事？  │
│   2. 方法可比性: 范式/品系/测量指标一致吗？  │
│   3. 统计完整性: 是否报告了效应量？          │
│   4. 时效性: 近 5 年 vs 经典文献            │
│   5. 引用权威性: 引用次数 / 期刊级别        │
│                                           │
│ 总分 < 阈值（如 60/100）→ 再检索一轮         │
│ 总分 > 阈值 → 进入合成阶段                  │
│ 3 轮后仍不足 → 触发 BRAG abstention        │
└──────────────┬────────────────────────────┘
               ▼
┌───────────────────────────────────────────┐
│ Phase 4: Citation-Aware Synthesis         │
│                                           │
│ 不是 "根据相关文献..."                     │
│ 而是:                                      │
│                                           │
│ "Smith et al. (2019) 在慢性不可预测应激     │
│  模型中报告 FST 不动时间增加 42%           │
│  (d=0.91, 95% CI[0.62,1.20], n=24),       │
│  p<0.01，与您当前批次的效应量一致。          │
│                                           │
│  Jones et al. (2021) 在 LPS 诱导的炎症     │
│  模型中也观察到类似趋势，但效应量较小        │
│  (d=0.45, n=32)，作者认为这可能反映了       │
│  疾病行为（sickness behavior）而非行为绝望。 │
│                                           │
│  ⚠️ 注意: Chen et al. (2020) 报告了相反的  │
│  结果——慢性应激后 FST 不动时间减少。         │
│  该研究使用了不同的应激方案（restraint       │
│  stress vs CMS），且品系为 BALB/c 而非      │
│  C57BL/6，可能解释了方向性差异。"            │
│                                           │
│ 每个事实性陈述绑定到具体的 (作者, 年份)       │
└──────────────┬────────────────────────────┘
               ▼
┌───────────────────────────────────────────┐
│ Phase 5: BRAG Answerability Gate          │
│                                           │
│ 证据充分 → 输出结论 + 引用                  │
│ 证据不足 → "目前文献中未找到直接可比的       │
│            效应量数据。最接近的是 X (2018)    │
│            但范式不同（TST 非 FST）。        │
│            建议在报告中注明这一局限。"        │
│ 证据矛盾 → 报告双方 + 分析可能原因 + 不站队   │
│                                           │
│ 幻觉率: 0.257 → 0.016（BRAG 论文数据）     │
└───────────────────────────────────────────┘
```

---

## 4. Knowledge Router：统一入口

```
用户提问
    │
    ▼
┌─────────────────────────────────────────────┐
│           Knowledge Router                  │
│                                             │
│  分类逻辑（rule-based + 置信度不足时 LLM）：   │
│                                             │
│  ┌─ 概念/事实查询 ──→ Layer 1: CAG prefix   │
│  │  判别信号: 问"是什么"/"多少"/"定义"        │
│  │  延迟: 0ms                               │
│  │                                          │
│  ├─ 操作步骤查询 ──→ Layer 2: Manual RAG    │
│  │  判别信号: 问"怎么做"/"如何设置"/"在哪"    │
│  │  延迟: ~200ms                            │
│  │                                          │
│  ├─ 异常诊断查询 ──→ Layer 3: GraphRAG KG   │
│  │  判别信号: 描述数据异常/报错/不符合预期     │
│  │  延迟: ~100ms                            │
│  │                                          │
│  ├─ 文献/证据查询 ──→ Layer 4: SciRAG        │
│  │  判别信号: 问"文献"/"报道"/"研究"/"证据"   │
│  │  延迟: ~1-3s                             │
│  │                                          │
│  └─ 多类型混合查询 ──→ 按优先级串联           │
│      "数据异常 + 文献参考" → 先 KG 排错      │
│      确认数据正常 → 再 SciRAG 找文献         │
└─────────────────────────────────────────────┘
```

Router 本身是 rule-based 优先（关键词匹配 + 意图分类），置信度不足时 fallback 到 LLM 分类。不需要为此训练模型。

---

## 5. 和 Agentic RL 的关系

**知识层和模型层解耦**：

```
┌─────────────────────────────────┐
│ 模型层（Agentic RL 训练）         │
│                                 │
│ 模型学会的：                      │
│ ✅ 用户问概念 → 直接从 context    │
│    回答（不调 tool）              │
│ ✅ 用户问操作 → 调 manual-search │
│ ✅ 用户描述异常 → 调 troubleshoot│
│ ✅ 用户问文献 → 调 literature-   │
│    search                        │
│ ✅ 知识层都找不到 → 明确告知     │
│ ✅ 多知识源 → 先排错再查文献     │
│                                 │
│ 模型不学的：                      │
│ ❌ 怎么检索（tool 内部逻辑）       │
│ ❌ 怎么合成引用（tool 内部逻辑）    │
│ ❌ 怎么遍历因果图（tool 内部逻辑）  │
└──────────────┬──────────────────┘
               │
               │ tool call
               ▼
┌─────────────────────────────────┐
│ 知识层（确定性工程）               │
│                                 │
│ CAG prefix    → 常驻 KV cache   │
│ Manual RAG    → 章节树索引      │
│ GraphRAG KG   → Neo4j 因果图    │
│ SciRAG        → 多轮检索-验证   │
│ BRAG Gate     → 幻觉拦截        │
└─────────────────────────────────┘
```

---

## 6. 和现有基础设施的关系

| 现有组件 | 在新架构中的位置 |
|---|---|
| **noldus-kb MCP**（6200+ 论文，当前禁用） | SciRAG 的本地向量库（Layer 4, Path A） |
| **ethoinsight 库** | 框架层，不归知识层管。计算决策走代码，不走检索 |
| **skill 文件** | 框架层。范式映射、Gate 规则、workflow 编排 |
| **golden-cases** | Agentic RL 的 Verifier 输入，不直接参与知识检索 |
| **training-data/auto-collected/** | Agentic RL 的 SFT + rollout 数据源 |
| **Semantic Scholar / Crossref API**（新增） | SciRAG 的外部 API fallback（Layer 4, Path C） |
| **Neo4j / Memgraph**（新增） | GraphRAG KG 的存储引擎（Layer 3） |
| **CAG prefix builder**（新增） | Layer 1 的构建工具 |

---

## 7. 执行优先级

| 优先级 | 组件 | 理由 |
|---|---|---|
| **P0（v0.1 前必须）** | CAG Prefix（基础知识） | 行为学基础概念和产品参数，每次会话都要用。构建成本低（一个 markdown → prefix 脚本），收益高（0ms 延迟） |
| **P1（v0.1 前）** | GraphRAG KG（设备排错） | 你记录的故障中，数据异常类有明确因果结构。先建 30-50 个已知故障模式，后续迭代扩展 |
| **P2（v0.1 期间）** | Manual RAG（操作步骤） | 用户问操作步骤的频率低于排错。v0.1 早期可用 CAG prefix 覆盖高频操作，后期补充章节树索引 |
| **P3（v0.1 后期）** | SciRAG（文献证据） | 依赖 noldus-kb MCP 恢复 + 外部 API 集成。文献引用是差异化能力但不是 v0.1 硬指标 |
| **P4（v1.0+）** | BRAG Gate | 幻觉拦截。v0.1 先靠 prompt engineering 兜底，v1.0 接正式 answerability gate |

---

## 8. 依赖和风险

| 依赖/风险 | 状态 | 缓解 |
|---|---|---|
| noldus-kb MCP 仍禁用 | SciRAG 本地向量检索无法使用 | 先接 Semantic Scholar API 做外部检索，内部库恢复后再补充 |
| Neo4j 运维复杂度 | 新增基础设施 | 可先用 Memgraph（更轻量）或直接用 NetworkX in-memory graph 做 PoC |
| 产品手册结构化 | 需产品文档团队配合 | 先用手册目录自动提取章节树，不要求文档团队改格式 |
| CAG prefix 更新 | 知识变更需要重建 prefix | build script 自动化，部署时一起更新 |
| 知识层和模型层解耦 | 需要模型学会正确路由到 tool | Agentic RL 训练时在 Verifier 里加入 tool selection 正确性评分 |

---

## 9. 关键论文参考

| 论文 | 贡献 | 出处 |
|---|---|---|
| **SciRAG** | 自适应检索 + Citation-Aware 符号推理 + Outline-Guided 合成 | EACL 2026 |
| **TechGraphRAG** | 13 步自主 pipeline + 100 分证据充分度 + 外部学术搜索 fallback | arXiv:2606.01613 |
| **BRAG** | Answerability gate：幻觉从 0.257 降到 0.016，Cohen's κ=0.778 | MDPI MLKE 2026 |
| **Hybrid Retrieval + Reranking** | 独立 judge 模型做 claim-level grounding，100% 准确率 | arXiv:2605.01664 |
| **RIRS** | 多 agent 迭代路由：知识摘要 + 中心路由 + 增量求解 | arXiv:2501.07813 |
| **Agentic KG for Enterprise** | 三层混合（Vector + Graph + Recursive Agent），95% vs 25% baseline | Google Dev 2026 |
| **Is Agentic RAG worth it?** | Agentic RAG vs Enhanced RAG 的系统对比，+2.8 NDCG@10 | arXiv:2601.07711 |
| **CAG vs RAG** | Cache-Augmented Generation 作为 RAG 的替代范式 | 2026 futureagi.com |

---

## 10. 决策请求

1. ✅/❌ 同意不做 CPT，行为学知识不进模型权重
2. ✅/❌ 同意四层知识架构（CAG + Manual RAG + GraphRAG KG + SciRAG）
3. ✅/❌ 同意执行优先级（P0 CAG Prefix → P1 GraphRAG KG → P2 Manual RAG → P3 SciRAG）
4. ✅/❌ 同意知识层和模型层解耦的设计原则
5. ✅/❌ 同意 v0.1 前完成 P0 + P1
