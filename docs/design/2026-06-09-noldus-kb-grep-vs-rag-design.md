# noldus-kb 知识检索方案重新评估：Grep-driven vs RAG

> 日期：2026-06-09 ｜ 状态：调研中

---

## 1. 背景：noldus-kb 现状

noldus-kb 是 EthoInsight 项目的知识库服务，为 agent 提供行为学领域知识（范式、产品、术语、论文、手册）。当前架构是标准 RAG 管线：

```
PDF papers/manuals → marker-pdf 解析 → BGE-M3 embedding (1024d)
    → PostgreSQL + pgvector 混合检索 → CLI / MCP / REST API 三层接口
```

**实际状态**（截至 2026-04-01 handoff，之后无更新记录）：

- 6200 篇论文中仅灌入 105 篇（~1.7%）
- 全量灌入预计 42 小时（T4 15GB GPU）
- GPU 与 phenotyper 训练共享，不能同时跑
- 搜索质量从未验证过
- 服务当前在 ethoinsight 中处于禁用状态（`extensions_config.json: "enabled": false`）
- 服务器 `180.184.84.124:7001` 大概率已不可达

**架构本身的问题**（来自项目自身的 architecture-improvement-notes.md）：

- P3 已指出：结构化查询（products/paradigms/terms）不需要向量搜索
- P4 已指出：多版本产品手册会互相矛盾
- P5 已指出：手册中 30-50% 的信息在图里，文本管线捕获不到

---

## 2. 触发讨论：用户提出 grep 替代 RAG

用户认为 RAG 已经过时，提出用 grep（全文检索）替代现有的 embedding + pgvector 方案。核心理由：

1. SSOT 原则：文件-based 知识天然符合单一事实来源
2. 精确性：agent 需要权威答案，不是"可能相关"的 chunk
3. 简洁性：不需要 GPU、不需要 embedding、不需要 42 小时灌入
4. 可维护性：改知识 = 改文件，git diff 一目了然

---

## 3. 深入讨论：grep 的边界

### 3.1 grep 擅长的场景

**Tier 1：结构化精确查找**

- "EPM 是什么范式" → `jq` 查 `paradigms.json`
- "thigmotaxis 的定义" → `jq` 查 `terms.json`
- "EthoVision XT 有哪些功能" → `jq` 查 `products.json`

目前 taxonomy 数据（products/paradigms/terms 三个 JSON）总计 ~30KB，是 grep/jq 的绝对甜点区。

### 3.2 grep 不擅长的场景（关键）

用户提出了两个真实 agent 场景，grep 解决不了：

**场景 A — 故障联想（关键词未知）**

agent 发现实验数据异常 → 需要联想到"是不是 EV19 的 zone 标定有问题？" → 需要查 manual 里 zone detection 的校准章节。

但 agent 不知道 "zone calibration" 这个关键词——它只知道"数据看着不对"。grep 的前提是知道搜什么。

**场景 B — 证据锚定（跨文献关联）**

agent 完成数据洞察："OFT center time 降低 40% (p<0.01)" → 需要找已发表论文里有没有类似发现来佐证判读 → 需要搜 "reduced center time in OFT under chronic stress" 相关的论文。

但 agent 不知道哪篇论文、哪个作者、哪年的。grep 全文检索可以搜关键词，但 6200 篇论文的原始 PDF/文本里搜 "center time reduced" 会返回大量噪声（methods 段、不相关的研究设计、不同动物模型等）。

### 3.3 传统 RAG 同样不擅长这些场景

Chunk-level RAG 的问题不是"搜不到"，而是**噪声-信号比太高**：

- 搜 "OFT center time reduced" → embedding 返回 5 个 chunk：一篇 2013 年论文的 methods 段（讲 arena 尺寸）、一篇 CatWalk 论文（碰巧有 "center" 但不是一回事）、一篇 review 的 introduction（泛泛而谈）
- Agent 拿到这 5 个 chunks，不知道哪些是权威的、哪些和当前实验参数匹配
- Agent 只能硬着头皮引用，产出的"证据"反而不可靠

**这与 EthoInsight 项目反复强调的 SSOT + 精确性原则根本冲突。** Chunk 级 RAG 把知识切成碎片，靠向量相似度碰运气，丢失了来源、上下文和权威性。

---

## 4. 提出的方案：三层检索 + 论文元数据摘要

### 4.1 总体架构

```
                    ┌─────────────────────────────────┐
                    │  Tier 1: 结构化精确查找           │
                    │  taxonomy JSON + curated markdown │
                    │  工具: grep / jq                  │
                    │  场景: "EPM 是什么"               │
                    │       "thigmotaxis 定义"          │
                    │       "EthoVision XT 功能"        │
                    └──────────────┬──────────────────┘
                                   │ 没命中
                    ┌──────────────▼──────────────────┐
                    │  Tier 2: 元数据过滤 + 全文搜索    │
                    │  论文摘要 + manual 章节           │
                    │  按范式/产品预过滤 + rg 全文检索   │
                    │  场景: "OFT center time 降低      │
                    │         的类似论文"               │
                    │        "EV19 zone 标定校准"       │
                    └──────────────┬──────────────────┘
                                   │ 还是没找到
                    ┌──────────────▼──────────────────┐
                    │  Tier 3: LLM 查询扩展 + 重试      │
                    │  agent 不知道关键词               │
                    │  → LLM 生成候选搜索词             │
                    │  → 回到 Tier 1/2 搜索             │
                    │  → 联想是 LLM 的活，不是检索的活   │
                    └─────────────────────────────────┘
```

核心区别：**检索粒度从 chunk（512 token 文本片段）提升到论文级（标题+作者+年份+结构化摘要）**。

| | Chunk RAG（现有） | 新方案 |
|---|---|---|
| 检索单元 | 512 token 文本片段 | 整篇论文的元数据 + 摘要 |
| 匹配方式 | embedding cosine | 范式标签过滤 → 全文检索摘要 |
| Agent 拿到 | 5 个来源不明的碎片 | 5 篇可追溯的论文（标题/作者/年份/摘要） |
| 可验证性 | 无法判断 chunk 权威性 | 可以看到论文信息，判断是否相关 |

### 4.2 Publications 线（6200 篇论文）

不把每篇论文切成 20 个 chunks 做 embedding。而是：

1. **保留现有元数据提取**（ingest 管线已有）：标题、作者、年份、paradigm、product、terms
2. **新增 LLM 结构化摘要**：每篇论文生成包含以下字段的摘要：
   - 实验范式、动物模型（品系/年龄/性别）
   - 实验处理（药物/剂量、应激模型、基因型）
   - 主要发现（指标名称 + 变化方向 + 统计学结果）
   - 实验条件（设备、光照、时间）
3. **按范式组织目录**：
   ```
   data/papers/OFT/
   ├── 2020_smith_chronic_stress_reduces_center_time.md
   ├── 2019_chen_anxiolytic_effects_of_compound_x.md
   └── ...
   ```
4. **检索流程**：agent 知道当前范式 → 进对应目录 → `rg "center time" --glob "*.md"` → 精确命中

6200 篇 ÷ 15 范式 ≈ 每个范式 ~400 篇。400 篇 markdown 的 `rg` 全文检索是秒级的。因为已按范式预过滤，搜索精度远高于全量向量搜索。

### 4.3 Manuals 线（Noldus 产品手册）

手册的知识结构与论文不同——它是操作性的、问题导向的：

- "EthoVision XT 的 zone 是怎么定义的？"
- "如果 body point detection 不准，可能是哪些设置的问题？"
- "数据导出有哪些格式？RawData 和 Statistics 的区别是什么？"

建议组织形式：

1. marker-pdf 把 manual PDF 转成 markdown（现有能力）
2. LLM 按主题半自动整理成问题导向的 curated markdown：
   ```
   data/manuals/EthoVisionXT/
   ├── zone-detection.md          # zone 定义、校准、常见问题
   ├── body-point-tracking.md     # 身体点检测、精度、故障排查
   ├── data-export.md             # 导出格式、字段含义
   ├── experiment-setup.md        # 实验设置流程
   └── troubleshooting.md         # 常见故障及解决方案
   ```
3. Agent grep 搜索时，标题层级提供上下文——搜 "zone 不准" 命中 `zone-detection.md`，打开就是完整章节

### 4.4 保留的现有资产

- CLI + MCP + REST API 三层接口架构：**不动**（接口设计是好的）
- taxonomy JSON 数据：**直接用**（已经是完美格式）
- 论文元数据提取管线：**保留**（LLM 提取标题/作者/范式/产品是核心价值）
- 要替换的：retriever 后端从 pgvector 换 filesystem + ripgrep
- 要新增的：LLM 论文摘要生成 + knowledge markdown 目录组织

---

## 5. 待调研的关键问题

以下问题需要由 Opus agent 做系统性调研：

### Q1: 检索粒度决策

文献中对于"领域知识库应该用 paper-level 摘要还是 chunk-level embedding"有没有系统性的比较研究？什么场景下 paper-level 优于 chunk-level？我们的场景（agent 需要引用论文支撑判读）属于哪一类？

### Q2: 论文结构化摘要的提取方案

LLM 从论文全文生成结构化摘要的效果如何？有没有现成的 prompt 模板或 pipeline？准确率（指标名称、变化方向、统计学结果）能到什么水平？有哪些已知的 failure mode？

### Q3: 跨范式检索（Tier 3 查询扩展）

当 agent 不知道搜什么关键词时，"LLM 生成查询假设 → 候选词搜索" 这个模式在实践中效果如何？和 embedding-based 语义搜索相比，各有什么优劣？

### Q4: 手册知识的组织方式

产品手册（操作手册）的知识检索在工业界有什么最佳实践？问题导向的 curated markdown vs 原文 chunk 检索，各自的适用场景是什么？

### Q5: 混合方案的可能性

是否存在一种方案同时保留"精确匹配的结构化查询"和"语义关联的论文检索"两种能力，而不用维护两套独立的检索系统？比如：全文索引（如 Elasticsearch/Meilisearch）vs 向量数据库？BM25 vs embedding？稀疏向量（SPLADE）？

### Q6: 类似项目的参考

有没有类似 EthoInsight 的领域-specific agent 项目（需要检索科学文献来支撑分析结论）？它们的知识库是怎么设计的？

---

## 6. 验证计划（调研结束后）

不急着全面改造。先用一个范式做端到端验证：

1. 从 6200 篇论文里筛出 OFT 相关的（用已有元数据或关键词）
2. 找 5-10 篇代表性论文，LLM 生成结构化摘要 markdown
3. 写简单的 `search_papers(paradigm, query)` → `rg` over 对应目录
4. 在真实 agent 场景跑一次：OFT 数据 → 计算指标 → agent 发现异常 → 调 `search_papers("OFT", "center time reduced stress")` → 拿到 3 篇相关论文摘要 → 引用支撑判读
5. 如果跑通，证明 grep + 范式目录 + curated 摘要的模式成立，再考虑规模化

---

## 7. 决策记录

| 决策 | 结论 | 理由 |
|------|------|------|
| Chunk-level embedding RAG | ❌ 不适合 | 噪声-信号比高，与 SSOT 哲学冲突，工程复杂度大（GPU/42h 灌入） |
| 纯 grep 全文检索 | ⚠️ 不够 | Tier 1（结构化查询）完美；Tier 2（论文检索）需要预组织；Tier 3（关键词未知）需要 LLM 辅助 |
| 三层架构 | ✅ 方向 | 精确匹配 → 范式预过滤+全文检索 → LLM 查询扩展，每层各司其职 |
| 检索粒度：paper-level 摘要 | ✅ 优于 chunk-level | Agent 需要可追溯、可验证的论文引用，不是来源不明的文本片段 |
| 保留现有 MCP/CLI/API 接口 | ✅ | 接口设计合理，只换后端实现 |
| 先验证再规模化 | ✅ | 用一个范式（OFT）端到端验证整体模式 |

---

## 8. 相关文档

- noldus-kb 仓库：`/home/wangqiuyang/noldus-kb`
- noldus-kb 架构改进笔记：`noldus-kb/docs/architecture-improvement-notes.md`
- 上次灌入 handoff：`noldus-kb/docs/handoffs/260401-noldus-kb-handoff.md`
- EthoInsight CLAUDE.md：知识注入三层 + noldus-kb 当前禁用状态
