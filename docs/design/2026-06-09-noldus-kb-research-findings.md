# noldus-kb 检索方案调研报告

> 日期：2026-06-09 ｜ 调研方式：Opus agent 系统性文献搜索（24 次搜索）｜ 配套设计文档：[2026-06-09-noldus-kb-grep-vs-rag-design.md](./2026-06-09-noldus-kb-grep-vs-rag-design.md)

---

## 总览判断

文献证据中度支持我们"chunk-level RAG 噪声太高、应该转向 paper-level 摘要 + 范式目录组织"的方向，但需要一个关键修正：**纯 grep/全文检索在科学文献检索上有明确的召回上限**（BM25 比最佳 dense 在 LitSearch 上低 24.8 个百分点）。

真正被反复证明有效的不是"扔掉 embedding"，而是 **"廉价召回（BM25/dense 都行）+ LLM 在召回结果上做 contextual rerank/summary"**（PaperQA2 的 RCS、RAPTOR 的摘要节点）。我们的"paper-level 结构化摘要 + 范式目录"恰好就是这个范式的一种静态预计算实现。

**核心建议**：保留 pgvector hybrid 作召回底座，把创新和工程投入放在摘要层和组织层，而不是"全文检索 vs 向量"的二选一。

---

## Q1: 检索粒度 — paper-level vs chunk-level

### 核心结论

文献明确支持"先 paper/summary 级定位、再下钻原文"的分层结构。纯 paper-level 摘要不是终点——最强系统都是"摘要层召回+过滤，原文用于最终引用"。我们的场景（agent 引用论文支撑判读）属于**证据召回+引用**类，最近似的专门 benchmark 是 SciFact（abstract 粒度）。

### 关键证据

**LlamaIndex Document Summary Index / Recursive (Parent-Document) Retriever**
这个模式的工业标准实现：build 时 LLM 给每篇文档生成摘要，query 时先按摘要相关性选中文档，选中后返回该文档的全部节点而非 top-k chunk。官方明确列出适用场景：chunk 缺全局上下文、top-k 难调、embedding 单独不够、多文档消歧。

- 来源：[LlamaIndex — A New Document Summary Index](https://www.llamaindex.ai/blog/a-new-document-summary-index-for-llm-powered-qa-systems-9a32ece2f9ec)、[Recursive Retriever + Document Agents](https://docs.llamaindex.ai/en/v0.10.19/examples/query_engine/recursive_retriever_agents.html)

**RAPTOR (Stanford, ICLR 2024)**
递归聚类+摘要建树，摘要节点与原文节点同时可检索。在 QuALITY 上 +20% 绝对准确率、QASPER 和 NarrativeQA 创 SOTA。核心论点："纯 contiguous chunk 检索丢失跨段全局上下文"。

- 来源：[arXiv:2401.18059](https://ar5iv.labs.arxiv.org/html/2401.18059)

**LitSearch (Princeton, EMNLP 2024) — 反直觉发现**
用全文（avg 6041 词）替代 title+abstract（avg 134 词）**并不能稳定提升检索，常常反而变差**。原因：embedding 模型上下文窗口有限，且在短文档上训练。这直接支持"paper-level 摘要/abstract 粒度优于盲目塞全文 chunk"。

- 来源：[arXiv:2407.18940](https://ar5iv.labs.arxiv.org/html/2407.18940)

**SciFact (UW/AI2, EMNLP 2020) — 场景 B 的 exact match benchmark**
1409 条专家 claim + 5183 篇 abstract，任务 = abstract retrieval → rationale selection → 支持/反驳判定。粒度就是 abstract 级。列出的核心难点直接命中我们用例：directionality（增/减方向推理）、numerical reasoning（统计结果解释）、cause-effect、claim/evidence 特异性不匹配——"OFT center time 降 40% (p<0.01)"正是 directionality + numerical reasoning 的典型。

- 来源：[arXiv:2004.14974](https://ar5iv.labs.arxiv.org/html/2004.14974)、[github.com/allenai/scifact](https://github.com/allenai/scifact)

### 对我们决策的建议

把 6200 篇论文建成 paper-level 结构化摘要索引是有文献背书的正确选择。但数据模型里**必须保留 summary→原文段落的指针**（parent-document 模式），让 agent 拿摘要佐证后能下钻到原句做精确引用——不要只存摘要丢掉原文定位能力。范式目录组织 = RAPTOR 树的人工/领域驱动版本，方向一致。

---

## Q2: 论文结构化摘要的 LLM 提取

### 核心结论

抽取"指标名+变化方向+统计结果"这类三元组，字段级单项准确率可达高位（实体 ~79%、事实一致性 ~92%），但**"完整一条带 P 值的假设"端到端召回会掉到 ~39%**。数值/P 值幻觉是真实且已被量化的 failure mode，但有成熟的 grounding 管线能压下去。

### 关键证据

**Scharfenberger & Funk, "Is it automatable yet?"**（最贴我们需求的数据）
从 SEM 论文抽 construct + 系数 + P 值（GPT-4o + Phi-3）：实体抽取召回 **79.2%**，关系 58.4%，**完整带 P 值的假设召回仅 39.3%**。做了 P 值分桶后处理（<0.001/<0.01/<0.05/<0.1/≥0.1）。

**SAFEPASSAGE (Barrow et al.) — 幻觉降低 85%**
三步管线：LLM 抽取 + 逐字 context 证据 → Smith-Waterman 字符串对齐验证 context 真实存在 → NLI 判定 context 是否真支撑抽取值。关键洞察：即便顶级 LLM 也有 **5–10% 的 exact string match 失败**——它们会"自动纠正"OCR 错误从而篡改原文。

**P-COD (Xie et al., 2025)**
跨文档比对抽出的数值（P 值/效应量）与语义相似论文，算"Surprising Score"，离群检测精度 **98%**，专为 human-in-the-loop 复核设计。

- 来源：[Reducing Hallucinations via Peer Context Outlier Detection](https://ar5iv.labs.arxiv.org/html/2604.01461)

**Sternfeld et al. (2024)**
Llama-3-8B few-shot 抽科学三元组，事实一致性 **~92%**（Levenshtein ≤2），接近人工标注。few-shot 优于 fine-tune（后者有时反而退化）。

- 来源：[EasyChair preprint](https://easychair.org/publications/preprint/Ldvs/download)

**已知 failure mode 清单**（多源汇总）：

| # | Failure Mode | 严重度 |
|---|-------------|--------|
| 1 | 数值幻觉 / P 值编造 | 高 |
| 2 | LLM 静默篡改原文（5–10% 不能逐字对回原文） | 高 |
| 3 | 混淆方法段与结果段 | 中 |
| 4 | claim 与 evidence 的特异性不匹配（SciFact 也列为核心难点） | 中 |

### 对我们决策的建议

1. 摘要 schema 里每个结构化字段（指标、方向、p、n、effect size）**必须带 verbatim evidence span + 源页码**，并在抽取后跑一道字符串对齐校验（SAFEPASSAGE 模式）。这与我们反复强调的 "parameters_used 必须反映实际 resolution path""禁止叙述黑洞" 是同一治理哲学。

2. **不要把"完整假设三元组"当作高置信结构化数据直接喂给判读**——端到端 39% 的召回意味着会大量遗漏/串台。更稳的做法：抽"指标+方向"作为可检索标签（高召回），P 值/效应量作为"需点击下钻到原句确认"的弱断言。这正是我们 "预填推测≠断言" 原则。

3. 可参考的现成 pipeline：SAFEPASSAGE（对齐+NLI 双校验）、P-COD（跨文献离群兜底）。

---

## Q3: 查询扩展 vs embedding 语义搜索

> 对应场景 A：agent 发现数据异常，但不知道搜什么关键词（"是不是 zone 标定有问题？"）

### 核心结论

"LLM 先生成查询假设→候选词搜索"就是学界的 **query expansion / HyDE / Query2doc** 范式，对稀疏检索（BM25/grep）增益巨大（BM25 nDCG@10 从 50.6→68.8），零样本、跨域泛化强——恰好契合我们 grep 底座 + agent 不知关键词的场景。直接拿 embedding 匹配延迟低但对短/模糊 query 敏感。**两者互补，不是二选一**。

### 关键证据

**Query2doc (Wang et al., 2023)**
LLM few-shot 生成伪文档拼到 query，BM25 上 TREC DL **+3–15%**，dense 上 +1–4%。

- 来源：[arXiv:2303.07678](https://arxiv.org/pdf/2303.07678)

**ThinkQE (Lei et al., 2025) — 关键数据**
加"思考过程"+检索反馈迭代，BM25 nDCG@10 **68.8 vs 裸 BM25 50.6**，超过部分训练过的 dense retriever。

- 来源：[ACL Anthology](https://aclanthology.org/2025.findings-emnlp.965.pdf)

**CSQE (Corpus-Steered Query Expansion) — 对我们最有用的变体**
用 LLM 知识 + 语料检索回来的句子共同 grounding 扩展词，缓解幻觉——对我们尤其有用：用 Noldus 手册/论文语料里的真实术语 grounding，而非让 LLM 凭空造词。

**SoftQE (Pimpalkhute et al., 2024)**
把 LLM 扩展知识离线蒸馏进 dense retriever，推理时不再调 LLM，BEIR OOD +2.83。长期优化方向。

- 来源：[arXiv:2402.12663](https://ar5iv.labs.arxiv.org/html/2402.12663)

**权衡总结**：

| | Query Expansion（Tier 3） | Embedding 语义搜索 |
|---|---|---|
| 对 BM25/grep 增益 | 大（+18pt nDCG） | 不适用 |
| 跨域泛化 | 强（零样本） | 需微调 |
| 延迟 | 高（每次调 LLM） | 低 |
| 幻觉风险 | 扩展词可能幻觉 | 无（但检索结果可能不相关） |
| 可解释性 | 高（能看到搜了什么词） | 低（向量距离不可解释） |

### 对我们决策的建议

场景 A 天然是 query expansion 的甜区——让 agent 把异常上下文先展开成"假设关键词集合"（zone calibration / arena boundary / detection threshold / tracking loss…）再走全文检索。务必用 **corpus-steered（CSQE）变体**：候选词从手册/论文真实术语 SSOT 出，而不是 LLM 自由发挥——防 deepseek 脑补假术语，与我们 "skill 不许内嵌结构化知识、查询词走 SSOT" 原则对齐。

---

## Q4: 手册/技术文档的知识检索

> 对应：EthoVision XT 等 Noldus 产品手册（100–500 页 PDF）的检索方案

### 核心结论

工业界最佳实践明确倾向**把原文重构成"自包含、问题导向的段落"再检索，而非纯原文 chunk**。诊断型（"数据异常 → 是不是 zone 标定问题？"）和 how-to 型（"这个功能怎么用？"）确实应该用**不同结构**——前者用 Symptom→Cause→Resolution，后者用顺序步骤+前置条件+预期结果。

### 关键证据

**AWS Prescriptive Guidance — Writing best practices for RAG**
核心主张：chunk 要语义自包含、消除"如上所述/见第 3 步"这类跨 chunk 依赖、标题用用户真实提问措辞、按内容类型打 metadata（task_type、error_code、symptom、product_version）。

- 来源：[AWS Prescriptive Guidance PDF](https://docs.aws.amazon.com/pdfs/prescriptive-guidance/latest/writing-best-practices-rag/writing-best-practices-rag.pdf)

**Farkas (UW HCDE) — Troubleshooting Procedures**
明确区分 diagnosis 阶段（匹配症状）与 resolution 阶段（解决路径），主张"两个差异很大的症状要写成独立的 troubleshooting 条目"。这对应业界 KCS（Knowledge-Centered Service）的 Symptom-Cause-Resolution 结构。

- 来源：[Farkas, Troubleshooting Procedures PDF](https://faculty.washington.edu/farkas/HCDE%20407-2013/FarkasTroubleshootingProcedures.pdf)

**厂商做法汇总**：主流是 BM25/sparse + dense 双路 + RRF + 可选 cross-encoder rerank；内容侧做语义切块而非机械定长切块。

### 对我们决策的建议

1. EthoVision 手册不要原样 chunk。**预先整理成问题导向 markdown**，两种 intent 分开建：

   **诊断型**（场景 A）：
   ```
   症状 → 可能原因 → 排查/解决步骤
   metadata: 症状关键词、功能模块、产品版本
   ```

   **How-to 型**：
   ```
   任务名 → 前置条件 → 顺序步骤 → 预期结果
   metadata: 任务类型、功能模块、产品版本
   ```

2. Agent 侧先做 intent 路由（diagnostic vs how-to），再查对应库——和我们已有的 intent guardrail 架构同构，可复用。

3. Curated 问题导向 markdown 本身就是把手册知识做成单一权威来源（SSOT），而不让 chunk 噪声+LLM 脑补两头漏。

---

## Q5: 全文索引 vs 向量检索 — 能否统一

> 核心问题：是否存在一种方案同时保留"精确匹配"和"语义关联"两种能力，而不维护两套独立系统？

### 核心结论

**能统一，而且我们现有的 pgvector hybrid 大概率已经够用，问题更可能在"没用对"而非"选错"**。SPLADE/ELSER（稀疏神经向量）是折中方案，但它做了 term expansion，**不等于精确字面匹配**，所以不能替代 grep 式精确匹配。

### 关键证据

**BM25 vs dense 在科学文献上的真实差距**

| Benchmark | BM25 | Best Dense | 差距 |
|-----------|------|------------|------|
| LitSearch recall@5 | 50.0% | 74.8% (GritLM-7B) | **+24.8pt** |
| LitSearch recall@20 | 39.9% | ~42% | 差距缩小 |
| SciFact nDCG@10 | 0.665 | ~0.744 | +11.9% |

BEIR 原始论文确认：BM25 是异常稳健的零样本基线，跨域不退化。在 broad/recall@20 上 BM25 接近甚至不输部分 dense。

- 来源：[LitSearch](https://ar5iv.labs.arxiv.org/html/2407.18940)、[BEIR NeurIPS 2021](https://ar5iv.labs.arxiv.org/html/2104.08663)

**SPLADE/ELSER 的关键限制**
BERT-MLM 做 term importance + term expansion（"pizza" 能匹配 "Margherita"），解决词表不匹配。但**它不保证精确字面匹配**——搜 "zone calibration" 可能匹配到 "arena setup" 而不是原文中的 "zone calibration"。所以即使上 SPLADE，仍需要精确 BM25/grep 那一路。

- 来源：[Pinecone SPLADE](https://archive.pinecone.io/learn/splade/)、[Qdrant modern sparse neural retrieval](https://qdrant.tech/articles/modern-sparse-neural-retrieval/)

**pgvector hybrid 生产可用性**
pgvector/pgvectorscale(HNSW/DiskANN) + BM25（ParadeDB pg_search 成熟） + RRF(k=60) 融合，是被反复验证的生产标准。OpenClaw 生产实测 hybrid ~84% precision vs vector-only ~62%。

- 来源：[ParadeDB — Hybrid Search in PostgreSQL: The Missing Manual](https://www.paradedb.com/blog/hybrid-search-in-postgresql-the-missing-manual)

**我们现有方案大概率"没用对"而非"选错"**。常见错配点：

| 错配 | 表现 | 正确做法 |
|------|------|---------|
| 在 chunk 上做 embedding 而非 paper-level 摘要 | Q1 已证全文 chunk 反而伤检索 | 索引粒度提到 paper/summary 级 |
| 缺 LLM rerank/RCS 层 | 噪声-信号比高（Q6：这才是 PaperQA2 拉开差距的关键） | 召回后 LLM 逐条打分+情境摘要 |
| RRF 权重/候选池没调 | 检索质量平庸 | 每路 50–100 候选，RRF k=60 |

### 对我们决策的建议

1. **不必维护两套系统**。一套 Postgres 内：精确字面/结构化查询用 BM25（ParadeDB pg_search）或 tsvector；语义关联论文检索用 pgvector；RRF 融合。SPLADE/ELSER 可作为"中间召回"补充但不能替代精确匹配那一路。

2. **先把"摘要层 + RCS rerank"补上**，再判断 hybrid 够不够，比直接推倒重来更有据。

3. 证据支持的解法是**把索引粒度提到 paper/summary 级 + 加 LLM rerank**，而不是放弃向量——这与我们"chunk-level RAG 噪声太高"的判断一致，但解法和我们最初想的"纯 grep"不同。

---

## Q6: 科学文献 + Agent 的类似项目

### 核心结论

有高度相似的开源项目。**首选参考 PaperQA2 / FutureHouse**（开源、有论文、有 ablation）。它的关键设计 RCS（LLM 对召回 chunk 逐条打分+情境化摘要再 rerank）直接回答了"chunk RAG 噪声太高"的痛点。Ablation 最关键结论：**chunk 大小/embedding 模型选择影响很小，rerank 层才是胜负手**。

场景 B（claim→找论文佐证）的最接近 benchmark 是 SciFact。最接近的产品形态是 FutureHouse 的 Owl（prior-work 检测："有没有人做过 X"——正是我们场景 B 的形态）。

### 关键证据

**PaperQA2 (FutureHouse, 开源, arXiv:2409.13740) — 超人类水平的 agentic RAG**

关键设计 — **RCS（Reranking + Contextual Summarization）**：
- Dense 召回 top-k(≥15) 后，**每个 chunk 独立送 LLM**
- LLM 输出 (相关性分 1–10 + ≤300 词情境摘要)
- 按分 rerank + 过滤低分
- 原文 ~9000 字符压缩 ~5.6×
- 平均处理 **14.5 篇论文/问**

这一步过滤关键词命中但语境无关的噪声、纠正 embedding 排序错误、让上下文窗口能塞下更多论文。

**Ablation 关键结论**：
- chunk 大小最优 7000–11000 字符
- **parser 选择几乎无影响**
- **embedding 模型在 depth≥20 时差异很小**
- → chunk/embedding 调参不是重点，**RCS rerank 才是**

**警告**：RCS 需要足够强的 LLM，GPT-3.5/Llama-3-70B 反而降低准确率。

还有 Citation Traversal（沿引用图滚雪球，~46% 问题用到）。

- 来源：[FutureHouse Engineering Blog](https://www.futurehouse.org/research-announcements/engineering-blog-journey-to-superhuman-performance-on-scientific-tasks)、[arXiv:2409.13740](http://arxiv.org/pdf/2409.13740)、[github.com/Future-House/paper-qa](https://github.com/Future-House/paper-qa)

**FutureHouse 平台（2025-05）**
- Crow：基于 PaperQA2 的通用科学 QA
- Falcon：跨千篇综述
- **Owl：prior-work 检测（"有没有人做过 X"）——正是我们场景 B 的形态**
- Phoenix：化学实验设计
- LitQA 上达 ~90% vs PhD ~67%

- 来源：[FutureHouse platform guide](https://intuitionlabs.ai/pdfs/futurehouse-ai-agents-a-guide-to-its-research-platform.pdf)

**Asta / AstaBench (AI2, 2025)**
9 类专门 agent，含 Paper Finder（LLM 相关性判定）、Scholar QA（带内联引用的长答）、Table Synthesis、DataVoyager（数据驱动统计发现）。orchestrator 路由准确率 100%。

- 来源：[Asta OpenReview](https://openreview.net/pdf?id=M7TNf5J26u)

**生物医药 copilot 检索层 (Boston U, medRxiv 2025)**
Tool-using agent + retrieval+rerank + cache-and-prune memory + 显式 evidence grounding 层，USMLE Step1/2 超 GPT-4。

- 来源：[medRxiv 2025.08.06.25333160](https://www.medrxiv.org/content/10.1101/2025.08.06.25333160v1)

### 对我们决策的建议

1. **直接对标 PaperQA2，复用其 RCS 设计**。我们的"paper-level 结构化摘要"可以理解为 RCS 的离线预计算版本——PaperQA2 是 query-time 现算情境摘要，我们是 build-time 预存通用摘要。两者可结合：用预存摘要做廉价召回（解决 6200 篇规模），命中后对少量候选再跑一次 query-aware 的 RCS rerank（解决"针对当前判读结论"的相关性）。这避免了"摘要太通用、匹配不上具体结论"的风险。

2. **范式目录 = Asta 的 paper-finder + 领域路由的静态版**，方向被验证可行（Asta orchestrator 路由准确率 100%）。

3. PaperQA2 的 ablation 是我们最有力的论据：**"chunk/embedding 调参收益很小，rerank/摘要层才是胜负手"**——正面支持"把工程投入从 chunk-embedding 调参转向 paper-level 摘要+组织"的决策方向。

---

## 跨题综合建议

### 对设计文档的关键修正

我们最初的设计文档（[2026-06-09-noldus-kb-grep-vs-rag-design.md](./2026-06-09-noldus-kb-grep-vs-rag-design.md)）中"用 grep 替代 RAG"的判断需要修正。证据收敛到以下结论：

1. **方向基本正确，但别走极端"grep 替代 RAG"**。胜负手是 **paper-level 摘要 + LLM rerank（RCS）**，召回底座用 BM25 还是 pgvector hybrid 反而次要（PaperQA2 ablation + LitSearch + RAPTOR）。

2. **保留 pgvector hybrid 作召回底座**，把它从 chunk 级提到 paper-level 摘要级，并补 LLM rerank 层——很可能现方案"没用对"而非"选错"（Q5）。

3. **场景 A（故障联想）用 corpus-steered query expansion**：候选词从手册/论文真实术语 SSOT 出，LLM 展开但不自由发挥——防脑补假术语（Q3）。

4. **场景 B（证据锚定）对标 SciFact 任务形态**：摘要里抽"指标+方向"做高召回标签，P 值/effect size 作弱断言需下钻原句确认，每字段带 verbatim span+页码并跑对齐校验（Q2）——与既有治理原则（SSOT、parameters_used、禁叙述黑洞、预填≠断言）完全同构。

5. **手册按 diagnostic / how-to 双结构 curated**，agent 先 intent 路由再查（Q4），复用现有 intent guardrail。

### 修正后的架构

```
                       ┌─────────────────────────────────┐
                       │  Tier 1: 结构化精确查找           │
                       │  taxonomy JSON + curated markdown │
                       │  工具: grep / jq                  │
                       │  场景: "EPM 是什么"               │
                       └──────────────┬──────────────────┘
                                      │ 没命中
                       ┌──────────────▼──────────────────┐
                       │  Tier 2: Paper-level 摘要召回     │
                       │  pgvector hybrid (摘要级,非chunk) │
                       │  + 范式目录预过滤                 │
                       │  场景: "OFT center time 降低      │
                       │         的类似论文"               │
                       └──────────────┬──────────────────┘
                                      │ 召回 top-k
                       ┌──────────────▼──────────────────┐
                       │  Tier 2.5: RCS Rerank (新增)     │
                       │  LLM 逐条打分(1-10)+情境摘要      │
                       │  过滤低分 → 压缩 5.6×            │
                       │  ← PaperQA2 的核心，胜负手        │
                       └──────────────┬──────────────────┘
                                      │
                       ┌──────────────▼──────────────────┐
                       │  Tier 3: Corpus-Steered QE       │
                       │  异常 → LLM 生成候选词            │
                       │  (从 SSOT 术语表 grounding)       │
                       │  → 回到 Tier 1/2 搜索             │
                       └─────────────────────────────────┘
```

### 最有用的 4 个直接参考

| 优先级 | 参考 | 为什么读 |
|--------|------|---------|
| 1 | [PaperQA2 论文+代码](https://github.com/Future-House/paper-qa) | 开源、有 ablation、RCS 设计可直接复用 |
| 2 | [LitSearch (arXiv:2407.18940)](https://ar5iv.labs.arxiv.org/html/2407.18940) | 科学文献检索 benchmark，BM25 vs dense 的真实差距 |
| 3 | [SciFact](https://github.com/allenai/scifact) | 场景 B 最接近的 benchmark |
| 4 | [ParadeDB Hybrid Search](https://www.paradedb.com/blog/hybrid-search-in-postgresql-the-missing-manual) | pgvector hybrid 怎么用对 |

---

## 调研方法说明

- 搜索次数：24 次
- 覆盖范围：学术论文（arXiv/ACL/EMNLP/ICLR/NeurIPS/medRxiv）、工业博客（LlamaIndex/Pinecone/Qdrant/ParadeDB）、开源项目（PaperQA2/SciFact）、官方文档（AWS）
- 时效：优先 2024–2025 年文献，核心 benchmark 追溯到原始发表年
- 每个结论均附可验证的来源 URL
