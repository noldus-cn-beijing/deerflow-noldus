# noldus-kb 改造方案：从 RAG 到 Agent-Native 文件系统知识库

> 日期：2026-06-09 ｜ 基于 2025–2026 年业界趋势调研 + noldus-kb 实际数据

---

## 1. 一句话结论

把 noldus-kb 从"embedding → pgvector → chunk 拼接"的 RAG 管线，改成"文件系统 + grep → agent 按需调用"的工具型知识库。知识内容用 markdown（+ 现有 taxonomy JSON）组织，检索用 ripgrep，agent 通过 MCP 工具主动查询而非被动接收拼接文本。

---

## 2. 2025–2026 年趋势：RAG 退潮，文件系统回归

### 2.1 为什么 RAG 在 agent 场景下是错的架构位置

RAG 的核心假设：检索发生在 LLM 理解问题之前。LLM 没有选择权——embedding 召回什么 chunk，就喂什么 chunk。这个假设在 agent 场景下崩溃了：

> Agent 范式下，检索不是预处理步骤，是 agent 认知循环里的一个**工具调用**。agent 自己判断什么时候调、调什么、结果够不够、要不要再调。

RAG 作为"透明预处理管线"嵌在 LLM 调用前面，和 agent 的工具调用循环是两套机制，根本不在一个架构层。

### 2.2 工业界共识

| 项目 | 核心设计 | 关键数据 |
|------|---------|---------|
| **Vercel Knowledge Agent** | grep + find + cat，零 embedding、零向量 DB | 单次调用成本从 $1.00 → $0.25，质量反而提升 |
| **Karpathy LLM Wiki** | `wiki/{index,concepts,sources,syntheses}/` + `[[wikilinks]]` + `grep -rli` | 业界最广泛采用的知识库文件结构 |
| **lat.md (v0.11, 2026-03)** | markdown 目录 + `[[wiki links]]` + `lat search/check` CLI | "AGENTS.md 不扩展"问题的解法 |
| **MinerU Document Explorer (v1, 2026-04)** | BM25 + LLM rerank + 15 MCP 工具，文件系统底层 | 零配置 BM25-only 模式内置 |
| **Alibaba GrepRAG** | ripgrep 比 graph RAG 快一个数量级 + 0.6B 模型生成 grep 命令超大型语义搜索 | 代码仓库检索 benchmark |

**共同趋势**：

1. **存储层**：markdown 文件系统（不是数据库）
2. **检索层**：`grep`/`find`/`cat` 优先，BM25/semantic 是可选增强
3. **结构层**：`index.md` + `concepts/` + `sources/` + `syntheses/` 分层目录 + `[[wikilinks]]` 交叉引用
4. **Agent 集成**：MCP 工具暴露文件操作（`search`, `read`, `list`），不是透明预处理
5. **可调试性**：确定性——能精确定位哪一行哪个文件产出了答案

### 2.3 一句话总结趋势

> **"Less scaffolding, more model."** 200K+ token 上下文窗口 + LLM 多步推理能力提升，让 agent 主动搜索（grep/LSP/code-graph）比预建向量索引更有效。

---

## 3. noldus-kb 现状 vs 目标

### 现状（RAG）

```
PDF papers/manuals
  → marker-pdf 解析 → 512-token chunks
  → BGE-M3 embedding (1024d)
  → PostgreSQL + pgvector 混合检索
  → CLI / MCP / REST API
```

问题：
- 索引在 chunk 级（噪声比信号高）
- 检索是透明预处理（agent 没有选择权）
- 需要 GPU（T4 15GB，和 phenotyper 训练抢）
- 6200 篇只灌了 105 篇（灌入 42 小时）
- 维护成本高（embedding 漂移、索引重建）
- **现已被禁用**

### 目标（Agent-Native grep）

```
data/
├── taxonomy/           # 已有，不变（JSON，精确查询用 jq）
│   ├── products.json
│   ├── paradigms.json
│   └── terms.json
├── paradigms/          # 论文知识：按范式分目录
│   ├── OFT/
│   │   ├── index.md              # OFT 知识索引
│   │   ├── 2020_smith_chronic_stress.md
│   │   ├── 2019_chen_anxiolytic.md
│   │   └── ...
│   ├── EPM/
│   ├── FST/
│   └── ...
├── manuals/            # 手册知识：问题导向 markdown
│   ├── EthoVisionXT/
│   │   ├── index.md              # 手册知识索引
│   │   ├── zone-detection.md     # 症状→原因→解决
│   │   ├── body-point-tracking.md
│   │   ├── data-export.md
│   │   └── troubleshooting.md
│   └── CatWalkXT/
└── concepts/           # 跨领域概念（wikilinks 连接）
    ├── thigmotaxis.md
    ├── anxiety-like-behavior.md
    └── ...
```

**Agent 调用方式**（不是预处理）：

```
agent 调 MCP 工具：
  search_knowledge("OFT", "center time reduced stress")
    → rg "center time" paradigms/OFT/ -l
    → 返回匹配文件列表 + 行号 + 上下文

  read_chunk("paradigms/OFT/2020_smith_chronic_stress.md", lines=10-30)
    → 返回该论文的完整摘要段落

  search_manual("EthoVisionXT", "zone detection calibration")
    → rg "zone" manuals/EthoVisionXT/ -C 3
    → 返回匹配段落

  get_term("thigmotaxis")
    → jq '.[] | select(.name=="thigmotaxis")' taxonomy/terms.json
    → 返回定义 + 相关概念链接
```

---

## 4. noldus-kb 的具体改造方案

### 4.1 要保留的

| 资产 | 理由 |
|------|------|
| MCP server 架构 | 接口设计合理，只换后端实现 |
| CLI client | `noldus-kb search "..."` 作为开发调试入口 |
| taxonomy JSON | 精确查询的最简方案，30KB，不动 |
| 论文元数据提取（ingest 管线） | LLM 提取标题/作者/范式/产品是核心价值，保留 |

### 4.2 要删的

| 组件 | 理由 |
|------|------|
| BGE-M3 embedding 管线 | 替换为文件系统 |
| pgvector 向量索引 | 替换为 ripgrep |
| chunker（512-token 切片） | 替换为 paper-level markdown 文件 |
| PostgreSQL 全文检索（可选保留为增强层） | 如果 rg 够用就不需要 |

### 4.3 要新增的

#### 4.3.1 论文摘要文件生成管线

改造现有 `ingest/pipeline.py`：
- **保留** marker-pdf 解析（PDF → markdown）
- **保留** LLM 元数据提取（标题、作者、年份、范式、产品）
- **新增** LLM 结构化摘要生成：每篇论文产出一个 markdown 文件，包含：

```markdown
# Chronic Stress Reduces Center Time in Open Field Test

- **作者**: Smith et al.
- **年份**: 2020
- **期刊**: Behavioural Brain Research
- **范式**: Open Field Test
- **动物**: C57BL/6J mice, male, 8 weeks
- **处理**: Chronic mild stress (CMS), 4 weeks
- **主要发现**:
  - center time: ↓42% vs control (p<0.01, Cohen's d=1.24)
  - total distance: 无显著差异
  - rearing: ↓35% vs control (p<0.05)
- **与当前判读的相关性**: [待 RCS 填充]

## 摘要

[LLM 生成的 200-300 字中文情境摘要]

## 关键原文段落

> The CMS group spent significantly less time in the center zone
> (42.3 ± 8.1 s) compared to controls (68.7 ± 10.2 s, t(18)=3.42, p<0.01),
> indicating increased anxiety-like behavior. — p.12

> No significant difference was observed in total distance traveled
> between groups (p=0.34), suggesting the effect was not due to
> locomotor impairment. — p.13
```

- **存储**：`data/paradigms/{范式}/{年份}_{第一作者}_{简短标题}.md`
- **去重**：已有 SHA-256 哈希去重逻辑，保留

#### 4.3.2 Manual 知识 curated 文件

半自动流程：
1. marker-pdf 把 manual PDF → markdown（现有能力）
2. LLM 按章节标题 + 内容聚类生成问题导向文件
3. 人工或 LLM 整理成 `症状 → 可能原因 → 排查/解决步骤` 结构

优先级：先做 EthoVision XT（最核心），再做 CatWalk、PhenoTyper 等。

#### 4.3.3 MCP 工具重新设计

从 6 个工具改为更精炼的 8 个工具：

| 工具 | 实现 | 用途 |
|------|------|------|
| `search_papers(paradigm, query)` | `rg query paradigms/{paradigm}/ -C 3` | 场景 B：按范式搜索论文摘要 |
| `search_manual(product, query)` | `rg query manuals/{product}/ -C 5` | 场景 A：搜索产品手册 |
| `get_term(term)` | `jq` 查 `taxonomy/terms.json` | 精确术语定义 |
| `get_paradigm(name)` | `jq` 查 `taxonomy/paradigms.json` | 范式基本信息 |
| `get_product(name)` | `jq` 查 `taxonomy/products.json` | 产品基本信息 |
| `read_paper(path, lines)` | `cat` / `sed` 读文件指定行 | 下钻读取论文完整摘要 |
| `read_manual(path, lines)` | `cat` / `sed` 读文件指定行 | 下钻读取手册章节 |
| `list_papers(paradigm)` | `ls` / `find` 列目录 | 浏览某范式下所有论文 |

去掉了 `list_products` 和 `list_paradigms`（数据量小，agent 自己 grep 即可），新增了下钻 `read_*` 工具（给 agent 从摘要定位到原文的能力）。

#### 4.3.4 概念层（concepts/）

跨范式的概念知识，用 `[[wikilinks]]` 连接：

```markdown
# thigmotaxis.md
## Thigmotaxis（趋触性）

定义：动物倾向于靠近墙壁/边缘的行为。高趋触性 = 焦虑样行为。

相关范式：[[OFT]], [[EPM]], [[LDB]]
相关术语：[[anxiety-like-behavior]], [[center-avoidance]]
相关产品：[[EthoVisionXT]]

## 已发表文献参考
- Smith 2020: CMS 增加 thigmotaxis（见 [[paradigms/OFT/2020_smith_chronic_stress]]）
- Chen 2019: 化合物 X 减少 thigmotaxis（见 [[paradigms/OFT/2019_chen_anxiolytic]]）
```

Agent 搜 "thigmotaxis" → grep 命中 `concepts/thigmotaxis.md` → 读到定义 + 看到 `[[paradigms/OFT/2020_smith_chronic_stress]]` → `read_paper` 下钻。

---

## 5. 实施路径

### Phase 1：核心管线（1-2 周）

1. **保留现有 MCP server 框架**，换后端实现
2. **实现 8 个 MCP 工具**（rg/jq/cat 实现）
3. **手工创建 3 个范式的示例目录**：OFT + EPM + FST（各 5-10 篇论文摘要 + 1 个 index.md）
4. **手工创建 EthoVision XT 手册目录**（zone-detection + troubleshooting + data-export）
5. **在 agent dogfood 中验证**：场景 A（故障联想）+ 场景 B（证据锚定）

### Phase 2：规模化（2-4 周）

6. **改造 ingest 管线**：从"chunk + embedding"改为"paper-level markdown + 范式目录"
7. **批量处理现有论文**：如果有原始 PDF，用新管线重新生成摘要文件
8. **扩展 manual 覆盖**：CatWalk XT、PhenoTyper 等
9. **概念层初始内容**：10-15 个核心概念（thigmotaxis、anxiety、locomotion 等）

### Phase 3：增强（按需）

10. **BM25 增强层**：如果 agent 反馈 grep 召回不够，加 ParadeDB pg_search 或 tantivy 做 BM25 排序
11. **LLM RCS rerank（可选）**：agent 拿到的候选太多时，加一层 LLM 相关性打分
12. **wikilinks 自动维护**：LLM 自动生成和校验跨文件链接（lat.md 的 `lat check` 模式）

---

## 6. 风险与应对

| 风险 | 应对 |
|------|------|
| grep 召回不足（recall 低于 embedding） | 范式目录预过滤 + CSQE 查询扩展 + 可选 BM25 增强层 |
| 论文摘要提取质量不稳定 | 每字段带 verbatim span + 字符串对齐校验（SAFEPASSAGE 模式） |
| 6200 篇论文生成摘要的 LLM 成本 | 先用范式筛选（每个范式论文量不同），优先 v0.1 已支持的 6 个范式 |
| Manual 整理人工成本高 | 先做 EthoVision XT（最常用），其他产品按需扩展 |
| 文件名/路径成为瓶颈 | 统一命名规范：`{年份}_{第一作者}_{简短标题}.md`，CLI 校验工具 |

---

## 7. 与现有方案的对比总结

| | 现有 RAG (noldus-kb) | 新方案 (Agent-Native grep) |
|---|---|---|
| 检索位置 | LLM 之前（透明预处理） | Agent 工具调用（agent 主动） |
| 索引粒度 | 512-token chunk | Paper-level markdown |
| 检索方式 | BGE-M3 embedding + cosine | ripgrep + jq |
| 基础设施 | GPU + pgvector + PostgreSQL | 纯文件系统 |
| 灌入时间 | ~42 小时（6200 篇） | LLM 摘要生成（按需，无需全量预灌） |
| 可调试性 | 低（向量距离不可解释） | 高（文件路径+行号） |
| 可纠错性 | agent 无法纠错（不知道 chunk 哪来的） | agent 可以换关键词重搜、下钻读原文 |
| 成本 | GPU 常驻 + embedding API | 零额外基础设施 |
| 接口 | MCP/CLI/REST（保留） | MCP/CLI/REST（保留，工具换个实现） |

---

## 8. 参考

- Vercel Knowledge Agent Template: [github.com/vercel-labs/knowledge-agent-template](https://github.com/vercel-labs/knowledge-agent-template)
- Vercel — Build Knowledge Agents Without Embeddings: [vercel.com/blog](https://vercel.com/blog/build-knowledge-agents-without-embeddings)
- Karpathy LLM Wiki Pattern: [lobehub.com/skills](https://lobehub.com/skills/chf3198-devenv-ops-llm-wiki-ops-portable)
- lat.md: [github.com/1st1/lat.md](https://github.com/1st1/lat.md)
- MinerU Document Explorer: [github.com/opendatalab/MinerU-Document-Explorer](https://github.com/opendatalab/MinerU-Document-Explorer)
- 中文社区趋势分析: [RAG退潮，文件系统+grep回归](https://www.e-com-net.com/article/2041002788447641600.htm)
- PaperQA2 (RCS rerank): [github.com/Future-House/paper-qa](https://github.com/Future-House/paper-qa)
