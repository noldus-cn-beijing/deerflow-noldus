# noldus-kb 演进路线：不要从 PDF 开始

> 日期：2026-06-09 ｜ 基于学术 API 调研 + 四层漏斗架构

---

## 1. 核心发现：PDF 是最差的起点

当前 noldus-kb 的灌入管线是 `PDF → marker-pdf 解析 → chunk → BGE-M3 embedding`。但行为学论文在 PDF 之前，有更好的格式：

| 格式 | 来源 | 结构 | 获取成本 |
|------|------|------|---------|
| **JATS XML** | PubMed Central / Europe PMC | `<abstract>`, `<methods>`, `<results>` 标签已分好 | 免费 API，无需 GPU |
| **JSON metadata** | Semantic Scholar / CrossRef / OpenAlex | 标题、作者、年份、摘要、引用、TLDR | 免费 API，无需 API key |
| **BioC JSON** | NCBI BioC API | 比 JATS 更简单，专为文本挖掘设计 | 免费，3 req/s |
| **PDF** | 本地 / 出版社 | 无结构、OCR 错误、格式混乱 | 需要 GPU + 42 小时 |

**关键数字**：
- Europe PMC：4000 万+ 文章，800 万+ 全文 XML，无需 API key
- Semantic Scholar：2 亿+ 论文，每篇免费 TLDR 摘要
- BioC API：PMC 开放获取论文的结构化 JSON，按段落分好

**大多数行为学论文不需要碰 PDF**。它们的 metadata + abstract 已经通过 API 结构化可用了。即使需要全文，JATS XML 也比 PDF 好一万倍——章节已分好，没有 OCR 错误。

---

## 2. noldus-kb 数据获取优先级

```
最优（优先使用）：
  Semantic Scholar API → 标题 + 摘要 + TLDR + 引用关系 + 范式分类
  Europe PMC API    → 全文 XML（OA 论文）+ 结构化段落
  PubMed E-utilities → MeSH 术语 + 搜索过滤

次优（XML 不可用时）：
  marker-pdf        → PDF → markdown（保留为 fallback）

仅当以上都不可用时：
  原始 PDF          → 传统灌入管线
```

---

## 3. 演进路线：三个 Phase

### Phase 1：用 API 重建论文知识库（1-2 周）

**目标**：不碰 PDF，用免费 API 拿到所有能拿到的结构化数据

```
Semantic Scholar API                    Europe PMC API
      │                                      │
      ├─ 论文搜索（按范式关键词）              ├─ 全文 XML（OA 论文）
      ├─ 元数据（标题、作者、年份、期刊）       ├─ 结构化段落
      ├─ TLDR 摘要（AI 生成，免费）            └─ 参考文献
      ├─ 引用关系
      └─ 领域分类
              │                              │
              └──────────┬───────────────────┘
                         ▼
              生成 paper-level markdown
              按范式分目录存储
              data/papers/{范式}/{年份}_{作者}_{slug}.md
                         │
                         ▼
              Postgres 元数据表（SQL 查询用）
              + 文件系统（grep 搜索用）
              + pgvector（语义兜底，按需）
```

**每篇论文的 markdown 模板**：

```markdown
# 标题

- **作者**: ...
- **年份**: 2020
- **期刊**: Behavioural Brain Research
- **DOI**: 10.xxx/xxx
- **范式**: Open Field Test
- **动物**: C57BL/6J, male, 8w
- **处理**: Chronic mild stress, 4w
- **Semantic Scholar TLDR**: [AI 自动摘要]
- **关键词**: stress, anxiety, center time, locomotion

## 摘要

[Europe PMC 的结构化 abstract，或 Semantic Scholar 摘要]

## 主要发现（LLM 提取）

- center time: ↓42% vs control (p<0.01, d=1.24)
- total distance: ns (p=0.34)
- rearing: ↓35% (p<0.05)

## 关键原文段落

> [从 JATS XML 提取的结果段原文]

## 参考文献

- [DOI 链接]
```

**此阶段产出**：
- `data/papers/OFT/`、`data/papers/EPM/` 等目录，每范式 50-200 篇 markdown
- Postgres `papers` 表（id, title, authors, year, journal, doi, paradigm, terms, file_path）
- MCP 工具 `search_papers(paradigm, query)` → rg 全文检索
- MCP 工具 `list_papers(paradigm, year_range, terms)` → SQL 查询

### Phase 2：Manual 知识整理（1-2 周）

**目标**：把 Noldus 产品手册整理成问题导向 markdown

```
EthoVision XT 手册 PDF
        │
        ▼
  marker-pdf → markdown（一次性）
        │
        ▼
  LLM 按主题聚类 → 诊断型 + How-to 型
        │
        ▼
  data/manuals/EthoVisionXT/
  ├── index.md                   # 手册知识索引
  ├── zone-detection.md          # 症状→原因→排查
  ├── body-point-tracking.md
  ├── data-export.md
  ├── experiment-setup.md
  └── troubleshooting.md
```

**此阶段产出**：
- `data/manuals/EthoVisionXT/` 完整目录
- MCP 工具 `search_manual(product, query)` → rg 全文检索
- MCP 工具 `read_manual(path, lines)` → 下钻读章节

### Phase 3：概念层 + Agent 集成（1 周）

**目标**：跨范式概念知识 + 接入 agent 工具循环

```
data/concepts/
├── index.md                    # 概念索引
├── thigmotaxis.md              # 定义 + 相关范式 + 相关论文链接
├── anxiety-like-behavior.md
├── locomotor-activity.md
└── ...

每个概念文件 = 定义 + [[wikilinks]] 到范式/论文/术语
```

**此阶段产出**：
- `data/concepts/` 目录（10-15 个核心概念）
- Agent 通过 MCP 工具调用知识库（不是预处理）
- Agent dogfood 验证：场景 A（故障联想）+ 场景 B（证据锚定）

---

## 4. 最终架构

```
┌─────────────────────────────────────────────────────────┐
│                    Agent 工具层                           │
│  search_papers | list_papers | read_paper | get_term    │
│  search_manual | read_manual | get_paradigm | get_product│
└──────┬────────────┬──────────────┬──────────────────────┘
       │            │              │
       ▼            ▼              ▼
┌─ SQL ────┐ ┌─ grep ──┐ ┌─ Vector ──┐
│ Postgres  │ │ ripgrep │ │ pgvector  │
│ 元数据过滤 │ │ 全文检索 │ │ 语义兜底   │
│           │ │         │ │ (摘要级)   │
│ "OFT 范式  │ │ "center  │ │ "zone     │
│  2015-2025│ │  time    │ │  tracking │
│  stress"  │ │  reduced"│ │  loss"    │
└─────┬─────┘ └────┬────┘ └─────┬─────┘
      │            │            │
      └────────────┼────────────┘
                   ▼
┌─────────────────────────────────────────────────────────┐
│                    数据层（文件系统）                      │
│                                                         │
│  data/papers/{范式}/*.md     ← 从 API 生成，不碰 PDF     │
│  data/manuals/{产品}/*.md    ← 问题导向 curated          │
│  data/concepts/*.md          ← 跨范式 [[wikilinks]]     │
│  data/taxonomy/*.json        ← 已有，不动                │
└─────────────────────────────────────────────────────────┘
```

**基础设施**：一台不需要 GPU 的服务器，Postgres + ripgrep + 文件系统。

---

## 5. 论文数据获取：具体 API 方案

### 5.1 Semantic Scholar（首选）

```
免费、无需 API key、200M+ 论文

搜索示例：
  GET https://api.semanticscholar.org/graph/v1/paper/search?
    query=open+field+test+center+time+stress+
    year=2015-2025+
    fields=title,authors,year,abstract,tldr,externalIds,openAccessPdf

返回：标题、作者、年份、摘要、AI TLDR、DOI、OA PDF URL（如有）

速率限制：100 req / 5min（无 key），100 req / sec（有 key）
```

### 5.2 Europe PMC（全文 XML）

```
免费、无需 API key、40M+ 文章、8M+ 全文 XML

搜索示例：
  GET https://www.ebi.ac.uk/europepmc/webservices/rest/search?
    query=open+field+test+AND+center+time+
    resultType=core+
    format=json

全文 XML：
  GET https://www.ebi.ac.uk/europepmc/webservices/rest/{PMCID}/fullTextXML
```

### 5.3 PubMed E-utilities（MeSH 术语）

```
搜索 + 过滤：
  esearch: db=pubmed, term=open field test[tiab] AND stress[mesh], retmax=100
  efetch:  db=pubmed, id={pmids}, rettype=abstract, retmode=xml
  elink:   dbfrom=pubmed, db=pmc, id={pmid} → 拿到 PMCID 去 Europe PMC 取全文

速率：3 req/s（无 key），10 req/s（有 key）
```

### 5.4 数据获取策略

```
第 1 轮：Semantic Scholar 搜索
  → 每个范式关键词搜索，拿 metadata + TLDR
  → 生成基础 markdown（标题、摘要、TLDR）
  → 存入 Postgres 元数据表 + data/papers/{范式}/

第 2 轮：Europe PMC 全文增强
  → 对有 PMCID/DOI 的论文，尝试取全文 XML
  → 从 XML 提取 <abstract>、<results>、关键段落
  → 更新 markdown 文件

第 3 轮：LLM 增强（可选）
  → 对重点论文（高引用、经典文献），LLM 提取结构化发现
  → 补充 "主要发现" 和 "关键原文段落" 字段

Fallback：PDF → marker-pdf
  → 仅用于 API 拿不到的论文（付费墙、会议论文、预印本）
```

---

## 6. 为什么这个路线优于现有 RAG 管线

| | 现有 RAG | 新路线 |
|---|---|---|
| 数据源 | PDF（OCR 错误、无结构） | API（JATS XML / JSON，已结构化） |
| 解析 | marker-pdf（GPU，~2 PDF/min） | API 调用（CPU，秒级） |
| 灌入 6200 篇 | ~42 小时 | ~2 小时（API 调用 + 文件生成） |
| 基础设施 | GPU + pgvector + embedding | 纯 CPU + Postgres + ripgrep |
| 摘要质量 | 512-token chunk 拼接 | API 提供的结构化 abstract + TLDR |
| 可搜索性 | embedding cosine（不可解释） | SQL 元数据 + grep 全文 + vector 语义兜底 |
| Agent 集成 | 透明预处理（agent 无法控制） | MCP 工具（agent 主动调用） |

---

## 7. 第一步行动

1. **验证 API 覆盖率**：从现有 6200 篇论文里随机抽 50 篇，查 Semantic Scholar + Europe PMC 能拿到多少 metadata/全文
2. **建一个范式试点**：选 OFT（论文量适中），完整走通 API → markdown → Postgres → MCP 工具的全链路
3. **Agent dogfood**：用试点数据在真实 agent 场景验证 SQL+grep+vector 四层漏斗

---

## 8. 参考

- [Semantic Scholar API](https://api.semanticscholar.org/) — 免费、200M+ 论文、TLDR 摘要
- [Europe PMC API](https://europepmc.org/RestfulWebService) — 40M+ 文章、8M+ 全文 XML、无需 key
- [NCBI BioC API](https://www.ncbi.nlm.nih.gov/research/bionlp/APIs/BioC-PMC/) — PMC OA 论文的 text-mining 友好 JSON
- [PubMed E-utilities](https://www.ncbi.nlm.nih.gov/books/NBK25501/) — MeSH 搜索 + PMCID 映射
- [OpenAlex API](https://openalex.org/) — 250M+ works, CC0, 引用网络
- [cadmus](https://www.lifescience.net/preprints/18027/) — 生物医学文献检索 pipeline，85% 全文获取率
- [BrainGPT](https://huggingface.co/datasets/BrainGPT/train_valid_split_pmc_neuroscience_2002-2022_filtered_subset) — 已构建的神经科学语料（332K abstract + 123K full text），可直接复用
