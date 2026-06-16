# Handoff — noldus-kb 知识库改造：从 RAG 到 Agent-Native grep + SQL + vector

> 日期：2026-06-10 ｜ 写给下一位接手的 AI Agent

---

## 0. 一句话现状

将 noldus-kb 从"embedding → pgvector → chunk 拼接"的 RAG 管线，改造成了"文件系统 markdown + grep + SQL + vector 语义兜底"的 Agent 工具型知识库。**346 篇 EthoVision 论文已完成分类、markdown 化、embedding。**下一步是建 MCP 搜索工具 + 补论文数量。

---

## 1. 仓库坐标

- **noldus-kb 仓库**：`/home/wangqiuyang/noldus-kb`
- **EthoInsight 仓库**：`/home/wangqiuyang/noldus-insight`
- **Python 环境**：`noldus-kb/.venv`（RTX 5090, CUDA 12.8, torch 2.11.0+cu128）
- **论文数据**：
  - XLSX：`noldus-kb/Noldus产品使用文献整理追新版.xlsx`（286 EthoVision）
  - CSV：`noldus-kb/研究文献整理-EV.csv`（~109 EthoVision）
  - PostgreSQL dump：`noldus-kb/noldus_kb.sql.gz`（2763 done papers, 116 with chunk text）
- **设计文档**：
  - `noldus-insight/docs/design/2026-06-09-noldus-kb-redesign-plan.md` — 改造方案
  - `noldus-insight/docs/design/2026-06-09-noldus-kb-evolution-roadmap.md` — 演进路线
  - `noldus-insight/docs/design/2026-06-09-noldus-kb-grep-vs-rag-design.md` — 初始讨论
  - `noldus-insight/docs/design/2026-06-09-noldus-kb-research-findings.md` — Opus 调研报告

---

## 2. 已完成的（✅）

### 数据收集与合并
- ✅ XLSX + CSV 合并去重 → 392 篇 EthoVision 论文标题
- ✅ OpenAlex API 反查（API key: `4M7eqPNtoxgOmD8sptXBhH`）→ 346 篇命中（88.3%）
- ✅ 拿到 DOI（344）、摘要（241）、作者、期刊、主题分类
- ✅ 与 PostgreSQL dump 交叉匹配 → 69 篇存在两边的标题交集（但 chunk 文本不完整）

### 论文分类与存储
- ✅ 正则范式分类（28 类：OFT 15, EPM 15, FST 3, LDB 2, Neurodegeneration 40 等）
- ✅ 346 篇 markdown 文件生成到 `data/papers/{范式}/*.md`
- ✅ 每篇含：DOI、作者、期刊、年份、摘要、OpenAlex 主题、关键词、Noldus 产品、PG 路径

### Embedding
- ✅ Qwen3-Embedding-0.6B 部署（比 BGE-M3 高 5.77 分 MTEB 检索）
- ✅ 346 × 1024d 向量 → `data/embeddings.npy`
- ✅ 论文索引 → `data/paper_index.json`

### 环境
- ✅ torch 2.11.0+cu128（RTX 5090 Blackwell sm_120 兼容）
- ✅ pyproject.toml 的 torch 源改为 cu128

---

## 3. 关键上下文

### 架构决策

**四层漏斗**（不是 grep vs RAG 的二选一）：

```
SQL（元数据过滤）→ grep/BM25（全文检索）→ Vector（语义兜底）→ Agent 整合
   Postgres              ripgrep             pgvector          Agent MCP 工具
```

核心洞察：PDF 是论文知识最差的起点。JATS XML / API metadata 远优于 PDF 原文。

### 知识库目录结构

```
data/
├── papers/{范式}/*.md      ← 346 篇 markdown（标题 + DOI + 摘要 + 主题）
├── embeddings.npy          ← 346 × 1024d 向量
├── paper_index.json        ← [id, title, paradigm, doi, filepath]
└── taxonomy/               ← 原有的 products/paradigms/terms JSON
```

### 论文数据源不完整的原因

PostgreSQL dump 只有 116/2763 篇有 chunk 文本。生产服务器（180.184.84.124）已不可达。69 篇标题匹配但只 2 篇拿到全文。278 篇 OA-only（不在 PG dump 里）。46 篇 OA 也未找到（标题乱码/太新）。

### 旧 noldus-kb RAG 管线

仍然存在但不使用：`src/noldus_kb/ingest/pipeline.py`、`src/noldus_kb/core/retriever.py`（pgvector hybrid search）。MCP server 框架保留（接口设计合理），后端实现要换。

### 已装依赖

```bash
cd /home/wangqiuyang/noldus-kb
uv run python3  # 使用 .venv，含 torch 2.11+cu128 + sentence-transformers
```

---

## 4. 未完成事项（按优先级）

### P0：MCP 搜索工具（知识库可用化的最后一步）

写 MCP 工具暴露给 agent：

| 工具 | 实现 | 用途 |
|------|------|------|
| `search_papers(query, paradigm?)` | RRF 融合：vector 余弦 + grep 关键词 | 场景 B：证据锚定 |
| `list_papers(paradigm, year_range?)` | SQL / grep 目录列表 | 浏览某范式论文 |
| `read_paper(path)` | cat markdown 文件 | 下钻读论文摘要 |
| `get_term(term)` | jq 查 taxonomy/terms.json | 术语定义 |
| `get_paradigm(name)` | jq 查 taxonomy/paradigms.json | 范式信息 |
| `get_product(name)` | jq 查 taxonomy/products.json | 产品信息 |

当前 embedding 检索代码参考 `regenerate.py` 末尾的搜索 demo（cosine 相似度 + top-k）。

grep 全文检索直接用 `rg query data/papers/{paradigm}/ -C 3`，范式参数做目录预过滤。

RRF 融合：vector top-20 + grep top-20 → Reciprocal Rank Fusion (k=60) → top-10。

### P1：补论文数量

- **278 篇 OA-only**：拿 DOI 去 Semantic Scholar / Europe PMC 补摘要。脚本已有 `enrich_openalex.py`，改指向 Semantic Scholar API 即可
- **46 篇未找到**：标题乱码或用作者名重搜
- **PostgreSQL dump 里的 2763 篇**：这些有标题+文件路径+产品信息，可以批量跑 OpenAlex/Semantic Scholar 补充 metadata，不依赖 chunk 文本

### P2：LLM 精细分类

当前正则分类粗（~70-80% 准确）。用 LLM 对每篇论文做一次判读：
- 这篇主要用什么行为学范式？
- 什么动物模型？
- 用什么 Noldus 产品？（除了 EthoVision 还有其他吗？）
- 主要发现方向（什么指标升高/降低）

可用现有 GLM API key 批量处理 346 篇。

### P3：Manual 知识整理

EthoVision XT 手册 → 问题导向 markdown（diagnostic + how-to）。流程：marker-pdf → markdown → LLM 按主题聚类 → `manuals/EthoVisionXT/` 目录。参考设计文档 §4.3.2。

### P4：概念层（concepts/）

跨范式概念知识：thigmotaxis、anxiety-like-behavior、locomotor-activity 等，用 `[[wikilinks]]` 连接论文和术语。参考 Karpathy LLM Wiki 模式。

---

## 5. 工具脚本清单

| 文件 | 作用 |
|------|------|
| `enrich_openalex.py` | OpenAlex API 反查（标题 → DOI + 摘要 + 主题） |
| `generate_markdown.py` | 从 enriched JSON 生成 markdown（旧版，不完整） |
| `merge_fulltext.py` | 合并 OpenAlex + PostgreSQL dump |
| `regenerate.py` | **主脚本**：合并数据 → markdown → embedding（当前使用的） |
| `ev_papers_enriched.json` | OpenAlex 反查结果（346 found + 46 not_found） |
| `ev_papers_merged_fulltext.json` | 合并后数据（OA + PG） |

---

## 6. 下一位 Agent 的第一步

```bash
cd /home/wangqiuyang/noldus-kb

# 1. 验证环境
uv run python3 -c "import torch; print(torch.cuda.is_available())"
# 应输出 True

# 2. 查看现有数据
ls data/papers/*/
uv run python3 -c "
import json, numpy as np
e = np.load('data/embeddings.npy')
with open('data/paper_index.json') as f:
    idx = json.load(f)
print(f'Papers: {len(idx)}, Embeddings: {e.shape}')
"

# 3. 快速搜索测试
uv run python3 -c "
from sentence_transformers import SentenceTransformer
import numpy as np, json
model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B', device='cuda', trust_remote_code=True)
e = np.load('data/embeddings.npy')
with open('data/paper_index.json') as f: idx = json.load(f)
q = model.encode(['Instruct: Retrieve relevant papers.\nQuery: open field test center time reduced anxiety'], normalize_embeddings=True)
top5 = np.argsort(np.dot(e, q[0]))[-5:][::-1]
for rank, i in enumerate(top5): print(f'{rank+1}. [{idx[i][\"paradigm\"]}] {idx[i][\"title\"][:70]}')
"
```

然后开始写 MCP 搜索工具（P0）。

---

## 7. 决策记录

| 决策 | 结论 | 理由 |
|------|------|------|
| 检索架构 | SQL + grep + vector 四层漏斗 | PaperQA2 ablation: chunk/embedding 调参收益小，rerank/摘要层是胜负手 |
| embedding 模型 | Qwen3-Embedding-0.6B | MTEB 检索 64.33 vs BGE-M3 54.60，同体积高 5.77 分 |
| PDF vs API | 优先 API，PDF 做 fallback | JATS XML > PDF chunk，LitSearch 证实全文检索不如 abstract |
| 检索粒度 | Paper-level 摘要（非 chunk） | RAPTOR + LitSearch + LlamaIndex Document Summary Index 背书 |
| 手册知识 | Diagnostic + How-to 双结构 curated markdown | AWS Prescriptive Guidance + KCS Symptom-Cause-Resolution 模式 |

## 8. 已确认不做的方向

- ❌ 纯 grep 替代所有检索（BM25 比 dense 在科学文献上低 24.8pt，不能只用 grep）
- ❌ 恢复旧的 chunk-level RAG 管线（与 agent 工具调用范式冲突，噪声比信号高）
- ❌ 从 PDF 开始重建（API metadata 远优于 PDF 解析）
