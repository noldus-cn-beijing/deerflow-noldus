# Init 设计：Experiment 跨范式对比层 —— 让 EthoInsight 从 chatbox 走向真正的 agent

> 状态：**init 设计文档（feature 立项）**，非实施 spec。落地拆分见末尾路线图。
> 日期：2026-06-26
> 代码基线：dev HEAD `776450b2`
> 性质：🟢 新 feature 立项 · 后端 persistence + gateway + subagent + 前端
> 阻塞：**判读层（"跨范式相同/不同"怎么算才科学）需行为学同事方法论**——工程骨架可先建，synthesizer 的判读规则挂着等同事（同 Golden Cases #90 的依赖性质）。已开 [Issue #226](https://github.com/noldus-cn-beijing/noldus-insight/issues/226) 给同事填写，最终落成新 skill `ethoinsight-cross-paradigm`。

---

## 〇、用户需求（原话提炼）

> 行为学实验有个关键叫**跨范式对比**：同一批老鼠先做 EPM（新建 thread 上传 28 个实验），再做 OFT（新开 thread）。用户有"跨范式对比"需求——主要对比的是 **data-analyst 的结论**：同样的老鼠做两个范式，跨范式有哪些相同与不同。
>
> 这个 feature 现在可以 **init**：每个 thread 可被 **archive 为一个实验**；用户把不同 thread archive 到**同一个实验**下；agent 做类似 **/dream 的功能**——把这些 thread 的关键信息拿去对比，对这个实验得出结论；用户之后还能加入新 thread 的结论。
>
> **"我希望我们这个项目更像一个真正的 agent，而不是单纯的 chatbox。"**

用户拍板的工程约束：
1. **import = archive**：把 thread 归入 experiment。
2. **工程化提取**：import 时**直接提取 imported thread 的具体文件**（落到 experiment 目录），**不靠运行时跨 thread 读**（thread 之间文件本就隔离，见配套路径 spec）。
3. **≥2 才可比**：只 import 1 个 thread 不可能对比；≥2 才触发 synthesize。
4. **不替用户识别**：不判断这两个 thread 是否"真属同一实验/同一批老鼠"——那是用户的责任，我们只做对比。

---

## 一、为什么这是"真 agent"而非 chatbox（范式正当性）

参考 blog《让 AI 主动管理自己的上下文》(https://blog.xlab.app/p/6a966aeb/，记于 memory `reference_agentic_context_engineering_session_tree_v1_direction`)：把多个会话视为可回溯的 **session tree**，**总结＝把分支合并到主线（mr-commit），重要内容沉淀、次要稀释——这本身接近"记忆"**。

Experiment 层正是这个范式的落地形态：
- 每个 thread＝一条已完成的分析分支（含 data-analyst 结论）。
- experiment＝把 N 条分支**归一**的容器。
- synthesizer＝blog 的 **SUM 节点**：把多分支关键信息总结成一条跨范式结论（主线）。
- "用户之后还能加新 thread 的结论"＝持续向主线追加 summary。

这给了 feature 理论靠山：不是拍脑袋发明的 UI 功能，是 **agentic context/记忆范式**在行为学场景的自然实例。**chatbox 是"一问一答即焚"，agent 是"跨会话沉淀出实验级结论并可持续生长"**——这正是用户要的区别。

> 注意边界：blog 的完整 agentic 上下文管理（context_tag/log/checkout 工具）是 **v1.0 方向**，本 feature **不实现那套工具**，只借"多分支总结到主线"的范式。

---

## 二、核心架构：新增 experiment 层级（DB + 磁盘 + agent）

### 2.1 与现有架构的关系

```
现有：  user ─┬─ thread A (EPM)  ── user-data/{workspace,uploads,outputs} ── handoff_data_analyst.json
              └─ thread B (OFT)  ── （thread 之间文件完全隔离，互相读不到）

新增：  user ── experiment X ─┬─ import(thread A) → 提取 A 的关键文件快照到 experiment 目录
                              ├─ import(thread B) → 提取 B 的关键文件快照
                              └─ synthesize → synthesizer 读两份快照 → 跨范式结论
```

**关键设计决策：快照而非引用**。import 时把 thread 的关键文件**复制**进 experiment 专属目录（独立副本），原 thread 即使后来被删，experiment 结论仍在。这直接绕开"thread 文件隔离、跨 thread 读不到"的结构障碍（见配套 `2026-06-26-file-path-reliability-loadbearing-convergence-spec.md`），且符合用户"工程化提取具体文件"的明确要求。

### 2.2 持久化层（新表，仿现有 persistence 模式）

现有表：`users` / `threads_meta` / `runs` / `run_events` / `feedback`，模型在 `packages/agent/backend/packages/harness/deerflow/persistence/<name>/model.py`，仓储仿 `thread_meta/sql.py` 的 `ThreadMetaRepository`，store 注入仿 `app/gateway/deps.py`。

**新增 3 表**（草案，实施 spec 细化）：
- `experiments`：`experiment_id` PK、`user_id`（index，user 隔离）、`display_name`、`description`、`created_at`/`updated_at`、`metadata_json`。
- `experiment_threads`（关联）：`experiment_id` + `thread_id` 唯一约束、`import_order`、`extracted_files_manifest`（JSON：提取的文件→experiment 内相对路径）、`imported_at`。**加外键**（守配套路径 spec 的承重墙原则：`experiment_id` → `experiments` ON DELETE CASCADE）。
- `experiment_synthesis`：`synthesis_id` PK、`experiment_id`、`status`（pending/running/completed/failed）、`result_json`、时间戳、`error`。

**迁移**：新文件仿 `migrations/versions/20260622_1700_*.py`；三张全新表是 CREATE 非 ALTER。**部署链不跑 alembic → 上线手动迁移**（守 memory `feedback_deploy_alembic_migration_for_added_columns`）。

### 2.3 磁盘层（experiment 专属目录）

在 `Paths` 加 experiment 目录方法（`config/paths.py` 受保护 → surgical 加方法不动既有）：

```
{base_dir}/users/{user_id}/experiments/{experiment_id}/
├── metadata.json
├── thread_{thread_id}/              # 每个 imported thread 的快照
│   ├── handoff_data_analyst.json    # ★ 跨范式对比核心
│   ├── handoff_code_executor.json   # fallback 原始统计
│   ├── report.md
│   ├── experiment_context.json      # 范式/分组元数据
│   └── charts/plot_*.png
└── synthesis/
    ├── handoff_synthesizer.json     # 综合结论
    └── charts/                       # 对比图（可选）
```

### 2.4 import 提取清单（"具体文件"= 哪些）

按优先级从 thread 的 `sandbox_work_dir` / `sandbox_outputs_dir` 提取（路径走解析单点，**不裸拼**）：

| 优先级 | 文件 | 源 | 用途 |
|---|---|---|---|
| P0 | `handoff_data_analyst.json` | workspace | 跨范式对比核心（key_findings/per_subject/statistics） |
| P0 | `experiment_context.json` | workspace | 范式标识、分组 |
| P1 | `report.md` | outputs | 人类可读结论 |
| P1 | `handoff_code_executor.json` | workspace | data-analyst 缺失时 fallback |
| P2 | `plot_*.png` | outputs | 图表快照 |

**import 前置校验**：源 thread 必须有 `handoff_data_analyst.json`（否则该 thread 还没产出可对比的结论）→ 缺则 400 提示"请先完成该 thread 的分析"。这避免 import 半成品（守 memory `feedback_subagent_consumption_via_first_party_tool`：消费包内文件走 first-party 路径）。

### 2.5 Synthesizer（跨范式对比 agent）

**两条接法，init 阶段倾向 A，实施时按 lead 触发复杂度定**：

- **A — 新增 `experiment-synthesizer` subagent**（推荐）：注册进 `subagents/builtins/__init__.py`（`BUILTIN_SUBAGENTS`，受保护→surgical）+ `handoff_registry.py` 加 `experiment_synthesizer: handoff_synthesizer.json` + lead prompt 加触发指引（守 CLAUDE.md 第 1 条三件套）。输入＝提取好的 N 份 handoff 快照（**不跨 thread 读、读 experiment 目录内的副本**），输出＝`handoff_synthesizer.json`（cross_paradigm_findings / paradigm_differences / integration_summary / recommendations / warnings）。
- **B — lead 直接对比**（n=2 轻量）：lead 读两份快照 inline 对比。**缺点**：lead 不受输出宪法约束、易产违禁词（同 memory 里"初次判读归 data-analyst 不归 knowledge-assistant"的教训 `feedback_...knowledge_assistant`）。故**倾向 A**——把跨范式判读关进受约束的 subagent。

**判读规则是同事方法论阻塞点**：synthesizer 的 system_prompt 里"什么算跨范式相同/不同、怎么对比才科学"需行为学同事给（如：同批鼠 EPM 开放臂时间↓ 与 OFT 中心区时间↓ 是否印证同一焦虑表型？效应量怎么跨范式比？）。**工程骨架可先建，判读规则挂 issue 等同事**（性质同 #90 Golden Cases）。

### 2.6 API 层（仿 threads router）

新建 `app/gateway/routers/experiments.py`，仿 `threads.py` 的 `@require_permission` + CRUD：
- `POST /api/experiments` 建 experiment
- `GET /api/experiments` / `GET /api/experiments/{id}` 列/详情
- `POST /api/experiments/{id}/import-thread` archive 一个 thread（执行文件提取）
- `POST /api/experiments/{id}/synthesize` 触发综合（**校验 ≥2 thread，否则 400**）
- `GET /api/experiments/{id}/synthesis` 取结论
- `DELETE /api/experiments/{id}` 删 experiment（DB 级联 + 删磁盘目录，守路径 spec 的删除事务化）

权限：`authz.py` 加 `experiments:read/write/delete`，user 隔离复用 `get_effective_user_id`。

### 2.7 前端

- `core/threads/`（或新建 `core/experiments/`）加 hooks/types/api。
- thread 卡片菜单加"归入实验"（import）入口。
- 新建 experiment 视图：imported thread 列表 + "综合分析"按钮（≥2 才亮）+ 并列展示各 thread 结论 + synthesizer 跨范式结论。
- 守前端铁律：组件可改性看 `components.json`；ai-elements/MagicUI/React Bits 禁改源、只消费侧 className override（memory `feedback_shadcn_ui_copyin_editable_vs_registry_pulled_generated`）。

---

## 三、与现有架构的冲突点 / 风险

| 风险 | 说明 | 缓解 |
|---|---|---|
| **import 半成品** | thread 分析没跑完就 import，快照缺 handoff | 前置校验 `handoff_data_analyst.json` 存在，缺则 400 |
| **快照陈旧** | thread 后续又追问改了结论，experiment 快照是旧的 | 设计为"import 时点快照"语义，UI 标注 imported_at；可选"重新 import 刷新" |
| **synthesizer 产违禁词** | lead 直接对比绕过输出宪法 | 用 subagent A 方案，关进受约束 subagent |
| **跨 user import** | 两 thread 属不同 user | `@require_permission(owner_check=True)` 已覆盖；import 校验两 thread 同 user |
| **judgment 规则空缺** | 跨范式判读方法论未定 | 工程骨架先建，规则挂 issue 等同事；synthesizer 先做"结构化并列+保守描述"不做强判读 |
| **路径可靠性** | 提取依赖路径解析正确 | **依赖配套 spec** `file-path-reliability-loadbearing-convergence`——它是本 feature 地基，建议先行或同期 |

---

## 四、落地路线图（分 Phase，逐 Phase 出实施 spec）

- **Phase 0（地基）**：配套路径可靠性承重墙 spec 落地（experiment 提取文件依赖它）。**建议先行**。
- **Phase 1（MVP 骨架）**：3 张表 + ExperimentRepository + experiments router（CRUD + import-thread 文件提取）+ Paths experiment 目录方法 + 前端 import 入口 + 并列展示视图。**不含 synthesizer 强判读**——先做"结构化并列展示两 thread 的 data-analyst 结论"。
- **Phase 2（synthesizer）**：等行为学同事跨范式判读方法论 → 新增 `experiment-synthesizer` subagent（三件套）+ synthesize 端点 + 结论展示。
- **Phase 3（体验）**：对比图表、experiment 级 memory（blog 范式延伸）、report 导出、重新 import 刷新快照。

---

## 五、需同事 / 需用户决策的开放项

1. **跨范式判读方法论**（同事）：同批鼠跨范式结论怎么对比才科学？相同/不同的判据？→ **已开 [Issue #226](https://github.com/noldus-cn-beijing/noldus-insight/issues/226)**（最终落成新 skill `ethoinsight-cross-paradigm`，仿 `ethovision-paradigm-knowledge` 渐进披露），性质同 #90。
2. **synthesizer 接法 A/B 终选**（实施时定）：倾向 A（受约束 subagent）。
3. **快照语义**（用户）：import 是"冻结时点快照"还是"始终跟随 thread 最新"？init 倾向**冻结快照**（简单、可追溯、不受原 thread 删除影响）。

---

*依据：实读坐实 thread 间文件隔离（`Paths` user/thread scoped）、data-analyst 结论是结构化 `handoff_data_analyst.json`、subagent 注册/handoff registry 模式（`subagents/builtins/__init__.py` + `handoff_registry.py`）、persistence 仿 `thread_meta` 可加新表、router 仿 `threads.py`。用户拍板"import=archive、工程化提取文件不跨 thread 读、≥2 才可比、不替识别"。范式正当性参 blog session-tree/记忆思想（memory `reference_agentic_context_engineering_session_tree_v1_direction`）。判读层挂同事方法论。*
