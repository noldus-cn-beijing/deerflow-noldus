# EV19 模板识别地基设计 — 渐进披露 skill + deerflow 现成能力复用

**日期**: 2026-05-08
**状态**: 设计稿（等用户 review 后转 implementation plan）
**作者**: 通过 brainstorming skill 与曲若衡共同推进

---

## 1. 背景与问题

### 1.1 PRD v2 的 6 范式 MVP 真实工程含义

PRD v2 (`docs/EthoInsight-PRD/EthoInsight-PRD-v2-DRAFT.md`) 把 MVP 收敛到 6 个范式（4 焦虑：EPM / OF / Zero Maze / LDB；2 抑郁：TST / FST）。但**实施 PRD 不是简单地写 6 个独立的 `templates/*.py`**——产品方向上每个 PRD 范式底下对应多个 EthoVision XT 19 真实模板变体，分析逻辑因变体而异（含 zone vs 不含 zone、不同动物种类、不同阵列规模）。

> 引用用户原话：「关键不是 shoaling，关键是实验模板，实验模板是大类，底下还有很多小类。」「shoaling 只是一个给领导展示的玩具」。

因此 v0.1 真正的工程地基 = **EV19 模板识别 + 渐进披露领域知识**，6 范式只是这个地基上长出来的第一批应用。

### 1.2 EthoVision 19 真实模板体系（20 大类 / 62 变体）

`demodata/ev19 templates/` 目录下是 EV19 出厂模板，每个目录对应一个变体，命名形如 `<大类>-<变体后缀>`：

| 大类 | 变体数 | 例 |
|---|---|---|
| OpenFieldCircle | 5 | -AllZones / -NovObjZones / -NoZones-Fish 等 |
| OpenFieldRectangle | 7 | -AllZones / -Subdivided3x3 等 |
| PhenoTyper | 15 | (单/Quad/16x) × (5 种 zone 配置) |
| PlusMaze | 3 | -AllZones / -FewZones / -NoZones |
| ZeroMaze | 2 | -AllZones / -NoZones |
| PorsoltCylinder | 2 | -AllZones / -NoZones |
| ...（其余 14 大类）| ... | ... |

**机器字段**已通过 `scripts/build_ev19_template_index.py` 抽取到 `docs/review-packages/2026-04-29-ev19-templates/_facts.json`（62 行 JSON），含每个变体的 `arena_template`、`zone_template`、`subject_hint`、`array_size`。**领域知识字段**等行为学同事（曲若衡）补 markdown。

### 1.3 当前 lead agent 的 18 范式分类与 EV19 不对应

`packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` 的「7 大类 18 范式」分类是项目早期自定义的扁平表，与 EV19 的 20 大类 62 变体无对应关系。这导致 Gate 1 反问机制不准（已在 [docs/plans/2026-04-29-ev19-template-paradigm-design.md](../../plans/2026-04-29-ev19-template-paradigm-design.md) 中识别）。

### 1.4 数据飞轮未启动 → 行为学知识必须可增量更新

微调 SFT 计划（Phase 1）尚未启动，6+ 个范式的领域知识不能编码进模型权重。当前可用机制：
- **prompt 注入**（静态、token 贵）
- **skill 渐进披露**（agent 按需 read_file）
- **MCP 知识库**（noldus-kb 当前禁用）

skill 渐进披露是当前阶段最匹配的载体，因为：行为学同事的 review PR 会持续到来，每次 PR 直接修 skill 下的 markdown 文件，agent 立刻能用。

### 1.5 已经验证的事实：raw txt meta 不包含原始模板 ID

通过实际读取 `golden-cases/case-001-shoaling-baseline/raw-data/` 下的 raw txt 文件验证：
- `观察区设定` 字段存的是用户起的名字（"Arena Settings 1"），**不是** `m_strArenaTemplate`
- `实验` 字段是用户起的实验名（"Shoaling behavior with JavaScript XT180"），不可靠
- 没有任何 meta 字段直接告诉你这是哪个 EV19 变体

**结论**：模板识别不能靠 meta 字段精确字典匹配。必须依靠 LLM 综合用户文字 + 文件名 + 必要时读 raw 列结构 + ask_clarification 反问。

---

## 2. 设计目标

| # | 目标 |
|---|---|
| G1 | Lead agent 通过对话识别 EV19 模板（大类 → 变体 → 实验范式），写入 `experiment-context.json` |
| G2 | 充分利用 deerflow 现成能力（skill 渐进披露 / GuardrailMiddleware / LoopDetectionMiddleware / ask_clarification），不重复造轮子 |
| G3 | 行为学知识与代码解耦——同事 review PR 直接修 skill markdown，agent 立刻生效 |
| G4 | 反问要精准、不烦扰用户（最多 1 次模板相关反问，否则用默认值降级） |
| G5 | 任何单点失败不导致流程崩溃（分层约束，详见 §6 鲁棒性分层） |

---

## 3. 关键架构决策

### 3.1 新建 `ethovision-paradigm-knowledge` skill

**目录结构**：

```
packages/agent/skills/custom/ethovision-paradigm-knowledge/
├── SKILL.md                                # 入口 — 决策树 + 大类索引
└── references/
    ├── _facts.md                           # 62 变体事实表（人类可读，从 _facts.json 转）
    ├── identification-decision-tree.md     # agent 用对话识别模板的决策流程
    ├── default-template-fallback.md        # 范式 → 默认变体降级表
    ├── by-template/                        # 大类知识（同事 review 包搬过来）
    │   ├── PlusMaze.md
    │   ├── OpenFieldCircle.md
    │   ├── OpenFieldRectangle.md
    │   ├── ZeroMaze.md
    │   ├── PorsoltCylinder.md
    │   └── ... (20 大类)
    └── by-experiment/                      # 实验范式知识（同事 review 包搬过来）
        ├── epm.md
        ├── open_field.md
        ├── zero_maze.md
        ├── light_dark_box.md
        ├── tail_suspension.md
        ├── forced_swim.md
        └── ... (20 实验)
```

**理由**：
- 渐进披露天然匹配 62 变体规模 — 入口 SKILL.md 小，按需 read_file 加载具体大类/实验
- 行为学知识 markdown 是同事 review PR 的产物 — 直接搬进 skill，无中间转换层
- agent 对识别结果有疑问时，可以读多个 by-template 和 by-experiment 文件交叉验证

**Alternatives 对比**：

| 方案 | 优点 | 缺点 | 决策 |
|---|---|---|---|
| 全量 prompt 注入 | 一次性加载 | 62 变体 token 太贵，更新需重启 | ❌ |
| MCP 服务（noldus-kb） | 检索更智能 | 当前禁用，启动周期长 | ❌（v0.1 后启用） |
| 渐进披露 skill | token 高效，agent 主导，与同事 PR 流程对齐 | LLM 需要多轮 read_file | ✅ |
| Python 模块（事实表）+ skill markdown 混合 | 工具校验用 Python，agent 阅读用 markdown | 双源维护成本 | ✅（事实表自动从 markdown 生成） |

### 3.2 复用 deerflow 现成功能 vs 自造轮子

| 需求 | 自造方案（弃） | deerflow 现成方案（采用） |
|---|---|---|
| 拦截 ev19_template=null 的 code-executor 派遣 | 改造 `GateEnforcementMiddleware` 加自定义检查逻辑 | `GuardrailMiddleware` + 自定义 `Ev19TemplateGuardrailProvider`（实现 `GuardrailProvider` 协议） |
| 防止 agent 反复反问模板 | thread state 加 `ev19_clarification_count` + 自写计数中间件 | `LoopDetectionMiddleware`（已现成）— 同 hash tool call ≥3 次警告，≥5 次 strip tool_calls |
| 渐进披露领域知识 | 自写 prompt 注入逻辑 | skill 系统 + agent 主动 `read_file` |
| 反问 UI 中断 | 自写中断流程 | `ask_clarification` 工具 + `ClarificationMiddleware`（`Command(goto=END)`）|

每一处复用都减少了维护面，并且 deerflow 上游修复会自动获益。

**关于现有 `GateEnforcementMiddleware` 的处理**：

CLAUDE.md 第 10 条提到当前 `GateEnforcementMiddleware` 在中间件链里。本 spec 的处理：
- **不删除** `GateEnforcementMiddleware`（它处理的是 `paradigm` 字段的 Gate 1 拦截，是 noldus 自加中间件）
- **新增** `GuardrailMiddleware` + `Ev19TemplateGuardrailProvider`，专门处理 `ev19_template` 字段拦截
- 两者职责正交：`GateEnforcement` 管 paradigm 完整性，`GuardrailMiddleware` 管 ev19_template 完整性 + 锁定语义
- **未来工作**（v0.1 后）：如果维护负担过重，再评估把 `GateEnforcement` 的逻辑也迁到 `GuardrailMiddleware` 自定义 provider，统一收口

### 3.3 review 包 markdown 直接合并到 skill

`docs/review-packages/2026-04-29-ev19-templates/by-template/*.md` 和 `by-experiment/*.md` 直接复制（git mv）到 skill `references/by-template/` 和 `references/by-experiment/` 下。同事下次 review PR 直接改 skill 里的 markdown，**不要做中间转换层**。

### 3.4 `set_experiment_paradigm` 工具签名升级

工具新增必填参数 `ev19_template`（变体级别字符串），加白名单校验：

```python
def set_experiment_paradigm(
    paradigm: str,                    # 学术范式 key（"epm", "open_field", ...）
    ev19_template: str,               # 必填，例 "PlusMaze-AllZones"
    confidence: Literal["high", "medium", "low"] = "medium",
    ...
):
    # 1. 白名单校验（来自 ev19_facts.py 的 EV19_VARIANTS 集合）
    if ev19_template not in EV19_VARIANTS:
        return {"error": f"未知 EV19 模板 '{ev19_template}'", "candidates": _nearby_matches(ev19_template)}
    # 2. paradigm × ev19_template 兼容性提示（来自 by-experiment 的事实）
    if paradigm not in EV19_TEMPLATE_PARADIGM_MAP.get(ev19_template, []):
        # 不阻塞，但记录 warning，让 agent 在 thinking 中考虑是否要 ask_clarification
        warning = f"模板 '{ev19_template}' 通常不用于范式 '{paradigm}'，请确认或反问用户"
    # 3. 写入 experiment-context.json
    ...
```

---

## 4. Agent 交互流程

### 4.1 整体流程（保持现有架构 + 新增 EV19 识别 step）

```
用户上传 + 提问（"帮我分析高架十字迷宫的数据"）
    ↓
Lead Agent
    ├─ 读 system prompt 看到 ethovision-paradigm-knowledge skill
    ├─ ethoinsight-planning skill 决定路径（端到端分析 vs 知识问答 vs 追问）
    │
    ├─ 【新】Gate 1 识别 EV19 模板：
    │   ├─ read_file SKILL.md → 看决策树
    │   ├─ 综合用户文字 + 文件名 推测候选大类
    │   ├─ 必要时 read_file by-experiment/<实验>.md 查候选模板
    │   ├─ 必要时 read_file by-template/<大类>.md 查变体差异
    │   ├─ 必要时 read_file（用户上传的第一个 raw txt 前 50 行）查 meta + 列结构
    │   ├─ 决策：
    │   │   ├─ 候选 = 1 高置信度 → 直接 set_experiment_paradigm
    │   │   ├─ 候选 2-3 → ask_clarification（结构化选项 + 推荐 + 默认值兜底）
    │   │   └─ 候选 0 或 ≥4 → ask_clarification（先问大实验类型）
    │
    ├─ 调 set_experiment_paradigm(paradigm, ev19_template, ...)
    │   └─ 工具内：白名单校验 → 写入 experiment-context.json（含 ev19_template 字段）
    │
    └─ 派遣 task("code-executor") → ...
            ↓
       handoff_planning.json / handoff_code_executor.json / handoff_data_analyst.json / handoff_report_writer.json
       ← subagent 之间靠 handoff JSON 文件传 hard fact（保持现有约定）
```

### 4.2 反问质量准则

写在 `references/identification-decision-tree.md`：

- **反问前必须读 raw meta** — 至少 read_file 第一个上传文件的前 50 行，看单位（毫米=鱼 / 厘米=啮齿）、追踪点（单点/三点）、zone 列结构
- **反问必须给 ≤3 个结构化选项** — 不要开放性问题
- **每个选项要标注差异** — "AllZones（含开闭臂 + 头探出区，最常见）"
- **推荐项放第一位** — agent 主动给推荐，不让用户从零选择
- **默认值兜底** — "如果不确定，选 A，绝大多数 EPM 实验都用这个"

### 4.3 默认值降级表

写在 `references/default-template-fallback.md`，agent 在反问被 LoopDetection 阻断或用户答 "不知道" 时按表选：

| 范式 | 默认 ev19_template | 原因 |
|---|---|---|
| epm | PlusMaze-AllZones | 90%+ EPM 实验用 AllZones |
| open_field | OpenFieldRectangle-AllZones | 大多数 OFT，圆形看 demodata 形状判断 |
| zero_maze | ZeroMaze-AllZones | 同上 |
| light_dark_box | （待行为学同事确认 EV19 中的对应大类）— v0.1 实施时如同事仍未确认，先用 `OpenFieldRectangle-Subdivided2x2` 兜底（明暗箱常用矩形 arena + zone 划分明暗区） | EV19 表里没有独立 LDB 模板大类 |
| tail_suspension | NoTemplate | TST 不用 zone，仅活动度 |
| forced_swim | PorsoltCylinder-AllZones | 默认 |

> 此表 v0.1 内不完美，行为学同事 review PR 会持续修正。

---

## 5. 实施范围

### 5.1 今天可做（不依赖同事 review PR）— D1-D10

| # | 内容 | 文件 |
|---|---|---|
| **D1** | 新建 skill 骨架 | `packages/agent/skills/custom/ethovision-paradigm-knowledge/{SKILL.md,references/}` |
| **D2** | review 包 markdown 复制（git mv）到 skill `references/by-template/` 和 `references/by-experiment/` | `references/by-template/*.md`、`references/by-experiment/*.md` |
| **D3** | `_facts.json` 转 `references/_facts.md` + Python 模块 `ev19_facts.py`（工具白名单 + paradigm 兼容性映射） | `references/_facts.md`、`packages/ethoinsight/ethoinsight/ev19_facts.py` |
| **D4** | SKILL.md 写好 — 何时使用 + 决策树概要 + 何时 read by-template/by-experiment + 何时 ask_clarification | `SKILL.md` |
| **D5** | 在 `extensions_config.json` 启用 skill | `packages/agent/extensions_config.json` |
| **D6** | Lead agent prompt 删除 "7 大类 18 范式" 段，保留极简一行引导新 skill | `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` |
| **D7** | `experiment-context.json` schema 加 `ev19_template` 字段（默认 null） | `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/experiment_context.py`（如无则新建定义点） |
| **D8a** | `set_experiment_paradigm` 工具加 `ev19_template` 必填 + 白名单校验 + paradigm 兼容性 warning | 工具定义文件 |
| **D8b** | 5 个 ethoinsight 分析步骤的入口加 "软门"：`parse_trajectories` / `compute_metrics` / `run_statistics` / `generate_charts` / `assess_and_handoff`（这些在 `ethoinsight-code` skill 里通过 sandbox bash 由 code-executor 调用，不是 deerflow tool），发现 ev19_template=null 时立即返回结构化错误并写 handoff 给 agent | 实际位置：`packages/ethoinsight/ethoinsight/templates/<paradigm>.py` 入口 + `ethoinsight-code` skill reference 教 code-executor 如何处理软门错误 |
| **D8c** | 实现 `Ev19TemplateGuardrailProvider`（继承 `GuardrailProvider` 协议），拒绝 ev19_template=null 时的 `task("code-executor")` 派遣；启用 `guardrails.enabled=true` 配置 | 新建 `packages/agent/backend/packages/harness/deerflow/guardrails/ethoinsight_provider.py` 或类似位置；`config.yaml` |
| **D8d** | 验证 `LoopDetectionMiddleware` 已启用；如 ask_clarification 的 hash 区分度不够，扩展 `_stable_tool_key` 给 ask_clarification 加按"涉及概念关键词"分组 | `packages/agent/backend/packages/harness/deerflow/agents/middlewares/loop_detection_middleware.py`（如需扩展） |
| **D8e** | `_facts.md` + `default-template-fallback.md` 写明范式→默认变体映射 | `references/default-template-fallback.md` |
| **D8f** | ev19_template 一旦设置就锁定 — 在 `Ev19TemplateGuardrailProvider` 内拒绝 `set_experiment_paradigm` 的二次修改（除非 confirmation 字段） | 同 D8c |
| **D9** | `references/identification-decision-tree.md` 写好 — agent 决策流程 + 反问质量准则 | `references/identification-decision-tree.md` |
| **D10** | 端到端测试 — 上传 EPM 数据，验证 lead agent 能 0-1 次反问后写入 ev19_template 进入分析（即使 6 范式分析模板尚未补完，至少能进入 code-executor 后被软门拦住给出 remediation 而不是崩溃） | 手工 e2e + `tests/test_ev19_identification_flow.py` |

### 5.2 明天起（依赖同事 review PR）

| # | 内容 |
|---|---|
| E1 | 同事 review PR 进来后，直接更新 skill `references/by-template/*.md` 和 `references/by-experiment/*.md` |
| E2 | 6 个 PRD 范式的分析模板补全 — `templates/epm.py` / `templates/open_field.py` / `templates/zero_maze.py` / `templates/light_dark_box.py` / `templates/tail_suspension.py` / `templates/forced_swim.py` |
| E3 | shoaling golden-case 校验（一次性，证明 v0.1 demo 不翻车） |
| E4 | 抽象 `templates/_base.py` 基类（参考 shoaling 重构） |

---

## 6. 鲁棒性分层

约束分散在多处，单点失败不导致整体崩溃：

| 层 | 机制 | 失效时下游能否兜住 |
|---|---|---|
| **L1 prompt 引导** | SKILL.md 决策树 + identification-decision-tree.md | 失效 → L2 接 |
| **L2 工具签名** | `set_experiment_paradigm` 加 `ev19_template` 必填 + 白名单 | 失效（agent 跳过工具）→ L4 接 |
| **L3 上下文软门** | 5 个 ethoinsight tool 开头检查 ev19_template 非 null | 失效（工具内未加门）→ L4 接 |
| **L4 GuardrailMiddleware** | 拒绝 ev19_template=null 时的 task("code-executor") | 失效（guardrails 未启用）→ L1+L2 兜底 |
| **L5 LoopDetectionMiddleware** | 防止反复反问死循环（≥3 次警告，≥5 次 strip） | deerflow 默认启用 |
| **L6 默认值降级表** | 反问失败时自动选最常见变体 | prompt 中明确 — 反问 = 0 次时直接走默认 |

---

## 7. 不在范围内（明确排除）

| 排除项 | 原因 |
|---|---|
| 6 范式分析模板的指标/统计/图表/解读实现 | 依赖同事 review，明天起做（E2） |
| shoaling 优化（补变体、改逻辑） | 玩具，"不再投入工程"已和用户确认 |
| Electron 打包 | "你先别管"已和用户确认 |
| 中途修正模板 | v0.1 用户搞错请开新 thread；v0.1 后用 update_agent 学习偏好 |
| 手动模式（用户选指标/检验） | PRD 加的，v0.1 不做 |
| Result 数据兼容 | 用户确认仅 Raw |
| 时间表（PRD 里的 5/10 公测等） | "事件相关的你都别管" |

---

## 8. 已识别风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| 同事 review 进度未知 | by-template / by-experiment 文件可能长期空白 | skill 内置 `_facts.md`（机器字段）+ default-template-fallback.md（默认值表）作为基线，agent 即使没有 by-template 详细差异也能选默认值进入分析 |
| Lead agent 跳过 skill 直接判断 | ev19_template 可能填错 | L2 工具签名白名单 + L4 GuardrailMiddleware 双层防御；填错的 thread 在 GuardrailProvider 给出错误 ToolMessage 后 agent 会自然回退到反问 |
| Raw txt meta 不含模板 ID | 识别只能靠对话 + 列结构 + 文件名 | 已通过 §1.5 验证；§4 的反问质量准则要求 agent 反问前必读 raw meta 头，弥补信息不足 |
| 用户搞错模板（中途发现） | 整个分析报告作废 | v0.1 建议开新 thread；v0.1 后启用 update_agent 工具让 agent 学习用户偏好 |
| LoopDetection 的 hash 对 ask_clarification 区分度不够 | 多次模板反问（args 文本不同）可能 hash 不重合，无法被 LoopDetection 拦截 | D8d 内含验证步骤；如发现不重合则扩展 `_stable_tool_key` 给 ask_clarification 加"涉及关键概念"分组 |
| Skill 文件 markdown 损坏（同事 PR 笔误） | agent read_file 后无法解析 | skill 系统的 `_validate_skill_frontmatter` 已做 SKILL.md 校验；references/ 下文件损坏只影响 agent 读到的内容，不会让流程崩 |

---

## 9. 未来工作（v0.1 之后，仅记录）

| 方向 | 说明 |
|---|---|
| **`update_agent` 工具启用** | 让 lead agent 在用户多次确认某模板偏好后自改 SOUL.md，下次默认那个变体。**优先级：v0.1 后第一批考虑**（用户已认可这个方向） |
| **Skill Evolution** | agent 自己沉淀新模板知识到 skills/custom（`SkillEvolutionConfig.enabled=true`）。等 review 包稳定 + 数据飞轮启动后启用 |
| **Feedback API 接前端** | 复用 deerflow 的 `/api/threads/{id}/runs/{rid}/feedback` 替代手写飞轮反馈通道（用户已认可"feedback api 比我们自己重新造轮子好"） |
| **6 范式分析模板补全** | E1-E4 |
| **shoaling golden-case 校验** | E3 |

---

## 10. 验收标准

| # | 标准 | 验证方式 |
|---|---|---|
| 1 | lead agent 能对每个 PRD 6 范式的 raw 数据：识别用户文字 → 读 skill → 0-1 次反问 → set_experiment_paradigm → 进入分析 | 手工 e2e |
| 2 | GuardrailMiddleware 在 ev19_template=null 时拒绝 `task("code-executor")` 派遣，agent 收到结构化错误 | unit test + e2e |
| 3 | LoopDetectionMiddleware 在 ≥3 次相同 ask_clarification 时注入警告 | unit test |
| 4 | 旧 "7 大类 18 范式" 段从 lead_agent/prompt.py 删除，agent 不再引用 | grep + e2e 看 agent 输出无引用 |
| 5 | 行为学同事 PR 进来后，更新 skill references markdown 即立即生效（无需改代码） | 模拟修改一份 markdown，重启 agent 验证 |
| 6 | `make test` 全绿 | CI |
| 7 | 一次完整 e2e：用户上传 EPM 数据 → agent 走完识别 → 进入分析（即使 6 范式分析模板尚未补完，至少能被软门拦住给出 remediation 而不是崩溃） | 手工 e2e |

---

## 11. 实施完成定义

- D1-D10 全部完成并 commit（中文 commit message，按 noldus 规范）
- `make test` 在 backend 和 ethoinsight 两侧全绿
- 一次完整 e2e（验收标准 #1 + #7）
- 写入交接文档 `docs/handoffs/2026-05-08-ev19-template-skill-foundation-handoff.md`，标注剩余 E1-E4 待办

---

## 12. 关联文档

- 上游设计：[docs/plans/2026-04-29-ev19-template-paradigm-design.md](../../plans/2026-04-29-ev19-template-paradigm-design.md)
- Review 包入口：[docs/review-packages/2026-04-29-ev19-templates/README.md](../../review-packages/2026-04-29-ev19-templates/README.md)
- PRD v2：[docs/EthoInsight-PRD/EthoInsight-PRD-v2-DRAFT.md](../../EthoInsight-PRD/EthoInsight-PRD-v2-DRAFT.md)
- MVP 验收清单：[docs/EthoInsight-PRD/EthoInsight-MVP-验收清单.md](../../EthoInsight-PRD/EthoInsight-MVP-验收清单.md)
- Raw 数据格式：[docs/EthoVision输出与文档/EthoVision_XT_Raw_Data_格式说明.md](../../EthoVision输出与文档/EthoVision_XT_Raw_Data_格式说明.md)
- 路线图 v0.1：[docs/roadmap.md](../../roadmap.md)
- 上次会话交接：[docs/handoffs/2026-05-08-multi-user-path-csrf-and-resources-fix-handoff.md](../../handoffs/2026-05-08-multi-user-path-csrf-and-resources-fix-handoff.md)

*文档结束*
