# Spec：EV19 列语义对齐 — Sprint 1（名字对齐 / 通用列别名表）

> 状态：实施 spec，可直接交付 agent 执行
> 日期：2026-06-05
> 范围：**仅 Sprint 1 = 名字对齐（命名差异，1:1 / 多列同概念）**。把 PR #89 的「单一裸 `in_zone` → 单一目标参数」泛化成「任意多个用户自定义列 → 各自 catalog 概念」的通用列别名表 + HITL 预填反诘 + 落盘 + guardrail 拦截。**不含** Sprint 2（结构聚合语义，如 EPM 4 区加权——见 §0.2）。
> 设计来源：[`docs/design/2026-06-05-column-semantics-hitl-design-v2.md`](../../design/2026-06-05-column-semantics-hitl-design-v2.md)（D1–D16）
> 前序已实现：[`2026-06-04-zone-override-unification-three-paradigm-design.md`](2026-06-04-zone-override-unification-three-paradigm-design.md)（PR #89 / commit 25eaebfc — `anonymous_zone_is` 三范式统一）
> 铁律：`feedback_oft_single_zone_must_ask_not_guess`、`feedback_single_source_of_truth`、`feedback_skill_describing_tool_output_enables_hallucination`、`feedback_pr_merge_must_run_full_suite_on_shared_logic`、`feedback_known_full_suite_test_pollution_4_tests`、`feedback_subagent_consumption_via_first_party_tool`

---

## 0. 给实施 agent 的必读

### 0.1 一句话

EV19 raw data 的分析区列名 100% 用户自定义（中心区/边缘区/Center/zone_A…），agent 无法从名字推断身份。现有 `anonymous_zone_is` 机制（PR #89）只能处理**单一裸 `in_zone`**——真实 OFT 数据是 `中心区`（normalize→`center`）且有**多个**自定义区，现机制不触发，直接 `columns_missing` raise。Sprint 1 把单区机制**泛化成通用列别名表**：HITL 让用户把每个不认识的列对齐到 catalog 概念（预填反诘、绝不字面猜），决议落 SSOT，resolve 在 `columns` 入口单点重映射，下游零改动。

### 0.2 与现状的精确对账（执行前必看，避免重复造轮子）

**已实现（PR #89，直接复用，不要重写）**：
- `anonymous_zone_is` 统一 key + 范式无关翻译层（`resolve.py:865-871`）：`{anonymous_zone_is: X}` → 按 catalog `anonymous_zone_override` 翻译成 `center_zone`/`open_zones`/`light_zone`，含 `wrap_list`。
- OFT/zero_maze/LDB 的 catalog 已声明 `anonymous_zone_override`（实测确认）。
- `_detect_anonymous_zone`（`resolve.py:472-519`）放行/报错逻辑。

**现状两个硬限制 = Sprint 1 要补的真实缺口**：

| # | 限制 | 现状代码 | 真实数据为什么撞上 |
|---|------|---------|------------------|
| ① | **只认裸 `in_zone`** | `resolve.py:495` `if "in_zone" not in available_columns: return None` | 34 文件是 `中心区`→`center`，无 `in_zone` → 不触发 → raise |
| ② | **单范式单匿名区** | 只有一个 `anonymous_zone_is` key | OFT 数据有 中心区+边缘区+边缘区到中心区 **三个**自定义区 |

**Sprint 1 = 泛化①②**：从「单一裸 in_zone → 单一目标参数」泛化成「N 个用户列 → N 个 catalog 概念」。`anonymous_zone_is` 机制作为**特例保留**（纵深防御，D10），新通用别名表是主路径。

### 0.3 不做（Sprint 2，明确排除）

- 多列**聚合语义**（EPM open_arm1+open_arm2 → open_arm 的 OR/加权/区分臂统计）。Sprint 1 只做 1:1 命名映射；多列同概念若恰好被现成 glob+OR 兜住（如 EPM `in_zone_open_arms_*` + `df[cols].max(axis=1)`）是副产品，**但不为它设计、不测它的聚合正确性**——那是 Sprint 2 与同事确认的范围。
- 跨文件列名不一致（归 Gate 2 数据质量，D14 场景 E）。

> **Sprint 2 勘察 issue**：[#98 — 6 范式分析区「结构差异」勘察](https://github.com/noldus-cn-beijing/noldus-insight/issues/98)（行为学专家逐范式确认聚合语义，工程不在确认前自行实现聚合）。

---

## 1. 设计总览

```
③ inspect_uploaded_file【增强】
     → 返回 column_assessment（recognized/unrecognized + 每列证据）+ open_questions
        ↓
④⑤ lead 用 catalog requires_columns 反推合法概念菜单（D12）
     → ask_clarification 预填反诘（"我的理解是：中心区→center，对吗？" D15）
        ↓ Command(goto=END) 中断
        ↓
⑥ set_experiment_paradigm(column_semantics={...})【增强】
     → 写 experiment-context.json（SSOT）
     → 写盘时单向投影 column_aliases（D11）
        ↓
⑦ prep_metric_plan → resolve_metrics(columns, column_aliases=...)【核心新建】
     → _apply_aliases 在 columns 入口单点重映射：中心区/center → in_zone_center
     → 下游 _missing_columns 等消费入口全不动
        ↓
⑧ Ev19TemplateGuardrailProvider【增强】
     → column_semantics.open_questions 未清空 → 拦截 task(code-executor)
        ↓
⑨⑩ code-executor → data-analyst → report-writer
```

**新建独立 tool/provider：0 个**（全是增强现有）。**新建 thin skill：1 个**。工程重量在 tool/resolve/guardrail 代码（确定性链路），skill 只承载交互方法论。

---

## 2. 实施改动（精确到文件:行）

### 2.1 `ethoinsight/utils.py` — 列识别判定（新增函数，~50 行）

新增 `assess_column_confidence()`：对每个原始列名判 `recognized` / `unrecognized`。

```python
def assess_column_confidence(
    raw_columns: list[str],
    required_patterns: list[str],   # 来自 catalog 各 metric 的 requires_columns（去重）
) -> dict:
    """对每列判定系统是否认得它。不新建第二份列名库（D5）：
    依据 = COLUMN_MAP 命中 OR normalize 后匹配某 required_pattern。

    Returns:
        {
          "recognized":   [ {raw, normalized} ... ],
          "unrecognized": [ {raw, normalized} ... ],   # 触发 HITL
        }
    判定规则（recognized 的充分条件，满足其一即认得）：
      a. raw in COLUMN_MAP（精确命中）
      b. normalize_column_name(raw) 命中某 required_pattern（fnmatch glob）
      c. normalize 结果是 L1 固定列之一（trial_time/x_center/... 见 _FIXED_COLUMNS）
    其余 → unrecognized。
    """
```

- 新增模块级常量 `_FIXED_COLUMNS`（L1 七列 normalize 后的名字 + distance_moved/velocity 等 COLUMN_MAP 必含项）。仅供 (c) 判定，**不作为列名库**（COLUMN_MAP 仍是唯一真源）。
- **判定纯函数、无 LLM 参与**（D 触发闸门客观可算）。

### 2.2 `inspect_uploaded_file_tool.py` — 返回 column_assessment（增强，~60 行）

文件：`packages/agent/backend/packages/harness/deerflow/tools/builtins/inspect_uploaded_file_tool.py`

- 在三条格式路径（txt/xlsx 单 sheet/csv）解析出 `columns` 后，调 `assess_column_confidence`。
  - `required_patterns` 来源：**lead 在调 inspect 时已知 paradigm**（identify_ev19_template 先行）。但 inspect 当前不接收 paradigm。**两种取法，选 A**：
    - **A（推荐）**：inspect 加可选入参 `paradigm: str | None`；若提供，加载该 catalog 的 requires_columns 当 required_patterns；不提供则 required_patterns=[]，只按 (a)(c) 判 recognized（degrade，不报假 unrecognized）。
    - B：inspect 不接 paradigm，column_assessment 只按 COLUMN_MAP+L1 判——会把所有 zone 列误报 unrecognized（即使标准命名的也报）。**否决 B**：标准数据会触发不必要反问，违反 D「标准数据零额外开销」。
- 复用已有 `_compute_anonymous_zone_evidence` 的证据产出思路，**泛化到任意未知列**：对每个 unrecognized 列，若是 0/1 值，附取值占比；非 0/1 附"疑似连续/距离值"标记。
- 返回值增：
  ```
  column_assessment: { recognized:[...], unrecognized:[{raw,normalized,evidence}...] }
  open_questions: [unrecognized 的 raw 列名 ...]   # 空 = 标准数据，不触发 HITL
  ```
- **反脑补铁律**（`feedback_skill_describing_tool_output_enables_hallucination`）：这些字段结构**只在工具 docstring 里有最小骨架**，SKILL.md **绝不**描述返回结构（见 §2.7）。

### 2.3 `ethoinsight/catalog/resolve.py` — 通用列别名重映射（核心新建，~45 行）

- `resolve_metrics()`（`resolve.py:116`）签名加参数：
  ```python
  column_aliases: dict[str, str] | None = None,   # {原始列 or normalized: catalog 概念}
  ```
- 在 `load_catalog` 之后、第一次 `_missing_columns` 之前（约 `resolve.py:164` 后）插入单点重映射：
  ```python
  if column_aliases:
      columns = _apply_aliases(columns, column_aliases)
  ```
- 新增 `_apply_aliases(columns, aliases)`：
  ```python
  def _apply_aliases(columns: list[str], aliases: dict[str, str]) -> list[str]:
      """把用户列重映射成 catalog 期望名。
      aliases value 为 None/"__ignore__" 的列 → 从 columns 移除（D4 用户确认忽略）。
      返回的列表喂给所有下游 _missing_columns —— 下游消费入口 0 改动。
      """
  ```
- **关键约束**：只动 `columns` 这一个入口（`resolve.py:118` 的参数，被 193/249/381/410 共享）。`_missing_columns`、`_detect_anonymous_zone`、`_compute_parameters_in_use` **全部不改**（承重墙消费入口不动）。
- `anonymous_zone_is` 机制保留不动（特例兜底）。

### 2.4 `experiment_context.py` — column_semantics 落盘 + 投影（增强，~50 行）

文件：`packages/agent/backend/packages/harness/deerflow/agents/middlewares/experiment_context.py`

- `set_experiment_paradigm_tool`（`:196`）加入参 `column_semantics: dict | None = None`。
- 写盘（`:321-331` 的 `data` dict）增字段 `column_semantics`（schema 见 §3）。
- **写盘时单向投影 column_aliases**（D8/D11）：从 column_semantics.columns 的 `resolves_to` 生成
  `{normalized_or_raw: resolves_to}`，写入一个新文件或 experiment-context.json 的 `column_aliases` 字段。
  prep_metric_plan 读它传给 `resolve_metrics(column_aliases=...)`。
  - 投影是**确定性纯函数、写盘时一次性算**（不在 resolve 时现算——保 analysis_config_id 输入时序确定，`feedback_parameters_used_must_reflect_actual_resolution_path`）。
- 照 `acknowledge_quality` 模式（`:262-281`）做"纯字段追加、不碰已有 paradigm 字段"的模板。

### 2.5 `ev19_template_provider.py` — guardrail 子检查（增强，~25 行）

文件：`packages/agent/backend/packages/harness/deerflow/guardrails/ev19_template_provider.py`

- 在 `task(code-executor)` 检查（现有 ev19_template 检查，`:128-151`）旁加子检查：
  读 experiment-context.json，若 `column_semantics.open_questions`（或等价：存在 `confirmed=false` 的列）**非空** → 拦截，reason 列出待对齐的列名。
- **仅 open_questions 非空时拦截**（标准数据 open_questions=[] 不触发，D「零额外开销」）。
- 与现有 ev19_template 检查正交叠加。

### 2.6 `lead_agent/prompt.py` — 触发 + 话术范例（增强，~15 行）

文件：`packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`

- 「反问合并规则」段（`:461-489`）加一条 bullet：
  ```
  - 如果 inspect 返回 column_assessment.open_questions 非空（有自定义分析区列未对齐）：
    用 catalog 合法概念菜单 + 各列证据，预填你的最佳理解让用户一键确认（见 skill）
  ```
- 加 skill 指针：`如需列对齐方法论，read skills/custom/ethoinsight-column-confirmation/SKILL.md`。
- **正面指令**（deepseek 不用"禁止"，`feedback` 第 6 条）：描述"预填理解+请确认"的想要行为，不写"不要猜"。

### 2.7 `skills/custom/ethoinsight-column-confirmation/` — thin skill（新建，~140 行）

> **三件套**（lead-only，subagent 侧不动 — 列对齐全在 lead）：① 文件 ② extensions_config 注册（§2.8）③ lead prompt read 指引（§2.6）。

```
ethoinsight-column-confirmation/
├── SKILL.md                       # ~60 行：触发条件 + 跳过条件 + 方法论概述 + 落盘指引
└── references/
    ├── presentation-template.md    # 预填反诘正例（§4.2 话术）+ 反例 3 个
    └── answer-mapping.md           # 用户自然语言 → column_semantics 参数映射表
```

**SKILL.md 存什么**（操作手册）：触发条件（open_questions 非空）、跳过条件（空→标准数据零开销）、预填反诘话术、答案映射、多列场景。
**绝不存**：column_assessment 字段结构（反脑补铁律）、列分类规则（在 utils.py）、范式知识（在 catalog/paradigm-knowledge skill）。

### 2.8 `extensions_config.json` — 注册 skill（3 行）

```json
"ethoinsight-column-confirmation": { "enabled": true }
```

---

## 3. SSOT schema：experiment-context.json `column_semantics`

```json
{
  "paradigm": "oft",
  "ev19_template": "OpenFieldRectangle-AllZones",
  "column_semantics": {
    "confirmed_at": "<ISO8601>",
    "columns": {
      "中心区":        { "raw_name": "中心区", "normalized": "center",       "resolves_to": "in_zone_center", "meaning_zh": "中心分析区", "confirmed": true },
      "边缘区":        { "raw_name": "边缘区", "normalized": "边缘区",       "resolves_to": "in_zone_border", "meaning_zh": "边缘分析区", "confirmed": true },
      "边缘区到中心区": { "raw_name": "边缘区到中心区", "normalized": "边缘区到center", "resolves_to": null, "ignore": true, "confirmed": true }
    }
  },
  "column_aliases": { "center": "in_zone_center", "边缘区": "in_zone_border" }
}
```

- `resolves_to` = 喂 resolve 的机器决议（投影成 column_aliases）。Sprint 1 取单值（1:1）。
- `meaning_zh` = 喂 report-writer 的叙述语义（D1 语义落 SSOT）。
- `ignore:true` 列也必须 `confirmed:true`（D4 确认无关）；不进 column_aliases，resolve 时移除该列。
- `column_aliases` = column_semantics 的派生产物（写盘时单向投影），prep_metric_plan 读它传 resolve。
- **Sprint 2 前向兼容**：`resolves_to` 将来可为列表/多列同概念，schema 不必改（§6.2 of design）。

---

## 4. HITL 话术（D13 反诘 + D15 预填，照搬 design §4）

### 4.1 场景 B 主反问（预填 + 可否决）

```
⚠️ 开始分析前需确认：
1. EV19 模板：OpenFieldRectangle / OpenFieldCircle
2. 你的数据有 3 列是自定义分析区。基于各列取值分布，我的初步理解是：
   ┌──────────────┬────────────────┬──────────────────┐
   │ 你的列名      │ 取值=1 时间占比 │ 我的理解（请确认） │
   ├──────────────┼────────────────┼──────────────────┤
   │ 中心区        │ 18%            │ → 中心区 center   │
   │ 边缘区        │ 82%            │ → 边缘区 border   │
   │ 边缘区到中心区 │ （非 0/1）      │ → 疑似距离列，忽略 │
   └──────────────┴────────────────┴──────────────────┘
   OFT 分析的区类型有：中心区 / 边缘区 / 角落 / 其它(忽略)。
   以上理解对吗？如有错误请告诉我正确对应。分组已识别为 KK/MM/MN/SS/TT，无需填。
```
约束：预填列标"请确认"；证据与预填并列让用户据证据核对；给否决口子；预填来自证据+catalog 菜单**不来自字面**。

### 4.2 场景 C 二次反诘（用户否决/矛盾）

```
确认一下：你把「边缘区」标为 中心区(center)。这列取值=1 占 82%。
（中心区通常是动物较少停留的区，仅供核对。）这是本意吗？回复"是"或更正。
```

---

## 5. 测试（~220 行，TDD 强制）

文件：`packages/ethoinsight/tests/` + `packages/agent/backend/tests/`

| 测试 | 内容 | 红锚点 |
|------|------|--------|
| `test_assess_column_confidence` | 34 文件真实列 → 前 7 列+移动距离/速度/Result1 = recognized；中心区/边缘区/边缘区到中心区 = unrecognized | — |
| `test_apply_aliases` | `{"center":"in_zone_center"}` 把 columns 里的 center 重映射；ignore 列被移除；下游 _missing_columns 匹配上 | — |
| `test_resolve_with_column_aliases` | **34 文件 fixture**：无 aliases → raise `columns_missing`（现状）；有 aliases → 算出 center_time_ratio 等 5 metric | xfail→pass 锚点 |
| `test_inspect_returns_column_assessment` | inspect(paradigm=oft) 返回 open_questions=[中心区,边缘区,边缘区到中心区]；标准命名数据 open_questions=[] | — |
| `test_guardrail_blocks_open_questions` | open_questions 非空 → task(code-executor) 被拦；清空 → 放行；ainvoke 等价（`feedback_langchain_tool_args_schema` 教训：异步必测） | — |
| `test_column_semantics_projection` | set_experiment_paradigm(column_semantics) → 写盘含 column_aliases 派生正确 | — |

**合并前必跑全量**（`feedback_pr_merge_must_run_full_suite_on_shared_logic`）：改了 `resolve.py`（6 范式 + 测试套共享承重墙）+ guardrail，**必须 `cd packages/ethoinsight && pytest` 全量 + `cd packages/agent/backend && make test` 全量**，grep 所有 `resolve_metrics(` / `set_experiment_paradigm` 调用方 fixture。已知 4 个污染测试（`feedback_known_full_suite_test_pollution_4_tests`）不算回归。

---

## 6. 验收标准

1. 真实 34 文件 OFT 数据：上传 → 一次合并反问（模板+3 列预填+分组已识别）→ 用户一句话确认 → 别名重映射 → 算出 center metric → 出报告。**端到端跑通**。
2. 标准命名数据（zone 列叫 `in_zone_center_*`）：open_questions=[] → **不触发列对齐反问**，流程同今天（零额外开销）。
3. 用户答忽略的列：`ignore:true` + resolve 时移除，不参与计算，不报错。
4. `anonymous_zone_is` 旧机制（裸 in_zone 单区）：回归不破（特例兜底仍工作）。
5. 全量测试绿（除已知 4 污染）。

---

## 7. 守住的铁律自检

- **不字面猜**（D2）：预填来自证据+catalog 菜单，不来自列名字面（"中心区"预填 center 因占时 18% 符合 center 典型，非因名字）。预填是可否决反诘（D15），不直接采纳。
- **SSOT 不双存**（D5/D12）：列名库复用 COLUMN_MAP+catalog；base fact 从 catalog requires 反推；column_semantics 是决议唯一落点，column_aliases 由它派生。
- **反脑补**：column_assessment 结构不进 SKILL.md。
- **承重墙不动**：别名重映射在 columns 入口单点，resolve 下游消费入口 0 改。
- **复用优先**：泛化 PR #89 单区机制，非从零；不新建独立 tool/provider。
- **不越界 Sprint 2**：只做 1:1 命名映射，不做聚合语义。
