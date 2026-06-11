# EV19 列语义对齐 (Column Semantics Alignment) — 设计 v2

> 本文档是 [`2026-06-05-column-semantics-hitl-design-discussion.md`](./2026-06-05-column-semantics-hitl-design-discussion.md) 的续写与重定向。
> v1 (discussion) 把问题误置为"列识别 + zone 确认"，并基于若干未经实测的假设。
> v2 用真实数据 (`DemoData/real_data/Raw data-OFT-Xuhui-34`) 重新锚定了问题本质，
> 并把所有设计张力收敛到一组已与用户逐条确认的决策 (D1–D14)。
>
> **v2 是可执行设计的基准；v1 仅作讨论历史保留。**

---

## 0. 一句话

用户在 EthoVision XT「数据选择配置-结果」里勾选「按分析区输出」时产生的分析区列，
**列名 100% 由用户自定义**，agent 无法从名字推断身份。当系统遇到这类不认识的列时，
必须进入 Human-in-the-Loop，**用客观证据让用户把自己的列对齐到范式概念**（绝不字面猜），
对齐结果落进 SSOT，再由 resolve 层通过「通用列别名表」把用户列重映射成 catalog 期望名，
metric 才能计算。

---

## 1. 问题本质（被真实数据重新锚定）

### 1.1 触发事件

用户上传 34 个真实 OFT XLSX（`Raw data-OFT-Xuhui-34`，英文版 EV19 导出）。
v1 推断根因是 `detect_ethovision` 不认英文 header marker —— **该 marker 已修**
（`parse/_core.py:57,95` 双语并存）。真实根因在更深一层。

### 1.2 真实根因（实测钉死）

对 34 个文件跑项目自己的 `normalize_column_name`，列结构 34 文件完全一致：

| 原始列名 | normalize 结果 | COLUMN_MAP? | catalog 怎么看 |
|---------|---------------|-------------|---------------|
| `Trial time` … `Elongation`（前 7 列） | ✅ 正确 | — | **L1 固定列**，EV19 骨架 |
| `移动距离` | `distance_moved` | ✅ 精确命中 | catalog 要的，匹配 |
| `速度` | `velocity` | ✅ 精确命中 | 匹配 |
| `Result 1` | `result_1` | ✅ 精确命中 | 无 metric 依赖，无害 |
| **`中心区`** | `center` | ❌ | **catalog 要 `in_zone_center_*`，`center` 不匹配** |
| **`边缘区`** | `边缘区` | ❌ | 半翻译脏名 |
| **`边缘区到中心区`** | `边缘区到center` | ❌ | 含"中心"字样的陷阱列 |

失败机制：catalog 的 `center_time_ratio` 等 default metric 要求 `in_zone_center_*`
（`catalog/oft.yaml:13-14`）。数据里没有 `in_zone`，只有 `中心区`→`center`。
`_detect_anonymous_zone` 因「无裸 in_zone」返回 `None`（`resolve.py:495-496`）→
落到 `columns_missing` → **raise，整个 plan 中止**（`resolve.py:205-217`）。

### 1.3 为什么列名不固定 —— EV19 机制

> 行为学同事 review package（`docs/review-packages/2026-05-13-常见行为学指标外分析要求.md:16`）：
> 「移动距离这个指标，动物在中心区走了多远？……普通的 `Raw` 是没办法获取分区信息的，
> **需要让用户在 EthoVision 里面的数据选择配置-结果中勾选「按分析区输出」后，才能在 `Raw` 中体现**。」

所以 `中心区`/`边缘区`/`边缘区到中心区` **不是固定数据列**——它们是用户在 EV19 里
**自己划的分析区 (zone)、自己命名、勾选「按分析区输出」生成的列**。

推论：
- 列名 **100% 用户自定义**——叫"中心区"、"Center"、"中央"、"C"、"zone_A"都行，EV19 不强制。
- **同一 OFT 实验、两个用户导出的 raw 列名可以完全不同**。
- `Result 1` 是默认块名；用户改它就带分组语义
  （同事 feedback Q3，`docs/review-packages/2026-05-12-feedback.md:22-23`）。

### 1.4 列的三档确定性（决定要不要进 HITL）

判据是**「系统认不认得这列」**，不是「它是不是 zone」：

```
L1 固定列        Trial time…Elongation（EV19 骨架，非用户划的区）  → 认得，不问
系统 default 认得  COLUMN_MAP 命中 / 标准模式（移动距离/速度/Result 1）  → 认得，不问
不认识的列        normalize 后既不在 COLUMN_MAP、也不匹配任何
                catalog requires_columns 模式（中心区/边缘区/…）      → ╳ 必须对齐
```

"不认识"是**可程序化判定的客观条件**，不靠 LLM 拍脑袋。

---

## 2. 已确认决策（D1–D14，逐条与用户确认）

| # | 决策 | 依据 |
|---|------|------|
| **D1** | 问题本质 = 用户在 EV19「按分析区输出」自定义的列，名字 100% 用户说了算，agent 无法从名字推断身份 → 必须 HITL 对齐 | 用户 + review package 第 16 行 |
| **D1'** | 对齐**锚定到范式 base fact**：agent 用范式合法分析区清单当**中性选项菜单**，让用户把自定义列映射过去。base fact = 范式知识（确定），映射 = 用户回答（确定），无一是猜测。**列含义最终落进 SSOT** | 用户对 D1 精化 |
| **D2** | **绝不字面猜** zone 身份。即使"中心区"字面像 center 也不许假设。用客观证据让用户确认 | 同事 feedback「虽然这次猜中了，但是不要猜」 |
| **D3** | **范围** = 除 L1 固定列 + COLUMN_MAP 已命中的派生列外，任何系统不认识的列都要对齐 | 用户「其他都能 customized」 |
| **D4** | **阻断粒度** = 任何不认识的列都阻断分析。阻断的是"未确认"，不是"无关"——无关列也要用户**确认它无关**才放行 | 用户「任何不认识列都阻断」 |
| **D5** | **default 列名库** = 复用现有 COLUMN_MAP + catalog requires_columns，不新建第二份知识 | 用户「复用现有」+ SSOT 铁律 |
| **D6** | **不在 catalog 存范式默认语义**（不写"OFT in_zone = 中心区"这种猜测） | 用户「不存默认语义」+ D2 |
| **D7** | **column_assessment（系统输出，带证据不带猜测）= 输入；column_semantics（用户决议）= 输出**，严格分离，猜测绝不漏进 SSOT | 用户「严格分离输入/输出」 |
| **D8** | **column_semantics 当 SSOT 容器**（全量统一），用户决议落这里 | 用户「column_semantics 全量统一」 |
| **D9** | **分组不必问**（头部 Group 字段，inspect 已能提取）——收窄反问范围 | 实测 + `inspect_uploaded_file` 已实现 |
| **D10** | **保留 zone_unnamed 等 resolve 层兜底**。Gate 1.5 主动对齐是前置仪式，兜底不移除（纵深防御） | 项目反复栽 prompt-only fix |
| **D11** | 新建能力 = **通用列别名表**：`原始列名 → catalog 概念` 映射，落进 column_semantics(SSOT)。resolve 检查 requires_columns 前先用别名表重映射用户列。比 anonymous_zone 一般化、不依赖裸 in_zone | 用户「通用列别名表」 |
| **D12** | **base fact 从 catalog requires_columns 反推**（OFT 要 `in_zone_center_*`/`in_zone_border_*` → 合法区 = center/border）。不另读范式 markdown，零新知识源 | 用户「从 catalog requires 反推」 |
| **D13** | **HITL 话术尺度 = 断言（禁）vs 反诘（允许）**。agent 永远只做：(1) 陈述客观事实 (2) 回放用户回答。永不产出"所以是 X / 推荐 X / X 更可能"。证据与选项解耦 | 用户「尺度抓对了」 |
| **D14** | **场景 C（用户指错）= 方案 Y**：用户回答与客观证据强烈矛盾时，系统**二次反诘**（并置事实与用户回答，问"这是本意吗"），不是替用户改，不是断言用户错。**跨文件列名不一致（场景 E）归 Gate 2 数据质量，不在本设计** | 用户「方案 Y」+「甲给数据质量关卡」 |
| **D15** | **预填推测 ≠ 断言**：反问**可以**预填系统最佳猜测让用户一键确认/否决（"我的理解是：中心区→center，对吗？不对请告诉我"），但**绝不直接采纳**。预填是反诘（用户可一票否决），采纳是断言（系统已据此计算）。这与 D13 一致——边界在"是否已据此动作"，不在"系统有没有形成猜测" | 用户「预填推测让用户确认」 |
| **D16** | **Sprint 拆分**：问题有两层。**Sprint 1 = 名字对齐**（命名差异，1:1 别名）已成型可实施。**Sprint 2 = 结构对齐**（粒度/聚合差异，如 EPM 4 区 → 标准 2 区）**开放式、按范式与行为学同事逐个迭代**——不可用单范式单 case 的发现窄定义它（过拟合） | 用户「这个只是 epm 发现了，还有其他情况 + 剩余 5 范式」 |

---

## 3. 架构

### 3.1 核心数据流

```
①  上传（含用户自定义分析区列）
        ↓
②  identify_ev19_template → paradigm + 模板候选
        ↓
③  inspect_uploaded_file【增强】
      → columns + data_preview
      + 【新】column_assessment = {
          recognized:   [L1 + COLUMN_MAP 命中列],
          unrecognized: [ {raw,normalized,evidence:{occupancy/取值分布}} … ],  ← 触发 HITL
          open_questions: [未对齐的原始列名 …] }
      （带证据，不带猜测 = D7 输入）
        ↓
④  lead 从 catalog requires_columns 反推合法区菜单（D12）
   + 头部 Group 已提取（D9 分组不问）
        ↓
⑤  ask_clarification（合并反问；话术守 D13 断言/反诘尺度）
        ↓ Command(goto=END) 中断等回复
        ↓
⑥  用户拍板 → set_experiment_paradigm(【新】column_semantics={…})
      → 写 experiment-context.json（column_semantics = SSOT, D8）
      → 【新】写盘时单向投影出 column_aliases（D11）一起落盘
        ↓
⑦  prep_metric_plan → resolve_metrics(columns, column_aliases=…)
      → 【唯一新建核心逻辑】单点重映射：中心区/center → in_zone_center
      → _missing_columns 匹配上 → metric 算出来
        ↓
⑧  guardrail【增强】column_semantics.open_questions 未清空 → 拦截（D4 阻断）
        ↓
⑨⑩ code-executor → data-analyst → report-writer
      → 报告引用 column_semantics 的语义（D1 语义落 SSOT 供叙述）
```

### 3.2 通用列别名表 —— 唯一新建核心能力

**插入点**：`resolve_metrics(paradigm, columns, …)`（`resolve.py:116`）的 `columns`
是所有 `_missing_columns` 检查（`resolve.py:193/249/381/410`）读的**同一个列表**。
别名重映射只需在 `columns` 进入后、第一次 `_missing_columns` 之前做**一次**：

```python
# resolve_metrics() 内，load_catalog 之后插入：
if column_aliases:                       # {原始列 or normalized: catalog 概念}
    columns = _apply_aliases(columns, column_aliases)
    # "中心区"→normalize→"center"，别名表 {"center": "in_zone_center"}
    # → columns 里出现 "in_zone_center" → 下游所有 _missing_columns 看到的是 catalog 期望名
```

**下游 0 改动**：`_missing_columns`、`_detect_anonymous_zone`、
`_compute_parameters_in_use`、所有 metric/chart 检查全部照旧——它们看到的 `columns`
已是重映射后的。承重墙的消费入口不动。

### 3.3 与 anonymous_zone 旧链路的关系

| | anonymous_zone（已有） | 通用列别名（新） |
|---|----------------------|-----------------|
| 触发前提 | 数据里有**裸 `in_zone`** | 无前提，任意用户列 |
| 能处理 | 单个匿名 zone 的身份 | 任意「未知列 → catalog 概念」映射 |
| 真实数据救得了吗 | ❌（这批没有裸 in_zone） | ✅ |
| 去留 | **保留为兜底**（D10），不移除 | 主路径 |

别名表是 anonymous_zone 的超集；anonymous_zone 作为纵深防御兜底继续存在。

### 3.4 base fact 从 catalog 反推（D12）

不另建知识源。OFT 的 `oft.yaml` 中 default metric 的 `requires_columns` 出现
`in_zone_center_*` / `in_zone_border_*` / `in_zone_corner_*` → 反推合法分析区菜单 =
{中心区(center), 边缘区(border), 角落(corner)}。范式知识只有一处真源（catalog）。

---

## 4. HITL 话术（D13 断言/反诘尺度 + D15 预填）

### 4.1 尺度定义

```
❌ 断言（agent 替用户下结论并据此动作，禁止）：
   "中心区→center（已采用）"          ← 系统直接拿去算
   "这是 center。"                     ← 下结论

✅ 预填反诘（agent 提出可一票否决的建议，允许 — D15）：
   "我的理解是：中心区→center、边缘区→border、边缘区到中心区→忽略。
    对吗？不对请告诉我正确的对应。"
   ← 系统给出最佳猜测当【预填项】，但必须：
     (a) 明示这是"我的理解/建议"，不是既成事实
     (b) 用户可一键否决/改写任意一项
     (c) 用户未确认前，系统不据此计算（guardrail 拦着，D4）
```

**边界澄清**：D13 禁的是"断言并据此动作"，不是"系统形成猜测"。
D15 允许系统把猜测作为**预填建议**呈现——因为预填是反诘（用户可否决），
采纳才是断言（系统已动作）。两者一致，区别只在"是否已据此动作"。

预填的依据 = 客观证据（占时分布）+ catalog 反推的合法区菜单（D12）。
系统**可以**说"中心区占时 18%，符合 center 的典型低占用，我猜它是 center"——
只要它**同时**把这句标成"我的猜测，请确认"，且把否决/改写的口子留给用户。

### 4.2 场景 B 反问模板（主场景，预填 + 可否决）

```
⚠️ 开始分析前需确认：

1. EV19 模板：OpenFieldRectangle / OpenFieldCircle

2. 你的数据有 3 列是自定义分析区。基于各列的取值分布，我的初步理解是：

   ┌──────────────┬──────────────────┬──────────────────┐
   │ 你的列名      │ 取值=1 时间占比   │ 我的理解（请确认） │
   ├──────────────┼──────────────────┼──────────────────┤
   │ 中心区        │ 18%              │ → 中心区 center   │
   │ 边缘区        │ 82%              │ → 边缘区 border   │
   │ 边缘区到中心区 │ （非 0/1）        │ → 疑似距离列，忽略 │
   └──────────────┴──────────────────┴──────────────────┘

   OFT 分析的区类型有：中心区 / 边缘区 / 角落 / 其它(忽略)。
   以上理解对吗？如有错误，请告诉我正确的对应（例："边缘区到中心区其实是角落"）。

   分组已自动识别为 KK/MM/MN/SS/TT，无需填。
```

约束（D15 + D13 共同约束）：
- 预填的"我的理解"列**必须标注"请确认"**，措辞是建议不是断言。
- 占时证据与预填并列展示，让用户能**据证据核对**预填对不对。
- 必须给"如有错误请告诉我"的否决口子；用户改写优先于预填。
- 预填来自证据+catalog 菜单，**不来自字面**（不因列名含"中心"就填 center —— 该列
  之所以预填 center 是因为占时 18% 符合 center 典型低占用，不是因为它叫"中心区"）。

### 4.3 场景 C 二次反诘模板（D14 方案 Y）

用户否决了预填、或给了与证据强烈矛盾的映射（如"边缘区=中心区"，但边缘区占时 82%）：

```
确认一下：你把「边缘区」这列标为 中心区(center)。
这列取值=1 的时间占 82%。
（中心区在 OFT 中通常是动物较少停留的区，仅供你核对。）
这是你的本意吗？回复"是"继续，或更正映射。
```

约束：不说"你错了"，不说"这应该是 border"，不替用户改。
用户回"是"则照办（可能这批动物异常，或用户有特殊划区）。

---

## 5. 分情况工作流走查

| 场景 | 条件 | 行为 | 兑现 |
|------|------|------|------|
| **A 标准数据** | 所有列系统认得 | column_assessment.unrecognized=[] → 跳过列对齐反问 → 流程同今天 | 标准数据零额外开销 |
| **B 自定义 zone（主场景）** | 中心区/边缘区/… | 一次合并反问（模板+3 列+确认分组无误）→ 用户一句话回 → 别名重映射 → 算 | D3/D4/D9/D13 |
| **C 用户指错** | 回答违背占时证据 | 二次反诘（并置事实与回答，问"本意吗"），不替改 | D14 方案 Y |
| **D 无关列** | catalog 不要的列（边缘区到中心区） | 进 open_questions，用户答"忽略"后才清空 → guardrail 放行。无关列也要确认无关 | D4 |
| **E 跨文件不一致** | A 文件「中心区」B 文件「Center」 | 不在本设计 → 报数据质量问题，归 Gate 2 | D14 |

---

## 6. Sprint 拆分：名字对齐 vs 结构对齐（D16）

列对齐问题有**两层**，难度与确定性差异很大，必须分 Sprint。

### 6.1 Sprint 1 — 名字对齐（命名差异，1:1）

**问题**：catalog 要某概念（如 OFT center），用户列叫什么都行（中心区/Center/C/zone_A），
**一个用户列 ↔ 一个 catalog 概念**。

**解法**：本文档 §1–§5 + §7–§9 的全部设计。通用列别名表是 1:1 映射，
预填反诘搞定身份确认。**已成型，可直接实施。**

**覆盖**：本场景 = 真实 34 文件 OFT 数据（实测主场景）。

### 6.2 Sprint 2 — 结构对齐（粒度/聚合差异，N:1）

**问题**：用户数据的**分区粒度**与 catalog 最佳实践不同，需要聚合/拆分才等价。
**已知一例**（用户提供）：

```
EPM 标准最佳实践按【2 区】算：open_arm, closed_arm
但有的用户数据是【4 区】：open_arm1, open_arm2, closed_arm1, closed_arm2
（用户把每条臂单独划了 zone）
需聚合：open_arm = open_arm1 ∪ open_arm2，closed_arm = closed_arm1 ∪ closed_arm2
聚合后才与标准 2 区等价，才能套 catalog metric。
```

**已勘察到的现成基础（但不等于 Sprint 2 已解决）**：
EPM 的 `metrics/epm.py` 脚本**已有 OR 聚合层**——
`compute_open_arm_time_ratio` 用 `df[cols].max(axis=1)`（`epm.py:96`）把多列按
"任一列=1 即在开臂"合并；`_get_open_zone_cols` 用正则 `in_zone.*open.?arm` 抓所有开臂列。
catalog 的 `requires_columns: in_zone_open_arms_*`（`epm.yaml:9`）glob 本就允许多列。
**所以 EPM 这一例**：只要用户的 open_arm1/2 都被映射成 `in_zone_open_arms_*` 能抓到的名字，
聚合自动发生——是 Sprint 1 别名表从"概念:单列"放宽到"概念:多列"的自然扩展。

**⚠️ 但这只是 6 范式中 1 个范式的 1 种结构 case，绝不可当 Sprint 2 全貌**：
- EPM 自身可能还有其他未勘察的结构差异。
- 其余 5 范式（OFT/LDB/FST/Zero Maze/TST）的结构差异**完全未勘察**。
- 聚合语义不一定都是 OR（可能加权、可能需区分臂身份的统计），**必须逐个与行为学同事确认，不能拍脑袋**。

**解法**：**开放式、按范式与行为学同事逐个迭代**。Sprint 2 不预先封闭定义，
每勘察一个范式的真实结构 case，与同事确认聚合语义，增量落地。

> **勘察 issue**：[#98 — 6 范式分析区「结构差异」勘察](https://github.com/noldus-cn-beijing/noldus-insight/issues/98)。
> **Sprint 1 实施 spec**：[`docs/superpowers/specs/2026-06-05-column-semantics-alignment-sprint1-spec.md`](../superpowers/specs/2026-06-05-column-semantics-alignment-sprint1-spec.md)。

**Sprint 1 与 Sprint 2 的接缝**：别名表结构预留"概念可接受多个用户列"的形态
（`resolves_to` 可为列表），使 Sprint 2 的多列聚合是 Sprint 1 的平滑扩展而非重写。

---

## 7. 实施清单（Sprint 1：名字对齐）

> 本清单只覆盖 Sprint 1。Sprint 2（结构对齐）按 §6.2 与同事迭代，单独立计划。

| 文件 | 改动 | 性质 | 估计量 |
|------|------|------|--------|
| `ethoinsight/parse/_core.py` | `detect_ethovision` L1 固定列签名兜底（marker 缺失时靠 XY+≥3 L1 列判定 EV19）。阈值**须用 34 文件实测**，不拍脑袋 | 加固 | ~20 行 |
| `ethoinsight/utils.py` | 新增 `assess_column_confidence()`：对每列判 recognized/unrecognized（依据 COLUMN_MAP + 传入的 catalog requires_columns 模式）。**不新建第二份列名库**（D5） | 新建 | ~60 行 |
| `ethoinsight/catalog/resolve.py` | `resolve_metrics` 加 `column_aliases` 参数 + `_apply_aliases()` 单点重映射。下游消费入口不动 | **核心新建** | ~40 行 |
| `inspect_uploaded_file_tool.py` | 返回值增 `column_assessment`（复用已有 `_compute_anonymous_zone_evidence` 的证据产出思路，泛化到任意未知列）+ `open_questions` | 增强 | ~60 行 |
| `experiment_context.py` | `set_experiment_paradigm` 加 `column_semantics` 入参 + 写盘时单向投影 `column_aliases`（照 `acknowledge_quality` 模式模板） | 增强 | ~50 行 |
| `ev19_template_provider.py` | guardrail 加子检查：`column_semantics.open_questions` 未清空 → 拦截 task(code-executor)。仅 open_questions 非空时拦截（标准数据不触发） | 增强 | ~25 行 |
| `lead_agent/prompt.py` | 「反问合并规则」段加列对齐范例（按 §4.2 话术）+ 触发条件 + skill 指针 | 增强 | ~15 行 |
| `skills/custom/ethoinsight-column-confirmation/` | 新建 thin SKILL.md + references（话术正例/反例、答案映射、跳过条件）。注意：lead-only skill，三件套 = 文件 + extensions_config 注册 + **lead prompt** read 指引（subagent 侧不动） | 新建 | ~140 行 |
| `extensions_config.json` | 注册新 skill | 3 行 |
| `tests/` | 回归：34 文件 fixture（未对齐时阻断 + 对齐后算出 center metric）+ 别名重映射单测 + 话术无断言断言（红锚点） | 测试 | ~220 行 |

---

## 8. SSOT schema：experiment-context.json `column_semantics`

```json
{
  "paradigm": "oft",
  "ev19_template": "OpenFieldRectangle-AllZones",
  "column_semantics": {
    "confirmed_at": "<ISO8601>",
    "columns": {
      "中心区": {
        "raw_name": "中心区",
        "normalized": "center",
        "resolves_to": "in_zone_center",
        "meaning_zh": "中心分析区",
        "confirmed": true
      },
      "边缘区": {
        "raw_name": "边缘区",
        "normalized": "边缘区",
        "resolves_to": "in_zone_border",
        "meaning_zh": "边缘分析区",
        "confirmed": true
      },
      "边缘区到中心区": {
        "raw_name": "边缘区到中心区",
        "normalized": "边缘区到center",
        "resolves_to": null,
        "ignore": true,
        "confirmed": true
      }
    }
  },
  "parameter_overrides": { "...": "由 column_semantics 单向投影派生（derived）" }
}
```

- `resolves_to` = 喂 resolve 的机器决议（投影成 column_aliases）。
- `meaning_zh` = 喂 report-writer 的叙述语义（D1 语义落 SSOT）。
- `ignore:true` 的无关列也必须 `confirmed:true`（D4：确认它无关）。
- `parameter_overrides` 是 column_semantics 的**派生产物**（写盘时单向投影），
  resolve 仍读 parameter_overrides → 下游零感知（爆炸半径锁在写盘点）。

**Sprint 2 前向兼容**（结构对齐，N:1）：上例是 Sprint 1 的 1:1 形态（`resolves_to` 取单值）。
Sprint 2 的多列聚合（如 EPM open_arm1+open_arm2 → open_arm）可表达为**多个用户列
`resolves_to` 同一个 catalog 概念的不同 glob 槽位**（如 `in_zone_open_arms_1` /
`in_zone_open_arms_2`，都被 `in_zone_open_arms_*` 抓到 → 现成 OR 聚合层合并）。
schema 不必为 Sprint 2 改结构，别名表"概念可接受多列"是同一机制的自然延伸（§6.2）。

---

## 9. 守住的铁律 / 反模式自检

- **D2 不猜 ≠ 不预填**：column_assessment 产证据；系统**可**把基于证据+catalog 菜单的猜测
  作为**可否决的预填建议**呈现（D15），但不直接采纳、不据此动作（采纳由用户确认后才发生）。
  不因列名字面预填（"中心区"不因名字含"中心"被填 center，而因占时符合 center 典型）。
- **不把单范式单 case 当全貌**：Sprint 2 结构对齐开放式迭代（D16）；EPM OR 聚合只是一例，
  其余 5 范式 + EPM 其他 case 未勘察，聚合语义须逐个问同事。
- **SSOT 不双存**：default 列名库复用 COLUMN_MAP+catalog（D5）；base fact 从 catalog 反推（D12）；
  column_semantics 是决议唯一落点，parameter_overrides 由它派生。
- **不在 skill 描述工具返回结构**（反脑补铁律）：column_assessment 字段结构只在工具内定义，
  SKILL.md 只教"何时触发对齐 / 怎么交互"，不写"inspect 返回什么"。
- **纵深防御**：Gate 1.5 主动对齐（prompt 驱动）+ resolve zone_unnamed 兜底（确定性）+ guardrail 硬拦（D10）。
- **承重墙消费入口不动**：别名重映射在 columns 入口单点完成，resolve 下游全不改。
