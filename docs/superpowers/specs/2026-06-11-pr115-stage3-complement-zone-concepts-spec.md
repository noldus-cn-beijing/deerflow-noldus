# Stage 3 — Q3 前置：补 border/dark/closed 概念枚举，让 catalog 完整

> 本 spec 可由单个 agent 独立执行。它是总纲 spec
> [`2026-06-11-pr115-catalog-concept-consolidation-and-gate-cnf-spec.md`](./2026-06-11-pr115-catalog-concept-consolidation-and-gate-cnf-spec.md)
> §2.3 的可执行展开。**只覆盖 Stage 3**，不实现 Stage 1/2/4 的内容。
>
> ✅ **执行前必读（决策门已闭合）**：OFT `border` 的形态已由 Fable 决策门 1 闭合（见 §8 D1）——落地为一个 `binding=None` 的一等概念条目 `ResolvedZoneConcept(concept="border", binding=None, source="explicit_concept")`，**不再有取舍、无需任何人再拍板**。本 spec 正文（§3/§5/§7/§6 实施顺序）只描述这条闭合路径，A1/A2/总纲 (b) 三个旧分支已作废（仅在 §8 决策溯源表留作历史）。LDB `dark` / ZM `closed` 走 `binding=ParamBinding(真 param)`（dark→`dark_zone`、closed→`closed_zones`），同样无决策门，可直接实施。
>
> ⚠️ 形态对齐铁律：**OFT border 用 `binding=None`（无注入点），LDB dark / ZM closed 用 `binding=ParamBinding(真列名参数)`（有注入点）——三者别搞混。**

---

## 1. 目标与在总纲中的位置

**目标（一句话）**：在 catalog 内**显式枚举**三个"补集区"概念——OFT `border` / LDB `dark` / ZM `closed`——并把它们纳入 Stage 2 建立的统一内部概念模型 `resolved_zone_concepts`，使 catalog 成为完整的概念 SSOT；**只补"概念可被对齐"这个结构，绝不实现任何 N:1 聚合语义**。

**依赖（上游 stage，必须先完成）**：

- **Stage 2 必须已合入**。Stage 2 给 `Catalog` 加了派生字段 `resolved_zone_concepts: dict[str, ResolvedZoneConcept]`（dataclass 定义见 Stage 2 spec §3 改动 1，字段 `concept: str` / `binding: ParamBinding | None` / `source: str`——`binding` **整体可空**，`ParamBinding` 内含 `param: str` + `wrap_list: bool`；`binding=None` 表示"可对齐但无注入点"的概念），loader 在加载期填充（先纳入 `zone_concept_params`，再对 `anonymous_zone_override` 跑 `_derive_concept_from_zone_patterns` 规范化）。Stage 3 把三个补集区**注入同一个 `resolved_zone_concepts` dict**——若 Stage 2 的派生字段不存在，本 stage 无处落地。
- **重要前提核实（实施第一步）**：`ResolvedZoneConcept`、`Catalog.resolved_zone_concepts` 当前在 dev 仓库**尚不存在**（grep 全仓为空，它们是 Stage 2 产物）。因此 §5 的所有 red 测试**只有在 Stage 2 已合入的分支上才有意义**——若在纯 dev（无 Stage 2）上起手，§5.1 会因 `AttributeError: 'Catalog' object has no attribute 'resolved_zone_concepts'` 而红，而非"概念缺失"红，红的语义会误导。**实施前先 `grep -rn "resolved_zone_concepts" packages/ethoinsight/` 确认 Stage 2 已落地**；若未落地，停下，先合 Stage 2。
- Stage 1（CNF 嵌套 requires_columns）与本 stage **正交**，不构成依赖；但若 Stage 1 已合，本 stage 改动的 catalog/loader 不得破坏其 flatten 逻辑（见 §7 风险）。

**阻塞（下游 stage，等本 stage 完成）**：

- **Stage 4**（从统一模型生成概念菜单）。Stage 4 的**构建期生成器**遍历 `resolved_zone_concepts` 产出概念速查菜单（菜单只读每条的 `concept` 名，**无 `binding is None` 分支**——对齐的本体是认领概念，注入与否不影响菜单条目是否出现）；今天 catalog 里 `border`/`dark`/`closed` **都未被枚举为"可对齐概念"**（OFT 没有 border 概念、LDB/ZM 的补集区只是 override 正区的隐式补），生成器跑出来的菜单就是残缺的。Stage 3 是 Stage 4「权威源完整」的前提。
  - **Stage 4 形态（Fable 决策门 2 已闭合，记此供 Stage 3 实施者理解下游契约，本 stage 不实现）**：概念菜单走**构建期生成 + CI staleness 兜底**，**不走运行期 CLI**。生成器**拥有整个生成文件**（`references/zone-concepts.generated.md` 等），手写 SKILL.md / answer-mapping 改为**链接指向生成文件、不内嵌概念表**（无哨兵块嵌入方案）。Stage 3 只负责让 `resolved_zone_concepts` 完整，不碰 Stage 4 的生成器/文件。

---

## 2. 已核实事实锚点（带文件:行号，2026-06-11 实读确认）

所有行号基于 `/home/wangqiuyang/noldus-insight/packages/ethoinsight/`，已现场 Read 核实。

### 2.1 metrics 端：三范式补集区的"可注入 param"形态异质（决定各自 binding 形态的核心证据）

| 范式 | 补集区 | metrics 端现状 | 独立可注入 param？ |
|------|--------|----------------|---------------------|
| **LDB** | `dark` | `ldb.py:49-53` `compute_transition_count(df, light_zone="in_zone_light", dark_zone="in_zone_dark")`——`dark_zone` 是**独立函数签名参数**（`:52`），函数体 `:63-79` 真正消费它（`has_dark`/`_count_0_to_1(df[dark_zone])` OR 计两列穿梭）。 | **有** → `binding=ParamBinding("dark_zone")` |
| **ZM** | `closed` | `zero_maze.py:40-50` `_get_closed_zone_cols(df, closed_zones)` + `:147-154` `compute_hesitation_count(df, open_zones=None, closed_zones=None, ...)`（签名 `:147`，`closed_zones` 形参 `:150`）——`closed_zones` 是**独立 list 参数**。 | **有** → `binding=ParamBinding("closed_zones")` |
| **OFT** | `border` | `oft.py:43-50` `_find_periphery_zone_column(df)` 用 regex `in_zone.*(peripher\|edge\|wall\|border\|outer)` **自动识别**列（无参数 hint）；但 `compute_thigmotaxis_index`（`:74-115`，下一个 def 在 `:118`）**无 `border_zone` / `periphery_zone` 函数参数**——它靠 ①`_find_periphery_zone_column` 自动 regex 找列（`:88`）②`1 − center_time_ratio` 补集（`:95-97`）③几何反推（`:100-115`）三级 fallback。center metric（`:58-71`）也只吃 `center_zone`，border 永远是 `1−center` 反推或自动找列。 | **无可注入 param** → `binding=None`（可对齐但无注入点，见 §8 D1 闭合结论） |

### 2.2 catalog 端：三个补集区今天都未被枚举为"可对齐概念"

- `oft.yaml`：只有 `center_zone` 参数，`anonymous_zone_override.target_param: center_zone`（`:6-8`）。**无任何 border 概念声明**。
- `ldb.yaml`：所有 metric 只暴露 `light_zone` 参数，`anonymous_zone_override.target_param: light_zone`（`:5-7`）。**`dark` 未被枚举为可对齐概念**——尽管 `compute_transition_count` 函数有 `dark_zone` 参数、且 `zone_entry_distribution` chart 的 `requires_columns` 已含字面量 `in_zone_dark`（`ldb.yaml:111`），catalog 从未把 `dark` 作为 `zone_concept_params` / `anonymous_zone_override` 里的**可对齐概念**声明出来（即没有任何 concept→param 的对齐入口）。
- `zero_maze.yaml`：所有 metric 只暴露 `open_zones` 参数，`anonymous_zone_override.target_param: open_zones`（`:6-8`）。**`closed` 未被枚举为可对齐概念**。

> 措辞校正（区分"字面量列名"与"可对齐概念"）：上面刻意不写"dark 在 catalog 中不存在"——`in_zone_dark` 字面量确实出现在 `ldb.yaml:111` 的 chart `requires_columns` 里。本 stage 要补的是把 `dark`/`closed`/`border` 枚举为 **`resolved_zone_concepts` 里可被 HITL 对齐的概念条目**，与某物理列名是否曾出现在某 chart 的 requires 里是两回事。

> 总纲 §1.4 结论："catalog 今天结构上根本没把 border/dark/closed 这些'补集区'枚举为可对齐概念。" 本 stage 修这一点。

### 2.3 注入与解析路径（Stage 3 改动落点的上下文）

- `schema.py:118-127` `ZoneConceptParam(param, wrap_list)`（`param: str` 在 `:126`），**按概念名 key** 存于 `Catalog.zone_concept_params`（`:142`）。今天仅 EPM 用（`open_arms→open_arm_zones` / `closed_arms→closed_arm_zones`）。
- `loader.py:154-177` 解析 `zone_concept_params` 块。**`:163` 是 `if not isinstance(param, str) or not param:`——强制 `param` 为非空 str**。Stage 3 给 LDB/ZM 的 yaml 新增 `zone_concept_params` 块时：LDB/ZM 的 param 是真列名 str（`dark_zone` / `closed_zones`）**复用这套已存在的解析器，零 loader/schema 改动**。**OFT border 不写 `zone_concept_params`**（它无注入点，落地为 `binding=None` 概念条目，由 Stage 2 规范化段直接构造，根本不进 `:163` 这条 str 校验路径）——因此 `:163` 全程不需要放松。
- Stage 2 后：loader 把 `zone_concept_params` 条目纳入 `resolved_zone_concepts`（Stage 2 spec §3 改动 3，loader `:131-135` 的 `for concept_key, zcp in zone_concept_params.items()` **无条件全量纳入**——结构上不可能只纳 EPM 的两个 key；这是 §7.1 的防御提醒能否成立的依据，实测 Stage 2 是全量纳入，故新概念会自动流入，每条 `binding=ParamBinding(param, wrap_list)`）。Stage 3 给 LDB/ZM 两个 yaml 补的 `zone_concept_params` 条目会**自动流入** `resolved_zone_concepts`，无需改 loader 填充逻辑。
- `resolve.py:666-677`：注入循环 `for concept_key, cols in concept_cols.items()` → `mapping = concept_param_map.get(concept_key)` → ... → `overrides[param_name] = cols if wrap_list else cols[0]`。**Stage 2 已在此循环顶端加 `if rc.binding is None: continue`**（语义=无注入点的概念不进 param 路由，非防御补丁）。OFT border（`binding=None`）因此被该 `continue` 跳过，不产生任何 override 键，现有注入输出零变化——这条 `continue` 就是 §5.4 border-alias 护栏要守的点。

---

## 3. 改动清单（逐文件逐处）

> **逐范式异质处理**——这是 Stage 3 的核心纪律。三个补集区的 metrics 形态不同，取向不同。**不跨范式复用、不因结构像而合并**（CLAUDE.md 第 14 条）。

### 3.1 LDB `dark` → 显式声明 `zone_concept_params`（`binding=ParamBinding("dark_zone")`）【无决策门，可直接实施】

**理由**：`compute_transition_count` 已有独立 `dark_zone` 函数参数并真正消费（`ldb.py:52,64,79`），存在可注入的 param 名 → 显式声明直接可行，无需新增推导规则。

**改 `packages/ethoinsight/ethoinsight/catalog/ldb.yaml`**：在文件顶层（与 `anonymous_zone_override` 同级）新增 `zone_concept_params` 块，声明 `dark` 概念：

```yaml
zone_concept_params:
  dark:
    param: dark_zone
    wrap_list: false
```

- `param: dark_zone` 对齐 `compute_transition_count` 的函数参数名（`ldb.py:52`）。
- `wrap_list: false`：`dark_zone` 是单列字符串（不是 list），对齐函数签名 `dark_zone: str = "in_zone_dark"`。
- **不动** `anonymous_zone_override`（仍 `target_param: light_zone`，承载关注点 (2) 统一输入键，§7 铁律）。
- **不动**任何 metric 的 `parameters` 块、不新增 metric、不改 `requires_columns`（不引入聚合）。

### 3.2 ZM `closed` → 显式声明 `zone_concept_params`（`binding=ParamBinding("closed_zones")`）【无决策门，可直接实施】

**理由**：`compute_hesitation_count` 已有独立 `closed_zones` list 参数（`zero_maze.py:150`），`_get_closed_zone_cols`（`:40-50`）消费它 → 显式声明直接可行。

**改 `packages/ethoinsight/ethoinsight/catalog/zero_maze.yaml`**：新增 `zone_concept_params` 块：

```yaml
zone_concept_params:
  closed:
    param: closed_zones
    wrap_list: true
```

- `param: closed_zones` 对齐 `compute_hesitation_count` 参数名（`zero_maze.py:150`）。
- `wrap_list: true`：`closed_zones: list[str] | None`，是列表，对齐 ZM `open_zones` 的 `wrap_list: true` 惯例（`zero_maze.yaml` override `:6-8`）。
- **不动** `anonymous_zone_override`（仍 `target_param: open_zones`）。
- **不动** metric `parameters` 块、不新增 metric、不改 `requires_columns`。

### 3.3 OFT `border` → 写一条 `binding=None` 概念条目【决策门已闭合，确定补 border】

**理由（异质点）**：metrics 端 `_find_periphery_zone_column`（`oft.py:43-50`）能 regex 自动找到 border/periphery 列，但 **`compute_thigmotaxis_index`（`:74-115`）签名无 `border_zone` 参数**——thigmotaxis 靠 `1−center` 反推或自动找列，center 靠 `center_zone`。**没有可注入的 param 名**。所以 LDB/ZM 那种 concept→param 字符串映射在 OFT 这里**无对应物**。

本 stage **不擅自给 OFT 加 `border_zone` compute 参数**（那是改 metrics 行为，超出"只补概念枚举结构"的边界——对应 §8 决策 D1 溯源表的 A3，留独立 sprint，本 stage 明确不做）。

**落地形态（Fable 决策门 1 闭合结论）**：把 border 编码为一个 `binding=None` 的一等概念条目——

```python
ResolvedZoneConcept(concept="border", binding=None, source="explicit_concept")
```

含义：border **存在、可被 HITL 认领、但无运行时注入点**。这比"把 border 绑回 center_zone"更诚实，且因 Stage 2 的 `binding` 本就可空（`ParamBinding | None`），**零额外 schema/loader 放松**。

**OFT 落地动作**：
- **不动** `oft.yaml`——border **不写进 `zone_concept_params`**（无 param 可写），oft.yaml 保持只有 `center_zone` + `anonymous_zone_override.target_param: center_zone`。
- **不动** `loader.py:163` 非空校验、**不动** `ResolvedZoneConcept`/`ParamBinding` 字段类型、**不动** metrics。
- border 条目由 **Stage 2 规范化段（或本 stage 的 OFT 填充点）直接构造**追加进 `resolved_zone_concepts`（见 §3.4）。
- 等价性天然成立：Stage 2 resolve 注入循环已加 `if rc.binding is None: continue`，border 进 `resolved_zone_concepts` 不污染任何现有注入输出（§5.4 守此）。

### 3.4 Stage 2 派生字段填充

- **LDB/ZM（`binding=ParamBinding(真 param)`）**：`resolved_zone_concepts` 填充**无需改 loader**——Stage 2 已写的"全量纳入所有 zone_concept_params 条目"逻辑（loader `:131-135`，无条件 `.items()` 循环）会自动吃下新条目，每条产出 `ResolvedZoneConcept(concept, binding=ParamBinding(param, wrap_list), source="zone_concept_params")`。**实测 Stage 2 是全量纳入而非只纳 EPM 两个 key**（§2.3 已核实），故此路无哑故障风险；§5.1 测试兜底。
- **OFT（`binding=None`）**：在 Stage 2 的 loader override 规范化段（Stage 2 spec §3 改动 3 的 1b 分支之后、`Catalog(...)` 构造之前）对 OFT **追加一条** `ResolvedZoneConcept(concept="border", binding=None, source="explicit_concept")`。这只改 Stage 2 的**填充逻辑**（让它在 center 之外再产一条 border 条目），**不改任何字段类型**——`binding` 本就可空，零 schema 放松。这是本 stage 唯一触及 Stage 2 loader 的点。

---

## 4. 接口契约

### 4.1 输入契约（本 stage 依赖上游 Stage 2 的产出）

- `Catalog` 已有派生字段 `resolved_zone_concepts: dict[str, ResolvedZoneConcept]`，loader 在 `_parse_catalog` 末尾（`loader.py:179` 的 `Catalog(...)` 之前）填充。
- loader 已把 `zone_concept_params` **全量**纳入 `resolved_zone_concepts`（Stage 2 spec §3 改动 3，无条件 `.items()`）。
- `resolve.py` 的 `concept_param_map` 已改读 `cat.resolved_zone_concepts`（Stage 2 spec §3 改动 4），删掉自跑 derive 的 Step 1 分支。
- `anonymous_zone_override` 的 softgate（`resolve.py:744-790` 区段）与 translate（`:1136-1139` 区段）**仍读 `cat.anonymous_zone_override`**，Stage 2 未动、Stage 3 也不动。

### 4.2 输出契约（本 stage 给下游 Stage 4 的产出）

- `resolved_zone_concepts` 对**每个范式包含其全部可对齐 zone 概念**：
  - EPM：`open_arms`、`closed_arms`（Stage 2 已纳入，本 stage 不碰）。
  - OFT：`center`（Stage 2 规范化 override 而来，`binding=ParamBinding("center_zone")`）+ `border`（本 stage 新增，`binding=None`，source=`explicit_concept`）。
  - LDB：`light`（override 而来）+ `dark`（本 stage 新增，`binding=ParamBinding("dark_zone")`，source=`zone_concept_params`）。
  - ZM：`open`（override 而来）+ `closed`（本 stage 新增，`binding=ParamBinding("closed_zones")`，source=`zone_concept_params`）。
- 每个概念条目的语义：concept 名 → `ResolvedZoneConcept(concept, binding, source)`，`binding` 为 `ParamBinding(param, wrap_list)` 或 `None`。**Stage 4 构建期生成器据此产概念菜单（只读 concept 名，不分 binding 是否为 None）。**
- **harness 侧零消费**：grep `packages/agent/` 对 `zone_concept_params` / `resolved_zone_concepts` / `ResolvedZoneConcept` 全空——harness 不直接消费本 stage 产出，契约只供 Stage 4 生成器（库内），方向闭合。
- **不变量**：本 stage **不改任何 metric/chart 的 resolve 输出**（param overrides 注入结果）。新增的 `dark`/`closed`/`border` 概念**仅扩充 `resolved_zone_concepts`**，不参与现有 plan 生成（除非用户经 HITL 主动对齐到这些概念，而 HITL 对齐通路本身不在本 stage 范围）。这是与 Stage 2"resolve 输出字节不变"等价性同源的硬约束。

---

## 5. TDD（red 先行）

> **前提声明**：本 stage 所有测试以 **Stage 2 已合入**为前提（`resolved_zone_concepts` / `ResolvedZoneConcept` 是 Stage 2 产物，当前 dev 不存在）。若在纯 dev 上起手，§5.1/§5.4 会因 `AttributeError`（字段缺失）而非"概念缺失"而红——红的语义会误导。**实施第一步 grep 确认 Stage 2 已落地，否则先合 Stage 2。**

新建测试文件 `packages/ethoinsight/tests/test_stage3_zone_concept_enumeration.py`。**先写全部断言并确认 red，再改 yaml/loader 转 green。**

### 5.1 概念枚举存在性（red → green）

- `test_ldb_catalog_enumerates_dark_concept`：load LDB catalog，断言 `"dark" in cat.resolved_zone_concepts`，且其 `binding.param == "dark_zone"`、`binding.wrap_list is False`、`source == "zone_concept_params"`。**改 ldb.yaml 前 red（概念缺失）**。
- `test_zm_catalog_enumerates_closed_concept`：load ZM catalog，断言 `"closed" in cat.resolved_zone_concepts`，`binding.param == "closed_zones"`、`binding.wrap_list is True`、`source == "zone_concept_params"`。**改 zero_maze.yaml 前 red**。
- `test_oft_catalog_enumerates_border_concept`：load OFT catalog，断言 `"border" in cat.resolved_zone_concepts`，且其 `binding is None`、`source == "explicit_concept"`（border 是可对齐但无注入点的概念）。**改 Stage 2 OFT 填充点前 red**。

### 5.2 param 名对齐 metrics 函数签名（防漂移，green 即守，**不依赖 Stage 2，现在就能写**）

- `test_ldb_dark_param_matches_compute_signature`：用 `inspect.signature(ldb.compute_transition_count)` 断言存在名为 `dark_zone` 的参数——证明 catalog 声明的 param 名真能注入到 compute（不是凭空起名）。
- `test_zm_closed_param_matches_compute_signature`：`inspect.signature(zero_maze.compute_hesitation_count)` 断言有 `closed_zones` 参数。
- （OFT 无此测试——border `binding=None`、无注入点，正是其语义；没有独立 border 参数可查。）

> 这两条是最干净的护栏：实测两参数存在（`ldb.py:52` / `zero_maze.py:150`），red→green **现在就能写，不依赖 Stage 2**。

### 5.3 不跨范式复用（守边界）

- `test_concepts_not_cross_paradigm_leaked`：断言 `dark` 只出现在 LDB 的 `resolved_zone_concepts`、`closed` 只在 ZM、`border` 只在 OFT；任一不出现在其它范式的 dict 中。

### 5.4 等价性回归（守"不改现有 resolve 输出"——本 stage 的可执行无损证明）

- `test_stage3_resolve_output_byte_equivalent`：对一组代表性 fixture 跑完整 resolve（生成 plan / param overrides），断言 Stage 3 改动**前后逐字节相同**。
  - **fixture 必须含两类**：
    1. **裸列集**（OFT 单 center 列、LDB light+dark 列、ZM open+closed 列、EPM 双臂列）——证明新增概念**不污染**普通注入路径。
    2. **border-alias 路径（守 `binding=None` 的注入跳过，缺它则 None-skip 回归测不到）**：构造 `column_aliases={"<某外周物理列>": "border"}` 跑 OFT resolve。断言改动**前**（无 border 概念 → 无 border override）与改动**后**（有 border 概念但 `binding=None` → 注入循环 `:666` 顶端 `if rc.binding is None: continue` 命中 → 仍无 border override，且**不抛异常**）**结果都不产 border override**。**没有这条 fixture，§5.4 恒绿、完全测不到 `binding is None` 跳过是否生效**（假绿——正是 MEMORY 反复警告的哑故障）。
  - 实现方式：在改 yaml/loader **之前**先把当前 resolve 输出 dump 成 golden snapshot（`json.dumps(..., sort_keys=True)`，提交进测试 fixture），改完后 diff。可用公共入口 `resolve_metrics` / `resolve_charts` / `plan_metrics_to_dict`（参照 `tests/test_plan_metrics_interpretation_fields.py` 先例）采 golden。
  - **若此测试红**：说明误改了注入语义（最可能是注入循环没正确跳过 `binding=None` 的概念），必须修到字节相等。

### 5.5 聚合语义未引入（守 issue#98 边界）

- `test_no_aggregation_semantics_introduced`：断言新增概念条目**不携带任何 OR/sum/聚合配置字段**；`ResolvedZoneConcept` 仍只是 `(concept, binding, source)`（`binding` 为 `ParamBinding(param, wrap_list)` 或 `None`），无聚合算子。防止有人借"补概念"夹带 N:1 聚合。

---

## 6. 验收闸门（客观判据）

全部满足才算 Stage 3 完成：

1. **Stage 2 前置已核实**：`grep -rn "resolved_zone_concepts" packages/ethoinsight/ethoinsight/catalog/` 命中 Stage 2 产出（字段 + loader 填充）。
2. **新测试全绿**：`test_stage3_zone_concept_enumeration.py` 全部 pass（含 5.1–5.5；OFT border 存在性测试确定包含，因 border 确定补）。
3. **全量回归绿**：`cd packages/ethoinsight && pytest tests/` 全绿（特别确认 `test_metrics_ldb.py` / `test_metrics_zero_maze.py` / `test_metrics_oft.py` / 现有 catalog & resolve 测试 + Stage 2 的 `test_resolve_zone_overrides_equivalence.py` 无回归——本 stage 给 Stage 2 OFT 填充点加了一条 `binding=None` border 条目，需重跑 Stage 2 等价性测试确认 resolve 输出不变；若该测试断言了 `resolved_zone_concepts` 的精确内容，按需重采其 golden 把 border 纳入）。
4. **等价性证明**：5.4 字节等价测试通过——Stage 3 前后 resolve 输出零变化（含 border-alias 的 `binding=None` 注入跳过护栏）。
5. **裸导入无环**：`cd packages/ethoinsight && python -c "from ethoinsight.catalog import loader, resolve, schema"` 0 退出（改动全在 ethoinsight 包内，无 harness 闭环风险；本 stage 不碰 harness 侧）。
6. **三补集区改动符合逐范式取向**：LDB=`zone_concept_params.dark`（`binding=ParamBinding("dark_zone")`）、ZM=`zone_concept_params.closed`（`binding=ParamBinding("closed_zones")`）、OFT=Stage 2 填充点追加 `ResolvedZoneConcept(concept="border", binding=None, source="explicit_concept")`（不写 oft.yaml、不动 loader `:163`、不动 metrics）。
7. **无聚合语义泄漏**：grep 三个 yaml 的新增块，确认无 OR/sum/aggregate 字段。

---

## 7. 风险与铁律核对

### 7.1 本 stage 特有坑

- **`anonymous_zone_override` 绝不能动**：它承载两个正交关注点，关注点 (2)（`anonymous_zone_is` 统一输入键，softgate `resolve.py:744-790` + translate `:1136-1139`）是运行期契约，与本 stage 正交。Stage 3（LDB/ZM）只补 `zone_concept_params`，**override 块保持原样**。误删/误改会破坏匿名区软门。**OFT 同样不动 override 块**——border 不走 override 派生，而是 Stage 2 填充点直接追加一条 `binding=None` 概念条目（§3.4），与 override 块无关。
- **OFT border 是纯填充逻辑改动、零 loader/schema/metrics 改动**：border `binding=None`，不写 `zone_concept_params`，因此**不碰 loader `:163` 非空校验**（那是 `param` str 校验路径，border 根本不进）、**不改 `ResolvedZoneConcept`/`ParamBinding` 字段类型**（Stage 2 的 `binding` 本就可空）、**不改 metrics**。resolve 注入循环已由 Stage 2 加 `if rc.binding is None: continue`，border 进 dict 后被该 continue 跳过、零 override 输出，由 §5.4 border-alias fixture 守住。**不要因"想统一三范式"而强行给 OFT 起个假 param 名**（那等于偷偷改了 metrics 契约，且不诚实——border 本就无注入点）。
- **Stage 1 共存**：若 Stage 1（CNF 嵌套 requires_columns）已合，`resolve.py` 与 Stage 2 loader 的 `pat.startswith("in_zone")` flatten 循环已被 Stage 1/Stage 2 改造。**Stage 3 本身不碰 `requires_columns`、不新增任何 zone glob**，故与 Stage 1 的 flatten 悬空**无关**——三 stage 同 PR 合并时，新 flatten 消费者（Stage 2 loader 的 1b 迭代）的 flatten 处理归 Stage 2/Stage 1 协调，Stage 3 干净。仅记录供 lead 协调合并顺序。
- **派生字段填充时机（防御提醒，实测无害）**：LDB/ZM（`binding=ParamBinding`）依赖 Stage 2 的"全量纳入所有 zone_concept_params"逻辑自动吃下新条目。**实测 Stage 2 loader `:131-135` 是无条件 `for ... in zone_concept_params.items()`，结构上不可能只纳 EPM 两个 key**，故此处无哑故障——本提醒留作护栏，§5.1 存在性测试是最终兜底。

### 7.2 铁律逐条核对

- **deepseek 正面提示**（CLAUDE.md 第 6 条）：本 stage 不产出 prompt 文案，无"禁止/不要"风险。yaml description 字段若新增，用正面陈述（"暗区列名，用于穿梭计数"而非"不要填亮区"）。✅
- **SSOT 单存**（MEMORY [[feedback_single_source_of_truth]]）：本 stage 正是为 SSOT 服务——把今天散在 metrics 函数签名里的隐式概念（dark_zone/closed_zones）收口到 catalog 这唯一权威源，为 Stage 4 删 skill md 双写铺路。新增概念只在各自 yaml 一份。✅
- **不跨范式复用**（CLAUDE.md 第 14 条）：LDB dark / ZM closed / OFT border **各补各的**——LDB/ZM 各一个独立 `zone_concept_params` 块、OFT 一条独立 `binding=None` 填充条目，不因"都是补集区结构一样"合并成共享配置。§5.3 测试守此。✅
- **TDD 强制 red 先行**：§5 全部测试先写、先确认 red（尤其 §5.1 存在性 + §5.4 等价性 golden snapshot 在改 yaml/填充前 dump），再改 yaml/loader 填充。✅
- **无 harness import 闭环**（CLAUDE.md「harness 模块顶层 import 闭环风险」）：改动全在 `packages/ethoinsight/` 包内（yaml + Stage 2 OFT 填充点），无任何 `from deerflow ... import`，§6.5 裸导入守；harness 侧零消费（§4.2 已 grep 确认）。✅
- **守 issue#98 / CLAUDE.md 第 14 条**：只补"概念枚举结构"让其可被 HITL 对齐，**不实现 N:1 聚合语义**（OR/sum 待 Golden Case）。§5.5 测试守此。✅

---

## 8. 待决策点

### 决策 D1 — OFT `border` 的表面形态【✅ 已由 Fable 2026-06-11 决策门 1 闭合】

> **✅ 闭合结论（Fable 决策门 1，见 [[feedback_fable_pr115_stage_decisions_parambinding_optional_and_buildtime_gen]]）**：统一内部模型**应当容纳"可对齐但无注入点"的概念**（蕴含关系反向：可注入 ⊂ 可对齐；对齐的本体职责是认领/消解歧义，注入只是可选绑定）。Stage 2 已据此把 `ResolvedZoneConcept` 改为 **`binding: ParamBinding | None`**（Stage 2 spec §3 改动 1 已更新）。
>
> **OFT border 落地形态 = 一个 `binding=None` 的一等概念条目**：
> ```python
> ResolvedZoneConcept(concept="border", binding=None, source="explicit_concept")
> ```
> 这**优于下表所有原始选项**：比总纲 (b)（把 border 绑回 center_zone）更诚实地编码"border 存在、可被 HITL 认领、但无运行时注入点"；且因 Stage 2 的 `binding` 本就可空，**零额外 schema 放松**——A2 的 `param: null` 哨兵 + 放松 loader `:163` + 改 schema 字段类型**全部不需要**（被 binding 可空形态取代）。等价性天然成立：Stage 2 resolve 注入循环已加 `if rc.binding is None: continue`，border 进 `resolved_zone_concepts` 不污染任何现有注入输出。
>
> **本 stage OFT 落地动作**：在 Stage 2 loader 规范化段（或本 stage 的 OFT 填充点）对 OFT 追加一条 `ResolvedZoneConcept(concept="border", binding=None, source="explicit_concept")`。**不改 oft.yaml 的 `param`（无 param 可写）、不改 loader `:163`、不改 metrics**。§3.3、§5.1、§5.4 按此形态执行（§5.4 的 border-alias None-skip 护栏 fixture 仍需要——它正是守 `if rc.binding is None: continue` 的，None 来自 `binding=None`）。
>
> **配套 TODO（Fable 挖出的静默精度降级缺口，本 stage 不做，记入总纲 follow-up）**：`_find_periphery_zone_column`（`oft.py:50`）用固定英文 regex 匹**物理列名**；用户把中文列（如 `外周区`）对齐到 border 后、列名不中 regex → `compute_thigmotaxis_index` 静默降级到 `1−center`、**无视用户对齐的列**（静默精度降级，非崩溃）。干净终态=给 OFT 加真实 `periphery_zone` param + 脚本把注入列作第零优先级（先于 regex）；届时只把 border 条目的 `binding` 从 `None` 换成 `ParamBinding("periphery_zone")`——**`binding=None` 形态天然支持此升级，无需重新发明概念**。本 stage 明确不实现（一次只动一层）。

下表为**决策溯源**（Fable 裁决前的候选空间，已被上方 ✅ 结论取代）：

| 选项 | 对应总纲 | 做法 | 评注 |
|------|----------|------|------|
| **A1（推后）** | 总纲外 | 本 stage 不补 OFT border | 已不必：binding=None 形态零成本，无推后理由 |
| **总纲 (b)** | 总纲早期 (b) 路 | 保留 override，对 center 补集 derive 一个 border 条目，param 复用 center_zone | 被取代：把 border 绑回 center_zone 不如 binding=None 诚实 |
| **A2** | 总纲外 | oft.yaml 写 `param: null` + 放松 loader `:163` + 改 schema `param: str\|None` + resolve 跳过 None | **被取代**：Fable 的 `binding: ParamBinding\|None` 形态使 `param: null` 哨兵与 loader 放松全不需要 |
| **A3** | 总纲外 | 给 OFT metrics 加 `border_zone` compute 参数 | 仍出界（改 metrics 行为）；即上方 TODO 的干净终态，留独立 sprint |

### 决策 D2 — Stage 2 `ResolvedZoneConcept` 形态【✅ 已闭合，无悬空】

Fable 决策门 1 已定 Stage 2 用 `binding: ParamBinding | None`（Stage 2 spec §3 改动 1 已更新）。OFT border 走 `binding=None` 复用此形态，**无需任何 Stage 2 字段类型的额外改动**（原 A2 才需要的 `param: str|None` 放松已不存在）。Stage 3 落 OFT border 时，Stage 2 的 shape 已就位——本决策无悬空、无连带。

### 决策 D3 — OFT `corner` 概念的处置【本 stage 明确不做，记录避免实施 agent 自由裁量】

总纲 §1.4（catalog 不完整点表）明确"OFT 的 `corner` 同事说罕见 → **删**"。**`corner` 不在本 stage 范围**：本 stage 只补 border/dark/closed 三个"补集区"概念，不新增也不删 `corner`（`corner` 今天本就不在 catalog 任何结构化声明里，无可删之物）。若 Stage 4 生成菜单时发现 skill md 手写表残留 `corner`，归 Stage 4 删手写表时处理。**实施 agent 不要在本 stage 自行处置 corner。**
