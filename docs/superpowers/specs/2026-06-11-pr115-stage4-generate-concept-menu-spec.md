# Stage 4 — Q3 主体：从统一模型生成概念菜单，消除双存

> PR #115 catalog 概念整合系列 · 第 4/4 篇 stage spec
> 总纲/索引：`docs/superpowers/specs/2026-06-11-pr115-catalog-concept-consolidation-and-gate-cnf-spec.md`（本 stage 见 §2 索引表「Stage 4」行 + 依赖图「轨道 B 末尾」；构建期裁决见 §0「第二轮 决策门 2」）
> 本篇可被单个 agent 独立执行，仅覆盖 Stage 4，不实现其它 stage。

---

## 1. 目标与定位

**一句话目标**：column-confirmation skill 里手写的「范式 → 合法分析区概念」速查菜单不再人手维护，改为从 catalog 的统一内部概念模型（Stage 2 产出的 `resolved_zone_concepts`）单点生成，物理上消除「catalog vs skill md」双存与漂移。

**在总纲中的位置**：

- **依赖（必须先完成）**：
  - **Stage 2** — 已把 `zone_concept_params`（EPM）与 `anonymous_zone_override`（OFT/LDB/ZM 的 concept→param 映射）规范化进 catalog 的派生字段 `resolved_zone_concepts`（加载期由 loader 填充）。**Stage 4 的"统一模型"就是这个字段**，没有它就没有单一可生成源。值是 `ResolvedZoneConcept` dataclass、键是 concept 名（见 §5.1 契约）。
  - **Stage 3** — 已把 OFT `border`、LDB `dark`、ZM `closed` 这三个"补集区"概念补进 `resolved_zone_concepts`，**三者均为 Stage 3 确定产出**。其中 LDB `dark`、ZM `closed` 走 `binding=ParamBinding(真 param)`；OFT `border` 经 Fable 决策门 1 闭合为 `binding=None` 的一等概念（`ResolvedZoneConcept(concept="border", binding=None, source="explicit_concept")`，详见 §5.1）。本 stage 菜单**确定含 border**。
- **Stage 1**（Q1 CNF requires_columns 嵌套列表）与本 stage **无直接耦合**，两者可并行；本 stage 不读 Stage 1 的嵌套语义。
- **阻塞的后续**：无。Stage 4 是本系列末尾，完成后三问题闭环（Q3 消灭手写副本 → Q2 依赖倒置压力消失）。

---

## 2. 已核实事实锚点（带文件:行号，动笔依据）

### 2.1 双存现状（本 stage 要消除的目标物）

- **副本 A — SKILL.md 速查菜单**：`packages/agent/skills/custom/ethoinsight-column-confirmation/SKILL.md:24-28`（已核实，逐行精确）：
  ```
  24: - EPM: `open_arms` / `closed_arms` / `center`
  25: - OFT: `center` / `border` / `corner`
  26: - LDB: `light` / `dark`
  27: - Zero Maze: `open` / `closed`
  28: - FST: 无自定义分析区
  ```
  其后 SKILL.md:30 已声明"具体的合法概念以 catalog YAML 的 `requires_columns` glob 为权威，上面只是速查"——即文档自己承认 catalog 是权威、本表是可漂移副本。
- **副本 B — answer-mapping.md 速查菜单**：`packages/agent/skills/custom/ethoinsight-column-confirmation/references/answer-mapping.md:11-17`（已核实）。**整个表块 = 表头 :11-12 + 5 行数据 :13-17**：
  ```
  11: | paradigm | 分析区概念关键词 |
  12: |----------|----------------|
  13: | open_field | `center` / `border` / `corner` |
  14: | epm | `open_arms` / `closed_arms` / `center` |
  15: | light_dark_box | `light` / `dark` |
  16: | zero_maze | `open` / `closed` |
  17: | forced_swim | （无自定义分析区） |
  ```
  其后 answer-mapping.md:19-21 同样声明"概念关键词的权威来源是 catalog YAML 的 `requires_columns`；上表是速查"。

### 2.2 漂移实证（双存为什么必须消除）

- **EPM 多 `center`**：两份 md 都列 EPM `center`（SKILL.md:24 / answer-mapping.md:14），但 EPM 概念集应为 `open_arms`/`closed_arms`（catalog `zone_concept_params` 只声明这两个，见 §2.3）。`center` 是手写副本里的多余项。
- **OFT 多 `corner`**：两份 md 都列 OFT `corner`（SKILL.md:25 / answer-mapping.md:13），但 catalog 不为 OFT corner 提供可对齐概念（既无 `zone_concept_params.corner` 也无对应 derive）。`corner` 是手写副本里的多余项。
- 即两份手写副本当前**已经**与 catalog 真相不一致，"该删项 = EPM center + OFT corner"。

### 2.3 统一模型（Stage 4 的生成源，由 Stage 2/3 产出）

- Catalog dataclass：`packages/ethoinsight/ethoinsight/catalog/schema.py:131`（`@dataclass(frozen=True) class Catalog`），现有 zone 相关字段在 `:141` `anonymous_zone_override`、`:142` `zone_concept_params`。**Stage 2 在此新增派生字段 `resolved_zone_concepts: dict[str, ResolvedZoneConcept]`**（Stage 2 spec §4 改动 1，`schema.py` 新增 `ResolvedZoneConcept` dataclass + `Catalog` 加 `resolved_zone_concepts` 字段，`default_factory=dict`）。**本 stage 读它，不定义它——定义归 Stage 2**。
- `ResolvedZoneConcept` 形状（Stage 2 spec，经 Fable 决策门 1 闭合）：frozen dataclass，字段 `concept: str` / `binding: ParamBinding | None`（整体可空，非裸 `param`；OFT `border` 的 `binding is None`）/ `source: str`。**dict 的键 = concept 名**（`open_arms`/`closed_arms`/`center`/`light`/`open`/`border`/…），值 = 该 dataclass。`list_zone_concepts` 读的是**键集合**（concept 名），不读 `binding`。
- 加载期 derive 逻辑现存于 `resolve.py`：`_derive_concept_from_zone_patterns`（**定义在 resolve.py:690**；调用点在 `:603`/`:651`。实测：OFT `center_zone`→`center`、LDB `light_zone`→`light`、ZM `open_zones`→`open` 全对得上）。Stage 2 已把这段 derive 的**结果**前移到 loader、写进 `resolved_zone_concepts`，resolve 改为直接读它（Stage 2 spec §4 改动 3/4）。本 stage 不碰这段。

### 2.4 落地点（生成形态 = 构建期，已由 Fable 决策门 2 闭合）

- **构建期形态（采纳）**：生成器渲染独立的 `.generated.md` 文件 → 进常规构建路径产出 → 提交进 harness；CI staleness check 整文件字节比对兜底。staleness check 住 `packages/agent/backend/tests/`（该目录已有同类 catalog↔skill 对账测试 `test_column_semantics.py`、`test_metric_catalog_live.py`、`test_lead_agent_skills.py`）。生成器拥有整个生成文件、手写 SKILL.md/answer-mapping.md 只链接指向（详见 §3 决策门 2 与 §4.3/§4.4）。
- 渐进披露事实：column-confirmation skill 当前是 agent `read_file references/...md`（SKILL.md 多处指向 read_file md 实体），即知识以 md 实体形式被渐进披露消费——构建期生成的 `.generated.md` 沿用这条已证通道。
- 运行期 CLI 形态已评估、不采纳（见 §3 决策门 2 闭合结论）。

### 2.5 范式清单 SSOT（生成器遍历的范式来源，不得内联第三份）

- v0.1 全部 6 范式的**唯一权威清单** = `packages/ethoinsight/ethoinsight/catalog/loader.py:60-69` 的 `_PARADIGM_ALIASES`（academic 名 → filename stem）：`forced_swim→fst` / `tail_suspension→tst` / `open_field→oft` / `light_dark_box→ldb` / `epm→epm` / `zero_maze→zero_maze`（已核实）。对应 6 个 yaml：`epm/oft/ldb/fst/tst/zero_maze.yaml` 均存在。
- **今天仅 EPM/OFT/LDB/ZM 有 zone 概念**（`zone_concept_params` 或 `anonymous_zone_override`）；**FST 与 TST 都是空集范式**（无自定义分析区）。生成器遍历范式时**必须从 `_PARADIGM_ALIASES` 的键取清单**，禁止在 `concept_menu.py` 内联第三份范式 list（那会制造本 stage 宗旨要消灭的双存，见 §4.1 与 §8.1）。

---

## 3. 决策点（已闭合）

> ✅ **两决策门均已由 Fable（2026-06-11）闭合**，本 stage 直接按闭合路径执行：①OFT border 作为 `binding=None` 一等概念进菜单（决策门 1）；②走构建期生成 `.generated.md` + CI staleness check，不走运行期 CLI（决策门 2）。下方各小节保留闭合结论与决策溯源表（历史），但**正文执行指令只描述闭合路径**。

### 决策门 1：OFT border 在菜单中的形态【✅ Fable 2026-06-11 闭合 — binding=None 一等概念，菜单含 border】

> **✅ 闭合（Fable 决策门 1，见 [[feedback_fable_pr115_stage_decisions_parambinding_optional_and_buildtime_gen]]）**：统一内部模型 `ResolvedZoneConcept` 的 `binding` 字段整体可空（`binding: ParamBinding | None`，非裸 `param: str | None`）。OFT `border` = 一等概念条目 `ResolvedZoneConcept(concept="border", binding=None, source="explicit_concept")`，确定进 `resolved_zone_concepts`、确定进本 stage 菜单、§6 含 border 断言。
>
> 原"推后/不补 border"（旧 A1）与"oft.yaml 写 param:null + 放松 loader/schema + resolve 跳过 None"（旧 A2）两分支均作废：`binding` 本就可空，无需任何放松；把 border 绑回 center_zone 的方案也作废——`binding=None` 更诚实。
>
> resolve 注入循环已在 Stage 2 加 `if rc.binding is None: continue`（语义 = 无注入点概念不进 param 路由，非防御）。border（`binding=None`）进 `resolved_zone_concepts` 不污染任何现有 resolve 注入输出，等价性天然成立。

- **执行 agent 启动仍现场确认（防 Stage 3 未合）**：先 `load_catalog("oft")` 检查 `"border" in cat.resolved_zone_concepts`。
  - 命中（Stage 3 已合）→ 菜单含 border、测断言 border。这是预期路径。
  - **未命中** → 说明 **Stage 3 尚未合入 dev**（本 stage 的硬前置，见 §1）→ **停手，先确认 Stage 3 落地**，不要在 border 缺失的状态下生成菜单（否则生成"缺 border 的 OFT 菜单"= 把 Stage 3 没做完的事固化进生成物）。
- **生成器读 `concept` 名即可**——`border` 的 `binding=None` 对菜单生成透明（菜单只列概念名、不涉注入），无需对 None 分支特判（Fable 决策门 1：菜单生成器只用 `concept`，无 None 分支）。
- **绝不**因为"菜单该有 border"就回头改 `oft.yaml`——那是 Stage 3 的边界，撞 §8.1 自禁条款。本 stage 只生成"catalog 当前真相"。

> **静默精度降级 TODO（本 stage 不做，仅记）**：`_find_periphery_zone_column`（`oft.py:50`）固定英文 regex 匹物理列名 → 用户对齐的中文列（`外周区`）不中 → thigmotaxis 静默降级 1−center 无视对齐列。干净终态 = 给 OFT 加真实 `periphery_zone` param + 注入列作第零优先级；届时 border 的 `binding` 从 `None` 换 `ParamBinding` 即可。**记 TODO，本 stage 不实现、不改 oft.py、不改 border 的 binding 形态**。

### 决策门 2：运行期 CLI vs 构建期生成 md【✅ 已由 Fable 2026-06-11 闭合 — 构建期】

> **✅ 闭合结论（Fable 决策门 2）**：**走构建期生成 md + CI staleness check**，不走运行期 CLI。三条积极理由（比 spec 原"不值得改契约"的消极理由更硬）：
> 1. **变更节奏匹配**：工具应服务**运行时才变**的数据（每实验列/每线程状态）；概念菜单只随 catalog 变、catalog 随**代码版本**变 = 构建期节奏。随代码变的知识就该物化成随代码提交的 artifact。运行期 CLI 把版本静态知识包装成动态查询 = 节奏错配，不是"更彻底"。
> 2. **新失败面（援引项目史）**：本 harness 里"加一个工具/CLI 调用"不只是多一次往返——是 **sandbox 白名单 + guardrail 正则 + 工具注册三层**都要打通且都可能静默漏（chart-maker 的 `catalog.resolve`/`dump_headers` 从未进 bash guardrail 白名单、潜伏到 dogfood 才爆，见 MEMORY [[feedback_chart_maker_bash_guardrail_must_allow_resolve_dumpheaders]]）。`read_file md` 是已证通道。用新失败面换 CI 能守死的 staleness = 负交易。
> 3. **构建期不是沉没成本**：真到要"运行时按当前实验列动态过滤菜单"那天，生成器读统一模型产 md、CLI 读同一模型产 JSON = 同 SSOT 两 renderer，迁移是**加一个出口**不是推倒。
>
> **"绝不双存"达标判据已被 Fable 精确化**：双存的祸害不是两份字节，是**两个独立可编辑的源头**。精确判据 = **一个 artifact 是危险副本，当且仅当人能手改它且手改能静默存活**。生成 + CI staleness 后，md 任何手改在 merge 关口必红 → 它从"第二个源"降格为"带一致性证明的物化视图"（同 `uv.lock` / 生成的 protobuf，无人说违反 SSOT）。所以构建期**算达标"生成才是唯一正解"**（那条标准针对手写副本，不是物理副本）。运行期消除的残余（merge 前本地窗口期 stale）本就到不了主干，为它付改契约 + 新失败面 = 为零买单。

下表为决策溯源（已被上方 ✅ 取代）：

| 形态 | 机制 | 评注 |
|------|------|------|
| **运行期** | 加 `list_zone_concepts <paradigm>` CLI；md 删手写表改为"调 CLI"指引 | 被取代：节奏错配 + 三层新失败面，换 CI 能守的 staleness 不值 |
| **构建期（✅ 采纳）** | 生成器整文件产出独立 `.generated.md`、手写 md 链接指向；CI 校验"重新生成 == 已提交"整文件字节比对 | 契合现有 `read_file` 渐进披露、零交互改动；Q2 测试转世为 staleness check 住 harness CI、方向顺 |

> **Fable 的三条硬性落地细节（§4/§5/§6 必须遵守，否则构建期形态守不住）**：
> 1. **生成器拥有整个文件，不在手写 md 里嵌生成区块**。混合文件 = staleness 校验变糊 + 手改混进生成段的温床。
>    → **修订 §4.3/§4.4**：原方案用 `<!-- BEGIN/END generated -->` 哨兵在 SKILL.md / answer-mapping.md **内部**嵌生成块，**违反此条**。改为：**把概念菜单拆成独立的生成文件**（如 `references/zone-concepts.generated.md`，整文件由生成器拥有、checker 整文件字节比对），手写的 SKILL.md / answer-mapping.md 改为**链接/指向**该生成文件（"该范式合法概念见 `zone-concepts.generated.md`"），不再内嵌概念表。这样生成物与手写物**物理分离**，staleness check 是干净的整文件比对。
> 2. **staleness 红时必须带指令**：CI 校验失败信息必须含可执行修复指令（如"运行 `make gen-references` 后重新提交"），不能只报"文件过期"（本项目实证：信息完备但不含指令的拒绝改不动行为，见 MEMORY [[feedback_deny_messages_must_direct]]）。
> 3. **生成进常规构建路径，CI 只兜底不当主机制**：把生成挂进常规 `make` 目标（如 `make gen-references` 或随某个已有 build 目标顺带跑），让"忘了重新生成"在**本地**就被消化；CI 的 staleness check 是最后兜底，不是唯一防线。

### 决策 D2：菜单文本的"展示别名"归属

- 现 md 用范式简称（`EPM`/`OFT`/`LDB`/`Zero Maze`）与 paradigm key（`open_field`/`epm`/`light_dark_box`/`zero_maze`/`forced_swim`）两种写法：SKILL.md 用简称、answer-mapping.md 用 academic key。生成器需要一个 paradigm key → 人类可读标签的映射，且**两套 label 风格各需一份**（skill 简称风 + answer-mapping academic key 风）。
- **paradigm key 列表本身从 §2.5 的 `_PARADIGM_ALIASES` 取**（已是 academic key，answer-mapping 风直接用）。skill 简称风（`EPM`/`OFT`/`LDB`/`Zero Maze`/`FST`/`TST`）若 catalog 无现成展示名字段，则在生成器内联**一张最小 key→简称表**——这是纯展示标签、不构成"概念知识"双存，且该映射只此一处（若新增也只这一处）。
- **需确认**：catalog 是否已有 paradigm 展示名字段；若无，接受生成器内联这张最小展示标签表。

---

## 4. 改动清单（逐文件逐处）

> 按 **构建期生成**（决策门 2 闭合形态）展开。OFT border 一律含在菜单（决策门 1 闭合：`binding=None` 一等概念）。生成器只读 `concept` 名、对 `binding` 透明，无 None 分支。

### 4.1 新增：生成器模块 `packages/ethoinsight/ethoinsight/catalog/concept_menu.py`

- **新建纯函数**（无副作用、可被 harness 测试顺方向 import）：
  - `def list_zone_concepts(paradigm: str) -> list[str]`：`load_catalog(paradigm)` → 读 `cat.resolved_zone_concepts` 的**键集合**（concept 名）→ 返回该范式的概念关键词**有序列表**（去重、确定性排序——见 §8.1，确保生成可复现）。**空集范式（FST `forced_swim` 与 TST `tail_suspension`）返回 `[]`**（二者今天都无 zone 字段，见 §2.5）。
    - **binding=None 概念入列**：OFT `border` 的 `ResolvedZoneConcept.binding is None`（Stage 3 落地形态）。`list_zone_concepts` **按键集合枚举，不过滤 binding=None 的条目**——border 是可被 HITL 对齐的合法概念，必须出现在菜单里。菜单生成只读 `concept` 名，对 `binding` 是否 None 透明（Fable 决策门 1：菜单生成器无 None 分支）。
  - `def _supported_paradigms() -> list[str]`：从 `loader._PARADIGM_ALIASES`（§2.5 SSOT）取 academic key 列表，**禁止内联第三份范式清单**。生成器遍历范式时调它。
  - `def render_skill_list(paradigms: list[str]) -> str` / `def render_answer_mapping_table(paradigms: list[str]) -> str`（或一个带 `style` 参数的 `render_concept_menu_markdown`）：对每个 paradigm 调 `list_zone_concepts`，渲染成 md 片段。**skill 形态 = 列表行（简称 label）**、**answer-mapping 形态 = 表格行（academic key label）**。空概念集渲染为"无自定义分析区"。两套 label 来源见 §3 D2（academic key 取自 `_PARADIGM_ALIASES` 键，简称取自生成器内最小展示表）。
- **为什么放 ethoinsight 包内**：生成源是 catalog（library 内），生成器读 catalog，方向顺。harness 侧测试 `import` 它是 harness→library 顺方向、安全（守 CLAUDE.md import 闭环铁律）。

### 4.2 新增：生成入口（独立子模块，不碰承重的 resolve CLI）

- 给 `concept_menu.py` 加 `main()` + `if __name__ == "__main__"`，由 `python -m ethoinsight.catalog.concept_menu --style skill|answer-mapping` 驱动，stdout 输出渲染好的 md 片段（构建期）。
- **范式清单由 `_supported_paradigms()` 在内部取自 §2.5 SSOT，不暴露 `--paradigms` 手填清单**（避免调用方漏填某范式——如旧草案漏 TST——也避免再造一份范式 list）。若必须支持子集调试，`--paradigms` 默认值必须 = `_supported_paradigms()` 全集。
- **不动现有 `cli.py` 的 `_build_parser`（:77）**，避免污染承重的 resolve 解析路径。
- 运行期 CLI 入口形态（`list_zone_concepts <paradigm>` 输出 JSON）已评估、不采纳（决策门 2 闭合）。

### 4.3 改写：`SKILL.md` 概念表（删手写 → 链接到独立生成文件）

> **⚠️ Fable 决策门 2 硬性细节 #1 取代了原"哨兵块"方案**：原方案在 SKILL.md 内嵌 `<!-- BEGIN/END generated -->` 哨兵块，违反"生成器拥有整个文件、不在手写 md 嵌生成区块"（混合文件 = staleness 校验糊 + 手改混进生成段）。**改为生成物与手写物物理分离。**

- 手写文件：`packages/agent/skills/custom/ethoinsight-column-confirmation/SKILL.md`
- 生成文件（新建、整文件由生成器拥有）：`packages/agent/skills/custom/ethoinsight-column-confirmation/references/zone-concepts.generated.md`
- **删 `SKILL.md:24-28`** 手写菜单（含该删的 EPM `center`、OFT `corner`），替换为一句**指向生成文件的链接 + 正向措辞**（如"各范式合法分析区概念见 `references/zone-concepts.generated.md`——该菜单由 catalog 自动生成、与 SSOT 同源，按此对齐"）。
- **保留 :30** 的"以 catalog requires_columns glob 为权威"说明（升级为正向句）。
- **生成文件整文件由 `render_skill_list` 产出**；checker 对**整文件**做"重新生成 == 已提交"字节比对（不是块内比对）。OFT 菜单含 `center`/`border`、不含 `corner`（决策门 1 已闭合 border）。

### 4.4 改写：`answer-mapping.md` 概念表（删手写 → 链接到独立生成文件）

- 手写文件：`packages/agent/skills/custom/ethoinsight-column-confirmation/references/answer-mapping.md`
- 生成文件（新建、整文件由生成器拥有）：`packages/agent/skills/custom/ethoinsight-column-confirmation/references/zone-concepts-mapping.generated.md`（answer-mapping 风 = academic key 表格）
- **删 `answer-mapping.md:11-17`** 手写表（表头 :11-12 + 5 行数据 :13-17 整块，含该删的 EPM `center`、OFT `corner`），替换为指向生成文件的链接（同 §4.3 措辞模式）。**保留 :19-21** 权威来源说明（升级正向措辞）。
- 生成文件整文件由 `render_answer_mapping_table` 产出；checker 整文件字节比对。

> **两份生成文件的整文件归属**：`zone-concepts.generated.md`（skill 简称风）+ `zone-concepts-mapping.generated.md`（academic key 风）各由生成器整文件拥有，文件名后缀 `.generated.md` 是给人和 checker 的"勿手改"信号。手写的 SKILL.md / answer-mapping.md 只**链接**它们、不内嵌内容。

### 4.5 改动边界（明确不碰）

- **不碰** `resolve.py` 的 concept→param 注入逻辑、`_derive_concept_from_zone_patterns`（那是 Stage 2 的活）。
- **不碰** `schema.py` 的 `ResolvedZoneConcept` / `resolved_zone_concepts` 定义（Stage 2 定义、本 stage 只读）。
- **不补/不改** 任何范式的概念集内容（OFT border / LDB dark / ZM closed 的补全是 Stage 3 的活；OFT border 经决策门 1 闭合为 `binding=None` 一等概念、由 Stage 3 落地，**本 stage 绝不自行增删概念、不改 oft.yaml、不改 border 的 binding 形态**，启动时仅 `load_catalog("oft")` 现场确认 border 已在 dict 中）。
- **不实现** 任何 N:1 聚合语义（守 issue#98 / CLAUDE.md 第 14 条）。

---

## 5. 接口契约

### 5.1 输入契约（依赖上游 stage 的产出）

- **来自 Stage 2**：`Catalog` 实例有派生字段 `resolved_zone_concepts: dict[str, ResolvedZoneConcept]`，**键 = concept 名**，值 = `ResolvedZoneConcept(concept, binding, source)`（`binding: ParamBinding | None`，Fable 决策门 1 已闭合形态）。`list_zone_concepts(paradigm)` 读 `cat.resolved_zone_concepts.keys()`（concept 名集合），**不读 binding**。
  - `binding` 可空：OFT `border` 的 `binding is None`。本 stage 不依赖 binding 取值（只枚举键），故 binding=None **不影响** `list_zone_concepts` 的正确性——border 的**键**仍在 dict 里、仍被枚举入菜单（菜单生成器无 None 分支）。
  - **若 `resolved_zone_concepts` 字段不存在 → Stage 4 不可启动**（red 测试会立即暴露 `AttributeError`，这是正确的阻塞信号，说明 Stage 2 未完成）。
- **来自 Stage 3**：`resolved_zone_concepts` 对 LDB 含 `dark`、ZM 含 `closed`、OFT 含 `border`（三者均为 Stage 3 确定产出，OFT border 经 Fable 决策门 1 闭合为 `binding=None` 一等概念）。
  - LDB/ZM/OFT 缺 dark/closed/border → 说明 **Stage 3 未合入**（本 stage 硬前置）→ §6 覆盖度断言失败（正确阻塞信号）→ 停手确认 Stage 3 落地，不在缺概念状态下生成菜单。

### 5.2 输出契约（给下游 = 给 agent 运行时）

- **构建期（采纳形态）**：两份独立 `.generated.md` 由 CI 保证 `重新生成 == 已提交`（整文件字节比对）。agent 运行时消费形态**不变**（仍 `read_file` 拿 md——手写 SKILL.md/answer-mapping.md 链接指向生成文件）。下游契约 = "agent 读到的菜单 == catalog 真相"。
- 运行期 CLI 输出契约（`list_zone_concepts <paradigm>` → JSON 概念数组）已评估、不采纳（决策门 2 闭合）。
- 对**其它系统零影响**：本 stage 不改 resolve 输出、不改 plan.json schema、不改 `column_semantics` 字段语义、不改 `set_experiment_paradigm` 工具。

---

## 6. TDD（red 先行）

> 测试归属铁律：**菜单↔catalog 对账测试放 harness 侧 `packages/agent/backend/tests/`**（依赖方向 harness→library 顺方向）；**生成器纯函数单测放 `packages/ethoinsight/tests/`**（library 内单测自身逻辑）。**绝不**把"skill md 与 catalog 一致性"测试放 `ethoinsight/tests/`（那是依赖倒置）。

### 6.1 library 侧：生成器纯函数单测

文件：`packages/ethoinsight/tests/test_concept_menu.py`（新建）

- `test_list_zone_concepts_epm_no_center`：`list_zone_concepts("epm")` 返回含 `open_arms`/`closed_arms`、**不含** `center`。
- `test_list_zone_concepts_oft_has_border_no_corner`：`list_zone_concepts("oft")` 返回含 `center`、含 `border`（Stage 3 经 Fable 决策门 1 闭合为 `binding=None` 一等概念）、**不含** `corner`。
- `test_list_zone_concepts_ldb_has_dark` / `test_list_zone_concepts_zm_has_closed`：分别含 `dark` / `closed`（Stage 3 确定产出）。
- `test_list_zone_concepts_fst_empty`：`forced_swim` 返回 `[]`。
- `test_list_zone_concepts_tst_empty`：`tail_suspension` 返回 `[]`（**TST 也是空集范式，§2.5；旧草案漏此条**）。
- `test_list_zone_concepts_deterministic`：连续两次调用返回**字节相同**的有序列表（生成可复现的前提，守 §8.1 排序）。
- `test_supported_paradigms_matches_alias_map`：`_supported_paradigms()` 返回的范式集 == `loader._PARADIGM_ALIASES` 的键集（**坐实范式清单不内联第三份**，守 §2.5）。
- `test_render_markdown_skill_style` / `test_render_markdown_answer_mapping_style`：渲染输出含预期概念、不含 `center`(EPM)/`corner`(OFT)；空集（FST+TST）渲染为"无自定义分析区"；含 `border`(OFT，决策门 1 闭合)。
- 依赖关系标注：上述测试中 **OFT border、LDB dark、ZM closed 三条依赖 Stage 3 已合**；若执行 agent 在 Stage 3 未合时启动，这三条 red 是"依赖未满足"而非"待实现"——此时应**阻塞回报"Stage 3 依赖未满足"**，而非自己补概念。`epm_no_center`/`fst_empty`/`tst_empty`/`deterministic`/`supported_paradigms` 五条只依赖 Stage 2，是"函数从无到有"的纯 red→green。

### 6.2 harness 侧：staleness / 覆盖度对账（Q2 测试转世）

文件：`packages/agent/backend/tests/test_concept_menu_staleness.py`（新建）

- **staleness 对账（整文件字节比对，哨兵块方案作废→比对独立生成文件）**：
  - `test_skill_generated_file_matches_render`：读已提交的 `references/zone-concepts.generated.md` **整文件内容**，与 `render_skill_list(_supported_paradigms())` 重新生成结果**逐字符比对相等**。
  - `test_answer_mapping_generated_file_matches_render`：同上，对 `references/zone-concepts-mapping.generated.md` 整文件用 `render_answer_mapping_table`。
  - **这两条是漂移回归探针，不验证"生成内容是否正确"**：baseline（已提交的 `.generated.md`）本身就是生成器的输出（§6.3 实施顺序：先实现生成器 → 跑它产生成文件 → 提交），所以本测试等价于"生成器输出 == 生成器输出"，**只能抓"人手改生成文件而不改 catalog"的漂移，抓不到生成器自身 bug**。生成内容的正确性由 §6.1（概念集对不对）+ 下方覆盖度断言（该删/该有的项）共同承担。执行 agent 不要误以为 staleness 能兜生成正确性。
  - staleness 测试失败信息必须含可执行修复指令（决策门 2 硬性细节 #2：如"运行 `make gen-references` 后重新提交"），不能只报"过期"。
- **覆盖度断言（承担"内容正确"职责）**：
  - `test_generated_menu_covers_all_supported_paradigms`：生成菜单覆盖 `_supported_paradigms()` 全部 **6** 范式（epm/oft/ldb/fst/tst/zero_maze，含 FST **与 TST** 的空集表达），无遗漏。
  - `test_generated_menu_dropped_concepts`：断言 EPM 无 `center`、OFT 无 `corner`（坐实"该删项已删"），且 LDB 有 `dark`、ZM 有 `closed`、**OFT 有 `border`**（坐实 Stage 3 产出已被消费、决策门 1 闭合的 border 已入菜单）。
- 这些测试 `import ethoinsight.catalog.concept_menu` 是 harness→library 顺方向，安全。

### 6.3 red→green 总览

- 全部测试为新增。只依赖 Stage 2 的（`epm_no_center`/`fst_empty`/`tst_empty`/`deterministic`/`supported_paradigms`）初始 red（函数不存在），实现 §4.1 后 green。依赖 Stage 3 的（dark/closed/border）在 Stage 3 已合前提下随生成器实现转 green；未合则阻塞回报。
- 实施顺序：现场确认 §3 决策门 1（`load_catalog("oft")` 核实 border 已在 dict，防 Stage 3 未合）→ 先写 §6.1+§6.2 全 red（OFT 含 border）→ 实现 §4.1 生成器 → §6.1 转 green → 运行生成器整文件产出两份 `.generated.md`、按 §4.3/§4.4 把手写 SKILL.md/answer-mapping.md 改为链接指向 → §6.2 staleness 转 green。

---

## 7. 验收闸门（客观判据）

本 stage 完成当且仅当：

1. **测试全绿**：
   - `cd packages/ethoinsight && pytest tests/test_concept_menu.py` 全过。
   - `cd packages/agent/backend && source .venv/bin/activate && make test` 全过（至少新增的 `test_concept_menu_staleness.py` 全过，且不引入新红；注意区分 CLAUDE.md 记录的 4 个已知污染失败，非本 stage 引入）。
2. **双存物理消除**：`SKILL.md:24-28` 与 `answer-mapping.md:11-17` 的手写概念表已删，替换为指向独立生成文件（`zone-concepts.generated.md` / `zone-concepts-mapping.generated.md`）的链接。`grep -nE "corner" SKILL.md references/answer-mapping.md` 不再出现 OFT corner；EPM 行不再含 `center`。
3. **同源性证明**：重新运行生成器产出与已提交 `.generated.md` 逐字符相等（staleness check 绿 = 机械证明生成文件是 catalog 的投影、无人手漂移）。
4. **覆盖完整**：生成菜单覆盖全部 **6** 范式（含 FST + TST 空集），含 Stage 3 补的 `dark`/`closed`、含决策门 1 闭合的 OFT `border`、不含该删的 `center`/`corner`。
5. **裸导入安全**：`cd packages/ethoinsight && python -c "import ethoinsight.catalog.concept_menu"` 0 退出；harness 侧 `cd packages/agent/backend && PYTHONPATH=. python -c "import app.gateway"` 与 `python -c "from deerflow.agents import make_lead_agent"` 均 0 退出（本 stage 不碰 harness 核心模块，仅 test 文件 import library，理论无闭环风险，仍跑这两条确认）。
6. **resolve 零回归**：`cd packages/ethoinsight && pytest tests/ -k "resolve or catalog"` 全过（本 stage 不碰 resolve 路径，应零影响）。

---

## 8. 风险与铁律核对

### 8.1 本 stage 特有坑

- **生成不确定性 → staleness check flaky**：若 `resolved_zone_concepts` 是 dict 且迭代序不稳定，渲染会非确定性，staleness check 间歇红。**对策**：`list_zone_concepts` 内部对概念关键词显式排序（`test_list_zone_concepts_deterministic` 守这条）。
- **范式清单内联 = 自造双存**：生成器若硬编码一份 `["epm","oft",...]`，就是本 stage 要消灭的那类双存复发。**对策**：`_supported_paradigms()` 必须取自 `loader._PARADIGM_ALIASES`（§2.5），`test_supported_paradigms_matches_alias_map` 守这条。
- **漏 TST**：TST 与 FST 同为空集范式，极易只覆盖 FST（旧草案即漏 TST）。**对策**：§6.1 `test_list_zone_concepts_tst_empty` + §6.2 覆盖全 6 范式断言。
- **OFT border 含在菜单（决策门 1 已闭合，非未决契约）**：border 经 Fable 决策门 1 闭合为 `binding=None` 一等概念、确定进 `resolved_zone_concepts`、确定进菜单、§6 含 border 断言。执行 agent 启动只需 `load_catalog("oft")` 现场确认 border 已在 dict（防 Stage 3 未合）——命中即按预期推进；未命中说明 Stage 3 未合，**停手回报、不自行补 border、不改 oft.yaml**（撞下条自禁）。
- **抢 Stage 3 的活**：若启动时 OFT border / LDB dark / ZM closed 不在 `resolved_zone_concepts`，**不要在本 stage 顺手补概念或改 yaml**——三者均为 Stage 3 确定产出，任一缺失 = Stage 3 未合入（阻塞回报）。本 stage 只生成"catalog 当前真相"。
- **抢 Stage 2 的活**：`resolved_zone_concepts` 字段不存在时**不要在本 stage 自己 derive**——读不到就是 Stage 2 未完成的信号，阻塞回报。
- **binding=None 概念被误过滤**：OFT border 的 `binding is None`。`list_zone_concepts` 必须**按键枚举、对 binding 透明（无 None 分支）**，否则 border 拿不到。§4.1 已规定，`test_list_zone_concepts_oft_has_border_no_corner` 守这条。
- **生成文件按行号/局部替换破坏 md**：生成器若只改手写 md 的某几行（旧哨兵块方案的隐患），会随 md 编辑漂移。**对策**：概念菜单拆成独立 `.generated.md`、由生成器**整文件产出**、checker 整文件字节比对；手写 SKILL.md/answer-mapping.md 只链接指向、不内嵌概念表（决策门 2 硬性细节 #1，生成物与手写物物理分离）。

### 8.2 全局铁律逐条核对

- **deepseek 正面提示**：改写后的 SKILL.md / answer-mapping.md 措辞用正面指令（"各范式合法概念见生成文件，与 catalog 同源，按此对齐"），不写"不要手改此表"之类负面句；`.generated.md` 文件名后缀 + CI staleness check 本身是机械约束、不依赖 prompt 措辞。✅
- **SSOT 单存**：本 stage 的**唯一目的**就是消除概念菜单双存——生成后概念知识只剩 catalog `resolved_zone_concepts` 一份，独立 `.generated.md` 是其机械投影（由 CI 整文件比对保同源）；范式清单也单点取自 `_PARADIGM_ALIASES`，不内联第三份。✅
- **不跨范式复用**：生成器逐范式独立调 `list_zone_concepts(paradigm)`，LDB dark / ZM closed / OFT border 各从各自 catalog 取，绝不因结构相似合并或互相填充。✅
- **TDD 强制 red 先行**：§6 全部测试先写、初始 red（函数不存在 + md 仍含 center/corner），再实现转 green；staleness check 是永久漂移回归探针。✅
- **import 闭环零风险**：改动全在 `ethoinsight` 包内（新增 concept_menu.py + 子模块入口）+ 两份 skill md（非 Python）+ harness 侧测试文件；harness 测试 `import ethoinsight.catalog.concept_menu` 是 harness→library 顺方向，安全；不碰 `subagents/`、`agents/`、`tools/builtins/` 任何核心模块顶层 import。✅
- **issue#98 / 第14条边界**：只生成"概念枚举"菜单，不实现任何 N:1 聚合语义；OFT border 即便入菜单也只是"可对齐概念"枚举，不触碰 metrics 注入。✅
