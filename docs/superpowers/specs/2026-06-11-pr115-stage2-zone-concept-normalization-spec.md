# Stage 2 — Q3 前置：concept→param 统一内部模型（加载期规范化）

> 本篇是 [2026-06-11-pr115-catalog-concept-consolidation-and-gate-cnf-spec.md](2026-06-11-pr115-catalog-concept-consolidation-and-gate-cnf-spec.md)（总纲/索引 spec：本 stage 见其 §2 索引表「Stage 2」行 + 依赖图「轨道 B 承重墙」；无损规范化的事实依据见 §1.2/§1.3）拆出的**可独立执行 stage spec**。只覆盖 Stage 2，不实现 Stage 1/3/4 的任何代码。

---

## 1. 目标与定位

**目标（一句话）**：在 `loader` 加载期把 `zone_concept_params` 与 `anonymous_zone_override` 的**关注点 (1)（concept→param 映射）** 无损规范化进 `Catalog` 上一个新的派生字段 `resolved_zone_concepts`，让 `resolve.py` 只读这个统一模型、删掉它自己重复跑 derive 的分支——且 Stage 2 前后所有 resolve 输出**逐字节不变**。

**在总纲中的位置**：

- **依赖（上游 stage）**：**无硬代码依赖**。Stage 1（requires_columns 升 CNF）与 Stage 2 改的是**同一对文件的不同行区**，不重叠（核实见下表）：

  | | Stage 1 改动行区 | Stage 2 改动行区 |
  |---|---|---|
  | `loader.py` | 三处 `isinstance(c, str)` 校验（`:286` / `:354` / `:458`） | Catalog 构造区（`:179-189`）+ 新增规范化段（插在 `:177` 与 `:179` 之间） |
  | `resolve.py` | `pat.startswith("in_zone")` 的 flatten 消费者（`:600` / `:617`） | `_build_zone_aliases_overrides` 的 Step 1 builder（`:582-607`） |
  | `schema.py` | `requires_columns` 字段类型 | 新增 `ResolvedZoneConcept` + `Catalog.resolved_zone_concepts` 字段 |

  两 stage **触及同一文件但行区不重叠**，可独立合入。本 spec 假设在 dev 上独立实施，**不读、不依赖 Stage 1 的 `requires_columns` 嵌套 list**——Stage 2 收集 zone_patterns 时遇到的 `requires_columns` 仍按当前生产形态（纯 list-of-str）处理（见 §7 风险 R3 的耦合点与实施第一步 grep）。
- **阻塞（下游 stage）**：**Stage 3** 在本 stage 产出的 `resolved_zone_concepts` 模型上**追加** border/dark/closed 概念条目（catalog 显式声明后纳入同一 dict）；**Stage 4** 的概念菜单生成器直接消费 `resolved_zone_concepts` 作为唯一数据源。Stage 2 是这两者的承重墙——它定义统一内部模型的**形状**（dataclass + loader 填充契约），Stage 3/4 只往里加内容、读内容，不改形状。

---

## 2. 已核实事实锚点（带文件:行号，2026-06-11 实读 + 实跑确认）

锚点路径前缀统一为 `packages/ethoinsight/ethoinsight/catalog/`（下文简称 catalog/）。所有行号已逐处实读核对；§2.3 / §2.5 的 derive 结果与 golden 形态为**实跑**所得。

### 2.1 当前两机制与 YAML 表面形态

- **`zone_concept_params`**（dict `concept → ZoneConceptParam{param, wrap_list}`，仅 EPM 用）
  - dataclass：`catalog/schema.py:119-127`（`ZoneConceptParam`，`@dataclass` 装饰器在 `:118`）。
  - `Catalog` 字段：`catalog/schema.py:142`（`zone_concept_params: dict[str, ZoneConceptParam] = field(default_factory=dict)`）。
  - loader 解析：`catalog/loader.py:154-177`（逐条校验 param 非空 str + wrap_list bool）。
  - YAML 实例（仅 `epm.yaml:215-221`）：
    ```yaml
    zone_concept_params:
      open_arms:   { param: open_arm_zones,   wrap_list: true }
      closed_arms: { param: closed_arm_zones, wrap_list: true }
    ```

- **`anonymous_zone_override`**（`AnonymousZoneOverride{target_param, wrap_list}`，OFT/LDB/ZM 用）
  - dataclass：`catalog/schema.py:103-115`（`AnonymousZoneOverride`，`@dataclass` 装饰器在 `:102`）。
  - `Catalog` 字段：`catalog/schema.py:141`（`anonymous_zone_override: AnonymousZoneOverride | None = None`）。
  - loader 解析：`catalog/loader.py:128-152`（注释 `:128`，`AnonymousZoneOverride(...)` 构造 `:149-152`）。
  - YAML 实例：`oft.yaml:6-8`（`target_param: center_zone`, `wrap_list: false`）、`ldb.yaml:5-7`（`target_param: light_zone`, `wrap_list: false`）、`zero_maze.yaml:6-8`（`target_param: open_zones`, `wrap_list: true`）。

- **`Catalog` dataclass 定义**：`catalog/schema.py:131-142`（`@dataclass(frozen=True)` 装饰器在 `:130`，两个 zone 字段在末尾 `:141`/`:142`，均带 default）。
- **loader 构造 `Catalog(...)` 处**：`catalog/loader.py:179-189`，其中 `default_metrics=`（`:182`）/ `optional_metrics=`（`:183`）/ `charts=`（`:184`）/ `anonymous_zone_override=`（`:187`）/ `zone_concept_params=`（`:188`）。这三个 metrics/charts 列表与两个 zone 字段在 `return` 时均为 `_parse_catalog` 局部变量（构造在 `_parse_catalog` 函数体内，非 `load_catalog`）。

### 2.2 anonymous_zone_override 的两个正交关注点（核心动机）

`anonymous_zone_override` 承载两件正交的事：

- **关注点 (1) concept→param 映射**：概念名由 `_derive_concept_from_zone_patterns` 从 `in_zone*` glob 反推（**确定性**）。derive 实现在 `catalog/resolve.py:690-704`（其依赖 `_extract_concept_keyword` 在 `:682-687`）；它被两处调用：`resolve.py:603-605`（1b fallback 路径，给 azo.target_param 推 concept）和 `resolve.py:651-653`（Step 3 匹配路径）。
- **关注点 (2) `anonymous_zone_is` 统一输入键**（运行期契约，与 SSOT 正交）：softgate `_detect_anonymous_zone`（定义在 `resolve.py:740`，区段约 `:740-787`）+ translate（`resolve.py:1137-1138`）。**本 stage 完全不动关注点 (2)**。

### 2.3 无损规范化结论（实测 derive 对得上）

`_derive_concept_from_zone_patterns(zone_patterns, target_param)`（`resolve.py:690-704`）逻辑 = 把 target_param 去掉 `_zones`/`_zone` 后缀得候选，再在 zone_patterns 提取的 keyword 里找包含关系。实跑三范式：

| 范式 | target_param | in_zone glob | derive 得 concept |
|------|--------------|--------------|-------------------|
| OFT  | `center_zone` | `in_zone_center_*` | `center` |
| LDB  | `light_zone`  | `in_zone_light*`   | `light`  |
| ZM   | `open_zones`  | `in_zone_open*`    | `open`   |

→ 关注点 (1) **能无损合进统一内部模型**：loader 加载期跑同一 derive，结果与 resolve 运行期跑 derive 一致。

### 2.4 现 resolve 端重复 derive 的分支（本 stage 要删/改的点）

`_build_zone_aliases_overrides`（`resolve.py:566-679`，唯一生产调用方 `resolve.py:202`）Step 1 构建 `concept_param_map`：

- `resolve.py:583`：`concept_param_map: dict[str, tuple[str, bool]] = {}` 声明。
- `resolve.py:585-587`（**1a**）：把每条 `zone_concept_params` 加入 map。
- `resolve.py:589-607`（**1b**）：对 `anonymous_zone_override`，当其 `target_param` 未被 1a 占用时，**在 resolve 内部现场收集 zone_patterns（`:596-601`，含 `pat.startswith("in_zone")` 于 `:600`）+ 跑 `_derive_concept_from_zone_patterns`（`:603-605`）** 推 concept 并写入 map（`:607`）。`azo = cat.anonymous_zone_override` 赋值在 `:591`，`occupied_params` 计算在 `:593`。

这段 1a+1b 正是「loader 与 resolve 两处各跑一次 derive 的隐性双存」。Stage 2 把 1a+1b 的**结果**前移到 loader 的 `resolved_zone_concepts`，resolve 改为直接读它。

> **可前移的根因（实证）**：`concept_param_map` 的构建（`:583-607`）**只依赖 `cat`**（catalog-static），**不依赖** `_build_zone_aliases_overrides` 的另两个运行期入参 `column_aliases` 与 `existing_overrides`——后两者只在 Step 3（`:623-662`）与 Step 4&5（`:664-679`）才参与。正因为 Step 1 是纯 catalog-static，把它整段搬到加载期（catalog 一旦加载即定）**语义无损**。这是 §5 字节等价成立的根因。

> 注：`resolve.py:648-656`（Step 3 内 `matched_concept` 为 None 时再调 derive 的额外分支，`if` 在 `:648`、derive 调用在 `:651-653`）与 `:612-621`（Step 2 收集 zone_patterns 供物理列匹配，`pat.startswith` 在 `:617`）**不是 concept_param_map 的构建源**，属运行期物理列→concept 路由逻辑，**本 stage 保留不动**（见 §3 改动 4 的边界说明）。

### 2.5 derive helper 的包内可达性陷阱（实施前必读，决定改动 2/3 的 import 写法）

**实证结论（2026-06-11 实跑）**：`catalog/__init__.py:22` 执行 `from ethoinsight.catalog.resolve import ... resolve ...`，把**函数 `resolve`** 绑进 `ethoinsight.catalog` 包命名空间，**遮蔽了同名子模块 `resolve`**。后果——以下两种「取模块」写法都**取到函数、不是模块**，访问 `_derive_concept_from_zone_patterns` 会 `AttributeError`：

- `from ethoinsight.catalog import resolve` → `resolve` 是 `<function>`（实跑 `hasattr(resolve, '_derive_concept_from_zone_patterns') == False`）。
- `import ethoinsight.catalog.resolve as m` → 包 `__init__` 跑完后 `m` 绑到属性 `ethoinsight.catalog.resolve`（=函数），同样取到函数（实跑 `type(m).__name__ == 'function'`）。

**唯一可靠的访问写法（实跑确认 `True`）**：

- **（推荐）函数名直接 import**：`from ethoinsight.catalog.resolve import _derive_concept_from_zone_patterns, _extract_concept_keyword`——绕过包属性遮蔽，直接绑函数对象。
- 或 `importlib.import_module('ethoinsight.catalog.resolve')`——返回真模块。

**且这条 import 必须放在函数体内（惰性）**，因为 `resolve.py:23` 顶层 `from ethoinsight.catalog.loader import ...`（实证：resolve→loader 是已存在的单向边）。若 loader 在**顶层**反向 import resolve，则闭成 `partially initialized module` 环。loader 当前**无**任何顶层 resolve import（实证 grep 为空），必须保持这个不变量。

> 这一节直接推翻草案与初版 review 的两个错误判断：① 草案的 `from ethoinsight.catalog import resolve` 写法是坏的；② review 提议的 `import ethoinsight.catalog.resolve as m` **也是坏的**。下文改动 3 与 §8 D1 已按实证写死正确写法。

---

## 3. 改动清单（逐文件逐处）

> 全部改动落在 `packages/ethoinsight/` 包内。loader↔resolve 共享 derive 走**惰性函数名 import**（见 §2.5），无 harness import 闭环风险。

### 改动 1 — `catalog/schema.py`：新增 `ResolvedZoneConcept` dataclass + `Catalog` 加派生字段

**1a. 新增 dataclass**（建议紧邻 `ZoneConceptParam`（`:119-127`）之后、`Catalog`（`:130`）之前，约 `schema.py:128` 空行处）：

```python
@dataclass(frozen=True)
class ParamBinding:
    """概念的运行时注入绑定（param 与 wrap_list 同生共死）。"""
    param: str
    wrap_list: bool = False


@dataclass(frozen=True)
class ResolvedZoneConcept:
    """统一内部 concept 模型（加载期规范化产物）。

    模型本体语义 = 「对齐目标 + 可选的注入绑定」，**不是「注入参数表」**：
    每个可注入概念必须可对齐，但不是每个可对齐概念必须可注入
    （Fable 2026-06-11 决策门 1：见 [[feedback_fable_pr115_stage_decisions_parambinding_optional_and_buildtime_gen]]）。

    binding=None 表示「可被 HITL 对齐/认领（消解歧义），但无运行时注入点」——
    Stage 3 的 OFT border 即此态（脚本靠 regex 自动识别 + 三级降级，不吃注入）。
    用 ParamBinding | None **整体可空**（非裸 param: str | None），让非法状态
    （param=None 但 wrap_list 有值）不可表达。

    来源三态（仅记录，不影响消费）：
      - "zone_concept_params": 直接来自 cat.zone_concept_params（EPM）
      - "anonymous_zone_override": 由 _derive_concept_from_zone_patterns 规范化（OFT/LDB/ZM）
      - "explicit_concept": Stage 3 catalog 显式声明的补集概念（border/dark/closed）
    """
    concept: str
    binding: ParamBinding | None = None
    source: str = "zone_concept_params"  # 三态枚举，便于调试与 Stage 4 菜单标注
```

> **形态决策已闭合（Fable 决策门 1）**：原草案用 `param: str`（非空）—— 这把「碰巧今天每条都有 param」误读成了不变式。Fable 裁决：对齐的本体职责是**认领/消解歧义**（对所有概念成立），注入只是**部分**概念额外的可选绑定。故用 `binding: ParamBinding | None`。**不拆成「可注入」「仅识别」两个并行 dict**——那会在统一模型内部再造小型双存。`None` 只在**注入点一处**需要 `if binding is not None`（语义本身，非防御）；菜单生成器只用 `concept`、HITL 认领只记 `concept`、Q1 门控引用 `param`（无 binding 的概念自然不出现），三处都无 None 分支。
> `source` 字段为 Stage 3/4 预留来源标记（与 MEMORY「来源标记可观测」一致），Stage 2 只产出前两种来源。**Stage 2 不实现 `explicit_concept` 分支**——那是 Stage 3 的工作（边界，见 §5 越界守护）。

**1b. `Catalog` 加派生字段**（`schema.py:142` 之后追加为最后一个字段）：

```python
    resolved_zone_concepts: dict[str, ResolvedZoneConcept] = field(default_factory=dict)
```

放在末尾、带 `default_factory=dict`，保证所有现有 `Catalog(...)` 关键字构造与测试构造**向后兼容**（不传则空 dict）。

### 改动 2 — derive helper 的可共享位置（loader 加载期需调用）

`_derive_concept_from_zone_patterns`（含其依赖 `_extract_concept_keyword`，`resolve.py:682-704`）当前在 `resolve.py`。loader 需要在加载期调用同一函数以保证「loader 跑的 derive == resolve 旧逻辑跑的 derive」（无损前提）。

**取向（默认，见 §8 D1）**：loader **惰性、按函数名** import（不抽函数、不挪位置）：

```python
from ethoinsight.catalog.resolve import _derive_concept_from_zone_patterns  # 函数体内惰性 import
```

理由：

- 复用现成函数 = 字节级保证同一 derive（避免「抽函数时手滑改一行 → 漂移」）。符合 MEMORY「只复用不自造」纪律。
- **必须用「函数名直接 import」而非「import 模块」**：包 `__init__.py:22` 已用同名函数 `resolve` 遮蔽 `resolve` 子模块（§2.5 实证），`from ethoinsight.catalog import resolve` 与 `import ethoinsight.catalog.resolve as m` 都会取到函数、`AttributeError`。
- **必须惰性放函数体**：`resolve.py:23` 顶层 import loader，loader 顶层反向 import resolve 会闭环（§2.5）。

**备选（D1 选项二，仅当团队倾向消除遮蔽脆弱性时主动选）**：把 `_derive_concept_from_zone_patterns` + `_extract_concept_keyword`（`resolve.py:682-704`）**整体下沉**到无依赖小模块 `catalog/_zone_concepts.py`，loader 与 resolve 都从该模块 import（**函数体字节不变，只挪位置**）。此选项不受包属性遮蔽影响、可顶层 import，是更稳的结构，但带搬动成本。决策见 §8 D1。

### 改动 3 — `catalog/loader.py`：构造 `resolved_zone_concepts` 并填入 `Catalog`

在 `loader.py:179`（`return Catalog(...)`）**之前**、已解析出 `zone_concept_params`（`:154-177`）与 `anonymous_zone_override`（`:128-152`）之后（即插在 `:177` 与 `:179` 之间），新增一段规范化构建（与 `resolve.py:583-607` 的 1a+1b **逻辑等价**，只是搬到加载期）：

```python
    # 规范化：把 zone_concept_params + anonymous_zone_override(关注点1) 合进统一内部模型。
    # 与 resolve._build_zone_aliases_overrides Step 1(1a+1b) 逻辑等价，提前到加载期。
    # 惰性 + 按函数名 import：__init__.py 用同名函数遮蔽了 resolve 子模块（见 spec §2.5），
    # 且 resolve.py 顶层 import 本 loader——顶层/取模块两种写法都会炸，只此写法可靠。
    from ethoinsight.catalog.resolve import _derive_concept_from_zone_patterns

    resolved_zone_concepts: dict[str, ResolvedZoneConcept] = {}
    # 1a: zone_concept_params 每条直接纳入（这些概念都有注入绑定）
    for concept_key, zcp in zone_concept_params.items():
        resolved_zone_concepts[concept_key] = ResolvedZoneConcept(
            concept=concept_key,
            binding=ParamBinding(param=zcp.param, wrap_list=zcp.wrap_list),
            source="zone_concept_params",
        )
    # 1b: anonymous_zone_override —— target_param 未被 1a 占用时，derive concept 并纳入
    if anonymous_zone_override is not None:
        occupied = {
            rc.binding.param for rc in resolved_zone_concepts.values() if rc.binding is not None
        }
        if anonymous_zone_override.target_param not in occupied:
            zone_patterns: set[str] = set()
            entries = list(default_metrics) + list(optional_metrics) + list(charts)
            for entry in entries:
                for pat in getattr(entry, "requires_columns", []) or []:
                    if pat.startswith("in_zone") and "*" in pat:
                        zone_patterns.add(pat)
            azo_concept = _derive_concept_from_zone_patterns(
                zone_patterns, anonymous_zone_override.target_param
            )
            if azo_concept and azo_concept not in resolved_zone_concepts:
                resolved_zone_concepts[azo_concept] = ResolvedZoneConcept(
                    concept=azo_concept,
                    binding=ParamBinding(
                        param=anonymous_zone_override.target_param,
                        wrap_list=anonymous_zone_override.wrap_list,
                    ),
                    source="anonymous_zone_override",
                )
```

并在 `Catalog(...)`（`:179-189`）追加：

```python
        resolved_zone_concepts=resolved_zone_concepts,
```

> **R1 关键约束**：这段 1b 收集 zone_patterns 的迭代必须与 `resolve.py:596-601` **逐字节同构**——`entries = list(default_metrics) + list(optional_metrics) + list(charts)`（resolve 用 `cat.default_metrics` 等，loader 用同名局部变量，元素相同）、`getattr(entry, "requires_columns", []) or []` 的空值兜底、`pat.startswith("in_zone") and "*" in pat`。任何差异都会破坏 §5 的等价性回归。**实施时并排 diff 两段**。
>
> **occupied 判据等价**：loader 用 `{rc.param for rc in resolved_zone_concepts.values()}`，与 resolve 原 `{p for p, _ in concept_param_map.values()}`（`:593`）等价（都是「已占用的 param 集」）。

### 改动 4 — `catalog/resolve.py`：`_build_zone_aliases_overrides` 改读 `resolved_zone_concepts`，删 1a+1b 现场 derive

把 `resolve.py:582-607`（Step 1 的声明 + 1a + 1b）替换为**直接读 `cat.resolved_zone_concepts`**：

```python
    # ── Step 1: 从 catalog 统一内部模型构建 (concept_keyword → param, wrap_list) 映射 ──
    concept_param_map: dict[str, tuple[str, bool]] = {}
    for concept_key, rc in cat.resolved_zone_concepts.items():
        if rc.binding is None:
            continue  # 无注入绑定的概念（如 Stage 3 OFT border）不进 param 路由 —— 语义本身
        concept_param_map[concept_key] = (rc.binding.param, rc.binding.wrap_list)

    azo = cat.anonymous_zone_override  # 仍保留：Step 3 :648-656 物理列路由分支引用 azo
```

> **`if rc.binding is None: continue` 是语义而非防御**：无注入点的概念本就不该出现在「物理列→param 注入」的路由里（注入无目标）。这是 Fable 决策门 1 说的「None 只在与注入相关的点出现」的体现。Stage 2 自身产出的条目 binding 都非 None（EPM/OFT/LDB/ZM 都有 param），此 `continue` 在 Stage 2 阶段不会触发——它是为 **Stage 3 引入 binding=None 的 border** 预留的、且必须现在就写对（否则 Stage 3 落地时 `rc.binding.param` 在 None 上抛 `AttributeError`）。

**精确边界**：

- **删掉**：`:585-587`（1a 循环）+ `:589-607`（1b：注释 + `occupied_params` 计算 + 现场收集 zone_patterns + 调 derive + 写入 map）。`:583` 的 `concept_param_map` 声明改为上面新循环的形式。
- **保留** `azo = cat.anonymous_zone_override`（原 `:591` 的赋值要留，因为 `:648` 等后续分支引用 `azo`）——把它移到新 Step 1 末尾即可（如上）。
- **保留不动** Step 2（`:612-621` 收集 zone_patterns 供物理列匹配）、Step 3（`:623-662`，含 `:648-656` 的 derive 分支）、Step 4&5（`:664-679`）。这些是运行期物理列→concept 路由，不是 concept_param_map 的**构建源**，本 stage 不碰。

> 结果：resolve 不再在 Step 1 现场跑 derive（那次 derive 的结果已被 loader 预先算进 `resolved_zone_concepts`），消除两处各跑一次 derive 的双存。`concept_param_map` 的**内容**应与改动前逐键逐值相等（这是 §5 等价性回归要锁的）。

---

## 4. 接口契约

### 4.1 输入契约（本 stage 对上游的依赖）

- **对 Stage 1**：无依赖。本 stage 读 `entry.requires_columns` 时按**当前生产形态（纯 list-of-str）** 处理（`pat.startswith(...)`）。若实施时 dev 已先合入 Stage 1（requires_columns 升 CNF 嵌套 list），见 §7 R3 的兼容处理——但本 spec 默认两 stage 独立、互不假设对方已落地。
- **对现有 catalog YAML**：依赖 `zone_concept_params`（epm.yaml:215）/ `anonymous_zone_override`（oft/ldb/zero_maze）的现有形态不变。本 stage **不改任何 YAML**。

### 4.2 输出契约（本 stage 给下游的产出）

给 **Stage 3 / Stage 4** 的稳定契约：

1. **`ResolvedZoneConcept` dataclass**（`schema.py`）：字段 `concept: str` / `binding: ParamBinding | None` / `source: str`，frozen（`ParamBinding(param: str, wrap_list: bool)` 同生共死；`binding=None` = 可对齐但无注入点，Fable 决策门 1）。Stage 3 追加条目（LDB `dark`/ZM `closed` 用 `binding=ParamBinding(...)`、OFT `border` 用 `binding=None`，`source="explicit_concept"`），Stage 4 读全部条目的 `concept` 名（不读 binding）。
2. **`Catalog.resolved_zone_concepts: dict[str, ResolvedZoneConcept]`**：键 = concept 名（如 `open_arms`/`closed_arms`/`center`/`light`/`open`），值 = `ResolvedZoneConcept`。loader 加载期填充。**这是 Stage 3 追加概念、Stage 4 生成菜单的唯一数据源**。
3. **derive 共享点**：`_derive_concept_from_zone_patterns` 在 loader 与 resolve 间共享（访问写法由 §2.5 锁定、位置由 §8 D1 决定）。Stage 3 若需对补集概念跑 derive，复用同一函数。
4. **不变量**：`anonymous_zone_override` 在 `Catalog` 上**仍存在且语义不变**（关注点 (2)）；resolve 的 softgate / translate 路径不受影响。

---

## 5. TDD（red 先行）

> CLAUDE.md「TDD 强制」+ MEMORY「TDD 强制 red 先行」。先写测试、确认红、再实现到绿。测试放 `packages/ethoinsight/tests/`。范式名用学术名 `open_field`/`light_dark_box`（实证 `_PARADIGM_ALIASES` 在 `loader.py:64-65` 把 `open_field→oft`、`light_dark_box→ldb`，与文件名 `oft.yaml`/`ldb.yaml` 经 alias 互通，`load_catalog("open_field")` 有效）。

### 测试文件 1：`tests/test_resolved_zone_concepts.py`（新建）

针对统一内部模型内容。fixture 用真实 catalog（`load_catalog("epm"|"open_field"|"light_dark_box"|"zero_maze")`，参照 `tests/test_catalog_loader_aliases.py` 的真实加载模式）。

| 测试 | 断言 | red→green |
|------|------|-----------|
| `test_epm_resolved_from_zone_concept_params` | `load_catalog("epm").resolved_zone_concepts` 含 `open_arms→(open_arm_zones, wrap_list=True, source="zone_concept_params")` 与 `closed_arms→(closed_arm_zones, True, ...)`；恰 2 条 | red（字段不存在 AttributeError）→ green |
| `test_oft_resolved_from_override` | OFT 含 1 条 `center→(center_zone, wrap_list=False, source="anonymous_zone_override")` | red→green |
| `test_ldb_resolved_from_override` | LDB 含 1 条 `light→(light_zone, False, "anonymous_zone_override")` | red→green |
| `test_zm_resolved_from_override` | ZM 含 1 条 `open→(open_zones, wrap_list=True, "anonymous_zone_override")` | red→green |
| `test_resolved_concept_param_matches_legacy_derive` | 对 OFT/LDB/ZM，`resolved_zone_concepts` 里 override 来源条目的 `concept`，等于直接对该范式跑 `_derive_concept_from_zone_patterns(<现场收集的 zone_patterns>, target_param)` 的返回值（锁 derive 同源）。import 该函数走 `from ethoinsight.catalog.resolve import _derive_concept_from_zone_patterns`（§2.5） | red→green |
| `test_resolved_default_empty_for_no_zone_paradigm` | 对无 zone 字段的范式（如 `fst`/`tst`）`resolved_zone_concepts == {}` | 可能 green（default_factory）——作回归护栏 |

### 测试文件 2：`tests/test_resolve_zone_overrides_equivalence.py`（新建，**字节等价回归 — Stage 2 验收核心**）

证明规范化无损：Stage 2 前后 `_build_zone_aliases_overrides` 的输出对所有真实输入逐字节相等。

实现方式（**golden 快照**，因为改动后无法再调「旧版函数」对比）：

1. **在改任何源码前**（red 阶段第一步），对每个 zone 范式（EPM/OFT/LDB/ZM）× 一组代表性 `column_aliases` 输入（覆盖：物理列命中 open_arms/closed_arms；命中 center/light/open；含 `__ignore__`/`None`；existing_overrides 含同名 target_param 的显式优先场景），调用当前生产 `_build_zone_aliases_overrides`，把返回 dict 序列化为 golden 快照（`json.dumps(..., ensure_ascii=False, sort_keys=True)`）内联进测试或存 `tests/fixtures/`。
2. 实现改动 1-4 后，同样输入再调一次，断言 **`新输出 == golden 快照`**（逐键逐值、wrap_list 决定的 list/scalar 类型也要一致）。
3. **已实跑锚定的最小 golden 基线**（2026-06-11 实测，可直接作为 fixture 起点）：

   | 输入 column_aliases | 范式 | 当前生产输出 |
   |---|---|---|
   | `{'中心区':'center'}` | open_field | `{"center_zone": "中心区"}`（scalar） |
   | `{'明区':'light'}` | light_dark_box | `{"light_zone": "明区"}`（scalar） |
   | `{'开放区A':'open','开放区B':'open'}` | zero_maze | `{"open_zones": ["开放区A","开放区B"]}`（list） |
   | `{'OA1':'open_arms','CA1':'closed_arms'}` | epm | `{"closed_arm_zones": ["CA1"], "open_arm_zones": ["OA1"]}`（双 concept） |

4. 至少覆盖：EPM（双 concept）、ZM（wrap_list=True 产 list）、OFT（wrap_list=False 产 scalar）、existing_overrides 命中走「显式优先 continue」路径（`resolve.py:670-675`）。

> 这是 Fable 判据「无损规范化」的**可执行证明**。golden 必须在改动前采集（基线），否则只是自证。

### 测试文件 3：扩 `tests/test_catalog.py` 或新增小护栏

- `test_catalog_construct_without_resolved_zone_concepts_field`：手工 `Catalog(...)` 不传 `resolved_zone_concepts` 仍能构造（验 default_factory 向后兼容，护住所有现有测试构造点）。

### 越界守护（守 Stage 边界，可选但推荐）

- `test_stage2_does_not_emit_explicit_concept_source`：对全部 6 范式，`resolved_zone_concepts` 中所有条目的 `source ∈ {"zone_concept_params", "anonymous_zone_override"}`，**绝无** `"explicit_concept"`（那是 Stage 3 才产出的；护住「Stage 2 不实现 explicit_concept 分支」）。

**回归全量**：实现后跑 `cd packages/ethoinsight && pytest tests/` 全绿（MEMORY「改共享 helper 合并前必跑全量 + grep 所有调用方」——`_build_zone_aliases_overrides` 被 `resolve.py:202` 调用，是核心 resolve 路径）。等价性回归之外，**必须确认以下直接覆盖 zone 路由/override 的测试不红**（实证存在且当前绿）：

- `tests/test_column_semantics.py`（**直接引用 `_build_zone_aliases_overrides`**，grep 实证唯一测试侧消费者）
- `tests/test_zone_unnamed_detection_all_paradigms.py`、`tests/test_oft_zone_unnamed_detection.py`（zone 路由 + override 核心回归）
- `tests/test_catalog_loader_aliases.py`、`tests/test_resolve_charts.py`、`tests/test_catalog_parameters.py`

---

## 6. 验收闸门（客观判据）

全部满足才算 Stage 2 完成：

1. **TDD red 先行可追溯**：测试文件 1/2/3 先提交为 red（字段缺失 / golden 基线已采），再实现到 green。
2. **新测试全绿**：`pytest tests/test_resolved_zone_concepts.py tests/test_resolve_zone_overrides_equivalence.py` 全通过。
3. **字节等价回归通过**：测试文件 2 的 golden 快照断言对 EPM/OFT/LDB/ZM × 代表性输入全相等——**这是 Stage 2 的中心验收项**。
4. **ethoinsight 全量绿**：`cd packages/ethoinsight && pytest tests/` 无新增失败（对照实施前基线；已知预存在失败不算回归）。§5「回归全量」点名的 6 个 zone 相关测试文件必须不红。
5. **裸导入 + 运行期实跑双重无环判据**（**升级版，因 §2.5 的遮蔽 bug 裸导入抓不到**）：以下两条都要过——
   ```bash
   cd packages/ethoinsight
   .venv/bin/python -c "from ethoinsight.catalog import loader, resolve, schema"        # 必须 0 退出
   .venv/bin/python -c "from ethoinsight.catalog.loader import load_catalog; \
       c = load_catalog('open_field'); \
       assert c.resolved_zone_concepts and 'center' in c.resolved_zone_concepts, c.resolved_zone_concepts"  # 必须 0 退出
   ```
   > **为何加第二条**：第一条裸导入即使 loader 的 derive import 写错（遮蔽取到函数）也会退 0——因为坏代码只在 `_parse_catalog` 运行（即 `load_catalog` 被调）时才触发。`load_catalog('open_field')` 实跑会逼出遮蔽/环 bug 并断言模型非空。**单凭裸导入会给出虚假绿灯**（§2.5 实证），故第二条是真正的守门。
6. **resolve Step 1 不再现场 derive**：`grep -n "_derive_concept_from_zone_patterns" resolve.py` 不再出现在 `_build_zone_aliases_overrides` 的 Step 1（原 `:582-607` 区段）；该函数 Step 1 只读 `cat.resolved_zone_concepts`。（Step 3 `:648-656` 与 helper 定义 `:690` 仍保留，属预期。）
7. **YAML 零改动**：`git diff` 对 `catalog/*.yaml` 为空。
8. **关注点 (2) 零改动**：`git diff` 不触及 softgate（`_detect_anonymous_zone`，`:740` 起）/ translate（`:1137-1138`）区段；`anonymous_zone_override` 字段保留。

---

## 7. 风险与铁律核对

### 本 stage 特有坑

- **R1（头号风险）等价性破在 zone_patterns 收集差异**：loader 的 1b（改动 3）收集 zone_patterns 的迭代必须与 `resolve.py:596-601` 逐字节同构（同样的 `entries` 来源 = default_metrics + optional_metrics + charts、同样的 `or []` 兜底、同样的 `startswith("in_zone") and "*" in pat`）。任何细微差异 → derive 输入集变 → concept 名漂移 → 等价性回归红。**实施时并排 diff 两段代码**。
- **R2 occupied 判据方向**：1b 只在 `target_param` 未被 1a 占用时才 derive。改动 3 用 `{rc.param for rc in resolved_zone_concepts.values()}`，与 resolve 原 `{p for p, _ in concept_param_map.values()}`（`:593`）等价。EPM 无 override、OFT/LDB/ZM 无 zone_concept_params，三范式实际不触发「占用」分支，但护栏逻辑要留以防未来同范式两机制并存。
- **R3 与 Stage 1 的 requires_columns 形态耦合（实施第一步必查）**：本 stage 在 loader/resolve 都迭代 `requires_columns` 找 `in_zone*` glob。若实施时 dev 已合 Stage 1（嵌套 list = CNF），则 `pat.startswith(...)` 会对内层 list 炸 AttributeError。**实施第一步**：
  ```bash
  cd packages/ethoinsight && grep -n "requires_columns" ethoinsight/catalog/resolve.py ethoinsight/catalog/loader.py
  ```
  确认当前形态。**缓解**：本 spec 默认两 stage 独立合入互不假设；若实测 dev 已含 Stage 1，则改动 3 的迭代需 flatten（与 Stage 1 同款 flatten 处理 `:600`/`:617` 两个消费者同步），并在等价性 golden 重新基于 CNF 输入采集。**此为「待执行 agent 现场确认」项**。
- **R4 import 写法 + 环（§2.5）**：loader 调 resolve 的 derive **必须**用 `from ethoinsight.catalog.resolve import _derive_concept_from_zone_patterns`（函数名直接 import，绕过包属性遮蔽）**且放函数体内惰性**（resolve.py:23 顶层 import loader，顶层反向 import 会闭环）。取模块写法（`from ethoinsight.catalog import resolve` / `import ethoinsight.catalog.resolve as m`）都会取到函数、AttributeError——**实证见 §2.5，勿用**。验收闸门 5 第二条（`load_catalog` 实跑）兜底。

### 铁律逐条核对

- **deepseek 正面提示**（CLAUDE.md 第 6 条）：本 stage 不产出 prompt/skill 文案，无「禁止/不要」措辞风险；spec 内对边界用「保留不动 / 仅读」正面描述。✅
- **SSOT 单存**（MEMORY [[feedback_single_source_of_truth]]）：本 stage **正是消灭双存**——loader 与 resolve 两处各跑一次 derive 收口为 loader 一次、resolve 只读。concept→param 知识此后单点在 `resolved_zone_concepts`。✅
- **不跨范式复用**（CLAUDE.md 第 14 条 / MEMORY [[feedback_no_cross_paradigm_reuse_accept_duplication]]）：`resolved_zone_concepts` 是**每个 Catalog（=每个范式）各自一份** dict，EPM 的 open_arms 与任何其他范式无共享。本 stage 不做任何跨范式合并。✅
- **TDD 强制 red 先行**（CLAUDE.md 测试段 / MEMORY）：§5 三测试文件 + 越界守护全要求 red 先行；等价性 golden 必须改动前采集。✅
- **无 harness import 闭环**（CLAUDE.md「harness 模块顶层 import 闭环风险」）：本 stage 改动全在 ethoinsight 包内，无 `from deerflow...` 顶层新增；loader↔resolve 共享走惰性函数名 import（§2.5），验收闸门 5 双条（裸导入 + `load_catalog` 实跑）兜底。✅
- **改共享 helper 跑全量 + grep 调用方**（MEMORY [[feedback_pr_merge_must_run_full_suite_on_shared_logic]]）：`_build_zone_aliases_overrides` 是 resolve 核心路径（`resolve.py:202` 唯一调用方），唯一测试侧消费者 `test_column_semantics.py` 已点名；§5「回归全量」+ §6 闸门 4 已锁。✅

---

## 8. 待决策点

### D1 — derive helper 的共享方式（**实施者可自决，记录即可，无需用户拍板**）

loader 加载期需要调 `_derive_concept_from_zone_patterns`。**前提已由 §2.5 实证锁定**：包 `__init__.py:22` 用同名函数遮蔽 `resolve` 子模块、且 `resolve.py:23` 顶层 import loader（resolve→loader 单向边已存在）。两种实现：

- **选项一（默认）**：loader 在函数体内惰性 `from ethoinsight.catalog.resolve import _derive_concept_from_zone_patterns`。零搬动、字节级保证同源。**写法已被实证约束死**（函数名直接 import + 惰性，见 §2.5/R4）——不可用「取模块」写法，不可放顶层。
- **选项二（结构更稳，团队若倾向消除遮蔽脆弱性可主动选）**：把 `_derive_concept_from_zone_patterns` + `_extract_concept_keyword`（`resolve.py:682-704`）**整体下沉**到无依赖小模块 `catalog/_zone_concepts.py`，loader 与 resolve 都从该模块 import；resolve 端原函数改为从该模块 import（**函数体字节不变，只挪位置**）。优点：新模块不被包 `__init__` 遮蔽、可顶层 import、不依赖 loader，从根上免疫 §2.5 的两个陷阱。代价：一次搬动 + resolve 内部引用点改 import。

**默认选项一**（零搬动、风险被 §2.5 写死的写法 + 闸门 5 第二条共同收敛）。**注意**：初版 review 曾基于「resolve 不太可能顶层 import loader」推荐选项一并预判无环——该预判**已被证伪**（`resolve.py:23` 确实顶层 import loader）。正确结论是：选项一可行但写法被严格约束（惰性 + 函数名 import）；选项二是免疫遮蔽与环的更干净结构。**二者皆工程内部选择，不需用户/同事拍板**；决策依据 = 实施者对「零搬动 vs 结构干净」的权衡 + 验收闸门 5 第二条最终守门。

### 本 stage 无需外部拍板的领域决策

Stage 2 是纯无损重构，不涉及范式语义、默认值、概念增删（那些在 Stage 3 的 OFT border「半声明」取向、Stage 4 的运行期 vs 构建期选型，分别归各自 stage spec 的待决策点 + 同事/用户拍板）。**本 stage 不引入任何需要行为学同事或用户拍板的决策。**
