# Stage Spec — Stage 1：Q1 `requires_columns` CNF 门控（组内 OR / 整体 AND）

> 状态：可独立执行（single-agent）。
> 日期：2026-06-11（对抗自检后定稿）
> 总纲：[2026-06-11-pr115-catalog-concept-consolidation-and-gate-cnf-spec.md](2026-06-11-pr115-catalog-concept-consolidation-and-gate-cnf-spec.md)（总纲已瘦身为索引型：Q1 裁决见 §0「第一轮」、本 stage 在 §2 索引表「Stage 1」行 + 依赖图「轨道 A」）。
> 关联裁决：Fable Q1 门控选 B（嵌套列表表达 CNF，非平行 `requires_any` key）；[[feedback_single_source_of_truth]]、[[feedback_no_cross_paradigm_reuse_accept_duplication]]。

---

## 1. 目标 + 总纲位置

**目标（一句话）**：把 catalog 的 `requires_columns` 从「list of str（pattern 间纯 AND）」升级为「list of (str 或 list-of-str)」——嵌套列表表示**组内 OR、整组之间 AND（CNF）**，使一个已内化多级列降级的脚本不再被门控误毙；纯 list-of-str 行为逐字节不变。

**在总纲中的位置**：
- **依赖**：无。Stage 1 是地基第一块，**不依赖任何上游 stage**。可独立先合（总纲 §2 依赖图「轨道 A：独立、可先合」）。
- **阻塞**：Stage 1 是 Stage 3 的软前置——Stage 3 给某些范式补「概念枚举」时，若该概念在 metrics 端是「正区取反 / 多列择一」语义，落到 catalog 可能用到 CNF 嵌套列表表达「A 或 B 满足即可」。Stage 1 不落地这类语义（那是 Golden Case 之后的事），但必须先让 schema/loader/resolve **能容纳**嵌套列表，否则 Stage 3 一旦写出嵌套 `requires_columns` 会在 loader 直接抛 `CatalogError`。Stage 2 / Stage 4 与本 stage 正交，不互相阻塞。

---

## 2. 已核实事实锚点（带文件:行号，2026-06-11 实读确认）

> 本节所有行号、函数名在定稿前**逐个实读核对**（不只信 FACTS）。自检纠正的两处编造函数名、一处遗漏崩溃触点、一处传递保护依赖已并入本节。

### 2.1 schema（要改的类型声明）
- `packages/ethoinsight/ethoinsight/catalog/schema.py:65` — `MetricEntry.requires_columns: list[str]`。
- `packages/ethoinsight/ethoinsight/catalog/schema.py:89` — `ChartEntry.requires_columns: list[str] = field(default_factory=list)`。
- （`StatisticsEntry` 无 `requires_columns` 字段，无需改。）

### 2.2 loader 三处硬校验 `isinstance(c, str)`（要放松）
- `loader.py:285-288`（metric 入口，`MetricEntry`）：
  ```python
  if not isinstance(requires_columns, list) or not all(
      isinstance(c, str) for c in requires_columns
  ):
      raise CatalogError(f"{source} {where}: 'requires_columns' must be list[str]")
  ```
- `loader.py:354`（chart 入口 `charts[i]`）：`if not isinstance(requires_columns, list) or not all(isinstance(c, str) for c in requires_columns):`
- `loader.py:458`（另一类 chart-like 入口 `{key}[i]`）：同上一行结构。

### 2.3 resolve 门控主循环（要加嵌套分支，内聚单点）
- `resolve.py:494` — `def _missing_columns(patterns, available, column_aliases=None)`，主循环 `:505-512`：
  ```python
  for pat in patterns:
      if any(fnmatch.fnmatchcase(col, pat) for col in available):
          continue
      if column_aliases:
          if _any_concept_matches_pattern(available, column_aliases, pat):
              continue
      missing.append(pat)
  ```
- **`_missing_columns` 的 4 个调用点全部把整个 `requires_columns` 列表传入**（`resolve.py:217 :273 :412 :441`），自身不迭代单 pattern → **门控判定逻辑完全内聚在 `_missing_columns` 内部，4 个调用点零改动**（已核实）。

### 2.4 `missing` 的下游消费者 —— **`_detect_anonymous_zone` 是硬崩触点（P1，自检补遗）**
`_missing_columns` 返回的 `missing` 列表在 `:224` 直接喂给 `_detect_anonymous_zone`：
- `resolve.py:224` — `zone_check = _detect_anonymous_zone(missing, columns, overrides, cat.anonymous_zone_override)`。
- `resolve.py:740` — `def _detect_anonymous_zone(missing_patterns, available_columns, ...)`。
- `resolve.py:755-757` — 函数体**裸迭代 missing_patterns 调 str 方法**：
  ```python
  has_zone_pattern = any(
      pat.startswith("in_zone") and len(pat) > len("in_zone")
      for pat in missing_patterns
  )
  ```
- **后果**：一旦 §3.3 让 `missing` 里出现 `list` 项（某个 `in_zone` OR-组全不满足），`pat.startswith(...)` 在 list 上抛 `AttributeError: 'list' object has no attribute 'startswith'`。**这不是渲染细节，是功能崩溃**——且 Stage 3 写出 `in_zone` OR-组、数据恰好缺该组时必然命中，§5.2/§5.3 任何「嵌套组全缺」用例也会命中。因此 `_detect_anonymous_zone` 是**强制改动触点**（见 §3.3），missing 对外形态必须在本 spec 拍死（见 §3.5），不下放给执行 agent。

### 2.5 flatten 全仓消费者（迭代 `requires_columns` 调 str 方法 / set.update）
升 CNF 后单项可能是 list → 不 flatten 会 `AttributeError`（str 方法）或 `TypeError`（set.update）：
- `resolve.py:599-600`（函数 **`_build_zone_aliases_overrides`**（`:566`）Step 1b 收集 zone_patterns）：
  ```python
  for pat in getattr(entry, "requires_columns", []) or []:
      if pat.startswith("in_zone") and "*" in pat:
  ```
- `resolve.py:616-617`（同函数 Step 2 收集 zone_patterns，结构同上）。
  > 自检纠正：FACTS / 草案曾把此函数误写为 `_build_zone_param_overrides`——该名全仓 0 命中（`grep -rn` 实测；总纲 spec 也沿用了错名）。**真实函数是 `_build_zone_aliases_overrides`**（`resolve.py:566`）。本 spec 全文用真实名。
- `inspect_uploaded_file_tool.py:63`（harness 侧，`packages/agent/backend/packages/harness/deerflow/tools/builtins/`，函数 `_extract_required_patterns`）：`patterns.update(m.requires_columns)`。
- `inspect_uploaded_file_tool.py:65`：`patterns.update(getattr(ch, "requires_columns", []) or [])`。
  - `:63` 与 `:65` 都是 **`set.update`**——把嵌套 list 当元素加入会引入 unhashable list，抛 **`TypeError: unhashable type: 'list'`**（**不是** `AttributeError`，自检纠正）。两行都要 flatten，别只盯 `:65`。

### 2.6 传递保护依赖（自检补遗，须在 §3.4 注明）
- `utils.py:311`（`assess_column_confidence`）：`if any(fnmatch.fnmatchcase(norm, pat) for pat in patterns):`。
  - `fnmatch.fnmatchcase(col, <list>)` → `TypeError: unhashable type: 'list'`（实测；fnmatch 内部 lru_cache 要求 pattern 可哈希）。
  - **当前唯一生产调用方**是 `inspect_uploaded_file_tool.py:122`，其 `patterns` 来自 `_extract_required_patterns`（即 `:63/:65`）。→ **只要 §3.4 把 `:63/:65` flatten 正确落地，`utils.py:311` 经传递被保护，今天不崩**。但这是一条**隐式传递依赖**：未来任何新调用方把裸 CNF `requires_columns` 直接喂进 `assess_column_confidence` 即崩。§3.4 须注明此保护关系（不在本 stage 改 `utils.py`，仅记录耦合，防后续 stage 误触）。

### 2.7 现状数据（确认今天无嵌套，向后兼容基线干净）
- 7 个 catalog yaml（`_common` + `epm`/`oft`/`ldb`/`zero_maze`/`fst`/`tst`）的 `requires_columns` 当前**全是纯 list-of-str**（grep 实测，`in_zone_light*` / `in_zone_open*` / `in_zone_center_*` 等均为裸字符串项）。→ Stage 1 落地后这些 yaml **一字不改**，行为基线即「字节不变」回归的对照。

---

## 3. 改动清单（逐文件逐处）

### 3.1 `schema.py`（类型声明，2 处）
- `:65` `MetricEntry.requires_columns: list[str]` → `requires_columns: list[str | list[str]]`。
- `:89` `ChartEntry.requires_columns: list[str] = field(default_factory=list)` → `requires_columns: list[str | list[str]] = field(default_factory=list)`。
- 仅类型注解放宽，`@dataclass(frozen=True)` 不变；嵌套 list 不可哈希不影响 frozen dataclass（其字段哈希由 dataclass 控制，且这两个类的现有用法未把 `requires_columns` 放进 set/dict key）。

### 3.2 `loader.py`（放松三处校验，复用同一新校验逻辑）
新增一个**模块内私有校验函数**（放在三处校验之前）：
```python
def _is_cnf_requires_columns(value: object) -> bool:
    """list，每项为非空 str，或非空 list-of-(非空 str)。"""
    if not isinstance(value, list):
        return False
    for item in value:
        if isinstance(item, str):
            if not item:
                return False
        elif isinstance(item, list):
            if not item or not all(isinstance(s, str) and s for s in item):
                return False
        else:
            return False
    return True
```
- `:285-288`（metric 入口）：把 `if not isinstance(requires_columns, list) or not all(isinstance(c, str) ...)` 整块替换为 `if not _is_cnf_requires_columns(requires_columns):`，错误信息改为 `'requires_columns' must be list of str or list-of-str groups`。
- `:354`（chart 入口）：同样替换为 `if not _is_cnf_requires_columns(requires_columns):`，错误信息同上风格（保留原有 `{source}: charts[{i}]` 前缀）。
- `:458`（chart-like `{key}[i]`）：同样替换（保留 `{source}: {key}[{i}]` 前缀）。
- **正面提示原则**（[[feedback_deny_messages_must_direct]] / CLAUDE.md 第 6 条）：错误信息描述「应当是什么」（list of str or list-of-str groups），不写「禁止 X」。

### 3.3 `resolve.py` 门控与 `missing` 下游（两处强制改动）

**(a) `_missing_columns`（`:494`）主循环加嵌套分支**，把 `:505-512` 改为对每个 `pat` 判类型：
```python
for pat in patterns:
    if isinstance(pat, list):
        # 组内 OR：任一 sub-pattern 满足 → 整组满足
        group_ok = False
        for sub in pat:
            if any(fnmatch.fnmatchcase(col, sub) for col in available):
                group_ok = True
                break
            if column_aliases and _any_concept_matches_pattern(available, column_aliases, sub):
                group_ok = True
                break
        if group_ok:
            continue
        missing.append(pat)          # 整组（list）计入 missing —— 见 §3.5 拍死的对外形态
        continue
    # 原有 str 路径，逐字节不变
    if any(fnmatch.fnmatchcase(col, pat) for col in available):
        continue
    if column_aliases:
        if _any_concept_matches_pattern(available, column_aliases, pat):
            continue
    missing.append(pat)
```
- str 分支与今天**完全相同**（保证纯 list-of-str 回归字节不变）。
- 门控判定**绝不 flatten**——flatten 会把组内 OR 退化成跨组 AND，改变语义（见 §3.4 末「flatten 语义说明」）。
- `_missing_columns` 的 4 个调用点（`:217 :273 :412 :441`）**不动**。

**(b) `_detect_anonymous_zone`（`:740`）入口 flatten missing_patterns（P1，硬崩修复）**
因 (a) 让 `missing` 可能含 list 项，而 `_detect_anonymous_zone` 在 `:755-757` 裸迭代 `pat.startswith(...)`，必须在函数入口先摊平：
```python
def _detect_anonymous_zone(missing_patterns, available_columns, overrides, anonymous_zone_override=None):
    flat_missing = _flatten_requires_columns(missing_patterns)   # 防 list 项炸 .startswith
    has_zone_pattern = any(
        pat.startswith("in_zone") and len(pat) > len("in_zone")
        for pat in flat_missing
    )
    ...   # 函数体后续把 missing_patterns 的迭代点统一改用 flat_missing
```
- **决策锁定**（与 §3.5 一致）：`missing` 对外**保留嵌套 list 原形态**（CNF 结构信息无损），由各下游消费者**自行 flatten**——此处即第一个、也是唯一一个会崩的下游。zone 检测只关心「出现过哪些 in_zone pattern」、不关心 OR/AND 结构，故 flatten 无损。
- **同处核实**：通读 `_detect_anonymous_zone` 函数体（`:740` 至函数末），把所有对 `missing_patterns` 的迭代/str 操作统一改走 `flat_missing`，不得残留对原 `missing_patterns` 的裸迭代。

### 3.4 flatten 消费者（全处理，否则嵌套 list 调 str 方法 / set.update 炸）
新增一个**共享 flatten helper**（放 `resolve.py` 模块级，见 §8 待决策点 #1，推荐选项 A）：
```python
def _flatten_requires_columns(requires_columns) -> list[str]:
    """把 CNF requires_columns 摊平成 str 列表（组拆散），供只关心『出现过哪些 pattern』的消费者用。
    None / [] → []；纯 str 列表 → 浅拷贝（顺序保持）；嵌套项就地展开。"""
    out: list[str] = []
    for item in (requires_columns or []):
        if isinstance(item, list):
            out.extend(item)
        else:
            out.append(item)
    return out
```
逐处改：
- `resolve.py:599-600`（`_build_zone_aliases_overrides` Step 1b）：循环改为 `for pat in _flatten_requires_columns(getattr(entry, "requires_columns", [])):`，循环体 `pat.startswith("in_zone")` 不变（pat 现在保证是 str）。
- `resolve.py:616-617`（同函数 Step 2）：同样改为 `for pat in _flatten_requires_columns(getattr(entry, "requires_columns", [])):`。
- `inspect_uploaded_file_tool.py:63`：`patterns.update(m.requires_columns)` → `patterns.update(_flatten_requires_columns(m.requires_columns))`。
- `inspect_uploaded_file_tool.py:65`：`patterns.update(getattr(ch, "requires_columns", []) or [])` → `patterns.update(_flatten_requires_columns(getattr(ch, "requires_columns", [])))`。
- **harness 侧 import 方向 + 位置**：`inspect_uploaded_file_tool.py` 现有所有 `ethoinsight` import 均在**函数体内**（`:55` `from ethoinsight.catalog.loader import load_catalog`、`:115`、`:289` 等，全惰性）。新加的 `from ethoinsight.catalog.resolve import _flatten_requires_columns` **必须同样放函数体内**（`_extract_required_patterns` 体内），**不得放模块顶层**——既符合 CLAUDE.md「harness import 惰性放函数体」闭环铁律，又与现状一致。这是 harness→ethoinsight 顺方向，无闭环风险；§6.4 裸导入验证是兜底而非替代此约束。

> **传递保护依赖（§2.6，须随 §3.4 一并记入实现注释，本 stage 不改 `utils.py`）**：`utils.py:311` 的 `assess_column_confidence` 经其唯一生产调用方 `inspect_uploaded_file_tool.py:122` 间接消费 `_extract_required_patterns` 的产物。**只要 `:63/:65` flatten 落地，`utils.py:311` 拿到的就是纯 str 列表，今天不崩**。但这条保护是隐式传递依赖——在 `_extract_required_patterns` 或 `assess_column_confidence` 处加一行注释说明「patterns 必须已 flatten 为 list[str]」，防后续 stage 新增调用方把裸 CNF 喂进来踩 `TypeError`。

> **flatten 语义说明（承重）**：flatten 只服务「我只想知道这个 catalog 提到过哪些列 pattern」的消费者（zone 概念推导 `_build_zone_aliases_overrides`、inspect 候选列收集、zone 检测的 has_zone_pattern 判断）——它们**不关心 AND/OR 结构**，拆散组是无损的。**门控判定（`_missing_columns`）绝不能 flatten**，否则组内 OR 退化成跨组 AND，改了门控语义。两类消费者必须分开对待（§3.3a 不 flatten vs §3.3b/§3.4 flatten），§6.6 grep 净空验收即为此分界守门。

---

## 4. 接口契约

### 4.1 输入契约（本 stage 对上游的依赖）
- **无上游 stage 依赖**。仅依赖现有 catalog 结构与 `fnmatch` 行为。
- 依赖现状事实：7 个 `<paradigm>.yaml` 的 `requires_columns` 当前为纯 list-of-str（§2.7），作为「字节不变」回归的对照基线。

### 4.2 输出契约（本 stage 给下游的产出）
- **给 Stage 3**：schema/loader/resolve **完整接受** `requires_columns` 项为 `str` 或 `list[str]`。Stage 3 若需在 catalog 写「A 或 B 满足即可」的概念门控，可直接用嵌套 list：loader 不再拒收、`_missing_columns` 正确按 CNF 判定、**且 `missing` 含 list 项时下游不再崩**（`_detect_anonymous_zone` 已 flatten 加固）。
- **稳定不变量（下游可依赖）**：
  1. 纯 list-of-str 的 `requires_columns` 经 loader→resolve 的输出与 Stage 1 前**逐字节相同**。
  2. `_missing_columns(patterns, available, aliases)` 对嵌套组：组内任一 sub-pattern（含 alias 概念匹配）命中 → 整组满足；全不命中 → 整组以 `list` 形态进 `missing`。
  3. `missing` 列表可能含 `list` 项（嵌套组形态保留，§3.5）；**所有迭代 `missing` 调 str 方法的下游必须先 `_flatten_requires_columns`**（当前唯一这样的下游 `_detect_anonymous_zone` 已加固）。
  4. flatten helper `_flatten_requires_columns` 对纯 str 列表返回浅拷贝（顺序保持），对嵌套项就地展开，`None`/`[]` 返回 `[]`。
- **不产出**：本 stage **不**修改任何 `.yaml`（无嵌套 list 写入 catalog），**不**实现 N:1 聚合 / OR 概念语义（那是 Stage 3 + Golden Case 的边界），**不**改 `utils.py`（仅记录其传递保护依赖）。

### 4.3 与并行 stage 的同处改动核对
- 与 Stage 2/4 **无同行/同函数冲突**（Stage 2 改 loader 的 zone_concept/anonymous_override 段 `:128-189` + Catalog 派生字段，与本 stage 改的 `:285-288/:354/:458` 校验块、resolve 门控不重叠）。
- **唯一软缝**：Stage 2 改动 3 会在 loader 侧**新增**一个裸迭代 `requires_columns` 找 `in_zone` 的 flatten 消费者（结构同 `resolve.py:599-601`）。Stage 2 spec §7 R3 已显式承认「若 Stage 1 先合，此处对内层 list 炸 `AttributeError`，需 flatten」——**该缝由 Stage 2 侧自查兜底**（见 §6.6 免责声明）。

---

## 5. TDD（red 先行）

新测试文件：`packages/ethoinsight/tests/test_requires_columns_cnf.py`（新建，集中本 stage ethoinsight 侧全部断言）。
harness 侧 flatten 回归：新建 `packages/agent/backend/tests/test_inspect_requires_columns_flatten.py`（或复用现有 `test_inspect_data_preview.py`，二选一）。

> **TDD 纪律**（CLAUDE.md「TDD 强制」+ [[feedback_pr_merge_must_run_full_suite_on_shared_logic]]）：以下每个测试**先写、先 red**（在改 schema/loader/resolve 之前跑必须失败），再改实现转 green。「字节等价」类测试在改动前应天然 green（基线），改动后保持 green——它是回归护栏而非 red→green，单列出。改了共享逻辑（`_missing_columns`/loader 校验）后**必须跑全量** ethoinsight 套，不只跑新文件。

### 5.1 loader 接受嵌套 / 拒收畸形（red→green）
- `test_loader_accepts_nested_requires_columns`：构造 metric/chart 的 dict，`requires_columns=["velocity", ["in_zone_center", "in_zone_periphery"]]`，调 loader 解析路径（或直接 `_is_cnf_requires_columns`）。
  - **red**（改前）：loader `:285/:354/:458` 旧 `all(isinstance(c, str))` 对嵌套 list 项为 False → 抛 `CatalogError`。
  - **green**（改后）：解析成功，`MetricEntry.requires_columns[1] == ["in_zone_center", "in_zone_periphery"]`。
- `test_loader_rejects_malformed_requires_columns`：分别喂 `[123]`、`[[]]`（空组）、`[["a", 1]]`（组内非 str）、`[""]`（空 str）。断言每个都抛 `CatalogError`（守 `_is_cnf_requires_columns` 的非空/类型边界）。

### 5.2 `_missing_columns` CNF 判定（red→green）
- `test_missing_columns_or_group_satisfied_by_either`：`patterns=[["in_zone_center", "in_zone_periphery"]]`，`available=["in_zone_periphery"]` → `missing == []`（OR：有一个就行）。
  - **red 机理（自检纠正）**：改前主循环对 list 项调 `fnmatch.fnmatchcase(col, pat)`，`pat` 是 list → **`fnmatch` 内部 lru_cache 要求 pattern 可哈希，list 不可哈希 → 直接抛 `TypeError: unhashable type: 'list'`**（**不是**「返回不匹配致 missing 非空」）。即改前此测试以**抛异常**形态 red；改后断言 `missing == []` green。
- `test_missing_columns_or_group_all_absent`：`patterns=[["in_zone_center", "in_zone_periphery"]]`，`available=["velocity"]` → `missing == [["in_zone_center", "in_zone_periphery"]]`（整组以 list 形态计入，§3.5 拍死的对外形态）。
- `test_missing_columns_mixed_cnf`：`patterns=["velocity", ["a", "b"]]`，`available=["velocity", "b"]` → `missing == []`（AND：str 项命中 + 组命中）；再 `available=["velocity"]` → `missing == [["a", "b"]]`。
- `test_missing_columns_or_group_via_alias`：组内 sub-pattern 经 `column_aliases` 概念匹配命中 → 整组满足（覆盖 `_any_concept_matches_pattern` 在组内分支生效）。

### 5.3 字节等价回归（基线护栏，改前 green / 改后保持 green）
- `test_pure_list_of_str_resolve_unchanged`：对真实 catalog（`epm`/`oft`/`ldb`/`zero_maze`/`fst`/`tst` 全跑）调公共入口 `resolve` / `resolve_metrics`（`resolve.py:122`）/ `resolve_charts`（`resolve.py:350`），用公共序列化器 `plan_to_dict`（`:1169`）/ `plan_metrics_to_dict`（`:1225`）/ `plan_charts_to_dict`（`:1277`）把输出序列化为规范 JSON，**断言改动前后逐字段相等**。
  - 实现方式：**先新建目录** `packages/ethoinsight/tests/fixtures/cnf_baseline/`（实测当前不存在，须 `mkdir`）；在改动**前**把每个范式的序列化输出落盘成 `cnf_baseline/<paradigm>.json`，改动后重跑断言 `== fixture`。
  - 这是 Fable「无损」判据在 Stage 1 的可执行证明（与 Stage 2 等价性回归同形，Stage 1 范围只覆盖 `requires_columns` 路径）。
- `test_missing_columns_str_path_unchanged`：参数化喂多组纯 str `patterns` + `available`（含 alias 路径），断言 `_missing_columns` 输出与一份手算 golden 完全一致——锁死 str 分支零行为漂移。
- `test_detect_anonymous_zone_str_path_unchanged`：纯 str `missing_patterns`（含 `in_zone_center_*` 一类）跑 `_detect_anonymous_zone`，断言加 flatten 前后返回值（`ResolveError`/`True`/`None` 三态）完全一致——锁死 zone 检测 str 路径零漂移。

### 5.4 flatten + 硬崩修复（red→green）
- `test_flatten_requires_columns`（ethoinsight 侧）：`_flatten_requires_columns(["a", ["b", "c"], "d"]) == ["a", "b", "c", "d"]`；纯 str `["a","b"]` 原样（浅拷贝、顺序保持）；`None`/`[]` 返回 `[]`。
- `test_detect_anonymous_zone_with_nested_missing`（ethoinsight 侧，**对应 §3.3b 硬崩修复**）：直接构造 `missing_patterns=[["in_zone_center", "in_zone_periphery"]]`（含 list 项）+ `available=["in_zone"]` + 一个带 `anonymous_zone_override` 的 catalog，调 `_detect_anonymous_zone`。
  - **red**（改前）：`:755-757` `pat.startswith` 在 list 项上抛 `AttributeError`。
  - **green**（改后）：入口 flatten 后正常返回（zone 检测照常工作，不崩）。
- `test_build_zone_aliases_overrides_with_nested`（ethoinsight 侧，**真实函数名**）：构造含嵌套 `requires_columns` 的 catalog（mock 或临时 yaml），跑 `_build_zone_aliases_overrides`（`resolve.py:566`）全程**不抛 `AttributeError`**（red：改前 `:599/:616` `pat.startswith` 在 list 上炸）。
- `test_inspect_collects_patterns_with_nested`（harness 侧 `backend/tests/`）：mock/load 含嵌套 `requires_columns` 的 catalog，调 `_extract_required_patterns`（经 inspect 工具 pattern 收集路径），断言 `set.update` 不抛 `TypeError: unhashable type: 'list'` 且收集到展开后的 str pattern（red：改前 `:63/:65` 把 list 塞进 set）。

---

## 6. 验收闸门（客观判据）

1. **新测试全绿**：`cd packages/ethoinsight && pytest tests/test_requires_columns_cnf.py -v` 全通过；harness 侧 inspect flatten 测试通过。
2. **全量回归绿**：`cd packages/ethoinsight && pytest tests/`（注：4 个 harness 侧污染测试与 ethoinsight 套无关，见 [[feedback_known_full_suite_test_pollution_4_tests]]，不在 ethoinsight 套内）；改 inspect 工具后 `cd packages/agent/backend && make test`。
3. **字节等价证明**：`test_pure_list_of_str_resolve_unchanged` 对全部 6 范式绿 + `test_detect_anonymous_zone_str_path_unchanged` 绿——证明纯 list-of-str 路径与 zone 检测 str 路径零漂移。
4. **裸导入验证**（因改了 harness 侧 `inspect_uploaded_file_tool.py`，按 CLAUDE.md 闭环铁律）：在 `packages/agent/backend/` 下
   ```bash
   PYTHONPATH=. python -c "import app.gateway"
   PYTHONPATH=. python -c "from deerflow.agents import make_lead_agent"
   ```
   两者 0 退出。
5. **catalog yaml 零改动**：`git diff packages/ethoinsight/ethoinsight/catalog/*.yaml` 为空（Stage 1 不写嵌套 list 进 catalog）。
6. **grep 净空**（**cwd = repo 根**，命令含 `packages/...` 绝对路径前缀，从 repo 根跑，从 `packages/ethoinsight` 跑会找不到 harness 路径）：
   ```bash
   grep -rn "requires_columns" \
     packages/ethoinsight/ethoinsight/catalog/resolve.py \
     packages/agent/backend/packages/harness/deerflow/tools/builtins/inspect_uploaded_file_tool.py
   ```
   逐处核对：每一处迭代 `requires_columns` 或 `missing`（含 list 项）的点，要么走 `_missing_columns`（CNF 感知、不 flatten），要么走 `_flatten_requires_columns`（flatten），**无裸 `for pat in <requires_columns|missing>: pat.<strmethod>`**。
   > **免责（自检 B3）**：本 gate 只证 **Stage 1 时刻**净空。后续 stage（如 Stage 2 改动 3 在 loader 新增的裸迭代点）由**该 stage 自查 + 各自「先 grep 当前形态」纪律**兜底，不在 Stage 1 验收范围内。

---

## 7. 风险与铁律核对

**本 stage 特有坑**：
- **最严重 = `missing` 含 list 项时 `_detect_anonymous_zone` 硬崩**（§2.4 / §3.3b）。不修则「门控判定能容纳嵌套、loader 也容纳嵌套，但 missing 下游不容纳」——输出契约半真，Stage 3 一写 `in_zone` OR-组且数据缺该组立刻 `AttributeError`。这是会产出「看似全绿、一遇 in_zone OR-组缺列即崩」哑故障的**头号风险**，§5.4 `test_detect_anonymous_zone_with_nested_missing` 为其 red 锚点。
- **flatten 回归（次严重）**：门控不能 flatten（OR 退化成 AND）、纯展示/收集消费者必须 flatten（否则 `AttributeError`/`unhashable`）。两类消费者已在 §3.3 vs §3.4 分开，落地时逐处对照 §2.5 清单核完，跑 §6.6 grep 净空确认无遗漏。
- **`set.update` 的 unhashable 陷阱**：`inspect_uploaded_file_tool.py:63/:65` 用 `set.update`，嵌套 list 是 unhashable，漏 flatten 抛的是 `TypeError`（**不是** `AttributeError`）——两行都改，别只盯 `:65`。
- **`utils.py:311` 传递保护**：今天经 `:63/:65` flatten 间接受保护、不崩；但属隐式传递依赖，§3.4 已要求加注释、本 stage 不改该文件（不过度扩面）。

**逐条铁律核对**：
- **deepseek 正面提示**：loader 错误信息写「should be list of str or list-of-str groups」，不写「禁止/不要」。✓（§3.2）
- **SSOT 单存**：本 stage 不引入任何知识副本；catalog 仍是 `requires_columns` 唯一来源，只是表达力升级；flatten 逻辑单点定义（§8 选项 A）。✓
- **不跨范式复用**：本 stage 不碰范式语义，纯结构能力升级，对 6 范式一视同仁，无跨范式归并。✓
- **TDD 强制 red 先行**：§5 每个 red→green 测试明确标了「改前 red 的原因 + 异常类型」；字节等价测试单列为基线护栏。✓
- **import 闭环风险**：改动主体在 ethoinsight 包内（schema/loader/resolve）无环；唯一 harness 侧改动 `inspect_uploaded_file_tool.py` 的新 import 是 harness→ethoinsight 顺方向且**惰性放函数体**（§3.4），加 §6.4 裸导入验证兜底。✓（CLAUDE.md 闭环铁律）

---

## 8. 待决策点

> 以下两点均为**实现细节级**，**不需要用户/同事/领域专家拍板**（那是 Stage 3 / Golden Case 的边界），由执行 agent 按推荐落地即可；列出仅为透明。
> 注意：自检曾把「missing 对外形态」误列为本节待决策点——它实际是**功能正确性**问题（决定 `_detect_anonymous_zone` 是否崩、`missing_patterns` 进 `ResolveError.details` 的结构是否变契约），已在 **§3.3b / §3.5 拍死为「missing 保留嵌套 list 原形态、下游各自 flatten」**，**不在本节、不下放**。

1. **flatten helper 放哪 / inspect 工具是否 import 私有名**（建议直接定，影响小）：
   - **选项 A（推荐）**：`_flatten_requires_columns` 放 `resolve.py` 模块级，`inspect_uploaded_file_tool.py` 在 `_extract_required_patterns` **函数体内**惰性 `from ethoinsight.catalog.resolve import _flatten_requires_columns`。优点：单点定义、SSOT；缺点：跨包 import 一个下划线私有名。
   - **选项 B**：inspect 工具内就地内联同款 3 行 flatten。优点：不跨包引私有名；缺点：逻辑两份（轻微 SSOT 瑕疵，flatten 是 trivial 纯函数，复制可接受）。
   - **推荐 A**——flatten 是承重的语义边界（「能 flatten 的消费者 vs 不能 flatten 的门控」），集中一处更利后续维护；harness→ethoinsight 惰性 import 已有先例（`load_catalog`）。**若 reviewer 反对引私有名，退 B 不阻塞。**

2. **`missing` 含 list 项的人类可读渲染**（仅限纯展示文案，不涉功能）：若某下游把 missing 渲染进给人看的 `notes` 文案（区别于结构化字段），list 项用 `" | ".join(group)`（OR 语义可读）还是 `str(group)`？**推荐 `" | ".join(group)`**。此项**仅影响文案可读性**，不影响门控/zone 检测/契约（结构化形态已由 §3.5 锁定为保留 list），故可由执行 agent 按 grep 实际下游就地决定。

---

## 9. §3.5 决策锁定汇总（被 §3.3b / §4.2 / §8 引用）

**`missing`（`_missing_columns` 返回值）的对外形态拍死为：保留嵌套 list 原形态。**
- 命中的组不进 `missing`；未命中的 str 项以 str 进 `missing`；未命中的 OR-组以**整个 `list`** 进 `missing`（保留「这一组都没满足」的 CNF 结构信息，无损）。
- **所有迭代 `missing` 调 str 方法的下游必须先 `_flatten_requires_columns`**。Stage 1 时刻唯一这样的下游是 `_detect_anonymous_zone`（§3.3b 已加固）。
- 此形态进 `ResolveError.details` 的 `missing_patterns` 字段，是**契约的一部分**（§4.2 不变量 3）；下游（data-analyst/lead）若消费该字段须按「项可能是 str 或 list-of-str」处理。**正因这是契约而非渲染，必须在 spec 拍死、不下放给执行 agent。**
