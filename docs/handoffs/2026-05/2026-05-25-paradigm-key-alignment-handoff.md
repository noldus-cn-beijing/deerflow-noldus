# 2026-05-25 范式 key 对齐实施 handoff (方案 B 根治)

> **状态**：5/25 端到端 trace `9f77adcc` 重现了 lead 调 prep_metric_plan 试错"forced_swim → fst"的问题。本 handoff 给独立 worktree 的新 agent 执行，**在 P0-P3 chart catalog 实施之前先做**，确保下游补图工作建立在对齐的 key 之上。
>
> **范围**：纯架构对齐，不改业务逻辑、不改 metric 算法、不改 chart 实现。**不在本任务范围**的事情明确列在末尾。

---

## 一句话目标

让整个系统**统一用学术名（`forced_swim` / `tail_suspension` / `open_field` / `light_dark_box` / `epm` / `zero_maze` / `shoaling`）作为 canonical paradigm key**，从识别（identify_ev19_template）→ 落盘（set_experiment_paradigm）→ 计划（prep_metric_plan）→ 知识注入（skill doc）→ 计算（metrics dispatcher）全程**一套 key 不需要任何转换**。

---

## 当前问题（背景，不需要再调研）

### 三套 key 共存

| 用途 | 标识符 | 例子 |
|---|---|---|
| identify_ev19_template 返回的 `paradigm_key` | **学术名** | `forced_swim`, `tail_suspension`, `open_field`, `light_dark_box` |
| set_experiment_paradigm 写盘的 paradigm 字段 | **学术名**（同上） | `forced_swim` |
| metrics dispatcher `compute_paradigm_metrics(paradigm=...)` | **学术名** | `forced_swim` |
| skill `ethovision-paradigm-knowledge/references/by-experiment/*.md` 的 slug | **学术名** | `forced_swim` |
| catalog yaml 文件名 + 文件内 `paradigm:` 字段 | **短 key** | `fst.yaml`、`paradigm: fst` |
| catalog loader `load_catalog(paradigm)` 接受的入参 | **短 key**（按文件名查找） | `"fst"` |

### 故障实证（trace `9f77adcc`）

lead 收到 identify 返回 `paradigm_key="forced_swim"`，按规则传给 prep_metric_plan：

1. 第一次调 `prep_metric_plan(paradigm="forced_swim")` → catalog 报 `unknown_paradigm`（因为找不到 `forced_swim.yaml`）
2. lead 收到 error → 再 read SKILL.md 找正确 key
3. 第二次调 `prep_metric_plan(paradigm="fst")` → ✅

这是 LLM 试错行为，**不应该让 LLM 担责**。架构问题。

### 调研结论（已 grep 全仓库）

- 24 个文件用学术名作为 paradigm key
- 9 个文件用短 key（其中 6 个在 catalog 自身：`{fst,tst,oft,ldb,epm,zero_maze,shoaling}.yaml`）
- dispatcher、skill doc、experiment_context 全都是学术名
- **只有 catalog 文件名 + catalog 文件内 `paradigm:` 字段 + loader 几个调用方用短 key**

---

## 方案

**让 `catalog/loader.load_catalog()` 接受学术名作为入参**，内部映射到现有的短 key 文件名。

这样所有上游（identify_ev19_template / prep_metric_plan / chart_maker / skill doc）都用学术名，**不需要任何转换**；底层 catalog yaml 文件名暂保留为短 key（不动文件系统）。

**这不是"加 alias 掩盖问题"——这是把 canonical key 定为学术名，让 loader 做唯一的物理路径解析**。

### 改动清单

#### 1. 加 alias 表 + 改 `load_catalog`（核心）

文件：`packages/ethoinsight/ethoinsight/catalog/loader.py`

在 `_DEFAULT_CATALOG_DIR` 之后加 alias 表：

```python
# Canonical paradigm key (academic name) → catalog YAML filename stem.
# Upstream code (identify_ev19_template, prep_metric_plan, metrics dispatcher,
# skill docs, experiment_context) ALL use the academic name as the
# canonical paradigm key. Catalog YAML filenames historically use shortened
# abbreviations (fst / tst / oft / ldb); the alias map below preserves that
# physical layout without exposing the inconsistency upstream.
#
# Filename-style keys (e.g. "fst") are also accepted for backward
# compatibility with existing scripts (plot_timeseries.py, test_catalog.py)
# that still pass abbreviations directly.
_PARADIGM_ALIASES: dict[str, str] = {
    # academic name → filename stem
    "forced_swim": "fst",
    "tail_suspension": "tst",
    "open_field": "oft",
    "light_dark_box": "ldb",
    # already aligned (no aliasing needed but listed for clarity)
    "epm": "epm",
    "zero_maze": "zero_maze",
    "shoaling": "shoaling",
}
```

修改 `load_catalog`:

```python
def load_catalog(paradigm: str, catalog_dir: str | Path | None = None) -> Catalog:
    """加载 <catalog_dir>/<paradigm>.yaml 并校验返回 Catalog。

    Args:
        paradigm: Canonical paradigm key (academic name, e.g. "forced_swim",
            "open_field", "epm"). Filename-style abbreviations (e.g. "fst",
            "oft") are also accepted for backward compatibility.
        catalog_dir: catalog YAML 目录；默认为本模块所在目录

    Raises:
        CatalogError: 文件不存在 / 必填字段缺失 / enum 越界 / id 重复 等
    """
    catalog_dir = Path(catalog_dir) if catalog_dir else _DEFAULT_CATALOG_DIR
    # Resolve canonical academic name → filename stem. Accept both directions:
    # if input is already the filename stem (no entry in alias map), use as-is.
    filename_stem = _PARADIGM_ALIASES.get(paradigm, paradigm)
    yaml_path = catalog_dir / f"{filename_stem}.yaml"
    if not yaml_path.is_file():
        raise CatalogError(
            f"Catalog file not found for paradigm '{paradigm}' "
            f"(resolved to '{filename_stem}.yaml'): {yaml_path}"
        )
    try:
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise CatalogError(f"YAML parse failed for {yaml_path}: {e}") from e

    if not isinstance(raw, dict):
        raise CatalogError(
            f"{yaml_path}: top-level must be a mapping, got {type(raw).__name__}"
        )

    return _parse_catalog(raw, source=yaml_path)
```

**注意**：
- 不改 `_parse_catalog` 内部对 yaml 内 `paradigm:` 字段的处理，那个字段读到的是 `"fst"`（跟文件名一致），不传到上游。
- 短 key 调用（如 `load_catalog("fst")`）仍然能工作——alias 表里不存在 `"fst"` 这个 key，所以 `.get("fst", "fst")` 返回 `"fst"` 自身。**完全向后兼容**。

#### 2. 更新 `prep_metric_plan_tool.py` docstring

文件：`packages/agent/backend/packages/harness/deerflow/tools/builtins/prep_metric_plan_tool.py`

Line 78 改：

```python
#       paradigm: 范式如 'epm' / 'oft' / 'fst' / 'ldb' / 'tst' / 'zero_maze' / 'shoaling'
#
# 改为：
#       paradigm: 范式 canonical key（学术名）：
#                 'epm' / 'open_field' / 'forced_swim' / 'light_dark_box' /
#                 'tail_suspension' / 'zero_maze' / 'shoaling'
#                 （filename-style 缩写如 'oft'/'fst'/'ldb'/'tst' 也接受，向后兼容）
```

不改 `resolve_metrics(paradigm=paradigm, ...)` 调用本身——`resolve_metrics` 内部调 `load_catalog`，loader 自己做 alias 解析。

#### 3. 单测：覆盖 alias 解析

新增 `packages/ethoinsight/tests/test_catalog_loader_aliases.py`：

```python
"""Test paradigm key alias resolution in catalog/loader.load_catalog().

Background: identify_ev19_template / prep_metric_plan / metrics dispatcher /
skill docs all use academic-name paradigm keys (e.g. "forced_swim").
Catalog YAML files historically use shortened stems (fst.yaml). load_catalog
must accept BOTH and resolve correctly.
"""

from __future__ import annotations

import pytest

from ethoinsight.catalog.loader import CatalogError, load_catalog


# Academic name → expected catalog paradigm field value (which equals the stem)
_ACADEMIC_TO_FILENAME: dict[str, str] = {
    "forced_swim": "fst",
    "tail_suspension": "tst",
    "open_field": "oft",
    "light_dark_box": "ldb",
}

# Catalog keys that don't need aliasing (academic name == filename stem)
_ALREADY_ALIGNED = ["epm", "zero_maze", "shoaling"]


@pytest.mark.parametrize("academic,filename_stem", list(_ACADEMIC_TO_FILENAME.items()))
def test_load_catalog_accepts_academic_name(academic: str, filename_stem: str) -> None:
    """load_catalog(academic_name) should resolve to the abbreviated YAML file."""
    cat = load_catalog(academic)
    # Catalog's internal paradigm field still uses the filename stem (unchanged)
    assert cat.paradigm == filename_stem, (
        f"load_catalog('{academic}') should return catalog with paradigm='{filename_stem}', "
        f"got '{cat.paradigm}'"
    )


@pytest.mark.parametrize("academic,filename_stem", list(_ACADEMIC_TO_FILENAME.items()))
def test_load_catalog_still_accepts_filename_stem(academic: str, filename_stem: str) -> None:
    """Backward compat: passing the short filename stem must still work."""
    cat = load_catalog(filename_stem)
    assert cat.paradigm == filename_stem


@pytest.mark.parametrize("paradigm", _ALREADY_ALIGNED)
def test_load_catalog_already_aligned_paradigms(paradigm: str) -> None:
    """Paradigms where academic name == filename stem (epm/zero_maze/shoaling)."""
    cat = load_catalog(paradigm)
    assert cat.paradigm == paradigm


def test_load_catalog_unknown_paradigm_reports_alias_resolution() -> None:
    """Error message should mention both the input and the resolved filename
    so debugging is straightforward."""
    with pytest.raises(CatalogError) as exc:
        load_catalog("not_a_real_paradigm")
    msg = str(exc.value)
    assert "not_a_real_paradigm" in msg


def test_load_catalog_unknown_academic_alias_falls_through() -> None:
    """If an academic name isn't in the alias map, it's used as-is. Make sure
    that produces a clean unknown-paradigm error (not a KeyError or other crash)."""
    with pytest.raises(CatalogError) as exc:
        load_catalog("morris_water_maze")  # not in alias map, not a real catalog file
    assert "morris_water_maze" in str(exc.value)
```

跑：
```bash
cd packages/ethoinsight && uv run pytest tests/test_catalog_loader_aliases.py -v
```
期望：6+ 个 test 全过。

#### 4. 文档：在 `loader.py` 顶部添加 docstring 段说明 canonical key 政策

```python
"""加载 packages/ethoinsight/ethoinsight/catalog/<paradigm>.yaml 并构造
Catalog dataclass。校验失败一律抛 CatalogError，含 paradigm + 问题点。

Canonical paradigm key policy (2026-05-25):
  The system uses ACADEMIC NAMES as canonical paradigm keys:
    - forced_swim (file: fst.yaml)
    - tail_suspension (file: tst.yaml)
    - open_field (file: oft.yaml)
    - light_dark_box (file: ldb.yaml)
    - epm (file: epm.yaml)
    - zero_maze (file: zero_maze.yaml)
    - shoaling (file: shoaling.yaml)

  ``load_catalog`` accepts either the academic name (preferred) or the
  filename stem (legacy) and resolves to the correct YAML via _PARADIGM_ALIASES.
"""
```

---

## 不在本任务范围（明确列出，不要顺手做）

- **不要改 catalog yaml 文件名**（不要把 `fst.yaml` 改名为 `forced_swim.yaml`）。文件名保留短 key，loader 做映射就够了。改文件名会拖出 git 历史 + 所有 catalog yaml 引用更新，没必要。
- **不要改 catalog yaml 内的 `paradigm:` 字段**（不要把 `paradigm: fst` 改成 `paradigm: forced_swim`）。这字段是文件内部标识，跟文件名一致，对上游不可见。
- **不要改 dispatcher 的 if/elif paradigm == ... 链**。它已经用学术名，没问题。
- **不要改 `identify_ev19_template_tool.py`**——它的输出已经是学术名，符合 canonical 政策。它内部那个 `"fst": "forced_swim"` 映射表是**用户输入歧义解析**（用户输入"fst"也认）的，跟我们的 canonical 政策不冲突，**保留**。
- **不要改 `experiment_context.py`**——它已用学术名，符合 canonical。
- **不要改 skill `ethovision-paradigm-knowledge/references/by-experiment/*.md`**——它们的 slug 已经是学术名（`forced_swim` / `tail_suspension` / `open_field` / `light_dark_box`），符合 canonical。**这是关键检查点**：lead read `forced_swim.md` 看到 `slug: forced_swim`，传给 `prep_metric_plan(paradigm="forced_swim")`，loader 内部转 `fst.yaml` 加载——整条链一气呵成。
- **不要改 `plot_timeseries.py` line 28-31** 的 `"oft"/"fst"/"tst"` 短 key 映射。这是脚本内部"短 key → column name"的查表，跟 canonical 政策正交，**保留**。
- **不要改 `chart_maker.py` line 47** 的示例文本——它示意 `--paradigm fst`，是个示例字符串，alias 后两种都能跑，**不动**。
- **不要改 `test_catalog.py` 中的 `["epm", "oft", "fst", "tst", "ldb", "zero_maze", "shoaling"]` 参数化列表**。alias 后短 key 仍然支持，**保留以测试向后兼容**。
- **不要做 P0-P3 chart catalog 实施**（[../docs/handoffs/2026-05/2026-05-25-chart-catalog-p0-p3-implementation-handoff.md](2026-05-25-chart-catalog-p0-p3-implementation-handoff.md)）。那是后续的独立工作，不要混在 worktree。

---

## 工作流程

### 步骤

1. **创建 worktree**：

```bash
cd /home/wangqiuyang/noldus-insight
git worktree add -b paradigm-key-alignment .claude/worktrees/paradigm-key-alignment dev
cd .claude/worktrees/paradigm-key-alignment
```

2. **Read 上下文文件**（不要凭记忆做）：
   - 本 handoff 全部
   - `packages/ethoinsight/ethoinsight/catalog/loader.py`（特别是 `load_catalog` 函数）
   - 一个 catalog yaml 例子：`packages/ethoinsight/ethoinsight/catalog/fst.yaml`
   - `packages/agent/backend/packages/harness/deerflow/tools/builtins/prep_metric_plan_tool.py:64-100` (docstring 段)

3. **实施 4 项改动**（按上面"改动清单"顺序）

4. **测试**：

```bash
# (a) ethoinsight 全套测试
cd packages/ethoinsight
uv run pytest tests/ -q
# 期望: 全过, 新增的 test_catalog_loader_aliases.py 6+ 个测试也过

# (b) agent backend 全套测试（防回归）
cd ../agent/backend
.venv/bin/python -m pytest tests/ -q
# 期望: 3017 passed, 18 skipped, 0 failed（跟当前 dev HEAD 一致）

# (c) 手测 — 模拟 trace 9f77adcc 的失败场景
cd ../../ethoinsight
uv run python -c "
from ethoinsight.catalog.loader import load_catalog
# 学术名应该 OK
cat_acad = load_catalog('forced_swim')
print('academic:', cat_acad.paradigm)  # 期望 'fst' (yaml 内部字段)
# 短 key 也应该 OK
cat_short = load_catalog('fst')
print('short key:', cat_short.paradigm)  # 期望 'fst'
assert cat_acad.paradigm == cat_short.paradigm
print('OK: academic name and filename stem produce identical catalog')
"
```

5. **commit + push** 到 dev（worktree 内）：

```bash
cd /home/wangqiuyang/noldus-insight/.claude/worktrees/paradigm-key-alignment
git add packages/ethoinsight/ethoinsight/catalog/loader.py \
        packages/ethoinsight/tests/test_catalog_loader_aliases.py \
        packages/agent/backend/packages/harness/deerflow/tools/builtins/prep_metric_plan_tool.py
git commit -m "fix(catalog): load_catalog accept academic-name paradigm keys (forced_swim → fst.yaml)

5/25 trace 9f77adcc lead 调 prep_metric_plan(paradigm='forced_swim') 报
unknown_paradigm，第二次改 'fst' 才成功。根因: identify_ev19_template /
metrics dispatcher / skill doc 全部用学术名 (forced_swim/tail_suspension/...),
但 catalog YAML 文件名用短 key (fst.yaml/tst.yaml/...), load_catalog 仅按
文件名查找,造成 LLM 必须试错才能找到正确 key。

修复: catalog/loader.py 加 _PARADIGM_ALIASES 映射表,load_catalog 先按学术
名解析到文件名,再加载 yaml。Canonical policy 落在学术名:
  forced_swim → fst.yaml
  tail_suspension → tst.yaml
  open_field → oft.yaml
  light_dark_box → ldb.yaml
  epm / zero_maze / shoaling: 学术名 == 文件名 (无需 alias)

短 key (fst/tst/oft/ldb) 仍接受 — 向后兼容,所有现有调用 (plot_timeseries.py
内部映射、test_catalog.py 参数化、chart_maker.py docstring 示例) 无需改动。

prep_metric_plan_tool.py docstring 同步更新 paradigm 参数说明,把学术名标为
preferred,短 key 标为 legacy backward-compat。

测试:
  - 新增 test_catalog_loader_aliases.py (6 个测试: 4 个学术名映射 + 4 个
    短 key backward-compat + 3 个 already-aligned 范式 + 2 个错误处理)
  - ethoinsight pytest 全过
  - agent backend pytest 3017 passed 不退化

不在本次范围:
  - catalog yaml 文件不重命名 (fst.yaml 保留)
  - catalog yaml 内 paradigm: 字段不改
  - dispatcher/skill doc/experiment_context 已用学术名,不动

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
git push origin paradigm-key-alignment
```

6. **回到主仓库**告知用户：

```bash
cd /home/wangqiuyang/noldus-insight
# worktree 在 .claude/worktrees/paradigm-key-alignment, branch=paradigm-key-alignment, 已 push
```

留着 worktree 不删（用户可能要 review）。

---

## 验证标准

任何一项不达标就**不要 commit**：

- [ ] `load_catalog("forced_swim")` 返回的 Catalog 对象 `.paradigm == "fst"`
- [ ] `load_catalog("fst")` 同上返回（向后兼容）
- [ ] `load_catalog("tail_suspension")` 返回 `.paradigm == "tst"`
- [ ] `load_catalog("open_field")` 返回 `.paradigm == "oft"`
- [ ] `load_catalog("light_dark_box")` 返回 `.paradigm == "ldb"`
- [ ] `load_catalog("epm")` 返回 `.paradigm == "epm"`（已对齐）
- [ ] `load_catalog("not_a_paradigm")` 抛 `CatalogError` 且 message 含 input 名
- [ ] ethoinsight `pytest tests/ -q` 全过
- [ ] agent backend `pytest tests/ -q` 3017+ passed / 0 failed
- [ ] `test_catalog_loader_aliases.py` 至少 6 个测试，全过
- [ ] prep_metric_plan_tool.py docstring 提到学术名为 preferred + 短 key 为 legacy
- [ ] **不要碰**任何 "不在本任务范围" 段落里列的文件

---

## 风险评估

- **风险等级**: 低。改动只在 loader.py 一处加映射，逻辑加法不减法，向后兼容。
- **可能的 false positive**: 如果有调用方传了 alias 表里**已映射过**的名字（例如 `load_catalog("forced_swim_fst")`），会作为 unknown 报错——但这不是真正的风险，因为这种名字根本不存在。
- **回滚策略**: 如果发现下游某个隐藏调用方传入了未在 alias 表中的非标准 key，可直接 revert commit；现有调用方都已查过，不会有此问题。

---

## 给新 agent 的第一步

1. Read 本 handoff 全部
2. `git worktree add -b paradigm-key-alignment .claude/worktrees/paradigm-key-alignment dev` 切到隔离 worktree
3. Read 列出的上下文文件
4. 按"改动清单"顺序 1→2→3→4 实施
5. 跑全套验证（ethoinsight + agent backend）
6. 全绿后 commit + push 到 `paradigm-key-alignment` 分支
7. 报告用户

**绝对不要**：跑 P0-P3 chart 实施、改 catalog yaml 文件名、改 dispatcher、改 skill doc、改 identify_ev19_template、改 experiment_context。
