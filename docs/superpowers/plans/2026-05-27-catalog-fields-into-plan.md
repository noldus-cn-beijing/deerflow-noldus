# Catalog 判读字段下沉到 plan_metrics.json — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 catalog YAML 的 5 个判读 / 展示字段(`unit_zh` / `one_liner` / `output_unit` / `direction_for_anxiety` / `statistical_default`)透传下沉到 plan_metrics.json,删除 SKILL.md / data-analyst / report-writer 中"read catalog YAML"的引导,让 subagent 通过 sandbox 白名单允许的 `/mnt/user-data/workspace/plan_metrics.json` 一次性拿到所有需要的字段。

**Architecture:** 在 `ethoinsight.catalog.resolve._metric_to_plan()` + `plan_metrics_to_dict()` 一处透传 MetricEntry 已有的 5 个字段(`PlanMetric` dataclass 加默认值,保持向前兼容);`prep_metric_plan_tool` 通过 `plan_metrics_to_dict()` 间接受益,无需改 tool 本身;三个 prompt / skill 文件(SKILL.md / data_analyst.py / report_writer.py)的指引改为指向 plan_metrics.json。

**Tech Stack:** Python 3.10+(ethoinsight)/ 3.12+(agent backend),pytest,dataclasses,YAML(PyYAML),existing prep_metric_plan_tool + LocalSandboxProvider 白名单基础设施。

**Spec:** [docs/superpowers/specs/2026-05-27-catalog-fields-into-plan-design.md](../specs/2026-05-27-catalog-fields-into-plan-design.md)

---

## 仓库背景速读(若你没读过 spec)

- 仓库根: `/home/wangqiuyang/noldus-insight`(也是 git 仓库)。CLAUDE.md 在仓库根。
- 当前分支: `dev`。HEAD `94867afd`。**注意:dev 有 3 个文件未 commit**(deploy 相关),实施时先 `git stash` 或在新分支开展,完工后再处理。建议本计划用新分支 `fix/catalog-fields-into-plan`。
- 主要工作目录:
  - `packages/ethoinsight/`(纯 Python 库,Python 3.10+,pytest tests/)
  - `packages/agent/backend/`(deerflow harness fork,Python 3.12+,`make test` / `make lint`)
- catalog YAML 在 `packages/ethoinsight/ethoinsight/catalog/` 下,**不要改**。
- plan_metrics.json 是 lead agent 通过 `prep_metric_plan_tool` 在派遣 subagent 前生成的工作空间文件,subagent 通过 `/mnt/user-data/workspace/plan_metrics.json`(sandbox 虚拟路径,白名单允许)读取。

## 文件结构

实施只改动这些文件,任何"顺手优化"都属于 scope creep。

**需要修改:**
- `packages/ethoinsight/ethoinsight/catalog/schema.py`(PlanMetric dataclass 扩 5 字段)
- `packages/ethoinsight/ethoinsight/catalog/resolve.py`(`_metric_to_plan` 透传 + `plan_metrics_to_dict` / `plan_to_dict` 序列化)
- `packages/ethoinsight/ethoinsight/catalog/__init__.py`(docstring 同步)
- `packages/agent/skills/custom/ethoinsight-metric-catalog/SKILL.md`(删 catalog YAML 引导,加 plan 字段字典)
- `packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py`(prompt 改为读 plan_metrics.json)
- `packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py`(同上)
- `docs/superpowers/specs/2026-05-13-metric-catalog-architecture-design.md`(在文末加 Addendum 段)

**需要新建:**
- `packages/ethoinsight/tests/test_plan_metrics_interpretation_fields.py`(回归)
- `packages/agent/backend/tests/test_subagent_prompts_no_catalog_yaml.py`(lint 用,防回滚)
- 扩展 `packages/agent/backend/tests/test_prep_metric_plan_tool.py`(若已存在则扩,e2e 验证 tool 写出的 JSON 含新字段)

---

## Task 1: PlanMetric dataclass 扩 5 个字段

**Files:**
- Modify: `packages/ethoinsight/ethoinsight/catalog/schema.py:96-105`
- Test: `packages/ethoinsight/tests/test_catalog_schema_plan_split.py`(扩展已存在文件)

### Step 1: 写失败测试 — 验证 PlanMetric 接受 5 个新字段

- [ ] **Step 1: 在文件末尾新增测试函数**

打开 `packages/ethoinsight/tests/test_catalog_schema_plan_split.py`,在末尾追加(保留所有已有 test):

```python
def test_plan_metric_accepts_interpretation_fields():
    """W27: PlanMetric 必须能透传 catalog MetricEntry 的 5 个判读 / 展示字段。"""
    metric = PlanMetric(
        id="open_arm_time",
        script="ethoinsight.scripts.epm.compute_open_arm_time",
        input="/tmp/raw.txt",
        output="/tmp/m.json",
        required=True,
        reason="paradigm.default",
        subject_index=0,
        display_name_zh="开放臂时间",
        unit_zh="秒",
        one_liner="动物在开放臂区域的累计停留时间",
        output_unit="seconds",
        direction_for_anxiety="lower_is_anxious",
        statistical_default="groupwise_compare",
    )
    assert metric.unit_zh == "秒"
    assert metric.one_liner == "动物在开放臂区域的累计停留时间"
    assert metric.output_unit == "seconds"
    assert metric.direction_for_anxiety == "lower_is_anxious"
    assert metric.statistical_default == "groupwise_compare"


def test_plan_metric_interpretation_fields_default_safe():
    """PlanMetric 不传新字段时也能构造（向前兼容）。"""
    metric = PlanMetric(
        id="x",
        script="s",
        input="/i",
        output="/o",
        required=True,
        reason="paradigm.default",
    )
    assert metric.unit_zh == ""
    assert metric.one_liner == ""
    assert metric.output_unit == ""
    assert metric.direction_for_anxiety is None
    assert metric.statistical_default == ""
```

- [ ] **Step 2: 运行测试,确认 FAIL**

```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
pytest tests/test_catalog_schema_plan_split.py::test_plan_metric_accepts_interpretation_fields tests/test_catalog_schema_plan_split.py::test_plan_metric_interpretation_fields_default_safe -v
```

Expected: 两个 test 都 FAIL, `TypeError: PlanMetric.__init__() got an unexpected keyword argument 'unit_zh'`。

- [ ] **Step 3: 修改 PlanMetric dataclass**

编辑 `packages/ethoinsight/ethoinsight/catalog/schema.py`,把 `PlanMetric` 类(line 96-105 附近)改成:

```python
@dataclass
class PlanMetric:
    id: str
    script: str
    input: str
    output: str
    required: bool
    reason: str  # PlanReasonEnum
    subject_index: int = 0  # 0-based index into inputs.raw_files; 0 for single-subject plans
    display_name_zh: str = ""           # 1.1: 中文指标名，透传自 MetricEntry
    # W27 (2026-05-27): catalog 判读 / 展示字段透传到 plan,subagent 直接读 plan 即可,
    # 不再 read catalog YAML。详见 docs/superpowers/specs/2026-05-27-catalog-fields-into-plan-design.md
    unit_zh: str = ""
    one_liner: str = ""
    output_unit: str = ""
    direction_for_anxiety: str | None = None
    statistical_default: str = ""
```

- [ ] **Step 4: 运行测试,确认 PASS**

```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
pytest tests/test_catalog_schema_plan_split.py -v
```

Expected: 全部 PASS(包括两个新 test 和已有 test 不退化)。

- [ ] **Step 5: 创建分支并 commit**

```bash
cd /home/wangqiuyang/noldus-insight
git checkout -b fix/catalog-fields-into-plan
git add packages/ethoinsight/ethoinsight/catalog/schema.py packages/ethoinsight/tests/test_catalog_schema_plan_split.py
git commit -m "feat(catalog): PlanMetric dataclass 扩 5 个判读/展示字段(默认值向前兼容)"
```

---

## Task 2: `_metric_to_plan()` 透传 5 个字段

**Files:**
- Modify: `packages/ethoinsight/ethoinsight/catalog/resolve.py:427-463`
- Test: `packages/ethoinsight/tests/test_catalog_resolve_multi_subject.py`(扩展) 或新建 case 文件

### Step 1: 写失败测试 — resolve 出的 PlanMetric 含新字段

- [ ] **Step 1: 新建 test 文件**

创建 `packages/ethoinsight/tests/test_plan_metrics_interpretation_fields.py`:

```python
"""W27: 验证 resolve_metrics 把 catalog MetricEntry 的判读 / 展示字段透传到 PlanMetric。

不实际跑 metric 脚本,只验证字段透传契约。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ethoinsight.catalog.loader import load_catalog
from ethoinsight.catalog.resolve import resolve_metrics

# 每个范式列举一组足以让 default_metrics 全部通过 columns 检查的列名。
# 来源:packages/ethoinsight/ethoinsight/catalog/<paradigm>.yaml 的 requires_columns 字段。
_MINIMAL_COLUMNS: dict[str, list[str]] = {
    "epm": ["in_zone_open_arms_subject", "in_zone_closed_arms_subject", "x_center", "y_center", "velocity"],
    "oft": ["in_zone_center_subject", "in_zone_periphery_subject", "x_center", "y_center", "velocity", "in_zone_subject"],
    "fst": ["mobility_state", "mobility_state_highly_mobile", "velocity"],
    "ldb": ["in_zone_light_subject", "in_zone_dark_subject", "x_center", "y_center", "velocity"],
    "zero_maze": ["in_zone_open_subject", "in_zone_closed_subject", "x_center", "y_center", "velocity", "in_zone_subject"],
}

_EXPECTED_NEW_FIELDS = {
    "unit_zh",
    "one_liner",
    "output_unit",
    "direction_for_anxiety",
    "statistical_default",
}


@pytest.mark.parametrize("paradigm", sorted(_MINIMAL_COLUMNS))
def test_resolve_metrics_transfers_interpretation_fields(paradigm: str, tmp_path: Path) -> None:
    """每个范式 resolve 后,每个 PlanMetric 必须含 catalog MetricEntry 的同源字段值。"""
    raw_file = str(tmp_path / "dummy.txt")
    Path(raw_file).write_text("placeholder", encoding="utf-8")

    pm = resolve_metrics(
        paradigm=paradigm,
        columns=_MINIMAL_COLUMNS[paradigm],
        raw_files=[raw_file],
        workspace_dir=str(tmp_path),
    )

    assert pm.metrics, f"{paradigm}: 没有 metric 输出,说明 _MINIMAL_COLUMNS 不全"

    # 取 catalog 原文做断言对照(确保字段值同源,不是凭空填的默认值)
    cat = load_catalog(paradigm)
    catalog_by_id = {m.id: m for m in cat.default_metrics}

    for plan_metric in pm.metrics:
        # 1. dataclass 必须有这 5 个 attribute(默认值就行,但下一步检查同源)
        for fld in _EXPECTED_NEW_FIELDS:
            assert hasattr(plan_metric, fld), f"{paradigm}/{plan_metric.id}: PlanMetric 缺 {fld}"

        # 2. 字段值必须等于 catalog MetricEntry 同名字段
        entry = catalog_by_id[plan_metric.id]
        assert plan_metric.unit_zh == entry.unit_zh
        assert plan_metric.one_liner == entry.one_liner
        assert plan_metric.output_unit == entry.output_unit
        assert plan_metric.direction_for_anxiety == entry.direction_for_anxiety
        assert plan_metric.statistical_default == entry.statistical_default
```

- [ ] **Step 2: 运行测试,确认 FAIL**

```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
pytest tests/test_plan_metrics_interpretation_fields.py -v
```

Expected: 全部 FAIL,因为 `_metric_to_plan` 还没透传新字段,`plan_metric.unit_zh` 是默认 `""` 而 catalog `entry.unit_zh` 是中文实际值。

如果有的范式 `_MINIMAL_COLUMNS` 不足让所有 default_metrics 通过 columns 检查导致 `ResolveError: columns_missing`,先 print 一下漏掉的 pattern,补到 dict 里。这是 fixture 数据问题,**不是**测试逻辑问题。

- [ ] **Step 3: 修改 `_metric_to_plan`**

打开 `packages/ethoinsight/ethoinsight/catalog/resolve.py`,定位 `_metric_to_plan` 函数(line 427 附近),把构造 `PlanMetric` 的循环改成:

```python
def _metric_to_plan(
    m: MetricEntry,
    raw_files: list[str],
    workspace_dir: str,
    *,
    required: bool,
    reason: str,
    virtual_workspace_dir: str | None = None,
) -> list[PlanMetric]:
    """Expand one MetricEntry into N PlanMetric (one per raw_file).

    W27 (2026-05-27): 透传 catalog 判读 / 展示字段到 PlanMetric,subagent 不再 read catalog YAML。
    详见 docs/superpowers/specs/2026-05-27-catalog-fields-into-plan-design.md
    """
    if not raw_files:
        return []
    effective_workspace = virtual_workspace_dir or workspace_dir
    multi = len(raw_files) > 1
    plans: list[PlanMetric] = []
    for idx, raw_file in enumerate(raw_files):
        suffix = f"_s{idx}" if multi else ""
        output_path = str(Path(effective_workspace) / f"m_{m.id}{suffix}.json")
        plans.append(
            PlanMetric(
                id=m.id,
                script=m.script,
                input=raw_file,
                output=output_path,
                required=required,
                reason=reason,
                subject_index=idx,
                display_name_zh=m.display_name_zh,
                unit_zh=m.unit_zh,
                one_liner=m.one_liner,
                output_unit=m.output_unit,
                direction_for_anxiety=m.direction_for_anxiety,
                statistical_default=m.statistical_default,
            )
        )
    return plans
```

(注意保留原 docstring 的"Fix 2026-05-20 (FST E2E)"段内容 — 多 subject 展开逻辑不变。)

- [ ] **Step 4: 运行测试,确认 PASS**

```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
pytest tests/test_plan_metrics_interpretation_fields.py -v
```

Expected: 5 个范式 parametrize case 全 PASS。

- [ ] **Step 5: 跑全量 ethoinsight 测试确认不回归**

```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
pytest tests/ -v
```

Expected: 全绿。

- [ ] **Step 6: Commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/ethoinsight/ethoinsight/catalog/resolve.py packages/ethoinsight/tests/test_plan_metrics_interpretation_fields.py
git commit -m "feat(catalog): _metric_to_plan 透传 5 个判读/展示字段 + 范式参数化回归"
```

---

## Task 3: `plan_metrics_to_dict()` 和 `plan_to_dict()` 序列化新字段

**Files:**
- Modify: `packages/ethoinsight/ethoinsight/catalog/resolve.py:698-742, 745-785`
- Test: `packages/ethoinsight/tests/test_plan_metrics_interpretation_fields.py`(扩展)

### Step 1: 写失败测试 — dict 输出含新字段

- [ ] **Step 1: 在 `test_plan_metrics_interpretation_fields.py` 末尾追加 dict 序列化测试**

```python
from ethoinsight.catalog.resolve import plan_metrics_to_dict, plan_to_dict, resolve


@pytest.mark.parametrize("paradigm", sorted(_MINIMAL_COLUMNS))
def test_plan_metrics_to_dict_includes_interpretation_fields(paradigm: str, tmp_path: Path) -> None:
    """plan_metrics_to_dict() 输出的 metrics[] 项必须含 5 个新字段。"""
    raw_file = str(tmp_path / "dummy.txt")
    Path(raw_file).write_text("placeholder", encoding="utf-8")
    pm = resolve_metrics(
        paradigm=paradigm,
        columns=_MINIMAL_COLUMNS[paradigm],
        raw_files=[raw_file],
        workspace_dir=str(tmp_path),
    )
    d = plan_metrics_to_dict(pm)
    assert d["metrics"], f"{paradigm}: dict 的 metrics 为空"
    for m in d["metrics"]:
        for fld in _EXPECTED_NEW_FIELDS:
            assert fld in m, f"{paradigm}/{m['id']}: dict 缺 key {fld}"
        # 字段值类型契约
        assert isinstance(m["unit_zh"], str)
        assert isinstance(m["one_liner"], str)
        assert isinstance(m["output_unit"], str)
        assert m["direction_for_anxiety"] in (None, "lower_is_anxious", "higher_is_anxious")
        assert m["statistical_default"] in ("groupwise_compare", "paired_compare")


@pytest.mark.parametrize("paradigm", sorted(_MINIMAL_COLUMNS))
def test_plan_to_dict_includes_interpretation_fields(paradigm: str, tmp_path: Path) -> None:
    """plan_to_dict()(旧 wrapper)也必须含新字段,保持两个序列化器一致。"""
    raw_file = str(tmp_path / "dummy.txt")
    Path(raw_file).write_text("placeholder", encoding="utf-8")
    plan = resolve(
        paradigm=paradigm,
        columns=_MINIMAL_COLUMNS[paradigm],
        raw_files=[raw_file],
        workspace_dir=str(tmp_path),
    )
    d = plan_to_dict(plan)
    assert d["metrics"]
    for m in d["metrics"]:
        for fld in _EXPECTED_NEW_FIELDS:
            assert fld in m, f"plan_to_dict {paradigm}/{m['id']}: dict 缺 key {fld}"
```

- [ ] **Step 2: 运行测试,确认 FAIL**

```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
pytest tests/test_plan_metrics_interpretation_fields.py::test_plan_metrics_to_dict_includes_interpretation_fields tests/test_plan_metrics_interpretation_fields.py::test_plan_to_dict_includes_interpretation_fields -v
```

Expected: FAIL,因为 `plan_metrics_to_dict` 和 `plan_to_dict` 的字面量还没列新 key。

- [ ] **Step 3: 修改 `plan_metrics_to_dict()`**

打开 `packages/ethoinsight/ethoinsight/catalog/resolve.py`,定位 `plan_metrics_to_dict`(line 745 附近)。把 `"metrics"` 列表推导式改成:

```python
        "metrics": [
            {
                "id": m.id,
                "script": m.script,
                "input": m.input,
                "output": m.output,
                "required": m.required,
                "reason": m.reason,
                "subject_index": m.subject_index,
                "display_name_zh": m.display_name_zh,
                # W27 (2026-05-27): 透传 catalog 判读 / 展示字段
                "unit_zh": m.unit_zh,
                "one_liner": m.one_liner,
                "output_unit": m.output_unit,
                "direction_for_anxiety": m.direction_for_anxiety,
                "statistical_default": m.statistical_default,
            }
            for m in pm.metrics
        ],
```

- [ ] **Step 4: 同样修改 `plan_to_dict()`(旧 backward-compat wrapper)**

定位 `plan_to_dict`(line 698 附近)。同样把 metrics 列表推导式改成:

```python
        "metrics": [
            {
                "id": m.id,
                "script": m.script,
                "input": m.input,
                "output": m.output,
                "required": m.required,
                "reason": m.reason,
                "subject_index": m.subject_index,
                "display_name_zh": m.display_name_zh,
                # W27 (2026-05-27): 透传 catalog 判读 / 展示字段
                "unit_zh": m.unit_zh,
                "one_liner": m.one_liner,
                "output_unit": m.output_unit,
                "direction_for_anxiety": m.direction_for_anxiety,
                "statistical_default": m.statistical_default,
            }
            for m in plan.metrics
        ],
```

- [ ] **Step 5: 运行测试,确认 PASS**

```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
pytest tests/test_plan_metrics_interpretation_fields.py -v
```

Expected: 4 个 parametrize 函数全部 PASS。

- [ ] **Step 6: 跑全量 ethoinsight 测试,确认不回归**

```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
pytest tests/ -v
```

Expected: 全绿。若有 test 因为新 key 出现而 fail(例如某测试用 dict 相等断言),证明它没把 plan_metrics.json schema 当作可扩展契约 — 在该测试里改用 `dict.items() >= expected.items()` 或显式断言只关心的 key。

- [ ] **Step 7: Commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/ethoinsight/ethoinsight/catalog/resolve.py packages/ethoinsight/tests/test_plan_metrics_interpretation_fields.py
git commit -m "feat(catalog): plan_metrics_to_dict / plan_to_dict 输出 5 个判读/展示字段"
```

---

## Task 4: `prep_metric_plan_tool` e2e 验证(只加 test,不改代码)

**Files:**
- Test: `packages/agent/backend/tests/test_prep_metric_plan_tool.py`(扩展已存在文件)

### Step 1: 看现有 test 文件,找一个写得最完整的 paradigm fixture 抄风格

- [ ] **Step 1: 先 cat 现有 test 文件看风格**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
grep -n "def test_\|paradigm=" tests/test_prep_metric_plan_tool.py | head -30
```

记下其中一个 paradigm 用的:
- 怎么构造 `runtime` / `thread_data`
- 怎么准备 fake uploaded_files
- 怎么 assert plan_metrics.json 内容

照抄一份测试逻辑。

- [ ] **Step 2: 在文件末尾追加 e2e test**

在 `packages/agent/backend/tests/test_prep_metric_plan_tool.py` 末尾加(若现有 fixture 函数名不同,沿用现有命名):

```python
def test_prep_metric_plan_writes_interpretation_fields_into_json(tmp_path, monkeypatch):
    """W27: prep_metric_plan_tool 写出的 plan_metrics.json 必须含 5 个判读 / 展示字段。

    这是 catalog → resolve → plan_metrics_to_dict → JSON 全链路的 e2e 防线。
    任何环节漏字段,这里 fail。
    """
    # 准备一个 EPM 范式 minimal fixture(沿用现有 test 的 helper,见本文件其他 test)
    # 如果现有 test 已有 _make_runtime / _prepare_epm_fixture,直接用;否则照抄第一个 test 函数。

    # 1) 准备 uploaded EPM 文件(EthoVision XT 标准 header + 必需列)
    # ...沿用现有 fixture / helper...

    # 2) 调 prep_metric_plan_tool
    result = prep_metric_plan_tool.invoke(
        {"uploaded_files": [f"/mnt/user-data/uploads/{Path(fake_data).name}"], "paradigm": "epm"},
        config={"configurable": {"runtime": runtime}},  # 沿用现有 test 调用方式
    )
    assert result["status"] == "ok"

    # 3) 读 plan_metrics.json,验证字段
    plan_path = Path(result["plan_path"])
    # 注意 plan_path 是 sandbox 虚拟路径(/mnt/user-data/workspace/plan_metrics.json),
    # 实际写在 thread_data.workspace_path 下,读时要用 thread_data["workspace_path"] / "plan_metrics.json"
    plan_file = Path(thread_data["workspace_path"]) / "plan_metrics.json"
    plan = json.loads(plan_file.read_text(encoding="utf-8"))

    assert plan["metrics"], "no metrics in plan_metrics.json"
    for m in plan["metrics"]:
        for fld in ("unit_zh", "one_liner", "output_unit", "direction_for_anxiety", "statistical_default"):
            assert fld in m, f"metric {m['id']}: 缺字段 {fld}"
```

**注意**:这个 test **必须沿用现有 `test_prep_metric_plan_tool.py` 的 fixture / helper 风格**,不要硬生生重写一遍 runtime / thread_data 构造逻辑。先看现有 test 怎么写的,照抄。

- [ ] **Step 3: 运行 test,确认 PASS**

(因为 Task 3 已经让 `plan_metrics_to_dict` 输出新字段,这个 e2e test 应该一次就过。)

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
PYTHONPATH=. uv run pytest tests/test_prep_metric_plan_tool.py::test_prep_metric_plan_writes_interpretation_fields_into_json -v
```

Expected: PASS。

如果 FAIL,先检查:
1. fixture 是否正确把 ethoinsight package 安装到了 .venv(应该是 editable install,可跑 `python -c "import ethoinsight.catalog.resolve; print(ethoinsight.catalog.resolve.__file__)"` 看路径)
2. fixture 的 EPM 数据是否含必需列(`in_zone_open_arms_*` 等)

- [ ] **Step 4: 跑现有 prep_metric_plan_tool 全部 test 确认不回归**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
PYTHONPATH=. uv run pytest tests/test_prep_metric_plan_tool.py -v
```

Expected: 全绿。

- [ ] **Step 5: Commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/backend/tests/test_prep_metric_plan_tool.py
git commit -m "test(prep_metric_plan): e2e 验证 plan_metrics.json 写出新判读字段"
```

---

## Task 5: 重写 `ethoinsight-metric-catalog/SKILL.md`

**Files:**
- Modify: `packages/agent/skills/custom/ethoinsight-metric-catalog/SKILL.md`
- Test: `packages/agent/backend/tests/test_subagent_prompts_no_catalog_yaml.py`(新建,见 Task 7)

### 这一步只改文档,不写代码测试。Task 7 会有 lint 测试兜底。

- [ ] **Step 1: 读现有 SKILL.md(确认你改的版本和 spec 引用的一致)**

```bash
cd /home/wangqiuyang/noldus-insight
wc -l packages/agent/skills/custom/ethoinsight-metric-catalog/SKILL.md
```

Expected: 104 行左右。如果文件已有人改过,先和 spec 引用的内容比对,有冲突先打住问用户。

- [ ] **Step 2: 用下述全文替换 SKILL.md**

把 `packages/agent/skills/custom/ethoinsight-metric-catalog/SKILL.md` **完整替换**为:

```markdown
---
name: ethoinsight-metric-catalog
description: >
  EthoInsight 范式指标 catalog 消费手册。lead 通过 prep_metric_plan 工具
  在派遣 subagent 前生成 plan_metrics.json;data-analyst 和 report-writer
  从 plan_metrics.json 直接读判读 / 展示字段,不再 read catalog YAML 文件
  (它在 Python 包内,sandbox 不暴露)。
type: knowledge
version: 0.2.0
author: noldus-insight
---

# EthoInsight Metric Catalog 入口

## 设计契约(2026-05-27 起)

catalog YAML 仍是 single source of truth,但**只有 lead 通过 deerflow first-party
工具 `prep_metric_plan` 间接消费**(工具在 sandbox 外的 deerflow 进程内 import
ethoinsight.catalog,把解析结果写到 `/mnt/user-data/workspace/plan_metrics.json`)。

**subagent 不直接读 catalog YAML** — sandbox 的 read_file 白名单只放行
`/mnt/user-data/*` / `/mnt/skills/*` / `/mnt/acp-workspace/*` / `/mnt/shared/*`
和配置的 custom mounts。Python 包内路径会被 PermissionError 拒。

详见设计 spec:`docs/superpowers/specs/2026-05-27-catalog-fields-into-plan-design.md`

## 你是哪种 role?

### lead

在派遣 code-executor **之前**,调 `prep_metric_plan` 工具一步完成列名解析 + catalog resolve。

```
prep_metric_plan(uploaded_files=["/mnt/user-data/uploads/<raw_file>.txt", ...], paradigm="<epm|oft|fst|...>")
```

**参数**:
- `uploaded_files`: 用户上传的数据文件虚拟路径数组(把当前 `<uploaded_files>` 中所有相关文件都传进来)
- `paradigm`: 范式 ID(`epm` / `oft` / `fst` / `ldb` / `zero_maze`)

**成功返回** (`status="ok"`):
```json
{
  "status": "ok",
  "plan_path": "/mnt/user-data/workspace/plan_metrics.json",
  "plan_summary": {
    "paradigm": "epm",
    "metric_count": 5,
    "subject_count": 2,
    "metric_ids": ["open_arm_time_ratio", ...]
  }
}
```

**失败返回** (`status="error"`):

| error_code | 含义 | 怎么反问(hint) |
|------------|------|------------------|
| `file_not_found` | 数据文件不存在,可能用户上传失败 | ask_clarification 让用户重新上传 |
| `format_unrecognized` | 文件不是 EthoVision XT 导出格式 | ask_clarification 让用户确认导出方式 |
| `parse_failed` | 数据文件损坏 | ask_clarification 让用户重新导出 |
| `unknown_paradigm` | 范式不在 catalog 内 | ask_clarification 让用户确认范式或检查 set_experiment_paradigm 调用 |
| `columns_missing` | 数据缺关键列 | ask_clarification 让用户确认实验录制设置 |
| `schema_violation` | catalog YAML 损坏——项目内部 bug | present_files 让用户报 bug |
| `empty_plan` | 按当前参数一项指标都跑不了 | ask_clarification 确认用户需求 |
| `unknown_metric` | 用户要求的指标不在 catalog 中 | ask_clarification 让用户选择 |

工具返回的 `hint` 字段已含上述反问话术,可直接用于 ask_clarification。

**派遣 code-executor**

派遣 prompt 中只需要:

```
范式:{paradigm}
plan 路径:/mnt/user-data/workspace/plan_metrics.json
分组:/mnt/user-data/workspace/groups.json

请按 plan.metrics[]、plan.statistics、plan.charts[] 逐条执行
```

不要把指标清单展开在 prompt 里。

### data-analyst

判读某个指标时,read plan_metrics.json,按 metric id 在 `metrics[]` 数组中匹配:

```
read_file:
    /mnt/user-data/workspace/plan_metrics.json
```

关注字段(由 lead 派遣前 resolve 透传自 catalog YAML):

- `direction_for_anxiety`: `"lower_is_anxious"` / `"higher_is_anxious"` / `null`
  - lower_is_anxious: 该指标越低 → 焦虑样行为越明显
  - higher_is_anxious: 该指标越高 → 焦虑样行为越明显
  - null: 该指标不属于焦虑判读维度(如运动量 control)
- `statistical_default`: `"groupwise_compare"` / `"paired_compare"`(验证 code-executor 用了正确的统计入口)

多 subject 场景下同一 metric id 会出现多次(`subject_index` 区分),判读字段在所有同 id 行上一致,取首个即可。

**不要尝试 read catalog YAML 文件** — 它在 Python 包内,sandbox 不暴露。

### report-writer

写 Results / Discussion 段时,read plan_metrics.json,按 metric id 匹配:

```
read_file:
    /mnt/user-data/workspace/plan_metrics.json
```

展示字段(由 lead 派遣前 resolve 透传自 catalog YAML):

- `display_name_zh`: 中文展示名
- `unit_zh`: 中文单位
- `one_liner`: 一句话解释(仅首次提及该指标时引用)
- `output_unit`: 物理单位标识(seconds / count,辅助 LLM 理解 `unit_zh` 含义)

多 subject 场景下同一 metric id 会出现多次,展示字段在所有同 id 行上一致,取首个即可。

**不要尝试 read catalog YAML 文件** — 它在 Python 包内,sandbox 不暴露。

## metric 字段字典 reference(完整版)

每个 `metrics[]` 元素的全部字段:

| 字段 | 类型 | 用途 | 谁消费 |
|---|---|---|---|
| id | str | metric 唯一标识 | 所有 |
| script | str | Python 调用路径 | code-executor |
| input | str | 输入文件虚拟路径 | code-executor |
| output | str | 输出 JSON 虚拟路径 | code-executor |
| required | bool | 是否 default metric | code-executor |
| reason | str | 入选理由(paradigm.default / user.include / ...) | (诊断用) |
| subject_index | int | 0-based subject 序号 | code-executor / report-writer 区分 multi-subject |
| display_name_zh | str | 中文展示名 | report-writer / data-analyst |
| unit_zh | str | 中文单位 | report-writer |
| one_liner | str | 一句话解释 | report-writer(首次提及) |
| output_unit | str | 物理单位标识 | report-writer |
| direction_for_anxiety | "lower_is_anxious"/"higher_is_anxious"/null | 焦虑判读方向 | data-analyst |
| statistical_default | "groupwise_compare"/"paired_compare" | 默认统计入口 | data-analyst 校验 |
```

- [ ] **Step 3: 验证 markdown 渲染正常 + 词数**

```bash
cd /home/wangqiuyang/noldus-insight
wc -l packages/agent/skills/custom/ethoinsight-metric-catalog/SKILL.md
grep -c "catalog.*\.yaml\|read.*catalog.*YAML" packages/agent/skills/custom/ethoinsight-metric-catalog/SKILL.md
```

Expected:
- 文件 ~120 行
- 第二个 grep 输出 0(只剩"不要 read catalog YAML"这种**反面**引用,不含 yaml 路径字面量)— 但因为"catalog YAML"在"不要尝试 read catalog YAML 文件"句中仍出现,所以期望值是 2(每个 subagent role 有一条反面提示)。先 grep 然后确认是否所有出现都是反面警告语境,**没有任何"应该 read"的指引**。

如果出现非预期数量,核对哪一处忘改。

- [ ] **Step 4: Commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/skills/custom/ethoinsight-metric-catalog/SKILL.md
git commit -m "docs(skill): SKILL.md 改为引导 subagent 读 plan_metrics.json"
```

---

## Task 6: 改写 data_analyst.py 和 report_writer.py 的 prompt

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py:137-148`
- Modify: `packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py:163-175`

### Step 1: 改 data_analyst.py

- [ ] **Step 1: 先打开文件看上下文(确保你定位的行号对得上)**

```bash
cd /home/wangqiuyang/noldus-insight
sed -n '130,160p' packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py
```

确认你看到的是 "## 指标元数据查询" 段。

- [ ] **Step 2: 把 line 137-148 那段(从 `## 指标元数据查询` 到 `- statistical_default`)替换成新内容**

定位 `## 指标元数据查询` 起,到 `<failure>` 标签之前。将该段替换为:

```
## 指标元数据查询

每个指标的判读字段已由 lead 在派遣前 resolve 到 plan_metrics.json,从那里取:

read_file:
    /mnt/user-data/workspace/plan_metrics.json

按 metric id 在 `metrics[]` 数组中匹配,读取以下字段:
- direction_for_anxiety: "lower_is_anxious" / "higher_is_anxious" / null
- statistical_default: "groupwise_compare" / "paired_compare"

多 subject 场景下同一 metric id 会出现多次(subject_index 区分),判读字段在所有
同 id 行上一致,取首个即可。

**不要尝试 read catalog YAML 文件** — 它在 Python 包内,sandbox 不暴露给 subagent。
plan_metrics.json 已经包含 subagent 需要的全部字段;详见 ethoinsight-metric-catalog
skill 的字段字典 reference。
```

- [ ] **Step 3: 改 report_writer.py — 先看上下文**

```bash
sed -n '155,185p' packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py
```

确认你看到的是 "## 指标展示元数据查询" 段。

- [ ] **Step 4: 把 line 163-175 那段(从 `## 指标展示元数据查询` 到 `禁止在本 prompt 内硬编码`)替换成**:

```
## 指标展示元数据查询

每个指标的中文展示字段已下沉到 plan_metrics.json,从那里取:

read_file:
    /mnt/user-data/workspace/plan_metrics.json

按 metric id 在 `metrics[]` 数组中匹配,读取以下字段:
- display_name_zh: 中文展示名
- unit_zh: 中文单位
- one_liner: 一句话解释(仅首次提及该指标时引用,不要在每段重复)

多 subject 场景下同一 metric id 会出现多次(subject_index 区分),展示字段在所有
同 id 行上一致,取首个即可。

禁止在本 prompt 内硬编码任何指标的中文名或单位 —— 全部走 plan_metrics.json。
**不要尝试 read catalog YAML 文件** — 它在 Python 包内,sandbox 不暴露给 subagent。
```

- [ ] **Step 5: 跑 backend 全量 test 确认 prompt 改动不破坏 subagent 注册 / lint**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
PYTHONPATH=. uv run pytest tests/ -x -q 2>&1 | tail -40
make lint
```

Expected:
- pytest: 全绿(prompt 是字符串字面量,改动不影响测试)
- lint: 干净

注意:**make test 一定要跑完整 suite**,因为 deerflow harness 内可能有 prompt 内容快照测试(`tests/test_*_prompt.py` 类),改 prompt 字符串会让快照失败。如果失败,根据现有快照测试的更新机制(通常是 `pytest --snapshot-update` 或类似)更新快照,**不要**绕过测试。

- [ ] **Step 6: Commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py
git commit -m "feat(subagent): data-analyst / report-writer 改为读 plan_metrics.json 取判读/展示字段"
```

---

## Task 7: 加 lint 测试 — 防 prompt 回滚到 catalog YAML

**Files:**
- Create: `packages/agent/backend/tests/test_subagent_prompts_no_catalog_yaml.py`

### Step 1: 写 lint 测试

- [ ] **Step 1: 新建 test 文件**

创建 `packages/agent/backend/tests/test_subagent_prompts_no_catalog_yaml.py`:

```python
"""W27: lint — data-analyst / report-writer prompt 不许再引导读 catalog YAML 文件。

这是防回滚 lint。如果后续编辑误把 prompt 改回"read catalog YAML",此测试 fail。
"""
from __future__ import annotations

from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_SUBAGENTS_DIR = _BACKEND_ROOT / "packages" / "harness" / "deerflow" / "subagents" / "builtins"

_FILES_UNDER_LINT = [
    _SUBAGENTS_DIR / "data_analyst.py",
    _SUBAGENTS_DIR / "report_writer.py",
]

# 禁止短语:任何引导 subagent read catalog YAML 文件路径的写法
_FORBIDDEN_PATTERNS: list[str] = [
    # 路径写法
    "/path/to/ethoinsight/catalog/",
    "ethoinsight.catalog as c",  # SKILL.md 顶部的 import + __file__ 招数
    "c.__file__",
    # 显式 "read catalog YAML" 类指引(必须出现在 prompt 中作为反面警告)
    # — 我们不能完全 ban "catalog YAML" 字面量,因为反面警告里要用,
    #   所以下面 _ALLOWED_NEGATION_CONTEXTS 是允许出现的语境。
]

_ALLOWED_NEGATION_CONTEXTS = [
    "不要尝试 read catalog YAML",
    "不再 read catalog YAML",
]


@pytest.mark.parametrize("py_file", _FILES_UNDER_LINT, ids=lambda p: p.name)
def test_subagent_prompt_has_no_catalog_yaml_read_guidance(py_file: Path) -> None:
    """断言文件不含禁用模式;必要时允许在反面警告语境出现"catalog YAML"。"""
    text = py_file.read_text(encoding="utf-8")

    for forbidden in _FORBIDDEN_PATTERNS:
        assert forbidden not in text, (
            f"{py_file.name}: 仍含禁用模式 {forbidden!r}。"
            f" 该 subagent 应改为读 /mnt/user-data/workspace/plan_metrics.json。"
            f" 详见 docs/superpowers/specs/2026-05-27-catalog-fields-into-plan-design.md"
        )

    # "catalog YAML" 字面量必须只出现在反面警告语境
    if "catalog YAML" in text:
        # 把文本切到 "catalog YAML" 周围 60 字符的窗,验证存在允许的反面短语
        idx = 0
        while True:
            pos = text.find("catalog YAML", idx)
            if pos == -1:
                break
            window = text[max(0, pos - 40): pos + 30]
            assert any(neg in window for neg in _ALLOWED_NEGATION_CONTEXTS), (
                f"{py_file.name}: 'catalog YAML' 出现在非反面警告语境(window={window!r}). "
                f"必须改成不要引导 subagent 读 catalog YAML 文件。"
            )
            idx = pos + len("catalog YAML")
```

- [ ] **Step 2: 跑 test,确认 PASS(因为 Task 6 已经清掉了引导)**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
PYTHONPATH=. uv run pytest tests/test_subagent_prompts_no_catalog_yaml.py -v
```

Expected: 2 个 parametrize case 全 PASS。

如果 FAIL,说明 Task 6 没改干净,回去补。

- [ ] **Step 3: Commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/backend/tests/test_subagent_prompts_no_catalog_yaml.py
git commit -m "test(subagent): lint 防 data-analyst / report-writer prompt 回滚到 catalog YAML"
```

---

## Task 8: 文档对齐 + spec Addendum

**Files:**
- Modify: `packages/ethoinsight/ethoinsight/catalog/__init__.py`(docstring)
- Modify: `docs/superpowers/specs/2026-05-13-metric-catalog-architecture-design.md`(末尾加 Addendum)

### Step 1: 改 catalog/__init__.py docstring

- [ ] **Step 1: 编辑 docstring**

打开 `packages/ethoinsight/ethoinsight/catalog/__init__.py`,把 docstring(line 1-9 附近)改成:

```python
"""ethoinsight.catalog — 范式 → 指标 catalog 模块.

承载 single source of truth:每个 paradigm 一份 YAML 文件,定义默认指标清单 +
脚本路径 + 列要求 + 展示元数据 + 判读方向性。

运行时消费契约(2026-05-27 起):
  - **lead agent**: 通过 deerflow first-party 工具 `prep_metric_plan` 间接消费 —
    工具在 sandbox 外的 deerflow 进程内调 resolve_metrics(),把结果写到
    /mnt/user-data/workspace/plan_metrics.json
  - **subagent(data-analyst / report-writer)**: 不直接读 catalog YAML,
    从 plan_metrics.json 取所有判读 / 展示字段
  - **dispatcher / 单测 / golden-case**: 直接 import 使用(沙箱外环境)

设计 spec:
  - docs/superpowers/specs/2026-05-13-metric-catalog-architecture-design.md(原始架构)
  - docs/superpowers/specs/2026-05-27-catalog-fields-into-plan-design.md(消费契约修正)
"""
```

- [ ] **Step 2: 在 2026-05-13 原 spec 文末追加 Addendum**

打开 `docs/superpowers/specs/2026-05-13-metric-catalog-architecture-design.md`,在文件末尾追加(不删除任何已有内容):

```markdown

---

## Addendum 2026-05-27 — subagent 消费路径修正

原 spec 设想 catalog YAML "被 lead / data-analyst / report-writer 多方共读"。这条
契约在运行时不成立 — deerflow sandbox 的 read_file 白名单不放行 Python 包内路径
(`/app/backend/packages/ethoinsight/ethoinsight/catalog/*.yaml` 会被
PermissionError 拒)。

修正后契约:catalog YAML **仅** 由 lead 通过 `prep_metric_plan` 工具在沙箱外消费,
所有 subagent 需要的字段(`unit_zh` / `one_liner` / `output_unit` /
`direction_for_anxiety` / `statistical_default` 等)由 resolve.py 透传到
plan_metrics.json,subagent 走 `/mnt/user-data/workspace/plan_metrics.json`
(白名单允许)一次读出。

详细设计:[2026-05-27-catalog-fields-into-plan-design.md](2026-05-27-catalog-fields-into-plan-design.md)
```

- [ ] **Step 3: Commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/ethoinsight/ethoinsight/catalog/__init__.py docs/superpowers/specs/2026-05-13-metric-catalog-architecture-design.md
git commit -m "docs: catalog __init__ docstring + 2026-05-13 spec Addendum 反映新消费契约"
```

---

## Task 9: 全量验证 + 手工 dev 复现

### Step 1: 跑全量后端测试

- [ ] **Step 1: ethoinsight 库全量**

```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
pytest tests/ -v 2>&1 | tail -30
```

Expected: 全绿。

- [ ] **Step 2: deerflow harness 全量 + lint**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
make test 2>&1 | tail -30
make lint
```

Expected:
- make test: 全绿
- make lint: 干净

如果 make test 因 prompt 快照失败(Task 6 已警告过),按现有快照机制更新,**不要**跳过测试。

### Step 2: 本地 make dev 手工复现(端到端验证)

- [ ] **Step 3: 启动本地 dev 环境**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent
make dev
```

等待启动完成,访问 http://localhost:2026

- [ ] **Step 4: 跑一次 FST 单样本场景**

复现原 handoff 中的症状路径:
1. 上传一份 FST 单样本数据(用 demo-data 里的 FST 样例,或之前 dogfood 用过的文件)
2. 让 lead 走完 prep_metric_plan → 派遣 data-analyst
3. 观察 data-analyst thinking(在浏览器内或后端日志里)

**Expected 行为变化**:

- ✅ data-analyst thinking **不再** 出现 "The catalog YAML file wasn't found at that path"
- ✅ data-analyst thinking **不再** 出现 "Let me try to find it via the python import approach"
- ✅ data-analyst 应该直接 read `/mnt/user-data/workspace/plan_metrics.json`,然后开始分析

如果以上任何一条不符合,说明 prompt 改动没生效或 prompt 还有残留引导;回去检查 Task 5 / Task 6 的改动是否落到了对应文件。

- [ ] **Step 5: 关闭 dev**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent
make stop
```

- [ ] **Step 6: 把 dev 验证观察写成 handoff 段**

不创建新 handoff 文件;在最终 commit message 或 PR description 里记一句:

```
本地 make dev 实测:FST 单样本场景 data-analyst thinking 不再尝试 read catalog YAML,
直接走 plan_metrics.json,trace 时间从 ~10 步猜路径减到 1 步直读。
```

---

## Task 10: PR 准备

### Step 1: 看 commit 历史 + 状态

- [ ] **Step 1: 看本分支 commit list**

```bash
cd /home/wangqiuyang/noldus-insight
git log --oneline dev..HEAD
git status
```

Expected: 7-8 个 commit(Task 1-8 的产物),工作区干净。

- [ ] **Step 2: 推到远端 + 开 PR**

```bash
git push -u origin fix/catalog-fields-into-plan
gh pr create --base dev --title "fix(catalog): 判读字段下沉到 plan_metrics.json,subagent 不再 read catalog YAML" --body "$(cat <<'EOF'
## Summary

- 根因:`ethoinsight-metric-catalog/SKILL.md` 引导 subagent 通过 `import ethoinsight.catalog; c.__file__` 拿包内物理路径再 read_file,但 deerflow sandbox 的 read_file 白名单只放行 `/mnt/user-data/*` / `/mnt/skills/*` / `/mnt/acp-workspace/*` / `/mnt/shared/*` / 配置的 custom mounts,Python 包内路径会被 PermissionError 拒
- 修复:把 catalog 的 5 个判读 / 展示字段透传到 plan_metrics.json(`unit_zh` / `one_liner` / `output_unit` / `direction_for_anxiety` / `statistical_default`),subagent 走 `/mnt/user-data/workspace/plan_metrics.json`(白名单允许)一次读出
- 删除 SKILL.md / data-analyst / report-writer 中"read catalog YAML"的引导;加 lint 测试防回滚

## Spec
docs/superpowers/specs/2026-05-27-catalog-fields-into-plan-design.md

## Test plan
- [x] `pytest packages/ethoinsight/tests/` 全绿
- [x] `cd packages/agent/backend && make test` 全绿
- [x] `make lint` 干净
- [x] 本地 make dev 实测:FST 单样本场景 data-analyst thinking 不再猜路径,直接读 plan_metrics.json

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: 把 PR URL 报给用户**

`gh pr create` 输出的 URL 复制给用户,等他 review / merge。

---

## Self-Review 已完成

- **Spec coverage**: §3.3 改动 1-7 → Task 1-8 一一对应;§3.4 测试 1-3 → Task 2/3/4/7;§3.5 文档对齐 → Task 8;§5 风险通过 TDD + lint 测试 + 多 paradigm parametrize 覆盖。
- **Placeholder 扫描**: 无 TBD / TODO。所有代码块带完整内容。Task 4 唯一的"沿用现有 fixture"指引是因为 prep_metric_plan_tool 现有 test 风格强烈;实施时打开文件就清楚。
- **Type 一致性**: PlanMetric 5 个新字段名在 Task 1(dataclass)/ Task 2(_metric_to_plan)/ Task 3(plan_metrics_to_dict / plan_to_dict)/ Task 5(SKILL.md 字段字典)中保持完全一致。
- **顺序约束**: Task 1 → 2 → 3 是 ethoinsight 库链路(必须按顺序);Task 5 / 6 / 7 是 agent 端 prompt 链路(5 → 6 → 7 顺序);Task 4 可并行(只要 Task 3 完成);Task 8 / 9 / 10 最后。

