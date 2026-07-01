# C2b 画廊后端产物注册表 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `artifacts.py` 三个并列 list 端点各自 ad-hoc 的 kind/ext + meta 构造收敛成一个 `ArtifactRegistry` SSOT + `list_artifacts_by_kind` 分发，端点变薄门面，行为字节级不变。

**Architecture:** 现状三个 list 函数（`list_chart_artifacts` / `list_report_artifacts` / `list_data_artifacts`）**已经是 module-level 函数**，端点已委托它们。C2b 加一个静态注册表 dict（`kind → builder`）+ 一个 `list_artifacts_by_kind(kind, thread_id, request)` 分发器，端点改为调分发器。builder 引用现有函数，**逻辑零改**——纯收敛调用入口。

**Tech Stack:** Python 3.12, FastAPI, pytest。后端 `app/gateway/` 层（deerflow 子树外）。

## Global Constraints

- **行为字节级不变**：重构后 `/charts` `/reports` `/data` 输出与重构前**逐字节相等**。这是纯重构，验收=等价。
- **只收敛现有三个 list 端点**（charts/reports/data）。**不新增产物类型。**
- **不动** `/metrics-table`（单文件 JSON）、`/data-table`（CSV 下载）、catch-all `/{path:path}` serve 端点——它们非「列表」逻辑。
- **不改前端**：`useThreadAssets` 消费的 JSON 契约字节级不变。
- **零改 deerflow 子树**（`packages/harness/deerflow/`）——全部在 `app/gateway/`，sync 友好。
- 测试放 `packages/agent/backend/tests/`，命名 `test_<feature>.py`。运行 `cd packages/agent/backend && make test`。
- **防 vacuous**：对账测试必须能在 builder 指错时变红（见 Task 3）。

---

### Task 1: 建 `ArtifactRegistry` 注册表 SSOT

**Files:**
- Create: `packages/agent/backend/app/gateway/artifact_registry.py`
- Test: `packages/agent/backend/tests/test_artifact_registry.py`

**Interfaces:**
- Consumes: 现有 `app.gateway.routers.artifacts` 的三个函数 `list_chart_artifacts(thread_id, request)` / `list_report_artifacts(thread_id)` / `list_data_artifacts(thread_id)`。
- Produces:
  - `KIND_BY_EXT: dict[str, str]` — `{".png":"chart", ".md":"report", ".html":"report", ".htm":"report", ".csv":"data", ".json":"data"}`
  - `kind_for_ext(ext: str) -> str | None` — 小写化 ext（带或不带前导点均可），返回 kind 或 None。
  - `REGISTRY: dict[str, str]` — `{"chart": "list_chart_artifacts", "report": "list_report_artifacts", "data": "list_data_artifacts"}`（builder 函数名，供 Task 2 分发；用名字避免顶层 import 环）。

> **为何用函数名字符串而非直接 import**：`artifact_registry.py` 若顶层 `from app.gateway.routers.artifacts import ...` 会与 artifacts.py 反向 import（Task 2 分发器要在 artifacts.py 里用注册表）形成环。注册表只存**名字**，Task 2 的分发器在 artifacts.py 内按名字取本模块函数——零环。

- [ ] **Step 1: Write the failing test**

```python
# packages/agent/backend/tests/test_artifact_registry.py
from app.gateway.artifact_registry import KIND_BY_EXT, REGISTRY, kind_for_ext


def test_kind_for_ext_maps_known_extensions():
    assert kind_for_ext(".png") == "chart"
    assert kind_for_ext(".md") == "report"
    assert kind_for_ext(".html") == "report"
    assert kind_for_ext(".htm") == "report"
    assert kind_for_ext(".csv") == "data"
    assert kind_for_ext(".json") == "data"


def test_kind_for_ext_is_case_insensitive_and_dot_optional():
    assert kind_for_ext(".PNG") == "chart"
    assert kind_for_ext("png") == "chart"


def test_kind_for_ext_unknown_returns_none():
    assert kind_for_ext(".xyz") is None
    assert kind_for_ext("") is None


def test_registry_maps_each_kind_to_builder_name():
    assert REGISTRY == {
        "chart": "list_chart_artifacts",
        "report": "list_report_artifacts",
        "data": "list_data_artifacts",
    }


def test_kind_by_ext_covers_registry_kinds():
    # Every kind reachable by ext must have a builder in REGISTRY.
    assert set(KIND_BY_EXT.values()) <= set(REGISTRY.keys())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/agent/backend && PYTHONPATH=. uv run pytest tests/test_artifact_registry.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.gateway.artifact_registry'`

- [ ] **Step 3: Write minimal implementation**

```python
# packages/agent/backend/app/gateway/artifact_registry.py
"""Artifact list registry (SSOT): ext→kind + kind→builder-name.

Consolidates the ad-hoc ext/kind logic that used to be duplicated across the
/charts, /reports, /data list endpoints. Pure data + one pure function — no
deerflow imports, no import of artifacts.py (holds builder *names*, resolved
by the dispatcher inside artifacts.py to avoid a reverse-import cycle).

Scope (spec 2026-07-01 C2b): only the three disk-list endpoints. Does NOT
cover /metrics-table (single file), /data-table (CSV download), or the
catch-all serve endpoint.
"""

from __future__ import annotations

# ext (lowercased, with leading dot) → artifact kind.
KIND_BY_EXT: dict[str, str] = {
    ".png": "chart",
    ".md": "report",
    ".html": "report",
    ".htm": "report",
    ".csv": "data",
    ".json": "data",
}

# kind → builder function NAME in app.gateway.routers.artifacts.
# (Name, not the function itself, to avoid an import cycle: the dispatcher
# lives in artifacts.py and resolves these names against that module.)
REGISTRY: dict[str, str] = {
    "chart": "list_chart_artifacts",
    "report": "list_report_artifacts",
    "data": "list_data_artifacts",
}


def kind_for_ext(ext: str) -> str | None:
    """Map a file extension (with or without leading dot, any case) to a kind."""
    if not ext:
        return None
    normalized = ext.lower()
    if not normalized.startswith("."):
        normalized = "." + normalized
    return KIND_BY_EXT.get(normalized)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/agent/backend && PYTHONPATH=. uv run pytest tests/test_artifact_registry.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add packages/agent/backend/app/gateway/artifact_registry.py packages/agent/backend/tests/test_artifact_registry.py
git commit -m "feat(artifacts): C2b Task1 — ArtifactRegistry SSOT (ext→kind + kind→builder-name)"
```

---

### Task 2: `list_artifacts_by_kind` 分发器 + 端点改薄门面

**Files:**
- Modify: `packages/agent/backend/app/gateway/routers/artifacts.py` (add dispatcher; rewrite 3 endpoint bodies)
- Test: `packages/agent/backend/tests/test_artifacts_list_by_kind.py`

**Interfaces:**
- Consumes: `app.gateway.artifact_registry.REGISTRY` (Task 1); existing `list_chart_artifacts(thread_id, request)` / `list_report_artifacts(thread_id)` / `list_data_artifacts(thread_id)`.
- Produces: `list_artifacts_by_kind(kind: str, thread_id: str, request: Request | None) -> list[dict[str, Any]]` — resolves the builder by name from REGISTRY, calls it (passing `request` only to the chart builder, which is the only one that takes it), returns its output verbatim.

> **签名差异处理**：现有三个 builder 签名不一致——`list_chart_artifacts(thread_id, request)` 收 request，`list_report_artifacts(thread_id)` / `list_data_artifacts(thread_id)` 不收。分发器按 builder 名判断是否传 request（chart 传，其余不传），保持每个 builder 调用与重构前逐字节一致。

- [ ] **Step 1: Write the failing test**

```python
# packages/agent/backend/tests/test_artifacts_list_by_kind.py
from app.gateway.routers import artifacts as art


def test_dispatch_chart_matches_direct_call(monkeypatch):
    sentinel = [{"path": "/x.png", "kind": "chart"}]
    monkeypatch.setattr(art, "list_chart_artifacts", lambda tid, req: sentinel)
    assert art.list_artifacts_by_kind("chart", "t1", None) == sentinel


def test_dispatch_report_matches_direct_call(monkeypatch):
    sentinel = [{"path": "/r.md", "kind": "report"}]
    monkeypatch.setattr(art, "list_report_artifacts", lambda tid: sentinel)
    assert art.list_artifacts_by_kind("report", "t1", None) == sentinel


def test_dispatch_data_matches_direct_call(monkeypatch):
    sentinel = [{"path": "/d.json", "kind": "data"}]
    monkeypatch.setattr(art, "list_data_artifacts", lambda tid: sentinel)
    assert art.list_artifacts_by_kind("data", "t1", None) == sentinel


def test_dispatch_unknown_kind_raises():
    import pytest
    with pytest.raises(KeyError):
        art.list_artifacts_by_kind("nope", "t1", None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd packages/agent/backend && PYTHONPATH=. uv run pytest tests/test_artifacts_list_by_kind.py -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'list_artifacts_by_kind'`

- [ ] **Step 3: Write minimal implementation**

Add the dispatcher near the other `list_*` functions in `artifacts.py` (after `list_data_artifacts`, before its endpoint). Add the import at top with the other imports:

```python
from app.gateway.artifact_registry import REGISTRY
```

```python
def list_artifacts_by_kind(kind: str, thread_id: str, request: Request | None) -> list[dict[str, Any]]:
    """Dispatch to the registered builder for ``kind`` (spec 2026-07-01 C2b).

    Resolves the builder NAME from REGISTRY against this module, then calls it.
    Only the chart builder takes ``request``; the others take just thread_id —
    the call shape per kind is byte-identical to the pre-refactor endpoints.
    """
    builder_name = REGISTRY[kind]  # KeyError on unknown kind — intentional (registry is SSOT).
    builder = globals()[builder_name]
    if kind == "chart":
        return builder(thread_id, request)
    return builder(thread_id)
```

Now rewrite the 3 endpoint bodies to delegate. Replace the body of `get_chart_artifacts`:

```python
@router.get(
    "/threads/{thread_id}/artifacts/charts",
    summary="List Chart Artifacts (disk + plan_charts.json)",
    description="List all generated chart images for the thread, joined with plan_charts.json metadata. Disk is the source of truth (independent of LangGraph state bubbling).",
)
@require_permission("threads", "read", owner_check=True)
async def get_chart_artifacts(thread_id: str, request: Request) -> Response:
    return Response(
        content=json.dumps(list_artifacts_by_kind("chart", thread_id, request)),
        media_type="application/json",
    )
```

Replace the body of `get_report_artifacts`:

```python
@router.get(
    "/threads/{thread_id}/artifacts/reports",
    summary="List Report Artifacts (disk)",
    description="List all report/document artifacts (.md/.html) on disk for the thread. Disk is the source of truth (independent of LangGraph state bubbling).",
)
@require_permission("threads", "read", owner_check=True)
async def get_report_artifacts(thread_id: str, request: Request) -> Response:
    return Response(
        content=json.dumps(list_artifacts_by_kind("report", thread_id, request)),
        media_type="application/json",
    )
```

Replace the body of `get_data_artifacts`:

```python
@router.get(
    "/threads/{thread_id}/artifacts/data",
    summary="List Data Artifacts (disk)",
    description="List metrics-table data artifacts on disk (metrics_table.json) for the thread. Disk is the source of truth.",
)
@require_permission("threads", "read", owner_check=True)
async def get_data_artifacts(thread_id: str, request: Request) -> Response:
    return Response(
        content=json.dumps(list_artifacts_by_kind("data", thread_id, request)),
        media_type="application/json",
    )
```

> Keep `list_chart_artifacts` / `list_report_artifacts` / `list_data_artifacts` **exactly as they are** — the dispatcher calls them unchanged.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd packages/agent/backend && PYTHONPATH=. uv run pytest tests/test_artifacts_list_by_kind.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Bare-import check (no cycle) + full artifacts suite**

Run: `cd packages/agent/backend && PYTHONPATH=. python -c "import app.gateway" && PYTHONPATH=. uv run pytest tests/test_artifacts_archive_router.py tests/test_artifacts_metrics_table_endpoint.py -v`
Expected: import exits 0 (no cycle); existing artifacts tests PASS.

- [ ] **Step 6: Commit**

```bash
git add packages/agent/backend/app/gateway/routers/artifacts.py packages/agent/backend/tests/test_artifacts_list_by_kind.py
git commit -m "feat(artifacts): C2b Task2 — list_artifacts_by_kind dispatcher + thin endpoint facades"
```

---

### Task 3: 字节级对账回归测试（防 vacuous）

**Files:**
- Test: `packages/agent/backend/tests/test_artifacts_registry_equivalence.py`

**Interfaces:**
- Consumes: `list_artifacts_by_kind` (Task 2); the three `list_*` functions (unchanged).

> **这是 C2b 的验收核心**：证明「经注册表分发」与「直接调 builder」输出逐字节相等，且证明测试非 vacuous（builder 指错时变红）。守 memory `feedback_pr115_stage1_equivalence_baseline_is_hollow`——对账 baseline 必须打开看内容。

- [ ] **Step 1: Write the equivalence + anti-vacuous test**

```python
# packages/agent/backend/tests/test_artifacts_registry_equivalence.py
"""C2b acceptance: registry dispatch is byte-identical to direct builder calls,
and the equivalence assertion is non-vacuous (fails if a builder is mis-wired)."""
import json

import pytest

from app.gateway.routers import artifacts as art


@pytest.fixture
def fake_builders(monkeypatch):
    """Distinct, content-rich outputs per kind so a mis-wire is detectable."""
    chart = [{"path": "/mnt/user-data/outputs/p.png", "kind": "chart", "chart_type": "box"}]
    report = [{"path": "/mnt/user-data/outputs/r.html", "kind": "report", "filename": "r.html"}]
    data = [{"path": "/mnt/user-data/outputs/metrics_table.json", "kind": "data", "filename": "metrics_table.json"}]
    monkeypatch.setattr(art, "list_chart_artifacts", lambda tid, req: chart)
    monkeypatch.setattr(art, "list_report_artifacts", lambda tid: report)
    monkeypatch.setattr(art, "list_data_artifacts", lambda tid: data)
    return {"chart": chart, "report": report, "data": data}


@pytest.mark.parametrize("kind", ["chart", "report", "data"])
def test_dispatch_byte_identical_to_direct_builder(kind, fake_builders):
    via_registry = art.list_artifacts_by_kind(kind, "t1", None)
    direct = fake_builders[kind]
    # Byte-level equivalence: serialize both, compare exact bytes.
    assert json.dumps(via_registry) == json.dumps(direct)


def test_equivalence_is_non_vacuous(monkeypatch, fake_builders):
    """Anti-vacuous probe: mis-wire the chart builder to the report output →
    the byte-equivalence assertion for 'chart' MUST now fail."""
    # Point chart's builder at the report content (simulate a registry mis-wire).
    monkeypatch.setattr(art, "list_chart_artifacts", lambda tid, req: fake_builders["report"])
    via_registry = art.list_artifacts_by_kind("chart", "t1", None)
    assert json.dumps(via_registry) != json.dumps(fake_builders["chart"])
```

- [ ] **Step 2: Run to verify it passes (equivalence holds; anti-vacuous probe green because it asserts inequality)**

Run: `cd packages/agent/backend && PYTHONPATH=. uv run pytest tests/test_artifacts_registry_equivalence.py -v`
Expected: PASS (4 tests: 3 parametrized + 1 anti-vacuous).

- [ ] **Step 3: Prove the probe truly bites (manual sanity, then revert)**

Temporarily break the dispatcher in `artifacts.py` — make `list_artifacts_by_kind` always call the report builder regardless of kind:

```python
    builder = globals()["list_report_artifacts"]  # TEMP break
    return builder(thread_id)
```

Run: `cd packages/agent/backend && PYTHONPATH=. uv run pytest tests/test_artifacts_registry_equivalence.py::test_dispatch_byte_identical_to_direct_builder -v`
Expected: FAIL for kind=chart and kind=data (proves the equivalence test is not vacuous). **Then revert the TEMP break** and re-run — Expected: PASS.

- [ ] **Step 4: Full backend suite green**

Run: `cd packages/agent/backend && make test`
Expected: no new failures vs baseline (run `make test` once before Task 1 to record baseline count).

- [ ] **Step 5: Commit**

```bash
git add packages/agent/backend/tests/test_artifacts_registry_equivalence.py
git commit -m "test(artifacts): C2b Task3 — byte-level equivalence + anti-vacuous probe"
```

---

## Self-Review

**Spec coverage:**
- 注册表 SSOT (ext→kind + 每 kind builder) → Task 1 ✅
- 端点变薄门面 → Task 2 ✅
- 对账测试逐字节相等 + 防 vacuous 探针 → Task 3 ✅
- chart plan_charts.json 元数据保真 → builder 零改，Task 3 对账覆盖 ✅
- 不动 /metrics-table /data-table /catch-all → Global Constraints + Task 2 只改 3 端点 ✅
- 前端契约不变 → 端点仍 `json.dumps(list)`，结构不变 ✅
- 零改 deerflow 子树 → 全在 app/gateway/ ✅

**Placeholder scan:** 无 TBD/TODO；每步有完整代码。

**Type consistency:** `list_artifacts_by_kind(kind, thread_id, request)` 签名在 Task 2 定义、Task 3 一致使用；`REGISTRY` / `kind_for_ext` 在 Task 1 定义、Task 2 消费一致。

**Note:** `kind_for_ext` 在 Task 1 建出但 Task 2/3 未消费——它是注册表 SSOT 的对外查询接口（未来加类型/前端一致性用），spec「ext→kind 映射 SSOT」要求它存在。保留，非孤儿。
