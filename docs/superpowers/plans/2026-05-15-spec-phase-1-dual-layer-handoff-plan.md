# Spec 阶段 1: 双层 Handoff 协议（L1/L2）实施 Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 落地 [2026-05-14 spec §2 + §4 + §8 阶段 1](../specs/2026-05-14-handoff-protocol-and-runtime-isolation-spec.md) 定义的"双层 handoff 协议"——L1 摘要（< 5 KB，进 message history，所有 agent 默认消费）+ L2 完整数据（落 workspace 文件，例外升级路径）。配合 `handoff_suffix` 让"同一 subagent 被多次调用"产物物理隔离、不互相覆盖。同时顺手完成 spec 阶段 3 的 `constitution_acknowledged` 字段（前置 G1 修复已经把 output-constitution.md + 4 个 subagent 读宪法落了，本 plan 把它进 L1 schema 收尾）。

**Architecture:** 单向改动管道：
1. **task tool 加 `handoff_suffix` 必填参数** → 影响 lead 派遣 API
2. **L1 schema 定为 `[handoff_summary]` 块 + 严格 JSON**（不写 Python 中间件强校验，先规范约定）
3. **4 个 ethoinsight subagent prompt 改造**：写 L2 时构造路径 `${RUN_DIR}/handoff_<type>__<suffix>.json`（run-scoped 路径绝缘 = spec 阶段 2 的预留，本 plan 暂时用 `${WORKSPACE_DIR}/handoff_<type>__<suffix>.json` 实现，留 TODO 标记等阶段 2 接入 RUN_DIR）；最终消息按 schema 返回 L1
4. **code-executor 加 catalog 字段投影**：写 L1 时从 catalog YAML 取 display_name_zh / direction_for_anxiety 等填进 `key_results[i]`
5. **lead prompt 改造**：每次 `task()` 必填 `handoff_suffix`；派下游 subagent 时把上一个 subagent 的 L1 inline 进 prompt；决策只看 L1
6. **HandoffIsolationProvider 适配**：现有按固定文件名授权（`handoff_code_executor.json`）→ 改为按"type__suffix" 匹配模式授权
7. **L1 加 `constitution_acknowledged` 字段**：4 个 subagent 写 L1 时强制 `true`（G1 修复已经让它们必读宪法）

不动 sandbox、不动 archiving middleware、不引入新的中间件（HandoffPendingActionsProvider 暂不实施，留到阶段 2 之后单独 plan）。

**Tech Stack:** Python 3.12+ (agent backend) / Python 3.10+ (ethoinsight) / pytest / pytest-asyncio / ruff (line length 240) / deerflow `GuardrailProvider` 协议（已存在）

**Context links（实现时必读）：**
- spec 主文档：`docs/superpowers/specs/2026-05-14-handoff-protocol-and-runtime-isolation-spec.md` §2 (L1 schema 完整定义) + §4 (端到端工作流) + §8 阶段 1 + §9 验收
- 前置 G1 修复：commit `007fb390`（已落地 output-constitution.md + 4 subagent read constitution）
- 前置 G5 修复：commit `f2a60122` + `acffce87`（CLI env-var fallback，路径系统经验）
- task tool 入口：`packages/agent/backend/packages/harness/deerflow/tools/builtins/task_tool.py`（315 行）
- HandoffIsolationProvider：`packages/agent/backend/packages/harness/deerflow/guardrails/handoff_isolation_provider.py`（88 行）
- 4 个 ethoinsight subagent 配置：`packages/agent/backend/packages/harness/deerflow/subagents/builtins/{code_executor,data_analyst,report_writer,knowledge_assistant}.py`（共 599 行）
- lead prompt：`packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`（line 416-440 已有 `{{handoff://}}` 占位符规约段；line 442+ 已有过程透明原则段）
- output-constitution：`packages/agent/skills/custom/ethoinsight/references/output-constitution.md`
- catalog YAML：`packages/ethoinsight/ethoinsight/catalog/<paradigm>.yaml`（每个 metric 含 display_name_zh / unit_zh / one_liner / direction_for_anxiety 等字段）
- 项目约束：`CLAUDE.md`（TDD 强制 / 中文 commit）+ `backend/CLAUDE.md`（boundary test：harness 不能 import app）

**Scope（明确不做）：**
- **不实施** spec §5.5.2 `HandoffPendingActionsProvider`——单独 plan 在阶段 2 之后做
- **不实施** spec §3 完整的 `runs/<run_id>/` 路径绝缘——本 plan L2 路径暂用 `${WORKSPACE_DIR}/handoff_<type>__<suffix>.json`，加 TODO 标记，等阶段 2 plan 接入 RUN_DIR env var
- **不实施** spec §6 catalog modifier（指标外分析）——独立 spec
- **不动** sandbox / ArchivingSummarizationMiddleware
- **不重写** output-constitution.md 内容（G1 修复 commit `007fb390` 已经写好；本 plan 只在 L1 schema 加 acknowledged 字段）
- **不删除** 旧的 `[gate_signals]` 文本块兼容路径——本 plan 让 subagent 同时输出 L1 JSON + 兼容旧文本块，让 lead 优先消费 L1；旧解析路径作为 fallback 不立即清理
- **不动** `general-purpose` / `bash` subagent（它们不在 ethoinsight 4 个之列）

**前置假设（执行前用 `git log -1` 验证）：**
- 当前在 `dev` 分支
- FIRST-TOKEN 回退 plan + G4 方案 C plan 已经合入或不阻塞（本 plan 与它们正交）
- backend / ethoinsight 测试全绿
- `output-constitution.md` 文件存在（commit `007fb390` 产物）—— 用 `ls packages/agent/skills/custom/ethoinsight/references/output-constitution.md` 验证

---

## File Structure

**修改 10 个文件 + 新建 1 个文件**：

| 文件 | 改动 |
|---|---|
| `packages/agent/backend/packages/harness/deerflow/tools/builtins/task_tool.py` | 加 `handoff_suffix: str` 必填参数 + HANDOFF_FILE_REGISTRY 改用 suffix 模板 + `_resolve_handoff_placeholders` 接受 suffix 参数 |
| `packages/agent/backend/packages/harness/deerflow/guardrails/handoff_isolation_provider.py` | 改 `_is_own_handoff` 匹配 pattern 含 suffix |
| `packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py` | prompt 加：1) 接收 `handoff_suffix` 2) 写 L2 路径 `handoff_code_executor__<suffix>.json` 3) 返回 L1 schema（含 catalog 字段投影）4) constitution_acknowledged 字段 |
| `packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py` | 同上 1+2+4，但 L1 schema 略不同（无 catalog 投影，只有 解读 summary） |
| `packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py` | 同 data_analyst |
| `packages/agent/backend/packages/harness/deerflow/subagents/builtins/knowledge_assistant.py` | 同 data_analyst（最简，只 acknowledged + summary） |
| `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` | 1) line 309-340 角色边界段加"每次 task() 必填 handoff_suffix"约束 2) line 416-440 `{{handoff://}}` 段更新为 `{{handoff://<type>__<suffix>}}` 3) line 1080-1100 添加"派下游 subagent 时 inline 上一 L1" 4) 在 Step 1.5 / Step 2 派遣指南里加 suffix 命名建议 |
| `packages/agent/backend/tests/test_task_tool_handoff_suffix.py` | **新建**：覆盖 task_tool handoff_suffix 必填参数 + 路径解析 + HandoffIsolationProvider 适配 |
| `packages/agent/backend/tests/test_handoff_isolation_provider.py` | **更新**：加测试用 suffix 化的 handoff 文件名 |
| `packages/ethoinsight/ethoinsight/catalog/loader.py` 或新建 `catalog/projection.py` | 新建：`project_metric_to_l1(metric_id, value, catalog) → dict`——把 catalog 字段投影成 L1 key_results entry 格式 |
| `packages/ethoinsight/tests/test_catalog_projection.py` | **新建**：覆盖 catalog 字段投影逻辑 |

**0 个删除文件**

---

## Plan 分 7 个 Task，按依赖顺序：

1. **Task 1：catalog 字段投影函数**（ethoinsight 库，独立可单测）
2. **Task 2：task_tool 加 handoff_suffix 必填参数**（API 改动 + 测试）
3. **Task 3：HandoffIsolationProvider 适配 suffix 路径**（guardrail 改动 + 测试更新）
4. **Task 4：code-executor subagent 改造**（写 L1 + 用 suffix L2 路径 + 引用 catalog 投影）
5. **Task 5：data-analyst / report-writer / knowledge-assistant 改造**（写 L1 + 用 suffix L2 路径）
6. **Task 6：lead prompt 改造**（每次 task() 必填 suffix + 派下游 inline L1 + 决策只看 L1）
7. **Task 7：dogfood 端到端验证 + spec §9 验收 + push**

---

### Task 1: catalog 字段投影函数（独立可测，最先做）

**Files:**
- Create: `packages/ethoinsight/ethoinsight/catalog/projection.py`
- Create: `packages/ethoinsight/tests/test_catalog_projection.py`

**为什么先做**：catalog 字段投影是 L1 schema 的核心数据契约（spec §2.4.1）；其他 task 引用此函数。先单测落地。

- [ ] **Step 1: 写失败测试**

完整内容 `packages/ethoinsight/tests/test_catalog_projection.py`：

```python
"""验证 catalog 字段到 L1 key_results entry 的投影逻辑。

spec §2.4.1: code-executor 写 L1 时从 catalog YAML 取下列字段放进 key_results[i]：
  - display_name_zh / unit_zh / one_liner / direction_for_anxiety
  - statistical_default / is_primary
让下游 subagent 不必各自再读 YAML。
"""

from __future__ import annotations

import pytest

from ethoinsight.catalog.projection import project_metric_to_l1
from ethoinsight.catalog.loader import load_catalog


def test_project_epm_open_arm_time_ratio_has_all_required_fields():
    cat = load_catalog("epm")
    entry = project_metric_to_l1(
        metric_id="open_arm_time_ratio",
        value=0.0799,
        catalog=cat,
    )
    # spec §2.4.1 要求字段全部存在
    required = {"id", "value", "unit", "display_name_zh", "unit_zh",
                "one_liner", "direction_for_anxiety", "statistical_default",
                "is_primary"}
    missing = required - set(entry.keys())
    assert not missing, f"missing fields: {missing}"
    # 类型 sanity check
    assert entry["id"] == "open_arm_time_ratio"
    assert entry["value"] == 0.0799
    assert isinstance(entry["display_name_zh"], str) and entry["display_name_zh"]
    assert isinstance(entry["is_primary"], bool)


def test_project_unknown_metric_raises():
    cat = load_catalog("epm")
    with pytest.raises(KeyError, match="not_a_real_metric"):
        project_metric_to_l1(
            metric_id="not_a_real_metric",
            value=0.5,
            catalog=cat,
        )


def test_project_passes_through_numeric_types():
    cat = load_catalog("epm")
    # int → 保持 int
    entry_int = project_metric_to_l1(
        metric_id="total_entry_count", value=21, catalog=cat
    )
    assert entry_int["value"] == 21 and isinstance(entry_int["value"], int)
    # float → 保持 float
    entry_float = project_metric_to_l1(
        metric_id="open_arm_time_ratio", value=0.0799, catalog=cat
    )
    assert entry_float["value"] == 0.0799 and isinstance(entry_float["value"], float)


def test_project_handles_missing_optional_fields_with_none():
    """catalog YAML 里 direction_for_anxiety 可能未定义（某些非焦虑范式）。
    投影函数应 graceful 处理：未定义 → None，不抛错。"""
    # shoaling 范式可能没有 direction_for_anxiety
    cat = load_catalog("shoaling")
    metrics = [m for m in cat.default_metrics if m.id]
    if not metrics:
        pytest.skip("shoaling catalog has no metrics to test")
    first_metric = metrics[0]
    entry = project_metric_to_l1(
        metric_id=first_metric.id, value=1.0, catalog=cat
    )
    # 必有字段
    assert "id" in entry and "value" in entry
    # 可选字段：可以是 None 或字符串
    assert entry.get("direction_for_anxiety") is None or isinstance(
        entry["direction_for_anxiety"], str
    )
```

Run:
```bash
cd /home/wangqiuyang/noldus-insight
pytest packages/ethoinsight/tests/test_catalog_projection.py -v
```

Expected: 全 FAIL（projection.py 不存在）。

- [ ] **Step 2: 实现 `projection.py`**

完整内容 `packages/ethoinsight/ethoinsight/catalog/projection.py`：

```python
"""Catalog 字段投影：把 metric 的 catalog 元数据投影到 L1 key_results entry。

spec §2.4.1: code-executor 写 L1 时调本函数，把 catalog YAML 的 display_name_zh /
direction_for_anxiety 等字段投影进 L1，让下游 subagent 不必再读 YAML。

设计原则：
  - 单一职责：只做 catalog → L1 的字段映射，不做计算
  - 容错：catalog 中可选字段缺失时返回 None，不抛错
  - 未知 metric_id 抛 KeyError，便于上游 fail-fast
"""

from __future__ import annotations

from typing import Any

from ethoinsight.catalog.schema import Catalog


def project_metric_to_l1(
    metric_id: str,
    value: int | float,
    catalog: Catalog,
) -> dict[str, Any]:
    """Project a metric's catalog metadata + value into L1 key_results entry format.

    Returns dict with required keys:
        id, value, unit, display_name_zh, unit_zh, one_liner,
        direction_for_anxiety, statistical_default, is_primary

    Optional catalog fields that are missing in YAML → returned as None.

    Raises:
        KeyError: if metric_id not found in catalog.default_metrics or
                  catalog.metrics（按 catalog schema 实际字段调整查找路径）.
    """
    # 在 catalog.default_metrics 中查找
    target = None
    for m in catalog.default_metrics:
        if m.id == metric_id:
            target = m
            is_primary = True
            break
    # fallback：在 catalog.optional_metrics（如果 schema 有）中找
    if target is None and hasattr(catalog, "optional_metrics"):
        for m in catalog.optional_metrics:
            if m.id == metric_id:
                target = m
                is_primary = False
                break
    if target is None:
        raise KeyError(
            f"metric_id '{metric_id}' not found in catalog "
            f"(paradigm={catalog.paradigm if hasattr(catalog, 'paradigm') else '?'})"
        )

    return {
        "id": metric_id,
        "value": value,
        "unit": getattr(target, "unit", None),
        "display_name_zh": getattr(target, "display_name_zh", None),
        "unit_zh": getattr(target, "unit_zh", None),
        "one_liner": getattr(target, "one_liner", None),
        "direction_for_anxiety": getattr(target, "direction_for_anxiety", None),
        "statistical_default": getattr(target, "statistical_default", None),
        "is_primary": is_primary,
    }
```

**注意**：实际 `MetricEntry` schema 是否含 `unit_zh` / `one_liner` / `direction_for_anxiety` 字段，需先 `grep -n "unit_zh\|one_liner\|direction_for_anxiety" packages/ethoinsight/ethoinsight/catalog/schema.py` 验证。如果 schema 里没有这些字段，需要先在 schema.py 加（dataclass 加 Optional 字段），然后给 EPM / OFT catalog YAML 加对应字段。如果加 schema 字段超出本 task 范围，**简化方案**：直接在 projection.py 里用 `target.__dict__.get("display_name_zh")` 兼容缺字段。

- [ ] **Step 3: 跑测试**

```bash
cd /home/wangqiuyang/noldus-insight
pytest packages/ethoinsight/tests/test_catalog_projection.py -v
```

Expected: 全 PASS。如果 catalog schema 缺字段导致 test_project_epm_*_has_all_required_fields 失败，需先扩 schema + EPM YAML 才能通过（这是本 plan 的"硬依赖任务"，必须做）。

- [ ] **Step 4: ethoinsight 全量测试**

```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
pytest tests/
```

Expected: 全绿（除已知预存 parse 失败）。

- [ ] **Step 5: Commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/ethoinsight/ethoinsight/catalog/projection.py \
        packages/ethoinsight/tests/test_catalog_projection.py
# 如果 schema.py 也改了，一起 add
git commit -m "$(cat <<'EOF'
feat(catalog): 加 metric → L1 key_results 投影函数（spec §2.4.1）

把 catalog YAML 的 display_name_zh / direction_for_anxiety / is_primary
等字段投影到 L1 key_results entry 字典。让下游 subagent（data-analyst /
report-writer）通过 L1 拿到 catalog 元数据，无需各自再读 YAML——
single source of truth 真正闭环。
EOF
)"
```

---

### Task 2: task_tool 加 handoff_suffix 必填参数

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/tools/builtins/task_tool.py`
- Create: `packages/agent/backend/tests/test_task_tool_handoff_suffix.py`

**改动思路**：
- task_tool 签名加 `handoff_suffix: str` 必填
- `HANDOFF_FILE_REGISTRY` 改为模板：name → filename pattern（含 `{suffix}` 占位符）
- `_resolve_handoff_placeholders` 接受 lead 给的占位符 `{{handoff://<type>__<suffix>}}` 形式，解析为 `handoff_<type>__<suffix>.json`
- 子函数 `_compose_handoff_path(subagent_type, suffix)` 集中文件路径构造

- [ ] **Step 1: 写失败测试**

完整内容 `packages/agent/backend/tests/test_task_tool_handoff_suffix.py`：

```python
"""验证 task_tool handoff_suffix 参数 + 占位符解析。"""

from __future__ import annotations

import pytest


def test_handoff_file_path_includes_suffix():
    from deerflow.tools.builtins.task_tool import _compose_handoff_path

    path = _compose_handoff_path("code-executor", "epm_basic")
    assert path == "/mnt/user-data/workspace/handoff_code_executor__epm_basic.json"


def test_handoff_file_path_with_different_suffixes_for_same_type():
    from deerflow.tools.builtins.task_tool import _compose_handoff_path

    p1 = _compose_handoff_path("code-executor", "epm_basic")
    p2 = _compose_handoff_path("code-executor", "time_segment_5bins")
    assert p1 != p2


def test_resolve_handoff_placeholder_with_suffix():
    from deerflow.tools.builtins.task_tool import _resolve_handoff_placeholders

    prompt = "请基于 {{handoff://code_executor__epm_basic}} 分析数据"
    replaced, authorized = _resolve_handoff_placeholders(prompt)
    assert "/mnt/user-data/workspace/handoff_code_executor__epm_basic.json" in replaced
    assert "{{handoff://" not in replaced
    assert len(authorized) == 1


def test_resolve_handoff_placeholder_unknown_subagent_raises():
    from deerflow.tools.builtins.task_tool import _resolve_handoff_placeholders

    prompt = "用 {{handoff://typo_name__suffix}} 分析"
    with pytest.raises(ValueError, match="Unknown handoff subagent"):
        _resolve_handoff_placeholders(prompt)


def test_resolve_handoff_placeholder_missing_suffix_falls_back_to_default():
    """向后兼容：lead 偶尔写 {{handoff://code_executor}}（无 suffix）时
    fallback 到 'default' suffix。"""
    from deerflow.tools.builtins.task_tool import _resolve_handoff_placeholders

    prompt = "用 {{handoff://code_executor}} 分析（无 suffix 形式）"
    replaced, authorized = _resolve_handoff_placeholders(prompt)
    # 兜底用 default
    assert "/mnt/user-data/workspace/handoff_code_executor__default.json" in replaced


@pytest.mark.asyncio
async def test_task_tool_requires_handoff_suffix():
    """task_tool 签名 introspection：handoff_suffix 是必填参数。"""
    from deerflow.tools.builtins.task_tool import task_tool

    # 查 langchain @tool 装饰器导出的 schema
    schema = task_tool.args_schema
    if hasattr(schema, "model_fields"):  # Pydantic v2
        fields = schema.model_fields
        assert "handoff_suffix" in fields
        # 必填 = 无 default
        suffix_field = fields["handoff_suffix"]
        # Pydantic v2: 必填字段的 default 是 PydanticUndefined
        from pydantic_core import PydanticUndefined
        assert suffix_field.default is PydanticUndefined, (
            "handoff_suffix should be required (no default)"
        )
    else:  # fallback v1
        fields = schema.__fields__
        assert "handoff_suffix" in fields
        assert fields["handoff_suffix"].required is True
```

Run:
```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
PYTHONPATH=. uv run pytest tests/test_task_tool_handoff_suffix.py -v
```

Expected: 全 FAIL（`_compose_handoff_path` 不存在；task_tool 签名无 handoff_suffix）。

- [ ] **Step 2: 改 task_tool.py**

具体 edits（按 line 38-79 现有代码）：

**Edit A**：把 `HANDOFF_FILE_REGISTRY` 改为模式 + 加 `_compose_handoff_path` 函数

old_string（line 41-46）：
```python
HANDOFF_FILE_REGISTRY: dict[str, str] = {
    "code_executor": "handoff_code_executor.json",
    "data_analyst": "handoff_data_analyst.json",
    "report_writer": "handoff_report_writer.json",
    "planning": "handoff_planning.json",
}
```

new_string：
```python
# Handoff-file naming: handoff_<subagent_type_underscored>__<suffix>.json
# Lead 在 task() 时必填 handoff_suffix；read 上游 handoff 时用
# {{handoff://<type>__<suffix>}} 占位符。
#
# 已知 subagent type 白名单（防 lead 拼错）。新增 subagent 时在这里登记。
KNOWN_HANDOFF_SUBAGENT_NAMES: set[str] = {
    "code_executor",
    "data_analyst",
    "report_writer",
    "knowledge_assistant",
    "planning",
}


def _compose_handoff_path(subagent_type: str, suffix: str) -> str:
    """Compose the L2 handoff file path for a given subagent + suffix.

    Naming: /mnt/user-data/workspace/handoff_<type_underscored>__<suffix>.json

    Example: code-executor + "epm_basic" → handoff_code_executor__epm_basic.json
    """
    type_normalized = subagent_type.replace("-", "_")
    return f"/mnt/user-data/workspace/handoff_{type_normalized}__{suffix}.json"
```

**Edit B**：改 `_resolve_handoff_placeholders` 支持 `<type>__<suffix>` 形式

old_string（line 48-78）：
```python
_HANDOFF_PLACEHOLDER_RE = re.compile(r"\{\{handoff://([^}]+)\}\}")


def _resolve_handoff_placeholders(prompt: str) -> tuple[str, set[str]]:
    """Replace ``{{handoff://<subagent_name>}}`` with the full workspace path.

    Returns:
        (replaced_prompt, authorized_absolute_paths) — the set is what
        HandoffIsolationProvider uses as its allowlist for the subagent.

    Raises:
        ValueError: if any placeholder references an unknown subagent name
            (fail-fast on typo so lead immediately learns about the error
            rather than silently dispatching a subagent with a broken prompt).
    """
    authorized: set[str] = set()

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1).strip()
        if name not in HANDOFF_FILE_REGISTRY:
            raise ValueError(
                f"Unknown handoff subagent '{name}' in placeholder. "
                f"Known: {sorted(HANDOFF_FILE_REGISTRY)}"
            )
        filename = HANDOFF_FILE_REGISTRY[name]
        full_path = f"/mnt/user-data/workspace/{filename}"
        authorized.add(full_path)
        return full_path

    replaced = _HANDOFF_PLACEHOLDER_RE.sub(_replace, prompt)
    return replaced, authorized
```

new_string：
```python
_HANDOFF_PLACEHOLDER_RE = re.compile(r"\{\{handoff://([^}]+)\}\}")


def _resolve_handoff_placeholders(prompt: str) -> tuple[str, set[str]]:
    """Replace ``{{handoff://<type>__<suffix>}}`` (or legacy ``<type>``) with full workspace path.

    Supported forms:
      - {{handoff://code_executor__epm_basic}}  →  handoff_code_executor__epm_basic.json
      - {{handoff://code_executor}}             →  handoff_code_executor__default.json (legacy fallback)

    Returns:
        (replaced_prompt, authorized_absolute_paths) — the set is what
        HandoffIsolationProvider uses as its allowlist for the subagent.

    Raises:
        ValueError: if any placeholder references an unknown subagent name
            (fail-fast on typo so lead immediately learns about the error).
    """
    authorized: set[str] = set()

    def _replace(match: re.Match[str]) -> str:
        spec = match.group(1).strip()
        # 拆 type / suffix
        if "__" in spec:
            type_part, suffix = spec.split("__", 1)
        else:
            # legacy fallback：无 suffix → "default"
            type_part = spec
            suffix = "default"
        if type_part not in KNOWN_HANDOFF_SUBAGENT_NAMES:
            raise ValueError(
                f"Unknown handoff subagent '{type_part}' in placeholder. "
                f"Known: {sorted(KNOWN_HANDOFF_SUBAGENT_NAMES)}"
            )
        full_path = _compose_handoff_path(type_part, suffix)
        authorized.add(full_path)
        return full_path

    replaced = _HANDOFF_PLACEHOLDER_RE.sub(_replace, prompt)
    return replaced, authorized
```

**Edit C**：改 task_tool 签名加 `handoff_suffix: str` 必填

old_string（line 81-120）：
```python
@tool("task", parse_docstring=True)
async def task_tool(
    runtime: ToolRuntime[ContextT, ThreadState],
    description: str,
    prompt: str,
    subagent_type: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    max_turns: int | None = None,
) -> str:
    """Delegate a task to a specialized subagent that runs in its own context.

    ...

    Args:
        description: A short (3-5 word) description of the task for logging/display. ALWAYS PROVIDE THIS PARAMETER FIRST.
        prompt: The task description for the subagent. Be specific and clear about what needs to be done. ALWAYS PROVIDE THIS PARAMETER SECOND.
        subagent_type: The type of subagent to use. ALWAYS PROVIDE THIS PARAMETER THIRD.
        max_turns: Optional maximum number of agent turns. Defaults to subagent's configured max.
    """
```

new_string（注意保留中间 description 不变，只示意签名 + Args 段）：
```python
@tool("task", parse_docstring=True)
async def task_tool(
    runtime: ToolRuntime[ContextT, ThreadState],
    description: str,
    prompt: str,
    subagent_type: str,
    handoff_suffix: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
    max_turns: int | None = None,
) -> str:
    """Delegate a task to a specialized subagent that runs in its own context.

    ... (中间描述不变) ...

    Args:
        description: A short (3-5 word) description of the task for logging/display. ALWAYS PROVIDE THIS PARAMETER FIRST.
        prompt: The task description for the subagent. Be specific and clear about what needs to be done. ALWAYS PROVIDE THIS PARAMETER SECOND.
        subagent_type: The type of subagent to use. ALWAYS PROVIDE THIS PARAMETER THIRD.
        handoff_suffix: Required namespace suffix for this dispatch's L2 handoff file. Subagent writes to /mnt/user-data/workspace/handoff_<type>__<suffix>.json. Lead must choose a stable, human-readable suffix like "epm_basic" / "time_segment_5bins" / "single_descriptive" so multiple dispatches of the same subagent don't overwrite each other. ALWAYS PROVIDE THIS PARAMETER FOURTH.
        max_turns: Optional maximum number of agent turns. Defaults to subagent's configured max.
    """
```

**实际 edit 操作**：拆成两步——

1. 先 Edit `def task_tool(...)` 签名行追加 `handoff_suffix: str,` 在 `subagent_type: str,` 后、`tool_call_id` 前
2. 再 Edit docstring 的 Args 段插入 handoff_suffix 描述

**Edit D**：在 task_tool 主体中把 handoff_suffix 传给 subagent prompt（构造一段说明）

在 `prompt = _resolve_placeholders(prompt)`（line 173 附近）之后、`# Resolve {{handoff://...}}` 之前，加一段：

```python
    # Inject handoff_suffix context so subagent knows where to write its L2 handoff.
    # L2 文件路径模板：/mnt/user-data/workspace/handoff_<type>__<suffix>.json
    # subagent prompt 中用 ${HANDOFF_SUFFIX} 替换。
    subagent_l2_path = _compose_handoff_path(subagent_type, handoff_suffix)
    prompt = prompt.replace("${HANDOFF_SUFFIX}", handoff_suffix)
    prompt = prompt.replace("${L2_HANDOFF_PATH}", subagent_l2_path)
```

这样 subagent prompt 模板可以这么写："写完毕请 write_file `${L2_HANDOFF_PATH}`"——task_tool 在 dispatch 前完成模板替换。

- [ ] **Step 3: 跑新测试 + 既有 task_tool 相关测试**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
PYTHONPATH=. uv run pytest tests/test_task_tool_handoff_suffix.py -v
PYTHONPATH=. uv run pytest tests/test_subagent_handoff_isolation_integration.py -v
```

Expected: 新测试全 PASS；既有集成测试可能因 task_tool 签名变化失败，**这是预期的**——在 Task 3 / Task 5 / Task 6 中会一起更新 lead prompt 和 subagent 的 prompt 模板让它们传 handoff_suffix。本 Task 不阻塞失败的旧测试。

如果旧测试因为缺 `handoff_suffix` 失败，先记下，**不要**改测试让它"刚好兼容"——等 Task 6 把 lead prompt 改完才更新这些测试。

- [ ] **Step 4: backend lint**

```bash
make lint
```

Expected: 无错误。

- [ ] **Step 5: Commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/backend/packages/harness/deerflow/tools/builtins/task_tool.py \
        packages/agent/backend/tests/test_task_tool_handoff_suffix.py
git commit -m "$(cat <<'EOF'
feat(task_tool): 加 handoff_suffix 必填参数 + 占位符 type__suffix 解析（spec §2 + §3）

task_tool 签名 + docstring 加 handoff_suffix。lead 派遣 subagent 时必须命名
一个 suffix（如 "epm_basic"），让多次调用同一 subagent 的产物物理隔离：
  handoff_code_executor__epm_basic.json
  handoff_code_executor__time_segment_5bins.json

_resolve_handoff_placeholders 解析 {{handoff://<type>__<suffix>}} 形式；
无 suffix 的旧形式 fallback 到 "default" suffix 不立即破坏 API。

task_tool 主体把 ${HANDOFF_SUFFIX} / ${L2_HANDOFF_PATH} 注入 subagent prompt，
subagent prompt 模板可直接引用，避免 lead 在 prompt 里硬编码物理路径。

旧的 test_subagent_handoff_isolation_integration 暂时失败——等 Task 3-6 更新
HandoffIsolationProvider 和 lead/subagent prompt 后绿。
EOF
)"
```

---

### Task 3: HandoffIsolationProvider 适配 suffix 路径

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/guardrails/handoff_isolation_provider.py`
- Modify: `packages/agent/backend/tests/test_handoff_isolation_provider.py`

**改动思路**：现有 `_is_own_handoff` 用 `f"handoff_{normalized}.json"` 简单匹配。新方案：文件名包含 `__<suffix>` 时仍是该 subagent 自己的——只要 `handoff_<normalized>__` 前缀匹配即可。

- [ ] **Step 1: 改 `_is_own_handoff` 加 suffix 支持**

用 Edit 工具修改 `handoff_isolation_provider.py:38-49`。

old_string：
```python
    def _is_own_handoff(self, file_path: str) -> bool:
        """Allow subagent to read its own handoff file (it just wrote it).

        e.g. data-analyst writes handoff_data_analyst.json, then may re-read
        for self-validation. This is not "peeking at peer".
        """
        if not self.self_outbox_subagent_name:
            return False
        # Subagent names use hyphens ('code-executor'); filenames use
        # underscores (handoff_code_executor.json). Normalize for comparison.
        normalized = self.self_outbox_subagent_name.replace("-", "_")
        return f"handoff_{normalized}.json" in file_path
```

new_string：
```python
    def _is_own_handoff(self, file_path: str) -> bool:
        """Allow subagent to read its own handoff file (it just wrote it).

        Match BOTH legacy and suffix-namespaced forms:
          - handoff_data_analyst.json (legacy, no suffix)
          - handoff_data_analyst__<suffix>.json (spec §2 + §3 suffix-namespaced)

        Match logic: filename contains "handoff_<normalized>" followed by
        either ".json" (legacy) or "__" (suffixed). Substring match on
        "handoff_<normalized>__" or "handoff_<normalized>.json" suffix-anchored.
        """
        if not self.self_outbox_subagent_name:
            return False
        normalized = self.self_outbox_subagent_name.replace("-", "_")
        # 提取文件名（去掉路径前缀）
        import os
        basename = os.path.basename(file_path)
        # 两种合法形式
        if basename == f"handoff_{normalized}.json":
            return True
        if basename.startswith(f"handoff_{normalized}__") and basename.endswith(".json"):
            return True
        return False
```

- [ ] **Step 2: 更新测试**

打开 `tests/test_handoff_isolation_provider.py`，找已有 "self_outbox" 相关测试加新 case。

用 Edit 工具在该测试类末尾追加新方法（具体语法看现有测试结构）：

```python
def test_self_outbox_matches_suffixed_filename(self):
    """spec §2 suffix-namespaced handoff files must be recognized as own outbox."""
    from deerflow.guardrails.handoff_isolation_provider import HandoffIsolationProvider
    from deerflow.guardrails.provider import GuardrailRequest

    provider = HandoffIsolationProvider(
        authorized_paths=set(),
        self_outbox_subagent_name="data-analyst",
    )
    req = GuardrailRequest(
        tool_name="read_file",
        tool_input={"file_path": "/mnt/user-data/workspace/handoff_data_analyst__epm_basic.json"},
        agent_id="subagent:data-analyst",
    )
    decision = provider.evaluate(req)
    assert decision.allow, "suffixed own-outbox handoff must be readable"


def test_self_outbox_does_not_match_other_subagent_suffixed(self):
    """data-analyst 不能因为 suffix 误判读 code-executor 的 handoff。"""
    from deerflow.guardrails.handoff_isolation_provider import HandoffIsolationProvider
    from deerflow.guardrails.provider import GuardrailRequest

    provider = HandoffIsolationProvider(
        authorized_paths=set(),
        self_outbox_subagent_name="data-analyst",
    )
    req = GuardrailRequest(
        tool_name="read_file",
        tool_input={"file_path": "/mnt/user-data/workspace/handoff_code_executor__epm_basic.json"},
        agent_id="subagent:data-analyst",
    )
    decision = provider.evaluate(req)
    assert not decision.allow, "must NOT allow reading peer's handoff without explicit authorization"
```

- [ ] **Step 3: 跑相关测试**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
PYTHONPATH=. uv run pytest tests/test_handoff_isolation_provider.py -v
```

Expected: 全 PASS（含新加 2 个）。

- [ ] **Step 4: Commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/backend/packages/harness/deerflow/guardrails/handoff_isolation_provider.py \
        packages/agent/backend/tests/test_handoff_isolation_provider.py
git commit -m "$(cat <<'EOF'
feat(guardrails): HandoffIsolationProvider 识别 suffix-namespaced 文件名

_is_own_handoff 同时识别：
- handoff_<type>.json（legacy）
- handoff_<type>__<suffix>.json（spec §2 + §3 suffix-namespaced）

防止 data-analyst 误读 code-executor 的 suffixed handoff——必须前缀完全匹配。
配套加 2 个测试 case。
EOF
)"
```

---

### Task 4: code-executor subagent 改造

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py`

**改动思路**：code-executor 的 system prompt（约 110 行 docstring）需要改 3 处：
1. 教它 L2 handoff 写到 `${L2_HANDOFF_PATH}`（task_tool 在 dispatch 时替换）
2. 最终消息按 L1 schema 返回（含 `[handoff_summary]` 块 + JSON）
3. JSON 里 `key_results` 调 catalog 投影函数生成

由于 system_prompt 是字符串字面量（Python），不能 import projection 函数运行——投影逻辑要写成 bash 调用（"调 python -c 'from ethoinsight.catalog.projection import project_metric_to_l1; ...'"）让 subagent 在执行时跑。或者更干净的：**新增一个 CLI** `python -m ethoinsight.catalog.summarize` 接受 metric_id + value list 返回 L1 key_results JSON 数组。

**Plan 选择**：新增 `python -m ethoinsight.catalog.summarize` CLI——避免 subagent 在 system_prompt 里硬编码长 Python 代码。

- [ ] **Step 1: 新建 `python -m ethoinsight.catalog.summarize` CLI**

完整内容 `packages/ethoinsight/ethoinsight/catalog/summarize.py`：

```python
"""CLI 入口：summarize — 把 metric_id + value list 投影成 L1 key_results JSON 数组。

用法：
  python -m ethoinsight.catalog.summarize \\
      --paradigm epm \\
      --metrics-json /mnt/user-data/workspace/metrics_values.json \\
      --output /mnt/user-data/workspace/key_results.json

metrics_values.json 格式：[{"id": "open_arm_time_ratio", "value": 0.0799}, ...]
key_results.json 输出格式：[{"id":..., "value":..., "display_name_zh":..., ...}, ...]

由 code-executor 在写 L1 handoff 前调用。集中投影逻辑、避免 subagent 在
system_prompt 里硬编码 Python。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ethoinsight.catalog.loader import CatalogError, load_catalog
from ethoinsight.catalog.projection import project_metric_to_l1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ethoinsight.catalog.summarize")
    parser.add_argument("--paradigm", required=True)
    parser.add_argument("--metrics-json", required=True,
                        help="JSON array of {id, value} dicts")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    try:
        cat = load_catalog(args.paradigm)
    except CatalogError as e:
        print(json.dumps({"code": "unknown_paradigm", "message": str(e)}),
              file=sys.stderr)
        return 1

    try:
        raw = json.loads(Path(args.metrics_json).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        print(json.dumps({"code": "schema_violation",
                          "message": f"Cannot read metrics-json: {e}"}),
              file=sys.stderr)
        return 1

    if not isinstance(raw, list):
        print(json.dumps({"code": "schema_violation",
                          "message": "metrics-json must be a JSON array"}),
              file=sys.stderr)
        return 1

    key_results = []
    for item in raw:
        try:
            entry = project_metric_to_l1(
                metric_id=item["id"],
                value=item["value"],
                catalog=cat,
            )
        except KeyError as e:
            print(json.dumps({"code": "unknown_metric",
                              "message": f"metric not in catalog: {e}"}),
                  file=sys.stderr)
            return 1
        key_results.append(entry)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(key_results, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"L1 key_results written to {args.output}: {len(key_results)} entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

加一个 smoke test `packages/ethoinsight/tests/test_catalog_summarize_cli.py`：

```python
"""smoke test for python -m ethoinsight.catalog.summarize CLI."""

from __future__ import annotations

import json
from pathlib import Path

from ethoinsight.catalog.summarize import main as summarize_main


def test_summarize_writes_key_results_with_catalog_fields(tmp_path: Path):
    metrics_json = tmp_path / "metrics.json"
    metrics_json.write_text(json.dumps([
        {"id": "open_arm_time_ratio", "value": 0.0799},
        {"id": "total_entry_count", "value": 21},
    ]), encoding="utf-8")
    output = tmp_path / "key_results.json"

    exit_code = summarize_main([
        "--paradigm", "epm",
        "--metrics-json", str(metrics_json),
        "--output", str(output),
    ])
    assert exit_code == 0

    key_results = json.loads(output.read_text(encoding="utf-8"))
    assert len(key_results) == 2
    for entry in key_results:
        assert "display_name_zh" in entry
        assert "direction_for_anxiety" in entry
        assert "is_primary" in entry
```

- [ ] **Step 2: 改 code-executor system prompt**

打开 `code_executor.py`，找到 system_prompt 里"聚合 / write_file handoff" 的段（约 line 34-65）。整段替换为：

要点（必须在 prompt 里说清楚）：
1. 接收 `handoff_suffix` 参数（task_tool 已通过 `${HANDOFF_SUFFIX}` / `${L2_HANDOFF_PATH}` 替换进 prompt）
2. 跑完 N 个 compute_* 后，先 `python -m ethoinsight.catalog.summarize` 生成 key_results.json
3. write_file `${L2_HANDOFF_PATH}`（含完整 metrics / stats / charts data，schema 见 ethoinsight-code skill）
4. 最终返回消息按 L1 schema 输出 `[handoff_summary]` JSON 块（spec §2.4 完整 schema）
5. L1 中 `constitution_acknowledged: true`（前置 G1 修复已让 subagent 必读宪法）
6. L1 中 `full_handoff_ref: "{{handoff://code_executor__${HANDOFF_SUFFIX}}}"`

提供完整 system_prompt 改写需要看现有 code_executor.py 全文，本 plan 不展开整段文本——执行 agent 按以下结构改：

```python
SYSTEM_PROMPT_TEMPLATE = """...（保留前面的角色定义不变）

## L2 handoff 文件命名

本次任务由 lead 显式分配的 handoff suffix：`${HANDOFF_SUFFIX}`
L2 handoff 完整路径：`${L2_HANDOFF_PATH}`

## 工作流（保留原步骤 1-4，改步骤 5-7）

5. 跑完所有 compute_*.py 后，构造 metrics_values.json：
   [{"id": "<metric_id>", "value": <number>}, ...]
   写到 /mnt/user-data/workspace/metrics_values.json

6. bash 调投影 CLI 生成 L1 key_results：
   python -m ethoinsight.catalog.summarize \\
       --paradigm <paradigm> \\
       --metrics-json /mnt/user-data/workspace/metrics_values.json \\
       --output /mnt/user-data/workspace/key_results.json

7. write_file `${L2_HANDOFF_PATH}`，包含完整：
   - paradigm, ev19_template, subject_id, group
   - metrics (full results), statistics, charts, data_quality_warnings, errors
   - constitution_acknowledged: true
   - 完整 schema 见 ethoinsight-code skill 的 templates/output-contract.md

## 最终返回消息（lead 看到的 L1）

最终消息**严格按下方格式**：

OK: code-executor completed

[handoff_summary]
{
  "schema_version": "1.0",
  "subagent": "code-executor",
  "handoff_suffix": "${HANDOFF_SUFFIX}",
  "run_id": "<run_id 从环境取或留空>",
  "status": "success" | "partial" | "failed",
  "key_results": <key_results.json 完整内容>,
  "gate_signals": {
    "data_quality": {
      "critical_count": <N>,
      "warning_count": <M>,
      "critical_items": [...]
    },
    "statistical_validity": "ok" | "warning" | "failed" | "skip",
    "errors_count": <N>,
    "needs_user_decision": <bool>
  },
  "summary_text": "<1-2 句中文给 lead 转述>",
  "full_handoff_ref": "{{handoff://code_executor__${HANDOFF_SUFFIX}}}",
  "constitution_acknowledged": true
}

不要在 [handoff_summary] 块前后加其他文字。
"""
```

执行 agent 实际改写时需要：
- 保留原 prompt 中"role 定义 / 错误处理 / catalog skill 路径"等段落不变
- 只改"步骤 5-7"和最终返回格式段
- 在 system_prompt 开头加 `${HANDOFF_SUFFIX} / ${L2_HANDOFF_PATH}` 说明

- [ ] **Step 3: 跑 ethoinsight 测试**

```bash
cd /home/wangqiuyang/noldus-insight/packages/ethoinsight
pytest tests/test_catalog_summarize_cli.py -v
pytest tests/
```

Expected: 新测试 PASS，全量绿（除预存）。

- [ ] **Step 4: backend lint**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
make lint
```

- [ ] **Step 5: Commit（拆 2 个 commit）**

```bash
# commit 1: summarize CLI
cd /home/wangqiuyang/noldus-insight
git add packages/ethoinsight/ethoinsight/catalog/summarize.py \
        packages/ethoinsight/tests/test_catalog_summarize_cli.py
git commit -m "feat(catalog): 加 summarize CLI — 把 metrics 投影成 L1 key_results JSON

集中 catalog 字段投影逻辑、避免 code-executor 在 system_prompt 硬编码 Python。
subagent 在工作流末尾调本 CLI 生成 key_results.json，写进 L1 handoff_summary。"

# commit 2: code-executor prompt 改造
git add packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py
git commit -m "feat(subagent): code-executor 写 L1+L2 handoff（spec §2.4）

system_prompt 改造：
- 接收 \${HANDOFF_SUFFIX} / \${L2_HANDOFF_PATH}（task_tool 注入）
- 工作流加 catalog.summarize 步骤投影 key_results
- L2 写完整 handoff JSON 到 \${L2_HANDOFF_PATH}
- 最终消息返回 [handoff_summary] L1 JSON 块（schema_version, key_results,
  gate_signals, full_handoff_ref, constitution_acknowledged）
- 不破坏旧的 [gate_signals] 兼容路径——lead 优先消费 L1"
```

---

### Task 5: data-analyst / report-writer / knowledge-assistant 改造

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py`
- Modify: `packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py`
- Modify: `packages/agent/backend/packages/harness/deerflow/subagents/builtins/knowledge_assistant.py`

**改动思路**：这三个 subagent 没有 catalog 字段投影需求，比 code-executor 简单。每个都需要：
1. 接收 `${HANDOFF_SUFFIX} / ${L2_HANDOFF_PATH}`
2. L2 写 `${L2_HANDOFF_PATH}` 含完整产出
3. 最终消息按 L1 schema 返回（简化版：无 key_results catalog 字段，但有 summary_text + 各自语义字段）
4. `constitution_acknowledged: true`

**data-analyst L1 关键字段**：除 schema 标配外，加 `insights` 数组（每条洞察）+ `method_warnings` 数组

**report-writer L1 关键字段**：除 schema 标配外，加 `report_summary_zh`（报告大纲）+ `output_artifacts: ["report.md"]`

**knowledge-assistant L1 关键字段**：除 schema 标配外，加 `answer_text` + `cited_sources` 数组

- [ ] **Step 1: 改 data_analyst.py**

按 Task 4 同款 pattern 改 data_analyst.py 的 system_prompt：加 `${HANDOFF_SUFFIX} / ${L2_HANDOFF_PATH}` 段、改 L2 路径、改最终消息格式。具体 schema：

```
OK: data-analyst completed

[handoff_summary]
{
  "schema_version": "1.0",
  "subagent": "data-analyst",
  "handoff_suffix": "${HANDOFF_SUFFIX}",
  "status": "success" | "partial" | "failed",
  "insights": [
    {"id": "insight_1", "metric_id": "<related>", "text": "...", "evidence": "..."},
    ...
  ],
  "method_warnings": ["...", ...],
  "summary_text": "<1-2 句中文给 lead 转述>",
  "full_handoff_ref": "{{handoff://data_analyst__${HANDOFF_SUFFIX}}}",
  "constitution_acknowledged": true
}
```

- [ ] **Step 2: 改 report_writer.py**

类似改造，最终消息：

```
OK: report-writer completed

[handoff_summary]
{
  "schema_version": "1.0",
  "subagent": "report-writer",
  "handoff_suffix": "${HANDOFF_SUFFIX}",
  "status": "success" | "partial" | "failed",
  "report_summary_zh": "<研究报告 3-5 句中文大纲>",
  "output_artifacts": ["/mnt/user-data/workspace/report__${HANDOFF_SUFFIX}.md"],
  "summary_text": "<1-2 句中文给 lead 转述>",
  "full_handoff_ref": "{{handoff://report_writer__${HANDOFF_SUFFIX}}}",
  "constitution_acknowledged": true
}
```

注意 report-writer 也用 suffix 命名 markdown 输出（`report__epm_basic.md`），让多次调用产物不互相覆盖。

- [ ] **Step 3: 改 knowledge_assistant.py**

最终消息：

```
OK: knowledge-assistant completed

[handoff_summary]
{
  "schema_version": "1.0",
  "subagent": "knowledge-assistant",
  "handoff_suffix": "${HANDOFF_SUFFIX}",
  "status": "success" | "partial" | "failed",
  "answer_text": "<完整回答中文>",
  "cited_sources": [
    {"type": "skill", "path": "..."},
    {"type": "mcp", "doc_id": "..."},
    ...
  ],
  "summary_text": "<1-2 句概括 answer>",
  "full_handoff_ref": "{{handoff://knowledge_assistant__${HANDOFF_SUFFIX}}}",
  "constitution_acknowledged": true
}
```

- [ ] **Step 4: lint + backend test**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
make lint
make test
```

Expected: lint 绿；test 中**部分原有 e2e 测试可能失败**，因为它们假设 subagent 最终消息是旧格式。**先记下哪些测试失败**，等 Task 6 lead prompt 改完后一并修复。

- [ ] **Step 5: Commit**

```bash
cd /home/wangqiuyang/noldus-insight
git add packages/agent/backend/packages/harness/deerflow/subagents/builtins/data_analyst.py \
        packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py \
        packages/agent/backend/packages/harness/deerflow/subagents/builtins/knowledge_assistant.py
git commit -m "$(cat <<'EOF'
feat(subagent): data-analyst / report-writer / knowledge-assistant 写 L1+L2（spec §2.4）

3 个 subagent 改造同款 pattern：
- 接收 \${HANDOFF_SUFFIX} / \${L2_HANDOFF_PATH}
- L2 写 \${L2_HANDOFF_PATH}
- 最终消息按 L1 schema 返回，各自语义字段：
  - data-analyst: insights[], method_warnings[]
  - report-writer: report_summary_zh, output_artifacts[]
  - knowledge-assistant: answer_text, cited_sources[]
- 全部含 constitution_acknowledged: true

report-writer 输出 markdown 也加 suffix 命名（report__<suffix>.md）。
EOF
)"
```

---

### Task 6: lead prompt 改造（最复杂、最后做）

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`
- Modify: 既有失败的 e2e / 集成测试（如 `test_subagent_handoff_isolation_integration.py` 等，按 Task 2/5 跑出的失败列表）

**改动思路**：lead prompt 改 4 处：
1. **角色边界硬约束段**（line 309-340）加："每次 task() 必须显式提供 handoff_suffix 参数，命名风格 lower_snake_case，反映本次调用的语义场景（如 `epm_basic` / `time_segment_5bins` / `single_descriptive`）"
2. **占位符规约段**（line 416-440）改示例为 `{{handoff://code_executor__epm_basic}}` 形式
3. **派下游 subagent 时 inline 上一 L1**——新加段："派 data-analyst 时，把上一个 code-executor 返回的 `[handoff_summary]` JSON inline 进 task prompt（用 ```json ... ``` 包裹），不要让 data-analyst 再读 L2 文件，除非 L1 信息不足"
4. **决策只看 L1**——新加段："看 subagent 返回的 [handoff_summary] JSON 块就能做决策（key_results / gate_signals / status / pending_actions（如有））。不要 read_file L2 完整 handoff，除非用户特别问详情"

- [ ] **Step 1: 改角色边界段加 handoff_suffix 必填约束**

用 Edit 工具，在 `### 角色边界硬约束（不可越权）` 段（line 309-340）追加一条约束。具体插入位置：在最后一条"3. 不要使用绝对参考术语"之后、`**正例**` 之前。

加入：

```markdown
4. **每次 `task()` 必填 `handoff_suffix`** — 必填参数。命名规则：`lower_snake_case`、反映本次调用的语义场景，例如：
   - 首次单只描述分析：`single_descriptive`
   - 标准 EPM 完整分析：`epm_basic`
   - 用户追加要的时间分段：`time_segment_5bins`
   - 重新跑同范式（参数变化）：`epm_v2`
   suffix 让多次调用同 subagent 的产物不互相覆盖。如果你不知道用什么 suffix，**先 `ask_clarification` 问用户场景类型**，不要随意用 `default`。
```

- [ ] **Step 2: 更新占位符规约段示例（line 416-440）**

用 Edit 工具修改占位符示例，从 `{{handoff://code_executor}}` 改为 `{{handoff://code_executor__epm_basic}}` 形式。具体 edits（按现有 prompt 内容）：

old_string：
```
| code-executor | `{{handoff://code_executor}}` | `/mnt/user-data/workspace/handoff_code_executor.json` |
| data-analyst | `{{handoff://data_analyst}}` | `/mnt/user-data/workspace/handoff_data_analyst.json` |
| report-writer | `{{handoff://report_writer}}` | `/mnt/user-data/workspace/handoff_report_writer.json` |
| planning | `{{handoff://planning}}` | `/mnt/user-data/workspace/handoff_planning.json` |
```

new_string：
```
| code-executor | `{{handoff://code_executor__<suffix>}}` | `/mnt/user-data/workspace/handoff_code_executor__<suffix>.json` |
| data-analyst | `{{handoff://data_analyst__<suffix>}}` | `/mnt/user-data/workspace/handoff_data_analyst__<suffix>.json` |
| report-writer | `{{handoff://report_writer__<suffix>}}` | `/mnt/user-data/workspace/handoff_report_writer__<suffix>.json` |
| knowledge-assistant | `{{handoff://knowledge_assistant__<suffix>}}` | `/mnt/user-data/workspace/handoff_knowledge_assistant__<suffix>.json` |

**关键**：`<suffix>` 必须填具体值，与你在 `task()` 调用时传的 `handoff_suffix` 完全一致。例如你派 code-executor 用 `handoff_suffix="epm_basic"`，下游派 data-analyst 时 prompt 里就写 `{{handoff://code_executor__epm_basic}}`。
```

- [ ] **Step 3: 新加"派下游 inline L1"段 + "决策只看 L1"段**

在占位符规约段之后、过程透明原则段之前，新加 markdown 段：

```markdown
### 派下游 subagent 时 inline 上一 L1（spec §4）

当你已经收到上游 subagent（如 code-executor）的 `[handoff_summary]` L1 JSON，再派下游 subagent（如 data-analyst）时：

**必须**把上游 L1 完整 JSON inline 进新派遣的 task prompt，用 ```json 围栏包裹。
**不要**只给 `{{handoff://code_executor__<suffix>}}` 占位符让 data-analyst 自己读 L2——L1 摘要已经足够下游做工作，避免不必要的 L2 文件读取。

**正例**：
```python
task(
    subagent_type="data-analyst",
    description="解读 EPM 指标",
    handoff_suffix="epm_basic",
    prompt='''请基于以下 code-executor L1 摘要做解读：

```json
{<将 code-executor 返回的 [handoff_summary] JSON 块完整粘贴在这里>}
```

如 L1 信息不足需要原始统计细节，用 read_file 读 {{handoff://code_executor__epm_basic}}。
'''
)
```

**仅当**下游明确需要 L2 中的完整数据（如 report-writer 要嵌入完整统计表 / 图表 base64）时，才用占位符让它自己读。

### 决策规则：lead 只消费 L1（spec §2.3）

收到 subagent 返回的 `[handoff_summary]` JSON 块之后，**你就有全部决策需要的信息**：
- `key_results` 含指标值 + catalog 元数据（display_name_zh / direction_for_anxiety）
- `gate_signals` 含 critical_count / warning_count / statistical_validity
- `status` 标本次任务成败
- `summary_text` 给你转述用户用的话术

**不要** `read_file` 完整的 L2 `handoff_*.json`，除非：
1. 用户特别问"原始 stats 数据" / "图表 base64" 等详情
2. L1 信息确实不够你回答用户的某个具体问题

这是 spec §2.3 "lead 不消费乐谱完整内容" 的硬约束。违反 = 浪费 context、增加漂移风险。
```

- [ ] **Step 4: 修复 Task 2 + Task 5 跑出失败的旧测试**

按 Task 2 Step 3 + Task 5 Step 4 记下的失败测试列表，逐个用 Edit 更新它们的 mock / assertion：
- 失败原因 A：测试调 task_tool 没传 handoff_suffix → 加 `handoff_suffix="test_suffix"` 参数
- 失败原因 B：测试 assert lead prompt 里有 `{{handoff://code_executor}}` → 改为 `__test_suffix`
- 失败原因 C：测试 assert subagent 最终消息含 `[gate_signals]` 旧格式 → 改为既兼容旧又兼容新（用 `or` 断言两种）

**不要"刚好让测试通过"——按改动语义更新测试**。

- [ ] **Step 5: 跑 backend 全量测试**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
make test
```

Expected: 全绿（除已知 6 个预存失败 auth/live/skill）。

- [ ] **Step 6: lint**

```bash
make lint
```

- [ ] **Step 7: Commit（拆 2 个 commit）**

```bash
cd /home/wangqiuyang/noldus-insight

# commit 1: lead prompt 改造
git add packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py
git commit -m "$(cat <<'EOF'
feat(lead): 双层 handoff 协议 lead 端接入（spec §2 + §4）

prompt 改 4 处：
- 角色边界硬约束加第 4 条：task() 必填 handoff_suffix（lower_snake_case 命名）
- 占位符规约段：{{handoff://<type>}} → {{handoff://<type>__<suffix>}}
- 新加段"派下游时 inline 上一 L1"——不依赖下游 read L2，L1 默认进 prompt
- 新加段"决策规则：lead 只消费 L1"——禁 read_file L2 除非用户特别问详情

衔接 task_tool 加 handoff_suffix 必填、4 个 subagent 写 L1+L2。
spec §2.3 抗漂移设计真正生效。
EOF
)"

# commit 2: 修复因 API 变化失败的旧测试
git add packages/agent/backend/tests/
git commit -m "$(cat <<'EOF'
test(spec-phase-1): 更新 e2e/集成测试适配 handoff_suffix 必填 API

按 Task 2/5/6 改动 update test mock 和 assertion：
- task_tool 调用加 handoff_suffix=
- {{handoff://}} 占位符示例加 __<suffix>
- subagent 最终消息断言同时兼容旧 [gate_signals] 和新 [handoff_summary] 格式

未改测试语义，只跟随 API 变化。
EOF
)"
```

---

### Task 7: dogfood 端到端验证 + spec §9 验收

**Files:**
- 不创建 / 不修改任何文件（纯测试）
- 末尾追加 `docs/handoffs/2026-05/2026-05-14-dogfood-followup-handoff.md`

**目的**：验证 spec §9 7 条验收标准。

- [ ] **Step 1: 起服务**

```bash
cd /home/wangqiuyang/noldus-insight/packages/agent
make stop
make dev  # 后台
until curl -sf --max-time 2 http://localhost:2026/ -o /dev/null 2>/dev/null; do sleep 2; done && echo "ready"
```

- [ ] **Step 2: Dogfood 验证（用 Playwright MCP 或手工）**

跑一次完整 EPM 单只分析（路径见之前 dogfood checklist）。期间观察 **lead 在 task() 调用时是否真的传了 handoff_suffix**——看 langgraph.log 的 SandboxAudit 或前端 SubtaskCard。

- [ ] **Step 3: 抓 thread workspace 文件 + grep 验证 spec §9 7 条**

```bash
THREAD_ID=$(ls -t packages/agent/backend/.deer-flow/users/*/threads/ 2>/dev/null | head -1)
echo "Testing thread: $THREAD_ID"

# §9.1 同 subagent 多次调用产物物理隔离
# 派两次 code-executor 不同 suffix，看 workspace 里是否有两个 handoff 文件
ls packages/agent/backend/.deer-flow/users/*/threads/$THREAD_ID/user-data/workspace/handoff_code_executor*.json
# 应看到 handoff_code_executor__<suffix1>.json + handoff_code_executor__<suffix2>.json

# §9.3 lead 不 read_file L2
grep "read_file" packages/agent/logs/langgraph.log | grep "handoff_.*\.json" | wc -l
# 应为 0 或极少（仅"用户特别问详情"场景下读）

# §9.4 catalog 字段不打架
# 看 data-analyst 报告中 display_name_zh 是否与 catalog YAML 一致
cat packages/agent/backend/.deer-flow/users/*/threads/$THREAD_ID/user-data/workspace/handoff_data_analyst__*.json \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d.get('insights',[]), ensure_ascii=False))"
# 看 insights 里的术语是否与 catalog YAML 的 display_name_zh 完全一致

# §9.5 下游 subagent L1 自给：grep data-analyst 调 read_file 频率
grep "subagent:data-analyst.*read_file.*handoff_" packages/agent/logs/langgraph.log | wc -l
# 应为 0 或极少

# §9.6 输出宪法生效（已在 G1 修复验证过，复测）
grep -hE "典型值|常模|金标准|参考范围|高焦虑|低焦虑" \
     packages/agent/backend/.deer-flow/users/*/threads/$THREAD_ID/archived_messages/*.json | wc -l
# 应为 0
```

- [ ] **Step 4: 记录验证结果**

在 `docs/handoffs/2026-05/2026-05-14-dogfood-followup-handoff.md` 末尾追加：

```markdown
## Spec 阶段 1 双层 handoff 协议 dogfood 验证（YYYY-MM-DD）

实施 commits: <Task 1-6 的 commit hashes>

dogfood thread: <UUID>

spec §9 验收：
- §9.1 同 subagent 多次调用产物隔离: ✅ / ❌
- §9.2 跨 run handoff 引用: ⏸️ 待阶段 2 落地后再验（本 plan 暂不支持）
- §9.3 lead 不 read_file L2: ✅ / ❌
- §9.4 catalog 字段不打架: ✅ / ❌
- §9.5 下游 L1 自给: ✅ / ❌
- §9.6 输出宪法生效: ✅ / ❌
- §9.7 5 分钟体感: ✅ / ❌

判定: 阶段 1 落地 ✅ / 部分 ⚠️ / 失败 ❌
```

- [ ] **Step 5: Commit 验证记录 + push**

```bash
cd /home/wangqiuyang/noldus-insight
git add docs/handoffs/2026-05/2026-05-14-dogfood-followup-handoff.md
git commit -m "$(cat <<'EOF'
docs(dogfood): spec 阶段 1 双层 handoff 协议端到端验证

dogfood thread <UUID> 验证 spec §9 7 条验收标准。阶段 1 落地 [✅/⚠️]，
[简短说明遗留项]。
EOF
)"

git push origin dev 2>&1 | tail -5
```

- [ ] **Step 6: 回报用户**

```
# Spec 阶段 1 双层 handoff 协议落地

## Commits（共 9 个）
<列 Task 1-6 + 验证 commits>

## Dogfood 验证（spec §9）
<对应 §9.1-§9.7 检查项的实测结果>

## 遗留
- §9.2 跨 run 引用：依赖阶段 2 落地，本 plan 不支持
- 其余按实测填

## 下一步候选
- 进 spec 阶段 2 run-scoped 路径绝缘
- 或开 spec §5.5.2 HandoffPendingActionsProvider 独立 plan
```

---

## 不要做的事（防止越权）

- ❌ **不要实施 spec §5.5.2 HandoffPendingActionsProvider**——单独 plan
- ❌ **不要实施 spec §3 完整 runs/<run_id>/ 路径绝缘**——本 plan L2 路径暂用 `${WORKSPACE_DIR}/`
- ❌ **不要重写 output-constitution.md 内容**——G1 修复已经写好
- ❌ **不要删除旧 [gate_signals] 兼容**——subagent 同时输出 L1 JSON + 旧文本块（双轨）让 lead 解析路径优先 L1、fallback 旧
- ❌ **不要改 sandbox / ArchivingSummarizationMiddleware**
- ❌ **不要 force push** / **不要用 `--no-verify`**
- ❌ **不要碰这 3 个无关文件**：
  - `docs/specs/llm-finetuning-strategy.md`
  - `docs/plans/2026-05-13-base-model-decision-memo.md`
  - `packages/agent/frontend/src/app/page.tsx`

---

## 实施完成后的状态

- 新建 3 个文件：projection.py + summarize.py + test_task_tool_handoff_suffix.py（+ test_catalog_projection.py + test_catalog_summarize_cli.py）
- 修改 7-9 个文件：task_tool / HandoffIsolationProvider / 4 subagent / lead prompt / 既有测试
- 约 8-9 个 commit 入 origin
- spec §2 + §4 + §8 阶段 1 + 部分 §3 + 收尾 §5.5 acknowledged 字段全部落地
- spec §9 验收：7 条中至少 6 条 ✅（§9.2 阶段 2 才能验）
- backend / ethoinsight 测试全绿
- dogfood: 同 subagent 多次调用不再互相覆盖
