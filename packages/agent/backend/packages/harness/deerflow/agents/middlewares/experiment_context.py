"""Read/write experiment-context.json and handoff_code_executor.json for Gate state persistence.

TWO PATH DOMAINS:
1. Container-side (lead agent): /mnt/user-data/workspace/experiment-context.json
   - Lead agent calls set_experiment_paradigm tool (container path, sandbox translates)
   - Code-executor reads via read_file tool (container path, sandbox translates)

2. Host-side (middleware): {workspace_path}/experiment-context.json
   - GateEnforcementMiddleware reads host-side path from state["thread_data"]["workspace_path"]
   - This module provides the host-side read functions

Robustness: file-not-found returns None (never raises).
"""

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from langchain.tools import ToolRuntime, tool
from langgraph.typing import ContextT

from deerflow.agents.thread_state import ThreadDataState, ThreadState

logger = logging.getLogger(__name__)

# Container-side path — used by lead agent prompt instructions and code-executor
CONTAINER_CONTEXT_PATH = "/mnt/user-data/workspace/experiment-context.json"


def read_context(workspace_dir: str) -> dict | None:
    """Read experiment-context.json from host-side workspace_dir. Returns None if absent."""
    path = Path(workspace_dir) / "experiment-context.json"
    try:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, PermissionError, OSError) as e:
        logger.warning("Failed to read experiment-context.json: %s", e)
        return None


def context_exists(workspace_dir: str) -> bool:
    """Check if experiment-context.json exists in host-side workspace_dir."""
    return (Path(workspace_dir) / "experiment-context.json").exists()


def resolve_workspace_from_state(state: dict) -> str | None:
    """Extract host-side workspace path from agent state (set by ThreadDataMiddleware).

    Returns None if state lacks thread_data — caller should treat as auto mode or old thread.
    """
    thread_data = state.get("thread_data")
    if not isinstance(thread_data, dict):
        return None
    return thread_data.get("workspace_path")


def read_handoff(workspace_dir: str, thread_data: ThreadDataState | None = None) -> dict | None:
    """Read handoff_code_executor.json from host-side workspace_dir. Returns None if absent.

    When ``thread_data`` is provided, host paths embedded in the JSON text are reverse-masked
    back to virtual paths (e.g. ``/home/.../uploads/x.txt`` -> ``/mnt/user-data/uploads/x.txt``)
    before parsing. This shields downstream consumers from subagents that leaked host paths
    via ``Path.resolve()``; the masking re-uses the same helper that ``write_file_tool`` runs
    on write.

    The result is also schema-validated via :class:`CodeExecutorHandoff`; on validation
    failure the schema errors are recorded in ``_schema_violations`` (a soft signal — the
    raw dict is still returned so existing gate-2 checks keep working).
    """
    path = Path(workspace_dir) / "handoff_code_executor.json"
    try:
        if not path.exists():
            return None
        raw_text = path.read_text(encoding="utf-8")
    except (PermissionError, OSError) as e:
        logger.warning("Failed to read handoff_code_executor.json: %s", e)
        return None

    if thread_data is not None:
        # Reverse-mask host paths embedded in the JSON text (defense in depth: write_file
        # already masks on write, but legacy handoffs or out-of-band writers may still leak).
        from deerflow.sandbox.tools import mask_local_paths_in_output

        raw_text = mask_local_paths_in_output(raw_text, thread_data)

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse handoff_code_executor.json: %s", e)
        return None

    if not isinstance(data, dict):
        return None

    # Soft schema validation: surface violations without dropping the handoff.
    try:
        from deerflow.subagents.handoff_schemas import CodeExecutorHandoff
        CodeExecutorHandoff.model_validate(data)
    except Exception as e:  # pragma: no cover - pydantic ValidationError
        logger.warning("handoff_code_executor.json schema violation: %s", e)
        violations = data.setdefault("_schema_violations", [])
        if isinstance(violations, list):
            violations.append(str(e))

    return data


def get_critical_warnings(workspace_dir: str, thread_data: ThreadDataState | None = None) -> list[dict]:
    """Extract severity='critical' warnings from handoff. Returns empty list if none or absent."""
    handoff = read_handoff(workspace_dir, thread_data=thread_data)
    if not handoff:
        return []
    warnings = handoff.get("data_quality_warnings", [])
    if not isinstance(warnings, list):
        return []
    return [w for w in warnings if isinstance(w, dict) and w.get("severity") == "critical"]


def is_quality_acknowledged(workspace_dir: str) -> bool:
    """Check if data quality gate has been acknowledged in experiment-context.json."""
    ctx = read_context(workspace_dir)
    if not ctx:
        return False
    gate_completed = ctx.get("gate_completed", [])
    if not isinstance(gate_completed, list):
        return False
    return "gate2_quality_acknowledged" in gate_completed


def _normalize_column_semantics(cs: dict) -> dict:
    """Normalize raw column_semantics dict — add confirmed_at if missing."""
    if "confirmed_at" not in cs:
        cs = {**cs, "confirmed_at": datetime.now(UTC).isoformat()}
    return cs


def _derive_column_aliases(cs: dict) -> dict[str, str]:
    """D8/D11: derive column_aliases from column_semantics.columns.

    For each confirmed entry with a non-None resolves_to that is not "__ignore__":
      → map BOTH the re-normalized raw_name AND the raw_name itself to resolves_to.

    CRITICAL: the alias source key is computed by re-running normalize_column_name()
    on raw_name — NOT by trusting the LLM-supplied ``normalized`` field. The real
    pipeline feeds resolve already-normalized columns (parse_header → normalize_columns),
    and e.g. normalize_column_name("中心区") == "中心区" (Chinese passes through slugify),
    NOT "center". If we trusted a wrong LLM ``normalized`` value the alias key would not
    match the actual column and the remap would silently miss → metric drops.

    Mapping both the normalized form and the raw_name makes the alias fire whether resolve
    receives raw or normalized column names.

    This is a deterministic pure function computed at write-time (no resolve-time
    recomputation — preserves analysis_config_id input timing determinism).
    """
    from ethoinsight.utils import normalize_column_name

    aliases: dict[str, str] = {}
    columns = cs.get("columns", {})
    if not isinstance(columns, dict):
        return aliases
    for col_key, entry in columns.items():
        if not isinstance(entry, dict):
            continue
        if not entry.get("confirmed"):
            continue
        resolves_to = entry.get("resolves_to")
        if resolves_to is None or resolves_to == "__ignore__":
            continue
        # Source of truth for the raw name: explicit raw_name, else the dict key.
        raw_name = entry.get("raw_name", col_key)
        # Re-normalize deterministically — do NOT trust the LLM-supplied "normalized".
        aliases[normalize_column_name(raw_name)] = resolves_to
        # Belt-and-suspenders: also map the raw name verbatim.
        aliases[raw_name] = resolves_to
    return aliases


@tool("set_experiment_paradigm", parse_docstring=True)
def set_experiment_paradigm_tool(
    paradigm: str | None = None,
    paradigm_cn: str | None = None,
    category: str | None = None,
    subject: str | None = None,
    ev19_template: str | None = None,
    acknowledge_quality: bool = False,
    column_semantics: dict | None = None,
    confirm_template_change: bool = False,
    workspace_dir: str = "/mnt/user-data/workspace/",
    runtime: ToolRuntime[ContextT, ThreadState] = None,
) -> str:
    """Record the user's experiment paradigm choice and/or acknowledge data quality.

    Two usage modes:
      1) Gate 1 paradigm confirmation:
         set_experiment_paradigm(paradigm="forced_swim", paradigm_cn="...", category="...",
                                 subject="...", ev19_template="...")
         → creates experiment-context.json with gate_completed=["gate1_paradigm"]
      2) Gate 2 quality acknowledgement:
         set_experiment_paradigm(acknowledge_quality=True)
         → reads existing experiment-context.json, appends "gate2_quality_acknowledged"
           to gate_completed (preserving all other fields). Requires Gate 1 already done.
      3) Column semantics alignment (Sprint 1):
         set_experiment_paradigm(column_semantics={...})
         → writes column_semantics + derived column_aliases into experiment-context.json.
           Can be combined with Gate 1 or called separately. column_semantics schema:
           {"columns": {"<raw column name verbatim>": {
              "raw_name": "中心区",            # exact data column header (NOT translated)
              "resolves_to": "center",         # CONCEPT KEYWORD (center/border/open_arms/...),
                                               #   NOT a machine column name — the catalog layer
                                               #   translates it to a matchable column. Use null
                                               #   + "ignore": true for irrelevant columns.
              "meaning_zh": "中心分析区",       # Chinese narrative meaning for report-writer
              "confirmed": true}, ...}}

    Args:
        paradigm: English paradigm name key. Required for Gate 1 mode.
        paradigm_cn: Chinese display name. Required for Gate 1 mode.
        category: Category name. Required for Gate 1 mode.
        subject: Subject type — "rodent" | "fish" | "insect" | "other". Required for Gate 1 mode.
        ev19_template: EthoVision 19 template variant ID (e.g. "PlusMaze-AllZones"). Required for Gate 1 mode.
        acknowledge_quality: Set True to acknowledge data quality warnings (Gate 2 mode).
                             When True, all paradigm fields may be omitted — the existing
                             experiment-context.json is read and only gate_completed is updated.
        column_semantics: Column semantics dict (Sprint 1). Written as-is; column_aliases
                          is derived from it as a deterministic projection.
        confirm_template_change: Set True to confirm changing an already-set ev19_template.
        workspace_dir: Workspace directory. Default: "/mnt/user-data/workspace/".

    Returns:
        JSON confirmation with the updated context.
    """

    # Resolve the actual host workspace path from thread state.
    actual_workspace = workspace_dir
    if runtime is not None and runtime.state is not None:
        thread_data: ThreadDataState | None = runtime.state.get("thread_data")
        if thread_data is not None:
            host_workspace = thread_data.get("workspace_path")
            if host_workspace is not None:
                actual_workspace = host_workspace

    existing = read_context(actual_workspace)

    # --- Gate 2: quality acknowledgement ---
    if acknowledge_quality:
        if existing is None:
            return json.dumps(
                {
                    "status": "error",
                    "message": "Cannot acknowledge quality before Gate 1. Call set_experiment_paradigm with paradigm fields first.",
                },
                ensure_ascii=False,
            )
        gate_completed = existing.get("gate_completed", [])
        if not isinstance(gate_completed, list):
            gate_completed = []
        if "gate2_quality_acknowledged" not in gate_completed:
            gate_completed.append("gate2_quality_acknowledged")
        data = {**existing, "gate_completed": gate_completed, "gate2_acknowledged_at": datetime.now(UTC).isoformat()}
        path = Path(actual_workspace) / "experiment-context.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        return json.dumps({"status": "ok", "path": str(path), "gate_completed": gate_completed}, ensure_ascii=False)

    # --- Gate 1: paradigm confirmation ---
    from ethoinsight.ev19_facts import is_paradigm_template_compatible, is_valid_ev19_template, suggest_nearby_templates

    required = {"paradigm": paradigm, "paradigm_cn": paradigm_cn, "category": category, "subject": subject, "ev19_template": ev19_template}
    missing = [k for k, v in required.items() if not v]
    if missing:
        return json.dumps({"status": "error", "message": f"Missing required fields for Gate 1: {missing}"}, ensure_ascii=False)

    # Validate ev19_template against the 62-variant whitelist
    if not is_valid_ev19_template(ev19_template):
        candidates = suggest_nearby_templates(ev19_template)
        logger.warning("Unknown ev19_template=%r; candidates=%s", ev19_template, candidates)
        return json.dumps(
            {
                "status": "error",
                "message": f"Unknown ev19_template: {ev19_template!r}. Choose from the 62 known EV19 variants.",
                "candidates": candidates,
            },
            ensure_ascii=False,
        )

    # Check paradigm–template compatibility (soft warning, does not block)
    warning: str | None = None
    if not is_paradigm_template_compatible(paradigm, ev19_template):
        warning = f"ev19_template {ev19_template!r} is not in the recommended list for paradigm {paradigm!r}. Proceeding anyway."
        logger.warning(warning)

    # Preserve gate2_quality_acknowledged if it was already set (user changing paradigm)
    prior_gate_completed = existing.get("gate_completed", []) if isinstance(existing, dict) else []
    gate_completed: list[str] = ["gate1_paradigm"]
    if "gate2_quality_acknowledged" in prior_gate_completed:
        gate_completed.append("gate2_quality_acknowledged")

    data = {
        "paradigm": paradigm,
        "paradigm_cn": paradigm_cn,
        "category": category,
        "subject": subject,
        "ev19_template": ev19_template,
        "paradigm_confirmed_at": datetime.now(UTC).isoformat(),
        "gate_completed": gate_completed,
    }

    # Sprint 1: column semantics alignment — write column_semantics + derive
    # column_aliases as a deterministic, write-time projection (D8/D11).
    if column_semantics is not None and isinstance(column_semantics, dict):
        cs = _normalize_column_semantics(column_semantics)
        data["column_semantics"] = cs
        aliases = _derive_column_aliases(cs)
        if aliases:
            data["column_aliases"] = aliases
    path = Path(actual_workspace) / "experiment-context.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    response: dict = {"status": "ok", "path": str(path), "paradigm": paradigm, "ev19_template": ev19_template}
    if warning is not None:
        response["warning"] = warning
    return json.dumps(response, ensure_ascii=False)


@tool("set_viz_choice", parse_docstring=True)
def set_viz_choice_tool(
    choice: Literal["yes", "no"],
    workspace_dir: str = "/mnt/user-data/workspace/",
    runtime: ToolRuntime[ContextT, ThreadState] = None,
) -> str:
    """Record the user's answer to the 'do you want a chart?' clarification.

    Use this AFTER ask_clarification has presented the viz question to the user
    and the user has replied. Writes gate3_viz_acknowledged + viz_choice to
    experiment-context.json so IntentPostStepAskGateProvider knows the user
    has answered and the lead can proceed to dispatch chart-maker (yes) or
    skip to report-writer (no).

    Args:
        choice: "yes" if user wants charts; "no" otherwise.
        workspace_dir: Workspace directory. Default: "/mnt/user-data/workspace/".
    """

    # Resolve the actual host workspace path from thread state.
    actual_workspace = workspace_dir
    if runtime is not None and runtime.state is not None:
        thread_data: ThreadDataState | None = runtime.state.get("thread_data")
        if thread_data is not None:
            host_workspace = thread_data.get("workspace_path")
            if host_workspace is not None:
                actual_workspace = host_workspace

    existing = read_context(actual_workspace)
    if existing is None:
        return json.dumps(
            {
                "status": "error",
                "message": "experiment-context.json missing; call set_experiment_paradigm first.",
            },
            ensure_ascii=False,
        )

    gate_completed = existing.get("gate_completed", [])
    if not isinstance(gate_completed, list):
        gate_completed = []
    if "gate3_viz_acknowledged" not in gate_completed:
        gate_completed.append("gate3_viz_acknowledged")

    data = {
        **existing,
        "gate_completed": gate_completed,
        "viz_choice": choice,
        "viz_acknowledged_at": datetime.now(UTC).isoformat(),
    }
    path = Path(actual_workspace) / "experiment-context.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    return json.dumps(
        {"status": "ok", "viz_choice": choice, "gate_completed": gate_completed},
        ensure_ascii=False,
    )
