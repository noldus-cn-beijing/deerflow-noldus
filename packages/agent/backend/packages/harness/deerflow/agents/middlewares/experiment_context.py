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

import enum
import hashlib
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

_EMERGENCY_DOWNGRADE_FILE = "/tmp/disable_strict_handoff"


class HandoffStrictMode(str, enum.Enum):
    OFF = "off"
    WARN = "warn"
    FAIL_CLOSED = "fail_closed"


class HandoffSchemaError(Exception):
    """Raised in FAIL_CLOSED mode when handoff schema validation fails."""


def _get_strict_mode() -> HandoffStrictMode:
    """读 config + 紧急降级文件。"""
    if Path(_EMERGENCY_DOWNGRADE_FILE).exists():
        logger.warning("emergency downgrade file present, forcing WARN mode")
        return HandoffStrictMode.WARN
    from deerflow.config import get_app_config

    cfg = get_app_config()
    mode_str = getattr(cfg, "handoff_strict_mode", "warn")
    try:
        return HandoffStrictMode(mode_str)
    except ValueError:
        logger.warning("invalid handoff_strict_mode %r, falling back to WARN", mode_str)
        return HandoffStrictMode.WARN


def compute_analysis_config_id(
    catalog_default: dict,
    overrides: dict[str, float | int | str],
) -> str:
    """Deterministic 16-char hex id for a unique (catalog_default + overrides) combination.

    Canonical ordering ensures key order does not affect the hash:
    normalize (sort keys) → json.dumps → sha256 → first 16 hex chars.

    Args:
        catalog_default: Paradigm catalog defaults (paradigm key, n_per_group, etc.).
        overrides: User-supplied parameter overrides (may be empty).

    Returns:
        16-character hex string, e.g. ``"a1b2c3d4e5f67890"``.
    """
    payload = {"catalog_default": catalog_default, "overrides": overrides}
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest[:16]


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

    # Three-tier strict mode validation
    mode = _get_strict_mode()
    if mode == HandoffStrictMode.OFF:
        return data

    try:
        from deerflow.subagents.handoff_schemas import CodeExecutorHandoff

        CodeExecutorHandoff.model_validate(data)
    except Exception as e:
        if mode == HandoffStrictMode.FAIL_CLOSED:
            raise HandoffSchemaError(
                f"handoff_code_executor.json schema violation (FAIL_CLOSED): {e}"
            ) from e
        # WARN mode
        logger.warning("handoff_code_executor.json schema violation (WARN mode): %s", e)
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


def _write_user_clarification_fact_to_memory(
    key: str,
    value: str,
    thread_id: str,
    agent_name: str | None = None,
    user_id: str | None = None,
) -> bool:
    """Write a single user-clarification fact to the memory storage.

    The fact is scoped to ``thread_id`` via a marker in the content so
    ``_get_resolved_facts_context`` can filter by thread later (C1).

    Returns True on success, False on failure (non-blocking).
    """
    import uuid

    from deerflow.agents.memory import get_memory_data
    from deerflow.agents.memory.storage import get_memory_storage

    try:
        storage = get_memory_storage()
        memory_data = get_memory_data(agent_name, user_id=user_id)

        fact = {
            "id": uuid.uuid4().hex,
            "content": f"[thread:{thread_id}] {key}: {value}",
            "category": "user_clarification",
            "confidence": 1.0,
            "source": "user_clarification",
            "createdAt": datetime.now(UTC).isoformat(),
        }

        facts: list = memory_data.get("facts", [])
        facts.append(fact)
        memory_data["facts"] = facts

        storage.save(memory_data, agent_name, user_id=user_id)
        logger.info("Wrote user_clarification fact: key=%r thread=%s", key, thread_id)
        return True
    except Exception as e:
        logger.error("Failed to write user_clarification fact key=%r: %s", key, e)
        return False


def _thread_id_from_runtime(runtime: ToolRuntime | None) -> str | None:
    """Extract thread_id from the tool runtime's context.

    ``ToolRuntime.context`` is a FLAT dict with ``thread_id`` at the top level
    (the same shape read by memory_middleware / thread_data_middleware /
    loop_detection_middleware / archiving_summarization etc.). The nested
    ``configurable.thread_id`` form belongs to the RunnableConfig, NOT to
    ``runtime.context`` — reading it there returns None in production and
    silently disables the memory projection. Use the flat key.
    """
    if runtime is None:
        return None
    try:
        ctx = runtime.context
        if isinstance(ctx, dict):
            return ctx.get("thread_id")
    except Exception:
        pass
    return None


def _apply_resolved_facts(data: dict, resolved_facts: list[dict]) -> None:
    """Merge resolved_facts into the data dict under the ``resolved`` key (SSOT §4 authority)."""
    resolved: dict = data.get("resolved", {})
    if not isinstance(resolved, dict):
        resolved = {}
    for item in resolved_facts:
        key = item.get("key")
        value = item.get("value")
        if key and value is not None:
            resolved[key] = value
    data["resolved"] = resolved


def _persist_resolved_facts_to_memory(resolved_facts: list[dict], thread_id: str | None) -> None:
    """Write resolved facts to memory storage as user_clarification facts (non-blocking).

    Each fact is scoped to ``thread_id``. Failures are logged, never raised.
    """
    if not thread_id:
        logger.warning("No thread_id available; skipping memory projection for resolved_facts")
        return
    for item in resolved_facts:
        key = item.get("key")
        value = item.get("value")
        if key and value is not None:
            _write_user_clarification_fact_to_memory(
                key=str(key),
                value=str(value),
                thread_id=thread_id,
            )


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
    user_confirmed_template: bool = False,
    parameter_overrides: dict[str, float | int | str] | None = None,
    resolved_facts: list[dict] | None = None,
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
      4) Resolved facts write-through (Spec B):
         set_experiment_paradigm(resolved_facts=[dict(key="groups", value="Trial1=control, ..."), ...])
         → writes to experiment-context.json "resolved" key (authoritative) AND
           projects each fact to memory storage as user_clarification fact (LLM injection).
           Can be combined with Gate 1 or called standalone (requires existing context).

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
        confirm_template_change: Set True to confirm intentional change of ev19_template
                                 when it was already set. Required to prevent accidental
                                 mid-analysis template switching. Default False.
        user_confirmed_template: Set True when identify_ev19_template returned ambiguous
                                 (2-3 candidates) and the user has explicitly chosen one.
                                 Required to prevent the agent from silently defaulting
                                 to "recommended" without user confirmation. Default False.
        parameter_overrides: User-confirmed parameter overrides. Examples:
            ``immobility_threshold=0.5`` or ``anonymous_zone_is=in_zone``.
            The unified key ``anonymous_zone_is`` works across all three zone
            paradigms (OFT / zero_maze / LDB); the backend translates it into
            the paradigm-specific parameter (center_zone / open_zones / light_zone).
                             Stored in experiment-context.json; used to compute analysis_config_id.
                             Pass None or {} when no overrides are needed (defaults apply).
        resolved_facts: Resolved clarification answers to persist (Spec B). Each item
                        is a dict with keys ``key`` and ``value`` (both str).
                        Written to experiment-context.json ``resolved`` field
                        (authoritative) AND projected to memory facts
                        (source=user_clarification) for LLM injection. Thread-scoped.
                        Requires existing experiment-context.json. Default None.
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

        # Spec B: resolved_facts write-through (can combine with Gate 2)
        if resolved_facts:
            _apply_resolved_facts(data, resolved_facts)

        path = Path(actual_workspace) / "experiment-context.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

        # Write memory projection (non-blocking)
        if resolved_facts:
            _persist_resolved_facts_to_memory(resolved_facts, _thread_id_from_runtime(runtime))

        response_data: dict = {"status": "ok", "path": str(path), "gate_completed": gate_completed}
        if resolved_facts:
            response_data["resolved_facts_saved"] = len(resolved_facts)
        return json.dumps(response_data, ensure_ascii=False)

    # --- Standalone column_semantics path (Sprint 1: no Gate 1/2 params) ---
    # Symmetric with the resolved_facts standalone path: read existing context,
    # merge column_semantics + derive column_aliases, write back preserving all
    # other fields. Allows the agent to align columns SEPARATELY after Gate 1,
    # instead of being forced through the full-fields Gate 1 path (which would
    # clobber resolved/groups — see Bug 3 in spec 2026-06-17).
    #
    # ⚠️ Placement: MUST come BEFORE the standalone resolved_facts path so that a
    # single call carrying BOTH column_semantics + resolved_facts is handled here.
    # Otherwise the resolved_facts path below — gated only on ``if resolved_facts:``
    # — would intercept it first and silently drop column_semantics. The
    # resolved_facts path still handles the column_semantics-absent case unchanged.
    if column_semantics is not None and isinstance(column_semantics, dict):
        if existing is None:
            return json.dumps(
                {
                    "status": "error",
                    "message": "No experiment-context.json found. Call set_experiment_paradigm with paradigm fields first.",
                },
                ensure_ascii=False,
            )
        data = dict(existing)  # inherit all existing fields (Bug 3 defensive: don't clobber)
        cs = _normalize_column_semantics(column_semantics)  # reuse existing pure fn
        data["column_semantics"] = cs
        aliases = _derive_column_aliases(cs)  # reuse existing pure fn
        if aliases:
            data["column_aliases"] = aliases

        # resolved_facts may ride the same call (same combination semantics as the
        # other channels: Gate 2 / resolved_facts standalone all accept it).
        if resolved_facts:
            _apply_resolved_facts(data, resolved_facts)

        path = Path(actual_workspace) / "experiment-context.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        if resolved_facts:
            _persist_resolved_facts_to_memory(resolved_facts, _thread_id_from_runtime(runtime))

        resp: dict = {"status": "ok", "path": str(path), "column_semantics_saved": True}
        if resolved_facts:
            resp["resolved_facts_saved"] = len(resolved_facts)
        return json.dumps(resp, ensure_ascii=False)

    # --- Standalone resolved_facts path (Spec B: no Gate 1/2 params) ---
    if resolved_facts:
        if existing is None:
            return json.dumps(
                {
                    "status": "error",
                    "message": "No experiment-context.json found. Call set_experiment_paradigm with paradigm fields first.",
                },
                ensure_ascii=False,
            )

        data = dict(existing)
        _apply_resolved_facts(data, resolved_facts)

        path = Path(actual_workspace) / "experiment-context.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

        # Write memory projection (non-blocking)
        _persist_resolved_facts_to_memory(resolved_facts, _thread_id_from_runtime(runtime))

        return json.dumps(
            {"status": "ok", "path": str(path), "resolved_facts_saved": len(resolved_facts)},
            ensure_ascii=False,
        )

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

    # Sprint 4.5: store parameter_overrides + compute deterministic analysis_config_id
    overrides = parameter_overrides if parameter_overrides is not None else {}
    catalog_default = {"paradigm": paradigm, "ev19_template": ev19_template, "subject": subject}
    config_id = compute_analysis_config_id(catalog_default, overrides)

    # Inherit the existing context first (Bug 3, spec 2026-06-17): a Gate 1 re-run
    # (user re-confirming the same paradigm/template) must NOT silently drop already-
    # aligned column_semantics/column_aliases or already-resolved groups. We start from
    # ``existing`` and let this call's Gate 1 fields override; this mirrors the line-566
    # ``prior_gate_completed`` preservation but extends it to every field.
    # NB: column_semantics (591) / resolved_facts (599) handlers below run AFTER this dict
    # literal, so an explicit this-call value still wins over the inherited one.
    base = dict(existing) if isinstance(existing, dict) else {}
    data = {
        **base,
        "paradigm": paradigm,
        "paradigm_cn": paradigm_cn,
        "category": category,
        "subject": subject,
        "ev19_template": ev19_template,
        "paradigm_confirmed_at": datetime.now(UTC).isoformat(),
        "gate_completed": gate_completed,
        "parameter_overrides": overrides,
        "analysis_config_id": config_id,
    }

    # Sprint 1: column semantics alignment — write column_semantics + derive
    # column_aliases as a deterministic, write-time projection (D8/D11).
    if column_semantics is not None and isinstance(column_semantics, dict):
        cs = _normalize_column_semantics(column_semantics)
        data["column_semantics"] = cs
        aliases = _derive_column_aliases(cs)
        if aliases:
            data["column_aliases"] = aliases

    # Spec B: resolved_facts write-through (can combine with Gate 1)
    if resolved_facts:
        _apply_resolved_facts(data, resolved_facts)

    path = Path(actual_workspace) / "experiment-context.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    # Write memory projection (non-blocking)
    if resolved_facts:
        _persist_resolved_facts_to_memory(resolved_facts, _thread_id_from_runtime(runtime))

    response: dict = {
        "status": "ok",
        "path": str(path),
        "paradigm": paradigm,
        "ev19_template": ev19_template,
        "analysis_config_id": config_id,
    }
    if warning is not None:
        response["warning"] = warning
    if resolved_facts:
        response["resolved_facts_saved"] = len(resolved_facts)
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
