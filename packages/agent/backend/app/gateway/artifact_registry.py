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
