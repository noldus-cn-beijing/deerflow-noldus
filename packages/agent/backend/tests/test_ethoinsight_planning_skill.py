"""Tests for ethoinsight-planning skill integration."""

import json
from pathlib import Path

from deerflow.skills.storage import get_or_new_skill_storage


def _find_skill(name: str):
    for skill in get_or_new_skill_storage().load_skills(enabled_only=False):
        if skill.name == name:
            return skill
    return None


def test_planning_skill_is_discovered():
    """ethoinsight-planning skill should be discoverable by skills loader."""
    skill = _find_skill("ethoinsight-planning")
    assert skill is not None, "ethoinsight-planning skill not found in skills/"


def test_planning_skill_is_enabled_in_config():
    """extensions_config.json should enable ethoinsight-planning."""
    # backend/tests/ -> backend/ -> packages/agent/
    repo_root = Path(__file__).resolve().parents[2]
    config_path = repo_root / "extensions_config.json"
    assert config_path.exists(), f"extensions_config.json not found at {config_path}"

    config = json.loads(config_path.read_text(encoding="utf-8"))
    skill_state = config.get("skills", {}).get("ethoinsight-planning")
    assert skill_state is not None, "ethoinsight-planning not registered in extensions_config.json"
    assert skill_state.get("enabled") is True, "ethoinsight-planning must be enabled"


def test_planning_skill_has_required_references():
    """Planning skill should ship 5 reference files for progressive loading."""
    skill = _find_skill("ethoinsight-planning")
    assert skill is not None

    references_dir = Path(skill.skill_dir) / "references"
    assert references_dir.is_dir(), f"references/ directory missing at {references_dir}"

    required = {
        "intent-classification.md",
        "planning-templates.md",
        "design-type-keywords.md",
        "quality-gates.md",
        "failure-recovery.md",
    }
    actual = {p.name for p in references_dir.iterdir() if p.is_file()}
    missing = required - actual
    assert not missing, f"Missing reference files: {missing}"


def test_planning_skill_description_mentions_trigger_conditions():
    """SKILL.md description should state when to load (uploaded data + analysis request)."""
    skill = _find_skill("ethoinsight-planning")
    assert skill is not None

    description = skill.description.lower()
    assert "uploaded" in description or "data" in description
    assert "analysis" in description or "analyze" in description
