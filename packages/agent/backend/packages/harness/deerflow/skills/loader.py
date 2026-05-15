"""Compatibility shim — re-export load_skills via new storage API.

Upstream 1ad1420e deleted loader.py. This shim keeps old test monkeypatch
targets (``deerflow.skills.loader.load_skills``) working until Phase D
completes the full migration.
"""

from deerflow.skills.storage import get_or_new_skill_storage


def load_skills(*, enabled_only: bool = False):
    return get_or_new_skill_storage().load_skills(enabled_only=enabled_only)
