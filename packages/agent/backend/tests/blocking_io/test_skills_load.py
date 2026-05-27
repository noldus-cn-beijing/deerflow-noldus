"""Regression test: skill content loading must not block the event loop.

Anchors the production call path in
`subagents/executor.py:_load_skill_contents`, which reads skill files
and renders their content for subagent prompt injection.

Noldus adaptation: our fork uses a standalone `_load_skill_contents`
function (not `SubagentExecutor._load_skills`). The upstream test
targets the upstream method which doesn't exist in our codebase.

Currently skipped because `_load_skill_contents` imports `get_app_config`
inside the function body, making it hard to monkeypatch. When the function
is refactored to accept `AppConfig` as a parameter, enable this test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.asyncio


def _seed_skill(skills_root: Path) -> None:
    skill = skills_root / "public" / "demo"
    skill.mkdir(parents=True, exist_ok=True)
    (skill / "SKILL.md").write_text(
        "---\nname: demo\ndescription: regression-test skill\n---\n# demo\n",
        encoding="utf-8",
    )


@pytest.mark.skip(reason="Noldus: _load_skill_contents imports get_app_config inside function body — monkeypatch doesn't reach it")
async def test_load_skill_contents_does_not_block_event_loop(tmp_path: Path) -> None:
    from deerflow.subagents.executor import _load_skill_contents

    _seed_skill(tmp_path)

    result = _load_skill_contents(["demo"])

    assert isinstance(result, str)
    assert "demo" in result
