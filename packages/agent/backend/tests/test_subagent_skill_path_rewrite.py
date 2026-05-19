"""Tests for ``_load_skill_contents`` skill path rewriting (PR-1 fix).

Background: ``SKILL.md`` documents commonly reference sibling files with
relative paths like ``references/foo.md``. When the file content is inlined
into a subagent's system_prompt the LLM has no notion of "current working
directory" — it sees the relative path and starts probing absolute paths
(``cat /mnt/user-data/skills/...``, ``find /mnt -name ...``), wasting turns
and sometimes giving up entirely.

Reproduced in 2026-05-19 dogfood thread 78ccb52b: code-executor ran
``cat /mnt/user-data/skills/ethoinsight/references/output-constitution.md``
3+ times before LoopDetection fired, simply because its SKILL.md referenced
``references/execution-conventions.md`` without an absolute prefix.

Fix: ``_load_skill_contents`` now:
  1. Wraps each skill with ``<skill name="X" base_path="/mnt/skills/<cat>/X">``
     so the LLM has the anchor in one obvious place.
  2. Rewrites in-line ``references/foo.md`` (and the same in markdown link
     bodies / bash invocations) into the absolute ``<base_path>/references/foo.md``.

Out of scope: rewrites are deliberately limited to ``references/...`` prefix.
Other relative references (``templates/...`` etc.) only get rewritten if
the skill ships such a folder — covered by the same prefix list in the impl.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from deerflow.skills.types import Skill, SkillCategory


def _make_fake_skill(tmp_path: Path, name: str, content: str, category: SkillCategory = SkillCategory.CUSTOM) -> Skill:
    skill_dir = tmp_path / category.value / name
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(content, encoding="utf-8")
    return Skill(
        name=name,
        description="desc",
        license=None,
        skill_dir=skill_dir,
        skill_file=skill_file,
        relative_path=Path(name),
        category=category,
        enabled=True,
    )


def _stub_storage(monkeypatch, skills: list[Skill]) -> None:
    fake_storage = MagicMock()
    fake_storage.load_skills = MagicMock(return_value=skills)
    monkeypatch.setattr(
        "deerflow.skills.render.get_or_new_skill_storage",
        lambda: fake_storage,
        raising=False,
    )
    # Also stub the import-time reference used inside render_skill_sections.
    import deerflow.skills.storage as storage_mod
    monkeypatch.setattr(storage_mod, "get_or_new_skill_storage", lambda: fake_storage)


def test_empty_skill_list_returns_empty_string(tmp_path, monkeypatch):
    from deerflow.skills.render import render_skill_sections

    _stub_storage(monkeypatch, [])
    assert render_skill_sections([]) == ""


def test_skill_not_found_skips_silently(tmp_path, monkeypatch):
    from deerflow.skills.render import render_skill_sections

    _stub_storage(monkeypatch, [_make_fake_skill(tmp_path, "real-skill", "# Real\n")])
    out = render_skill_sections(["unknown-skill"])
    assert out == ""


def test_skill_wrapper_includes_base_path_attribute(tmp_path, monkeypatch):
    """The <skill> XML wrapper must surface the absolute base path so the LLM
    can stop guessing where references live."""
    from deerflow.skills.render import render_skill_sections

    skill = _make_fake_skill(
        tmp_path,
        "ethoinsight-code",
        "# Skill body\n\n执行宪法见 `references/execution-conventions.md`。\n",
        SkillCategory.CUSTOM,
    )
    _stub_storage(monkeypatch, [skill])
    out = render_skill_sections(["ethoinsight-code"])
    assert 'name="ethoinsight-code"' in out
    assert 'base_path="/mnt/skills/custom/ethoinsight-code"' in out


def test_inline_references_path_rewritten_to_absolute(tmp_path, monkeypatch):
    """references/foo.md inside the SKILL.md body gets prefixed with the
    skill's absolute container path."""
    from deerflow.skills.render import render_skill_sections

    body = (
        "# Code Executor\n\n"
        "执行约束:`references/execution-conventions.md`\n"
        "错误恢复:`references/error-recovery.md`\n"
        "质检:`references/quality-checks.md`\n"
    )
    skill = _make_fake_skill(tmp_path, "ethoinsight-code", body, SkillCategory.CUSTOM)
    _stub_storage(monkeypatch, [skill])
    out = render_skill_sections(["ethoinsight-code"])
    # All three references rewritten
    assert "/mnt/skills/custom/ethoinsight-code/references/execution-conventions.md" in out
    assert "/mnt/skills/custom/ethoinsight-code/references/error-recovery.md" in out
    assert "/mnt/skills/custom/ethoinsight-code/references/quality-checks.md" in out
    # Bare relative form no longer appears (preventing LLM from grabbing the stale form)
    assert "`references/execution-conventions.md`" not in out


def test_yaml_frontmatter_still_stripped(tmp_path, monkeypatch):
    from deerflow.skills.render import render_skill_sections

    body = "---\nname: x\ndescription: d\n---\n\n# X\n\n`references/a.md`\n"
    skill = _make_fake_skill(tmp_path, "x-skill", body, SkillCategory.CUSTOM)
    _stub_storage(monkeypatch, [skill])
    out = render_skill_sections(["x-skill"])
    assert "name: x" not in out, "Frontmatter must still be stripped"
    assert "/mnt/skills/custom/x-skill/references/a.md" in out


def test_markdown_link_body_rewritten(tmp_path, monkeypatch):
    """Markdown link syntax [text](references/foo.md) — body part also rewritten."""
    from deerflow.skills.render import render_skill_sections

    body = "See [conventions](references/execution-conventions.md) for details.\n"
    skill = _make_fake_skill(tmp_path, "ethoinsight-code", body, SkillCategory.CUSTOM)
    _stub_storage(monkeypatch, [skill])
    out = render_skill_sections(["ethoinsight-code"])
    assert "[conventions](/mnt/skills/custom/ethoinsight-code/references/execution-conventions.md)" in out


def test_already_absolute_paths_not_double_rewritten(tmp_path, monkeypatch):
    """If the doc already uses /mnt/skills/... absolute paths, leave them alone."""
    from deerflow.skills.render import render_skill_sections

    body = "See `/mnt/skills/custom/ethoinsight/references/output-constitution.md`.\n"
    skill = _make_fake_skill(tmp_path, "ethoinsight-code", body, SkillCategory.CUSTOM)
    _stub_storage(monkeypatch, [skill])
    out = render_skill_sections(["ethoinsight-code"])
    # Path appears exactly once — no double prefixing
    assert out.count("/mnt/skills/custom/ethoinsight/references/output-constitution.md") == 1
    assert "/mnt/skills/custom/ethoinsight-code//mnt/skills/" not in out


def test_public_category_path_uses_public_prefix(tmp_path, monkeypatch):
    """Public skills resolve to /mnt/skills/public/..., not custom/..."""
    from deerflow.skills.render import render_skill_sections

    body = "# Public\n\n`references/a.md`\n"
    skill = _make_fake_skill(tmp_path, "public-skill", body, SkillCategory.PUBLIC)
    _stub_storage(monkeypatch, [skill])
    out = render_skill_sections(["public-skill"])
    assert 'base_path="/mnt/skills/public/public-skill"' in out
    assert "/mnt/skills/public/public-skill/references/a.md" in out


def test_multiple_skills_each_get_own_base_path(tmp_path, monkeypatch):
    from deerflow.skills.render import render_skill_sections

    skill_a = _make_fake_skill(tmp_path, "skill-a", "`references/a.md`\n", SkillCategory.CUSTOM)
    skill_b = _make_fake_skill(tmp_path, "skill-b", "`references/b.md`\n", SkillCategory.CUSTOM)
    _stub_storage(monkeypatch, [skill_a, skill_b])
    out = render_skill_sections(["skill-a", "skill-b"])
    assert "/mnt/skills/custom/skill-a/references/a.md" in out
    assert "/mnt/skills/custom/skill-b/references/b.md" in out
    # Cross-contamination check: skill-a's body should NOT mention skill-b's base
    a_section = out.split("</skill>")[0]
    assert "skill-b" not in a_section
