"""Skill content rendering for subagent prompt injection.

Extracted from ``deerflow.subagents.executor`` so it can be unit-tested
without triggering the circular-import workaround in tests/conftest.py.

## What this module does

Given a list of enabled-skill names, produce a single string that can be
appended to a subagent's system_prompt. Each skill is rendered as:

    <skill name="X" base_path="/mnt/skills/<category>/X">
    {body of SKILL.md with YAML frontmatter stripped and references/* rewritten}
    </skill>

The ``base_path`` attribute + the in-body path rewrites together solve the
"subagent LLM probes filesystem to find a relative reference" failure mode
seen in 2026-05-19 dogfood thread 78ccb52b (code-executor running
``cat /mnt/user-data/skills/...`` 3+ times because its SKILL.md referenced
``references/execution-conventions.md`` without an absolute prefix).
"""
from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


# Prefixes inside a SKILL.md body that, when relative, should be rewritten
# to absolute container paths. Anything else (e.g. http(s):// links, paths
# starting with `/`) is left alone.
_RELATIVE_PREFIXES = ("references/", "templates/", "scripts/", "assets/")

# Match a relative path appearing anywhere in the body. The path itself
# must start with one of the recognized prefixes and end at the first
# character outside ``[A-Za-z0-9._/-]``. We deliberately do NOT anchor to
# any specific markdown syntax — this catches:
#   `references/foo.md`           backticked
#   [text](references/foo.md)     markdown link body
#   read_file references/foo.md   bare in instructions
#   cat references/foo.md         bare in bash sample
_RELATIVE_PATH_RE = re.compile(
    r"(?<![/A-Za-z0-9])(?:" + "|".join(re.escape(p) for p in _RELATIVE_PREFIXES) + r")[A-Za-z0-9._/-]+"
)


def _rewrite_relative_paths(body: str, base_path: str) -> str:
    """Replace relative ``references/foo.md`` (and siblings) with absolute paths.

    Idempotent: if the doc already uses absolute ``/mnt/skills/...`` paths,
    the regex's lookbehind ``(?<![/A-Za-z0-9])`` prevents matching them.
    """

    def _replace(match: re.Match[str]) -> str:
        rel = match.group(0)
        return f"{base_path}/{rel}"

    return _RELATIVE_PATH_RE.sub(_replace, body)


def _strip_frontmatter(content: str) -> str:
    if not content.startswith("---"):
        return content
    end = content.find("---", 3)
    if end == -1:
        return content
    return content[end + 3 :].strip()


def render_skill_sections(skill_names: list[str], container_base_path: str = "/mnt/skills") -> str:
    """Render the named skills into a single XML-tagged block.

    Args:
        skill_names: SKILL.md names to render. Names not present in the
            enabled-skills storage are silently skipped (with a warning).
        container_base_path: Base path where skills are mounted in the
            sandbox container. Defaults to ``/mnt/skills`` (matches the
            default in ``SkillsConfig.container_path``).

    Returns:
        Concatenated ``<skill>`` blocks separated by blank lines. Empty
        string when no skills resolve.
    """
    from deerflow.skills.storage import get_or_new_skill_storage

    all_skills = get_or_new_skill_storage().load_skills(enabled_only=True)
    skill_map = {s.name: s for s in all_skills}

    sections: list[str] = []
    for name in skill_names:
        skill = skill_map.get(name)
        if skill is None:
            logger.warning("Skill '%s' not found or not enabled, skipping injection for subagent", name)
            continue
        try:
            raw = skill.skill_file.read_text(encoding="utf-8")
        except Exception:
            logger.warning("Failed to read skill file for '%s'", name, exc_info=True)
            continue

        base_path = skill.get_container_path(container_base_path)
        body = _rewrite_relative_paths(_strip_frontmatter(raw), base_path)
        sections.append(f'<skill name="{name}" base_path="{base_path}">\n{body}\n</skill>')

    return "\n\n".join(sections)
