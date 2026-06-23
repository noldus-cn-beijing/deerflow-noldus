"""Tests for identify_ev19_template per_file_grouping (S2, 2026-06-12).

Verifies the identify tool now iterates ALL uploaded_files and returns
per_file_grouping, so the lead agent can determine groups without
inspecting each file individually.

Test coverage:
  1. per_file_grouping with single file → grouping fields extracted
  2. multi-file traversal → per_file_grouping has N keys
  3. tolerance — missing/corrupt non-first file doesn't block identify
  4. performance contract — only parse_header called, never parse_trajectory
  5. SSOT — identify and inspect use same extract_grouping_fields import
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from deerflow.tools.builtins.identify_ev19_template_tool import (
    identify_ev19_template_tool,
)

# Backend root for resolving prompt.py path
BACKEND_ROOT = Path(__file__).resolve().parents[1]

# Modules that identify imports lazily inside the function body.
_SANDBOX_TOOLS = "deerflow.sandbox.tools"
_PARSE_CORE = "ethoinsight.parse._core"
_EV19_FACTS = "ethoinsight.ev19_facts"
_ID_MOD = "deerflow.tools.builtins.identify_ev19_template_tool"

# A minimal EV19 template map that ensures the tool enters the "ok" path
# (single candidate) instead of "unknown" (which doesn't include per_file_grouping).
_MOCK_TEMPLATE_MAP = {"open_field": ["OpenFieldRectangle-AllZones"]}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_runtime(workspace_path: str) -> MagicMock:
    """Build a mock ToolRuntime with thread_data containing workspace_path."""
    runtime = MagicMock()
    runtime.state = {
        "thread_data": {
            "workspace_path": workspace_path,
            "uploads_path": "/mnt/user-data/uploads",
            "outputs_path": "/mnt/user-data/outputs",
        }
    }
    return runtime


def _header_with_grouping(group: str = "", treatment: str = "") -> dict:
    """Return a minimal parse_header result with grouping metadata."""
    raw = {}
    if group:
        raw["Group"] = group
    if treatment:
        raw["Treatment"] = treatment
    return {
        "columns": ["Trial time", "X center", "Y center", "in_zone"],
        "raw_metadata": raw,
        "experiment": "Test Experiment",
        "trial_name": "Trial 1",
        "arena": "Arena 1",
        "subject": "Subject 1",
    }


# ---------------------------------------------------------------------------
# 1. per_file_grouping: single file extraction
# ---------------------------------------------------------------------------
class TestPerFileGroupingSingleFile:
    def test_single_file_with_group_field(self, tmp_path):
        """identify returns per_file_grouping with extracted Group field."""
        workspace = str(tmp_path / "workspace")
        Path(workspace).mkdir(parents=True, exist_ok=True)

        upload_path = tmp_path / "uploads"
        upload_path.mkdir(parents=True, exist_ok=True)
        real_file = upload_path / "test_data.xlsx"
        real_file.write_text("dummy")

        runtime = _make_runtime(workspace)
        header = _header_with_grouping(group="XX")

        with patch(f"{_SANDBOX_TOOLS}.replace_virtual_path", return_value=str(real_file)), \
             patch(f"{_PARSE_CORE}.detect_ethovision", return_value=True), \
             patch(f"{_PARSE_CORE}.parse_header", return_value=header), \
             patch(f"{_ID_MOD}._resolve_skills_ref_dir", return_value=tmp_path / "skills_ref"), \
             patch(f"{_ID_MOD}._read_markdown_section", return_value=None), \
             patch(f"{_ID_MOD}._extract_template_recommendations", return_value=[]), \
             patch(f"{_EV19_FACTS}.EV19_TEMPLATE_PARADIGM_MAP", _MOCK_TEMPLATE_MAP):
            result = identify_ev19_template_tool.func(
                runtime=runtime,
                uploaded_files=["/mnt/user-data/uploads/test_data.xlsx"],
                user_message="旷场实验数据",
            )

        assert "per_file_grouping" in result
        assert result["per_file_grouping"] == {"test_data.xlsx": {"Group": "XX"}}

    def test_single_file_with_treatment_field(self, tmp_path):
        """Grouping extraction picks up Treatment field."""
        workspace = str(tmp_path / "workspace")
        Path(workspace).mkdir(parents=True, exist_ok=True)
        upload_path = tmp_path / "uploads"
        upload_path.mkdir(parents=True, exist_ok=True)
        real_file = upload_path / "epm_trial1.txt"
        real_file.write_text("dummy")

        runtime = _make_runtime(workspace)
        header = _header_with_grouping(treatment="DrugA")

        with patch(f"{_SANDBOX_TOOLS}.replace_virtual_path", return_value=str(real_file)), \
             patch(f"{_PARSE_CORE}.detect_ethovision", return_value=True), \
             patch(f"{_PARSE_CORE}.parse_header", return_value=header), \
             patch(f"{_ID_MOD}._resolve_skills_ref_dir", return_value=tmp_path / "skills_ref"), \
             patch(f"{_ID_MOD}._read_markdown_section", return_value=None), \
             patch(f"{_ID_MOD}._extract_template_recommendations", return_value=[]), \
             patch(f"{_EV19_FACTS}.EV19_TEMPLATE_PARADIGM_MAP", _MOCK_TEMPLATE_MAP):
            result = identify_ev19_template_tool.func(
                runtime=runtime,
                uploaded_files=["/mnt/user-data/uploads/epm_trial1.txt"],
                user_message="旷场实验",
            )

        assert "per_file_grouping" in result
        assert result["per_file_grouping"] == {"epm_trial1.txt": {"Treatment": "DrugA"}}

    def test_single_file_with_no_grouping_fields(self, tmp_path):
        """per_file_grouping is empty dict when no grouping metadata found."""
        workspace = str(tmp_path / "workspace")
        Path(workspace).mkdir(parents=True, exist_ok=True)
        upload_path = tmp_path / "uploads"
        upload_path.mkdir(parents=True, exist_ok=True)
        real_file = upload_path / "no_group.txt"
        real_file.write_text("dummy")

        runtime = _make_runtime(workspace)
        header = _header_with_grouping()  # no group or treatment

        with patch(f"{_SANDBOX_TOOLS}.replace_virtual_path", return_value=str(real_file)), \
             patch(f"{_PARSE_CORE}.detect_ethovision", return_value=True), \
             patch(f"{_PARSE_CORE}.parse_header", return_value=header), \
             patch(f"{_ID_MOD}._resolve_skills_ref_dir", return_value=tmp_path / "skills_ref"), \
             patch(f"{_ID_MOD}._read_markdown_section", return_value=None), \
             patch(f"{_ID_MOD}._extract_template_recommendations", return_value=[]), \
             patch(f"{_EV19_FACTS}.EV19_TEMPLATE_PARADIGM_MAP", _MOCK_TEMPLATE_MAP):
            result = identify_ev19_template_tool.func(
                runtime=runtime,
                uploaded_files=["/mnt/user-data/uploads/no_group.txt"],
                user_message="旷场实验",
            )

        assert "per_file_grouping" in result
        assert result["per_file_grouping"] == {}  # empty, no grouping found


# ---------------------------------------------------------------------------
# 2. multi-file traversal
# ---------------------------------------------------------------------------
class TestPerFileGroupingMultiFile:
    def test_three_files_each_with_grouping(self, tmp_path):
        """per_file_grouping contains one entry per file with grouping metadata."""
        workspace = str(tmp_path / "workspace")
        Path(workspace).mkdir(parents=True, exist_ok=True)
        upload_path = tmp_path / "uploads"
        upload_path.mkdir(parents=True, exist_ok=True)

        files = [
            ("Trial 1.xlsx", "XX"),
            ("Trial 2.xlsx", "YY"),
            ("Trial 3.xlsx", "ZZ"),
        ]
        for fname, _ in files:
            (upload_path / fname).write_text("dummy")

        runtime = _make_runtime(workspace)

        headers_by_file = {
            str(upload_path / fname): _header_with_grouping(group=grp)
            for fname, grp in files
        }

        def _side_effect_parse_header(path):
            return headers_by_file[path]

        with patch(f"{_SANDBOX_TOOLS}.replace_virtual_path",
                   side_effect=lambda p, td: str(upload_path / Path(p).name)), \
             patch(f"{_PARSE_CORE}.detect_ethovision", return_value=True), \
             patch(f"{_PARSE_CORE}.parse_header", side_effect=_side_effect_parse_header), \
             patch(f"{_ID_MOD}._resolve_skills_ref_dir", return_value=tmp_path / "skills_ref"), \
             patch(f"{_ID_MOD}._read_markdown_section", return_value=None), \
             patch(f"{_ID_MOD}._extract_template_recommendations", return_value=[]), \
             patch(f"{_EV19_FACTS}.EV19_TEMPLATE_PARADIGM_MAP", _MOCK_TEMPLATE_MAP):
            result = identify_ev19_template_tool.func(
                runtime=runtime,
                uploaded_files=[f"/mnt/user-data/uploads/{fname}" for fname, _ in files],
                user_message="旷场实验",
            )

        assert "per_file_grouping" in result
        assert len(result["per_file_grouping"]) == 3
        assert result["per_file_grouping"]["Trial 1.xlsx"] == {"Group": "XX"}
        assert result["per_file_grouping"]["Trial 2.xlsx"] == {"Group": "YY"}
        assert result["per_file_grouping"]["Trial 3.xlsx"] == {"Group": "ZZ"}


# ---------------------------------------------------------------------------
# 3. failure tolerance (non-first file missing/corrupt)
# ---------------------------------------------------------------------------
class TestPerFileGroupingTolerance:
    def test_missing_nonfirst_file_does_not_block_identify(self, tmp_path):
        """Non-first file missing is skipped; per_file_grouping contains remaining."""
        workspace = str(tmp_path / "workspace")
        Path(workspace).mkdir(parents=True, exist_ok=True)
        upload_path = tmp_path / "uploads"
        upload_path.mkdir(parents=True, exist_ok=True)

        # file1 must exist for Step 3 validation to pass
        (upload_path / "file1.xlsx").write_text("dummy")
        (upload_path / "file3.xlsx").write_text("dummy")
        # file2 intentionally NOT created — it's missing

        runtime = _make_runtime(workspace)

        headers = {
            str(upload_path / "file1.xlsx"): _header_with_grouping(group="XX"),
            str(upload_path / "file3.xlsx"): _header_with_grouping(group="ZZ"),
        }

        def _side_effect_parse_header(path):
            return headers[path]

        with patch(f"{_SANDBOX_TOOLS}.replace_virtual_path",
                   side_effect=lambda p, td: str(upload_path / Path(p).name)), \
             patch(f"{_PARSE_CORE}.detect_ethovision", return_value=True), \
             patch(f"{_PARSE_CORE}.parse_header", side_effect=_side_effect_parse_header), \
             patch(f"{_ID_MOD}._resolve_skills_ref_dir", return_value=tmp_path / "skills_ref"), \
             patch(f"{_ID_MOD}._read_markdown_section", return_value=None), \
             patch(f"{_ID_MOD}._extract_template_recommendations", return_value=[]), \
             patch(f"{_EV19_FACTS}.EV19_TEMPLATE_PARADIGM_MAP", _MOCK_TEMPLATE_MAP):
            result = identify_ev19_template_tool.func(
                runtime=runtime,
                uploaded_files=[
                    "/mnt/user-data/uploads/file1.xlsx",  # valid (first file)
                    "/mnt/user-data/uploads/file2.xlsx",  # MISSING on disk
                    "/mnt/user-data/uploads/file3.xlsx",  # valid
                ],
                user_message="旷场实验",
            )

        assert "per_file_grouping" in result
        assert "file2.xlsx" not in result["per_file_grouping"]  # skipped
        assert result["per_file_grouping"]["file1.xlsx"] == {"Group": "XX"}
        assert result["per_file_grouping"]["file3.xlsx"] == {"Group": "ZZ"}

    def test_parse_failure_nonfirst_file_does_not_block(self, tmp_path):
        """Non-first file that throws on parse_header is skipped silently."""
        workspace = str(tmp_path / "workspace")
        Path(workspace).mkdir(parents=True, exist_ok=True)
        upload_path = tmp_path / "uploads"
        upload_path.mkdir(parents=True, exist_ok=True)

        (upload_path / "good.xlsx").write_text("dummy")
        (upload_path / "bad.xlsx").write_text("dummy")

        runtime = _make_runtime(workspace)
        header_good = _header_with_grouping(group="XX")

        def _side_effect_parse_header(path):
            if "bad" in path:
                raise RuntimeError("simulated parse failure")
            return header_good

        with patch(f"{_SANDBOX_TOOLS}.replace_virtual_path",
                   side_effect=lambda p, td: str(upload_path / Path(p).name)), \
             patch(f"{_PARSE_CORE}.detect_ethovision", return_value=True), \
             patch(f"{_PARSE_CORE}.parse_header", side_effect=_side_effect_parse_header), \
             patch(f"{_ID_MOD}._resolve_skills_ref_dir", return_value=tmp_path / "skills_ref"), \
             patch(f"{_ID_MOD}._read_markdown_section", return_value=None), \
             patch(f"{_ID_MOD}._extract_template_recommendations", return_value=[]), \
             patch(f"{_EV19_FACTS}.EV19_TEMPLATE_PARADIGM_MAP", _MOCK_TEMPLATE_MAP):
            result = identify_ev19_template_tool.func(
                runtime=runtime,
                uploaded_files=[
                    "/mnt/user-data/uploads/good.xlsx",
                    "/mnt/user-data/uploads/bad.xlsx",
                ],
                user_message="旷场实验",
            )

        assert "per_file_grouping" in result
        assert "bad.xlsx" not in result["per_file_grouping"]
        assert result["per_file_grouping"]["good.xlsx"] == {"Group": "XX"}


# ---------------------------------------------------------------------------
# 4. performance contract — only parse_header, never parse_trajectory
# ---------------------------------------------------------------------------
class TestPerFileGroupingPerformanceContract:
    def test_parse_trajectory_not_called(self, tmp_path):
        """Step 3.5 only calls parse_header, never parse_trajectory (performance contract)."""
        workspace = str(tmp_path / "workspace")
        Path(workspace).mkdir(parents=True, exist_ok=True)
        upload_path = tmp_path / "uploads"
        upload_path.mkdir(parents=True, exist_ok=True)

        (upload_path / "t1.xlsx").write_text("dummy")
        (upload_path / "t2.xlsx").write_text("dummy")

        runtime = _make_runtime(workspace)
        header = _header_with_grouping(group="XX")

        # Spy on parse_trajectory to assert it is NEVER called.
        traj_spy = MagicMock()

        with patch(f"{_SANDBOX_TOOLS}.replace_virtual_path",
                   side_effect=lambda p, td: str(upload_path / Path(p).name)), \
             patch(f"{_PARSE_CORE}.detect_ethovision", return_value=True), \
             patch(f"{_PARSE_CORE}.parse_header", return_value=header), \
             patch(f"{_ID_MOD}._resolve_skills_ref_dir", return_value=tmp_path / "skills_ref"), \
             patch(f"{_ID_MOD}._read_markdown_section", return_value=None), \
             patch(f"{_ID_MOD}._extract_template_recommendations", return_value=[]), \
             patch(f"{_EV19_FACTS}.EV19_TEMPLATE_PARADIGM_MAP", _MOCK_TEMPLATE_MAP), \
             patch(f"{_PARSE_CORE}.parse_trajectory", traj_spy):
            identify_ev19_template_tool.func(
                runtime=runtime,
                uploaded_files=[
                    "/mnt/user-data/uploads/t1.xlsx",
                    "/mnt/user-data/uploads/t2.xlsx",
                ],
                user_message="旷场实验",
            )

        traj_spy.assert_not_called()


# ---------------------------------------------------------------------------
# 5. SSOT — identify and inspect share the same extract_grouping_fields import
# ---------------------------------------------------------------------------
class TestGroupingFieldsSSOT:
    def test_same_function_used_by_both_tools(self):
        """inspect imports extract_grouping_fields from the shared _ev19_grouping module."""
        from deerflow.tools.builtins._ev19_grouping import extract_grouping_fields as shared_func
        from deerflow.tools.builtins.inspect_uploaded_file_tool import extract_grouping_fields as inspect_func

        assert inspect_func is shared_func, (
            "inspect_uploaded_file_tool must import extract_grouping_fields "
            "from deerflow.tools.builtins._ev19_grouping (SSOT, no dupe)"
        )

    def test_identify_also_uses_shared_module(self):
        """Spot-check: identify's source contains the SSOT import."""
        import inspect

        # identify_ev19_template_tool is a StructuredTool; get its underlying func
        source = inspect.getsource(identify_ev19_template_tool.func)
        assert "from deerflow.tools.builtins._ev19_grouping import extract_grouping_fields" in source, (
            "identify_ev19_template_tool must import extract_grouping_fields from _ev19_grouping (SSOT)"
        )

    def test_shared_extract_handles_animal_id(self):
        """Animal ID is returned alongside grouping fields as a subject identifier."""
        from deerflow.tools.builtins._ev19_grouping import extract_grouping_fields

        result = extract_grouping_fields({"Animal ID": "1", "Group": "Control"})
        assert result["Animal ID"] == "1"
        assert result["Group"] == "Control"

    def test_shared_extract_animal_id_takes_first_variant_only(self):
        """Only the FIRST matching Animal ID variant is kept (break semantics).

        Locks the parity with the original inspect _extract_grouping_fields: Animal ID
        is a subject identifier (not a grouping field), so when multiple aliases appear
        in one header we keep only the first per _ANIMAL_ID_KEYS order to avoid polluting
        the grouping dict with duplicate identifiers. Regression guard for the SSOT
        extraction (the shared function must not collect every alias).
        """
        from deerflow.tools.builtins._ev19_grouping import extract_grouping_fields

        # Two Animal ID aliases present → only the first ("Animal ID") survives.
        result = extract_grouping_fields({"Animal ID": "1", "Animal": "mouse-7"})
        assert result == {"Animal ID": "1"}

        # Chinese aliases: "动物 ID" precedes "动物编号" in _ANIMAL_ID_KEYS → it wins.
        result_cn = extract_grouping_fields({"动物编号": "7", "动物 ID": "8"})
        assert result_cn == {"动物 ID": "8"}

        # Animal ID coexists with a real grouping field → both kept, single Animal ID.
        result_mixed = extract_grouping_fields({"Group": "XX", "Animal ID": "1", "Animal": "m7"})
        assert result_mixed == {"Group": "XX", "Animal ID": "1"}

    def test_shared_extract_behavior_empty_and_none(self):
        """Empty values skipped; None raw_metadata returns empty dict."""
        from deerflow.tools.builtins._ev19_grouping import extract_grouping_fields

        assert extract_grouping_fields({"Treatment": "", "Group": "  "}) == {"Group": "  "}
        assert extract_grouping_fields({}) == {}
        assert extract_grouping_fields(None) == {}


# ---------------------------------------------------------------------------
# 6. prompt contract — lead prompt mentions per_file_grouping
# ---------------------------------------------------------------------------
class TestPromptGroupingContract:
    def test_prompt_mentions_per_file_grouping(self):
        """Lead prompt has a positive instruction to use per_file_grouping."""
        prompt_path = BACKEND_ROOT / "packages/harness/deerflow/agents/lead_agent/prompt.py"
        prompt_text = prompt_path.read_text(encoding="utf-8")

        assert "per_file_grouping" in prompt_text, (
            "lead_agent prompt must reference per_file_grouping"
        )
        # ETHO-5 (2026-06-23): grouping upgraded from soft "优先用" suggestion to a
        # hard constraint that the determination MUST cover all uploaded files.
        # Assert the hard-constraint phrasing rather than the obsolete soft string.
        assert ("必须基于全部" in prompt_text or "覆盖全部文件" in prompt_text), (
            "prompt must hard-require grouping over ALL files' per_file_grouping (ETHO-5)"
        )
