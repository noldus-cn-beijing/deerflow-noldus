"""Tests for the analysis template tool."""

import ast
import json

import pytest

from ethoinsight.templates.tool import (
    get_analysis_template_tool,
    get_available_paradigms,
    render_template,
)


class TestGetAvailableParadigms:
    def test_discovers_shoaling(self):
        paradigms = get_available_paradigms()
        assert "shoaling" in paradigms

    def test_excludes_init_and_tool(self):
        paradigms = get_available_paradigms()
        assert "__init__" not in paradigms
        assert "tool" not in paradigms

    def test_returns_sorted_list(self):
        paradigms = get_available_paradigms()
        assert paradigms == sorted(paradigms)


class TestRenderTemplate:
    def test_shoaling_basic(self):
        result = render_template(
            paradigm="shoaling",
            file_pattern="/mnt/user-data/uploads/*.txt",
            groups={"control": ["Subject 1"], "treatment": ["Subject 3"]},
        )
        assert "FILE_PATTERN = " in result
        assert "/mnt/user-data/uploads/*.txt" in result
        assert "PARADIGM = " in result
        assert "shoaling" in result
        assert "'control'" in result
        assert "'Subject 1'" in result

    def test_custom_metrics(self):
        result = render_template(
            paradigm="shoaling",
            file_pattern="/mnt/user-data/uploads/*.txt",
            groups={"control": ["Subject 1"], "treatment": ["Subject 3"]},
            metrics=["distance_moved"],
        )
        assert "METRICS_TO_COMPUTE = ['distance_moved']" in result

    def test_custom_chart_types(self):
        result = render_template(
            paradigm="shoaling",
            file_pattern="/mnt/user-data/uploads/*.txt",
            groups={"control": ["Subject 1"], "treatment": ["Subject 3"]},
            chart_types=["box_plot", "violin_plot"],
        )
        assert "CHART_TYPES = ['box_plot', 'violin_plot']" in result

    def test_output_is_valid_python(self):
        result = render_template(
            paradigm="shoaling",
            file_pattern="/mnt/user-data/uploads/*.txt",
            groups={"control": ["Subject 1", "Subject 2"], "treatment": ["Subject 3"]},
        )
        # Should parse without SyntaxError
        ast.parse(result)

    def test_preserves_customizable_markers(self):
        result = render_template(
            paradigm="shoaling",
            file_pattern="/mnt/user-data/uploads/*.txt",
            groups={"control": ["Subject 1"], "treatment": ["Subject 3"]},
        )
        assert "# CUSTOMIZABLE:" in result
        assert "# CUSTOMIZABLE: add extra charts" in result

    def test_preserves_fixed_workflow_markers(self):
        result = render_template(
            paradigm="shoaling",
            file_pattern="/mnt/user-data/uploads/*.txt",
            groups={"control": ["Subject 1"], "treatment": ["Subject 3"]},
        )
        assert "fixed workflow, do not modify" in result

    def test_unknown_paradigm_raises(self):
        with pytest.raises(FileNotFoundError, match="No template for paradigm"):
            render_template(
                paradigm="nonexistent_paradigm",
                file_pattern="/mnt/user-data/uploads/*.txt",
                groups={"control": ["Subject 1"]},
            )

    def test_custom_output_dir(self):
        result = render_template(
            paradigm="shoaling",
            file_pattern="/mnt/user-data/uploads/*.txt",
            groups={"control": ["Subject 1"], "treatment": ["Subject 3"]},
            output_dir="/custom/output/",
        )
        assert "OUTPUT_DIR = " in result
        assert "/custom/output/" in result

    def test_custom_handoff_path(self):
        result = render_template(
            paradigm="shoaling",
            file_pattern="/mnt/user-data/uploads/*.txt",
            groups={"control": ["Subject 1"], "treatment": ["Subject 3"]},
            handoff_path="/custom/handoff.json",
        )
        assert "HANDOFF_PATH = " in result
        assert "/custom/handoff.json" in result


class TestGetAnalysisTemplateTool:
    """Test the LangChain tool wrapper."""

    def test_tool_name(self):
        assert get_analysis_template_tool.name == "get_analysis_template"

    def test_shoaling_via_tool(self):
        groups_json = json.dumps({"control": ["Subject 1"], "treatment": ["Subject 3"]})
        result = get_analysis_template_tool.invoke({
            "paradigm": "shoaling",
            "file_pattern": "/mnt/user-data/uploads/*.txt",
            "groups": groups_json,
        })
        assert "parse.parse_batch" in result
        assert "PARADIGM" in result

    def test_invalid_groups_json(self):
        result = get_analysis_template_tool.invoke({
            "paradigm": "shoaling",
            "file_pattern": "/mnt/user-data/uploads/*.txt",
            "groups": "not valid json{",
        })
        assert "Error: invalid groups JSON" in result

    def test_unknown_paradigm_via_tool(self):
        groups_json = json.dumps({"control": ["Subject 1"]})
        result = get_analysis_template_tool.invoke({
            "paradigm": "nonexistent",
            "file_pattern": "/mnt/user-data/uploads/*.txt",
            "groups": groups_json,
        })
        assert "No template for paradigm" in result
        assert "shoaling" in result  # lists available paradigms

    def test_metrics_parameter(self):
        groups_json = json.dumps({"control": ["Subject 1"], "treatment": ["Subject 3"]})
        result = get_analysis_template_tool.invoke({
            "paradigm": "shoaling",
            "file_pattern": "/mnt/user-data/uploads/*.txt",
            "groups": groups_json,
            "metrics": "distance_moved,mean_iid",
        })
        assert "METRICS_TO_COMPUTE = ['distance_moved', 'mean_iid']" in result

    def test_chart_types_parameter(self):
        groups_json = json.dumps({"control": ["Subject 1"], "treatment": ["Subject 3"]})
        result = get_analysis_template_tool.invoke({
            "paradigm": "shoaling",
            "file_pattern": "/mnt/user-data/uploads/*.txt",
            "groups": groups_json,
            "chart_types": "box_plot,violin_plot",
        })
        assert "CHART_TYPES = ['box_plot', 'violin_plot']" in result
