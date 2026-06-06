"""Tests for inspect_uploaded_file_tool, column_semantics projection,
and ev19_template guardrail column_semantics sub-check.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── inspect_uploaded_file column_assessment tests ──────────────────────

class TestInspectReturnsColumnAssessment:
    """inspect_uploaded_file returns column_assessment + open_questions."""

    @pytest.mark.asyncio
    async def test_inspect_returns_open_questions_for_custom_columns(
        self, tmp_path, ethovision_txt_file
    ):
        """With a real-like EthoVision file, inspect returns column assessment."""
        from deerflow.tools.builtins.inspect_uploaded_file_tool import (
            inspect_uploaded_file_tool,
        )

        # Build a minimal TXT file that parse_header can handle
        txt_path = tmp_path / "test.txt"
        # EthoVision TXT files need UTF-16 LE with BOM (\xff\xfe)
        with open(txt_path, "wb") as f:
            f.write(b"\xff\xfe")
            f.write(ethovision_txt_file.encode("utf-16-le"))

        runtime = MagicMock()
        runtime.state = {
            "thread_data": {
                "workspace_path": str(tmp_path),
            }
        }

        result = inspect_uploaded_file_tool.func(
            runtime=runtime,
            uploaded_files=[str(txt_path)],
            paradigm="open_field",
        )
        assert result["status"] == "ok"
        assert "column_assessment" in result
        assert "open_questions" in result
        assert "recognized" in result["column_assessment"]
        assert "unrecognized" in result["column_assessment"]

    @pytest.mark.asyncio
    async def test_inspect_without_paradigm(self, tmp_path, ethovision_txt_file):
        """Without paradigm, inspect still works (degraded but no false unrecognized)."""
        from deerflow.tools.builtins.inspect_uploaded_file_tool import (
            inspect_uploaded_file_tool,
        )

        txt_path = tmp_path / "test.txt"
        with open(txt_path, "wb") as f:
            f.write(b"\xff\xfe")
            f.write(ethovision_txt_file.encode("utf-16-le"))

        runtime = MagicMock()
        runtime.state = {
            "thread_data": {
                "workspace_path": str(tmp_path),
            }
        }

        result = inspect_uploaded_file_tool.func(
            runtime=runtime,
            uploaded_files=[str(txt_path)],
            # paradigm=None — should still return ok (degraded mode)
        )
        assert result["status"] == "ok"
        assert "column_assessment" in result

    @pytest.mark.asyncio
    async def test_inspect_no_files(self):
        """Empty uploaded_files returns error."""
        from deerflow.tools.builtins.inspect_uploaded_file_tool import (
            inspect_uploaded_file_tool,
        )

        runtime = MagicMock()
        result = inspect_uploaded_file_tool.func(
            runtime=runtime,
            uploaded_files=[],
        )
        assert result["status"] == "error"
        assert result["error_code"] == "no_files_provided"


# ── column_semantics projection tests ─────────────────────────────────

class TestColumnSemanticsProjection:
    """set_experiment_paradigm writes column_semantics + derives column_aliases."""

    def test_derive_column_aliases_basic(self):
        """_derive_column_aliases keys aliases by RE-NORMALIZED raw_name (+ raw verbatim).

        It does NOT trust the LLM-supplied 'normalized' field. For 中心区,
        normalize_column_name("中心区") == "中心区", so the alias key is 中心区 (not 'center').
        resolves_to holds a concept keyword which the resolve layer translates later.
        """
        from deerflow.agents.middlewares.experiment_context import (
            _derive_column_aliases,
        )

        cs = {
            "confirmed_at": "2026-06-05T00:00:00Z",
            "columns": {
                "中心区": {
                    "raw_name": "中心区",
                    "normalized": "center",  # WRONG LLM value — must be ignored by the code
                    "resolves_to": "center",  # concept keyword
                    "meaning_zh": "中心分析区",
                    "confirmed": True,
                },
                "边缘区": {
                    "raw_name": "边缘区",
                    "resolves_to": "border",
                    "meaning_zh": "边缘分析区",
                    "confirmed": True,
                },
            },
        }
        aliases = _derive_column_aliases(cs)
        # Keyed by re-normalized raw_name (中心区 stays 中心区) — NOT the bogus "center".
        assert aliases["中心区"] == "center"
        assert aliases["边缘区"] == "border"
        assert "center" not in aliases  # the wrong LLM "normalized" did not leak in as a key

    def test_derive_column_aliases_skips_unconfirmed(self):
        """Unconfirmed columns are NOT included in aliases."""
        from deerflow.agents.middlewares.experiment_context import (
            _derive_column_aliases,
        )

        cs = {
            "columns": {
                "中心区": {
                    "raw_name": "中心区",
                    "normalized": "center",
                    "resolves_to": "in_zone_center_point",
                    "confirmed": False,  # not confirmed yet
                },
            },
        }
        aliases = _derive_column_aliases(cs)
        assert aliases == {}

    def test_derive_column_aliases_skips_ignored(self):
        """Ignored columns (resolves_to=None or __ignore__) are excluded."""
        from deerflow.agents.middlewares.experiment_context import (
            _derive_column_aliases,
        )

        cs = {
            "columns": {
                "忽略列": {
                    "raw_name": "忽略列",
                    "normalized": "忽略列",
                    "resolves_to": None,
                    "ignore": True,
                    "confirmed": True,
                },
            },
        }
        aliases = _derive_column_aliases(cs)
        assert aliases == {}

    def test_derive_column_aliases_empty(self):
        """Empty columns dict produces empty aliases."""
        from deerflow.agents.middlewares.experiment_context import (
            _derive_column_aliases,
        )

        assert _derive_column_aliases({}) == {}
        assert _derive_column_aliases({"columns": {}}) == {}

    @pytest.mark.asyncio
    async def test_set_experiment_paradigm_with_column_semantics(
        self, tmp_path
    ):
        """set_experiment_paradigm with column_semantics writes to experiment-context.json."""
        from deerflow.agents.middlewares.experiment_context import (
            read_context,
            set_experiment_paradigm_tool,
        )

        runtime = MagicMock()
        runtime.state = {
            "thread_data": {
                "workspace_path": str(tmp_path),
            }
        }

        # First, set up Gate 1
        set_experiment_paradigm_tool.func(
            paradigm="open_field",
            paradigm_cn="旷场",
            category="anxiety",
            subject="rodent",
            ev19_template="OpenFieldRectangle-AllZones",
            workspace_dir=str(tmp_path),
            runtime=runtime,
        )

        # Now add column_semantics (resolves_to = concept keyword)
        cs = {
            "columns": {
                "中心区": {
                    "raw_name": "中心区",
                    "resolves_to": "center",
                    "meaning_zh": "中心分析区",
                    "confirmed": True,
                },
            },
        }
        result_raw = set_experiment_paradigm_tool.func(
            paradigm="open_field",
            paradigm_cn="旷场",
            category="anxiety",
            subject="rodent",
            ev19_template="OpenFieldRectangle-AllZones",
            column_semantics=cs,
            workspace_dir=str(tmp_path),
            runtime=runtime,
        )
        result = json.loads(result_raw)
        assert result["status"] == "ok"

        # Verify the file on disk — alias key is the re-normalized raw_name (中心区), and
        # the value is the concept keyword (translated to a column later by resolve).
        ctx = read_context(str(tmp_path))
        assert ctx is not None
        assert "column_semantics" in ctx
        assert "column_aliases" in ctx
        assert ctx["column_aliases"]["中心区"] == "center"
        assert ctx["column_semantics"]["columns"]["中心区"]["confirmed"] is True


# ── guardrail column_semantics sub-check tests ────────────────────────

class TestGuardrailBlocksOpenQuestions:
    """Ev19TemplateGuardrailProvider blocks task(code-executor) when
    column_semantics has unconfirmed columns."""

    def test_guardrail_allows_when_no_column_semantics(self, tmp_path):
        """When experiment-context.json has ev19_template but no column_semantics,
        guardrail allows task(code-executor)."""
        from deerflow.guardrails.ev19_template_provider import (
            Ev19TemplateGuardrailProvider,
        )
        from deerflow.guardrails.provider import GuardrailRequest

        # Write a valid experiment-context.json with template but no column_semantics
        ctx = {
            "paradigm": "open_field",
            "ev19_template": "OpenFieldRectangle-AllZones",
            "gate_completed": ["gate1_paradigm"],
        }
        (tmp_path / "experiment-context.json").write_text(
            json.dumps(ctx), encoding="utf-8"
        )

        provider = Ev19TemplateGuardrailProvider(
            workspace_resolver=lambda: str(tmp_path)
        )
        request = GuardrailRequest(
            tool_name="task",
            tool_input={"subagent_type": "code-executor", "description": "test"},
        )
        decision = provider.evaluate(request)
        assert decision.allow is True

    def test_guardrail_blocks_unconfirmed_columns(self, tmp_path):
        """Unconfirmed column_semantics columns block task(code-executor)."""
        from deerflow.guardrails.ev19_template_provider import (
            Ev19TemplateGuardrailProvider,
        )
        from deerflow.guardrails.provider import GuardrailRequest

        ctx = {
            "paradigm": "open_field",
            "ev19_template": "OpenFieldRectangle-AllZones",
            "gate_completed": ["gate1_paradigm"],
            "column_semantics": {
                "columns": {
                    "中心区": {
                        "raw_name": "中心区",
                        "normalized": "center",
                        "resolves_to": "in_zone_center_point",
                        "confirmed": False,  # NOT confirmed!
                    },
                },
            },
        }
        (tmp_path / "experiment-context.json").write_text(
            json.dumps(ctx), encoding="utf-8"
        )

        provider = Ev19TemplateGuardrailProvider(
            workspace_resolver=lambda: str(tmp_path)
        )
        request = GuardrailRequest(
            tool_name="task",
            tool_input={"subagent_type": "code-executor", "description": "test"},
        )
        decision = provider.evaluate(request)
        assert decision.allow is False
        assert any(
            "ethoinsight.column_semantics_unconfirmed" in r.code
            for r in decision.reasons
        )

    def test_guardrail_allows_confirmed_columns(self, tmp_path):
        """All-confirmed column_semantics allows task(code-executor)."""
        from deerflow.guardrails.ev19_template_provider import (
            Ev19TemplateGuardrailProvider,
        )
        from deerflow.guardrails.provider import GuardrailRequest

        ctx = {
            "paradigm": "open_field",
            "ev19_template": "OpenFieldRectangle-AllZones",
            "gate_completed": ["gate1_paradigm"],
            "column_semantics": {
                "columns": {
                    "中心区": {
                        "raw_name": "中心区",
                        "normalized": "center",
                        "resolves_to": "in_zone_center_point",
                        "confirmed": True,
                    },
                },
            },
        }
        (tmp_path / "experiment-context.json").write_text(
            json.dumps(ctx), encoding="utf-8"
        )

        provider = Ev19TemplateGuardrailProvider(
            workspace_resolver=lambda: str(tmp_path)
        )
        request = GuardrailRequest(
            tool_name="task",
            tool_input={"subagent_type": "code-executor", "description": "test"},
        )
        decision = provider.evaluate(request)
        assert decision.allow is True

    def test_guardrail_does_not_block_non_code_executor(self, tmp_path):
        """Guardrail only blocks code-executor, not other subagents."""
        from deerflow.guardrails.ev19_template_provider import (
            Ev19TemplateGuardrailProvider,
        )
        from deerflow.guardrails.provider import GuardrailRequest

        ctx = {
            "paradigm": "open_field",
            "ev19_template": "OpenFieldRectangle-AllZones",
            "column_semantics": {
                "columns": {
                    "中心区": {"confirmed": False},
                },
            },
        }
        (tmp_path / "experiment-context.json").write_text(
            json.dumps(ctx), encoding="utf-8"
        )

        provider = Ev19TemplateGuardrailProvider(
            workspace_resolver=lambda: str(tmp_path)
        )
        # task(data-analyst) should NOT be blocked
        request = GuardrailRequest(
            tool_name="task",
            tool_input={"subagent_type": "data-analyst", "description": "test"},
        )
        decision = provider.evaluate(request)
        assert decision.allow is True

    @pytest.mark.asyncio
    async def test_guardrail_async_equivalent(self, tmp_path):
        """aevaluate gives same result as evaluate (LangChain async lesson)."""
        from deerflow.guardrails.ev19_template_provider import (
            Ev19TemplateGuardrailProvider,
        )
        from deerflow.guardrails.provider import GuardrailRequest

        ctx = {
            "paradigm": "open_field",
            "ev19_template": "OpenFieldRectangle-AllZones",
            "column_semantics": {
                "columns": {
                    "中心区": {"confirmed": False},
                },
            },
        }
        (tmp_path / "experiment-context.json").write_text(
            json.dumps(ctx), encoding="utf-8"
        )

        provider = Ev19TemplateGuardrailProvider(
            workspace_resolver=lambda: str(tmp_path)
        )
        request = GuardrailRequest(
            tool_name="task",
            tool_input={"subagent_type": "code-executor", "description": "test"},
        )
        sync_decision = provider.evaluate(request)
        async_decision = await provider.aevaluate(request)
        assert sync_decision.allow == async_decision.allow


class TestPrepMetricPlanReadsColumnAliases:
    """END-TO-END seam: prep_metric_plan must read column_aliases from
    experiment-context.json and pass them to resolve. This is the integration
    point that was severed in the original Sprint 1 implementation (CRITICAL-1):
    column_aliases were written but never read, so the feature was dead on arrival.

    Uses a REAL EthoVision TXT with custom zone columns (中心区/边缘区) — the exact
    shape of the 34-file OFT upload that originally failed.
    """

    # Custom-zone TXT: 7 L1 columns + custom 中心区/边缘区 (no standard in_zone_* columns).
    _CUSTOM_ZONE_TXT = "\r\n".join(
        [
            '"标题行数：";"6"',
            '"实验";"OFT-Xuhui"',
            '"对象名称";"Subject1"',
            '"开始时间";"2024-01-01T00:00:00"',
            '"Trial time";"Recording time";"X center";"Y center";"Distance moved";"Velocity";"中心区";"边缘区"',
            '"s";"s";"cm";"cm";"cm";"cm/s";"";""',
            "0.00;0.00;100.0;200.0;0.0;0.0;1;0",
            "0.04;0.04;101.0;201.0;1.5;37.5;1;0",
            "0.08;0.08;102.0;202.0;1.5;37.5;0;1",
        ]
    )

    def _setup_workspace(self, tmp_path, column_aliases):
        """Create uploads + workspace dirs, write the TXT and experiment-context.json."""
        uploads = tmp_path / "uploads"
        workspace = tmp_path / "workspace"
        uploads.mkdir()
        workspace.mkdir()
        txt_path = uploads / "oft.txt"
        # EthoVision TXT must be UTF-16 LE *with BOM* (detect_ethovision requires \xff\xfe).
        txt_path.write_bytes(b"\xff\xfe" + self._CUSTOM_ZONE_TXT.encode("utf-16-le"))

        ctx = {
            "paradigm": "open_field",
            "ev19_template": "OpenFieldRectangle-AllZones",
            "gate_completed": ["gate1_paradigm"],
        }
        if column_aliases is not None:
            ctx["column_aliases"] = column_aliases
        (workspace / "experiment-context.json").write_text(
            json.dumps(ctx, ensure_ascii=False), encoding="utf-8"
        )

        runtime = MagicMock()
        runtime.state = {
            "thread_data": {
                "workspace_path": str(workspace),
                "uploads_path": str(uploads),
                "outputs_path": str(tmp_path / "outputs"),
            }
        }
        return runtime, workspace

    def test_without_aliases_resolve_fails_on_custom_zone(self, tmp_path):
        """No column_aliases → custom zone columns unrecognised → columns_missing."""
        from deerflow.tools.builtins.prep_metric_plan_tool import (
            prep_metric_plan_tool,
        )

        runtime, _ = self._setup_workspace(tmp_path, column_aliases=None)
        result = prep_metric_plan_tool.func(
            uploaded_files=["/mnt/user-data/uploads/oft.txt"],
            paradigm="open_field",
            runtime=runtime,
        )
        assert result["status"] == "error"
        assert result["error_code"] == "columns_missing"

    def test_with_concept_aliases_resolve_succeeds(self, tmp_path):
        """column_aliases (concept keyword) IS read and passed → center metrics resolve.

        This is the regression guard for CRITICAL-1: if prep_metric_plan stops reading
        column_aliases, this test fails (back to columns_missing).
        """
        from deerflow.tools.builtins.prep_metric_plan_tool import (
            prep_metric_plan_tool,
        )

        runtime, workspace = self._setup_workspace(
            tmp_path,
            column_aliases={"中心区": "center", "边缘区": "border"},
        )
        result = prep_metric_plan_tool.func(
            uploaded_files=["/mnt/user-data/uploads/oft.txt"],
            paradigm="open_field",
            runtime=runtime,
        )
        assert result["status"] == "ok", result
        metric_ids = set(result["plan_summary"]["metric_ids"])
        assert "center_time_ratio" in metric_ids
        # plan_metrics.json must have been written for the downstream code-executor.
        assert (workspace / "plan_metrics.json").exists()


# ── fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def ethovision_txt_file():
    """Minimal EthoVision TXT content in the semicolon-delimited quoted format
    that parse_header expects. The caller must write it with UTF-16 LE encoding.

    parse_header expects:
      Line 0: "标题行数：","<N>"  (quoted, semicolon-separated)
      Lines 1..N-3: metadata key-value pairs (quoted, semicolon-separated)
      Line N-2: column names
      Line N-1: units
      Lines N..: data rows
    """
    lines = [
        '"标题行数：";"6"',
        '"实验";"旷场"',
        '"对象名称";"Subject1"',
        '"开始时间";"2024-01-01T00:00:00"',
        '"Trial time";"Recording time";"X center";"Y center";"Distance moved";"Velocity";"in_zone_center_point";"in_zone_border_point";"中心区";"边缘区"',
        '"s";"s";"cm";"cm";"cm";"cm/s";"";"";"";""',
        "0.00;0.00;100.0;200.0;0.0;0.0;0;1;0;1",
        "0.04;0.04;101.0;201.0;1.5;37.5;0;1;0;1",
    ]
    return "\r\n".join(lines)
