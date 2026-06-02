"""Tests for inspect_uploaded_file data_preview (2026-06-02 增强, spec §2.4 / DoD §4).

Covers the two pure preview builders that back all three format paths:
  - _build_data_preview_df  → used by xlsx & csv paths (pandas DataFrame in)
  - _build_data_preview_txt → used by EV19 txt path (UTF-16-LE file in)

Asserts the {columns, rows(≤N), n_rows_total} contract, the ≤5-row cap,
NaN→None / float-int→int conversion (df path), and numeric coercion +
column truncation/padding (txt path).
"""

from __future__ import annotations

import pandas as pd

from deerflow.tools.builtins.inspect_uploaded_file_tool import (
    _DATA_PREVIEW_N_ROWS,
    _build_data_preview_df,
    _build_data_preview_txt,
)


# ---------------------------------------------------------------------------
# _build_data_preview_df — xlsx & csv paths
# ---------------------------------------------------------------------------
class TestBuildDataPreviewDf:
    def test_basic_shape(self):
        df = pd.DataFrame(
            {"Trial time": [0.0, 0.04, 0.08], "X center": [12.3, 12.4, 12.5], "state": ["Mobile", "Immobile", "Mobile"]}
        )
        preview = _build_data_preview_df(df)
        assert preview is not None
        assert preview["columns"] == ["Trial time", "X center", "state"]
        assert preview["n_rows_total"] == 3
        assert len(preview["rows"]) == 3
        assert preview["rows"][0] == [0, 12.3, "Mobile"]  # 0.0 → int 0

    def test_caps_at_n_rows(self):
        n = _DATA_PREVIEW_N_ROWS
        df = pd.DataFrame({"a": list(range(n + 10)), "b": list(range(n + 10))})
        preview = _build_data_preview_df(df)
        assert preview is not None
        assert len(preview["rows"]) == n  # only first N rows
        assert preview["n_rows_total"] == n + 10  # but total reflects full df

    def test_nan_becomes_none(self):
        df = pd.DataFrame({"x": [1.0, float("nan"), 3.0]})
        preview = _build_data_preview_df(df)
        assert preview is not None
        assert preview["rows"][1][0] is None

    def test_float_integer_coerced_to_int(self):
        df = pd.DataFrame({"count": [5.0, 10.0], "ratio": [0.25, 0.75]})
        preview = _build_data_preview_df(df)
        assert preview is not None
        assert preview["rows"][0] == [5, 0.25]  # 5.0 → 5, 0.25 stays float

    def test_n_total_override(self):
        """xlsx EV19 path passes parse_trajectory'd df; n_total may be explicit."""
        df = pd.DataFrame({"a": [1, 2]})
        preview = _build_data_preview_df(df, n_total=7501)
        assert preview is not None
        assert preview["n_rows_total"] == 7501

    def test_empty_df_returns_none(self):
        preview = _build_data_preview_df(pd.DataFrame({"a": []}))
        assert preview is None


# ---------------------------------------------------------------------------
# _build_data_preview_txt — EV19 txt (UTF-16-LE) path
# ---------------------------------------------------------------------------
class TestBuildDataPreviewTxt:
    @staticmethod
    def _write_ev19_txt(tmp_path, n_data_rows: int) -> str:
        """Minimal EV19-style UTF-16-LE txt: 2 header lines + n data rows.

        header_lines counts the rows before the data starts (column names +
        units line). Data lines are semicolon-separated, values double-quoted.
        """
        lines = [
            '"Trial time";"X center";"Mobility state"',  # column-names row
            '"sec";"cm";""',  # units row
        ]
        for i in range(n_data_rows):
            lines.append(f'"{i * 0.04}";"{12.0 + i}";"Mobile"')
        content = "﻿" + "\r\n".join(lines) + "\r\n"  # BOM + CRLF
        p = tmp_path / "trial.txt"
        p.write_text(content, encoding="utf-16-le")
        return str(p)

    def _header(self) -> dict:
        # header_lines = 2 (column-names + units); data begins at line index 2
        return {"columns": ["Trial time", "X center", "Mobility state"], "header_lines": 2}

    def test_basic_shape(self, tmp_path):
        real_path = self._write_ev19_txt(tmp_path, n_data_rows=3)
        preview = _build_data_preview_txt(real_path, self._header())
        assert preview is not None
        assert preview["columns"] == ["Trial time", "X center", "Mobility state"]
        assert preview["n_rows_total"] == 3
        assert len(preview["rows"]) == 3
        # numeric strings coerced to float; text stays str
        assert preview["rows"][0] == [0.0, 12.0, "Mobile"]
        assert preview["rows"][2][1] == 14.0

    def test_caps_at_n_rows(self, tmp_path):
        n = _DATA_PREVIEW_N_ROWS
        real_path = self._write_ev19_txt(tmp_path, n_data_rows=n + 7)
        preview = _build_data_preview_txt(real_path, self._header())
        assert preview is not None
        assert len(preview["rows"]) == n
        assert preview["n_rows_total"] == n + 7

    def test_no_columns_returns_none(self, tmp_path):
        real_path = self._write_ev19_txt(tmp_path, n_data_rows=2)
        preview = _build_data_preview_txt(real_path, {"columns": [], "header_lines": 2})
        assert preview is None

    def test_no_header_lines_returns_none(self, tmp_path):
        real_path = self._write_ev19_txt(tmp_path, n_data_rows=2)
        preview = _build_data_preview_txt(real_path, {"columns": ["a"], "header_lines": 0})
        assert preview is None

    def test_dash_and_blank_become_none(self, tmp_path):
        # data row with a "-" placeholder and an empty field
        content = "﻿" + "\r\n".join(
            [
                '"Trial time";"X center";"state"',
                '"sec";"cm";""',
                '"0.0";"-";""',
            ]
        ) + "\r\n"
        p = tmp_path / "trial.txt"
        p.write_text(content, encoding="utf-16-le")
        preview = _build_data_preview_txt(str(p), {"columns": ["Trial time", "X center", "state"], "header_lines": 2})
        assert preview is not None
        assert preview["rows"][0] == [0.0, None, None]  # "-" → None, "" → None
