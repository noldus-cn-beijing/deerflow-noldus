"""W3: _common.yaml 加载 + 校验。"""
from __future__ import annotations

import pytest

from ethoinsight.catalog.loader import CatalogError, load_common_catalog


def test_load_common_catalog_returns_two_charts():
    cc = load_common_catalog()
    chart_ids = [c.id for c in cc.common_charts]
    assert "trajectory_plot" in chart_ids
    assert "timeseries_plot" in chart_ids


def test_common_charts_have_when_field():
    cc = load_common_catalog()
    for c in cc.common_charts:
        assert c.when, f"common chart '{c.id}' missing 'when' field"
        assert "total_subjects" in c.when or c.when == "always", (
            f"common chart '{c.id}' should use total_subjects-based when "
            f"(got '{c.when}')"
        )


def test_common_chart_script_path_format():
    cc = load_common_catalog()
    for c in cc.common_charts:
        assert c.script.startswith("ethoinsight.scripts._common."), (
            f"chart '{c.id}' script '{c.script}' must be under ethoinsight.scripts._common"
        )


def test_load_common_catalog_handles_missing_file(tmp_path):
    """Custom catalog_dir 指向无 _common.yaml 的目录 → CatalogError。"""
    with pytest.raises(CatalogError, match="_common.yaml"):
        load_common_catalog(catalog_dir=tmp_path)
