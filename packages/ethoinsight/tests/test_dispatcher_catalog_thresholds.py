"""Sprint 2a: dispatcher catalog threshold lookup — unit tests.

Verifies dispatcher reads thresholds from catalog YAMLs
with proper fallback when catalog is unavailable.
"""

from __future__ import annotations

import importlib
from unittest.mock import patch

import pytest

from ethoinsight.catalog.loader import CatalogError
from ethoinsight.metrics import dispatcher


@pytest.fixture(autouse=True)
def _clear_cache():
    """Clear lru_cache between tests to avoid stale values."""
    dispatcher._get_shared_param.cache_clear()
    dispatcher._get_paradigm_param.cache_clear()
    yield
    dispatcher._get_shared_param.cache_clear()
    dispatcher._get_paradigm_param.cache_clear()


class TestDispatcherCatalogThresholds:
    def test_reads_zm_distance_threshold_from_catalog(self):
        """Zero maze distance threshold read from catalog (default 10.0)."""
        val = dispatcher._get_paradigm_param("zero_maze", "zm_low_distance_threshold", 99.0)
        assert val == 10.0

    def test_reads_epm_motor_threshold_from_catalog(self):
        """EPM motor threshold read from catalog (default 8)."""
        val = dispatcher._get_paradigm_param("epm", "motor_low_entries_threshold", 99)
        assert val == 8

    def test_reads_shared_sample_size_from_catalog(self):
        """Sample size underpowered threshold from shared params (default 5)."""
        val = dispatcher._get_shared_param("sample_size_underpowered_threshold", 99)
        assert val == 5

    def test_reads_ldb_transition_from_catalog(self):
        """LDB transition threshold from catalog (default 4)."""
        val = dispatcher._get_paradigm_param("light_dark_box", "signal_low_transition_threshold", 99)
        assert val == 4

    def test_fallback_when_catalog_missing_param(self):
        """Unknown param name → returns hardcoded fallback."""
        val = dispatcher._get_paradigm_param("epm", "nonexistent_param", 42)
        assert val == 42

    def test_fallback_when_catalog_load_fails(self):
        """load_catalog raises → returns fallback without crashing."""
        with patch("ethoinsight.catalog.loader.load_catalog", side_effect=CatalogError("fail")):
            # Clear cache so it re-evaluates with the mock
            dispatcher._get_paradigm_param.cache_clear()
            val = dispatcher._get_paradigm_param("epm", "motor_low_entries_threshold", 8)
            assert val == 8

    def test_fallback_when_shared_catalog_load_fails(self):
        """load_common_catalog raises → returns fallback without crashing."""
        with patch("ethoinsight.catalog.loader.load_common_catalog", side_effect=CatalogError("fail")):
            dispatcher._get_shared_param.cache_clear()
            val = dispatcher._get_shared_param("sample_size_underpowered_threshold", 5)
            assert val == 5
