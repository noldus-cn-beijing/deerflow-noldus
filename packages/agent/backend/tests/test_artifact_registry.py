from app.gateway.artifact_registry import KIND_BY_EXT, REGISTRY, kind_for_ext


def test_kind_for_ext_maps_known_extensions():
    assert kind_for_ext(".png") == "chart"
    assert kind_for_ext(".md") == "report"
    assert kind_for_ext(".html") == "report"
    assert kind_for_ext(".htm") == "report"
    assert kind_for_ext(".csv") == "data"
    assert kind_for_ext(".json") == "data"


def test_kind_for_ext_is_case_insensitive_and_dot_optional():
    assert kind_for_ext(".PNG") == "chart"
    assert kind_for_ext("png") == "chart"


def test_kind_for_ext_unknown_returns_none():
    assert kind_for_ext(".xyz") is None
    assert kind_for_ext("") is None


def test_registry_maps_each_kind_to_builder_name():
    assert REGISTRY == {
        "chart": "list_chart_artifacts",
        "report": "list_report_artifacts",
        "data": "list_data_artifacts",
    }


def test_kind_by_ext_covers_registry_kinds():
    # Every kind reachable by ext must have a builder in REGISTRY.
    assert set(KIND_BY_EXT.values()) <= set(REGISTRY.keys())
