"""Unit tests for ThreadState reducers.

Regression coverage for issue #3123: todos list disappearing after streaming
completes because a downstream node's partial state update with `todos=None`
overwrites the previously accumulated value.
"""

from typing import get_type_hints

from deerflow.agents.thread_state import (
    ThreadState,
    merge_artifacts,
    merge_charts_status,
    merge_todos,
    merge_viewed_images,
)


class TestMergeTodos:
    """Reducer for ThreadState.todos - keeps last non-None value."""

    def test_new_value_overrides_existing(self):
        existing = [{"id": 1, "text": "old", "done": False}]
        new = [{"id": 1, "text": "old", "done": True}]
        assert merge_todos(existing, new) == new

    def test_none_new_preserves_existing(self):
        """THE KEY FIX for #3123: a node that doesn't touch todos must NOT
        wipe them out by returning an implicit None."""
        existing = [{"id": 1, "text": "task", "done": False}]
        assert merge_todos(existing, None) == existing

    def test_none_existing_accepts_new(self):
        new = [{"id": 1, "text": "first todo"}]
        assert merge_todos(None, new) == new

    def test_both_none_returns_none(self):
        assert merge_todos(None, None) is None

    def test_empty_list_is_explicit_clear(self):
        """An explicit empty list means 'user cleared all todos' and must
        win over the previous list."""
        existing = [{"id": 1, "text": "task"}]
        assert merge_todos(existing, []) == []


class TestMergeArtifacts:
    """Sanity check for the existing artifacts reducer."""

    def test_dedupes_and_preserves_order(self):
        assert merge_artifacts(["a", "b"], ["b", "c"]) == ["a", "b", "c"]

    def test_none_new_preserves_existing(self):
        assert merge_artifacts(["a"], None) == ["a"]

    def test_none_existing_accepts_new(self):
        assert merge_artifacts(None, ["a"]) == ["a"]


class TestMergeArtifactsArtifactMeta:
    """spec 2026-06-24-frontend-phase0-3-artifact-gallery：artifacts 升级成
    ArtifactMeta[]（向后兼容裸 string）。merge 必须按 path 去重，新值覆盖旧值。"""

    def test_dedup_by_path_mixed_string_and_meta(self):
        """混合 string + meta + 重复 path → 按 path 去重，不崩。"""
        existing = [
            "/mnt/user-data/outputs/a.png",  # 裸 string（旧数据）
            {"path": "/mnt/user-data/outputs/b.png", "kind": "chart", "chart_id": "b"},
        ]
        new = [
            {"path": "/mnt/user-data/outputs/a.png", "kind": "chart", "chart_id": "a", "output_mode": "aggregate"},
            "/mnt/user-data/outputs/c.md",  # 报告（裸 string）
        ]
        merged = merge_artifacts(existing, new)
        # 按 path 去重：a/b/c 三条；a 的新值（meta 带 chart_id）覆盖旧的裸 string。
        by_path = {m["path"] if isinstance(m, dict) else m: m for m in merged}
        assert set(by_path.keys()) == {
            "/mnt/user-data/outputs/a.png",
            "/mnt/user-data/outputs/b.png",
            "/mnt/user-data/outputs/c.md",
        }
        # a 的新 meta 覆盖了旧裸 string（追问轮补全元数据）。
        assert by_path["/mnt/user-data/outputs/a.png"]["chart_id"] == "a"
        assert by_path["/mnt/user-data/outputs/a.png"]["output_mode"] == "aggregate"
        # c 仍是裸 string（无 plan 命中的报告）。
        assert by_path["/mnt/user-data/outputs/c.md"] == "/mnt/user-data/outputs/c.md"

    def test_new_overrides_existing_same_path(self):
        """同 path 的新 meta 覆盖旧 meta（元数据补全场景）。"""
        existing = [{"path": "/x.png", "kind": "chart", "output_mode": "per_subject"}]
        new = [{"path": "/x.png", "kind": "chart", "output_mode": "aggregate", "chart_id": "x"}]
        merged = merge_artifacts(existing, new)
        assert len(merged) == 1
        assert merged[0]["output_mode"] == "aggregate"
        assert merged[0]["chart_id"] == "x"

    def test_preserves_distinct_subjects(self):
        """per_subject 多张图 path 不同 → 各自保留（不被 chart_id 互吞）。"""
        metas = [
            {"path": f"/o_{i}.png", "kind": "chart", "chart_id": "o", "subject": str(i)}
            for i in range(5)
        ]
        merged = merge_artifacts(None, metas)
        assert len(merged) == 5

    def test_bare_strings_still_dedupe(self):
        """纯裸 string 列表仍按值去重（向后兼容锚点）。"""
        assert merge_artifacts(["a", "b"], ["b", "c"]) == ["a", "b", "c"]


class TestMergeChartsStatus:
    """charts_status reducer（spec §四 Step 5：failed/remaining 摘要进 state）。"""

    def test_none_new_preserves_existing(self):
        existing = {"n_rendered": 5, "failed": [], "remaining": []}
        assert merge_charts_status(existing, None) == existing

    def test_new_overrides_existing(self):
        existing = {"n_rendered": 5, "failed": [{"chart_id": "x"}], "remaining": []}
        new = {"n_rendered": 6, "failed": [], "remaining": []}
        assert merge_charts_status(existing, new) == new


class TestMergeViewedImages:
    """Sanity check for the existing viewed_images reducer."""

    def test_merges_dicts(self):
        existing = {"k1": {"base64": "x", "mime_type": "image/png"}}
        new = {"k2": {"base64": "y", "mime_type": "image/jpeg"}}
        merged = merge_viewed_images(existing, new)
        assert set(merged.keys()) == {"k1", "k2"}

    def test_empty_dict_clears(self):
        existing = {"k1": {"base64": "x", "mime_type": "image/png"}}
        assert merge_viewed_images(existing, {}) == {}


class TestThreadStateAnnotations:
    """Regression guards: ensure reducer wiring on ThreadState fields.

    These tests protect against silent regressions where a field's
    ``Annotated[..., reducer]`` is reverted to a plain type, which would
    re-introduce bugs even when the reducer functions themselves remain
    correct.
    """

    def test_todos_field_is_wired_to_merge_todos(self):
        """ThreadState.todos must use merge_todos.

        Without this Annotated binding, LangGraph falls back to last-value-wins
        behavior, and partial state updates that omit todos will silently clear
        previously streamed values.
        """
        hints = get_type_hints(ThreadState, include_extras=True)
        todos_hint = hints["todos"]
        assert hasattr(todos_hint, "__metadata__"), "ThreadState.todos must be Annotated with a reducer"
        assert merge_todos in todos_hint.__metadata__, "ThreadState.todos must be wired to merge_todos reducer (see #3123)"

    def test_artifacts_field_is_wired_to_merge_artifacts(self):
        """Sanity check that existing reducer wiring is preserved."""
        hints = get_type_hints(ThreadState, include_extras=True)
        assert merge_artifacts in hints["artifacts"].__metadata__

    def test_charts_status_field_is_wired_to_merge_charts_status(self):
        """spec phase0-3: charts_status must be Annotated with merge_charts_status."""
        hints = get_type_hints(ThreadState, include_extras=True)
        assert merge_charts_status in hints["charts_status"].__metadata__


class TestMergeArtifactsRunScoped:
    """spec 2026-06-26 §任务2（路 A）：artifact 按 (run_id, path) 去重。

    同一 thread 内多 run 若产同名 chart（chart_id 来自 catalog 固定、与 run 无关），
    旧 reducer 按 path 去重会让后一 run 覆盖前一 run 的产物（前端列表不变但磁盘
    内容被换）。带 run_id 维度后：同 run 同 path 覆盖（正确），跨 run 同 path 不互覆盖。
    """

    def test_cross_run_same_path_not_overwritten(self):
        """两个 run 各 present 同 path → 两条 artifact（按 run 区分）。"""
        existing = [{"path": "/o.png", "kind": "chart", "run_id": "run-A"}]
        new = [{"path": "/o.png", "kind": "chart", "run_id": "run-B"}]
        merged = merge_artifacts(existing, new)
        assert len(merged) == 2
        run_ids = {m.get("run_id") for m in merged}
        assert run_ids == {"run-A", "run-B"}

    def test_same_run_same_path_dedupes(self):
        """同 run 同 path 仍去重为一条（追问轮补全元数据，不回归）。"""
        existing = [{"path": "/o.png", "kind": "chart", "run_id": "run-A", "output_mode": "per_subject"}]
        new = [{"path": "/o.png", "kind": "chart", "run_id": "run-A", "output_mode": "aggregate", "chart_id": "o"}]
        merged = merge_artifacts(existing, new)
        assert len(merged) == 1
        # 同 run 内新值覆盖旧值（元数据补全语义不变）
        assert merged[0]["output_mode"] == "aggregate"
        assert merged[0]["chart_id"] == "o"

    def test_run_id_absent_falls_back_to_path_dedup(self):
        """裸 string / 无 run_id 的 meta 仍按 path 去重（向后兼容旧产物）。"""
        existing = ["/o.png"]  # 裸 string，无 run_id
        new = [{"path": "/o.png", "kind": "chart"}]  # dict 无 run_id
        merged = merge_artifacts(existing, new)
        assert len(merged) == 1  # 都无 run_id → 按 path 去重

    def test_distinct_paths_within_same_run_all_kept(self):
        """per_subject 多张图 path 不同 → 各自保留（与 run_id 维度正交）。"""
        metas = [
            {"path": f"/o_{i}.png", "kind": "chart", "run_id": "run-A", "subject": str(i)}
            for i in range(3)
        ]
        merged = merge_artifacts(None, metas)
        assert len(merged) == 3

