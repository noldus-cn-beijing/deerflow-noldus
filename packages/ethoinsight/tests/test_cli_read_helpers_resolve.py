"""Spec 2026-06-16 缺陷 1a + 1b — read_inputs_json / read_groups_json I/O 边界对称。

缺陷 1a（red 锚点）：``read_inputs_json`` / ``read_groups_json`` 的 ``path`` 参数此前
是 ``Path(path)`` 直接读（spec #2 当时只修了 ``parse_trajectory`` + ``save_output_json``，
漏了这两个读函数）。statistics 链读 workspace 下的 inputs.json / groups.json（/mnt 虚拟
路径）→ ``FileNotFoundError``，致 ``statistics: {}``。

缺陷 1b（red 锚点）：``read_groups_json`` docstring 认为文件就是
``{group: [files]}``，但 SSOT（``prep_metric_plan`` 写、``metric_aggregation`` 主读）是
``{file: group}``（flat map）。旧透传 ``{k: [resolve(s) for s in v]}`` 会把字符串组名
当可迭代拆字符（``"control" → ["c","o","n","t","r","o","l"]``）或抛 TypeError。

修复（1a）：``path`` 也包 ``resolve_sandbox_path``。修复（1b）：函数内反转 SSOT flat map
成 ``{group: [files]}``（派生视图，非双存），同时兼容遗留直通形态。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ethoinsight.scripts._cli import read_groups_json, read_inputs_json


# /mnt/user-data/workspace → DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE


def _set_workspace_env(monkeypatch, real_ws: Path) -> None:
    monkeypatch.setenv("DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE", str(real_ws))


# ---------------------------------------------------------------------------
# 缺陷 1a — read 函数 resolve 自身 path 参数
# ---------------------------------------------------------------------------


class TestReadHelpersResolvePath:
    def test_read_inputs_json_resolves_mnt_path(self, tmp_path, monkeypatch):
        """path 参数 /mnt 虚拟路径能读到真实 inputs.json，返回条目也已 resolve。

        red 锚点：修复前 ``FileNotFoundError: '/mnt/user-data/workspace/inputs.json'``。
        """
        real_ws = tmp_path / "real_workspace"
        real_ws.mkdir()
        real_subj = tmp_path / "real_subject.txt"
        real_subj.write_text("x")
        # inputs.json 内列 /mnt 虚拟路径（与 path 同一 workspace 前缀）
        inputs_payload = ["/mnt/user-data/uploads/subject1.txt"]
        (real_ws / "inputs.json").write_text(json.dumps(inputs_payload), encoding="utf-8")
        # uploads 也设 env，让条目 resolve
        monkeypatch.setenv("DEERFLOW_PATH_MNT_USER_DATA_UPLOADS", str(tmp_path))

        _set_workspace_env(monkeypatch, real_ws)
        result = read_inputs_json("/mnt/user-data/workspace/inputs.json")
        assert result == [str(tmp_path / "subject1.txt")]

    def test_read_groups_json_resolves_mnt_path(self, tmp_path, monkeypatch):
        """path 参数 /mnt 虚拟路径能读到真实 groups.json。

        red 锚点：修复前 ``FileNotFoundError: '/mnt/user-data/workspace/groups.json'``。
        """
        real_ws = tmp_path / "real_workspace"
        real_ws.mkdir()
        groups_payload = {"/mnt/user-data/uploads/Trial 1.xlsx": "control"}
        (real_ws / "groups.json").write_text(json.dumps(groups_payload), encoding="utf-8")
        monkeypatch.setenv("DEERFLOW_PATH_MNT_USER_DATA_UPLOADS", str(tmp_path))

        _set_workspace_env(monkeypatch, real_ws)
        result = read_groups_json("/mnt/user-data/workspace/groups.json")
        # SSOT flat map 反转后 + 条目 resolve
        assert "control" in result
        assert result["control"] == [str(tmp_path / "Trial 1.xlsx")]

    def test_read_helpers_idempotent_on_real_path(self, tmp_path):
        """守护 fail-safe：喂真实路径（无 /mnt）零变化，对现有调用方无影响。"""
        real_a = tmp_path / "a.txt"
        real_a.write_text("x")
        inputs = tmp_path / "inputs.json"
        inputs.write_text(json.dumps([str(real_a)]), encoding="utf-8")
        result = read_inputs_json(inputs)
        assert result == [str(real_a)]

        groups = tmp_path / "groups.json"
        groups.write_text(json.dumps({str(real_a): "control"}), encoding="utf-8")
        g = read_groups_json(groups)
        # SSOT flat map 反转：{file: group} → {group: [file]}；真实路径原样返回。
        assert g == {"control": [str(real_a)]}


# ---------------------------------------------------------------------------
# 缺陷 1b — groups 格式反转（SSOT {file:group} → {group:[files]}）
# ---------------------------------------------------------------------------


class TestReadGroupsInvertsFormat:
    def test_inverts_flat_map(self, tmp_path, monkeypatch):
        """SSOT {file: group} 读回反转成 {group: [resolved file]}。

        red 锚点：修复前要么 TypeError（迭代字符串），要么产出
        ``{"<file>": ["c","o","n","t","r","o","l"]}``（拆组名字符）。
        """
        real_uploads = tmp_path / "real_uploads"
        real_uploads.mkdir()
        groups_payload = {
            "/mnt/user-data/uploads/Trial 1.xlsx": "control",
            "/mnt/user-data/uploads/Trial 8.xlsx": "treatment",
        }
        groups_file = tmp_path / "groups.json"
        groups_file.write_text(json.dumps(groups_payload), encoding="utf-8")
        monkeypatch.setenv("DEERFLOW_PATH_MNT_USER_DATA_UPLOADS", str(real_uploads))

        result = read_groups_json(groups_file)
        assert result == {
            "control": [str(real_uploads / "Trial 1.xlsx")],
            "treatment": [str(real_uploads / "Trial 8.xlsx")],
        }

    def test_flat_map_multiple_subjects_same_group(self, tmp_path):
        """同组多个 subject 反转后聚到同一 key 下。"""
        groups_payload = {
            "/real/a.xlsx": "control",
            "/real/b.xlsx": "control",
            "/real/c.xlsx": "treatment",
        }
        groups_file = tmp_path / "groups.json"
        groups_file.write_text(json.dumps(groups_payload), encoding="utf-8")
        result = read_groups_json(groups_file)
        assert set(result.keys()) == {"control", "treatment"}
        assert result["control"] == ["/real/a.xlsx", "/real/b.xlsx"]
        assert result["treatment"] == ["/real/c.xlsx"]

    def test_passes_through_legacy_group_to_list(self, tmp_path):
        """遗留形态 {group: [files]} 直通 + resolve（兼容性守护）。"""
        groups_payload = {"control": ["/real/f1.xlsx", "/real/f2.xlsx"]}
        groups_file = tmp_path / "groups.json"
        groups_file.write_text(json.dumps(groups_payload), encoding="utf-8")
        result = read_groups_json(groups_file)
        assert result == {"control": ["/real/f1.xlsx", "/real/f2.xlsx"]}

    def test_empty_dict(self, tmp_path):
        """空 dict 不炸，返回空。"""
        groups_file = tmp_path / "groups.json"
        groups_file.write_text("{}", encoding="utf-8")
        assert read_groups_json(groups_file) == {}

    def test_rejects_non_object(self, tmp_path):
        """非 dict 输入明确报错（ValueError）。"""
        groups_file = tmp_path / "groups.json"
        groups_file.write_text(json.dumps(["not", "an", "object"]), encoding="utf-8")
        with pytest.raises(ValueError, match="must be a JSON object"):
            read_groups_json(groups_file)
