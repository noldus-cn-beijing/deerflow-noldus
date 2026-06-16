"""Subprocess-level tests for ethoinsight.scripts.epm.*."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _run_script(module: str, args: list[str]) -> subprocess.CompletedProcess:
    """Run `python -m <module> <args>` and return CompletedProcess."""
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        capture_output=True,
        text=True,
        check=False,
    )


class TestComputeOpenArmTimeRatio:
    def test_happy_path_writes_json_and_emits_result(
        self, epm_trajectory_file: Path, tmp_path: Path
    ):
        out_path = tmp_path / "metric.json"
        result = _run_script(
            "ethoinsight.scripts.epm.compute_open_arm_time_ratio",
            ["--input", str(epm_trajectory_file), "--output", str(out_path)],
        )

        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert out_path.exists()

        payload = json.loads(out_path.read_text())
        assert payload["metric"] == "open_arm_time_ratio"
        assert isinstance(payload["value"], float)
        assert 0.0 <= payload["value"] <= 1.0

        # stdout must contain [result] marker
        assert "[result]" in result.stdout
        result_line = next(
            l for l in result.stdout.splitlines() if l.startswith("[result]")
        )
        result_payload = json.loads(result_line[len("[result] ") :])
        assert result_payload["metric"] == "open_arm_time_ratio"

    def test_missing_input_arg_exits_nonzero(self, tmp_path: Path):
        result = _run_script(
            "ethoinsight.scripts.epm.compute_open_arm_time_ratio",
            ["--output", str(tmp_path / "x.json")],
        )
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "input" in result.stderr.lower()

    def test_file_without_open_arm_columns_returns_value_none(
        self, tmp_path: Path, make_epm_df
    ):
        from tests.scripts.conftest import _df_to_ethovision_file

        # Build a df without any open-arm zone columns
        df = make_epm_df(open_arm_cols=["in_zone_closed_arm_1"])  # only closed arm
        df = df.drop(columns=["in_zone_closed_arm_1"])  # remove all zone cols
        df["x_center"] = df["x_center"]  # keep position cols
        path = tmp_path / "no_open_arm.txt"
        _df_to_ethovision_file(df, path)

        out_path = tmp_path / "metric.json"
        result = _run_script(
            "ethoinsight.scripts.epm.compute_open_arm_time_ratio",
            ["--input", str(path), "--output", str(out_path)],
        )

        assert result.returncode == 0
        payload = json.loads(out_path.read_text())
        assert payload["metric"] == "open_arm_time_ratio"
        assert payload["value"] is None


class TestComputeOpenArmEntryCount:
    def test_happy_path(self, epm_trajectory_file: Path, tmp_path: Path):
        out_path = tmp_path / "metric.json"
        result = _run_script(
            "ethoinsight.scripts.epm.compute_open_arm_entry_count",
            ["--input", str(epm_trajectory_file), "--output", str(out_path)],
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        payload = json.loads(out_path.read_text())
        assert payload["metric"] == "open_arm_entry_count"
        assert isinstance(payload["value"], int)
        assert payload["value"] >= 0


class TestComputeOpenArmEntryRatio:
    def test_happy_path(self, epm_trajectory_file: Path, tmp_path: Path):
        out_path = tmp_path / "metric.json"
        result = _run_script(
            "ethoinsight.scripts.epm.compute_open_arm_entry_ratio",
            ["--input", str(epm_trajectory_file), "--output", str(out_path)],
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        payload = json.loads(out_path.read_text())
        assert payload["metric"] == "open_arm_entry_ratio"
        # ratio is None or float in [0, 1]
        assert payload["value"] is None or (
            isinstance(payload["value"], float) and 0.0 <= payload["value"] <= 1.0
        )


class TestComputeOpenArmTime:
    def test_happy_path(self, epm_trajectory_file: Path, tmp_path: Path):
        out_path = tmp_path / "metric.json"
        result = _run_script(
            "ethoinsight.scripts.epm.compute_open_arm_time",
            ["--input", str(epm_trajectory_file), "--output", str(out_path)],
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        payload = json.loads(out_path.read_text())
        assert payload["metric"] == "open_arm_time"
        assert isinstance(payload["value"], float)
        assert payload["value"] >= 0.0


class TestComputeTotalEntryCount:
    def test_happy_path(self, epm_trajectory_file: Path, tmp_path: Path):
        out_path = tmp_path / "metric.json"
        result = _run_script(
            "ethoinsight.scripts.epm.compute_total_entry_count",
            ["--input", str(epm_trajectory_file), "--output", str(out_path)],
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        payload = json.loads(out_path.read_text())
        assert payload["metric"] == "total_entry_count"
        assert isinstance(payload["value"], int)
        assert payload["value"] >= 0


class TestPlotBoxOpenArm:
    def test_plot_with_groups(self, epm_trajectory_files: list[Path], tmp_path: Path):
        inputs_file = tmp_path / "inputs.json"
        inputs_file.write_text(json.dumps([str(p) for p in epm_trajectory_files]))

        # First 3 = control, last 3 = treatment
        groups_file = tmp_path / "groups.json"
        groups_file.write_text(
            json.dumps(
                {
                    "control": ["Subject 1", "Subject 2", "Subject 3"],
                    "treatment": ["Subject 4", "Subject 5", "Subject 6"],
                }
            )
        )

        out_path = tmp_path / "box.png"
        result = _run_script(
            "ethoinsight.scripts.epm.plot_box_open_arm",
            [
                "--inputs",
                str(inputs_file),
                "--groups",
                str(groups_file),
                "--output",
                str(out_path),
            ],
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert out_path.exists()
        assert out_path.stat().st_size > 1000


class TestRunGroupwiseStats:
    def test_stats_with_two_groups(
        self, epm_trajectory_files: list[Path], tmp_path: Path
    ):
        inputs_file = tmp_path / "inputs.json"
        inputs_file.write_text(json.dumps([str(p) for p in epm_trajectory_files]))

        groups_file = tmp_path / "groups.json"
        groups_file.write_text(
            json.dumps(
                {
                    "control": ["Subject 1", "Subject 2", "Subject 3"],
                    "treatment": ["Subject 4", "Subject 5", "Subject 6"],
                }
            )
        )

        out_path = tmp_path / "stats.json"
        result = _run_script(
            "ethoinsight.scripts.epm.run_groupwise_stats",
            [
                "--inputs",
                str(inputs_file),
                "--groups",
                str(groups_file),
                "--output",
                str(out_path),
            ],
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        payload = json.loads(out_path.read_text())
        assert "comparisons" in payload
        assert "alpha" in payload

    def test_stats_ssot_flat_map_format(
        self, epm_trajectory_files: list[Path], tmp_path: Path
    ):
        """Spec 2026-06-16 缺陷 1b：groups.json 用 SSOT ``{file: group}`` 格式（生产
        ``prep_metric_plan`` 实际写的格式）也能跑通 statistics。

        旧的 ``read_groups_json`` docstring 认为文件是 ``{group: [files]}``，透传会把
        字符串组名当可迭代拆字符。修复后函数内反转 flat map。本测试用真实 EPM
        statistics 脚本端到端证明 SSOT 格式不再炸/不再错乱。
        """
        inputs_file = tmp_path / "inputs.json"
        inputs_file.write_text(json.dumps([str(p) for p in epm_trajectory_files]))

        # SSOT flat map：subject_file -> group_name（与 prep_metric_plan 写的形态一致）。
        ssot_groups = {str(epm_trajectory_files[i]): ("control" if i < 3 else "treatment") for i in range(6)}
        groups_file = tmp_path / "groups.json"
        groups_file.write_text(json.dumps(ssot_groups))

        out_path = tmp_path / "stats_ssot.json"
        result = _run_script(
            "ethoinsight.scripts.epm.run_groupwise_stats",
            [
                "--inputs", str(inputs_file),
                "--groups", str(groups_file),
                "--output", str(out_path),
            ],
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        payload = json.loads(out_path.read_text())
        # 缺陷 1b 核心：SSOT flat map 不再炸（不再把组名字符串当可迭代拆）。
        # payload 成功产出 paradigm + comparisons 即证明 read_groups_json 反转生效。
        assert payload["paradigm"] == "epm"
        assert "comparisons" in payload

    def test_stats_resolves_mnt_virtual_paths(
        self, epm_trajectory_files: list[Path], tmp_path: Path
    ):
        """Spec 2026-06-16 缺陷 1a：inputs.json / groups.json 自身的 ``/mnt`` 虚拟路径
        经 ``resolve_sandbox_path`` 读到——真实 statistics 链不再 FileNotFoundError。

        red 锚点：修复前 ``read_inputs_json('/mnt/.../inputs.json')`` 直接 FileNotFoundError
        （脚本 rc≠0、run_metric_plan 记 ``{"id":"statistics"}`` 进 failures、payload 恒
        ``statistics:{}``）。修复后脚本 rc=0、payload 成功产出。

        断言精确到 spec red 锚点（不再 FileNotFoundError 失败），comparisons 是否非空
        受 fixture subject 命名 vs 文件路径匹配影响，与 1a/1b 正交，不在断言内。
        """
        real_workspace = tmp_path / "real_workspace"
        real_workspace.mkdir()
        real_uploads = tmp_path / "real_uploads"
        real_uploads.mkdir()

        # 把 6 个轨迹文件复制到 real_uploads，inputs.json 列 /mnt/uploads 虚拟路径。
        upload_virtual_paths = []
        for i, src in enumerate(epm_trajectory_files):
            dst = real_uploads / f"subj_{i + 1}.txt"
            dst.write_bytes(src.read_bytes())
            upload_virtual_paths.append(f"/mnt/user-data/uploads/subj_{i + 1}.txt")

        inputs_virtual = "/mnt/user-data/workspace/inputs.json"
        (real_workspace / "inputs.json").write_text(json.dumps(upload_virtual_paths), encoding="utf-8")

        # SSOT flat map：key 用 /mnt/uploads 虚拟路径（resolve 后指 real_uploads 里的文件）。
        ssot_groups = {upload_virtual_paths[i]: ("control" if i < 3 else "treatment") for i in range(6)}
        groups_virtual = "/mnt/user-data/workspace/groups.json"
        (real_workspace / "groups.json").write_text(json.dumps(ssot_groups), encoding="utf-8")

        out_path = tmp_path / "stats_mnt.json"

        # _run_script 跑 subprocess，env 经 env= 注入（DEERFLOW_PATH_* 让子进程能 resolve）。
        env = {
            **os.environ,
            "DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE": str(real_workspace),
            "DEERFLOW_PATH_MNT_USER_DATA_UPLOADS": str(real_uploads),
        }
        result = subprocess.run(
            [sys.executable, "-m", "ethoinsight.scripts.epm.run_groupwise_stats",
             "--inputs", inputs_virtual,
             "--groups", groups_virtual,
             "--output", str(out_path)],
            capture_output=True, text=True, check=False, env=env,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert out_path.exists()
        payload = json.loads(out_path.read_text())
        # 缺陷 1a 核心：/mnt 虚拟路径下的 inputs.json/groups.json 成功被读到、脚本产出 payload。
        assert payload["paradigm"] == "epm"
        assert "comparisons" in payload
