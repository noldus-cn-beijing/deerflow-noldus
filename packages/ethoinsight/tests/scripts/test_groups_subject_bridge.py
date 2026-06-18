"""第三层 bug 修复（spec 2026-06-16）：file→subject 标识桥接。

缺陷 1a/1b 修好后 statistics payload 能产出，但 ``comparisons`` 在真实数据上
**必然空**：``read_groups_json`` 反转后 groups 成员是**文件路径**，而
``compute_paradigm_metrics`` 按 ``parse_batch()["subjects"]`` 的 **subject key**
（EV19 "对象名称"，真实数据常为空串 ``''``）匹配——两者无交集 → ``matched`` 空
→ ``group_summary`` 空 → ``comparisons`` 空。

修法：``parse_batch`` 新增 ``file_subjects`` 映射（``{file_path: subject_key}``），
``run_groupwise_stats`` 用 ``bridge_groups_to_subjects`` 按**文件**（非 index）把
groups 的文件路径翻译成 subject key，再喂 dispatcher。

本测试是该第三层 bug 的独立红→绿证（与缺陷 1a/1b/2 的根因不同，单独立证）：
  - 红锚点：不经桥接、直接把文件路径 groups 喂 dispatcher → comparisons 空。
  - 绿：经桥接 → comparisons 非空。
  - 桥接按文件而非 index：某文件被 parse_batch 静默过滤时只丢该成员、不错位。
  - 真实数据 subject 名为空串：桥接照样工作（不依赖 subject 名非空）。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tests.scripts.conftest import _df_to_ethovision_file, _make_epm_df

from ethoinsight.metrics.dispatcher import compute_paradigm_metrics
from ethoinsight.parse import parse_batch
from ethoinsight.scripts._cli import bridge_groups_to_subjects
from ethoinsight.statistics import compare_groups


# EPM metric list the production stats script tests (mirrors run_groupwise_stats).
_EPM_METRICS = [
    "open_arm_time_ratio",
    "open_arm_entry_count",
    "open_arm_entry_ratio",
    "open_arm_time",
    "total_entry_count",
]


def _write_epm_files(
    tmp_path: Path, n: int, *, subject_name: str | None = None
) -> list[Path]:
    """Write n synthetic EPM trajectory files.

    ``subject_name=None`` (default): each file gets a **distinct** subject name
    ("S0".."Sn-1") so parse_batch yields n distinct subject keys and group sizes
    are meaningful — this is the faithful analogue of real data, whose blank
    subject + blank trial yields distinct ``''/_1/_2/...`` keys.

    ``subject_name=""``: all files share a blank EV19 "对象名称" (mimics the raw
    EV19 reality). Note the shared fixture writes a constant ``试验名称`` "Trial 1",
    so blank-subject + same-trial collides into 2 keys via parse_batch's
    ``{subject}_{trial}`` dedup branch — used only by the blank-name map test,
    NOT the comparison tests (which need distinct keys).
    """
    files: list[Path] = []
    for i in range(n):
        # control half: high open-arm; treatment half: low — so metrics differ.
        if i < n // 2:
            pattern = ([1] * 30 + [0] * 10) * 20
        else:
            pattern = ([1] * 5 + [0] * 35) * 20
        df = _make_epm_df(n_frames=400, open_arm_pattern=pattern[:400])
        path = tmp_path / f"epm_{i}.txt"
        name = subject_name if subject_name is not None else f"S{i}"
        _df_to_ethovision_file(df, path, subject=name)
        files.append(path)
    return files


# ============================================================================
# parse_batch file_subjects map
# ============================================================================


class TestParseBatchFileSubjectsMap:
    def test_map_links_each_file_to_its_subject_key(self, tmp_path: Path):
        files = _write_epm_files(tmp_path, 3, subject_name="")
        parsed = parse_batch([str(p) for p in files])

        fs = parsed["file_subjects"]
        # One entry per parsed file, keyed by the exact path passed in.
        assert set(fs.keys()) == {str(p) for p in files}
        # Values are the subjects-dict keys (empty EV19 name → ''/_1/_2 dedup).
        assert set(fs.values()) == set(parsed["subjects"].keys())

    def test_empty_subject_names_still_mapped(self, tmp_path: Path):
        """真实数据坐实：EV19 对象名空 → subject key 是 ''/_1/_2，但 file_subjects
        仍把每个文件链到它的 subjects-dict key（这正是文件名/stem 桥接做不到的）。

        注：合成 fixture 的 ``试验名称`` 都是 "Trial 1"（同名），blank-subject + 同
        trial 会触发 parse_batch 的 ``{subject}_{trial}`` 去重分支并**键碰撞**（真实
        数据 trial_name 为空、走 ``_{len}`` 分支不碰撞，见 handoff）。这里只断言桥接
        真正依赖的契约：**每个输入文件都在 file_subjects 里、且映射到某个存活的
        subjects key**——碰撞导致的 subject 折叠是 parse 层既有行为，与桥接正交。"""
        files = _write_epm_files(tmp_path, 3, subject_name="")
        parsed = parse_batch([str(p) for p in files])
        assert "" in parsed["subjects"]  # blank-name subject exists
        # 每个输入文件都被映射（桥接的核心契约：无文件遗漏）。
        assert set(parsed["file_subjects"].keys()) == {str(p) for p in files}
        # 每个映射值都是一个真实存活的 subjects key（非悬空）。
        assert all(v in parsed["subjects"] for v in parsed["file_subjects"].values())

    def test_empty_result_has_empty_map(self, tmp_path: Path):
        # No real trajectory files → empty subjects AND empty file_subjects.
        parsed = parse_batch([str(tmp_path / "does_not_exist.txt")])
        assert parsed["subjects"] == {}
        assert parsed["file_subjects"] == {}


# ============================================================================
# bridge_groups_to_subjects (pure helper)
# ============================================================================


class TestBridgeGroupsToSubjects:
    def test_translates_file_paths_to_subject_keys(self):
        file_subjects = {
            "/u/a.xlsx": "",
            "/u/b.xlsx": "_1",
            "/u/c.xlsx": "_2",
            "/u/d.xlsx": "_3",
        }
        groups = {
            "control": ["/u/a.xlsx", "/u/b.xlsx"],
            "treatment": ["/u/c.xlsx", "/u/d.xlsx"],
        }
        bridged = bridge_groups_to_subjects(groups, file_subjects)
        assert bridged == {
            "control": ["", "_1"],
            "treatment": ["_2", "_3"],
        }

    def test_drops_filtered_file_without_index_shift(self, capsys):
        """parse_batch 静默过滤掉某文件时（不在 file_subjects），桥接只丢该成员，
        不让后续成员错位到错误文件（按文件桥接，非 index）。"""
        # /u/b.xlsx was filtered out (not in file_subjects).
        file_subjects = {"/u/a.xlsx": "", "/u/c.xlsx": "_1"}
        groups = {"g": ["/u/a.xlsx", "/u/b.xlsx", "/u/c.xlsx"]}
        bridged = bridge_groups_to_subjects(groups, file_subjects)
        # a→'', c→'_1'; b dropped (not silently mapped onto the wrong subject).
        assert bridged == {"g": ["", "_1"]}
        # Drop is observable on stderr (no silent truncation).
        assert "/u/b.xlsx" in capsys.readouterr().err

    def test_empty_groups_yield_empty(self):
        assert bridge_groups_to_subjects({}, {"/u/a.xlsx": ""}) == {}


# ============================================================================
# End-to-end: red (file-path groups, no bridge) → green (bridged)
# ============================================================================


class TestComparisonsEmptyWithoutBridgeNonEmptyWith:
    def test_dispatcher_self_bridges_filepath_groups(self, tmp_path: Path):
        """权威契约（2026-06-18 box plot No-data 修复）：dispatcher 自己桥接文件路径 groups。

        历史：本测试原是「红锚点」，断言 raw dispatcher + 文件路径 groups → comparisons
        **空**——当时 file→subject 桥接只在 statistics 脚本层（bridge_groups_to_subjects）。
        但 box plot 走 compute_paradigm_metrics 直调这条路从未接桥 → group_summary 空 →
        箱线图 "No data"。修复把桥接下沉进 dispatcher（用 parsed_data["file_subjects"]），
        **dispatcher 成为唯一权威桥**，覆盖 box plot / statistics / 任何直调方。

        因此 raw dispatcher + 文件路径 groups 现在**应产出非空** comparisons。脚本层的
        bridge_groups_to_subjects 保留为幂等兵架（先翻成 subject_key，dispatcher 再桥时
        因不在 file_subjects 里而直接透传，安全无害）。
        """
        files = _write_epm_files(tmp_path, 6)
        paths = [str(p) for p in files]
        parsed = parse_batch(paths)

        # groups 成员是文件路径（read_groups_json 反转后的形态）——不经脚本层桥接，
        # 直接喂 dispatcher。dispatcher 现在用 file_subjects 自行桥接。
        filepath_groups = {"control": paths[:3], "treatment": paths[3:]}
        metrics = compute_paradigm_metrics(parsed, paradigm="epm", groups=filepath_groups)
        # group_summary 非空（旧 bug 这里恒空 → 箱线图 No data 的根因）。
        assert metrics["group_summary"], (
            "dispatcher 应用 file_subjects 自桥接文件路径 groups → group_summary 非空"
        )
        stats = compare_groups(metrics, metrics_to_test=_EPM_METRICS)
        assert stats.get("comparisons", {}), (
            "dispatcher 自桥接后文件路径 groups 应产出非空 comparisons"
        )

    def test_green_bridged_groups_yield_nonempty_comparisons(self, tmp_path: Path):
        """绿：桥接后 → comparisons 非空（含 EPM 指标）。"""
        files = _write_epm_files(tmp_path, 6)
        paths = [str(p) for p in files]
        parsed = parse_batch(paths)

        filepath_groups = {"control": paths[:3], "treatment": paths[3:]}
        bridged = bridge_groups_to_subjects(filepath_groups, parsed["file_subjects"])
        metrics = compute_paradigm_metrics(parsed, paradigm="epm", groups=bridged)
        stats = compare_groups(metrics, metrics_to_test=_EPM_METRICS)

        comparisons = stats.get("comparisons", {})
        assert comparisons, "桥接后 comparisons 应非空"
        # 至少一个 EPM 指标有真实组间比较结果。
        assert any(c for c in comparisons.values())


# ============================================================================
# End-to-end through the real production script (subprocess)
# ============================================================================


class TestRunGroupwiseStatsBridgedEndToEnd:
    def test_ssot_filepath_groups_produce_comparisons(self, tmp_path: Path):
        """端到端：用 SSOT ``{file: group}`` 格式（prep_metric_plan 实际写的形态）跑真实
        ``run_groupwise_stats`` 脚本，桥接生效 → comparisons 非空。

        这是第三层 bug 的端到端绿证：缺陷 1b 让 SSOT 格式不再炸，第三层桥接让
        comparisons 真正产出（而非空 payload）。"""
        files = _write_epm_files(tmp_path, 6)
        inputs_file = tmp_path / "inputs.json"
        inputs_file.write_text(json.dumps([str(p) for p in files]))

        # SSOT flat map: file -> group（与 prep_metric_plan 写的形态一致）。
        ssot_groups = {str(files[i]): ("control" if i < 3 else "treatment") for i in range(6)}
        groups_file = tmp_path / "groups.json"
        groups_file.write_text(json.dumps(ssot_groups))

        out_path = tmp_path / "stats.json"
        result = subprocess.run(
            [
                sys.executable, "-m", "ethoinsight.scripts.epm.run_groupwise_stats",
                "--inputs", str(inputs_file),
                "--groups", str(groups_file),
                "--output", str(out_path),
            ],
            capture_output=True, text=True, check=False,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        payload = json.loads(out_path.read_text())
        assert payload["paradigm"] == "epm"
        # 第三层修复核心：comparisons 真正非空（不再是产出 payload 但内容空）。
        assert payload.get("comparisons"), f"comparisons 应非空，payload={payload}"
