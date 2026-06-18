"""spec 2026-06-18: per-subject plot 脚本承接列对齐 zone 参数（红→绿 TDD）。

根因：HITL 列语义对齐（把用户列名 `open` 确认为开臂）对 compute 脚本生效、但对
per-subject plot 脚本未生效——plot 脚本裸调 `compute_*(df)` 不传 zone 参数，metric
函数走列名模式 fallback（找 `in_zone.*open_arm`），匹配不上用户列名 `open` → None →
chart-maker 报「no open-arm zone columns」。修：plot 脚本也接收并透传 zone 参数，
参数由 resolve_charts 从 column_aliases 单点投影注入 --parameters-json。

覆盖（全范式根治 + 跨范式守门）：
1. make_plot_parser 解析 --parameters-json（list/scalar 两形态）
2. resolve_charts 把 column_aliases 投影成 --parameters-json 注入 chart entry.args
3. per-subject plot 端到端：列名 `open` + --parameters-json → 出图（坐实红：裸调失败）
4. 错误信息区分「传参仍 None」vs「没传参」
5. OFT scalar center_zone / LDB scalar light_zone 跨范式守门
6. aggregate plot 透传 zone_overrides 给 dispatcher
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ethoinsight.catalog.resolve import resolve_charts
from ethoinsight.scripts._cli import make_plot_parser, parse_parameters


# ============================================================================
# helpers — 写最小 EthoVision 风格轨迹文件（UTF-16-LE BOM / 分号分隔）
# 与 test_plot_epm_single_subject_cli 同款
# ============================================================================


def _df_to_ethovision_file(df: pd.DataFrame, path: Path, *, subject: str = "Subject 1") -> None:
    columns = list(df.columns)
    n_header_lines = 6
    lines: list[str] = []
    lines.append(f'"标题行数";"{n_header_lines}"')
    lines.append(f'"对象名称";"{subject}"')
    lines.append('"试验名称";"Trial 1"')
    lines.append('"竞技场名称";"Arena 1"')
    lines.append(";".join(f'"{c}"' for c in columns))
    lines.append(";".join(['""'] * len(columns)))
    for _, row in df.iterrows():
        values = []
        for v in row.values:
            if pd.isna(v):
                values.append('"-"')
            else:
                values.append(f'"{v}"')
        lines.append(";".join(values))
    content = "\n".join(lines) + "\n"
    with open(path, "wb") as f:
        f.write(b"\xff\xfe")
        f.write(content.encode("utf-16-le"))


def _epm_trajectory_with_user_named_columns(path: Path, *, subject: str = "Subject 1") -> Path:
    """FewZones EPM 轨迹：开臂/闭臂列叫 `open`/`closed`（非 in_zone_* 标准）。

    复现 dogfood：用户列名经 HITL 对齐后是裸 `open`/`closed`，
    metric 自动检测的正则 `in_zone.*open.?arm` 匹配不上 → 必须靠 zone 参数。
    """
    rng = np.random.default_rng(42)
    n = 60
    in_open = np.zeros(n, dtype=int)
    in_open[5:10] = 1
    in_open[30:40] = 1
    in_closed = np.zeros(n, dtype=int)
    in_closed[10:25] = 1
    in_closed[40:55] = 1
    in_center = np.clip(np.ones(n, dtype=int) - in_open - in_closed, 0, 1)
    df = pd.DataFrame(
        {
            "trial_time": np.arange(n, dtype=float) * 0.04,
            "x_center": rng.uniform(100, 500, n),
            "y_center": rng.uniform(100, 500, n),
            "open": in_open,
            "closed": in_closed,
            "center": in_center,
        }
    )
    _df_to_ethovision_file(df, path, subject=subject)
    return path


def _oft_trajectory_with_user_named_columns(path: Path, *, subject: str = "Subject 1") -> Path:
    """OFT FewZones 轨迹：中心区列叫 `center`（非 in_zone_center 标准）。

    用户自定义分析区列经 HITL 对齐后是裸 `center`，metric 自动检测的
    `_find_center_zone_column` 默认 hint=in_zone_center 精确不匹配 → 必须靠 zone 参数。
    """
    rng = np.random.default_rng(7)
    n = 60
    in_center = np.zeros(n, dtype=int)
    in_center[10:30] = 1
    in_periphery = np.clip(np.ones(n, dtype=int) - in_center, 0, 1)
    df = pd.DataFrame(
        {
            "trial_time": np.arange(n, dtype=float) * 0.04,
            "x_center": rng.uniform(100, 500, n),
            "y_center": rng.uniform(100, 500, n),
            "center": in_center,
            "periphery": in_periphery,
        }
    )
    _df_to_ethovision_file(df, path, subject=subject)
    return path


def _run_script(module: str, args: list[str]) -> subprocess.CompletedProcess:
    """Run a plot script as a subprocess (real CLI, exercises argparse + parse_parameters).

    cwd 设为 worktree 的 packages/ethoinsight，使 ``-m ethoinsight.scripts.*`` 解析到
    worktree 源（而非主仓 editable 安装版本）——sys.path[0]=cwd 优先于 editable .pth。
    """
    cwd = str(Path(__file__).resolve().parent.parent)
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        capture_output=True, text=True, cwd=cwd,
    )


# ============================================================================
# 1. make_plot_parser 接受 --parameters-json（list / scalar 两形态）
# ============================================================================


def test_make_plot_parser_accepts_parameters_json_list():
    """list 型 zone 参数（EPM open_arm_zones=['open']）正确解析。"""
    parser = make_plot_parser(description="t", supports_groups=False)
    args = parser.parse_args(["--input", "/x.txt", "--output", "/o.png", "--parameters-json", '{"open_arm_zones": ["open"]}'])
    params = parse_parameters(args)
    assert params == {"open_arm_zones": ["open"]}


def test_make_plot_parser_accepts_parameters_json_scalar():
    """scalar 型 zone 参数（OFT center_zone='中心区'）正确解析。"""
    parser = make_plot_parser(description="t", supports_groups=False)
    args = parser.parse_args(["--input", "/x.txt", "--output", "/o.png", "--parameters-json", '{"center_zone": "中心区"}'])
    params = parse_parameters(args)
    assert params == {"center_zone": "中心区"}


def test_make_plot_parser_defaults_to_empty_parameters():
    """不传 --parameters-json 时 parse_parameters 返回 {}（向后兼容，走 metric 自动检测）。"""
    parser = make_plot_parser(description="t", supports_groups=False)
    args = parser.parse_args(["--input", "/x.txt", "--output", "/o.png"])
    assert parse_parameters(args) == {}


# ============================================================================
# 2. resolve_charts 把 column_aliases 投影成 --parameters-json 注入 chart entry.args
# ============================================================================


def test_resolve_charts_injects_zone_params_into_per_subject_plot(tmp_path):
    """EPM column_aliases {open: open_arms} → per_subject chart entry.args 含
    --parameters-json 且反序列化后是 open_arm_zones=['open']。"""
    pc = resolve_charts(
        paradigm="epm",
        columns=["open", "closed", "trial_time", "x_center", "y_center"],
        raw_files=["/tmp/raw1.txt"],
        workspace_dir=str(tmp_path),
        total_subjects=1, n_per_group=1, n_groups=1,
        column_aliases={"open": "open_arms", "closed": "closed_arms"},
    )
    bar_charts = [c for c in pc.charts if c.id == "open_arm_time_ratio_bar"]
    assert bar_charts, f"open_arm_time_ratio_bar 未生成，charts={[c.id for c in pc.charts]}"
    args = bar_charts[0].args
    assert "--parameters-json" in args, f"entry.args 缺 --parameters-json: {args}"
    idx = args.index("--parameters-json")
    payload = json.loads(args[idx + 1])
    assert payload.get("open_arm_zones") == ["open"], f"open_arm_zones 投影错: {payload}"
    assert payload.get("closed_arm_zones") == ["closed"], f"closed_arm_zones 投影错: {payload}"


def test_resolve_charts_injects_zone_params_into_aggregate_plot(tmp_path):
    """aggregate chart（box_open_arm, needs_groups）同样注入 --parameters-json。"""
    pc = resolve_charts(
        paradigm="epm",
        columns=["open", "closed", "trial_time", "x_center", "y_center"],
        raw_files=["/tmp/r1.txt", "/tmp/r2.txt"],
        workspace_dir=str(tmp_path),
        total_subjects=6, n_per_group=3, n_groups=2,
        groups={"control": ["/tmp/r1.txt", "/tmp/r2.txt"]},
        column_aliases={"open": "open_arms", "closed": "closed_arms"},
    )
    box = [c for c in pc.charts if c.id == "box_open_arm"]
    assert box, "box_open_arm 未生成（需 n_per_group>=3 + groups）"
    args = box[0].args
    assert "--parameters-json" in args
    idx = args.index("--parameters-json")
    assert json.loads(args[idx + 1])["open_arm_zones"] == ["open"]


def test_resolve_charts_no_aliases_no_parameters_json(tmp_path):
    """无 column_aliases（标准列名场景）→ entry.args 不含 --parameters-json（不污染）。"""
    pc = resolve_charts(
        paradigm="epm",
        columns=["in_zone_open_arms_center", "in_zone_closed_arms_center", "trial_time"],
        raw_files=["/tmp/raw1.txt"],
        workspace_dir=str(tmp_path),
        total_subjects=1, n_per_group=1, n_groups=1,
    )
    bar_charts = [c for c in pc.charts if c.id == "open_arm_time_ratio_bar"]
    assert bar_charts
    assert "--parameters-json" not in bar_charts[0].args


# ============================================================================
# 3. per-subject plot 端到端：列名 `open` + --parameters-json → 出图（红→绿）
# ============================================================================


@pytest.fixture
def epm_user_named_file(tmp_path: Path) -> Path:
    return _epm_trajectory_with_user_named_columns(tmp_path / "epm_user.txt")


def test_plot_open_arm_bar_with_aliased_column_succeeds(epm_user_named_file: Path, tmp_path: Path):
    """绿：列名 `open` + --parameters-json '{"open_arm_zones":["open"]}' → exit 0 + 出图。"""
    output = tmp_path / "bar.png"
    result = _run_script(
        "ethoinsight.scripts.epm.plot_open_arm_time_ratio_bar",
        ["--input", str(epm_user_named_file), "--output", str(output),
         "--parameters-json", '{"open_arm_zones": ["open"]}'],
    )
    if result.returncode != 0:
        pytest.fail(f"应成功却失败: stdout={result.stdout} stderr={result.stderr}")
    assert output.exists() and output.stat().st_size > 0
    assert "[result]" in result.stdout


def test_plot_open_arm_bar_without_params_fails_on_user_named_column(epm_user_named_file: Path, tmp_path: Path):
    """坐实红：列名 `open` 裸调（无 --parameters-json）→ exit 1 + 「no open-arm zone columns」。

    这正是 dogfood 的失败形态：metric 自动检测匹配不上用户列名。
    """
    output = tmp_path / "bar_fail.png"
    result = _run_script(
        "ethoinsight.scripts.epm.plot_open_arm_time_ratio_bar",
        ["--input", str(epm_user_named_file), "--output", str(output)],
    )
    assert result.returncode == 1, f"裸调应失败: stdout={result.stdout} stderr={result.stderr}"
    assert "no open-arm zone columns" in result.stderr


# ============================================================================
# 4. 错误信息区分「传参仍 None」vs「没传参」（便于诊断）
# ============================================================================


def test_plot_error_message_distinguishes_missing_param_vs_missing_column(epm_user_named_file: Path, tmp_path: Path):
    """没传 --parameters-json 的 stderr 应提示「未收到 zone 对齐参数」。

    传了参数但列名仍不匹配（用错误列名）的 stderr 应提示「已传 zone 参数仍无值」。
    两种失败文案不同，便于诊断是注入链断了还是真无该列。
    """
    # 没传参数
    r_no_param = _run_script(
        "ethoinsight.scripts.epm.plot_open_arm_time_ratio_bar",
        ["--input", str(epm_user_named_file), "--output", str(tmp_path / "a.png")],
    )
    assert r_no_param.returncode == 1
    assert "未收到 zone 对齐参数" in r_no_param.stderr, f"缺「未收到参数」提示: {r_no_param.stderr}"

    # 传了参数但列名错（数据里没 nonexistent_arm 列）
    r_param_no_col = _run_script(
        "ethoinsight.scripts.epm.plot_open_arm_time_ratio_bar",
        ["--input", str(epm_user_named_file), "--output", str(tmp_path / "b.png"),
         "--parameters-json", '{"open_arm_zones": ["nonexistent_arm"]}'],
    )
    assert r_param_no_col.returncode == 1
    assert "已传 zone 参数仍无值" in r_param_no_col.stderr, f"缺「已传参数仍无值」提示: {r_param_no_col.stderr}"


# ============================================================================
# 5. 跨范式守门：OFT scalar center_zone / LDB scalar light_zone
# ============================================================================


@pytest.fixture
def oft_user_named_file(tmp_path: Path) -> Path:
    return _oft_trajectory_with_user_named_columns(tmp_path / "oft_user.txt")


def test_plot_center_time_ratio_bar_with_aliased_column(oft_user_named_file: Path, tmp_path: Path):
    """OFT scalar：列名 `center` + --parameters-json '{"center_zone":"center"}' → 出图。"""
    output = tmp_path / "center_bar.png"
    result = _run_script(
        "ethoinsight.scripts.oft.plot_center_time_ratio_bar",
        ["--input", str(oft_user_named_file), "--output", str(output),
         "--parameters-json", '{"center_zone": "center"}'],
    )
    if result.returncode != 0:
        pytest.fail(f"OFT 应成功: stdout={result.stdout} stderr={result.stderr}")
    assert output.exists() and output.stat().st_size > 0


def test_plot_center_time_ratio_bar_without_params_fails(oft_user_named_file: Path, tmp_path: Path):
    """坐实红：OFT 列名 `center` 裸调 → exit 1（自动检测默认找 in_zone_center）。"""
    output = tmp_path / "center_fail.png"
    result = _run_script(
        "ethoinsight.scripts.oft.plot_center_time_ratio_bar",
        ["--input", str(oft_user_named_file), "--output", str(output)],
    )
    assert result.returncode == 1
    assert "no center zone columns" in result.stderr


def _ldb_trajectory_with_user_named_columns(path: Path, *, subject: str = "Subject 1") -> Path:
    """LDB FewZones 轨迹：明区/暗区列叫 `light`/`dark`（非 in_zone_light/dark）。"""
    rng = np.random.default_rng(11)
    n = 60
    in_light = np.zeros(n, dtype=int)
    in_light[0:30] = 1
    in_dark = np.clip(np.ones(n, dtype=int) - in_light, 0, 1)
    df = pd.DataFrame(
        {
            "trial_time": np.arange(n, dtype=float) * 0.04,
            "x_center": rng.uniform(100, 500, n),
            "y_center": rng.uniform(100, 500, n),
            "light": in_light,
            "dark": in_dark,
        }
    )
    _df_to_ethovision_file(df, path, subject=subject)
    return path


@pytest.fixture
def ldb_user_named_file(tmp_path: Path) -> Path:
    return _ldb_trajectory_with_user_named_columns(tmp_path / "ldb_user.txt")


def test_plot_ldb_zone_entry_with_aliased_column(ldb_user_named_file: Path, tmp_path: Path):
    """LDB scalar：列名 `light`/`dark` + --parameters-json → zone_entry_distribution 出图。

    compute_light_entry_count 走 light_zone 参数（scalar）；裸调会因找不到 in_zone_light 列返回 0。
    """
    output = tmp_path / "ldb_zone.png"
    result = _run_script(
        "ethoinsight.scripts.ldb.plot_zone_entry_distribution",
        ["--input", str(ldb_user_named_file), "--output", str(output),
         "--parameters-json", '{"light_zone": "light", "dark_zone": "dark"}'],
    )
    if result.returncode != 0:
        pytest.fail(f"LDB 应成功: stdout={result.stdout} stderr={result.stderr}")
    assert output.exists() and output.stat().st_size > 0


# ============================================================================
# 6. aggregate plot 透传 zone_overrides 给 dispatcher
# ============================================================================


def test_plot_box_open_arm_uses_zone_overrides(tmp_path: Path):
    """aggregate box（subprocess）：多 subject 列名 `open`/`closed` + --parameters-json → 出图。

    进程内守门见 test_dispatcher_aggregate_uses_zone_overrides_on_user_named_columns。
    """
    raw1 = _epm_trajectory_with_user_named_columns(tmp_path / "s1.txt", subject="S1")
    raw2 = _epm_trajectory_with_user_named_columns(tmp_path / "s2.txt", subject="S2")
    inputs_json = tmp_path / "inputs.json"
    inputs_json.write_text(json.dumps([str(raw1), str(raw2)]), encoding="utf-8")
    groups_json = tmp_path / "groups.json"
    groups_json.write_text(json.dumps({"control": ["S1", "S2"]}), encoding="utf-8")
    output = tmp_path / "box.png"
    result = _run_script(
        "ethoinsight.scripts.epm.plot_box_open_arm",
        ["--inputs", str(inputs_json), "--groups", str(groups_json), "--output", str(output),
         "--parameters-json", '{"open_arm_zones": ["open"], "closed_arm_zones": ["closed"]}'],
    )
    if result.returncode != 0:
        pytest.fail(f"aggregate box 应成功: stdout={result.stdout} stderr={result.stderr}")
    assert output.exists() and output.stat().st_size > 0


def test_dispatcher_aggregate_uses_zone_overrides_on_user_named_columns(tmp_path: Path):
    """进程内守门：列名 `open`/`closed` 时，dispatcher 带 zone_overrides 算出非 None 的
    open_arm_time_ratio；不带 zone_overrides（裸调，走自动检测）则返回 None。

    证明 aggregate plot 透传 zone_overrides 是必要的（不只是装饰）——
    plot_box_open_arm 等改前不传 zone_overrides，与 compute 脚本结果不一致。
    """
    from ethoinsight.metrics.dispatcher import compute_paradigm_metrics
    from ethoinsight.parse import parse_batch

    raw1 = _epm_trajectory_with_user_named_columns(tmp_path / "s1.txt", subject="S1")
    raw2 = _epm_trajectory_with_user_named_columns(tmp_path / "s2.txt", subject="S2")
    parsed = parse_batch([str(raw1), str(raw2)])
    groups = {"control": ["S1", "S2"]}

    # 裸调：自动检测匹配不上用户列名 `open` → None
    metrics_bare = compute_paradigm_metrics(parsed, paradigm="epm", groups=groups)
    per_subject_bare = metrics_bare["per_subject"]
    assert all(v["open_arm_time_ratio"] is None for v in per_subject_bare.values()), (
        f"裸调应因列名不匹配返回 None: {per_subject_bare}"
    )

    # 传 zone_overrides：命中用户列名 `open` → 非 None
    metrics_aligned = compute_paradigm_metrics(
        parsed, paradigm="epm", groups=groups,
        zone_overrides={"open_arm_zones": ["open"], "closed_arm_zones": ["closed"]},
    )
    per_subject_aligned = metrics_aligned["per_subject"]
    for name, m in per_subject_aligned.items():
        assert m["open_arm_time_ratio"] is not None, f"{name} open_arm_time_ratio 应非 None: {m}"
    # 且两种算法结果一致性：传参后值应与单 subject compute 一致（同源同语义）
