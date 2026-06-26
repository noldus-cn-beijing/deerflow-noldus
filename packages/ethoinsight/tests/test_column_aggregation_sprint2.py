"""EV19 列语义对齐 Sprint 2 — 结构聚合（N 列 → 1 概念）坐实 / 固化 / 补测。

spec: docs/superpowers/specs/2026-06-26-column-semantics-sprint2-structural-aggregation-spec.md

本 sprint 不从零造聚合——N 列→1 概念的 OR 聚合全链路已存在并工作（
catalog/resolve.py:_build_zone_aliases_overrides 多列收集 + metrics/epm.py:max(axis=1) OR +
metrics/_common.py:_count_zone_entries OR-then-transition）。这里补的是「锁行为」的
直接回归测试 + 同事方法论的 special 范式规则固化，防止未来回归。

任务 1：多列 OR 聚合的直接测试（显式 open_arm_zones=['A','B']）+ resolve 端到端 +
       累积分析区去重陷阱（同事原话：同一分析区群组内 OR 安全；不同群组间堆叠会双重计数）。
任务 2：LDB 隐藏区忽略（__ignore__）/ FST·TST 不分区（不触发 zone 反问）。
任务 3：仅当任务 1 陷阱测试暴露双重计数 bug 才做（条件性）。

聚合正确性断言（spec 四.2 常驻回归纪律，守 memory
feedback_2026-06-16_io_boundary_asymmetry_and_aggregator_half_built 的「mean 正确性常驻断言」）：
两列 OR 的占比/计数全部手算对照，不是「跑通即过」。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ethoinsight.catalog.resolve import resolve_metrics


# ============================================================================
# Helpers
# ============================================================================


def _df_two_open_arms(
    *,
    n: int = 100,
    dt: float = 0.04,
    oa_a_bouts: list[tuple[int, int]] | None = None,
    oa_b_bouts: list[tuple[int, int]] | None = None,
) -> pd.DataFrame:
    """构造一份有两列开放臂指标（in_zone_open_arm_A / in_zone_open_arm_B）的 df。

    bout = (start, end_inclusive)。两列各自在不同帧 ==1，制造「按帧 OR」的语义：
    动物在 A 臂或 B 臂任一即视为「在开放臂」。
    """
    if oa_a_bouts is None:
        oa_a_bouts = [(10, 19), (50, 54)]  # A 臂停留：帧 10-19、50-54
    if oa_b_bouts is None:
        oa_b_bouts = [(30, 39), (70, 79)]  # B 臂停留：帧 30-39、70-79
    a = np.zeros(n, dtype=int)
    b = np.zeros(n, dtype=int)
    for s, e in oa_a_bouts:
        a[s : e + 1] = 1
    for s, e in oa_b_bouts:
        b[s : e + 1] = 1
    return pd.DataFrame(
        {
            "trial_time": np.arange(n, dtype=float) * dt,
            "x_center": np.zeros(n),
            "y_center": np.zeros(n),
            "in_zone_open_arm_A": a,
            "in_zone_open_arm_B": b,
        }
    )


# ============================================================================
# 任务 1a — 多列 OR 聚合：显式 open_arm_zones=['A','B'] 路径（手算对照）
# ============================================================================
#
# 数据布局（_df_two_open_arms 默认）：
#   A 臂 ==1 的帧：10-19（10 帧）、50-54（5 帧）  → A 共 15 帧
#   B 臂 ==1 的帧：30-39（10 帧）、70-79（10 帧） → B 共 20 帧
#   A、B 不重叠 → OR 合并后 in-open-arm 帧 = 15 + 20 = 35 帧
#   占比 = 35 / 100 = 0.35
#   OR 合并后的 0→1 跳变：A 在 10 进入、B 在 30 进入、A 在 50 进入、B 在 70 进入 = 4 次
#     （若逐列分别计数再加总 = 2 + 2 = 4，此例不重叠所以巧合相同——重叠场景见 1a-entry-overlap）
# ============================================================================


class TestMultiColumnOpenArmAggregation:
    """两列开放臂显式注入 open_arm_zones 的 OR 聚合数值正确性（手算对照）。"""

    def test_time_ratio_is_or_not_sum(self):
        """compute_open_arm_time_ratio(df, open_arm_zones=['A','B']) == OR 后占比 0.35。

        关键：必须是两列 OR（max）的占比 0.35，不是两列占比之和（0.15+0.20=0.35，此例
        不重叠巧合相同）。重叠场景在 test_time_ratio_overlap 下用 OR≠sum 钉死。
        """
        from ethoinsight.metrics import compute_open_arm_time_ratio

        df = _df_two_open_arms()
        ratio = compute_open_arm_time_ratio(df, open_arm_zones=["in_zone_open_arm_A", "in_zone_open_arm_B"])
        assert ratio == pytest.approx(0.35)

    def test_time_ratio_overlap_proves_or_not_sum(self):
        """重叠帧：A=帧0-49、B=帧25-74。OR=帧0-74（75 帧，占比0.75）；sum 占比=1.25。

        这条钉死「OR 合并」语义——若实现错成逐列相加会把占比算成 1.25（>1，物理不可能）。
        """
        from ethoinsight.metrics import compute_open_arm_time_ratio

        n = 100
        a = np.zeros(n, dtype=int)
        b = np.zeros(n, dtype=int)
        a[0:50] = 1
        b[25:75] = 1
        df = pd.DataFrame(
            {
                "trial_time": np.arange(n, dtype=float) * 0.04,
                "in_zone_open_arm_A": a,
                "in_zone_open_arm_B": b,
            }
        )
        ratio = compute_open_arm_time_ratio(df, open_arm_zones=["in_zone_open_arm_A", "in_zone_open_arm_B"])
        assert ratio == pytest.approx(0.75)  # OR；不是 1.25（sum）

    def test_time_is_or_frame_count_times_dt(self):
        """compute_open_arm_time(df, open_arm_zones=['A','B']) == OR 帧数 × dt。

        OR 帧数 = 35，dt = 0.04 → 1.4 秒。不是 A_time + B_time（也 1.4s 此例，但
        重叠场景由 test_time_ratio_overlap 钉 OR 语义）。
        """
        from ethoinsight.metrics import compute_open_arm_time

        df = _df_two_open_arms()
        t = compute_open_arm_time(df, open_arm_zones=["in_zone_open_arm_A", "in_zone_open_arm_B"])
        assert t == pytest.approx(35 * 0.04)

    def test_entry_count_or_transitions_not_per_column_sum(self):
        """compute_open_arm_entry_count 两列 OR 后的跳变计数。

        重叠 + 相邻场景：A 臂帧 10-29、B 臂帧 25-44（帧 25-29 重叠）。
        OR 合并序列：帧 10-44 连续为 1 → 只有 1 次 0→1 跳变（在帧 10）。
        若逐列分别计数再加总：A 1 次 + B 1 次 = 2（双重计数，这正是同事说的陷阱）。
        守 epm.py compute_total_entry_count 注释「OR to avoid overcounting」。
        """
        from ethoinsight.metrics import compute_open_arm_entry_count

        n = 100
        a = np.zeros(n, dtype=int)
        b = np.zeros(n, dtype=int)
        a[10:30] = 1
        b[25:45] = 1  # 与 A 在帧 25-29 重叠
        df = pd.DataFrame(
            {
                "trial_time": np.arange(n, dtype=float) * 0.04,
                "in_zone_open_arm_A": a,
                "in_zone_open_arm_B": b,
            }
        )
        count = compute_open_arm_entry_count(df, open_arm_zones=["in_zone_open_arm_A", "in_zone_open_arm_B"])
        assert count == 1  # OR 合并后仅 1 次进入；不是 2

    def test_total_entry_count_groups_or_then_sum(self):
        """compute_total_entry_count：每组组内 OR 防重复计数，再跨组求和。

        开放臂 A/B（帧10-29 重叠成一次）+ 闭臂 C（帧 50-69 一次）→ 总进入 = 1 + 1 = 2。
        若组内不做 OR（逐列加总）会把开放臂算成 2 → 总 3（错）。
        """
        from ethoinsight.metrics import compute_total_entry_count

        n = 100
        a = np.zeros(n, dtype=int)
        b = np.zeros(n, dtype=int)
        c = np.zeros(n, dtype=int)
        a[10:30] = 1
        b[25:45] = 1  # 与 A 重叠 → 开放臂组 OR 后 1 次
        c[50:70] = 1  # 闭臂 1 次
        df = pd.DataFrame(
            {
                "trial_time": np.arange(n, dtype=float) * 0.04,
                "in_zone_open_arm_A": a,
                "in_zone_open_arm_B": b,
                "in_zone_closed_arm_C": c,
            }
        )
        total = compute_total_entry_count(
            df,
            open_arm_zones=["in_zone_open_arm_A", "in_zone_open_arm_B"],
            closed_arm_zones=["in_zone_closed_arm_C"],
        )
        assert total == 2  # 开放臂组 1 + 闭臂组 1


# ============================================================================
# 任务 1b — resolve 端到端：column_aliases 两列同 concept → plan 两列都进、不丢
# ============================================================================


class TestResolveMultiColumnZoneAliases:
    """resolve_metrics 把两列同 concept 的 column_aliases 投影成 list，两列都保留。"""

    def test_two_open_arm_columns_both_in_plan(self):
        """column_aliases {'A':'open_arms','B':'open_arms'} → open_arm_time_ratio 的
        parameters_in_use['open_arm_zones'] == ['in_zone_open_arm_A','in_zone_open_arm_B']。

        这锁住「N 列收集不丢列」——Sprint 2 核心行为。
        """
        columns = [
            "trial_time",
            "x_center",
            "y_center",
            "in_zone_open_arm_A",
            "in_zone_open_arm_B",
            "in_zone_closed_arm_C",
        ]
        pm = resolve_metrics(
            "epm",
            columns,
            raw_files=["/tmp/stub_epm.csv"],
            workspace_dir="/tmp",
            column_aliases={
                "in_zone_open_arm_A": "open_arms",
                "in_zone_open_arm_B": "open_arms",
                "in_zone_closed_arm_C": "closed_arms",
            },
        )
        ratio_metric = next(m for m in pm.metrics if m.id == "open_arm_time_ratio")
        zones = ratio_metric.parameters_in_use.get("open_arm_zones")
        assert zones is not None, f"open_arm_zones missing from parameters_in_use: {ratio_metric.parameters_in_use}"
        assert set(zones) == {"in_zone_open_arm_A", "in_zone_open_arm_B"}, (
            f"both open-arm columns must be collected, got {zones}"
        )


# ============================================================================
# 任务 1c — 累积分析区去重陷阱（同事原话：不同群组间堆叠会双重计数）
# ============================================================================
#
# 同事方法论 SSOT（docs/review-packages/2026-06-09-feedbacks/自定义数据列识别对齐聚合.md）：
# 同一分析区群组内互斥（OR 安全）；**不同群组间可能堆叠**——如 open_arm_A/B（分量列）
# + open_arm_all（EV 内已聚合的累积列）三者都输出时，三者 OR 会双重计数（all 已含 A/B）。
# 这是聚合的唯一真正陷阱。
#
# 本测试构造该陷阱数据：all 列 == A OR B（all 是 A/B 的并集），三者同 alias 成 open_arms。
# 预期（同事规则）：择 all（累积列）或只 OR A/B，两者数值应一致（all 单列的占比）。
#   A=帧10-19, B=帧30-39 → all = A|B = 帧10-19∪30-39（20 帧，占比 0.20）
# 若实现把三者无脑 OR：max(A,B,all) = all（因为 all⊇A 且 all⊇B）→ 占比仍 0.20。
#   → 当前 max(axis=1) OR 在「all 是 A/B 的并集」时数值恰好正确（all 支配）。
# 但 entry_count 会暴露：all 的跳变 = A、B 各一次跳变的并集。
#   关键判据：all(A|B) 的 0→1 跳变数 == A、B OR 后的跳变数（都是 2），不是 3。
# 若未来有人把三者「分别计数再加总」就会得 3（双重计数）。
# ============================================================================


class TestCumulativeZoneDedupTrap:
    """累积分析区（EV 内已聚合）+ 分量列同 alias 的去重陷阱守护。"""

    def _trap_df(self) -> pd.DataFrame:
        """A/B 分量列 + all 累积列（all = A|B 并集）的陷阱数据。"""
        n = 100
        a = np.zeros(n, dtype=int)
        b = np.zeros(n, dtype=int)
        all_col = np.zeros(n, dtype=int)
        a[10:20] = 1  # 帧 10-19 在 A
        b[30:40] = 1  # 帧 30-39 在 B
        all_col[10:20] = 1
        all_col[30:40] = 1  # all = A ∪ B
        return pd.DataFrame(
            {
                "trial_time": np.arange(n, dtype=float) * 0.04,
                "in_zone_open_arm_A": a,
                "in_zone_open_arm_B": b,
                "in_zone_open_arm_all": all_col,
            }
        )

    def test_time_ratio_not_double_counted(self):
        """三者同 alias 后 time_ratio 应等于 all 单列占比（0.20），不被双重计数放大。

        当前 max(axis=1) OR 实现：max(A,B,all) == all（all 支配 A、B）→ 0.20，正确。
        本测试锁住「即使把 all 也喂进 OR，占比也不膨胀」——若未来改为 sum 或别的
        算子导致 >0.20，立即回归报警。
        """
        from ethoinsight.metrics import compute_open_arm_time_ratio

        df = self._trap_df()
        ratio = compute_open_arm_time_ratio(
            df,
            open_arm_zones=["in_zone_open_arm_A", "in_zone_open_arm_B", "in_zone_open_arm_all"],
        )
        assert ratio == pytest.approx(0.20), (
            f"cumulative-all + components must not double-count; expected 0.20, got {ratio}"
        )

    def test_entry_count_not_double_counted(self):
        """三者同 alias 后 entry_count 应等于 A|B OR 后的跳变数（2），不是 3。

        all 的跳变（帧10、帧30）== A、B 各自跳变的并集。三者 OR 后 max 序列 = all
        序列 → 跳变 2 次。若逐列分别计数加总 = 1+1+2 = 4（严重双重计数）。
        """
        from ethoinsight.metrics import compute_open_arm_entry_count

        df = self._trap_df()
        count = compute_open_arm_entry_count(
            df,
            open_arm_zones=["in_zone_open_arm_A", "in_zone_open_arm_B", "in_zone_open_arm_all"],
        )
        assert count == 2, f"cumulative-all must not inflate entry count; expected 2, got {count}"


# ============================================================================
# 任务 2a — LDB 隐藏区忽略（alias 成 __ignore__，不报缺列、不参与聚合）
# ============================================================================
#
# 同事方法论：LDB 明区 + 暗区；隐藏分析区（入口）→ 忽略（其坐标最终落入暗区，
# 直接算暗区即可）。隐藏区列**不参与聚合、不报缺列**——alias 成 __ignore__ 由
# _apply_aliases 移除。
# ============================================================================


class TestLDBHiddenZoneIgnore:
    """LDB 隐藏区列被 alias 成 __ignore__ 后不污染 plan、不报缺列。"""

    def test_hidden_zone_aliased_ignore_removed_from_columns(self):
        """隐藏区列 alias=__ignore__ → resolve 不报 columns_missing、plan 正常生成。

        数据有 in_zone_light（明区）、in_zone_dark（暗区）、in_zone_hidden（隐藏区，
        用户标 __ignore__）。隐藏区不参与明/暗聚合、不应触发缺列。
        """
        columns = ["trial_time", "x_center", "y_center", "in_zone_light", "in_zone_dark", "in_zone_hidden"]
        # 不应抛 columns_missing；隐藏区被移除
        pm = resolve_metrics(
            "light_dark_box",
            columns,
            raw_files=["/tmp/stub_ldb.csv"],
            workspace_dir="/tmp",
            column_aliases={
                "in_zone_light": "light",
                "in_zone_dark": "dark",
                "in_zone_hidden": "__ignore__",
            },
        )
        ids = {m.id for m in pm.metrics}
        # 明区核心指标应在 plan 里（不因隐藏区缺失而失败）
        assert "light_time_ratio" in ids
        assert "transition_count" in ids

    def test_hidden_zone_not_in_dark_aggregation(self):
        """暗区指标只用暗区列，隐藏区列（__ignore__）不混入 dark_zone。

        column_aliases 里隐藏区=__ignore__ → _build_zone_aliases_overrides 跳过它
        （concept in (None,'__ignore__') 的 continue 分支，resolve.py L691），
        dark_zone 只含暗区列。
        """
        from ethoinsight.catalog.resolve import _build_zone_aliases_overrides
        from ethoinsight.catalog.loader import load_catalog

        cat = load_catalog("light_dark_box")
        overrides = _build_zone_aliases_overrides(
            {
                "in_zone_light": "light",
                "in_zone_dark": "dark",
                "in_zone_hidden": "__ignore__",
            },
            cat,
            {},
        )
        assert "dark_zone" in overrides, f"dark_zone should be present, got {overrides}"
        assert overrides["dark_zone"] == "in_zone_dark", (
            f"dark_zone must be only the dark column, hidden zone must not leak in; got {overrides['dark_zone']}"
        )
        # 隐藏区绝不应出现在任何 override 值里
        for v in overrides.values():
            assert "in_zone_hidden" not in str(v), f"hidden zone leaked into override: {overrides}"


# ============================================================================
# 任务 2b — FST / TST 不分区：列对齐链路不触发 zone 反问（negative 守护）
# ============================================================================
#
# 同事方法论：FST/TST 不分区。这两范式无 zone metric、列对齐链路不应对它们触发
# zone 反问。_build_zone_aliases_overrides 对无 zone_pattern 的范式返回 {}（已坐实
# resolve.py L685-686：zone_patterns 为空即返回 {}）。本测试锁住这个 negative。
# ============================================================================


class TestFstTstNoZoneReverseAsk:
    """FST/TST 不分区——_build_zone_aliases_overrides 对它们返回 {}，无 zone 反问。"""

    @pytest.mark.parametrize("paradigm", ["fst", "tst"])
    def test_no_zone_overrides_for_non_zone_paradigm(self, paradigm):
        """FST/TST 即便喂了带 zone 味道的 column_aliases，也不产任何 zone override。

        这锁住「不分区范式绝不被列对齐链路触发 zone 反问」——若有人误给 FST/TST
        加 zone_concept_params 或 in_zone requires_columns，本测试立即红。
        """
        from ethoinsight.catalog.resolve import _build_zone_aliases_overrides
        from ethoinsight.catalog.loader import load_catalog

        cat = load_catalog(paradigm)
        # 故意喂一个看起来像 zone 的 alias，确认 FST/TST 不会被它激活
        result = _build_zone_aliases_overrides(
            {"in_zone_open_arms": "open_arms"},
            cat,
            {},
        )
        assert result == {}, (
            f"{paradigm} must not produce zone overrides (no zones), got {result}"
        )

    @pytest.mark.parametrize("paradigm", ["fst", "tst"])
    def test_catalog_has_no_zone_concepts(self, paradigm):
        """FST/TST catalog 无 zone_concept_params、无 anonymous_zone_override。

        结构性守护：不分区范式的 catalog 不该声明任何 zone 注入点。
        """
        from ethoinsight.catalog.loader import load_catalog

        cat = load_catalog(paradigm)
        assert not cat.resolved_zone_concepts, (
            f"{paradigm} must have NO resolved zone concepts (it is a non-zone paradigm), "
            f"got {list(cat.resolved_zone_concepts.keys())}"
        )
        assert cat.anonymous_zone_override is None, (
            f"{paradigm} must have NO anonymous_zone_override, got {cat.anonymous_zone_override}"
        )
