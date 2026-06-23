"""spec 2026-06-23-etho3-code-executor-triage-metadata-sidecar：triage 段指向 sidecar 契约。

ETHO-3 残留缺口：code-executor 的失败分诊（<triage>）查"指标定义/单位/期望集"时应读
去重元数据旁路 `_metric_metadata.json`（~5 条、几 KB），不读 133K 的 plan_metrics.json
全文（按 subject 重复 140 条，触发 read_file 截断）。复用 PR#169 已建的 sidecar，
只改 code_executor.py 的 system_prompt 指引（纯 prompt，零新机制）。

覆盖 spec §四三个测试项：
  - 项1 红→绿：triage 段指向 _metric_metadata.json（改前红、改后绿）。
  - 项2 守边界：triage 段保留 m_<metric_id>.json 单 subject 文件出口。
  - 项3 守 scope：本 spec 没碰 report-writer 的 sidecar 旁路（仍指向 _metric_metadata.json）。

ethoinsight 侧 metric_metadata_to_dict 的去重单测见
packages/ethoinsight/tests/test_metric_metadata_sidecar.py；
prep 落盘 + report-writer/data-analyst prompt 的契约见
tests/test_metric_metadata_sidecar_contract.py（本文件不重复）。
"""

from __future__ import annotations

from deerflow.subagents.builtins.code_executor import CODE_EXECUTOR_CONFIG
from deerflow.subagents.builtins.report_writer import REPORT_WRITER_CONFIG


def _triage_section() -> str:
    """抽出 <triage>...</triage> 段。

    用分段而非全 system_prompt，避免 happy-path plan_metrics.json 措辞（workflow /
    critical_rules 段，本 spec 不动）误触发断言——code_executor 的 triage 段是否含
    才是本 spec 的点。
    """
    sp = CODE_EXECUTOR_CONFIG.system_prompt
    start = sp.index("<triage>")
    end = sp.index("</triage>") + len("</triage>")
    return sp[start:end]


def test_triage_points_to_metadata_sidecar() -> None:
    """项1 红→绿：triage 段指向去重元数据旁路 _metric_metadata.json。

    改前此断言红（_metric_metadata.json 不在 triage 段），改后绿。
    用 _triage_section() 而非全 system_prompt，避免 report_writer/data_analyst
    （本就含此串）造成假绿——code_executor 的 triage 段是否含才是本 spec 的点。
    """
    triage = _triage_section()
    assert "_metric_metadata.json" in triage, (
        "triage 段必须指引分诊查指标定义/单位/期望集时读 _metric_metadata.json（去重旁路），"
        "而非扫 plan_metrics.json 133K 全量施工单"
    )
    # 正面措辞：点到去重投影（与 report_writer 范本对齐）
    assert "去重" in triage or "几 KB" in triage
    # 守 catastrophic forgetting：三分类 + 不重跑 run_metric_plan 原句仍在
    assert "plan 层错误" in triage
    assert "数据层错误" in triage
    assert "环境层错误" in triage
    assert "不要重新跑 run_metric_plan" in triage


def test_triage_keeps_single_subject_file_exit() -> None:
    """项2 守边界：triage 段保留 m_<metric_id>.json 单 subject 文件出口。

    分诊查「某 subject 的某 metric 具体产物写没写出」时，出口是单文件
    m_<metric_id>.json（<metric_id> 已含 __s<subject> 后缀，如
    m_open_arm_time_ratio__s0.json），不是读全量 plan_metrics。
    命名核实自 tests/test_run_metric_plan_validation_env.py:95,111 的真实 fixture。
    """
    triage = _triage_section()
    assert "m_<metric_id>.json" in triage, "triage 段必须保留单 subject 产物文件出口 m_<metric_id>.json"
    assert "m_open_arm_time_ratio__s0.json" in triage, (
        "给出具体命名示例（双下划线 __s<subject>），避免 LLM 用错格式"
    )


def test_metadata_sidecar_generation_unchanged() -> None:
    """项3 守 scope：本 spec 没碰 report-writer 的 sidecar 旁路。

    选「断言 report_writer prompt 仍指向 sidecar」而非「import metric_metadata_to_dict
    assert callable」的理由：
      - metric_metadata_to_dict 住在 ethoinsight 库（prep_metric_plan_tool import 它），
        本 spec 的改动域是 harness 的 code_executor.py prompt，根本没碰生成逻辑。
        import 它只能证明「函数还在」，不能证明「本 spec 没动它」——语义错位。
      - report_writer.py 是上一个 spec（2026-06-22-metric-metadata-sidecar）落地的 sidecar
        消费者，它的 prompt 仍指向 _metric_metadata.json = 「旁路契约整体未被本 spec 破坏」，
        正好是「本 spec 守 scope」的可观测投影。
      - 且 test_metric_metadata_sidecar_contract.py 已覆盖生成路径（prep 落盘），
        这里不重复，只补「邻域消费者契约未漂移」这层。
    """
    rw = REPORT_WRITER_CONFIG.system_prompt
    assert "_metric_metadata.json" in rw, (
        "report-writer prompt 仍应指向 _metric_metadata.json（本 spec 不改它，回归守边界）"
    )
