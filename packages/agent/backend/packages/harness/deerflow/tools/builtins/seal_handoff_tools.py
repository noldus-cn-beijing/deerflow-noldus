"""4 个 first-party tool — subagent 调用本 tool 结构化 seal handoff 到 workspace。

设计原则（grill 锁定 Sprint 0）：
1. LLM 只填 tool 参数（LangChain tool_call schema 自动校验类型/必填）
2. tool 内部 Pydantic 校验 + atomic write + .lineage/manifest.json 记录
3. 4 个 tool 共享 _seal_handoff helper，避免重复
4. 调用方:
    - code-executor → seal_code_executor_handoff
    - data-analyst → seal_data_analyst_handoff
    - chart-maker → seal_chart_maker_handoff
    - report-writer → seal_report_writer_handoff
"""

from __future__ import annotations

import base64
import hashlib
import html.parser
import json
import logging
import os
import re
import threading
import weakref
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from langchain.tools import tool
from pydantic import ValidationError

from deerflow.subagents.handoff_schemas import (
    ChartMakerHandoff,
    CodeExecutorHandoff,
    DataAnalystHandoff,
    ReportWriterHandoff,
)
from deerflow.tools.types import Runtime

logger = logging.getLogger(__name__)


# ============================================================================
# Sprint 6: experiment_summary memory fact
# ============================================================================


def _extract_n_per_group(workspace: Path) -> str:
    """Read n_per_group from handoff_code_executor.json (deterministic, no LLM).

    Priority: metadata.n_per_group > first metrics_summary group's n > "unknown".
    """
    ce_path = workspace / "handoff_code_executor.json"
    if not ce_path.exists():
        return "unknown"
    try:
        ce = json.loads(ce_path.read_text(encoding="utf-8"))
        # Priority 1: explicit metadata field
        meta = ce.get("metadata")
        if isinstance(meta, dict) and "n_per_group" in meta:
            return str(meta["n_per_group"])
        # Priority 2: scan metrics_summary for first group's first metric's n
        ms = ce.get("metrics_summary")
        if isinstance(ms, dict):
            for _group, metrics in ms.items():
                if isinstance(metrics, dict):
                    for _metric, stats in metrics.items():
                        if isinstance(stats, dict) and "n" in stats:
                            return str(stats["n"])
        return "unknown"
    except Exception:
        return "unknown"


def _extract_key_findings_count(workspace: Path) -> int:
    """Read key_findings count from handoff_data_analyst.json."""
    da_path = workspace / "handoff_data_analyst.json"
    if not da_path.exists():
        return 0
    try:
        da = json.loads(da_path.read_text(encoding="utf-8"))
        return len(da.get("key_findings", []))
    except Exception:
        return 0


def _write_experiment_summary_memory(
    workspace: Path,
    paradigm: str,
    config_id: str,
    thread_id: str,
    user_id: str | None,
) -> None:
    """Write an experiment_summary fact to memory (deterministic, no LLM).

    Non-fatal: all exceptions are caught and logged as warnings.
    """
    try:
        from deerflow.agents.memory.updater import create_memory_fact

        n_per_group = _extract_n_per_group(workspace)
        key_findings_count = _extract_key_findings_count(workspace)

        # Lineage (thread_id/config_id) is folded into content because
        # create_memory_fact() hardcodes source="manual" and does not accept
        # a source kwarg — keeping it in content preserves traceability.
        content = (
            f"{paradigm} analysis on {datetime.now(UTC).strftime('%Y-%m-%d')}: "
            f"n_per_group={n_per_group}; "
            f"key_findings_count={key_findings_count}; "
            f"analysis_config_id={config_id}; "
            f"thread={thread_id}"
        )
        create_memory_fact(
            content=content,
            category="experiment_summary",
            confidence=1.0,
            user_id=user_id,
        )
        logger.info("experiment_summary fact written for config_id=%s", config_id)
    except Exception as e:
        logger.warning("Failed to write experiment_summary memory fact: %s", e)


# ============================================================================
# 内部 helper
# ============================================================================


def _resolve_workspace(runtime: Runtime) -> Path:
    """从 runtime state 取 host-side workspace 路径。"""
    state = runtime.state
    if not isinstance(state, dict):
        raise RuntimeError("seal_*_handoff: runtime.state is not a dict")
    thread_data = state.get("thread_data")
    if not isinstance(thread_data, dict):
        raise RuntimeError("seal_*_handoff: thread_data missing from state")
    workspace_path = thread_data.get("workspace_path")
    if not workspace_path:
        raise RuntimeError("seal_*_handoff: workspace_path missing")
    return Path(workspace_path)


# Regex matching markdown image paths that do NOT use the canonical virtual
# prefix. Covers three LLM-written variants that must be normalised to canonical:
#   (outputs/file.png)             — relative, no prefix
#   (mnt/user-data/outputs/…)     — correct prefix but missing the leading slash
#   (/mnt/user-data/outputs/…)    — already canonical; matched idempotently
# The optional ``/?mnt/user-data/`` prefix absorbs both slash-less and slashed
# forms; the literal ``outputs/`` then matches; capture group 1 = the file name.
_BAD_IMG_PATH_RE = re.compile(
    r"\((?:/?mnt/user-data/)?outputs/([^)]+\.(?:png|jpg|jpeg|svg|gif|webp))\)"
)

# Regex matching {{img:<basename>}} placeholders in report.md.
# Layer 1 of the chart image placeholder resolution system: LLM writes
#   ![Figure 1]({{img:plot_trajectory_s0.png}})
# and seal_report_writer_handoff resolves it to the canonical virtual path
# from handoff_chart_maker.json.chart_files.
_IMG_PLACEHOLDER_RE = re.compile(r"\{\{img:([^}]+)\}\}")


# ============================================================================
# report 图片路径规范形态（SSOT）—— 2026-06-18
# spec: docs/superpowers/specs/2026-06-18-report-image-path-ssot-spec.md
#
# report.md 内图片路径的**唯一规范形态**是带前导斜杠的虚拟绝对路径：
#   /mnt/user-data/outputs/<name>.<ext>
#
# 前导斜杠让前端 ``src.startsWith("/mnt/")`` 判断稳定命中；前端把它原样
# 交给 artifact API（``/api/threads/{tid}/artifacts/mnt/user-data/outputs/…``），
# 后端 ``resolve_virtual_path`` 内部 ``lstrip("/")`` 后命中 ``mnt/user-data/`` 前缀。
#
# 这是 report 图片路径规范形态的**唯一定义点**——seal 的两个产出点
# (placeholder 解析 + path normalize) 都调本函数，保证字节一致（SSOT 铁律）。
# ============================================================================
_CANONICAL_PREFIX = "/mnt/user-data/outputs/"


def _to_canonical_artifact_path(name: str) -> str:
    """Return the canonical report-image path for *name*.

    规范形态 = ``/mnt/user-data/outputs/<name>``（带前导斜杠）。
    对已是规范形态（或带多余前导斜杠）的输入幂等归一，不会二次加前缀。

    Args:
        name: 图片文件名（basename 或已是 ``/mnt/user-data/outputs/…`` 全路径）。

    Returns:
        带前导斜杠的规范虚拟绝对路径。
    """
    stripped = name.lstrip("/")
    if stripped.startswith("mnt/user-data/outputs/"):
        return f"/{stripped}"
    return f"{_CANONICAL_PREFIX}{stripped}"


def _load_chart_files_map(workspace: Path) -> dict[str, str]:
    """Return {basename: canonical_virtual_path} from handoff_chart_maker.json.

    Returns empty dict when file absent, unparseable, or chart_files empty.
    The value is the **canonical** form (leading slash) via
    ``_to_canonical_artifact_path`` — the SSOT for report image paths.
    """
    chart_handoff = workspace / "handoff_chart_maker.json"
    if not chart_handoff.exists():
        return {}
    try:
        data = json.loads(chart_handoff.read_text(encoding="utf-8"))
        chart_files = data.get("chart_files", [])
        if not isinstance(chart_files, list):
            return {}
        result: dict[str, str] = {}
        for f in chart_files:
            if isinstance(f, str):
                result[Path(f).name] = _to_canonical_artifact_path(Path(f).name)
        return result
    except Exception:
        return {}


def _resolve_report_image_placeholders(
    report_host_path: Path,
    workspace: Path,
) -> None:
    """Resolve {{img:<basename>}} placeholders in report.md.

    Reads handoff_chart_maker.json from workspace, builds a {basename: full_path}
    mapping, then replaces every {{img:<basename>}} placeholder in the report
    with the canonical virtual path.

    Unmatched basenames → replaced with visible error stub listing available files.
    Missing/empty chart_files → no-op (placeholders survive for human diagnosis).
    """
    if not report_host_path.is_file():
        return

    chart_files_map = _load_chart_files_map(workspace)

    try:
        original = report_host_path.read_text(encoding="utf-8")

        def _replace(match: re.Match[str]) -> str:
            basename = match.group(1).strip()
            if not chart_files_map:
                return match.group(0)          # 无映射时保留原样
            if basename in chart_files_map:
                return chart_files_map[basename]
            # 不匹配 → 可见错误文本
            available = ", ".join(
                sorted(chart_files_map.keys())[:5]
            )
            suffix = f"；可用: {available}" if available else ""
            return f"[图表 '{basename}' 未找到{suffix}]"

        resolved = _IMG_PLACEHOLDER_RE.sub(_replace, original)

        if resolved != original:
            report_host_path.write_text(resolved, encoding="utf-8")
            logger.info(
                "seal_report_writer_handoff: resolved image placeholders in %s",
                report_host_path,
            )
    except Exception:
        logger.warning(
            "seal_report_writer_handoff: image placeholder resolution skipped",
            exc_info=True,
        )


# ============================================================================
# report.html —— HTML 模式：占位符 base64 内联 + XSS 消毒（spec 2026-06-29）
#
# 报告载体从 Markdown 改 HTML 后，seal 时对 .html 报告新增两层确定性处理：
#   1. ``{{img:<basename>}}`` → 读对应 .png → base64 → 内联
#      ``<img src="data:image/png;base64,...">``。只内联代表性图（prompt 已指导
#      report-writer 只放 1-3 张），自包含、下载离线可看，治「下载丢图」。
#   2. ``sanitize_report_html`` 剥 ``<script>`` / 内联 ``on*`` 事件 / ``<iframe>``
#      等——HTML 来自 LLM 产出，封存时做一层确定性消毒（前端二次 sanitize 为兜底）。
#
# 纯 stdlib（``html.parser``）实现，不引新依赖、不增顶层 import（守导入环铁律）。
# ============================================================================


class _ReportHtmlSanitizer(html.parser.HTMLParser):
    """确定性 HTML 消毒器：剥危险标签/属性，保留结构与 data URI 图。

    - 删除整段危险标签及其内容：``<script>`` / ``<style>`` / ``<iframe>`` /
      ``<object>`` / ``<embed>`` / ``<link>`` / ``<meta>`` / ``<base>``。
      （``<style>`` 也能藏 expression()/@import 攻击，一并剥；报告样式由前端提供。）
    - 剥所有 ``on*`` 内联事件属性（含 ``on click`` 这类带空白/制表的混淆形式），
      以及 ``javascript:`` 伪协议的 href/src。
    - 其余标签/属性原样保留（含 ``<img src="data:image/png;base64,...">``）。

    注：``convert_charrefs=True``（默认）已把文本实体转回字符再交付 handle_data，
    避免实体拆分绕过。本消毒器面向 report-writer LLM 产出（非任意网页），配合前端
    DOMPurify 二次 sanitize，构成纵深防御。
    """

    # 标签名小写。这些标签**整段连同内容删除**（script/style 可执行，title 不该进正文，
    # 其余远程加载风险）。
    _DROP_TAGS_WITH_CONTENT = frozenset({
        "script",
        "style",
        "iframe",
        "object",
        "embed",
        "noscript",
        "template",
        # title 整段删除（连内容）：LLM 常产 <head><title>报告标题</title></head>，
        # 若只剥标签留内容，标题文字会裸露在 <head>，前端解析时落入 body → 报告顶部
        # 冒出一行裸标题（dogfood thread 73b41dc3 证实）。标题本就不该进正文渲染区。
        "title",
    })
    # 这些标签删除标签本身但**保留内容**（meta/link/base 在 head 内、报告用不到）。
    _STRIP_TAGS_KEEP_CONTENT = frozenset({"link", "meta", "base"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._out: list[str] = []
        self._drop_depth = 0  # >0 表示当前在某 _DROP_TAGS_WITH_CONTENT 内，跳过输出

    @property
    def result(self) -> str:
        return "".join(self._out)

    @staticmethod
    def _is_event_attr(name: str) -> bool:
        """``on*`` 事件属性判定，容忍 ``on click`` / ``on\terror`` 等混淆形式。

        规范化：去空白/制表后看是否以 ``on`` 开头且长度 > 2（on + 至少 1 字符）。
        """
        compact = name.strip().replace(" ", "").replace("\t", "").lower()
        return compact.startswith("on") and len(compact) > 2

    @staticmethod
    def _has_script_protocol(value: str) -> bool:
        """``javascript:`` / ``vbscript:`` / ``data:text/html`` 伪协议判定。"""
        compact = value.strip().replace(" ", "").lower()
        return compact.startswith(("javascript:", "vbscript:")) or compact.startswith("data:text/html")

    def _filter_attrs(self, attrs: list[tuple[str, str | None]]) -> list[tuple[str, str | None]]:
        kept: list[tuple[str, str | None]] = []
        for name, value in attrs:
            if self._is_event_attr(name):
                continue
            if name in ("href", "src", "xlink:href") and value is not None and self._has_script_protocol(value):
                continue
            kept.append((name, value))
        return kept

    @staticmethod
    def _format_attrs(attrs: list[tuple[str, str | None]]) -> str:
        parts: list[str] = []
        for name, value in attrs:
            if value is None:
                parts.append(f" {name}")
            else:
                escaped = value.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")
                parts.append(f' {name}="{escaped}"')
        return "".join(parts)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        low = tag.lower()
        if low in self._DROP_TAGS_WITH_CONTENT:
            self._drop_depth += 1
            return
        if low in self._STRIP_TAGS_KEEP_CONTENT:
            return  # 剥标签、留内容（内容经 handle_data 输出）
        filtered = self._filter_attrs(attrs)
        self._out.append(f"<{low}{self._format_attrs(filtered)}>")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """自闭合标签（``<img ... />``）——同样过滤属性，不进 drop 栈。"""
        low = tag.lower()
        if low in self._DROP_TAGS_WITH_CONTENT or low in self._STRIP_TAGS_KEEP_CONTENT:
            return
        filtered = self._filter_attrs(attrs)
        self._out.append(f"<{low}{self._format_attrs(filtered)} />")

    def handle_endtag(self, tag: str) -> None:
        low = tag.lower()
        if low in self._DROP_TAGS_WITH_CONTENT:
            if self._drop_depth > 0:
                self._drop_depth -= 1
            return
        if low in self._STRIP_TAGS_KEEP_CONTENT:
            return
        self._out.append(f"</{low}>")

    def handle_data(self, data: str) -> None:
        if self._drop_depth > 0:
            return
        self._out.append(data)


def sanitize_report_html(html_text: str) -> str:
    """对 report-writer 产出的 HTML 做一层确定性 XSS 消毒。

    剥 ``<script>`` / 内联 ``on*`` 事件 / ``<iframe>`` 等可执行或远程加载内容，
    保留结构化标签与 ``data:`` URI 图。返回消毒后的 HTML 字符串。

    纯 stdlib、无新依赖。前端会对返回结果再过一遍 DOMPurify（纵深防御）。
    """
    parser = _ReportHtmlSanitizer()
    parser.feed(html_text)
    parser.close()
    return parser.result


def _resolve_html_report_image_placeholders(
    report_host_path: Path,
    workspace: Path,
) -> None:
    """Resolve ``{{img:<basename>}}`` placeholders in report.html → base64-inline.

    与 markdown 模式的 ``_resolve_report_image_placeholders`` 对称。占位符契约：
    prompt 让 LLM 把占位符写进 ``<img src="{{img:<basename>}}">``——占位符是 src 的
    **值**，因此这里只产 data URI 字符串（或空串降级），替换后得
    ``<img src="data:image/png;base64,...">``，自包含、下载离线即可看图（治「下载丢图」）。

    - 命中且磁盘 .png 存在 → base64 内联 data URI 值。
    - 命中但磁盘文件缺失，或 basename 不在 chart_files → 降级为空串（``<img src="">``，
      前端加载失败显示 alt），诊断进日志。绝不产伪 data URI、绝不产文本 stub 塞进 src
      （后者经 sanitize 会残留裸文本/坏 src——2026-06-29 嵌套 ``<img>`` 回归根因）。
    - chart_files 为空/不可读 → 保留占位符原样（人工诊断）。

    report 文件不存在/不可读 → 静默跳过（seal 仍成功）。
    """
    if not report_host_path.is_file():
        return

    chart_files_map = _load_chart_files_map(workspace)
    outputs_dir = _outputs_dir_for(workspace)

    try:
        original = report_host_path.read_text(encoding="utf-8")
    except Exception:
        logger.warning(
            "seal_report_writer_handoff: report.html unreadable, skip base64 inline",
            exc_info=True,
        )
        return

    def _replace(match: re.Match[str]) -> str:
        basename = match.group(1).strip()
        # 占位符契约：prompt 让 LLM 写 ``<img src="{{img:X}}">``——占位符是 src 的
        # **值**，因此这里只产 data URI 字符串（或空串降级），绝不能产整个
        # ``<img>`` 元素，否则得到嵌套 ``<img src="<img ...">">``，经
        # sanitize_report_html 后 ``/`` 被拆成空格、图全废（#234 回归根因）。
        if not chart_files_map:
            return match.group(0)  # 无映射保留原样（人工诊断）
        if basename not in chart_files_map:
            available = ", ".join(sorted(chart_files_map.keys())[:5])
            suffix = f"；可用: {available}" if available else ""
            logger.warning(
                "seal_report_writer_handoff: chart %r not in chart_files%s, "
                "degrade img to empty src (alt shown)",
                basename,
                suffix,
            )
            return ""  # <img src=""> 前端显示 alt，不破坏 HTML
        # 命中 → 读磁盘 .png → base64 内联
        png_path = outputs_dir / basename
        try:
            raw = png_path.read_bytes()
        except Exception:
            logger.warning(
                "seal_report_writer_handoff: chart png missing on disk for %r, "
                "degrade img to empty src (alt shown)",
                basename,
            )
            return ""  # <img src=""> 前端显示 alt，不破坏 HTML
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:image/png;base64,{b64}"

    try:
        resolved = _IMG_PLACEHOLDER_RE.sub(_replace, original)
    except Exception:
        logger.warning(
            "seal_report_writer_handoff: html base64 inline substitution skipped",
            exc_info=True,
        )
        return

    if resolved != original:
        try:
            report_host_path.write_text(resolved, encoding="utf-8")
            logger.info(
                "seal_report_writer_handoff: base64-inlined image placeholders in %s",
                report_host_path,
            )
        except Exception:
            logger.warning(
                "seal_report_writer_handoff: report.html rewrite failed after base64 inline",
                exc_info=True,
            )


def _normalize_report_image_paths(report_host_path: Path) -> None:
    """Rewrite image paths in report.md to the canonical form (SSOT).

    规范形态 = 带前导斜杠的虚拟绝对路径 ``/mnt/user-data/outputs/<name>.<ext>``
    （见 ``_to_canonical_artifact_path``）。LLM 常写相对形态 ``outputs/file.png``
    或缺前导斜杠的 ``mnt/user-data/outputs/file.png``，两者前端
    ``startsWith("/mnt/")`` 都不命中 → 404。本函数把它们统一到规范形态，
    这样无论 LLM 写成什么，结果都正确（前端只认这一种）。

    幂等：已是规范形态的路径不变。report 文件不存在时静默跳过。
    """
    if not report_host_path.is_file():
        return
    try:
        original = report_host_path.read_text(encoding="utf-8")
        # 替换目标用 canonical helper 拼接，与 _load_chart_files_map 字节一致（SSOT）。
        normalised = _BAD_IMG_PATH_RE.sub(
            lambda m: f"({_to_canonical_artifact_path(m.group(1))})",
            original,
        )
        if normalised != original:
            report_host_path.write_text(normalised, encoding="utf-8")
            logger.info(
                "seal_report_writer_handoff: normalised image paths in %s",
                report_host_path,
            )
    except Exception as e:
        # Non-fatal: log and continue; the seal still succeeds.
        logger.warning("seal_report_writer_handoff: image path normalisation skipped: %s", e)


def _read_analysis_config_id(workspace: Path) -> str:
    """从 experiment-context.json 读 analysis_config_id (Sprint 4.5 填)。

    Sprint 0 阶段: experiment-context.json 可能还没有此字段，返回 "PENDING_SPRINT_4.5"
    占位，Sprint 4.5 实施后会自动正常填入。
    """
    ctx_path = workspace / "experiment-context.json"
    if not ctx_path.exists():
        return "PENDING_SPRINT_4.5"
    try:
        ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
        return ctx.get("analysis_config_id", "PENDING_SPRINT_4.5")
    except Exception as e:
        logger.warning("read experiment-context.json failed: %s", e)
        return "PENDING_SPRINT_4.5"


def _update_manifest(workspace: Path, handoff_filename: str, sha256: str, analysis_config_id: str) -> None:
    """写 .lineage/manifest.json。

    Sprint 5.5 会进一步用本 manifest 做下游 hash 校验；Sprint 0 只负责写。
    """
    lineage_dir = workspace / ".lineage"
    lineage_dir.mkdir(exist_ok=True)
    manifest_path = lineage_dir / "manifest.json"

    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                manifest = {}
        except Exception:
            manifest = {}
    else:
        manifest = {}

    manifest[handoff_filename] = {
        "sha256": sha256,
        "analysis_config_id": analysis_config_id,
        "timestamp": datetime.now(UTC).isoformat(),
    }

    # atomic write for manifest itself
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(manifest_path)


def _build_task_context(payload: dict[str, Any]) -> dict[str, Any]:
    """从 handoff payload 已有字段确定性组装 task_context 的 4 个字段。

    纯函数、无 LLM、无副作用、任何异常返回部分结果（防御性）。
    """
    tc: dict[str, Any] = {
        "file_changes": [],
        "verify_commands": [],
        "failed_paths": [],
        "pending_items": [],
    }
    try:
        # file_changes: 从 output_files 的 value（路径）提取
        output_files = payload.get("output_files") or {}
        paths: list[str] = []
        for v in output_files.values():
            if isinstance(v, str):
                paths.append(v)
            elif isinstance(v, list):
                paths.extend(p for p in v if isinstance(p, str))
        tc["file_changes"] = paths

        # verify_commands: 对每个产物文件生成存在性 + JSON 校验命令（模板）
        cmds: list[str] = []
        for p in paths:
            if p.endswith(".json"):
                cmds.append(f"python -m json.tool {p} > /dev/null")
            else:
                cmds.append(f"ls {p}")
        tc["verify_commands"] = cmds

        # failed_paths: 从 errors 提取（errors 是产出方记录的失败事实）
        errors = payload.get("errors") or []
        tc["failed_paths"] = [e for e in errors if isinstance(e, str)]

        # pending_items: 暂留空。
        # 真实 partial 语义是"指标已算完、但统计检验因样本量(n=1/n=2)被跳过"，
        # 而非"指标未算完"——partial 的原因已由 gate_signals.statistical_validity="skipped"
        # + data_quality_warnings 充分表达，lead 据此决策无需本字段。
        # 当前无可靠的"未完成明细"数据源（errors 在此类 partial 时恒空）。
        # TODO: 若未来出现"指标脚本失败导致 partial"的真实场景，再从
        #   plan_metrics.json(计划) vs metrics_summary(实际) 的差集派生。
        # 不从 errors 派生（恒空且误导）。
    except Exception:
        pass  # 防御性：组装失败不影响 seal 主流程
    return tc


# ============================================================================
# chart-maker 封存对账 —— spec 2026-06-22
#   根因 A：failed_charts[].reason 是 LLM free-text，可伪造（prod 实测对 plan 里
#     没 skip 的 box_open_arm 编 "missing columns"，被 plan_charts.json.skipped=[]
#     证伪）。用 plan 的 skipped[] 机读真相订正。
#   根因 B：status=completed 但 plan 内 aggregate 图未全部落盘 → 现有 validator
#     只看 chart_files 非空（trajectory 在），放行了。对账 plan 的 aggregate 图
#     集合 ⊆ outputs/ 实际 png，缺口非空且 completed → 响亮拒绝。
# 单一注入点：_seal_handoff_to_workspace 在 model_cls is ChartMakerHandoff 时调用，
#   覆盖 seal_chart_maker_handoff 工具与 executor auto-seal 兜底两条路径。
# ============================================================================


def _load_plan_charts(workspace: Path) -> dict[str, Any] | None:
    """读 {workspace}/plan_charts.json；缺文件/不可解析返回 None（不 crash）。"""
    plan_path = workspace / "plan_charts.json"
    if not plan_path.exists():
        return None
    try:
        data = json.loads(plan_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        logger.warning("chart-maker seal: plan_charts.json unreadable, skip reconciliation")
    return None


def _outputs_dir_for(workspace: Path) -> Path:
    """outputs/ 是 workspace 的兄弟目录（同 _resolve_workspace 推导）。"""
    return workspace.parent / "outputs"


def _chart_file_exists_on_disk(virtual_path: str, outputs_dir: Path) -> bool:
    """把虚拟路径 /mnt/user-data/outputs/<name> resolve 成物理 outputs_dir/<name>，
    确定性 exists() 核磁盘。与 2.2 门 rendered 集同源（basename 拼接）。

    前缀不符（schema 已拦，防御性）或 basename 为空 → 视为不存在（剔进 remaining）。
    """
    basename = virtual_path.rsplit("/", 1)[-1]
    if not basename:
        return False
    return (outputs_dir / basename).exists()


def _reconcile_chart_files_against_disk(
    payload: dict[str, Any],
    workspace: Path,
) -> None:
    """产物真实性核对（spec 2026-06-24 ETHO-10）。

    chart_files 里每条虚拟路径 resolve 成物理路径、确定性 exists() 核磁盘：
      - 存在 → 留在 chart_files；
      - 不存在 → 从 chart_files 剔除，挪进 remaining_charts 留痕
        （语义：声称要画但未落盘，留痕供用户再要）。

    这是对所有 status 生效的通用不变式（partial/failed 的幻影路径也会致下游 404）。
    不靠 LLM 自报，磁盘是唯一真相——与 present_file_tool 守同一不变式。

    鲁棒性：outputs 读盘异常（路径不是目录、权限问题等）→ warning + 跳过核对、
    原样保留 chart_files（绝不 crash、绝不误剔真图）。注意「completed 且 chart_files
    全空」的响亮 ValueError 不在此函数内（那是上层 2.2 段的有意拒绝，与读盘异常无关）。
    """
    chart_files = payload.get("chart_files") or []
    if not chart_files:
        return  # 空集无幻影可剔；completed 空集的拒绝由上层 2.2 段处理。

    outputs_dir = _outputs_dir_for(workspace)
    try:
        kept: list[str] = []
        purged: list[dict[str, str]] = []
        for virtual_path in chart_files:
            if not isinstance(virtual_path, str) or not virtual_path:
                continue
            try:
                real = _chart_file_exists_on_disk(virtual_path, outputs_dir)
            except OSError as e:
                # 单条路径核盘异常（如 outputs 是文件）：保守不剔（绝不误剔真图）。
                logger.warning(
                    "chart-maker seal: chart_files reality check OSError on %r (%s); "
                    "keep as-is (never purge on read error)",
                    virtual_path,
                    e,
                )
                real = True  # 保守保留
            if real:
                kept.append(virtual_path)
            else:
                basename = virtual_path.rsplit("/", 1)[-1] or virtual_path
                purged.append(
                    {
                        "chart_id": basename,
                        "reason": (
                            "claimed in chart_files but not rendered on disk "
                            "(likely loop-detection hard-stop / max_turns early-exit)"
                        ),
                    }
                )
    except (OSError, NotADirectoryError) as e:
        # outputs 整体读盘异常 → warning + 跳过核对、原样封存（绝不 crash）。
        logger.warning(
            "chart-maker seal: outputs dir reality check unreadable (%s); "
            "skip reality reconciliation, keep chart_files as-is",
            e,
        )
        return

    if purged:
        logger.warning(
            "chart-maker seal: 产物真实性核对剔除 %d 条幻影路径（磁盘不存在）"
            "进 remaining_charts（sealed_by 可观测：reward hacking 治理触发率）",
            len(purged),
        )
        payload["chart_files"] = kept
        existing_remaining = payload.get("remaining_charts") or []
        if not isinstance(existing_remaining, list):
            existing_remaining = []
        # 去重：同 chart_id 不重复进 remaining（保守合并）
        seen = {
            r.get("chart_id")
            for r in existing_remaining
            if isinstance(r, dict)
        }
        for p in purged:
            if p["chart_id"] not in seen:
                existing_remaining.append(p)
                seen.add(p["chart_id"])
        payload["remaining_charts"] = existing_remaining


def _reconcile_chart_maker_payload(
    payload: dict[str, Any],
    workspace: Path,
) -> None:
    """封存前对 chart-maker payload 做确定性对账（spec 2026-06-22 + 2026-06-24 ETHO-10）。

    机读真相来源：plan_charts.json（resolver 真实 skip 列表 + 计划要画的图）+
    outputs/ 文件系统（实际落盘的 png）。LLM 自述的 failed_charts[].reason 不可信，
    LLM 自报的 chart_files 路径也不可信（2026-06-24：磁盘是唯一真相）。

    2.0（产物真实性不变式，spec 2026-06-24 ETHO-10）：chart_files 每条虚拟路径
        resolve 成物理路径、确定性 exists() 核磁盘——存在的留，不存在的剔除挪进
        remaining_charts 留痕。对所有 status 生效（partial 的幻影也致 404）。与
        present_file_tool 守同一不变式「声称的产物文件必须真存在」。剔除率记 warning
        可观测（reward hacking 治理触发率）。

    2.1（订正伪造 reason）：对每条 failed_charts，
        - chart_id 在 plan.skipped[] → 用 plan 的真实 detail 覆盖 LLM 自述；
        - chart_id 不在 plan.skipped[] → resolver 没 skip 它，这是「执行/未执行」
          失败，reason 规整为机读形态 "resolved in plan but not rendered"，LLM
          原文作为引用保留（绝不保留它编的 "missing columns" 当权威 reason）。

    2.2（堵漏执行）：planned_aggregate（plan 内 output_mode=="aggregate" 的图
        basename 集）- rendered（outputs/ 实际 *.png basename 集）= missing。
        status==completed 且 missing 非空 → 抛 ValueError（响亮拒绝，让 chart-maker
        补画；与 _completed_requires_core_output 同风格）。
        另：2.0 真实性核对后若 completed 且 chart_files 为空（核心图一张没真画）
        → 抛 ValueError（堵 `if not planned_aggregate: return` 的 plan-无-aggregate
        放行漏洞，不依赖 plan 有没有 aggregate）。

    per_subject 图（被 chart_budget 截断的）不进 2.2 aggregate 门：只对账 aggregate。

    鲁棒性：plan_charts.json 不存在 → 跳过 2.1/2.2、原样封存（但 2.0 真实性核对仍生效，
    因它只看 outputs/ 磁盘不依赖 plan）。任何读盘异常 → warning + 跳过该段，绝不让 seal
    因读不到 plan/outputs 而 crash。

    Args:
        payload: 待封存的 chart-maker handoff payload（in-place 修订 failed_charts
            / chart_files / remaining_charts）。
        workspace: host-side workspace 目录。

    Raises:
        ValueError: status==completed 且（核磁盘后 chart_files 全空，或 plan 内
            aggregate 图有未落盘者）。
    """
    # --- 2.0：产物真实性核对（spec 2026-06-24 ETHO-10，磁盘是唯一真相）---
    # 先于一切 plan 依赖逻辑：真实性核对只读 outputs/ 磁盘，不需要 plan。
    # 即使 plan_charts.json 不存在（下面早退），2.0 仍生效（spec 六：2.0 不依赖 plan）。
    _reconcile_chart_files_against_disk(payload, workspace)

    plan = _load_plan_charts(workspace)
    if plan is None:
        # 极端早退：无 plan 可对账 → 不订正 reason、不做 aggregate 完整性校验。
        # 但 2.0 真实性核对已跑（剔除幻影路径）；completed 空集拒绝仍须守。
        if payload.get("status") == "completed" and not payload.get("chart_files"):
            raise ValueError(
                "status='completed' but chart_files is empty after disk reality check "
                "(and no plan_charts.json to reconcile). outputs/ has no real png "
                "matching any claimed path. 补画真实图再 completed，或诚实改 "
                "status=partial/failed。绝不留幻影路径（下游 present 时会 404）。"
            )
        return

    # --- 2.1：用 plan.skipped[] 订正 failed_charts[].reason ---
    skipped_by_id: dict[str, str] = {}
    for s in plan.get("skipped") or []:
        if isinstance(s, dict) and s.get("id"):
            # detail 比 reason 信息更全（含缺失列名等）；回落 reason。
            skipped_by_id[str(s["id"])] = str(
                s.get("detail") or s.get("reason") or "skipped by resolver"
            )

    failed_charts = payload.get("failed_charts") or []
    reconciled: list[dict[str, str]] = []
    for fc in failed_charts:
        if not isinstance(fc, dict):
            # 防御性：非 dict 条目（schema 不会产，但 extra="allow" 下理论可能）规整化。
            reconciled.append({"chart_id": "?", "reason": str(fc)})
            continue
        chart_id = str(fc.get("chart_id", ""))
        llm_reason = str(fc.get("reason", ""))
        if chart_id in skipped_by_id:
            # plan 证实的真 skip → 用机读真相覆盖 LLM 自述。
            new_reason = skipped_by_id[chart_id]
        else:
            # plan 里没 skip 它 → 是「执行/未执行」失败，不是「resolve 失败」。
            # 规整为机读形态，LLM 原文作引用（堵 "missing columns" 类脑补当权威）。
            new_reason = (
                f"resolved in plan but not rendered before seal "
                f"(likely max_turns/early-exit); chart-maker note: {llm_reason!r}"
            )
            logger.warning(
                "chart-maker seal: failed_charts[%r] reason订正——plan.skipped=[] "
                "未含此 id，LLM 自述 %r 疑似伪造，规整为机读形态",
                chart_id,
                llm_reason,
            )
        reconciled.append({"chart_id": chart_id, "reason": new_reason})
    payload["failed_charts"] = reconciled

    # --- 2.2：completed 时对账 plan 内 aggregate 图全部落盘 ---
    # 改动 2（spec 2026-06-24）：2.0 真实性核对已剔除幻影路径（见函数顶部，先于 plan）。
    # 剔除后若 completed 且 chart_files 为空（核心图一张没真画）→ 抛 ValueError。
    # 这条不依赖 plan 有没有 aggregate，直接堵死下方 `if not planned_aggregate: return`
    # 的 plan-无-aggregate 放行漏洞（2026-06-24 dogfood：57 张全 per_subject + outputs
    # 0 png 仍标 completed）。
    if payload.get("status") != "completed":
        return
    if not payload.get("chart_files"):
        raise ValueError(
            "status='completed' but chart_files is empty after disk reality check — "
            "every claimed output path is missing on disk (outputs/ has 0 real png). "
            "Likely chart-maker hit a loop-detection hard-stop / max_turns early-exit "
            "before rendering anything. 补画真实图再 completed，或诚实改 status=partial/"
            "failed 并在 failed_charts/remaining_charts 留痕。绝不留幻影路径（下游 "
            "present 时会 404）。"
        )

    planned_aggregate: set[str] = set()
    for c in plan.get("charts") or []:
        if not isinstance(c, dict):
            continue
        # output_mode=="aggregate" 是组间对比 must_have（box/bar），必须画。
        # per_subject 图被 chart_budget 截断是合法的（remaining_charts 指纹），豁免。
        if c.get("output_mode") == "aggregate":
            output = c.get("output")
            if isinstance(output, str) and output:
                planned_aggregate.add(Path(output).name)

    if not planned_aggregate:
        return  # plan 里没 aggregate 图 → 无完整性约束可对账。

    outputs_dir = _outputs_dir_for(workspace)
    rendered: set[str] = set()
    if outputs_dir.exists():
        rendered = {p.name for p in outputs_dir.glob("*.png")}

    missing_aggregate = planned_aggregate - rendered
    if missing_aggregate:
        missing_sorted = sorted(missing_aggregate)
        raise ValueError(
            f"status='completed' but plan-charts aggregate 图未全部落盘: "
            f"{missing_sorted}. 这些是组间对比 must_have 图（output_mode=aggregate），"
            f"plan_charts.json 声明了但 outputs/ 里没有对应 png——把缺的 aggregate "
            f"图画完再 completed，或确实失败则 status=partial 并在 failed_charts "
            f"写真实原因。注意 per_subject 截断不在此门内（remaining_charts 留痕）。"
        )


def _seal_handoff_to_workspace(
    model_cls: type,
    filename: str,
    payload: dict[str, Any],
    workspace: Path,
    *,
    force: bool = False,
) -> str:
    """Pure-function variant of _seal_handoff: Pydantic validate → atomic write → manifest.

    Takes an explicit workspace ``Path`` instead of resolving it from ``Runtime``.
    Used by harness-level auto-seal where no Runtime is available (Spec C).

    Same contract as _seal_handoff: all failure paths raise ValueError.

    Args:
      force: 强制覆盖已有确定性封存（spec 2026-06-25 M1）。仅 ``run_chart_plan`` 工具
        （确定性来源，重跑全量是合法覆盖场景）传 ``force=True``。``seal_chart_maker_handoff``
        （LLM 手调）与 auto-seal 都不传 force → 走默认 ``False`` → 撞已有 ``run_plan`` 封存
        即被拒。守「LLM 提议，确定性门定生死」——LLM 无法绕过。
    """
    # 1. 注入 analysis_config_id (subagent 不用手动传)
    payload.setdefault("analysis_config_id", _read_analysis_config_id(workspace))

    # 1.4. chart-maker「封存只允许一次」不变式（spec 2026-06-25 M1 / R1）：
    #      磁盘已有 ``sealed_by=run_plan`` 封存时，拒绝任何**非 force** chart-maker seal。
    #      治 double-seal 覆盖：chart-maker 手调 ``seal_chart_maker_handoff``（或 auto-seal）
    #      会把 ``run_chart_plan`` 的确定性封存（sealed_by=run_plan）覆盖成 LLM 自报
    #      （sealed_by=model）。``run_chart_plan`` 是唯一传 force=True 的合法覆盖场景。
    #      放在 reconcile / 校验 / atomic write 之前——拒绝时 manifest 不被污染。
    if model_cls is ChartMakerHandoff and not force:
        existing = workspace / filename
        prev_sealed_by: str | None = None
        if existing.exists():
            try:
                prev = json.loads(existing.read_text(encoding="utf-8"))
                prev_sealed_by = prev.get("sealed_by") if isinstance(prev, dict) else None
            except Exception:
                prev_sealed_by = None
        if prev_sealed_by == "run_plan":
            raise ValueError(
                f"seal_{filename}: 已存在确定性封存 (sealed_by=run_plan)，拒绝覆盖。"
                f"run_chart_plan 已确定性封存真相（chart_files 是磁盘真相），"
                f"不要再调 seal_chart_maker_handoff。"
                f"如需重跑，调 run_chart_plan（它带 force=True 合法覆盖）。"
            )

    # 1.5. 自动组装 task_context —— 仅当目标 schema 仍声明该字段时注入。
    # ethoinsight 4 个 handoff 已移除该字段（拆为旁路 lineage，spec 2026-06-18）；
    # 通用 handoff schema 若仍有该字段则保持组装，向前兼容。task_context 是死重量
    # （下游不消费），无条件注入会把主 handoff 顶过 sandbox read_file 50K 截断线。
    if "task_context" in getattr(model_cls, "model_fields", {}):
        payload.setdefault("task_context", _build_task_context(payload))

    # 1.6. chart-maker 封存对账（spec 2026-06-22）：堵伪造 failed_charts reason +
    #      堵 plan 内 aggregate 图漏画却标 completed。仅在 chart-maker 时触发，
    #      不影响其余 3 个 handoff。覆盖 seal 工具与 auto-seal 两条路径（单一注入点）。
    if model_cls is ChartMakerHandoff:
        _reconcile_chart_maker_payload(payload, workspace)

    # 2. Pydantic 校验
    try:
        handoff = model_cls(**payload)
    except ValidationError as e:
        raise ValueError(
            f"seal_{filename}: schema validation failed: {e}. "
            f"Check field names/types against {model_cls.__name__} schema."
        ) from e

    # 3. Atomic write (tmp + rename)
    final_path = workspace / filename
    tmp_path = workspace / f"{filename}.tmp"
    json_bytes = handoff.model_dump_json(indent=2, exclude_none=False).encode("utf-8")
    tmp_path.write_bytes(json_bytes)
    os.rename(tmp_path, final_path)  # POSIX atomic

    # 3.5. chmod 0o644（Spec1 教训：文件权限 — downstream 工具需可读）
    os.chmod(final_path, 0o644)

    # 4. 写 manifest
    sha256 = hashlib.sha256(json_bytes).hexdigest()
    _update_manifest(workspace, filename, sha256, payload["analysis_config_id"])

    return f"OK: sealed {filename} (sha256={sha256[:12]}...)"


def _seal_handoff(
    model_cls: type,
    filename: str,
    payload: dict[str, Any],
    runtime: Runtime,
) -> str:
    """共享 helper: Pydantic 校验 → atomic write → 写 manifest → 返回 OK。

    所有失败路径都返回 ValueError（LangChain 会自动转 error ToolMessage 给 LLM）。
    """
    workspace = _resolve_workspace(runtime)
    return _seal_handoff_to_workspace(model_cls, filename, payload, workspace)


# ============================================================================
# 4 个 first-party tool
# ============================================================================


@tool("seal_code_executor_handoff", parse_docstring=True)
def seal_code_executor_handoff(
    status: str,
    summary: str,
    paradigm: str,
    metrics_summary: dict[str, dict[str, dict[str, Any]]] | None = None,
    per_subject: dict[str, dict[str, Any]] | None = None,
    statistics: dict[str, Any] | None = None,
    output_files: dict[str, Any] | None = None,
    data_quality_warnings: list[dict[str, Any]] | None = None,
    errors: list[str] | None = None,
    confidence: float | None = None,
    ev19_template: str | None = None,
    inputs: dict[str, Any] | None = None,
    gate_signals: dict[str, Any] | None = None,
    runtime: Runtime = None,
) -> str:
    """Code-executor 完成指标计算后，封存 handoff_code_executor.json。

    严禁直接用 write_file 写 handoff_code_executor.json，必须走本 tool。

    Args:
        status: 执行状态: "completed" / "partial" / "failed"
        summary: 一句话总结
        paradigm: 范式名，如 "fst" / "epm"
        metrics_summary: 嵌套 dict: group -> metric -> {mean, std, n, parameters_used, ...}
        per_subject: 每个 subject 的原始数据: {subject_name: {metric: value}}
        statistics: 组间统计检验结果
        output_files: 产物文件路径表
        data_quality_warnings: 警告列表，每条含 severity/code/metric/message/evidence/blocks_downstream
        errors: 错误信息列表
        confidence: 整体置信度 [0,1]
        ev19_template: EV19 模板 ID，如 'fst-modified'
        inputs: 输入信息: {raw_files: [...], groups: {...}}
        gate_signals: 决策信号
    """
    payload = {
        "status": status,
        "summary": summary,
        "paradigm": paradigm,
        "metrics_summary": metrics_summary or {},
        "per_subject": per_subject or {},
        "statistics": statistics or {},
        "output_files": output_files or {},
        "data_quality_warnings": data_quality_warnings or [],
        "errors": errors or [],
        "confidence": confidence,
        "ev19_template": ev19_template,
        "inputs": inputs,
        "gate_signals": gate_signals,
    }
    return _seal_handoff(CodeExecutorHandoff, "handoff_code_executor.json", payload, runtime)


@tool("seal_data_analyst_handoff", parse_docstring=True)
def seal_data_analyst_handoff(
    status: str,
    key_findings: list[str] | None = None,
    outlier_findings: list[dict[str, Any]] | None = None,
    excluded_metrics: list[str] | None = None,
    method_warnings: list[str] | None = None,
    recommendations: list[str] | None = None,
    errors: list[str] | None = None,
    gate_signals: dict[str, Any] | None = None,
    quality_warnings: list[dict[str, Any]] | None = None,
    parameter_audit_findings: list[dict[str, Any]] | None = None,
    runtime: Runtime = None,
) -> str:
    """Data-analyst 完成分析后，封存 handoff_data_analyst.json。

    严禁直接用 write_file 写 handoff_data_analyst.json，必须走本 tool。

    Args:
        status: "completed" / "failed"
        key_findings: 1-5 条核心发现
        outlier_findings: 异常 subject 列表，每条含 subject/metric/value/deviation/counterfactual
        excluded_metrics: 因质量问题被排除的指标
        method_warnings: 统计方法警告
        recommendations: 建议后续操作
        errors: 错误信息
        gate_signals: 决策信号
        quality_warnings: 从 handoff_code_executor.json 透传的 data_quality_warnings
        parameter_audit_findings: 恒传空数组 []。2026-06-18 起 data-analyst 不再产出
            参数审计（判据行为学上造不出来，移出判读路径）。字段保留为向前兼容 + 将来以
            确定性代码接入时复用；调用时 parameter_audit_findings=[] 即可，gate_signals
            的 parameter_audit_findings_count / parameter_audit_critical_count 恒为 0。
    """
    payload = {
        "status": status,
        "key_findings": key_findings or [],
        "outlier_findings": outlier_findings or [],
        "excluded_metrics": excluded_metrics or [],
        "method_warnings": method_warnings or [],
        "recommendations": recommendations or [],
        "errors": errors or [],
        "gate_signals": gate_signals,
        "quality_warnings": quality_warnings or [],
        "parameter_audit_findings": parameter_audit_findings or [],
    }
    return _seal_handoff(DataAnalystHandoff, "handoff_data_analyst.json", payload, runtime)


# ============================================================================
# data-analyst 分步填模板（产物 + 封口）—— spec
# 2026-06-23-data-analyst-seal-stepwise-fill-template
#
# 根因：data-analyst 的判读是唯一「LLM 当场生成、直接当 seal tool_call args 一次性
# 吐出」的内容；它与 reasoning_tokens 共享单次响应 max_tokens，flash 默认产
# 3300-4096 reasoning → args 被挤穿腰斩成未终止 JSON → invalid_tool_calls →
# handoff 永不落盘 → FAILED → lead 降级跳过专业判读。
#
# 根治（与其他三个 subagent 的「先变落盘产物、seal 只封口记元数据」心智模型对齐）：
#   harness 预置合法 in_progress 空模板（_preset_data_analyst_template，纯 Python、
#       无 LLM、无 max_tokens，确定性 100% 成功）
#   → data-analyst 用 fill_* 工具逐字段填（每次 args 天然小，绕过狭颈）
#   → finalize 确定性 gate 判核心字段填足才改 status=completed/partial/failed。
#
# 全程工具签名粒度 + 确定性 gate，零新 prompt 规则（守 HarnessX Telecom 禁令）。
# ============================================================================

_DATA_ANALYST_HANDOFF_FILENAME = "handoff_data_analyst.json"

# ============================================================================
# data-analyst handoff 读-改-写并发安全（spec 2026-06-24-fill-handoff-concurrent-write-race）
#
# 根因：data-analyst 同一条 AIMessage 并行发多个 fill_data_analyst_* tool_call，
# 它们都走「_load_da_payload 读 → 改 payload → _write_da_payload 写固定 tmp → os.rename」
# 无锁且 tmp 路径固定共享 → 竞态（丢字段 / FileNotFoundError rename / 读到 0 字节）。
#
# 治本：所有 data-analyst handoff 读-改-写按 (workspace, filename) 用模块级 threading.Lock
# 串行化。照 sandbox/file_operation_lock.py 模式（WeakValueDictionary + guard lock）。
# fill 是 sync 工具用 threading（非 asyncio）；fill 只有 runtime 拿不到 sandbox，不能
# 复用 get_file_operation_lock(sandbox, path)，按 workspace/filename 自建同款锁。
# 临界区内只调 _load/_write/_da_progress（均不取锁）→ 无重入死锁。
# ============================================================================

# Use WeakValueDictionary to prevent memory leak in long-running processes
# (locks auto-removed when no thread references them) — mirrors file_operation_lock.py.
_DA_HANDOFF_LOCKS: weakref.WeakValueDictionary[str, threading.Lock] = weakref.WeakValueDictionary()
_DA_HANDOFF_LOCKS_GUARD = threading.Lock()


def _get_da_handoff_lock(workspace: Path) -> threading.Lock:
    """按 (workspace, handoff filename) 返回串行化锁（同 path 同一 Lock，不同 path 不阻塞）。

    Args:
        workspace: host-side workspace 目录。

    Returns:
        该 workspace 的 data-analyst handoff 专用 threading.Lock。
    """
    key = str(workspace / _DATA_ANALYST_HANDOFF_FILENAME)
    with _DA_HANDOFF_LOCKS_GUARD:
        lock = _DA_HANDOFF_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _DA_HANDOFF_LOCKS[key] = lock
        return lock


# DataAnalystHandoff 的 list[str] 字段（工具 A）。
_DATA_ANALYST_TEXT_LIST_FIELDS = (
    "key_findings",
    "excluded_metrics",
    "method_warnings",
    "recommendations",
    "errors",
)

# DataAnalystHandoff 的 list[dict] 字段（工具 B）。每条 dict 工具内按 field 选
# OutlierFinding / DataQualityWarning 子模型校验。
_DATA_ANALYST_RECORD_LIST_FIELDS = ("outlier_findings", "quality_warnings")


def _data_analyst_record_model(field: str) -> type:
    """Return the Pydantic sub-model for a record-list field (lazy, no top-level cycle)."""
    from deerflow.subagents.handoff_schemas import DataQualityWarning, OutlierFinding

    if field == "outlier_findings":
        return OutlierFinding
    if field == "quality_warnings":
        return DataQualityWarning
    raise ValueError(f"unknown data-analyst record-list field: {field}")


def preset_data_analyst_template_to_workspace(workspace: Path, *, config_id: str | None = None) -> str:
    """预置合法 in_progress 空 handoff 模板到 workspace（纯 Python、无 LLM）。

    spec §二 结构 1：executor 派遣 data-analyst 前（agent 跑之前）用本函数生成
    ``handoff_data_analyst.json``：``status="in_progress"`` + 所有字段空默认 +
    运行时填 ``analysis_config_id``（从 experiment-context.json 读，缺则 "PENDING"）。

    - **幂等覆盖**（grill 问题 1）：每次派遣无条件覆盖——无论 workspace 已有上次的
      in_progress（FAILED 残留，覆盖防污染）还是终态模板（新分析，覆盖重置），原子写
      保证不留半新半旧。
    - Pydantic 先校验模板合法（status Literal 已含 in_progress）→ 原子写（tmp+rename）。
    - 不放静态文件：analysis_config_id + 路径都是 per-thread 运行时值。

    Args:
        workspace: host-side workspace 目录。
        config_id: 可选 analysis_config_id；None 时从 experiment-context.json 读。

    Returns:
        写盘路径的字符串。

    Raises:
        ValueError: Pydantic 校验失败（不应发生——in_progress + 空默认合法）。
        OSError: 磁盘满/权限/workspace 不可写（响亮报错，让 executor 终止派遣）。
    """
    resolved_config_id = config_id if config_id is not None else _read_analysis_config_id(workspace)
    payload: dict[str, Any] = {
        "status": "in_progress",
        "key_findings": [],
        "outlier_findings": [],
        "excluded_metrics": [],
        "method_warnings": [],
        "recommendations": [],
        "errors": [],
        "gate_signals": None,
        "analysis_config_id": resolved_config_id,
        "quality_warnings": [],
        "parameter_audit_findings": [],
        "sealed_by": "preset",
    }
    # 进锁（spec 2026-06-24 #3）：防 preset 与首个 fill 竞态。preset 在派遣路径、fill 在
    # subagent turn，理论有先后，但显式加锁更稳。Pydantic 校验 + 原子写整段在锁内。
    with _get_da_handoff_lock(workspace):
        # Pydantic 校验模板合法（防 schema 漂移写坏 in_progress 模板）。
        handoff = DataAnalystHandoff(**payload)
        final_path = workspace / _DATA_ANALYST_HANDOFF_FILENAME
        # tmp 唯一后缀（纵深，同 _write_da_payload）。
        tmp_path = workspace / f"{_DATA_ANALYST_HANDOFF_FILENAME}.{os.getpid()}.{threading.get_ident()}.tmp"
        json_bytes = handoff.model_dump_json(indent=2, exclude_none=False).encode("utf-8")
        tmp_path.write_bytes(json_bytes)
        os.replace(tmp_path, final_path)  # POSIX atomic
        os.chmod(final_path, 0o644)
    return str(final_path)


def _load_da_payload(workspace: Path) -> dict[str, Any]:
    """读 handoff_data_analyst.json。不存在/不可解析 → 抛 ValueError（让 fill 失败回响亮）。"""
    path = workspace / _DATA_ANALYST_HANDOFF_FILENAME
    if not path.exists():
        raise ValueError(
            "data-analyst template not found in workspace. The harness should have "
            "pre-populated an in_progress template before dispatching the subagent. "
            "This indicates a dispatch-path bug — the preset step failed silently or "
            "did not run."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise ValueError(f"handoff_data_analyst.json is unreadable/invalid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ValueError("handoff_data_analyst.json is not a JSON object")
    return data


def _write_da_payload(workspace: Path, payload: dict[str, Any]) -> None:
    """整体 DataAnalystHandoff 重校验 → 原子重写。校验失败抛 ValueError（回响亮给 LLM）。"""
    try:
        handoff = DataAnalystHandoff(**payload)
    except ValidationError as e:
        raise ValueError(
            f"data-analyst handoff validation failed after fill: {e}. "
            f"Check field names/types against DataAnalystHandoff schema."
        ) from e
    final_path = workspace / _DATA_ANALYST_HANDOFF_FILENAME
    # tmp 路径加 pid+tid 唯一后缀（纵深防御）：锁已保证不并发，万一锁失效也不抢同一 tmp。
    # 用 os.getpid()/threading.get_ident()（确定性、可测），不用随机数/时间戳（spec §七）。
    tmp_path = workspace / f"{_DATA_ANALYST_HANDOFF_FILENAME}.{os.getpid()}.{threading.get_ident()}.tmp"
    json_bytes = handoff.model_dump_json(indent=2, exclude_none=False).encode("utf-8")
    tmp_path.write_bytes(json_bytes)
    os.replace(tmp_path, final_path)
    os.chmod(final_path, 0o644)


def _da_progress(workspace: Path) -> str:
    """读回刚写的 payload，返回极简 JSON 进度（字段填充计数）。"""
    try:
        data = _load_da_payload(workspace)
    except Exception:  # noqa: BLE001 — 进度是 best-effort，绝不让它盖过 fill 主结果
        return '{"status":"in_progress","progress":"unreadable"}'
    counts = {
        f: len(data.get(f) or []) for f in _DATA_ANALYST_TEXT_LIST_FIELDS + _DATA_ANALYST_RECORD_LIST_FIELDS
    }
    gs = data.get("gate_signals")
    return json.dumps(
        {
            "status": data.get("status"),
            "sealed_by": data.get("sealed_by"),
            "key_findings": counts["key_findings"],
            "outlier_findings": counts["outlier_findings"],
            "method_warnings": counts["method_warnings"],
            "recommendations": counts["recommendations"],
            "quality_warnings": counts["quality_warnings"],
            "gate_signals_set": bool(gs),
        },
        ensure_ascii=False,
    )


@tool("fill_data_analyst_text_list", parse_docstring=True)
def fill_data_analyst_text_list(
    field: Literal[
        "key_findings",
        "excluded_metrics",
        "method_warnings",
        "recommendations",
        "errors",
    ],
    mode: Literal["set", "append"],
    value: list[str],
    runtime: Runtime = None,
) -> str:
    """逐字段填 data-analyst handoff 的 list[str] 字段（每次 args 天然小，绕过 max_tokens 狭颈）。

    harness 已在 workspace 预置合法 in_progress 空模板（status="in_progress"）。
    本工具读模板 → 填/追加该字段 → 整体 DataAnalystHandoff 重校验（保持 in_progress 合法）
    → 原子重写 → 返回极简 JSON 进度。

    Args:
        field: 要填的字段名。key_findings=1-5 条核心发现；excluded_metrics=被排除指标；
            method_warnings=方法学警告；recommendations=给研究者的建议；errors=非致命错误。
        mode: "set"=用 value 整体覆盖该字段（一次填一个字段完整内容）；"append"=把 value
            追加到该字段末尾（兜底超长字段，分多次 append）。
        value: 字符串列表。
    """
    workspace = _resolve_workspace(runtime)
    # 读-改-写整段进锁（spec 2026-06-24 #2）：同消息多 fill 并行不再竞态。
    with _get_da_handoff_lock(workspace):
        payload = _load_da_payload(workspace)
        current = payload.get(field) or []
        if not isinstance(current, list):
            current = []
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ValueError(f"fill_data_analyst_text_list: value must be list[str], got {type(value).__name__}")
        if mode == "set":
            payload[field] = list(value)
        elif mode == "append":
            payload[field] = [*current, *value]
        else:  # 被 Literal 拦住，防御性
            raise ValueError(f"fill_data_analyst_text_list: unknown mode {mode!r}")
        _write_da_payload(workspace, payload)
        progress = _da_progress(workspace)  # 读回也在锁内，保证进度一致
    return f"OK: filled {field} ({mode}, {len(payload[field])} items). progress={progress}"


@tool("fill_data_analyst_record_list", parse_docstring=True)
def fill_data_analyst_record_list(
    field: Literal["outlier_findings", "quality_warnings"],
    mode: Literal["set", "append"],
    value: list[dict],
    runtime: Runtime = None,
) -> str:
    """逐字段填 data-analyst handoff 的 list[dict] 字段（outlier_findings / quality_warnings）。

    工具内按 field 选 OutlierFinding / DataQualityWarning 子模型校验每条 dict，再整体
    DataAnalystHandoff 重校验 → 原子重写 → 返回极简 JSON 进度。

    Args:
        field: "outlier_findings"=按受试者的离群诊断（每条含 subject/metric/value/deviation/
            counterfactual）；"quality_warnings"=从 handoff_code_executor.json 透传的
            data_quality_warnings（每条含 severity/code/message/evidence/blocks_downstream 等）。
        mode: "set"=整体覆盖；"append"=追加（兜底超长字段）。
        value: dict 列表，每条 dict 的字段须匹配对应子模型（OutlierFinding / DataQualityWarning）。
    """
    workspace = _resolve_workspace(runtime)
    # 读-改-写整段进锁（spec 2026-06-24 #2）。
    with _get_da_handoff_lock(workspace):
        payload = _load_da_payload(workspace)
        sub_model = _data_analyst_record_model(field)
        # 逐条用子模型校验 → 规整成 dict（拿子模型的默认/校验）。校验失败抛 ValueError 回响亮。
        validated: list[dict[str, Any]] = []
        if not isinstance(value, list):
            raise ValueError(f"fill_data_analyst_record_list: value must be list[dict], got {type(value).__name__}")
        for i, item in enumerate(value):
            if not isinstance(item, dict):
                raise ValueError(
                    f"fill_data_analyst_record_list: value[{i}] must be a dict for {field} ({sub_model.__name__}), "
                    f"got {type(item).__name__}"
                )
            try:
                validated.append(sub_model(**item).model_dump())
            except ValidationError as e:
                raise ValueError(
                    f"fill_data_analyst_record_list: value[{i}] failed {sub_model.__name__} validation: {e}"
                ) from e
        current = payload.get(field) or []
        if not isinstance(current, list):
            current = []
        if mode == "set":
            payload[field] = validated
        elif mode == "append":
            payload[field] = [*current, *validated]
        else:  # 防御性
            raise ValueError(f"fill_data_analyst_record_list: unknown mode {mode!r}")
        _write_da_payload(workspace, payload)
        progress = _da_progress(workspace)
    return f"OK: filled {field} ({mode}, {len(payload[field])} items). progress={progress}"


@tool("fill_data_analyst_gate_signals", parse_docstring=True)
def fill_data_analyst_gate_signals(value: dict, runtime: Runtime = None) -> str:
    """一次整体填 data-analyst handoff 的 gate_signals（GateSignals 单 dict，无 append 语义）。

    字段不多、一次装得下；不分次补子键留口（否则每次重传整 dict、args 可能变大）。
    工具内用 GateSignals 子模型校验 → 整体 DataAnalystHandoff 重校验 → 原子重写。

    Args:
        value: GateSignals dict，字段如 statistical_validity / data_quality /
            quality_warnings_critical_count / statistics_status 等（详见 GateSignals schema）。
    """
    workspace = _resolve_workspace(runtime)
    from deerflow.subagents.handoff_schemas import GateSignals

    # 读-改-写整段进锁（spec 2026-06-24 #2）。
    with _get_da_handoff_lock(workspace):
        payload = _load_da_payload(workspace)
        if not isinstance(value, dict):
            raise ValueError(f"fill_data_analyst_gate_signals: value must be a dict, got {type(value).__name__}")
        try:
            payload["gate_signals"] = GateSignals(**value).model_dump()
        except ValidationError as e:
            raise ValueError(f"fill_data_analyst_gate_signals: GateSignals validation failed: {e}") from e
        _write_da_payload(workspace, payload)
        progress = _da_progress(workspace)
    return f"OK: filled gate_signals. progress={progress}"


@tool("finalize_data_analyst_handoff", parse_docstring=True)
def finalize_data_analyst_handoff(
    final_status: Literal["completed", "partial", "failed"],
    runtime: Runtime = None,
) -> str:
    """唯一能把 data-analyst handoff 的 status 从 in_progress 改成终态的封口入口（确定性 gate）。

    读模板 → 改 status=final_status + sealed_by="finalize" → 整体 DataAnalystHandoff
    重校验（**让 _completed_requires_core_output validator 自然触发**，绝不自己写
    key_findings 判空——守 SSOT，memory feedback_single_source_of_truth）→ 写
    manifest(sealed_by="finalize") → 返回 OK。gate 拒绝时抛 ValueError（LangChain 转
    error ToolMessage 引导补 fill）。

    final_status=completed → 必须 key_findings 非空（既有 validator 触发）；
    partial/failed 不要求 key_findings 非空（fast-fail 路径合法）。

    Args:
        final_status: 终态。completed=判读完成且 key_findings 已填；partial=fast-fail/
            描述性路径（如 n<3、statistics 空）；failed=无法判读（handoff 读取失败等）。
    """
    workspace = _resolve_workspace(runtime)
    # 读-改-写整段进锁（spec 2026-06-24 #2）：防末尾 fill 与 finalize 竞态。
    with _get_da_handoff_lock(workspace):
        payload = _load_da_payload(workspace)
        payload["status"] = final_status
        payload["sealed_by"] = "finalize"
        # 整体重校验让 _completed_requires_core_output validator 自然触发（守 SSOT）。
        # 校验失败（如 completed 但 key_findings 空）→ ValueError → error ToolMessage 引导补 fill。
        _write_da_payload(workspace, payload)
        # 写 manifest，sealed_by="finalize" 可观测（spec §七）。
        sha256 = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        ).hexdigest()
        _update_manifest(
            workspace,
            _DATA_ANALYST_HANDOFF_FILENAME,
            sha256,
            payload.get("analysis_config_id", _read_analysis_config_id(workspace)),
        )
        progress = _da_progress(workspace)
    return (
        f"OK: finalized handoff_data_analyst.json "
        f"(status={final_status}, sealed_by=finalize, sha256={sha256[:12]}...). "
        f"progress={progress}"
    )


@tool("seal_chart_maker_handoff", parse_docstring=True)
def seal_chart_maker_handoff(
    paradigm: str,
    summary: str,
    chart_files: list[str] | None = None,
    failed_charts: list[dict[str, str]] | None = None,
    remaining_charts: list[dict[str, str]] | None = None,
    status: str = "completed",
    gate_signals: dict[str, Any] | None = None,
    runtime: Runtime = None,
) -> str:
    """Chart-maker 完成绘图后，封存 handoff_chart_maker.json。

    严禁直接用 write_file 写 handoff_chart_maker.json，必须走本 tool。

    Args:
        paradigm: 范式名
        summary: 一句话描述生成的图表
        chart_files: 成功的图表 png 路径（必须在 /mnt/user-data/outputs/ 下）
        failed_charts: 失败列表，每条 {chart_id, reason}
        remaining_charts: P5 预算降级指纹——被 chart_budget 截断未画的 per_subject 图，
            每条 {chart_id, reason}（reason 通常 "chart_budget_truncated"）。
            来自 prep_chart_plan 返回的 plan_summary.budget_remaining_ids。无截断时省略/[]。
        status: "completed" / "partial" / "failed"（全部失败时为 failed）
        gate_signals: 决策信号
    """
    payload = {
        "status": status,
        "paradigm": paradigm,
        "summary": summary,
        "chart_files": chart_files or [],
        "failed_charts": failed_charts or [],
        "remaining_charts": remaining_charts or [],
        "gate_signals": gate_signals,
    }
    return _seal_handoff(ChartMakerHandoff, "handoff_chart_maker.json", payload, runtime)


@tool("seal_report_writer_handoff", parse_docstring=True)
def seal_report_writer_handoff(
    status: str,
    report_path: str,
    sections_written: list[str] | None = None,
    errors: list[str] | None = None,
    gate_signals: dict[str, Any] | None = None,
    runtime: Runtime = None,
) -> str:
    """Report-writer 完成写报告后，封存 handoff_report_writer.json。

    严禁直接用 write_file 写 handoff_report_writer.json，必须走本 tool。

    Args:
        status: "completed" / "failed"
        report_path: 报告文件路径（report.html / 旧 report.md）
        sections_written: 已写的段落，如 ["Results", "Discussion"]
        errors: 错误信息
        gate_signals: 决策信号
    """
    payload = {
        "status": status,
        "report_path": report_path,
        "sections_written": sections_written or [],
        "errors": errors or [],
        "gate_signals": gate_signals,
    }

    # 报告载体分支（spec 2026-06-29）：
    #   .html → 占位符 base64 内联 + XSS 消毒（自包含 HTML，治「下载丢图」）。
    #   .md   → 沿用既有 markdown 占位符解析 + 路径规整（旧报告不回归）。
    _report_name = Path(report_path).name
    _is_html = _report_name.lower().endswith((".html", ".htm"))

    try:
        _ws = _resolve_workspace(runtime)
        _report_host = _ws.parent / "outputs" / _report_name
        if _is_html:
            # HTML 模式：{{img:}} → base64 data URI（仅代表性图，prompt 已指导）
            _resolve_html_report_image_placeholders(_report_host, _ws)
            # 消毒：剥 <script>/on*/<iframe> 等（LLM 产出，封存时做一层确定性消毒）
            if _report_host.is_file():
                try:
                    _raw = _report_host.read_text(encoding="utf-8")
                    _clean = sanitize_report_html(_raw)
                    if _clean != _raw:
                        _report_host.write_text(_clean, encoding="utf-8")
                        logger.info("seal_report_writer_handoff: sanitized report.html %s", _report_host)
                except Exception:
                    logger.warning("seal_report_writer_handoff: html sanitize skipped", exc_info=True)
        else:
            # markdown 模式（legacy）：占位符解析 + 路径规整，原样保留不回归。
            # Layer 1: resolves LLM-written placeholders to canonical virtual paths
            # from handoff_chart_maker.json.chart_files before the Layer 2
            # _normalize_report_image_paths prefix fix.
            _resolve_report_image_placeholders(_report_host, _ws)
    except Exception as _e:
        logger.warning("seal_report_writer_handoff: image placeholder resolution failed: %s", _e)

    # 1. 规范化图片路径前缀（仅 markdown 模式；HTML 模式图已 base64 内联，无路径可规整）
    # Normalise image paths in the report file before sealing: the artifacts API
    # requires ``mnt/user-data/outputs/file.png`` (no leading slash), but LLMs
    # often write ``outputs/file.png`` or ``/mnt/user-data/outputs/file.png``,
    # both of which return 400 Bad Request in the frontend. Fix it server-side
    # so the result is correct regardless of what the model wrote.
    if not _is_html:
        try:
            _ws = _resolve_workspace(runtime)
            # report_path is a virtual path like /mnt/user-data/outputs/report.md;
            # derive the host path by replacing the virtual prefix with the outputs dir.
            _report_host = _ws.parent / "outputs" / _report_name
            _normalize_report_image_paths(_report_host)
        except Exception as _e:
            logger.warning("seal_report_writer_handoff: image normalisation pre-step failed: %s", _e)

    result = _seal_handoff(ReportWriterHandoff, "handoff_report_writer.json", payload, runtime)

    # Sprint 6: write experiment_summary memory fact on successful completion
    if status == "completed":
        try:
            workspace = _resolve_workspace(runtime)
            thread_data = runtime.state.get("thread_data", {})
            config_id = payload.get("analysis_config_id", _read_analysis_config_id(workspace))

            # Read paradigm from experiment-context.json
            ctx_path = workspace / "experiment-context.json"
            paradigm = "unknown"
            if ctx_path.exists():
                try:
                    ctx = json.loads(ctx_path.read_text(encoding="utf-8"))
                    paradigm = ctx.get("paradigm", "unknown")
                except Exception:
                    pass

            thread_id = runtime.state.get("thread_id", "unknown")
            user_id = thread_data.get("user_id")

            _write_experiment_summary_memory(
                workspace=workspace,
                paradigm=paradigm,
                config_id=config_id,
                thread_id=thread_id,
                user_id=user_id,
            )
        except Exception as e:
            logger.warning("S6 memory injection skipped: %s", e)

    return result
