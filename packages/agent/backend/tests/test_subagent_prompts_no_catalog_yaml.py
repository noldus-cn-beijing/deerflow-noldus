"""W27: lint — data-analyst / report-writer prompt 不许再引导读 catalog YAML 文件。

这是防回滚 lint。如果后续编辑误把 prompt 改回"read catalog YAML",此测试 fail。
"""
from __future__ import annotations

from pathlib import Path

import pytest

_BACKEND_ROOT = Path(__file__).resolve().parents[1]
_SUBAGENTS_DIR = _BACKEND_ROOT / "packages" / "harness" / "deerflow" / "subagents" / "builtins"

_FILES_UNDER_LINT = [
    _SUBAGENTS_DIR / "data_analyst.py",
    _SUBAGENTS_DIR / "report_writer.py",
]

# 禁止模式:任何引导 subagent read catalog YAML 文件路径的写法
_FORBIDDEN_PATTERNS: list[str] = [
    # 路径写法
    "/path/to/ethoinsight/catalog/",
    "ethoinsight.catalog as c",  # SKILL.md 顶部的 import + __file__ 招数
    "c.__file__",
]

_ALLOWED_NEGATION_CONTEXTS = [
    "不要尝试 read catalog YAML",
    "不再 read catalog YAML",
]


@pytest.mark.parametrize("py_file", _FILES_UNDER_LINT, ids=lambda p: p.name)
def test_subagent_prompt_has_no_catalog_yaml_read_guidance(py_file: Path) -> None:
    """断言文件不含禁用模式;必要时允许在反面警告语境出现 'catalog YAML'。"""
    text = py_file.read_text(encoding="utf-8")

    for forbidden in _FORBIDDEN_PATTERNS:
        assert forbidden not in text, (
            f"{py_file.name}: 仍含禁用模式 {forbidden!r}。"
            f" 该 subagent 应改为读 /mnt/user-data/workspace/plan_metrics.json。"
            f" 详见 docs/superpowers/specs/2026-05-27-catalog-fields-into-plan-design.md"
        )

    # "catalog YAML" 字面量必须只出现在反面警告语境
    if "catalog YAML" in text:
        # 把文本切到 "catalog YAML" 周围 60 字符的窗,验证存在允许的反面短语
        idx = 0
        while True:
            pos = text.find("catalog YAML", idx)
            if pos == -1:
                break
            window = text[max(0, pos - 40): pos + 30]
            assert any(neg in window for neg in _ALLOWED_NEGATION_CONTEXTS), (
                f"{py_file.name}: 'catalog YAML' 出现在非反面警告语境(window={window!r})。"
                f" 必须改成不要引导 subagent 读 catalog YAML 文件。"
            )
            idx = pos + len("catalog YAML")
