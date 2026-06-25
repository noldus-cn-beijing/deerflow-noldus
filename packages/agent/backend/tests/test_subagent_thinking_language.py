"""TDD for thinking-channel language constraint (spec 2026-06-25).

EPM dogfood (thread 0e72d605, 用户全程中文) 暴露：lead + 4 个 subagent 的 thinking
（思考过程）全英文，只有最终输出中文。根因=语言约束只覆盖「输出」channel，无一点名
thinking。修法=语言约束显式扩展到 thinking channel（正面指令「使用与用户相同的语言」
覆盖思考 + 输出 + write_file + handoff），进 5 个 agent system_prompt。

本文件断言各 system_prompt 含 thinking 语言约束（不仅「输出」）。
"""
import re

from deerflow.agents.lead_agent import prompt as prompt_module
from deerflow.agents.lead_agent.prompt import apply_prompt_template
from deerflow.subagents.builtins.chart_maker import CHART_MAKER_CONFIG
from deerflow.subagents.builtins.code_executor import CODE_EXECUTOR_CONFIG
from deerflow.subagents.builtins.data_analyst import DATA_ANALYST_CONFIG
from deerflow.subagents.builtins.report_writer import REPORT_WRITER_CONFIG

_SUBAGENTS = [
    ("code-executor", CODE_EXECUTOR_CONFIG.system_prompt),
    ("data-analyst", DATA_ANALYST_CONFIG.system_prompt),
    ("chart-maker", CHART_MAKER_CONFIG.system_prompt),
    ("report-writer", REPORT_WRITER_CONFIG.system_prompt),
]


def test_t1_subagents_constrain_thinking_language():
    """4 个 subagent system_prompt 必须把语言约束显式点名到 thinking channel。

    红：code_executor / chart_maker 当前只有「输出」无「思考」。
    绿：M1a/M1b 后含「思考过程（thinking / reasoning）」。
    """
    for name, sp in _SUBAGENTS:
        # 语言段存在
        assert "<语言>" in sp, f"{name} 缺少 <语言> 段"
        # 必须点名 thinking —— 不仅是「输出」
        assert "思考" in sp or "thinking" in sp.lower(), (
            f"{name} 语言约束未覆盖 thinking channel（缺「思考」/「thinking」）"
        )
        # 思考约束必须与语言规则同段（在 <语言>.../</语言> 内出现「思考」）
        lang_section = sp.split("<语言>")[1].split("</语言>")[0]
        assert "思考" in lang_section or "thinking" in lang_section.lower(), (
            f"{name} 「思考」字样不在 <语言> 段内——thinking 约束未真正进入语言规则"
        )


def test_t2_lead_thinking_style_constrains_language(monkeypatch):
    """lead 的 <thinking_style> 段必须含语言约束（thinking 跟随用户语言）。

    走与 test_lead_agent_prompt.py 相同的 monkeypatch 姿势：stub 掉 config/skills，
    不依赖磁盘 config.yaml（gitignored）。
    """
    from types import SimpleNamespace

    config = SimpleNamespace(
        sandbox=SimpleNamespace(mounts=[]),
        skills=SimpleNamespace(container_path="/mnt/skills"),
    )
    monkeypatch.setattr("deerflow.config.get_app_config", lambda: config)
    monkeypatch.setattr(prompt_module, "_get_enabled_skills", lambda: [])
    monkeypatch.setattr(prompt_module, "get_deferred_tools_prompt_section", lambda **kwargs: "")
    monkeypatch.setattr(prompt_module, "_build_acp_section", lambda **kwargs: "")
    monkeypatch.setattr(prompt_module, "_get_memory_context", lambda agent_name=None, **kwargs: "")
    monkeypatch.setattr(prompt_module, "get_agent_soul", lambda agent_name=None: "")

    sp = apply_prompt_template()
    assert "<thinking_style>" in sp, "lead system prompt 缺 <thinking_style> 段"
    # 抽出 thinking_style 段。注意：confidentiality 段会作为示例文本提到 <thinking_style>，
    # 故用 regex 抓真正的 <thinking_style>...</thinking_style> 块（最后一处、含真实内容）。
    thinking_sections = re.findall(r"<thinking_style>\n.*?\n</thinking_style>", sp, flags=re.DOTALL)
    assert thinking_sections, "未找到完整 <thinking_style>...</thinking_style> 块"
    thinking_section = thinking_sections[-1]
    lower = thinking_section.lower()
    assert "same language" in lower or "用户说中文" in thinking_section, (
        "lead <thinking_style> 段缺少语言约束（thinking 跟随用户语言）"
    )


def test_t3_report_writer_sections_written_still_fixed_chinese():
    """回归：report_writer 的 sections_written 固定中文章节名规则保留不变。

    该字段是下游消费的固定 key，不跟随用户语言（守现有设计）。
    """
    sp = REPORT_WRITER_CONFIG.system_prompt
    assert "sections_written" in sp, "report_writer 缺 sections_written 字段说明"
    assert "中文章节名" in sp, (
        "report_writer 的 sections_written 固定中文章节名规则被破坏（应保留不变）"
    )
