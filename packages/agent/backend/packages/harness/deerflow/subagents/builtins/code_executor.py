"""Code execution subagent for behavioral data analysis."""

from deerflow.subagents.config import SubagentConfig

CODE_EXECUTOR_CONFIG = SubagentConfig(
    name="code-executor",
    description=(
        "行为数据分析的代码执行专家。"
        "使用 run_paradigm_analysis 一步完成分析。"
    ),
    system_prompt="""你是行为数据分析的代码执行专家。

<environment>
- ethoinsight 和所有依赖包已预装在系统 Python 中，可直接 import。
- 不要创建虚拟环境，不要运行 pip install。这会浪费你的执行轮次。
- 如果需要运行 Python 代码，直接使用 `python` 命令。
</environment>

请严格按照下方注入的 skill 指南执行分析任务。如果没有 skill 注入，按以下默认流程：
第一步：从任务描述中提取 paradigm、file_pattern、groups。
第二步：立即调用 run_paradigm_analysis 工具。
第三步：检查返回 JSON 的 status 字段，将结果返回。""",
    tools=["run_paradigm_analysis", "get_analysis_template",
           "bash", "read_file", "write_file", "ls", "str_replace"],
    disallowed_tools=["task", "ask_clarification", "present_files",
                       "web_search", "web_fetch", "image_search"],
    model="inherit",
    max_turns=8,
    timeout_seconds=600,
    skills=["ethoinsight-analysis"],
)
