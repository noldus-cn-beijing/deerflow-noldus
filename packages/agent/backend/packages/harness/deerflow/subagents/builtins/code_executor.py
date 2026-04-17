"""Code execution subagent for behavioral data analysis."""

from deerflow.subagents.config import SubagentConfig

CODE_EXECUTOR_CONFIG = SubagentConfig(
    name="code-executor",
    description=(
        "行为数据分析的代码执行专家。"
        "按 ethoinsight-analysis skill 指示，依次调用 parse_trajectories、"
        "compute_metrics、run_statistics、generate_charts、assess_and_handoff。"
    ),
    system_prompt="""你是行为数据分析的代码执行专家。

<environment>
- ethoinsight 和所有依赖包已预装在系统 Python 中，可直接 import。
- 不要创建虚拟环境，不要运行 pip install。这会浪费你的执行轮次。
- 分析所需的所有能力已封装为 5 个细粒度工具，通过文件在 /mnt/user-data/workspace/ 传递中间状态。
</environment>

<workflow>
严格按照注入的 ethoinsight-analysis skill 执行 6 步分析流程：
parse → compute → statistics → charts → assess_and_handoff → return。

每一步工具调用完成后，读取对应的 *_summary.json 文件检查质量信号，
再调用下一步。遇到范式不支持或工具报错，参考 skill 的 fallback 和 error-recovery 指南。
</workflow>

<output>
最后返回给 lead agent 的消息包含：
- handoff JSON 路径（assess_and_handoff 产出）
- 关键输出文件列表
- 数据质量警告摘要（如有）
</output>""",
    tools=[
        "parse_trajectories",
        "compute_metrics",
        "run_statistics",
        "generate_charts",
        "assess_and_handoff",
        "get_analysis_template",
        "bash",
        "read_file",
        "write_file",
        "ls",
        "str_replace",
    ],
    disallowed_tools=[
        "task",
        "ask_clarification",
        "present_files",
        "web_search",
        "web_fetch",
        "image_search",
    ],
    model="inherit",
    max_turns=12,
    timeout_seconds=900,
    skills=["ethoinsight-analysis"],
)
