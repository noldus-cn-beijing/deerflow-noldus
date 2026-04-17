# Fallback 流程（范式不支持时）

当 `compute_metrics` 返回 `paradigm not supported` 或核心指标无法计算时，
切换到旧流程：

## 步骤

1. 调用 `get_analysis_template(paradigm, file_pattern, groups)` 获取 Python 脚本
2. `write_file` 把返回的脚本内容写到 `/mnt/user-data/workspace/analysis.py`
3. `bash` 执行：`cd /mnt/user-data/workspace && python analysis.py`
4. `ls /mnt/user-data/outputs/` 确认产物
5. `read_file /mnt/user-data/workspace/handoff_code_executor.json` 确认 handoff 已生成
6. 按 output-contract 返回给 lead agent

## 何时走 fallback

- `compute_metrics` 返回 paradigm not supported
- 任务描述的范式不在 `parse_trajectories` 的 `paradigm_hint` 识别范围内
- 需要自定义分析逻辑（用户明确要求非标准指标）

## 注意

- Fallback 不会生成 parsed.pkl/metrics.pkl 等中间文件，因此步骤 5 的 handoff 由 analysis.py 自身写出
- 确保 `/mnt/user-data/outputs/` 目录中的输出文件被纳入 handoff 的 `output_files.charts` 字段
