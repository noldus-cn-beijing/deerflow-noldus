# Fallback 流程

仅当 run_paradigm_analysis 返回不支持该范式时，使用旧流程：

1. 调用 get_analysis_template 获取 Python 脚本
2. write_file 写入脚本到 /mnt/user-data/workspace/analysis.py
3. bash 执行脚本
4. 验证 /mnt/user-data/outputs/ 下的输出文件
5. 确认 handoff JSON 已生成
