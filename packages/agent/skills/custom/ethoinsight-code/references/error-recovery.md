# 错误恢复指南

任何工具返回 `status: failed` 时按本指南排查。

## parse_trajectories 失败

返回 `error: "parse_batch raised: ..."`：
1. 用 bash 检查文件路径：`ls /mnt/user-data/uploads/`
2. 检查文件编码：`file /mnt/user-data/uploads/*.txt`（EthoVision 标准导出是 UTF-16 LE，parse_batch 已自动处理；若 file 显示 ASCII 可能是用户自己截短的格式，不一定能解析）
3. 检查文件首 50 行：`head -c 500 /mnt/user-data/uploads/<一个文件名>.txt | iconv -f utf-16 -t utf-8`

返回 `error: "No files matched pattern ..."`：
1. 用 bash `ls /mnt/user-data/uploads/` 确认文件存在
2. 看返回的 `matched_files` 字段是否给出了实际文件名，根据实际文件名修正 glob

## compute_metrics 失败

返回 `error: "missing dependency: .../parsed.pkl"`：
→ 回到步骤 1 先调 `parse_trajectories`

返回 `error: "Invalid groups JSON"`：
→ 检查 groups 字符串：必须是 JSON 字符串，键是组名，值是 subject ID 列表。例如：
`'{"control":["Subject 1","Subject 2"],"treatment":["Subject 3"]}'`

若脚本 stderr 包含 `"compute_paradigm_metrics"` 相关错误：
1. 读 `workspace/parsed_summary.json` 看 `columns` 是否有 `x_center_mm`, `y_center_mm`
2. 若列缺失，说明数据不适合该范式——调 `get_analysis_template` 走 fallback
3. 若 stderr 含 "unknown paradigm"，范式名拼写错误或未实现——参考 `fallback-workflow.md`

## run_statistics 失败

返回 `error: "Need at least 2 groups ..."`：
→ groups 只有 1 个组，无法做组间比较。确认分组信息。

返回 `error: "compare_groups raised: Input data has range zero"`：
→ 某组的数据方差为零（所有 subject 值完全相同）。读 `metrics_summary.json` 定位是哪个指标，在 handoff 的 errors 中标注。

## generate_charts 失败

部分 chart 失败（`errors` 非空但 `chart_paths` 非空）：
→ 非致命，继续下一步。在 handoff 中保留 errors。

全部 chart 失败（`chart_paths` 为空）：
→ 读 errors 列表，常见原因：metrics.pkl 损坏（回步骤 2）、chart_types 拼写错误。

