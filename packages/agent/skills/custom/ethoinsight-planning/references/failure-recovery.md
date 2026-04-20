# 失败场景降级策略

## code-executor 失败类型分类

根据失败信息的关键词分类：

| 关键词 | 类型 | 策略 |
|--------|------|------|
| "范式不支持"/"尚未支持"/"无模板"/"not supported" | 能力边界 | 降级选项 A |
| "文件解析失败"/"编码错误"/"No trajectory files found"/"parse error" | 数据格式 | 降级选项 B |
| "分组信息缺失"/"groups"/"missing groups" | 参数不足 | 降级选项 C |
| 超时无输出 / timeout | 执行复杂度 | 降级选项 D |
| 其他 | 未知 | 降级选项 E |

## data-analyst 失败

```python
ask_clarification(
    question="分析解读步骤遇到问题：<简短原因>。以下几种处理方式，您倾向哪一种？",
    clarification_type="approach_choice",
    context="data-analyst 失败",
    options=[
        "重试一次（通常是临时性错误）",
        "直接展示 code-executor 的原始统计结果（跳过专家解读）",
        "中止本次分析"
    ]
)
```

- ❌ 不要 bypass data-analyst 继续派 report-writer：没有专家解读，报告质量必然劣化
- ❌ 不要静默重试（浪费 token 且用户无感）

## report-writer 失败

```python
ask_clarification(
    question="APA 报告生成遇到问题：<简短原因>。以下几种处理方式，您倾向哪一种？",
    clarification_type="approach_choice",
    context="report-writer 失败",
    options=[
        "重试一次",
        "只要分析洞察就够了（不要报告）",
        "中止"
    ]
)
```

- ❌ 不要把 analysis_summary.md 原文当作最终报告返回给用户（用户期望的是 APA 格式）
- ❌ 不要输出残缺的报告（比如只有 Results 没有 Discussion）

## 降级选项

### A. 能力边界（范式不支持）

```python
ask_clarification(
    question="该范式的自动分析流程尚未完善（当前支持：shoaling, epm, open_field, ...）。可选方案：",
    clarification_type="approach_choice",
    context="code-executor 返回：<原错误信息>",
    options=[
        "尝试基础指标计算（移动距离、区域停留时间等，可能不完整）",
        "展示数据结构，我来指定分析内容",
        "暂时跳过"
    ]
)
```

### B. 数据格式问题

```python
ask_clarification(
    question="数据文件解析失败：<具体错误>。可能原因：",
    clarification_type="missing_info",
    context="code-executor 返回：<原错误信息>",
    options=[
        "文件不是 EthoVision XT 导出格式 → 重新导出",
        "编码问题（UTF-16 等）→ 尝试重新读取",
        "文件损坏 → 重新上传"
    ]
)
```

### C. 参数不足

重新检查分组定义，通常是 lead agent 自己的 prompt 传递问题。回到 Step 2 重新询问分组。

### D. 执行超时

```python
ask_clarification(
    question="分析耗时较长，可能因为样本量大或数据复杂。可选方案：",
    clarification_type="approach_choice",
    context="code-executor 超时",
    options=[
        "继续等待（再次派遣，给更长超时）",
        "简化分析（只跑核心指标，跳过复杂统计）",
        "分批分析（先做对照组）"
    ]
)
```

### E. 未知错误

```python
ask_clarification(
    question="分析遇到未预期的问题：<错误信息>。需要您帮助判断下一步：",
    clarification_type="approach_choice",
    context="code-executor 未知错误",
    options=[
        "重试",
        "跳过这次分析",
        "联系技术支持"
    ]
)
```

## 绝对禁止

- ❌ 同一轮对话中重新派遣 code-executor 执行相同范式（用户明确指示除外）
- ❌ 自己用 bash/read_file 替代 code-executor 完成整个分析流程
- ❌ 假设"换个参数"能解决范式不支持的问题
- ❌ 不告知用户就静默重试

## 连续失败处理

- 同一范式连续失败 2 次 → 必须放弃当前路径，询问用户整体方向
- 不同 subagent 连续失败 2 次 → 必须 `ask_clarification`，不能继续盲目流水线
