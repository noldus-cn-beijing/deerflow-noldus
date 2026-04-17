# 意图分类决策树

## 信号来源

1. `<uploaded_files>` 中 "uploaded in this message" 部分（本轮新上传）
2. 用户消息文本
3. 已有会话历史（workspace 中是否有 analysis_report.md / metrics.csv）

## 决策树

```
本轮是否有新上传数据文件？
├── 是
│   ├── 用户消息包含分析词汇（"分析"/"统计"/"看看"/"处理"/"可视化"/"报告"）？
│   │   ├── 是 → 主意图: 端到端分析
│   │   └── 否（仅打招呼/无分析指令） → 主意图: 需澄清
│   │       → 动作: ask_clarification("用户刚上传了文件 X，但未提出分析请求，是否要分析？")
│   └── (已处理)
└── 否
    ├── 用户消息指代已有结果（代词"这个"/"刚才"/"上面"/提及具体指标如"p 值"/"NND"/"Cohen's d"）？
    │   ├── 是 → 主意图: 追问已有结果
    │   │       → 动作: 派遣 knowledge-assistant，附 workspace/analysis_report.md 路径
    │   └── 否
    │       └── 消息是概念性问题（"什么是 X"/"怎么做 X"/"X 和 Y 的区别"）？
    │           ├── 是 → 主意图: 知识问答
    │           │       → 动作: 派遣 knowledge-assistant
    │           └── 否 → 主意图: 闲聊/确认
    │                   → 动作: 自己回复，不派遣
```

## 特殊路径

### "重新分析"/"换种方式"
- 触发词：re, 重新, 换, different, 不一样
- 动作: 端到端分析，但在 code-executor prompt 中注明"重新分析"
- 如果用户具体指了某一步（如"只换图表类型"）→ 仅派遣 code-executor

### "只重写报告"
- 触发词：重写, 换个格式, 翻译, 精简, APA, 中文/英文
- 前置条件: workspace 中有 analysis_report.md 或 code_summary.json
- 动作: 仅派遣 report-writer，传入原分析结果路径
- 无前置数据 → `ask_clarification` 询问是否先做分析

### 混合意图
- 例: "帮我分析这批数据，顺便解释下什么是 NND"
- 处理: 主意图（端到端分析）优先 + 末尾追加一个 knowledge-assistant 回答副意图
- 或者在主流水线完成后，用 report-writer 的报告中自然包含术语解释

## 边界 case

| 情况 | 判断 |
|------|------|
| 有新文件但是非数据文件（如 PDF、图片） | 先 `ask_clarification` 询问用途 |
| 有新文件 + "?"（纯问号） | 等价"无明确指令"，`ask_clarification` |
| 有新文件 + 用户说"先不分析" | 不规划，直接回复"收到，需要分析时告诉我" |
| 无新文件 + "继续上次的分析" | 如果 workspace 有数据 → 根据上次的 paradigm 继续 |
