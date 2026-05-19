# 意图分类决策树

[user message arrives]
   ├── 有 uploaded_files?
   │     ├── Yes + "分析并画图/报告/全套" → E2E_FULL
   │     ├── Yes + "分析一下"/"看看" → E2E_MIN
   │     └── Yes + 无分析意图 → QA_KNOWLEDGE
   ├── workspace 有 handoff_code_executor.json?
   │     ├── Yes + 要图 → CHART
   │     ├── Yes + 要报告 → REPORT
   │     └── Yes + 追问数据 → QA_FACT
   ├── 问领域知识(EPM 是什么/焦虑模型) → QA_KNOWLEDGE
   └── 信息不够 → CLARIFY

## 边界 case
- 上传数据 + "我先了解一下 EPM" → QA_KNOWLEDGE
- 没上传 + "继续上次的分析" → 读 workspace 看历史 handoff
- 上传新数据 + 已有旧 handoff → 重新走 E2E_FULL/E2E_MIN
