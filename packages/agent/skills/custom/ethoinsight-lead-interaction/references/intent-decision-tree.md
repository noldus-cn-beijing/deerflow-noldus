# 意图分类决策树

[user message arrives]
   ├── 有 uploaded_files?
   │     ├── Yes + 模糊总称("分析"/"看看"/"研究下"/"整一下", 不含明确出图意愿)
   │     │       → E2E_FULL_ASKVIZ (code → data → ask(viz?) → [yes]chart → ask(report?))
   │     ├── Yes + 明确出图意愿("画"/"图"/"可视化"/"箱线"/"轨迹"/"趋势"/"表"/...)
   │     │       → E2E_FULL (code → data → chart → ask(report?))
   │     ├── Yes + 只"算"/"计算" (无解读/出图意愿)
   │     │       → E2E_MIN (code → ask(four-choice))
   │     └── Yes + 无任何动词类别 → QA_KNOWLEDGE
   ├── workspace 有 handoff_code_executor.json?
   │     ├── Yes + 要图 → CHART
   │     ├── Yes + 要报告 → REPORT
   │     └── Yes + 追问数据 → QA_FACT
   ├── 问领域知识(EPM 是什么/焦虑模型) → QA_KNOWLEDGE
   └── 信息不够 → CLARIFY

## 出图意愿置信度判定 (E2E_FULL_ASKVIZ vs E2E_FULL 的分水岭)

**高置信(直接 chart-maker,跳过反问):** 用户消息含任一明确出图意愿触发词。

| 触发词类别 | 触发词(中) | 触发词(英) |
|---|---|---|
| **直接"图"概念** | 画 / 图 / 可视化 / 画出来 / 画一下 / 画几张 / 出图 / 图表 | plot / chart / draw / graph / visualize |
| **表格类(视为可视化)** | 表 / 表格 / 列出来 / 一览表 | table / list |
| **具体图种名** | 箱线 / 箱型 / 轨迹 / 趋势 / 热图 / 散点 / 柱状 / 时序 | boxplot / heatmap / scatter / bar |
| **明确展示意图** | 用图说 / 展示 / 呈现 / 给我看看图 | show / present |

**低置信(走 E2E_FULL_ASKVIZ, 跑完解读后反问"要不要出图"):** 用户消息只含模糊总称,无任何出图意愿触发词。

模糊总称清单:
- 「**分析一下**」/「分析下」/「分析这个数据」
- 「**看看**」/「帮我看看」/「帮我看下」
- 「**研究下**」/「**整一下**」
- 「**搞一下**」/「弄一下」(数据)

例子:
- "帮我分析一下大鼠强迫游泳数据" → 模糊总称 + 无出图词 → **E2E_FULL_ASKVIZ**
- "看看这两组数据" → 模糊总称 + 无出图词 → **E2E_FULL_ASKVIZ**
- "分析并画图" → 含"画图" → **E2E_FULL**
- "帮我做描述性分析和可视化" → 含"可视化" → **E2E_FULL**
- "画几张箱线图" → 含"画"+"箱线" → **E2E_FULL** (但其实更适合 CHART,如果已有 handoff)
- "把指标列成表格" → 含"表格" → **E2E_FULL**
- "算一下指标" → 只 CALC → **E2E_MIN**

## E2E_FULL_ASKVIZ 反问模板

data-analyst 完成后,lead 调:

```python
ask_clarification(
    question="📊 指标和解读已完成。需要我把结果可视化成图吗?",
    options=[
        "A. 是,把刚才的结论画成图(默认推荐,箱线图/轨迹图/时序图)",
        "B. 不用,直接给我报告"
    ]
)
```

- 用户选 A → 派 chart-maker → 完成后再 ask(要不要报告?)
- 用户选 B → 直接 ask(要不要报告?),跳过 chart-maker

## 歧义兜底

拿不准时偏 **E2E_FULL_ASKVIZ**:
- 错判 E2E_FULL_ASKVIZ 的代价是多一次反问(用户可一步选 B 跳过出图)
- 错判 E2E_FULL 的代价是给只想要数字 + 报告的用户硬塞 4 张图(用户反馈过"不必要的图")
- 错判 E2E_MIN 的代价是 lead 拿到 code-executor 结果后又得 ask(four-choice),解读环节缺失

## 4 类动词归类 fallback

仅在 fast-path 未命中(消息中没有任何明确出图触发词,也没有"分析/看看"模糊总称)时启用,把消息里的动词归 4 类:

| 类别 | 触发词(中) | 触发词(英) |
|---|---|---|
| **CALC**(算)        | 计算 / 算 / 跑指标            | compute / calculate / run metric |
| **ANALYZE**(解读)   | 解读 / 描述 / 描述性 / 评估 / 比较 / 对比 | analyze / interpret / describe / assess / compare |
| **VISUALIZE**(出图) | (已在 fast-path 高置信触发词里穷举,这里不重复) | visualize / plot / chart |
| **REPORT**(成稿)    | 报告 / 总结 / 汇总 / 写成稿     | report / summary / summarize / writeup |

- 出现 VISUALIZE 类 → E2E_FULL (这条 fast-path 已 cover,几乎不会落到 fallback)
- 只 ANALYZE 一类(如"解读这组数据") → E2E_FULL_ASKVIZ
- 只 CALC 一类 → E2E_MIN
- 含 REPORT 类(无 handoff) → E2E_FULL (按 4 类计为 ≥2 类复合语义)

历史例子(走 fast-path 后已无需 fallback,保留作参考):
- "帮我做**描述性分析**和**可视化**" → ANALYZE + VISUALIZE = 复合 → E2E_FULL(fast-path 命中"可视化")
- "**分析**完了**写个报告**" → ANALYZE + REPORT = 复合 → E2E_FULL(fast-path 命中"报告")

## 边界 case
- 上传数据 + "我先了解一下 EPM" → QA_KNOWLEDGE
- 没上传 + "继续上次的分析" → 读 workspace 看历史 handoff
- 上传新数据 + 已有旧 handoff → 重新走 E2E_FULL_ASKVIZ / E2E_FULL / E2E_MIN
