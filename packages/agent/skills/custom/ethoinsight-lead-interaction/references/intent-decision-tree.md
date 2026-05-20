# 意图分类决策树

[user message arrives]
   ├── 有 uploaded_files?
   │     ├── Yes + 复合语义 (≥2 个动词类别)        → E2E_FULL
   │     ├── Yes + 单一动词类别 (仅"分析"/仅"看看") → E2E_MIN
   │     └── Yes + 无任何动词类别                  → QA_KNOWLEDGE
   ├── workspace 有 handoff_code_executor.json?
   │     ├── Yes + 要图 → CHART
   │     ├── Yes + 要报告 → REPORT
   │     └── Yes + 追问数据 → QA_FACT
   ├── 问领域知识(EPM 是什么/焦虑模型) → QA_KNOWLEDGE
   └── 信息不够 → CLARIFY

## 复合语义判定 (E2E_FULL vs E2E_MIN 的唯一分水岭)

把用户消息里出现的动词归入下列 4 个类别,出现 ≥2 个类别 → 复合语义 → E2E_FULL:

| 类别 | 触发词(中) | 触发词(英) |
|---|---|---|
| **CALC**(算)        | 计算 / 算 / 跑指标            | compute / calculate / run metric |
| **ANALYZE**(解读)   | 分析 / 解读 / 描述 / 描述性 / 评估 / 看看 / 比较 / 对比 | analyze / interpret / describe / assess / compare |
| **VISUALIZE**(出图) | 可视化 / 出图 / 画图 / 画一下 / 图表 / 可视 | visualize / plot / chart / draw / graph |
| **REPORT**(成稿)    | 报告 / 总结 / 汇总 / 写成稿     | report / summary / summarize / writeup |

例子(都是 E2E_FULL):
- "帮我做**描述性分析**和**可视化**" → ANALYZE + VISUALIZE
- "**分析并画图**" → ANALYZE + VISUALIZE
- "**计算指标**然后**出图**" → CALC + VISUALIZE
- "**分析**完了**写个报告**" → ANALYZE + REPORT
- "**算完**给我**总结**一下" → CALC + REPORT

E2E_MIN 反例(只有 1 个动词类别):
- "**分析一下**这个数据" → 仅 ANALYZE
- "**看看**这个数据" → 仅 ANALYZE
- "**画个图**" → 仅 VISUALIZE(但其实更适合 CHART,需先有 handoff)
- "**算一下**指标" → 仅 CALC

歧义兜底: 拿不准时偏向 E2E_FULL — E2E_MIN 在 code-executor 跑完后会 ask(four-choice) 让用户选下一步,但**如果用户最初消息已经明确想要分析+可视化,跳过反问直接 E2E_FULL 更顺**。错判到 E2E_FULL 的代价是多跑一个 chart-maker(用户反正想要);错判到 E2E_MIN 的代价是 lead 拿到 chart-maker 结果后没下一步、容易自己读 handoff 撞硬限。

## 边界 case
- 上传数据 + "我先了解一下 EPM" → QA_KNOWLEDGE
- 没上传 + "继续上次的分析" → 读 workspace 看历史 handoff
- 上传新数据 + 已有旧 handoff → 重新走 E2E_FULL/E2E_MIN
