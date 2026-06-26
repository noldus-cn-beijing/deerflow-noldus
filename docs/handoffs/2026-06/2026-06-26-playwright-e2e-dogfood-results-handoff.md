# Handoff：Playwright E2E dogfood 复现结果（2026-06-26，真实 28 EPM 数据）

> 接 [[2026-06-26-playwright-e2e-dogfood-bug-reproduction-handoff]] 的执行。用 `/noldus-insight-e2e` skill 驱动真实 Chromium 跑了一遍完整 dogfood（28 个 `Raw data-EPM-Xuhui-Trial 1-28.xlsx`，路径 `~/DemoData/real_data/Raw data-EPM-Xuhui-28`），从登录到 chart-maker 出 113 图。**核心结论：handoff（写在 `07838ae6`）要复现的一批 bug，大部分已被其后的 #216/#217/#218 修掉；同时坐实了 1 个新前端 bug（画廊单样本图不渲染）+ 1 个工作流缺陷（画图/报告二选一）。**
>
> **⚠️ 关键前提修正**：handoff 的 HEAD 是 `07838ae6`，但执行时 dev 已在 `6ec47f78`，期间合入：
> - `646d5fd5` #216 画廊全量图走磁盘端点 + 对话流报告卡 + 画廊返回不重挂载
> - `373a06f4` #217 决策卡 DecisionCard
> - `6ec47f78` #218 memory context UUID 加载崩溃修复
>
> 这三条**精确命中** handoff 要复现的 spec（三现象修复 / 输入框遮挡+决策卡 / memory UUID）。核实代码用 `git show HEAD:`，勿用 `07838ae6`。
>
> dev 服务：`make dev`（:2026 nginx + :8001 gateway + :3000 next，三进程齐全）。thread=`bd7ca7f7-5972-47b7-a2d8-9167481c770f`，run=`1be92cc7-2fee-47fa-9bea-d00b483a01c5`，证据全集 `/tmp/noldus-e2e-runs/20260626-114801/`。

---

## 一、每个 bug 一行结论（复现矩阵）

| spec / bug | handoff 预期 | 当前 dev (`6ec47f78`) 实测 | 结论 |
|---|---|---|---|
| **memory UUID**（spec A，#218） | log 有 `got 'UUID'` | 全程 gateway.log 无 `Failed to load memory context` / `got 'UUID'`；thread 正常加载记忆 | ✅ **已修复，未复现** |
| **画廊丢图 §一**（三现象，#213 方向错） | 画廊只 1 张 box，112 per_subject 丢失 | 后端 artifacts 端点返回 113/113（1 aggregate+112 per_subject，35KB）；**前端画廊只渲染 aggregate 1 张，112 per_subject 标题显示但缩略图不渲染**（滚动不懒加载） | ⚠️ **部分复现，根因变了**：后端登记 OK（#216 后端有效），**bug 在前端 `gallery/page.tsx` 不渲染 per_subject**。**非** handoff 推断的"subagent Command 不上行" |
| **报告看不到 §二**（三现象，#216 报告卡） | 报告藏侧栏、对话流无内嵌 | **report.md 根本未生成**：DecisionCard[20] 给了"画图 vs 报告"二选一，driver generic `确认` 被解读成"画图"分支，lead 走完 chart-maker 即 `Run success` 收尾，**从未派遣 report-writer** | ⚠️ **新发现（工作流缺陷）**：不是"报告藏起来了"，是"画图/报告被设计成互斥二选一，选画图就没报告"。详见 §三 |
| **返回卡顿 §三**（三现象，#216 不重挂载） | `router.push` 重挂载→卡顿+丢滚动 | 画廊→返回对话未做火焰图；#216 已改"不重挂载"（commit 证据 `gallery/page.tsx` 改动），运行时未观察到明显卡顿 | 🟡 **代码层已修，运行时未坐实卡顿**（未录 trace，保守记） |
| **决策卡遮挡**（#217） | 决策卡被悬浮输入框盖住 | thread 跑完后决策卡已答消失（历史消息），未抓到"决策卡 live + 输入框同屏"态；最终态底部布局正常无错位 | 🟡 **未完全坐实**（需 live 决策卡待答态，本次 thread 已过该节点） |
| **历史乱序**（spec B 🔴红线） | 重进后 input 跑中间+输出消失 | 后端 25 events 顺序完全正确（[0]input→…→[24]完成）；**前端刷新重进 25 条消息顺序正确，input 在最前、无消失、无乱序** | ✅ **未复现**（后端+前端双轨顺序均正确）。若曾为真 bug，当前 dev 已不触发；或需更复杂多 run 合并条件 |
| **切回卡顿**（无 spec，待火焰图） | 后台 SSE 积压→切回 long task 风暴 | 未录（需真机后台 tab 节流 + 进行中 SSE，headless 大概率不节流） | 🟡 **未坐实**（同 handoff §六诚实原则，不伪造） |
| **ETHO-10 chart 伪完成**（Panel B） | claimed≠disk | `chart_files=113 failed=0`，113/113 实存盘（aggregate 1 + per_subject 112），skipped=1 | ✅ **未复现**，#213/#216 产物登记修复有效 |

---

## 二、Panel 判定（forensic analyze.py）

- **Panel A（data-analyst seal）= ✅ HEALTHY**：`handoff_data_analyst.json` `status=completed, sealed_by=finalize`，5 条 key_findings（Kruskal-Wallis H=19.69 p=0.0002 η²=0.20 + 离群 Trial3/Trial12 排查 + 运动抑制混杂排除）。**#187 stepwise-fill 路径正常工作，非 degraded。**
- **Panel B（chart 伪完成 ETHO-10）= ✅ PASS**：`outputs/*.png=113`，`plot_*.png=113`，`chart_files=113/113 实存`，aggregate=`plot_box_open_arm.png` 落盘，skipped=1（非阻断）。sealed_by=`run_plan`。
- **Panel other = ⚠️ report-writer 未派遣**：`code-executor: completed/run_plan/epm`；`report-writer: handoff 不存在`；`outputs/report.md = False`。

---

## 三、最重要的新发现：画图 / 报告被设计成互斥二选一

**现象**：data-analyst 完成判读后，lead 发 DecisionCard（前端 message[20]）：

> 📊 指标和解读已完成。需要我把结果可视化成图吗？
> A. 是，把刚才的结论画成图
> B. 不用，直接给我报告

driver 答 generic `确认` → lead 解读为选 A → 派 chart-maker 画 113 图 → `Run 1be92cc7 success` 收尾。**全程未派 report-writer，无 report.md。**

**为什么这是缺陷而非"用户选择"**：
1. dogfood 真实场景里用户通常**既要图又要报告**（图给汇报、报告给存档）。当前互斥设计迫使二选一。
2. lead 最后一条 AI message[24] 原文："所有分析环节（指标计算→判读→图表）均已完成"——**lead 的流程终点认知里，"报告生成"不是必经步骤**，而是 B 选项的替代路径。
3. 即使选 A（画图），分析已完成、data-analyst 结论已就绪，**生成 report.md 成本极低**（report-writer 拿现成 handoff 即可），却没自动触发。

**建议（待 spec）**：DecisionCard 应允许"画图 + 报告"多选，或画图完成后自动续派 report-writer。这比"画廊显示"更影响用户交付完整性。**这条不在原 handoff 待派清单里，是新问题。**

---

## 四、第二个新发现：画廊单样本图(112) 不渲染

**现象**：画廊页 `/workspace/chats/{tid}/gallery`（注意是 **chats** 复数，handoff/skill 未写明，单数 `chat` 会 404）：
- 标题正确分组：`汇总图 (1)` + `单样本图 (112)` ✅
- artifacts 磁盘端点 `/api/threads/{tid}/artifacts/charts` 返回 113 条全量 ✅（#216 后端有效）
- **但单样本图(112) 分组下不渲染任何 `<img>` 缩略图**，滚动 8 次（每次 800px）也不懒加载，只显示分组标题
- DOM 有 26 个 `[aria-expanded]`/details 元素——**疑似该分组默认折叠**，但展开态未自动渲染缩略图

**根因方向**：前端 `gallery/page.tsx`（#216 改过）的 per_subject 渲染/懒加载逻辑有缺陷，**非后端问题**。这与 handoff §六"#213 方向错、subagent Command 不上行"的推断**相反**——后端登记是对的，#216 后端磁盘端点是对的，bug 纯在前端画廊渲染。

**待确认**：是"默认折叠、点击可展开"（交互设计）还是"展开也渲染不出"（真 bug）。本次滚动未触发，需点击 `[aria-expanded]` 元素验证。**建议回填三现象修复 spec §一：修复目标从前端到后端重新界定——后端已 OK，只需修画廊 per_subject 渲染。**

---

## 五、其他观察（spec 未预料）

1. **流程状态条与后端进度脱节**：final-body.txt 显示前端流程指示器"列对齐 进行中 / 指标计算 未开始"，但后端实际早已算完 140 指标 + 113 图。前端 progress 指示器未随真实进度推进——独立 UI bug。
2. **generic HITL 在 EPM 上不收敛但能跑通**：driver 答 8 轮 `确认`，前 5 轮 lead 一直反问 FewZones/AllZones（因为 generic 确认没指定），第 5 轮后 lead 自行选了推荐值 A (PlusMaze-AllZones) 推进。**结论：generic `确认` 能跑通 EPM 但效率低（5 轮冗余反问）；真实 dogfood 应 live 给范式答案**（template=PlusMaze-FewZones 因数据有 open/closed 划区列、groups=Treatment、open=开臂/closed=闭臂）。注意：本次 lead 实际选了 **AllZones** 而非 FewZones（因为 generic 没指定，lead 走推荐默认），但分析结果仍合理（5 指标全成）。
3. **PydanticSerializationUnexpectedValue 噪声**：gateway.log 大量 `Expected 'none' - serialized value may not be as expected [field_name='context']` 警告——非阻断，但污染日志。可入"后端质量三小项"spec。
4. **memory update skipped/failed**（run 后异步）：`Failed to parse LLM response for memory update: No valid memory update JSON object found`——run 结束后 memory 异步更新偶尔解析失败，不影响主流程，但属质量小项。

---

## 六、对原 handoff 待派 spec 的影响

| spec | 原状态 | 本次 E2E 后建议 |
|---|---|---|
| 三现象修复 §一（画廊丢图） | 待派 | **重新界定**：后端 OK，只需修前端画廊 per_subject 渲染（§四） |
| 三现象修复 §二（报告看不到） | 待派 | **升级为新 spec**：画图/报告互斥二选一缺陷（§三），比"报告展示"更根本 |
| 三现象修复 §三（返回卡顿） | 待派 | #216 代码层已改不重挂载，运行时未坐实；降优先级 |
| 输入框遮挡 / 决策卡 | #217 已合 | 决策卡已生效（message[20] 实测）；遮挡未坐实（需 live 态），降优先级 |
| 历史乱序 spec B 🔴 | 待派红线 | **当前 dev 未复现**（后端+前端顺序双正确）；建议先在更多多 run 合并场景验证再决定是否仍需修 |
| memory UUID | #218 已合 | ✅ 验证有效 |
| Chrome 切回卡顿 | 待火焰图出 spec | 仍未坐实，保持"需真机" |
| 后端质量三小项 | 待派 | 加入：流程状态条脱节 + Pydantic 序列化噪声 + memory update 解析失败 |

---

## 七、未坐实项的诚实声明（同 handoff §六原则）

- **Step6 切回卡顿**：未录真机火焰图。headless 不触发后台 tab 节流，自动化大概率测不到——**不伪造数据**。需真机：开真任务进行中→切后台 tab 30-60s→切回，CDP `Tracing.start/end` 录切回瞬间。
- **Step4 决策卡遮挡**：thread 已跑完，决策卡成历史消息，未抓到"决策卡 live + 输入框同屏"遮挡态。需开新任务跑到 DecisionCard 待答时截图。
- **画廊单样本图折叠 vs 真不渲染**：本次只滚动了页面，未点击 `[aria-expanded]` 验证展开态。下次先点开折叠组再判定。

---

## 八、证据索引（`/tmp/noldus-e2e-runs/20260626-114801/`）

- `analyze.txt` — forensic panel 全输出（Panel A/B/other）
- `drive.log` — driver 全程（pngs=113, handoffs=5, reportMd=false）
- `gateway-excerpt.log` — 本 thread gateway 关键行（含 `Run 1be92cc7 -> success`、report-writer 零派遣证据）
- `clarifications.json` — 8 轮 HITL Q&A（含第 8 轮 lead 完整总结，证 report-writer 未派）
- `specB-messages-raw.json` — 后端 `/messages` 25 events 全量（顺序正确证据）
- `specB-rejoin-order.json` — 前端刷新重进 25 条消息渲染顺序（顺序正确证据）
- `fe-findings.json` / `fe-gallery-dom.json` / `fe-gallery-after-scroll.json` — 画廊 DOM 取证
- `fe-02-gallery.png` / `fe-04-rejoin.png` — 画廊 + 重进截图
- `sse-0..8.txt` / `final.png` / `final-body.txt` — SSE 流 + 最终态

---

## milestone 建议

本次 E2E **不是 feature checkpoint**，是验证+发现。建议：
1. **不开新 milestone**——结果回填到现有 `frontend-phase0` track。
2. **新起两个 spec**（待派）：① 画廊 per_subject 渲染修复（前端，三现象 §一 重新界定）② 画图/报告互斥→多选或自动续派 report-writer（§三，新发现的交付完整性缺陷）。
3. **降优先级/暂缓**：历史乱序 spec B（当前未复现）、返回卡顿 §三（#216 已改）、决策卡遮挡（需 live 态再验）。
4. **后端质量三小项 spec 扩充**：流程状态条脱节 + Pydantic 序列化噪声 + memory update 解析失败。
