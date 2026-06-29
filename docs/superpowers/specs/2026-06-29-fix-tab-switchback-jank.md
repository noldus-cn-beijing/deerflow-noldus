# Spec: 修复「切回 tab 时长卡顿」（后台流式积压 + in-flight 大消息整条重渲染）

> 状态：待实施（根因已定位；**实施前必须先 CDP 实测坐实**，守 handoff 铁律 + `feedback_perf_is_efficient_impl_not_visual_downgrade`）
> 归属：#232 流式性能的补充场景（#232 未覆盖）
> 红线：不动 useStream / mergeMessages / dedupe。

## Context（活体症状）

#232 合入后，跑 dogfood 到 **data-analyst 一大串流式输出**时，用户**从 VS Code 切回 Chrome 页面**，会有**一段很长的卡顿**。注意是「切回瞬间」卡，不是持续卡——这是与 #232 治的稳态开销**不同的场景**。

## 根因（已定位，待实测坐实）

#232 治的是「持续流式时历史消息每 token 被 O(n) 重扫」——用 extraction-cache 缓存**终态**消息、**有意 bypass in-flight 那条**（message-list.tsx:102-114 注释明说："the in-flight one is bypassed"）。

但「切回 tab」是另一条机制：
1. **后台 tab 被浏览器激进节流**：`requestAnimationFrame` 暂停、React 并发 scheduler（`useDeferredValue` 依赖它）基本停摆 → 渲染冻结。
2. **但 SSE 不受 tab 隐藏影响**（`useStream` 在 hooks.ts 继续把 token 灌进 state）→ 后台期间 data-analyst 那条 in-flight 消息持续变大（几 KB→几十 KB），`messages` 多次更新积压。
3. **切回前台瞬间**：浏览器恢复渲染，React 一次性追平后台积压的多次 state 更新 + 把那条**未缓存的巨型 in-flight 消息**整条重新 group + 抽取 + markdown 解析（streamdown）→ 主线程被这一下打满 → 长卡顿。

**#232 为何没挡住**：data-analyst 大输出**就是 in-flight 那条**，而 #232 对 in-flight 消息**故意不缓存**；`useDeferredValue` 在此还**放大**切回卡顿（后台积压的更新切回时集中 flush）。

## 方案候选（实施前用 CDP 实测选择 + 量化；都属"同视觉省开销"）

### Step 0（不可跳）：CDP 实测坐实
在 **prod build**（dev build perf 是噪声）复现「切走→后台等 data-analyst 流式积压→切回」，录 performance trace，确认切回瞬间主线程 self-time 大头 = ① 积压更新集中 flush ② in-flight 大消息整条 markdown 重解析 / 抽取。给出 before 数字。
- 复现可不依赖用户的跨用户 thread：用 e2e 用户自跑一个产生大段 data-analyst 流式的 run，用 CDP `Emulation`/`Page` 把页面置 hidden 模拟后台，再切回录 trace。

### 候选 A（推荐，针对根因 2）：in-flight 大消息「冻结前缀 + 活跃尾部」
把流式那条消息拆成「已落定前缀（可缓存/memo 冻结）+ 活跃尾部（只它重解析）」，让 streamdown 的块级 memo 对 in-flight 也生效——目前 #232 对 in-flight bypass 缓存削弱了它。切回时只有尾部重渲染，前缀不动。
- 这是业界标准做法（committed prefix + live tail），spec 1（streaming-render-perf）已提及但 #232 未实现这层。

### 候选 B（针对根因 1+3）：visibilitychange 感知
监听 `document.visibilitychange`：tab 隐藏时**降低/暂停**对 deferred 更新的强制追平（或在隐藏期间合并 state 不触发渲染），切回时**分帧**追平（`requestIdleCallback`/分批）而非一次性 flush；或切回时**只渲染最终态**、丢弃后台积压的中间帧。
- 风险：别破坏 `useStream` 的 state（只在渲染层做，不碰 SSE/merge）。

### 候选 C（兜底）：切回时对超大 in-flight 消息降级渲染
in-flight 消息超过阈值（如 >N KB）时，流式期间用轻量渲染（纯文本/降频），终态再上完整 markdown。需评估是否影响"流式可见性"体验。

**优先 A**（治本、最不影响体验）；A 不足再叠 B。C 仅兜底。**最终选择以 Step 0 的 trace 数据为准。**

## 改动文件（视方案）
- 候选 A：`message-list.tsx`（in-flight 拆分逻辑）、`markdown-content.tsx` / `message-list-item.tsx`（前缀/尾部分别渲染 + memo）、可能扩展 `extraction-cache.ts`（in-flight 前缀也可缓存）。
- 候选 B：新 `useDocumentVisibility` hook + message-list 消费；只动渲染层。
- 不碰 `core/threads/hooks.ts` 的 useStream/merge/dedupe。

## TDD / 验收
- **CDP perf（prod build）**：Step 0 基线 vs 改后，「切走→积压→切回」主线程长任务 self-time 降 ≥50%、切回后首帧时间显著下降、无 >Xms 长任务。**dev build 不作数**。
- 功能不回归：流式仍逐字可见（候选 A/B 不降低可见性）；`pnpm check` 0；`npx vitest run` 绿（in-flight 拆分/缓存的新单测：前缀冻结、尾部更新、终态与原渲染字节级一致）。
- 守红线：useStream/mergeMessages/dedupe 未碰（git diff 证）。

## 不做
- 不动 SSE/merge 核心。
- 不为了"切回不卡"牺牲流式逐字可见性（守 perf=高效实现非视觉降级）。
