# 实施 spec：切回 tab 卡顿——#238 候选B 已合但真机仍卡，定位真凶（2026-06-29 续）

> **接续 spec**，承 `2026-06-29-fix-tab-switchback-jank.md`（原 Step0+候选A/B/C）与其 Step0 实测报告 `docs/superpowers/reports/2026-06-29-tab-switchback-jank-step0-trace-report.md`。
> **新事实**：#238 只实施了候选 B；用户真机手工 dogfood 实测**仍卡十几秒**。这恰好是原 Step0 报告建议的「真机坐实路径」，**坐实了卡顿真实存在**，推翻 headless「零 longtask」假象。本 spec 定位候选 B 未覆盖的真凶。

---

## 〇、为什么需要这份接续 spec（事实链）

| 时间 | 事实 | 证据 |
|---|---|---|
| 原 spec | 写了 Step0（CDP 实测坐实）+ 候选 A（in-flight 冻结前缀）/B（visibilitychange）/C（降级兜底），红线=实测前不盲上 | `2026-06-29-fix-tab-switchback-jank.md` |
| Step0 实测 | headless `Page.setWebLifecycleState('frozen')` **冻结整页 JS**（含 SSE 回调）→ `frozen_grew_by ≤ 0`、切回零 longtask。**复现不出**真实 hidden 的不对称性。报告判 A/B/C 均「不应盲上」，并建议**真机 GUI + OS 级切 tab** 才能坐实 | Step0 trace report §40-47 |
| #238 | 只实施了**候选 B**（`useDocumentVisibility` hook：hidden 时用 live messages 绕过 `useDeferredValue` 积压） | commit `18b412c3` |
| **现在** | 用户真机 `make dev` + dogfood，切到 VS Code 再切回，**仍卡十几秒** | 用户报告（= Step0 报告 §62 的「选项2 手工 dogfood 坐实」已发生） |

**结论**：候选 B 没解决问题。原因（来自调查，按概率排序）：

1. **(A 最可能) `visibilitychange` 在真机延迟/不可靠** → `useDocumentVisibility` 感知失败 → 照样走 deferred 积压。这正是 headless 测不出、必须真机 profile 才能区分的点。
2. **(B) 即便感知成功，后台期间 `messages` 本身已被 SSE 灌得很大** → 切回时 `groupMessages(messages)` 仍 O(n)、in-flight 大消息 markdown 重解析（Shiki 高亮）成本未减。
3. **(C/D) 113 张图 + 长对话**：虚拟化重测量 + 进度挂载在 visibility 恢复时级联。
4. **(E) Chrome 后台节流的 timer/rAF 债**切回补偿执行。

**deerflow 上游有无现成修复？——没有**：上游前端**根本不用 `useDeferredValue`**（那是我们 #232 独有优化），所以上游没这个积压问题、也无对应修复可拉。这是我们自引优化的坑，得自己解。（已 grep 上游无 visibilitychange/jank 相关 commit。）

---

## 一、红线（继承原 spec + Step0，不可违）

1. **不动 `useStream` / `mergeMessages` / dedupe**（流式核心，原 spec 红线）。
2. **dev build perf 不作数**——所有量化必须 prod build（`make start`）。
3. **未用 trace 坐实具体真凶前，不盲上任何修法**（守 `feedback_perf_is_efficient_impl_not_visual_downgrade`：perf=高效实现先量再改，不视觉降级、不 over-engineer）。
4. perf=同视觉省开销，**不许砍视觉/降流式逐字可见性**保性能。

---

## 二、Step 1（不可跳）：真机 profile 坐实真凶

headless 已证无效（Step0）。这一步**必须在有 GUI 的机器上、headed Chrome、OS 级真切 tab**。

### 操作
1. prod build 起：`cd packages/agent && make start`（确认非 dev：无 dev markers）。
2. dogfood：上传 EPM 多文件（如现有 thread `dcde1446-...` 的 28 个 xlsx）→ 发分析请求 → 跑到 **data-analyst 大段流式**（in-flight 消息 >15k chars）。
3. DevTools → Performance 面板**开始录制** → **用 OS 真切到别的窗口（VS Code）停 N 秒** → 切回 Chrome → 等卡顿结束 → 停录制。
4. 同时在 `use-document-visibility.ts` 临时加 `console.log(Date.now(), document.hidden, document.visibilityState)` + 在 message-list 加 `console.log("messages.length", messages.length)`，录 Console 时间线。

### 要从 trace 读出的判别（决定走哪个修法）
| 观察 | 指向真凶 | 对应修法 |
|---|---|---|
| `visibilitychange` 事件**晚于**切回数百 ms 才触发，或 `document.hidden` 在后台仍报 false | (A) hook 感知失败 | 修法 1（见下） |
| hidden 期间 Console 显示 `messages.length` 持续增长 | (B) 后台 SSE 确在灌 + 切回 O(n) flush | 修法 2 |
| 切回长任务里 "Evaluate Script" 大头是 Shiki/markdown parse | in-flight 大消息重解析 | 候选 A（冻结前缀） |
| 长任务大头是 virtualizer `measureElement` / `getTotalSize` | (D) 虚拟化重测量 | 修法 3 |
| 长任务集中在 timer/rAF 回调补偿 | (E) 节流债 | 需进一步定位排了什么 |

**产出**：一份 trace 数据小报告（落 `docs/superpowers/reports/`），明确「切回 N 秒卡顿的主线程 self-time 大头是哪类任务 + visibilitychange 时序」。**这份报告决定下面选哪个修法——不先出报告不写实现代码。**

---

## 三、Step 2：按 trace 选定修法（候选，trace 定生死）

> 以下都是「同视觉省开销」候选；**不在 Step1 坐实前实施**。trace 指向哪个就做哪个，可叠加。

### 修法 1（若真凶=A，visibilitychange 感知失败）
`useDocumentVisibility` 的感知不可靠是 #238 失效主因。改进方向：
- 不只依赖 `visibilitychange`，叠加 `window` 的 `blur`/`focus` 事件（OS 级窗口失焦比 tab 隐藏更早更可靠地触发）。
- 或：在 SSE 层（`hooks.ts`，**注意红线不动 useStream 核心逻辑**，只在其外围加 visibility 感知的节流闸）——但这碰红线，需谨慎，优先在渲染消费层解决。
- 文件：`use-document-visibility.ts`（已有配套 `.test.ts`，改后同步更新）+ `message-list.tsx` 消费点。

### 修法 2（若真凶=B，切回 O(n) flush + 大消息重解析）
即原 spec **候选 A**：in-flight 大消息「冻结前缀 + 活跃尾部」——已落盘的前缀进缓存不重解析，只重渲染正在变的尾部。
- 文件：`message-list.tsx`（in-flight 拆分）、`markdown-content.tsx`、扩展 `extraction-cache.ts`（in-flight 前缀也缓存）。
- 守原 spec 验收：终态与原渲染**字节级一致**（新单测断言）。

### 修法 3（若真凶=D，虚拟化重测量级联）
切回时抑制 virtualizer 的全量重测量（visibility 恢复时不触发 remount/remeasure，复用上次测量值）。
- 文件：`virtualized-groups.tsx`。

### 修法 C（兜底，仅当 A/2/3 都不足）
切回瞬间对超大 in-flight 消息临时降级渲染（纯文本骨架），下一帧再升级为完整 markdown。**最后手段**，因触及「视觉降级」红线边缘，仅在前述修法证实不够时用。

---

## 四、验收（prod build）

1. **Step1 trace 报告**先行：明确真凶 + 选定修法。
2. CDP perf（prod）：改后「切走→积压→切回」主线程长任务 self-time 较基线降 **≥50%**，切回后首帧显著下降，无 >Xms 长任务（X 由 Step1 基线定）。**dev build 不作数。**
3. **真机主观复核**：用户/agent 在 GUI 机器重跑 dogfood，切回不再卡十几秒（守「代码有修复≠现象消除」——必须真机现象消除才算过，见 `feedback_code_has_fix_not_equal_bug_eliminated_seal_react_floor`）。
4. 功能不回归：流式逐字可见性不降；`pnpm check` 0；`npx vitest run` 绿（含修法的新单测）。
5. 守红线：`useStream`/`mergeMessages`/dedupe 未碰（git diff 证）。

---

## 五、不做什么

- ❌ 不在 Step1 trace 坐实前写任何实现代码（守 Step0 报告的红线）。
- ❌ 不动流式核心（useStream/mergeMessages/dedupe）。
- ❌ 不砍视觉 / 不降流式逐字可见性换性能。
- ❌ 不去 deerflow 上游拉「修复」——已证上游无此问题、无对应修复。
- ❌ 不在 headless 复现（已证无效，必须真机）。

---

## 六、注意

1. **headless 陷阱**（Step0 血泪）：`Page.setWebLifecycleState('frozen')` ≠ 真实 hidden，会冻结整页 JS 致复现不出。**必须 headed + OS 级切窗口**。
2. **#238 已在 dev**：修法 1 是在它基础上改进 `useDocumentVisibility`，不是推倒——先判它到底有没有部分生效（trace 看 visibilitychange 时序）。
3. 这是 EPM 113 图 + 长对话的重负载场景，trace 时保持同等负载才有代表性。
