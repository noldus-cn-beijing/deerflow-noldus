# 2026-06-02 Spec — 前端上游现代化（上游 infra 基底 + Noldus 功能用上游新写法重新嫁接）

> **一句话目标**：把当前「停在旧 Noldus 版、故意没跟上上游重构」的前端，升级到 **「上游新代码做基底 + Noldus 功能（auto/flywheel、feedback 飞轮、quality-warning、clarification、stage-broadcast）用上游新写法重新嫁接」**。功能完全不变，底层结构现代化到上游最新。
>
> **与上一份 spec 的关系（必读）**：[`2026-06-02-frontend-sync-noldus-restoration-spec.md`](./2026-06-02-frontend-sync-noldus-restoration-spec.md) 做的是「恢复 dev 版止血」（sync 误覆盖 → tsc 58→0），是**临时**方案，**方向与本 spec 相反**。本 spec 是那份的「下一步」：不再退回 dev，而是前进吃上游。**本 spec 必须基于 #78 已 merge 进 dev 后的最新 dev 做**（已满足：#78 已 merge，本地 dev 已 pull rebase 到 PR #78 之上）。

---

## 0. 锁定的方向决策（2026-06-02 与用户对齐，不可动摇）

**现代化吃的是上游 infra，不是 mode 映射语义。**

- ✅ **吃上游 infra**：消息分组系统（getMessageGroups）、run messages 分页（before_seq lazy-load）、隐藏消息去重、token-usage 抽象层（api.ts/token-usage.ts/usage-model.ts）、onStart 回调扩展、UI 原语升级。
- 🔒 **不动 mode 语义**：前端 mode **保持 `auto/flywheel` 两态**，`workflow_mode` 字段**原样保留**。不扩成上游的 flash/thinking/pro/ultra 四态。
- **理由（flag 级实证，见 §2.1）**：我们的 auto/flywheel 本质就是上游 mode flag 空间里的两个特定点（`auto≈flash+固定subagent`，`flywheel≈ultra−按模式开关的subagent`）。flag 映射层面上游**没有我们该补的硬改进**；`workflow_mode` 是我们独有、上游没有的后端 Gate 承重墙（上游"删掉它"不是最佳实践，是它没有这个 Gate 概念）。所以 mode 这一块是「用上游新代码结构承载我们的两态语义」，**不是把产品特性也升级成四态**。

---

## 1. Context — 为什么做、做到什么程度

### 1.1 当前状态
- DeerFlow 是我们的 infra 底座（2026-06-02 策略锁定：全量跟随 + surgical 守护）。前端因 sync 时**无 protected-files 守卫**，被整文件覆盖 → 上一 agent 用 restoration spec 恢复回 dev 版止血。
- 结果：**后端 + 部署已是上游最新（Gateway 迁移完成），但前端停在旧 Noldus 结构**，没跟上上游对 hooks/utils/message-list/input-box 的重构。这是临时的、刻意标记为"待现代化"的状态。

### 1.2 上游 `deerflow/main@74e3e80c` 相对我们 dev 的 4 大演进（实证规模）
| 文件 | 上游行数 | dev 行数 | 上游引入的机制 |
|---|---|---|---|
| `src/core/threads/hooks.ts` | 1179 | 1122 | run messages 分页、隐藏消息去重、onStart 加 runId、token-usage 改用抽象层 |
| `src/components/workspace/input-box.tsx` | 976 | 640 | 四态 mode + DropdownMenu UI（**mode 部分我们不跟，UI 原语可借鉴**）|
| `src/components/workspace/messages/message-list.tsx` | 468 | 359 | 采纳 getMessageGroups 分组迭代 |
| `src/core/messages/utils.ts` | 581 | 537 | 消息分组系统（getMessageGroups/getStreamingMessageLookup/getAssistantTurnUsageMessages 等）|

### 1.3 范围边界
- **只动前端**。后端/部署/Gateway 迁移已完成验证，**不碰** `packages/agent/backend/**`。
- 本 spec 落地后，前端是「上游 infra 基底 + Noldus 功能嫁接」的自洽整体；mode 仍是 auto/flywheel。

---

## 2. 关键事实（全部已 git show / diff 现场核实，实施前仍建议复核关键点）

> 三方路径映射：我们 `packages/agent/frontend/X` ↔ `git show deerflow/main:frontend/X` ↔ `git show origin/dev:packages/agent/frontend/X`。
> Noldus 标记检测正则：`flywheel|quality|clarification|stageBroadcast|feedback|FeedbackVerdict|reasoning_effort|workflow_mode|[一-龥]`

### 2.1 mode→runtime flag 逐字段对比（核心证据，决定 mode 不动）

我们 dev（`hooks.ts:679-688`）：
```ts
thinking_enabled: context.reasoning_effort !== undefined || false,
is_plan_mode:     context.mode === "flywheel",
workflow_mode:    context.mode === "flywheel" ? "manual" : "auto",   // ← 上游没有此字段
subagent_enabled: true,                                              // ← 固定开（产品决定）
reasoning_effort: context.reasoning_effort ?? (context.mode === "flywheel" ? "high" : undefined),
```
上游（`hooks.ts:774-785`）：
```ts
thinking_enabled: context.mode !== "flash",
is_plan_mode:     context.mode === "pro" || context.mode === "ultra",
subagent_enabled: context.mode === "ultra",                          // ← 按 mode 开关
reasoning_effort: context.reasoning_effort ?? (ultra:"high" | pro:"medium" | thinking:"low" | _:undefined),
// 无 workflow_mode
```

| flag | 上游 flash/thinking/pro/ultra | 我们 auto / flywheel | 结论 |
|---|---|---|---|
| `thinking_enabled` | false/true/true/true | false / true | 等价 |
| `is_plan_mode` | false/false/true/true | false / true | 等价（flywheel≈pro/ultra）|
| `subagent_enabled` | false/false/false/true | **true / true（固定）** | **我们固定开，不跟上游按 mode 开关** |
| `reasoning_effort` | undefined/low/medium/high | undefined / high | 我们只用两端，不需要中间档 |
| `workflow_mode` | ❌ 无 | auto / manual | **我们独有，后端 Gate 承重墙，必保** |

→ **mode 映射层面无可吃的上游改进**。现代化只把这段映射**搬到上游 hooks.ts 新结构里**原样保留，不改语义。

### 2.2 上游 infra 新文件（我们 dev 没有，现代化要吃的）
| 上游文件 | 作用 | 处理 |
|---|---|---|
| `src/core/threads/api.ts` | `fetchThreadTokenUsage(threadId)` API 函数 | **吃**（B 档，token-usage 抽象）|
| `src/core/threads/token-usage.ts` | `threadTokenUsageQueryKey()` + `threadTokenUsageToTokenUsage()` 转换器 | **吃** |
| `src/core/messages/usage-model.ts` | token usage 数据模型 | **吃**（配套）|
| `src/components/workspace/messages/message-token-usage.tsx` | usage 渲染组件 | **吃**（配套，纯新组件无 Noldus 定制）|

### 2.3 上游整文件新增但**不吃**（用不上，吃进来是死代码）
`src/core/threads/static-mode.ts`、`src/core/threads/static-demo.ts`、`src/core/auth/static-user.ts`、`src/app/blog/**`、`src/app/[lang]/docs/**`、`src/components/landing/**`、`src/content/**`、`src/core/blog/index.ts`、`src/mdx-components.ts` 等——这些是上游的**不同业务功能**（演示模式、博客、文档站、它自己的认证/落地页），不是"最佳实践基础设施"。**全部跳过。**

### 2.4 Noldus 前端定制清单（现代化后必须 grep 验证全部健在）
- **auto/flywheel 模式** + `workflow_mode` 字段（`input-box.tsx` 的 `InputMode = "auto"|"flywheel"`、`hooks.ts` 的映射）
- **feedback 飞轮**：`api-client.ts` 的 `submitFeedback`/`listFeedback`/`FeedbackVerdict`/`FeedbackItem`/`FeedbackRequest`；`feedback-buttons.tsx`；`hooks.ts` 的 `mergedMessageRunIds`（每条消息 run_id）
- **quality-warning**：`quality-warning-banner.tsx`；`messages/utils.ts` 的 `QualityWarning`/`extractQualityWarnings`/`hasQualityWarnings`
- **clarification**：`messages/utils.ts` 的 `stripClarificationOptionsFromContent`；`clarification-options.tsx`
- **stage-broadcast**：`core/tools/stage-broadcast.ts` + i18n `stageBroadcast` key
- **中文 i18n key**：`autoMode`/`autoModeDescription`/`flywheelMode`/`flywheelModeDescription`/`clarification`/`stageBroadcast`/`taskResult`/`taskDescription`/`feedback`（workspace 命名空间）
- **Noldus 自有 token-usage**：dev `hooks.ts` 当前用自己的 queryKey（`hooks.ts` 注释明说不用上游 fetchThreadTokenUsage）→ 现代化后**改用上游 api.ts/token-usage.ts**，但要确认数据形状对得上（见 §3 B3）
- **Gateway 改造**：`next.config.js` 的 langgraph rewrite 指向 gateway（**不碰**）；`artifacts/utils.ts` 的 `normalizeArtifactImageSrc`、`artifact-file-detail.tsx` 的三路合并、`artifacts/preview.ts`（这些 #78 已合好，**不退回**）

---

## 3. 实施分档（A/B/C）

> 总原则：**A 先做（低风险打底）→ B 再做（嫁接，逐文件 typecheck）→ C 收尾（可选）**。每改一档跑一次 `pnpm typecheck`，绿了再下一档。

### A 档：吃下上游对「无 Noldus 定制文件」的改进（低风险，直接取上游）

判定方法：`git show origin/dev:packages/agent/frontend/<path> | grep -cE "<Noldus正则>"` == 0 **且** dev 与上游有 diff（上游领先）→ 直接取上游版。

候选（实施时先 grep 确认 Noldus 标记为 0 再取）：
- 消息分组系统的**纯新导出**部分（并入 utils.ts，见 B3，因为 utils.ts 同时含 Noldus 导出，**不能裸覆盖**——A 档只取其中无定制的新函数体）
- UI 原语 / 通用组件（`code-block.tsx`/`copy-button.tsx`/`streamdown/*` 等 dev 无 Noldus 标记的）——若上游领先则取上游
- §2.2 的 token-usage infra 新文件（`api.ts`/`token-usage.ts`/`usage-model.ts`/`message-token-usage.tsx`）：**整文件从上游取**（`git checkout deerflow/main -- frontend/...` 后挪到我们路径，或手抄），它们是纯新增、无 Noldus 定制。

> ⚠️ A 档"直接取上游"仅限**确认无 Noldus 定制**的文件。任何 grep 命中 Noldus 标记的，一律走 B 档嫁接。

### B 档：在上游新基底上重新嫁接 Noldus 功能（中高风险，逐文件手工）

#### B1 `src/core/threads/hooks.ts` —— **最难，最高风险**
**上游新结构要吃**：
- run messages 分页：`buildRunMessagesUrl`/`getOldestRunMessageSeq`/`getNextRunMessagesBeforeSeq`/`runMessagesPageHasMore`
- 隐藏消息去重：`isHiddenFromUIMessage` + `lastVisibleIndexByIdentity`
- `onStart` 回调扩展：`(threadId, runId) => void`（+ 可能的 `onSend`）
- token-usage 改用 §2.2 的 `api.ts`/`token-usage.ts`（替换 dev 内嵌的 `queryKey: ["threads/usage", threadId]`）

**Noldus 必保（原样搬进新结构，不改语义）**：
- mode→context 映射的 5 行（§2.1 dev 版），**含 `workflow_mode`**。
- `mergedMessageRunIds`（feedback 飞轮用，每条消息绑 run_id）——**这是与上游"隐藏消息去重"最易冲突的点**：去重逻辑改变 message↔run_id 绑定时，必须确认 `mergedMessageRunIds` 仍准确。
- 返回值：dev 是 tuple `[mergedThread, sendMessage, isUploading, mergedMessageRunIds]`；上游是 object。**建议保 tuple 但扩展**（或改 object 但所有消费方同步改——见 §3 全局调用点）。以 typecheck + 消费方不破为准。

**嫁接顺序建议**：先以**上游 hooks.ts 为基底**复制，再逐块把 dev 的 Noldus 段（mode 映射 / workflow_mode / mergedMessageRunIds / 固定 subagent_enabled / 返回值形状）改回来。**不要**反过来（在 dev 基底上贴上游片段——会漏掉上游新机制）。

#### B2 `src/components/workspace/input-box.tsx` —— 中风险
- **mode 保两态**：`type InputMode = "auto" | "flywheel"`（不改四态）。
- 可借鉴上游的 UI 原语（DropdownMenu、mode 支持检查的写法），但选项只有 auto/flywheel，文案保中文（autoMode/flywheelMode）。
- `ModeHoverGuide` 改回/保持支持 auto/flywheel 的文案。
- **判断**：如果上游 UI 升级对我们两态收益不大，B2 可最小化——只保证 input-box 与新 hooks.ts 的接口对齐即可，UI 不强行升级。以"功能不变 + typecheck 绿"为底线。

#### B3 `src/core/messages/utils.ts` —— 中风险（不能裸覆盖）
- **吃上游新导出**：`getMessageGroups`/`getStreamingMessageLookup`/`getAssistantTurnUsageMessages`/`getAssistantTurnCopyData`/`isAssistantMessageGroupStreaming`/`stripInternalMarkers`/`INTERNAL_MARKER_TAGS`/`MessageGroup` 类型等。
- **保 Noldus 导出**：`stripClarificationOptionsFromContent`/`findToolCallArgs`/`QualityWarning`/`extractQualityWarnings`/`hasQualityWarnings`/`findToolCallResult`。
- 做法：以**上游 utils.ts 为基底**，把 dev 独有的 5~6 个 Noldus 导出补进去。**两套并存**（分组系统用于聚合渲染，clarification/quality-warning 是独立渲染规则，互不干扰）。

#### B4 `src/components/workspace/messages/message-list.tsx` —— 中风险
- 采纳上游 `getMessageGroups` 分组迭代（替换 dev 的扁平遍历）。
- **保 Noldus 渲染分支**：clarification-options、feedback-buttons、quality-warning-banner、stage-broadcast 的条件渲染必须在分组迭代里**正确触发**（这是"能编译≠跑对"的高发区，见 §4）。
- 配套：`message-group.tsx`/`message-list-item.tsx`/`subtask-card.tsx`/`clarification-options.tsx` 跟随上游分组系统调整（按 typecheck 驱动）。

#### B5 i18n —— 低风险机械
- 以上游 `i18n/locales/{en-US,zh-CN,types}.ts` 为基底（吃上游可能新增的 key），**补回 9 组 Noldus key**（§2.4）。types.ts 同步加类型。

#### B6 `api/api-client.ts` —— 低风险机械
- 以上游为基底，**补回 feedback 飞轮导出**（`submitFeedback`/`listFeedback`/`FeedbackVerdict`/`FeedbackItem`/`FeedbackRequest`）。注意上游也新增了 `api/feedback.ts`（§2.2 清单里有）——确认我们的 feedback 飞轮是走 api-client 还是要并入上游 feedback.ts，**以不破坏后端 `/api/threads/{tid}/runs/{rid}/feedback` 契约为准**（后端 SQLite verdict 三分类 + revised_text，不能改）。

### C 档：可选现代化（低优先，可留 v0.2）
- run messages 分页（`before_seq`）的**完整 UI 接入**（"加载更多历史"按钮等）——基础函数在 B1 已吃，但 UI 触点可后置。Noldus 短周期工作流消息不多，优先级低。
- 若 B 档已把 token-usage 切到上游抽象，C 档无额外项；否则 token-usage 切换留 C。

### 不做（明确跳过，见 §2.3）
static-demo/static-mode/auth-static-user/blog/docs/landing/content/mdx——全部不吃。

---

## 4. 风险点与验收（能编译 ≠ 跑对）

> 现代化最大的陷阱：**前端改完 tsc 全绿，但运行时 Noldus 功能静默失效**。以下每条必须 dogfood 实测，不能只看编译。

### 🔴 风险 1：mode→flag 映射丢 workflow_mode → flywheel 静默失效
- **症状**：编译过，但后端 `GateEnforcementMiddleware` 收不到 `workflow_mode="manual"` → Gate 1 范式反问永不激活。
- **验证**：dogfood 选 flywheel 发消息，DevTools Network 看 `POST .../runs` 请求体 `context` 含 `workflow_mode:"manual"` + `is_plan_mode:true` + `reasoning_effort:"high"`；选 auto 时 `workflow_mode:"auto"`。

### 🔴 风险 2：隐藏消息去重破坏 mergedMessageRunIds → feedback 飞轮断
- **症状**：feedback 按钮无法对应到正确 run_id，或刷新后历史消息无法反馈。
- **验证**：发消息→确认每条 AI 消息绑到正确 run_id；feedback 三按钮提交成功（`POST .../runs/{rid}/feedback`）；刷新页面后历史消息仍可反馈。

### 🟡 风险 3：分组系统改变迭代顺序 → clarification 双重分组失效
- **验证**：触发 Gate 1 范式 clarification，确认 clarification-options 组件出现且可点选提交。

### 🟡 风险 4：utils.ts 嫁接漏导出 → quality-warning 不渲染
- **验证**：上传有质量问题的数据（如缺列），报告顶部出现 quality-warning-banner。

### 🟡 风险 5：stage-broadcast i18n key 丢失 → 阶段进度不显示
- **验证**：跑分析，确认各阶段（分析/清洗/报告）的 stage-broadcast 进度提示正常。

---

## 5. 验收标准（Definition of Done）

### 编译/构建
- [ ] `cd packages/agent/frontend && pnpm typecheck` exit 0，0 个 `error TS`。
- [ ] `pnpm build` 成功（dev server 真正能起的最强信号）。
- [ ] `pnpm lint` 无 error。

### Noldus 定制健在（grep 验证，不是只看编译）
- [ ] `grep -rn 'flywheel' .../input-box.tsx` 有命中；`grep -n 'workflow_mode' .../hooks.ts` 有命中。
- [ ] `api-client.ts`（或并入的 feedback 模块）有 `submitFeedback`/`FeedbackVerdict`；`feedback-buttons.tsx` 在。
- [ ] `quality-warning-banner.tsx` 存在 + `messages/utils.ts` 有 `extractQualityWarnings`。
- [ ] i18n 有 `clarification`/`stageBroadcast`/`autoMode`/`flywheelMode` 等 9 组 key。
- [ ] `hooks.ts` 有 `mergedMessageRunIds`。

### 上游 infra 确实吃进来了（不是只保留 dev）
- [ ] `messages/utils.ts` 有上游 `getMessageGroups`/`getStreamingMessageLookup`。
- [ ] `message-list.tsx` 用 `getMessageGroups` 迭代（不是旧扁平遍历）。
- [ ] `hooks.ts` 有 run messages 分页函数 + 隐藏消息去重 + onStart(threadId, runId)。
- [ ] token-usage 走上游 `threads/api.ts`/`token-usage.ts`（除非 C 档后置，需在 handoff 注明）。

### Gateway 迁移未被退回
- [ ] `next.config.js` langgraph rewrite 仍指 gateway；`artifact-file-detail.tsx` 同时含 `normalizeArtifactImageSrc` 和 `appendHtmlPreviewScrollRestoration`；`artifacts/preview.ts` 存在。

### 运行时 dogfood（§4 全部 5 条 + 端到端）
- [ ] auto↔flywheel 切换：后端实收对应 `workflow_mode`（DevTools 实证）。
- [ ] EV19 短流程跑通：上传→Gate 1 clarification→code-executor→data-analyst→report，无 artifact 404。
- [ ] feedback 飞轮端到端：提交→feedback 列表可见→刷新后历史仍可反馈。
- [ ] quality-warning / stage-broadcast 渲染正常。

### 后端未动
- [ ] `git status` 只应有 `packages/agent/frontend/**` 改动（不碰 backend）。

---

## 6. 反模式（别踩）

- ❌ **把 mode 扩成上游四态 flash/thinking/pro/ultra**。auto/flywheel 是已锁定的行为学产品特性（见 memory `feedback_version_boundary_*` 与本 spec §0）。
- ❌ **删 `workflow_mode` 跟随上游**。它是后端 Gate 承重墙，上游没有 ≠ 不需要。
- ❌ **把 `subagent_enabled` 改成按 mode 开关**。我们固定 true（行为学流水线永远要 subagent）。
- ❌ **裸 `git checkout` 覆盖含 Noldus 定制的文件**（hooks/utils/api-client/i18n/input-box）。这些走 B 档手工嫁接，不能整文件取上游。
- ❌ **裸覆盖 `next.config.js`/`artifacts/utils.ts`/`artifact-file-detail.tsx`/`artifacts/preview.ts`**。承载 Gateway 改造 + PR #75 + 三路合并（#78 已合好）。
- ❌ **吃上游 static-demo/blog/docs/landing/auth 整套**（§2.3）。死代码 + 维护负担。
- ❌ **只看 tsc 绿就算完**。§4 的运行时行为必须 dogfood（编译过 ≠ 功能没断）。
- ❌ **在 dev 基底上贴上游片段**。B 档要以**上游文件为基底**再改回 Noldus 段，反过来会漏上游新机制。

---

## 7. 执行环境与命令速查

> **在哪做**：建议开 worktree（基于最新 dev，即 #78 已 merge 之后）。本 spec 文件本身写在 dev（纯文档）。worktree 前端需 `pnpm install` 或软链主仓库 node_modules（仅 typecheck 用，完事删，别提交）。

```bash
# 三方对比某文件
diff <(git show deerflow/main:frontend/<path>) <(git show origin/dev:packages/agent/frontend/<path>)

# 上游某文件行数 vs dev
git show deerflow/main:frontend/<path> | wc -l
git show origin/dev:packages/agent/frontend/<path> | wc -l

# 判定某文件是否含 Noldus 定制（>0 = 含，走 B 档嫁接）
git show origin/dev:packages/agent/frontend/<path> | grep -cE "flywheel|quality|clarification|stageBroadcast|feedback|FeedbackVerdict|reasoning_effort|workflow_mode|[一-龥]"

# 取上游某「纯新文件」（A 档 infra）
git show deerflow/main:frontend/src/core/threads/api.ts > packages/agent/frontend/src/core/threads/api.ts

# typecheck（worktree 软链 node_modules 后）
( cd packages/agent/frontend && node_modules/.bin/tsc --noEmit --pretty false 2>&1 | grep -c "error TS" )
```

路径映射：我们 `packages/agent/frontend/X` ↔ `git show deerflow/main:frontend/X` ↔ `git show origin/dev:packages/agent/frontend/X`。
上游基线：`deerflow/main@74e3e80c`（= 上次 sync 基线，`.deerflow-sync-state`）。

---

## 8. 工时估计（参考，约 18-22h / 2-3 天）

- A 档（infra 打底）：token-usage 4 文件吃下 + 通用原语（~3h）
- B 档：types.ts/i18n（1.5h）→ input-box（1-2h）→ **hooks.ts（4-5h，高风险）** → messages/utils.ts（3-4h）→ message-list + 配套（2-3h）→ api-client/feedback（1h）
- 全局调用点适配（hooks 返回值形状变更的消费方）（2h）
- 本地 dogfood 验收（§4 五条 + 端到端）（3-4h）

---

## 9. 关联文档 / memory
- 前置（临时止血，方向相反）：[`2026-06-02-frontend-sync-noldus-restoration-spec.md`](./2026-06-02-frontend-sync-noldus-restoration-spec.md)
- sync 设计：[`2026-06-02-deerflow-upstream-sync-design.md`](./2026-06-02-deerflow-upstream-sync-design.md)
- PR #78 交接（含 §4 现代化调研素材）：`docs/handoffs/2026-06/2026-06-02-pr78-merge-conflict-resolution-handoff.md`
- 善后待办（本 spec 的同源根因）：sync 脚本加**前端 PROTECTED_FILES**（hooks.ts/messages/utils.ts/api-client.ts/i18n/input-box.tsx/feedback-buttons.tsx/quality-warning-banner.tsx/stage-broadcast.ts/settings/local.ts/threads/utils.ts/threads/types.ts），+ SOP 加「sync 后必跑前端 pnpm typecheck」。关联 memory `feedback_sync_protected_files_registry_loss` 应扩展到前端。
- 版本边界：v0.1 仍是 insight harness；本现代化属 infra 跟随，不触愿景层。
