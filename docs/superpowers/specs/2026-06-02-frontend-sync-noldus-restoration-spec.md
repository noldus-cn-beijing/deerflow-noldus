# 2026-06-02 Spec — 修复 DeerFlow 全量 sync 对前端 Noldus 定制的覆盖（typecheck 回绿）

> **执行环境**：sync worktree `/home/wangqiuyang/noldus-insight/.claude/worktrees/sync-0602`，分支 `worktree-sync-0602`（PR #77）。**不要在主仓库改**。
> **目标**：把 worktree 前端从 **58 个 tsc 错误** 修到 **`pnpm typecheck` 全绿**，同时保住 Noldus 前端定制（auto/flywheel 模式、feedback 飞轮、quality-warning、clarification、stage-broadcast），不丢上游对*非定制*文件的真实改进。
> **不碰**：本 spec 只处理前端。Gateway 迁移（后端 + 部署文件）已完成且验证通过（后端 3262 passed / 0 failed，`test_gateway_runtime_cleanup` 8 测试全绿），**不要回退或重做任何 Gateway 迁移改动**。

---

## 1. Context — 为什么要做这件事

PR #77 是 DeerFlow「全量跟随上游」sync（`fa3418ec`，f9b70713→74e3e80c，34 commit）。后端有 `scripts/sync-deerflow.sh:51` 的 22 个 PROTECTED_FILES 守卫，所以后端只崩了少数受保护文件（已 surgical 修好）。**但前端没有任何 protected-files 守卫**，于是 sync 对前端 Noldus 定制文件**整文件取了上游版本**，把 Noldus 的定制覆盖掉了。

结果是一次「精神分裂式不一致合并」：

- sync 对一批 **plumbing/依赖文件** 取了**上游**（丢了 Noldus 的导出/i18n key/模式逻辑）：`hooks.ts`、`messages/utils.ts`、`api/api-client.ts`、`i18n/locales/{en-US,zh-CN,types}.ts` 等
- sync 对一批 **消费组件** 保留了 **dev/Noldus**：`input-box.tsx`、`message-list.tsx`、`subtask-card.tsx`、`feedback-buttons.tsx`、`recent-chat-list.tsx`、`stage-broadcast.ts` 等
- 还**整文件删除**了一个 dev-only Noldus 文件：`quality-warning-banner.tsx`

→ 保留的 Noldus 组件去引用被丢掉的 Noldus helper / i18n key / API 导出 → **58 个 type error / 18 个文件**。

**基线证据**：主仓库 dev HEAD 前端 `tsc --noEmit` = **0 错误**。worktree = 58 错误。所以这 58 个是**纯 sync 回归**，不是项目固有问题。dev 的前端是一个**自洽的整体**。

**修复方向（已现场核实，关键）**：**对「被覆盖的 Noldus 定制文件」恢复 dev 版本**（前端版的 protected-files 还原），**不是**把 `local.ts`/组件正向迁到上游的 flash/pro/ultra 枚举。理由见 §3 的方向论证。

---

## 2. 关键事实（全部已 grep/diff 现场核实，新 agent 可直接信任，但仍建议自己复核一遍）

### 2.1 模式系统冲突的真相（最容易改错的地方）
- 上游模式枚举：`"flash" | "thinking" | "pro" | "ultra"`；上游 `hooks.ts:774-783` 据此映射 `thinking_enabled = mode !== "flash"` / `is_plan_mode = mode === "pro" || "ultra"` / `subagent_enabled = mode === "ultra"`。
- **Noldus（dev）模式枚举：`"auto" | "flywheel"`**，是**刻意的 Noldus 产品特性**（`input-box.tsx:74 type InputMode = "auto" | "flywheel"`）。dev `hooks.ts:679-688` 有完全不同的映射：`is_plan_mode = mode === "flywheel"` / `workflow_mode = mode === "flywheel" ? "manual" : "auto"` / `reasoning_effort = ... mode === "flywheel" ? "high" : undefined`。
- **sync 误取了上游 `hooks.ts`（flash/pro/ultra），但保留了 Noldus `local.ts` + `input-box.tsx`（auto/flywheel）** → 二者不兼容 → hooks.ts 报 7 个 TS2367「no overlap」。
- ✅ **正解：`hooks.ts` 恢复 dev 版**（auto/flywheel 自洽），**不是**把 `local.ts` 改成上游枚举（那会砸掉 Noldus 模式功能）。
- `settings/local.ts` 现在 `up=44 dev=0`（已经是 Noldus 版），**无需改动**。

### 2.2 sync 误取上游、丢掉的 Noldus 导出（必须随文件恢复 dev 而回来）
- `messages/utils.ts`：dev 独有 `stripClarificationOptionsFromContent`、`findToolCallArgs`、`QualityWarning`、`extractQualityWarnings`、`hasQualityWarnings`。**注意**：上游 utils.ts 也有大量 dev 没有的新导出（`getMessageGroups`/`getStreamingMessageLookup`/`stripInternalMarkers`/`INTERNAL_MARKER_TAGS` 等，是上游新的 message-grouping 系统）。见 §3 对该文件的处理决策。
- `api/api-client.ts`：dev 独有 `submitFeedback`、`listFeedback`、`FeedbackItem`、`FeedbackRequest`、`FeedbackVerdict`（feedback 飞轮）。
- `i18n/locales/{en-US,zh-CN}.ts`：dev 独有 key `autoMode`、`autoModeDescription`、`flywheelMode`、`flywheelModeDescription`、`clarification`、`stageBroadcast`、`taskResult`、`taskDescription`、`feedback`（workspace 命名空间）。`i18n/locales/types.ts` 同步需要 dev 版（key 的类型定义）。

### 2.3 sync 整文件删除的 Noldus 文件（必须从 dev 恢复）
- `src/components/workspace/messages/quality-warning-banner.tsx`（worktree 不存在，dev 存在）。

### 2.4 dev 不依赖的「上游新文件」→ 走 dev 方向后这些**不需要**补
sync 漏带了上游新文件 `src/core/static-mode.ts`、`src/core/threads/api.ts`、`src/core/threads/token-usage.ts`、`src/core/threads/static-demo.ts`。**但已核实 dev 的 `hooks.ts` / `api-client.ts` 根本不 import 它们**（dev 用自己的 token-usage 实现，`hooks.ts:1017` 注释明说 "Upstream uses fetchThreadTokenUsage... 我们不用"）。所以**恢复这些文件到 dev 版后，对这些缺失上游文件的引用一并消失，无需补这些文件**。

### 2.5 我（上一个 agent）已经在 worktree 做的前端改动（不要覆盖掉）
为修 PR #77 的 artifact 三路冲突，我已经在 worktree 改了 2 个文件（**这些是 Gateway/冲突修复的一部分，要保留**）：
- `src/core/artifacts/utils.ts`：加了 `normalizeArtifactImageSrc`（dev PR #75 修复）。当前 `up=20 dev=47`。
- `src/components/workspace/artifacts/artifact-file-detail.tsx`：三路合并（上游 scroll-restoration + dev 的 img-rewrite）。
- `src/core/artifacts/preview.ts`：我已从上游恢复（sync 漏带的新文件，artifact tsx 依赖它）。**保留**。
- `next.config.js`：我已做 Gateway 改造（langgraph rewrite 指向 gateway）。**保留，不要碰**。

→ **新 agent 处理 `artifacts/utils.ts` 和 `artifact-file-detail.tsx` 时要特别小心**：不能简单 `git checkout origin/dev` 覆盖，否则丢掉上游 scroll-restoration + 我的 Gateway 无关合并。见 §3 对这两个文件的专门说明。

---

## 3. 修复策略 — 「恢复 Noldus 定制文件到 dev 版」+ 精准例外

**核心原则**：对**确有 Noldus 定制**的前端文件，恢复 dev 版（`git checkout origin/dev -- <path>`，注意路径前缀）。对**无 Noldus 定制、纯上游领先**的文件，**保持 worktree（上游）版不动**（别为还原 dev 而丢上游改进）。

### 3.1 判定某文件「是否 Noldus 定制」的方法
对每个候选文件：
```bash
# Noldus 标记检测（dev 版里有这些 = 是 Noldus 定制 = 恢复 dev）
git show origin/dev:packages/agent/frontend/<path> | grep -cE \
  "flywheel|quality|qualityWarning|clarification|stageBroadcast|feedback|FeedbackVerdict|reasoning_effort|workflow_mode|[一-龥]"
```
- 命中 >0 → Noldus 定制 → 恢复 dev 版。
- 命中 0 且 `diff vs 上游 == 0`（worktree 就是上游）→ 不是 Noldus 定制 → **保留 worktree 不动**。

### 3.2 A 类：直接 `git checkout origin/dev` 恢复（确认有 Noldus 定制、且我没动过）

> 命令模板（在 worktree 根执行）：`git checkout origin/dev -- packages/agent/frontend/<path>`

按当前已核实的清单，**恢复以下文件到 dev 版**（已确认 dev 版含 Noldus 定制、worktree 当前是被覆盖的上游版、且不在 §2.5 我动过的列表里）：

| 文件 | 证据（dev=N 表示距 dev 行数，越大说明被覆盖越多）|
|---|---|
| `src/core/threads/hooks.ts` | dev=519，含 auto/flywheel 映射 + Noldus token-usage |
| `src/core/messages/utils.ts` | 见 §3.4（**特殊**，不能裸 checkout）|
| `src/core/api/api-client.ts` | dev=167，含 feedback 导出 |
| `src/core/i18n/locales/en-US.ts` | dev=148，含 9 组 Noldus key |
| `src/core/i18n/locales/zh-CN.ts` | dev=158 |
| `src/core/i18n/locales/types.ts` | dev=98，key 类型 |
| `src/core/threads/types.ts` | dev≠0，含 Noldus 标记 ×2 |
| `src/components/workspace/input-box.tsx` | dev=0（已是 Noldus，但报错因依赖）→ 实际**无需改**，依赖修好即绿。仍建议确认 |
| `src/components/workspace/messages/message-list.tsx` | dev=0 → 无需改，依赖修好即绿 |
| `src/components/workspace/messages/subtask-card.tsx` | dev=0 → 无需改 |
| `src/components/workspace/messages/message-group.tsx` | dev=0 → 无需改 |
| `src/components/workspace/messages/clarification-options.tsx` | dev=0 → 无需改 |
| `src/components/workspace/messages/markdown-content.tsx` | dev=17 → 恢复 dev（rehypePlugins 默认值在 dev 版）|
| `src/components/workspace/recent-chat-list.tsx` | dev=43，pathOfThread overload 用法 |
| `src/components/workspace/mode-hover-guide.tsx` | dev=0 → 无需改 |
| `src/components/workspace/workspace-nav-chat-list.tsx` | dev=0 → 无需改 |
| `src/components/feedback/feedback-buttons.tsx` | dev=0 → 无需改（依赖 api-client feedback 导出）|
| `src/core/tools/stage-broadcast.ts` | dev=0 → 无需改（依赖 i18n stageBroadcast）|
| `src/app/workspace/agents/new/page.tsx` | dev=63，用 auto/flywheel + pathOfThread |
| `src/app/workspace/chats/[thread_id]/page.tsx` | dev=0 → 无需改（依赖 hooks 的 tuple 返回）|
| `src/app/workspace/agents/[agent_name]/chats/[thread_id]/page.tsx` | dev=0 → 无需改 |
| `src/core/threads/utils.ts` | dev=0 → 已是 Noldus（pathOfThread overloads 在此）。确认即可 |

> **重要**：很多报错文件本身 `dev=0`（已经是 Noldus 版），它们的错误是**依赖**被覆盖造成的。把依赖（hooks/utils/api-client/i18n）恢复 dev 后，这些 `dev=0` 文件**自动变绿，无需 checkout**。新 agent 应**先恢复依赖文件，再 typecheck**，剩下的错误才逐个看。

### 3.3 B 类：恢复 sync 删除的 Noldus 文件
```bash
git checkout origin/dev -- packages/agent/frontend/src/components/workspace/messages/quality-warning-banner.tsx
```

### 3.4 C 类：特殊文件，**不能裸 checkout**，需判断

**`src/core/messages/utils.ts`** —— 这是最棘手的。dev 和上游**都有对方没有的导出**：
- dev 独有（Noldus，必须在）：`stripClarificationOptionsFromContent`、`findToolCallArgs`、`QualityWarning`、`extractQualityWarnings`、`hasQualityWarnings`。
- 上游独有（新 message-grouping 系统）：`getMessageGroups`、`getAssistantTurnUsageMessages`、`getStreamingMessageLookup`、`isAssistantMessageGroupStreaming`、`getAssistantTurnCopyData`、`stripInternalMarkers`、`INTERNAL_MARKER_TAGS` 等。
- **判定哪边对**：看**保留下来的 Noldus 组件**（message-list/message-group/subtask-card，它们是 dev 版）import 的是哪一套。
  ```bash
  # 这些 dev 组件 import 了 utils 的哪些符号？
  for f in message-list message-group message-list-item subtask-card clarification-options; do
    echo "== $f =="; git show origin/dev:packages/agent/frontend/src/components/workspace/messages/$f.tsx | grep -A20 'from "@/core/messages/utils"' | grep -oE '\b[a-zA-Z]+\b' | head -30
  done
  ```
- **大概率结论**：dev 组件用的是 dev 版 utils 的符号集 → **恢复 dev 版 utils.ts**（`git checkout origin/dev -- .../messages/utils.ts`），上游那套 message-grouping 导出是给上游组件用的，而我们用的是 dev 组件，所以不需要。**但必须先验证**：恢复 dev utils 后，typecheck 不能冒出「上游某保留文件找不到 getMessageGroups」之类的新错误。若冒出，说明有上游组件被保留了 → 那个组件也要恢复 dev，或保留上游 utils 并补 dev 的 5 个 Noldus 导出（二选一，以 typecheck 全绿为准）。

**`src/core/artifacts/utils.ts`** 和 **`src/components/workspace/artifacts/artifact-file-detail.tsx`** —— 见 §2.5，**我已经手工合并好**（上游 scroll-restoration + dev 的 normalizeArtifactImageSrc）。
- **默认不要动这两个**。先跑 typecheck，如果这两个文件**没报错**，就别碰。
- 如果报错，**不能裸 checkout dev**（会丢上游 scroll-restoration + preview.ts 那条线）。需手工：保证 `normalizeArtifactImageSrc` 在 utils.ts、`preview.ts` 存在、artifact-file-detail.tsx 同时 import 两边。参照 git 历史里我这次的改动。

### 3.5 D 类：Bucket C，test-only，**dogfood 不阻塞，可暂留**
- `src/core/messages/utils.test.ts`（2 错）、`tests/e2e/thread-history-mermaid.spec.ts`（2 错，缺 `@playwright/test` dev-dep + `tests/e2e/utils/mock-api`）。
- 这些**不影响 `make dev` / `pnpm dev`**（dev server 不编译 test）。
- 处理：`utils.test.ts` 多半随 §3.4 恢复 dev utils.ts 后自动绿（它测的是 dev 的函数）。`thread-history-mermaid.spec.ts` 是上游新带进来的 E2E，依赖未装的 `@playwright/test` + 上游 `tests/e2e/utils/mock-api`——**最省事**是从上游补 `tests/e2e/utils/mock-api.ts`（`git checkout deerflow/main -- ...`，注意上游路径 `frontend/tests/...`），`@playwright/test` 装不装取决于 tsconfig 是否 include tests。**若 `pnpm typecheck` 的 tsconfig 不含 tests/e2e，这两个错根本不出现，忽略即可。** 先看 tsconfig include。

---

## 4. 执行步骤（建议顺序）

> ⚠️ worktree 前端**没装 node_modules**。typecheck 需要 node_modules。两个办法：
> - **(A) 推荐**：`cd packages/agent/frontend && pnpm install`（package.json 与 dev 一致，几分钟）。
> - **(B) 快速**：软链主仓库 node_modules（**仅 typecheck 用，完事删掉，别提交**）：
>   `ln -sfn /home/wangqiuyang/noldus-insight/packages/agent/frontend/node_modules packages/agent/frontend/node_modules`（已验证 package.json 一致，可用于 `tsc --noEmit`）。结束务必 `rm -f packages/agent/frontend/node_modules`。

1. **基线**：`cd packages/agent/frontend && node_modules/.bin/tsc --noEmit --pretty false 2>&1 | grep -c "error TS"` → 应为 58。
2. **恢复依赖层**（§3.2 A 类里 dev≠0 的 + §3.4 utils.ts + §3.3 quality-warning-banner + i18n 三件 + hooks/api-client/types）：逐个 `git checkout origin/dev -- packages/agent/frontend/<path>`。
3. **typecheck**：看错误从 58 掉到多少。**预期大部分消失**（依赖修好，dev=0 的组件自动绿）。
4. **逐个清剩余**：对剩下的报错文件，按 §3.1 判定方向；`dev=0` 的不该再报错（报错说明还有依赖没恢复），`dev≠0` 的恢复 dev。
5. **守住 §2.5**：确认 `next.config.js`、`artifacts/utils.ts`、`artifact-file-detail.tsx`、`artifacts/preview.ts` 没被退回（它们承载 Gateway 改造 + PR #75 + 三路合并）。
6. **全绿确认**：`pnpm typecheck` exit 0（或 `tsc --noEmit` 0 error，test-only 的 Bucket C 若 tsconfig 不含可忽略）。
7. **删软链**（若用了办法 B）：`rm -f packages/agent/frontend/node_modules`。
8. **lint**（可选但建议）：`pnpm lint`（或 `node_modules/.bin/eslint`）。

---

## 5. 验收标准（Definition of Done）

- [ ] `cd packages/agent/frontend && pnpm typecheck`（即 `tsc --noEmit`）**exit 0**，0 个 `error TS`（test-only 的 2 个若 tsconfig 不含 tests 则不计；若含，也要修到 0）。
- [ ] **Noldus 前端定制全部健在**（grep 验证，不是只看编译过）：
  - auto/flywheel 模式：`grep -rn 'flywheel' packages/agent/frontend/src/components/workspace/input-box.tsx` 有命中。
  - feedback 飞轮：`api-client.ts` 有 `submitFeedback`/`FeedbackVerdict`；`feedback-buttons.tsx` 在。
  - quality-warning：`quality-warning-banner.tsx` 存在 + `messages/utils.ts` 有 `extractQualityWarnings`。
  - clarification / stage-broadcast：i18n 有 `clarification`/`stageBroadcast` key。
- [ ] **Gateway 迁移改动未被退回**：`next.config.js` 的 langgraph rewrite 仍指 gateway（`grep -n 'gatewayURL}/api' packages/agent/frontend/next.config.js`）；`artifact-file-detail.tsx` 同时含 `normalizeArtifactImageSrc` 和 `appendHtmlPreviewScrollRestoration`；`artifacts/preview.ts` 存在。
- [ ] **后端没被动**：不需要重跑后端全量，但确认本次没改 `packages/agent/backend/**`（`git status` 只应有 frontend 改动）。
- [ ] worktree 没残留软链 node_modules（`git status` 不应出现 `frontend/node_modules`）。
- [ ] （推荐）`pnpm build` 跑通一次（dev server 真正能起的最强信号）；若太慢，至少 `pnpm typecheck` 全绿 + `pnpm lint` 无 error。

---

## 6. 反模式（别踩）

- ❌ **把 `local.ts` / 组件正向迁到上游 flash/thinking/pro/ultra 枚举**。auto/flywheel 是 Noldus 产品特性，方向反了（上一个诊断 agent 在这里判错过）。
- ❌ **裸 `git checkout origin/dev` 覆盖 `artifacts/utils.ts` / `artifact-file-detail.tsx` / `next.config.js`**。会丢 Gateway 改造 + 上游 scroll-restoration + 我的三路合并。
- ❌ **为了「全量跟随上游」把无 Noldus 定制的纯上游文件也退回 dev**（`code-block.tsx`/`copy-button.tsx`/`clipboard.ts`/`streamdown/*`/`agents/api.ts` 等 dev 无 Noldus 标记的，保留上游，别动）。
- ❌ **补上游新文件 `static-mode.ts`/`threads/api.ts`/`threads/token-usage.ts`/`static-demo.ts`**。走 dev 方向后它们无人引用，补了是死代码（§2.4）。
- ❌ **只看「编译过」就算完**。必须 grep 验证 §5 的 Noldus 定制确实在（编译过 ≠ 定制没丢）。

---

## 7. 善后（修完后建议，可写进 handoff，不强制本 spec 内做）

这次暴露的**根因是 sync 缺前端版 PROTECTED_FILES**。建议：
- 在 `scripts/sync-deerflow.sh` 增加前端受保护清单（至少：`frontend/src/core/threads/hooks.ts`、`frontend/src/core/messages/utils.ts`、`frontend/src/core/api/api-client.ts`、`frontend/src/core/i18n/locales/*`、`frontend/src/components/workspace/input-box.tsx`、`feedback-buttons.tsx`、`quality-warning-banner.tsx`、`stage-broadcast.ts`、`settings/local.ts`、`threads/utils.ts`、`threads/types.ts`），sync 时对它们做 surgical 而非整文件覆盖。
- 在 sync SOP 增加一条：**sync 后必须跑前端 `pnpm typecheck`**（本次缺这一步，所以"全绿 3254"只是后端，前端编译不过没被发现）。
- 关联 memory：`feedback_sync_protected_files_registry_loss`（后端版教训）应扩展到前端。

---

## 8. 速查（命令）

```bash
# 进 worktree
cd /home/wangqiuyang/noldus-insight/.claude/worktrees/sync-0602

# 软链 node_modules 仅供 typecheck（完事删）
ln -sfn /home/wangqiuyang/noldus-insight/packages/agent/frontend/node_modules packages/agent/frontend/node_modules

# typecheck
( cd packages/agent/frontend && node_modules/.bin/tsc --noEmit --pretty false 2>&1 | grep "error TS" )

# 恢复某文件到 dev 版
git checkout origin/dev -- packages/agent/frontend/<相对 frontend 的路径，但要带 packages/agent/frontend 前缀>
# 例：git checkout origin/dev -- packages/agent/frontend/src/core/threads/hooks.ts

# 某文件 dev/上游 标记检测
git show origin/dev:packages/agent/frontend/<path> | grep -cE "flywheel|quality|clarification|stageBroadcast|feedback|reasoning_effort|workflow_mode|[一-龥]"

# 删软链
rm -f packages/agent/frontend/node_modules
```

上游 git 路径映射：我们 `packages/agent/frontend/X` ↔ `git show deerflow/main:frontend/X` ↔ `git show origin/dev:packages/agent/frontend/X`。
