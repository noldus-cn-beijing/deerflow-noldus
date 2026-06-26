# Handoff：EthoInsight 前端 Phase 0 收尾 + 多 bug 修复 spec 待派（2026-06-26）

> ## 🔄 2026-06-26 E2E 后状态更新（最新，先读这个）
>
> 一次完整 Playwright dogfood E2E（结果见 `2026-06-26-playwright-e2e-dogfood-results-handoff.md`）+ 后续 PR 合并，**本 handoff 原列的待派 spec 大半已实施**。当前 dev HEAD = **`6ec47f78`**（不是下文的 `07838ae6`）。最新真相：
>
> **✅ 已实施/已修，作废下文对应 spec（勿重做）**：
> - **三现象修复**（画廊丢图/报告/返回卡顿）→ **#216 已合**（`/artifacts/charts` 磁盘端点 + 对话流报告卡 + gallery 返回 `router.back` 不重挂载）。**采用了"画廊按路径从磁盘取图"方案，后端 113/113 全量返回——成功。** 之前"subagent Command 不上行"的纠结已绕开。
> - **#5 决策卡** → **#217 已合**（DecisionCard）。**Phase 0 八份 spec 全部完成。**
> - **memory UUID** → **#218 已合**（按 spec A 原方案：prompt.py:845 `str()` + storage 入口防御，E2E 验证未复现）。
>
> **⏸️ 暂缓/降级（E2E 未复现或非 bug）**：
> - **历史乱序**（spec B 🔴）→ E2E 实测**后端+前端顺序双正确，未复现**。暂缓，除非更多 multi-run 合并场景再现。
> - **画图/报告"互斥二选一"**（E2E 误报为缺陷）→ **非 bug**：`prompt.py:431` 流程本就是"画完图后再 ask(report?)"按需多轮决策；E2E driver 用 generic `确认` 机械应答测不出真实决策。**不做。**
> - **画廊 per_subject 不渲染**（E2E 误报）→ **非 bug**：`useState(false)` 默认折叠、点击展开（设计正确）。只需加明显提示（见新 spec）。
>
> **🆕 仍待派（E2E 后确认有效）**：
> - **后端质量五小项**（已扩充）：`2026-06-26-backend-quality-three-minor-fixes-spec.md` —— 原 3 项 + E2E 新增 Pydantic context 噪声 + 前端进度条与后端脱节。
> - **画廊折叠提示增强**（新）：`2026-06-26-gallery-per-subject-collapse-affordance-spec.md` —— 保持折叠 + 加明显展开提示（用户拍板）。
> - **Chrome 切回卡顿** → 仍待真机火焰图（headless 测不到后台节流），未出 spec。
> - **输入框遮挡** → #217 决策卡已合，但"决策卡 live 待答 + 输入框同屏遮挡"态 E2E 未抓到；spec 仍在 `2026-06-26-input-box-overlaps-bottom-content-fix-spec.md`，**需 live 态再验是否仍遮挡**（可能 #217 已缓解）。
>
> **教训补记**：本会话"画廊丢图"我判过两次根因（先 #213 subagent Command，后纠正为磁盘端点），**#216 用磁盘端点方案修对了**。再次印证 memory `feedback_subagent_command_artifacts_not_bubble_to_lead_executor_drops_state` + 「dogfood 实测 > 代码推断」。
>
> ---
> **以下为 2026-06-26 早些时候的原始 handoff（基线 `07838ae6`），保留作历史；状态以上面的更新块为准。**

---

> 给下一个 agent：本会话产出了一批 spec（前端 Phase 0 收尾 + dogfood 发现的多个 bug 修复），**全部已写成 spec 文档，但大部分尚未实施/派发**。本文档给你：①每份 spec 的待派状态 ②并行/串行依赖图 ③关键背景与陷阱（尤其一个已合但方向错的修复）。
> dev HEAD：`07838ae6`（核实基线用 `git show HEAD:`，**别用工作树 grep**——会把别人未提交草稿误判成已合）。


---

## 〇、一句话现状

前端 Phase 0（8 spec）**只剩 #5 决策卡**未做；dogfood 一次真实 EPM 端到端跑出**一串 bug**（产物丢失/报告不显示/返回卡顿/输入框遮挡/历史乱序/memory 崩溃/后端质量），已逐一取证定根因、写成 spec。**一个严重纠正**：之前"产物丢失"的修复（#213 已合 dev）**方向是错的**，需用新 spec 重做。

---

## 一、待派 spec 清单（已核实 dev 落地状态）

| spec | 文件 | 范围/风险 | dev 状态 |
|---|---|---|---|
| **#5 决策卡** | `2026-06-24-frontend-phase0-5-decision-card-spec.md` | 前端 / 🟡 | ❌ 未做 |
| **三现象修复**（画廊丢图+报告+返回卡顿） | `2026-06-26-artifact-bubbling-report-display-gallery-return-fix-spec.md` | 后端端点+前端 / 🔴 | ❌ 未做（端点未落、gallery 仍 router.push）|
| **memory UUID 崩溃** | `2026-06-26-memory-context-uuid-load-crash-fix-spec.md` | 后端 / 🟡 一行 | ❌ 未做（prompt.py:838 仍裸 `.id`）|
| **输入框遮挡底部** | `2026-06-26-input-box-overlaps-bottom-content-fix-spec.md` | 前端布局 / 🟢 | ❌ 未做（无 pb）|
| **重进 thread 历史乱序** | `2026-06-26-rejoin-thread-history-merge-disorder-fix-spec.md` | 前端**流式红线** / 🔴 | ❌ 未做（**诊断 spec，需坐实+grill**）|
| **后端质量三小项** | `2026-06-26-backend-quality-three-minor-fixes-spec.md` | 后端 / 🟢 | ❌ 未做 |
| **Chrome 切回卡顿** | （未写 spec，用户选"先实测坐实再出 spec"）| 前端性能 / 🔴 | ❌ 未诊断完 |

**已合 dev（参考存档，勿重做）**：Phase 0 #1/#2/#3/#4/#6/#7/#8、附件堆叠补全(#215)、read-file 越界(#206)、orphan-grouping(#204)、thinking-language(#209)、#213（产物丢失旧修复——**但方向错，见 §三**）。

---

## 二、并行/串行依赖图（核心）

### 可立即并行（互不依赖、互不碰同一文件）

| 轨 | spec | 为什么独立 |
|---|---|---|
| **后端轨1** | memory UUID 崩溃 | 改 `prompt.py` 一行，纯后端 |
| **后端轨2** | 后端质量三小项 | 改 charts.py / updater.py / middleware，三项还各自独立 |
| **前端轨1** | 输入框遮挡 | 改 page.tsx/message-list className padding，纯布局 |
| **前端轨2** | #5 决策卡本体 | 独立组件（accent bar/键盘1-9/依据），**只"进度轨 waiting 联动"依赖 #4——#4 已合，故现在全可做** |

→ 这 4 条**可同时派 4 个 agent**，零冲突。

### 有依赖/需串行

- **三现象修复**：内部三个现象**可并行**（现象1 后端端点+前端画廊数据源 / 现象2 报告呈现 / 现象3 gallery 返回改 router.back）。但与 **#5 决策卡都可能碰 chat-box/page.tsx 布局**——若同时做，先约定布局改动分层。建议三现象修复**单独一个 agent 一次做完三个**（同源、同文件区）。
- **重进 thread 历史乱序（🔴红线）**：**必须串行、单独、先坐实**。它碰 `mergeMessages`/`dedupeMessagesByIdentity`（踩坑沉淀重写必复发）。**不可与任何前端 spec 并行**（都可能间接影响消息渲染）。流程：先拿原始 history API 数据复现 → 隔离失败测试 → 才动手 → 全量回归。**派它的 agent 必须 grill，别随手改。**
- **Chrome 切回卡顿**：先实测坐实（Performance 面板录"切后台→跑→切回"）再决定要不要出 spec、改哪。**与历史乱序可能同源**（都在消息流/流式层），建议**这两个由同一个 agent 串行处理**，避免两人同时碰流式层。

### 文件冲突协调点（多 spec 改同文件）

- `app/.../page.tsx`：输入框遮挡（加 padding）+ 三现象修复（gallery 返回）+ #5（决策卡布局）—— **先派遮挡（最小、先落），再 #5/三现象在其上加**。
- `message-list.tsx`：遮挡（pb）+ 历史乱序（红线）+ Chrome 卡顿（红线）—— **红线两项串行最后做**，遮挡先做。

### 推荐波次

1. **第 1 波（4 路并行）**：memory UUID / 后端质量三小项 / 输入框遮挡 / #5 决策卡 —— 全独立低风险，先清。
2. **第 2 波**：三现象修复（一个 agent 做完三个）—— 等遮挡先落（page.tsx 布局先定）。
3. **第 3 波（红线，单 agent 串行）**：历史乱序坐实+修 → Chrome 卡顿坐实+修。**最谨慎、最后做、必 grill。**

---

## 三、关键背景与陷阱（必读）

### ⚠️ 1. #213 产物丢失修复方向是错的（已合 dev 但无效）

- `2026-06-25-run-chart-plan-auto-register-artifacts-spec.md`（=#213）让 `run_chart_plan` 在 chart-maker subagent 内返回 `Command(update={artifacts})`。**dogfood 实测无效**（thread `e9837b33`，修复后跑，画廊仍只 1 张）。
- **根因（铁律级，已记 memory `feedback_subagent_command_artifacts_not_bubble_to_lead_executor_drops_state`）**：subagent 工具返回的 `Command(artifacts=...)` 写在 subagent 独立 graph 的 state，**DeerFlow executor 只捞文本结果、丢弃子 state，不上行到 lead**。subagent 的 artifacts 到不了前端。`present_files` 之所以能上行，是因为它是 **lead** 工具（lead 调它写 lead 主图），不是 subagent 工具。
- **正确修复 = 三现象修复 spec §一**：用户拍板方案——**画廊按路径从磁盘/plan 直取图**（新只读端点 `/artifacts/charts` join `plan_charts.json` 元数据），绕开 state 冒泡（磁盘是唯一真相）。后端已有现成"按 thread 列 outputs 目录"逻辑（`archive_artifacts`）可复用。
- **#213 保留无害**（不依赖它），但**别再往"让 subagent Command 上行"方向加码**。

### 2. Command 是 LangGraph 原生，不是 DeerFlow（认知）

`Command(update=...)` 是 LangGraph 能力，更新**当前 graph** state。LangGraph 也有 `Command(graph=PARENT)` 跨层更新父 state，但 **DeerFlow executor 没用它、把 subagent 当文本黑盒**。所以"subagent 改 state 给 lead 看"不能靠 Command——走文件（handoff/磁盘）。

### 3. 核实铁律

判"某代码在不在 dev"用 `git show HEAD:<file>`，**不要 `grep <工作树文件>`**——工作树含别人未提交草稿会误判（本会话初期 spec#3 后端误判即此）。本会话每个"已合/未做"判断都用 `git show HEAD:` 复核过。

### 4. dogfood 取证账号

测真实 thread 用 owner 账号登录本地 dev（端口 **2027**，不是 2026）：`qiuyang.wang@noldus.com.cn` / `19961031`（注意 `.com.cn`，不是 `.com`——那是另一个账号，会因用户隔离查不到 thread）。checkpoints 解码看 state：`packages/agent/backend/.deer-flow/checkpoints.db`，用 `langgraph.checkpoint.serde.jsonplus.JsonPlusSerializer`。

### 5. 后端改动铁律

改 `subagents/`/`tools/builtins/`/`agents/` 后**必裸导入两生产入口**（`PYTHONPATH=. python -c "import app.gateway"` + `from deerflow.agents import make_lead_agent`）——`pytest` 全绿是假绿（conftest mock 了 executor）。

---

## 四、下一步建议（接手第一步）

1. **先派第 1 波 4 路**（memory UUID / 后端质量 / 遮挡 / #5）——全独立低风险，立刻推进、清掉积压。
2. **再派三现象修复**（一个 agent，等遮挡的 page.tsx 布局先落）——这是研究员最痛的（画廊看不到图、报告看不到、返回卡）。
3. **红线两项（历史乱序 + Chrome 卡顿）单独留给一个会 grill 的 agent 串行最后做**，先坐实再改，别碰崩流式核心。
4. Phase 0 做完 #5 后，**Phase 1（replay/组件注册表/思考链降噪/todo）尚未出 spec**——建议先真机体验一轮 Phase 0 全貌，再决定 Phase 1 开不开、先开哪项（母方案 §8 有路线图描述）。

---

## 五、相关记忆/文档

- memory `feedback_subagent_command_artifacts_not_bubble_to_lead_executor_drops_state`（#213 错因根因）
- memory `feedback_code_has_fix_not_equal_bug_eliminated_seal_react_floor`（代码有修复≠现象消除，必 dogfood）
- memory `feedback_shadcn_ui_copyin_editable_vs_registry_pulled_generated`（ui/ 是 shadcn copy-in 可改，ai-elements 是 registry 禁改）
- 母方案 `docs/plans/2026-06-24-frontend-generative-ux-upgrade.md`（§8 分期路线图，Phase 1/2 定义）
- 前端 Phase 0 dev 代码核对 spec `2026-06-25-frontend-phase0-dev-code-fix-spec.md`（A/B 节核对已做完，全绿，仅参考）
