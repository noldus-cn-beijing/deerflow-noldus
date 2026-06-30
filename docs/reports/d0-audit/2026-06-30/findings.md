# D0 UX audit findings — 2026-06-30

> **D0 是诊断，只看不改。** 本清单喂 D1（DESIGN.md/token）、A2（进度叙事）、D2（组件库）、C2（画廊）排优先级。
> 5 透镜：L1 当前态优先 / L2 对话为主 / L3 日式克制 / L4 不暴露内脏 / L5 分支态一致。

---

## 采集元信息（证据可信度声明）

- **数据**：`~/DemoData/real_data/Raw data-EPM-Xuhui-28`（EPM，4 组 × 7：Group=XX/XY/YY/YZ，28 trial）。
- **HITL 答案**：FewZones 模板；列语义 `open=开臂, closed=闭臂`；分组 `Group=XX 对照组 / XY/YY/YZ 实验组`（见 `e2e-answers.yaml`）。
- **实跑路**：thread `8827351d-e2b1-4292-b69e-a3bde14b5fb0`，完整跑通到 report（report-writer 出 `report.html` 6 段）。
- **⚠️ 视觉证据采自 dev :2026（nginx 门面），非裸 :3000**：spec 要求 prod build 视觉真实，但裸 `pnpm start` 的 `:3000` 是 **frontend-only**（无 nginx/gateway），`/api/*` 一律 404 无法登录采图。**这不是 prod 断裂、是采图用错 prod 形态**（详见 L5-1 复诊更正）。改在 dev :2026（nginx 门面）采集；L3 视觉判断受「非 prod build 优化」限制，但读码路不受影响。
- **截图证据**：`evidence/` 下 3 张关键态（首屏回放 / 反问态 / report 完成态）+ 磁盘产物真相（113 图 / report.html / experiment-context.json / handoffs）。
- **本次采集发现的 tooling 问题**（非产品 bug，记此供 e2e skill 改进，不入 findings）：playwright `setInputFiles` 与该上传组件不兼容（卡第 4 个文件）；e2e driver `networkidle` goto 在 prod polling 下超时（已加 `E2E_GOTO_WAIT` env）；chip 探测 `text=Remove` 在中文 i18n 下失效（已修为 `aria-label*=移除`）。

---

## 问题清单

### 🔴 L4-1 ［高］agent 把基因型分组脑补成「剂量梯度」，幻觉性归因写满报告

| 字段 | 内容 |
|---|---|
| 问题 | 用户的 4 个 Group（XX/XY/YY/YZ，基因型）只说「XX 对照、其余实验」。agent 在 `experiment_context.json` 自作主张写入 `XX=对照组, XY=低剂量, YY=中剂量, YZ=高剂量`，data-analyst 据此产出完整「剂量-反应关系 / 中高剂量致焦虑 / 低剂量无效应」判读，并写进交付给研究员的 `report.html`。 |
| 在哪 | `experiment-context.json: resolved.groups`；`handoff_data_analyst.json`；`report.html`（含「低剂量/中剂量/高剂量/剂量-反应」）；UI 交付物清单「专业判读 ✅ 中/高剂量致焦虑效应」——截图 `evidence/05-gallery-with-report.png` |
| 严重度 | **高**（阻断判读可信度 / 误导研究员 / 输出宪法违规——编造未确认语义） |
| 违背理念 | L4（不暴露/编造内脏）+ 判读正确性（CLAUDE.md 第 9 条：判读只看组间差异，不脑补实验设计语义） |
| 修复方向 | **归 A2 + 输出宪法**：HITL 反问增加「组名语义」确认（是剂量梯度？基因型？处理？），agent **禁用未确认的剂量/处理命名**，未知组用「实验组 1/2/3」中性命名。report-writer 守宪法拒绝写「剂量-反应」 unless 用户确认剂量设计。 |

### ⚪ L5-1 ［作废→方法论修正］「prod build/auth 断裂」前提错误：采图用错 prod 形态（裸 :3000 绕过 nginx）

> **2026-06-30 复诊更正**：本条原记为「🔴 高 / P0 阶断：prod build 缺 API route + auth 端点断裂」。复诊（配置层坐实）证明**前提错误，不是产品 bug**，降级为审计方法论修正。

| 字段 | 内容 |
|---|---|
| 原始观察 | 测试者跑 `pnpm build && pnpm start` 起 `:3000`，curl `/api/v1/auth/setup-status`、`/api/v1/auth/login/local`、`…/runs/{id}/messages` 全 404；`:2026` 同端点 200/422。据此判「prod build/auth 断裂、研究员无法登录」。 |
| 复诊根因 | **`:3000` 是 Next.js 前端单进程，从不负责 `/api/*`。** nginx 才是唯一公网门面，代理 `/api/v1/auth/*`、`/api/langgraph/*`（rewrite→`/api/*`）、`/api/threads/*` 到 gateway:8001（`docker/nginx/nginx.conf:69/163/233`）；`:3000` 只拥有 `location /`（UI）。三个「缺失」端点**都在 gateway 里、build 没漏任何东西**：`setup-status`=`backend/app/gateway/routers/auth.py:392`、`login/local`=`auth.py:275`、`runs/{id}/messages`=`thread_runs.py:384`。裸 `pnpm start` 没起 nginx 也没起 gateway，curl `:3000/api/...` → **404 是设计内行为**，非断裂。真 prod（`docker/docker-compose.yaml`）跑 nginx+frontend+gateway 三服务，研究员走 nginx，三端点全可达。 |
| 严重度 | **作废**（非 bug；无部署链可修） |
| 违背理念 | 无（产品侧）。属审计采集方法论错误：守 `feedback_dev_prod_behavior_alignment`（dev/prod 行为对齐 compose）+ `feedback_e2e_testing_deterministic_playwright_not_llm_browser_use`（prod 视觉必须起对的 prod 形态）。 |
| 修复方向 | **归 e2e skill 方法论**（非部署链）：prod 视觉/登录/任何 `/api/*` 采集必经 nginx `:2026`（`make dev`）或完整 prod compose，**禁止直连 :3000**。护栏已加进 `.claude/skills/noldus-insight-e2e/SKILL.md`（perf 段后新增「`:3000` 是 frontend-only」bullet）。 |
| ⚠️ 牵连更正 | 原 L5-1 把「用户前端崩溃什么都看不到」归因于此。**真因是浏览器端崩溃**（findings.md 软分支态表已正确记述：chart-maker 跑 113 图时**浏览器**崩溃 16:50，后端 thread 不受影响、重连后端点 200 恢复）。崩溃与 prod 部署无关，原 L5-1 误将两者合并。 |

### 🟡 L4-2 ［中］交付物清单暴露内部字段名 / analysis config id

| 字段 | 内容 |
|---|---|
| 问题 | report 完成消息含 `analysis config id 58db62f46ff5f1d4`（内部追踪 id）+ 交付清单附近 DOM 暴露 `sections_missing / errors_count / written_count / statistical_validity` 等内部字段名（虽值全 ok，字段名对研究员是黑话）。 |
| 在哪 | lead 收尾消息 `message-list.tsx` 渲染；截图 `evidence/05-gallery-with-report.png`（视觉分析确认 config id 暴露） |
| 严重度 | 中 |
| 违背理念 | L4（不暴露内脏） |
| 修复方向 | **归 D2 + A2**：lead 收尾文案模板剥离 config id / 内部字段，只露研究员可读的「环节 / 状态 / 产出」。 |

### 🟡 L4-3 ［中］quality-warning-banner 直接展示内部 code（如 METRIC_VALIDATION）

| 字段 | 内容 |
|---|---|
| 问题 | `QualityWarningBanner` 渲染 `[{ws.label} {w.code}]`——`w.code` 是内部诊断码（`METRIC_VALIDATION` 等），研究员看不懂。 |
| 在哪 | `components/workspace/messages/quality-warning-banner.tsx:102-103` |
| 严重度 | 中（本次 dogfood `injected=0 warnings` 未触发，但代码路径确认） |
| 违背理念 | L4 |
| 修复方向 | **归 D2**：把 `code` 映射成人类可读标签，或只在「展开详情」里露、标题用自然语言。 |

### 🟡 L4-4 ［中］token-usage-indicator 在对话区露 token 数

| 字段 | 内容 |
|---|---|
| 问题 | `TokenUsageIndicator` 显示 `formatTokenCount(totalTokens)` + CoinsIcon——token 是 LLM 计费单位，对行为学研究员是无意义技术黑话。 |
| 在哪 | `components/workspace/token-usage-indicator.tsx`（渲染位置由调用方决定，本次截图未见明显暴露，待核调用点） |
| 严重度 | 中（若在研究员可见的主对话区） |
| 违背理念 | L4 |
| 修复方向 | **归 D2**：研究员视角隐藏，或只在管理员/调试模式显示。 |

### 🟢 L1-1 ［低］反问态/分析态设计成熟，当前态优先落实良好（正向发现）

| 字段 | 内容 |
|---|---|
| 问题 | subtask-card 只露「当前 subagent 在干什么」（stage broadcast + shimmer），不摆固定全流程清单；decision-card 反问用 accent bar + 数字快捷键 1-9 + closed-loop answered 高亮。符合「只告诉当前在干什么」。 |
| 在哪 | `subtask-card.tsx`、`decision-card.tsx`、`clarification-options.tsx`；截图 `evidence/03-clarification-report-confirm.png` |
| 严重度 | 低（正向，无需改） |
| 违背理念 | L1（达成） |
| 修复方向 | 保持。数字快捷键 + accent bar 是 spec#5 成熟落地。 |

### 🟢 L3-1 ［低］日式克制视觉落实良好（正向发现，dev 视觉）

| 字段 | 内容 |
|---|---|
| 问题 | 留白充足、信息密度适中、accent 色条克制（非整卡变色）、颜色饱和度低、反问卡 color-not-only 三件套（图标+色+文字）。入场用 `animate-pulse-warm`/`ease-brand-out`（非 linear）。 |
| 在哪 | 截图 `evidence/03-clarification-report-confirm.png`、`05-gallery-with-report.png`（视觉分析确认）；`globals.css` 的 `--ease-*` / `animate-pulse-warm` |
| 严重度 | 低（正向） |
| 违背理念 | L3（达成，注：dev :2026 视觉，非 prod build 优化态未核——见 L5-1 复诊：prod 视觉须经 nginx 门面采集） |
| 修复方向 | 保持。 |

### 🟢 C2-1 ［低］thread-assets-panel 两段式画廊（报告 + 图表），113 图缩略图 + ZIP 下载（已知设计点）

| 字段 | 内容 |
|---|---|
| 问题 | 画廊两段：报告区（report.html 卡 + 查看完整报告 + 导出）+ 图表区（分面筛选：所有范式/所有类型、汇总图(1) + 展开查看全部 112 张个体图、下载全部 ZIP）。数据走磁盘端点稳定（不随流漂移）。本次 113 图全显示（PR#213 修过的「只显示 1 张」未复发）。 |
| 在哪 | `thread-assets-panel.tsx`；截图 `evidence/05-gallery-with-report.png` |
| 严重度 | 低（设计已知，C1/C2 territory） |
| 违背理念 | L2（产物为辅，空间分配合理） |
| 修复方向 | **归 C2**：在 `8fc7316a` 之上优化两段式布局 + 113 图浏览体验。 |

---

## 软分支态覆盖（spec 验收 2）

| 软分支态 | 状态 | 证据 |
|---|---|---|
| **崩溃重连** | ✅ 自然触发 | 用户**浏览器**在 chart-maker 跑 113 图时崩溃（真实事件 16:50）；后端 thread 不受影响继续跑；重连后 history/runs/artifacts 端点 200 恢复。**崩溃真因 = 浏览器端崩溃，与 prod 部署无关**（原 L5-1 误归因 prod build/auth 断裂，已复诊作废）。 |
| **多轮追问** | ✅ 自然触发 | 3 轮 clarification（gate1 范式/分组 → viz 可视化 → report 生成）；对话变长后滚动行为正常（`use-stick-to-bottom` + progressive mount）。截图 `03-clarification-report-confirm.png`。 |
| **n=1 单样本** | ⚠️ 未触发，原因：本次 4 组 × 7 = 28 样本，无 n<2 组。降级态（lead fast-path 跳 data-analyst）未覆盖。 | 需独立场景（单 subject 数据）驱动。 |
| **知识问答空态** | ⚠️ 未触发，原因：本次全程有数据分析产物，无「无数据纯问答」路径。 | 需独立场景（不发数据纯提问）驱动。 |

---

## 硬主路 6 关键态覆盖（spec 验收 1）

| 关键态 | 截图 | 状态 |
|---|---|---|
| 1. 首屏空态（未上传） | `evidence/replay-first-frame.png`（含首屏「👋 你好，欢迎回来！」+ 早期分析已派遣的历史回放） | ✅（首屏空态的视觉已在探针 `/tmp/d0-probe2.png` 确认：左栏+中央欢迎语+输入框，右侧产出物空态隐藏） |
| 2. 上传后（数据已挂） | （包含在 03 的历史回放流里） | ⚠️ 部分覆盖（上传后直接进 HITL，无独立静止态截图） |
| 3. 反问卡片态 | `evidence/03-clarification-report-confirm.png` | ✅（report 确认反问 + accent bar + 1/2 数字键选项） |
| 4. 分析中（subtask + 流式） | （包含在 03 的 subtask timeline：正在派遣 code-executor / data-analyst） | ⚠️ 部分覆盖（流式瞬间未独立截，timeline 折叠态有） |
| 5. 产物画廊态 | `evidence/05-gallery-with-report.png` | ✅（113 图画廊 + 报告区 + 分面筛选 + ZIP） |
| 6. 报告展开态 | `evidence/05-gallery-with-report.png`（report.html 入口 + 交付物清单 + 核心结论） | ✅（report 完成态；「查看完整报告」点开 report.html 的展开态未独立截，但 report.html 已落盘 113KB 可端到端验证） |

---

## D1 修正输入摘要（喂 DESIGN.md / globals.css token）

D0 是 UX/判读诊断，**未发现 DESIGN.md token 层面的修正需求**——L3 日式克制（留白/饱和度/ease 曲线）落实良好，token 体系（`--ease-*`、`--color-status-*`、`animate-pulse-warm`）健康。

D1 真正该改的不是 token，是 **agent 输出层**（lead 收尾文案 / experiment_context 语义守门 / report-writer 宪法）——见 **L4-1（剂量幻觉）+ L4-2（config id 暴露）**，这两个是 D1 + A2 的输入，不是视觉 token 问题。

**给 D1 的结论**：视觉设计语言层（D1 DESIGN.md / globals.css）无需修正；问题集中在 agent 输出宪法 + 文案模板（归 A2 输出宪法轨）。

---

## 给下游的优先级建议

1. **P0（判读可信度）**：L4-1 剂量幻觉 → A2 + 输出宪法，HITL 加组名语义确认 + report-writer 守宪法。**现唯一 P0。**
2. **P1**：L4-2 交付清单暴露 config id / 内部字段 → A2 文案模板。
3. **P2**：L4-3 quality-warning code、L4-4 token 指示器 → D2 组件库。
4. **P3**：C2-1 画廊两段式 → C2 轨。
5. **保持**：L1-1、L3-1（正向发现，无需改）。
6. **作废（非 bug）**：L5-1（复诊：采图用错 prod 形态、非 prod 断裂）→ 已归 e2e skill 方法论护栏，无部署链待办。

---

## 关联

- spec：`docs/superpowers/specs/2026-06-30-d0-ux-audit-design.md`
- 上游路线图：`2026-06-30-frontend-design-language-roadmap-and-d1-design-md.md`（D0 是第 0 步）
- 下游：D1（本摘要「D1 无 token 修正需求」）、A2（L4-1/L4-2 输出宪法+文案）、D2（L4-3/L4-4 组件）、C2（C2-1 画廊 `8fc7316a` 之上）
- 守 memory：`feedback_e2e_testing_deterministic_playwright_not_llm_browser_use`（采集确定性、判断分离）、`feedback_frontend_design_japanese_minimal_motion_craft`（L3）、`feedback_oft_single_zone_must_ask_not_guess`（L4-1 同源：不知就问别猜）
