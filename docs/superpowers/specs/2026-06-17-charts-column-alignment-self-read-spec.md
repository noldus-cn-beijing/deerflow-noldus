# Spec 3 (P3): charts 路径列对齐靠 LLM 拼 CLI —— 改为从 experiment-context 自读

> 日期：2026-06-17
> 类型：结构病根治（红线二：共享源 + 自取）
> 触发源：2026-06-17 EPM dogfood —— 真实 FewZones 数据（列名 open/closed），chart-maker 跑 resolve 没传 `--column-aliases-file` → resolve 判"缺 in_zone_* 列" → box/bar/rose/zone_distribution 全被 skip，只出了 trajectory/heatmap 纯坐标图。
> 状态：待 review → 批准后 worktree 实施。
> **遵循工程实践**：[红线二 —— session 级横切状态用"共享源 + 自取"，禁止"传参 + 各路径单独接线"](../../refs/2026-06-17-langgraph-deerflow-agent-engineering-best-practices.md#红线二session-级横切状态用共享源--自取禁止传参--各路径单独接线)。修法照"正模式 1：横切状态有唯一 reader 收口，工具内部自取，绝不让 LLM 拼 CLI 参数传 session 常量"。
> 关联：P4（shared-state-sourcing）是本 spec 的元收口——P3 修 charts 这一条路径，P4 加"所有路径都接了"的 parity 回归。

---

## 0. 症状

dogfood 输出实证："因原始数据仅含轨迹坐标列…缺 in_zone_* 区带驻留列，所以 catalog 跳过了箱线图、条形图、玫瑰图、区域分布图（共 4 种）。" —— 但用户**已经在 Gate 1 对齐了 open→open_arms、closed→closed_arms**，对齐信息就在 experiment-context.json 里。box/bar/rose 本该能画。它们被跳，纯粹因为 chart-maker 跑 resolve 时**没把 column_aliases 传进去**。

## 1. 根因：charts 路径退化成"LLM 拼 CLI 传 session 状态"，metrics 路径却是"工具自读 context"

| 路径 | 怎么拿 column_aliases | 靠谁 | 结果 |
|---|---|---|---|
| **metrics** | `prep_metric_plan_tool.py:213-218` 工具内部 `read_context()` 自读 | 代码（确定性） | ✅ 对齐总是生效 |
| **charts** | chart-maker 在 bash 里拼 `catalog.resolve --column-aliases-file ...`（SKILL.md:22 纯 bash 模板，`--column-aliases-file` 是 cli.py:101 的**可选参数 default=None**） | **LLM 记得拼**（祈祷） | ❌ LLM 没拼 → resolve 收到 None → 缺列 → skip |

同一个 session 状态（column_aliases，SSOT 在 experiment-context.json），两条路径两种拿法。错的那条把"机制"换成了"LLM 记性"。这正是红线二的病根：**横切状态靠传参 + 各路径单独接线，每条路径都是一个"忘了接"的点。**

代码层早就支持（cli.py:247 把 column_aliases 透传给 `resolve_charts`，resolve_charts 完整消费）——**纯粹是没人把 session 里的 aliases 喂进这个入口**。

## 2. 修法（照红线二正模式 1：工具自读，不靠 LLM 拼 CLI）

核心原则：**charts 路径获取 column_aliases 必须像 metrics 路径一样确定性，不依赖 LLM 在 bash 里记得拼参数。**

两个候选实现，**优先方案 A**：

### 方案 A（推荐，结构对称）：给 charts 一个 prep 工具，内部自读 context
仿照 `prep_metric_plan_tool`，新增（或扩展）一个 `prep_chart_plan` 工具：
- 工具内部 `read_context()` 拿 `column_aliases` + `groups` + `paradigm`（与 metrics 路径同一个 reader，红线二正模式 1）。
- 工具内部调 `resolve_charts(..., column_aliases=...)`，产出 plan_charts.json。
- chart-maker 改为调这个工具，而非手拼 `catalog.resolve` bash 命令。
- **column_aliases 永远来自 context，LLM 无从遗漏。**

这样 charts 和 metrics 在"如何拿 session 状态"上结构对称，都走"工具自读 context"，消灭"LLM 拼 CLI"这条退化路径。

### 方案 B（轻量但不彻底）：CLI 自己兜底读 context
若不新增工具，`catalog.resolve` 的 cli.py 在 `--column-aliases-file` 未传时，**自动从 `--workspace-dir/experiment-context.json` 读 column_aliases 兜底**（charts 和 metrics 模式都兜底）。

> 方案 B 的风险：resolve CLI 开始隐式读 context，与"resolve 是纯函数式 CLI"的设计有张力；且 workspace-dir 得保证传对。**方案 A 更符合红线二**（reader 收口在工具层，CLI 保持纯粹）。**实施者优先 A，A 工作量过大时退 B 并在 spec 注明理由。**

无论 A/B：**SKILL.md 不再教 LLM 拼 `--column-aliases-file`**（那是把状态传递交给 LLM 记性 = 红线二反模式）。SKILL 改动只是"调 prep_chart_plan 工具"或"resolve 会自动对齐"，不承载状态传递。

## 3. 测试（红→绿，照红线二可执行判据）

> **真实数据（红线三）**：端到端验收用 `/home/wangqiuyang/DemoData/real_data/Raw data-EPM-Xuhui-28`（FewZones，列名 open/closed），对齐后断言 box/bar/rose 真能画出。单测可用其列结构子集。

- `test_charts_resolve_picks_up_aliases_from_context`（**修复前必红**）：experiment-context.json 有 `column_aliases={open:open_arms, closed:closed_arms}`，**不手动传 aliases**，跑 charts 路径（方案 A：调 prep_chart_plan；方案 B：调 resolve 不带 --column-aliases-file）→ 断言 plan_charts 含 box_open_arm/open_arm_time_ratio_bar（不再被 skip）。当前代码不自读 → 红。
- `test_charts_skip_when_truly_no_alignment`：无 column_aliases 且列名确实非标准 → box/bar 仍合理 skip（守住"真缺列才跳"，不是无脑全画）。
- **parity 测试（属 P4，但这里先放一条 charts 的）**：同一份 column_aliases，metrics 路径和 charts 路径都能解析出 zone 对齐（两路径行为一致）。

## 4. 风险边界

- 不改 resolve_charts 的列对齐算法（PR#141 已做对，只是没被喂数据）；本 spec 只改"aliases 怎么到达 resolve"。
- 方案 A 新增工具要注册（tools.py + builtins __init__）+ 裸导入验证导入环（红线/CLAUDE.md 铁律）。
- groups 同理：charts 路径的 groups 也应走 context 自读（trace 里 chart-maker 也在手动处理 groups.json）。本 spec 一并收口 column_aliases + groups 两个 session 状态。
- 不碰 4 图预算（那是 P5）。
