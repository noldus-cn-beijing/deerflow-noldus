# Spec：e2e dogfood skill 增强 —— HITL 答案预填 + 性能/视觉回归

> 状态：**实施 spec，可直接交付 agent 执行**
> 日期：2026-06-26
> 代码基线：dev HEAD（写时 `df92db3d`）
> 性质：🟡 中 · 改 `noldus-insight-e2e` skill（driver 脚本 + 取证面板），不碰产品代码
> 目标：补现有 e2e 的两个结构性空白——① **HITL 答案随数据而异、要能预先给**（现在 driver 写死通用"确认"，跑不了需专业答案的数据）② **前端"太卡/视觉问题"测不出**（现在只录一张事后截图，零性能维度）。

---

## 〇、现状与两个缺陷（坐实）

`noldus-insight-e2e` skill 现在：driver `scripts/run-e2e.cjs` 用 Playwright(chromium) 上传 xlsx → 正则识别 HITL → **填写死的通用答案**（`run-e2e.cjs:118` 检测 clarif，`:128` `ans='确认'`，`:129` 重复则升级 `继续推进...`）→ 录 SSE body（CDP Network）+ **一张 final.png**（`:154`）→ `analyze.py` 跑后端取证面板（data-analyst seal / chart / report 存在性）。

**缺陷 A — HITL 答案写死、范式无关**：SKILL.md 明确"driver 故意 paradigm-agnostic，真要具体答案另起一步手动输入"。后果：**只能跑"通用确认就能过"的数据**；需专业答案的数据（OFT "中心区是哪列"、EPM "4 区怎么聚合"、FewZones 列对齐）跑不通，且无法无人值守。违背 memory 铁律 `feedback_oft_single_zone_must_ask_not_guess`（不知道哪列就要问、不能猜）——driver 替用户瞎"确认"恰是猜。

**缺陷 B — 前端性能/视觉测不出**：driver 只录事后截图，**无性能 trace、无长任务、无帧率、无视觉回归**。前几轮 dogfood 的"切回卡顿"是靠**手动**临时 CPU profile 脚本（`/tmp/perf-profile.cjs`）抓的，**没沉淀进 e2e → 下次回归没人守**（"代码有修复≠现象消除"的反面）。

---

## 一、任务 A：HITL 答案预填（用户提前说明每步答案）

### A.1 答案文件（用户提前写、跟数据走）

在 data 目录旁放一个 `e2e-answers.yaml`（**与数据同位、可版本化、一份数据一份答案**）。driver 默认读 `$E2E_DATA_DIR/e2e-answers.yaml`，调用时可用 env 覆盖路径。格式：

```yaml
# e2e-answers.yaml —— 用户提前想好、按问题关键词匹配
answers:
  - match: ["模板", "PlusMaze", "范式"]      # 问题含任一关键词 → 用此答案
    answer: "这是 EPM 高架十字迷宫"
  - match: ["中心区", "边缘区", "哪.{0,4}列", "分析区"]
    answer: "中心区对应 zone_center 列，边缘区对应 zone_border 列"
  - match: ["分组", "Treatment", "对照"]
    answer: "按 Treatment 列分组，control vs drug"
on_unmatched: fail        # fail（默认，遇未预期问题停下报错）| generic（兜底"确认"+标记）
```

### A.2 匹配方式：关键词/类型匹配（用户拍板）

- driver 检测到 HITL 后，**按 `answers[].match` 关键词正则去匹配问题文本**（问题是 agent 动态生成、措辞不固定、顺序随数据变，所以**不按出现顺序**——顺序脆，漏一个全错位）。第一个命中的条目用其 `answer`。
- 匹配在 `run-e2e.cjs:118` 现有 clarif 检测之后插入：命中预填 → 填 `answer`；否则走 A.3。
- 匹配是 driver 侧纯文本逻辑（确定性），不引 LLM。

### A.3 未匹配处理：fail-loud 停下报错（用户拍板）

- 预填没覆盖到的实际问题 → **默认 fail-loud**：driver 停下、写一条 `unmatched_clarification`（含问题全文 + 已有 answers）到 `$E2E_OUT/`、退出非零。**防"静默乱答污染结果"**（守 memory `feedback_oft_single_zone_must_ask_not_guess` + "代码有修复≠现象消除"）。
- 仅当 `on_unmatched: generic` 时才 fallback 到通用"确认"并在 `clarifications.json` 标 `kind=generic-fallback`（保留旧行为给"通用确认即可过"的数据）。
- **保留**现有"重复同问题升级 `继续推进`"防死循环逻辑（`run-e2e.cjs:129`），但仅在 generic 模式下生效；预填模式下重复同问题应直接 fail（说明预填答案没真正解决该问，避免假装跑通）。

### A.4 SKILL.md 同步

改 SKILL.md 的"Hard rules"段：把"driver 故意 paradigm-agnostic、不接受范式答案"**改为**"默认读 `e2e-answers.yaml` 按关键词应答；无答案文件时退化为通用确认（旧行为）；未匹配默认 fail-loud"。argument-hint 补 answers 文件说明。

---

## 二、任务 B：性能/视觉回归（治"太卡测不出"）

### B.0 头号约束：必须跑 prod build（否则数据全是噪声）

**dev build 性能数据不可信**（Turbopack 未优化 + HMR 注入 + source map）——前一轮 handoff 自己记了"dev build 失真、验收以 prod build 为准"。所以**性能采集必须对 prod build**：
- 新增一个 `E2E_PERF_BASE_URL` 指向 prod 前端（`pnpm build && pnpm start` 起的，或 prod compose 的 :2026）。
- driver 性能段若检测到目标是 dev（端口/header 判别），**warn 并拒绝出性能断言**（只录原始 trace 不给结论），防假绿/假红。

### B.1 采集维度（按痛感优先级，P0 必做，其余可选）

driver 已持有 CDP session（`run-e2e.cjs:43`），顺势扩：

| 维度 | 优先级 | 采集方式 | 抓什么 |
|---|---|---|---|
| **长任务（main-thread block）** | **P0**（切回卡顿直接信号、当初手动 profile 抓的就是它） | 注入 `PerformanceObserver('longtask')` 或 CDP `Performance`，量 >50ms 任务 | 切回 thread / 开图廊 / 打开页时主线程被阻塞 |
| **关键交互耗时** | **P0** | `page.evaluate` 量"点击→渲染稳定"的 ms（包住切回 thread、开图廊这几个已知痛点动作） | 交互到可见的延迟 |
| 帧率/掉帧 | P1 | CDP `Tracing` 录 trace 算 FPS | 滚动/动画掉帧 |
| Web Vitals(INP/CLS/LCP) | P1 | 注入 web-vitals 采集 | 输入延迟/布局抖动/首屏 |
| 视觉回归 | P2 | 关键画面截图对基线（扩展现有 final.png） | 图廊空、布局错位 |

P0 两项是本 spec 必做（直接对应你说的"太卡"）；P1/P2 列为可选，实施时按改动面增删，**实现哪几项就在取证面板诚实标注，没测的不算过**（守 `feedback_fallback_trigger_rate_must_be_observable`）。

### B.2 固定动作脚本（让"卡"可复现可断言）

新增 driver 阶段：流水线跑完到达 terminal 后，**确定性地执行一组已知易卡的交互**并各自采性能：
1. 切回该 thread（reload 或导航离开再回）——量长任务总时长 + 交互耗时。
2. 打开对话流图廊 / artifact 视图——量长任务。
3. （若有）滚动消息流——量掉帧。

每个动作的指标落 `$E2E_OUT/perf.json`。

### B.3 阈值断言（进取证面板）

`analyze.py` 取证面板加性能段：对 `perf.json` 的 P0 指标做**阈值断言**（如"切回交互的最长 longtask > 200ms = 退化告警"、"切回总阻塞 > 阈值 = 红"）。阈值**首次跑后按 prod baseline 标定写进 skill**（不是拍脑袋），后续做回归守护。**dev 数据进来则面板标 `PERF: skipped (dev build)` 不出红绿**。

---

## 三、关键文件

- `.claude/skills/noldus-insight-e2e/scripts/run-e2e.cjs`（A：HITL 匹配应答；B：性能采集 + 固定动作脚本）
- `.claude/skills/noldus-insight-e2e/scripts/lib.js`（共享：answers 文件解析、CDP 性能 helper）
- `.claude/skills/noldus-insight-e2e/scripts/analyze.py`（B：性能阈值断言进取证面板）
- `.claude/skills/noldus-insight-e2e/SKILL.md`（A/B：改 Hard rules、env、argument-hint；加 `e2e-answers.yaml` 说明 + prod-build 要求）
- `.claude/skills/noldus-insight-e2e/references/forensic-panel.md`（B：性能面板说明）
- 新增样例 `references/e2e-answers.example.yaml`

---

## 四、不做 / 边界

- **不引入 browser-use 或任何 LLM 驱动浏览器**——测试要确定性，被测系统已是 LLM 非确定性，再叠 LLM 测试器会让故障归因崩坏（且它不测性能、更慢）。维持 Playwright/CDP 确定性驱动。
- **driver 不替用户编范式答案**——答案全部来自用户预填的 `e2e-answers.yaml`；未匹配默认 fail-loud，不猜（守 `feedback_oft_single_zone_must_ask_not_guess`）。
- **性能不测 dev build**——只对 prod build 出断言。
- **不动产品前端/后端代码**——纯增强测试 skill。

---

## 五、验证

1. **A 回归**：用一份需专业答案的真实数据（如 OFT 自定义列）+ 配套 `e2e-answers.yaml`，无人值守跑通到 terminal；故意删一条答案 → 断言 driver fail-loud 报 `unmatched_clarification` 而非瞎确认。
2. **A 兼容**：无 answers 文件时退化为旧的通用"确认"行为（不回归现有能跑的数据）。
3. **B prod 门**：对 dev build 跑 → 面板标 `PERF: skipped (dev build)`；对 prod build 跑 → 出 P0 性能指标 + 阈值判定。
4. **B 能抓卡**：在 prod build 上，对一个**已知卡的版本**（如 revert 掉 #223 渲染优化）跑 → 性能面板应红/告警（坐实"能测出卡"，守"代码有修复≠现象消除"——亲手验证它真能抓回归）。
5. skill 改动后自测 driver 脚本可独立跑（`node scripts/run-e2e.cjs` 在 server 起着时）。

---

*依据：实读 `run-e2e.cjs:118-136`（HITL 写死通用确认）、`:43`（已持 CDP session 可扩性能）、`:154`（仅事后截图）、SKILL.md（故意 paradigm-agnostic 的硬规则）。两缺陷：HITL 答案随数据而异需预填、前端性能/视觉测不出（切回卡顿靠手动 profile 未沉淀）。用户拍板：关键词匹配 / 未匹配 fail-loud / answers 文件随数据走。性能必须 prod build（dev 失真）。不引 browser-use（LLM 非确定性、不测性能、归因崩坏）。*
