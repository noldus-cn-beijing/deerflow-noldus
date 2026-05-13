# 2026-05-12 真数据指标 review

> **2026-05-13 校正：** 同事已回 [`../2026-05-12-feedback.md`](../2026-05-12-feedback.md)。本目录下的 review 是 5-12 初稿，部分推断已被同事推翻（Q1/Q2/Q4/Q5），保留作历史档案。当前结论以 feedback.md 为准；后续行动追踪见 [`2026-05-13 同事反馈校正交接`](../../handoffs/2026-05/2026-05-13-feedback-corrections-handoff.md)（待写）。
>
> `review.html` 顶部已加红色校正横幅，标明哪些段落作废。

## 给行为学同事

请打开 [`review.html`](./review.html) — 单文件 HTML，浏览器双击即可看。

**3 个范式（EPM / OFT / FST）我们已经用你提供的真数据跑通了所有指标**。你需要做的事情：

1. ~~**判断数字是否合理**：每个指标我都标了"典型范围（参考）"和初步判断（✅ 合理 / ⚠️ 偏低 / 等）~~ 已作废 — 同事确认不以常模/基线判读，所有指标都要等对照组数据出来再做组间比较。
2. **回答 6 个待确认问题**（Q1-Q6 黄色高亮区）— ✅ 已于 5-13 回流，见 [`../2026-05-12-feedback.md`](../2026-05-12-feedback.md)
3. **勾选**你希望最终报告里呈现哪些指标（Q6）— ✅ 已给白名单，见 feedback

## 给软件同事

- 数据源：`/home/wangqiuyang/DemoData/newdemodata/`
- 跑指标的临时脚本：`/tmp/run_metrics_on_real_data.py`
- 这一轮的修复 commit：`db630ec7..d8311bd3`（dev 分支，3 个 fix commit）
- 同步 handoff 文档：[`docs/handoffs/2026-05/2026-05-12-real-data-metrics-verified-handoff.md`](../../handoffs/2026-05/2026-05-12-real-data-metrics-verified-handoff.md)

## 同事 review 回流之后（已更新 2026-05-13）

实际同事 feedback 给出的方向：

- **Q1（EPM 单只判读）**：单只无对照组不做结论，等对照组数据齐再比 — 不需要额外开发。
- **Q2（OFT 单 zone fallback）**：列名歧义时改为反问用户，移除 `oft.py` 里裸 `in_zone` silent fallback；后续 agent 层接 `AmbiguousZoneError` 触发反问。
- **Q3（分组 ground truth）**：raw 最后一列是「数据选择配置中对结果块的命名」（不是 user-defined variable）；`parse.py` 自动 group 推断按这个语义重写。
- **Q4（FST immobility 数值过小）**：作废 — 同事原话"还是陷入了和常模对比的幻觉，不要参考常模或基线"。「新增基于 `mobility_continuous` 算法路径」的提议同步作废。
- **Q5（Mobility_1 / Mobility_10）**：真实含义是 **采样率**（每帧 vs 每 10 帧平均），不是阈值。一般用户只会给一种采样率；同时给两份属非典型。
- **Q6（指标白名单）**：见 [`feedback.md`](../2026-05-12-feedback.md)；待补 EPM `open_arm_entry_ratio`、OFT `center_distance` + `center_distance_ratio`、FST 命名收口为 `immobility_time / _latency / _count`。

