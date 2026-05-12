# 2026-05-12 真数据指标 review

## 给行为学同事

请打开 [`review.html`](./review.html) — 单文件 HTML，浏览器双击即可看。

**3 个范式（EPM / OFT / FST）我们已经用你提供的真数据跑通了所有指标**。你需要做的事情：

1. **判断数字是否合理**：每个指标我都标了"典型范围（参考）"和初步判断（✅ 合理 / ⚠️ 偏低 / 等）
2. **回答 6 个待确认问题**（Q1-Q6 黄色高亮区）
   - 最重要的是 **Q4**（FST immobility 比文献小 10-100 倍，疑似 EthoVision Mobility threshold 设定问题）和 **Q5**（Mobility_1 vs Mobility_10 目录命名的真实含义）
3. **勾选**你希望最终报告里呈现哪些指标（Q6）

## 给软件同事

- 数据源：`/home/wangqiuyang/DemoData/newdemodata/`
- 跑指标的临时脚本：`/tmp/run_metrics_on_real_data.py`
- 这一轮的修复 commit：`db630ec7..d8311bd3`（dev 分支，3 个 fix commit）
- 同步 handoff 文档：[`docs/handoffs/2026-05/2026-05-12-real-data-metrics-verified-handoff.md`](../../handoffs/2026-05/2026-05-12-real-data-metrics-verified-handoff.md)

## 同事 review 回流之后

根据 Q1-Q6 答复决定：

- 若 Q4 = (A)：让同事重新导一份合理 Mobility threshold 的 FST 数据
- 若 Q4 = (C)：增加基于 `mobility_continuous` 连续值的 immobility 算法
- 若 Q6 列了我们没实现的指标：开新 task 补全
- 若 Q2 答案是"应规范命名"：在 `golden-cases/SCHEMA.md` 加 zone 命名约定
- 若 Q3 / Q5 确认了分组语义：在 `parse.py` 加自动 group 推断（从 user-defined variable 列）
