# Spec：data-analyst 判读残留耗时 —— 06-18 thinking-overload 修复后仍 ~3min + 2 次 seal-gate 催促

> **状态**：待实施（**第一步是取证，不是改 prompt**）
> **来源**：EPM dogfood thread `339512dd`（2026-06-22，跑在已含 PR#161 的 dev 上）
> **优先级**：🟠 P1 性能（端到端最大单项耗时，但已能 seal，非崩溃）
> **前置**：[2026-06-18-data-analyst-thinking-overload-spec.md](2026-06-18-data-analyst-thinking-overload-spec.md)（已合 PR#161）。本 spec 是它的**残留追踪**，不重复其结论。

## 〇、给实施 agent 的一句话

06-18 那刀（删参数审计 + outlier 下沉 + prompt 收窄）解决了「turn 内超时、永不 seal、原地打转」的**崩溃级**问题。但 6-22 dogfood（跑在含该修复的 dev 上）显示 data-analyst **仍**花了 ~3min 且被 `SealGateMiddleware` 催了 **2 轮**才封存——说明还有**残留 thinking 成本**。**本 spec 第一步是取证这个残留 thinking 到底在算什么**（当前训练数据 JSONL 不录 subagent thinking，必须换源拿到），**证据到手再决定改不改、改什么**。在没有 thinking 内容实证前，不要凭猜改 prompt。

## 一、现象（实证，thread `339512dd` gateway log 时间线）

```
16:39:47  data-analyst 派遣（trace=21a5af11）
16:40:54  SealGateMiddleware: seal reminder for data-analyst (count=1)   ← 第 1 次催
16:41:40  SealGateMiddleware: seal reminder for data-analyst (count=2)   ← 第 2 次催
16:42:15  data-analyst 收尾（subagent 序列结束）
                                                  ≈ 2min28s 墙钟，2 次 seal 催促
```

对比：code-executor 全程 ~44s。data-analyst 是端到端最大单项（lead 派遣间隙另算）。

与 06-18 的关键区别：
- **06-18**：turn 根本没结束 → 900s 超时 → handoff 从未产出 → SealGate 永不触发（结构救不了）。
- **6-22（本次）**：turn **结束了**、handoff **产出了**、SealGate **触发并催了 2 次**最终 seal 成功。**这是「收尾时漏 call、被门催回」类**（SealGate 设计内能 cover），不是「收不了尾」类。

所以残留问题更可能是：**判读 thinking 仍偏重（但没到撑爆 turn），且收尾时没一次性 seal、要门催**。

## 二、第一步：取证（实施 agent 必须先做这步）

当前 `TrainingDataMiddleware` 写的 `.deer-flow/training-data/auto-collected/<thread>.jsonl` **只录最终 turn、不含 subagent 中间 thinking**（本次排查实测：data-analyst 那条 record 的 `thinking` 字段为空）。要拿 data-analyst 实际 thinking，用下列之一：

1. **复跑一次 EPM dogfood**，开 debug 抓 subagent 的 model 输入输出（含 thinking delta），或
2. 解码 lead 层 checkpoint / subagent transcript（`.deer-flow` 下 subagent 的 jsonl，若有留存），或
3. 临时在 data-analyst 链上加一次性 thinking-size 探针（`len(thinking)` 落盘），复跑取数后撤。

**取证要回答**：
- data-analyst 这 ~2.5min 里 thinking 写了多少 token？主要在干什么？（还在搬 outlier？还在草拟 seal JSON？还在走某种隐式决策树？）
- 2 次 seal 催促之间它在做什么——是「判读没做完」还是「判读做完了但没调 seal、在 thinking 里反复确认」？

## 三、可能的残留根因（**假设，待 §二 证据确认或推翻**）

> ⚠️ 以下是基于 06-18 spec 经验的**假设清单**，不是结论。逐条需 §二 证据支持才动。

- **H1：outlier 引用仍偏重**。06-18 把 outlier 搬运下沉到 `_outliers.json`，data-analyst 只「引用最该警惕的几条」。但本次 `outlier_count=5`、`outlier_diagnostics_count=65`——若 prompt 仍诱导它通读 65 条再挑 5 条，thinking 仍会展开 65 条。验证：看 thinking 是否逐条过 65 条。
- **H2：seal JSON 在 thinking 里反复草拟**。06-18 现象之一是「用散文草拟 seal 参数两三遍」。若残留，2 次 seal 催促正是「草拟→没真调→门催→再草拟」。验证：看 2 次催促间 thinking 是否在拼 JSON。
- **H3：判读判据没钉死 read**。06-18 spec §1.6 指出判据已在 `by-experiment/epm.md`，但 data-analyst 仍可能现编判读。验证：看 thinking 是否 read 了 epm.md、还是在重新推「开臂↓=焦虑↑」这类已成文的判据。
- **H4：thinking_enabled 本身对判读 agent 的边际收益 vs 成本**。判读确实需要推理，但若大头花在机械确认而非真推理，可考虑对 data-analyst 调 thinking 预算（**谨慎**——见 §五风险，别把响亮问题变哑）。

## 四、设计原则（确定方向后遵循）

1. **守 06-18 的边界**：失败若仍是「thinking 背机械动作」→ 继续把机械动作逐出判读路径（下沉到工具/旁路），**不在 seal 门打补丁**。
2. **若是「收尾漏 call、门催回」**（H2）→ 这是 SealGate 设计内能 cover 的，残留成本主要是「催回的额外 turn」。治本是让 data-analyst **判读完一次性 seal**（prompt 收尾纪律），减少催回轮次。
3. **deepseek 正面提示**：prompt 写「判读完成后，立即用以下字段调 `seal_data_analyst_handoff`」，给一个**最小字段清单**让它照填，而非让它自行组织 JSON（减少 thinking 里草拟）。
4. **判读 agent prompt 绝不含「遍历 N 条」「走 a–f 决策树」**（memory `feedback_seal_fourth_root_cause_thinking_overload_turn_timeout` 铁律）——若 §二发现 prompt 仍有此类诱导，删之。

## 五、风险与注意事项

1. **别调 thinking_enabled/timeout 旋钮当首选**（memory + 06-18 spec §6.2）：会把「响亮的慢」变「哑的判读质量下降」。只有 §二 证据明确指向「thinking 花在机械确认而非推理」时才考虑，且要有判读质量回归保护。
2. **判读质量最敏感**：data-analyst 产出直接进 report 给导师看。任何 prompt 收窄都要确保「混杂排查 / 效应量 / 离群警示」judgment 不退化——改前后用 golden case（若有 EPM baseline）或人工核对同一 thread 的判读质量。
3. **取证优先**：没有 thinking 内容实证就改 prompt = 重蹈「凭猜叠补丁」。本 spec 故意把取证设成强制第一步。
4. **可能结论是「不用改」**：~2.5min 判读 + 2 次门催，若证据显示 thinking 主要在做**真判读推理**（而非机械搬运），那这是 deepseek 推理的合理成本，SealGate 催回也按设计工作——此时**记录结论、关闭即可**，不强行优化。

## 六、验收标准（方向确定后）

- 取得 data-analyst 残留 thinking 的实证（token 量 + 在算什么）。
- 据证据判定：① 仍有机械动作可逐出 → 逐出并验证 thinking 缩小、seal 催促降到 ≤1 次；或 ② 是合理判读成本 → 记录结论关闭。
- 若改 prompt：data-analyst 判读质量不退化（混杂/效应量/离群三项 judgment 仍在）。

## milestone 建议

属 data-analyst 判读链打磨，与 06-18 thinking-overload 同一 track。本 spec 完成后更新该 track 的 milestone：标注「崩溃级已修（PR#161），残留耗时已取证 + 处置」。
