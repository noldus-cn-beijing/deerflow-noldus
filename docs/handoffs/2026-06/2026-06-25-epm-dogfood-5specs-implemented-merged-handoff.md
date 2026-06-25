# 交接：EPM dogfood (thread 0e72d605) 端到端成功后,5 份 spec 全实施 merge dev

> **会话日期**:2026-06-25
> **代码基线**:dev HEAD `1bc3dbf1`(#209,最后一份 spec D 合入)
> **本会话角色**:取证 + 写 spec(零生产代码改动,实施交别的 agent;别的 agent 已全部实施 merge)
> **一句话**:EPM 真实数据 dogfood(thread `a6e3775c` 双 thread + `0e72d605` 单 thread)端到端**成功**,但暴露 5 类问题。本会话**全部取证定位真根因 + 写 spec**,其中**纠正了多处误诊**(守 ETHO-2:不跑代码不凭推断)。5 份 spec 已由实施 agent 全部 merge 到 dev(B 按设计暂缓)。**关键贡献:推翻了「核盘竞态」「double-seal 良性」「read_file 回绕」「Gateway 被画图拖垮」等多个表象归因,落到真根因。**

---

## 〇、给接手 agent 的一句话

本会话围绕**两次 EPM dogfood** 暴露的 5 类问题,**全部取证 + 写 spec + 由实施 agent merge dev**。接手前先读本文 §三「5 份 spec 状态表」——尤其注意 **4 个问题里有 2 个是同一根因链**(read_file 死循环 → run_metric_plan max_turns 中断),和 **B 暂缓是「假设的痛点经实测不存在」而非否决**。**剩余唯一动作 = §五 dogfood 验收**(代码已 merge,但 merge≠运行时生效,需 `make stop && make dev` 重跑核实)。

---

## 一、本会话完整脉络(从 dogfood 到 5 份 spec)

1. **起点**:用户跑完 EPM 真实数据(`Raw data-EPM-Xuhui-28`,28 trials,XX/XY/YY/YZ 四组)的 dogfood,thread `a6e3775c`(112/112 画图成功) + `0e72d605`(113 图 + 6 段报告,端到端成功)。
2. **dogfood 虽成功,但暴露 5 类问题**,逐一取证:
   - ① chart-maker handoff `sealed_by=model`(应为 run_plan)——表象像「核盘竞态良性」,**取证推翻** → 真根因=**double-seal 覆盖**。
   - ② run_chart_plan results dict 用 chart id 做 key,per_subject 112 图共享 4 id 互相覆盖(实测顺带发现)。
   - ③ Gateway 画几百张图压力(用户提 celery 化)——**实测推翻** → sync 工具本就跑 worker 线程,event loop 没被阻塞。
   - ④ code-executor 反复 read_file 找 statistics 死循环 20+ 轮 → 真根因=**read_file 两个越界静默 bug** + 策略错。
   - ⑤ lead + 4 subagent thinking 全英文(用户全程中文)——根因=语言约束只覆盖「输出」不覆盖 thinking。
   - ⑥ 前端 console `Unexpected tool message outside a processing group {}` ——表象像空占位/中断悬挂,**取证推翻** → 真根因=**流式分片时序**(tool_calls 未到时不开组)。
   - ⑦ run_metric_plan「第一次中断」——**查明**=read_file 死循环耗尽 max_turns=20 → turn 20 被 hard-limit break 掐断(是 ④ 的下游症状)。
3. **5 份 spec 写完 + 实施 agent 全部 merge dev**(见 §三)。

---

## 二、纠正的误诊(本会话核心价值,守 ETHO-2)

| 问题 | 原表象/误诊 | 取证推翻 → 真根因 |
|---|---|---|
| ① double-seal | problem doc:「ProcessPoolExecutor 未 flush 就核盘的竞态,良性」 | 代码 `shutdown(wait=True)` 已在,核盘时 worker 全退出。`sealed_by=model` + 中文长句 summary = chart-maker **手调 seal_chart_maker_handoff 覆盖** run_plan 确定性封存(LLM 自报,病非善)。`n_rendered=4` 是 chart-maker 自己 bash ls 的 LLM 自述,非工具竞态。 |
| ③ Gateway 压力 | 用户直觉:「画几百张图把 Gateway 压垮,要 celery」 | 实测(代码路径 + P99) sync `@tool` 经 `ainvoke→run_in_executor` 跑 **worker 线程,不阻塞 event loop**;112 图 5s,非瓶颈。**痛点不存在,B 暂缓。** |
| ④ read_file | LLM 自述:「read_file 回绕到第 1 行」 | 真因是 **两个 bug**:只传 start_line 不传 end_line 时切片整段被忽略返回全文件头部(像回绕);start_line 越界静默返回空无提示(死循环根源)。 |
| ⑥ 前端孤儿 | 初判:DanglingToolCall 空占位 / 中断悬挂 | 解 langgraph checkpoint 真实 37 条序列 + 核各 group 渲染安全性,**排除**占位/中断 → 真因=**流式分片时序**(AI message 的 tool_calls 分片未到、content 暂空那一帧不开组,tool result 流到成孤儿)。 |
| ⑦ 中断 | LLM 自述:「run_metric_plan was interrupted」(像工具崩) | 解 checkpoint 24-turn timeline,真因=**max_turns=20 hard-limit 在 tool_call/tool_return 之间 break**(④ 死循环耗尽 turn 预算的下游)。 |

---

## 三、5 份 spec 状态表(全部已 merge dev,代码层坐实)

| spec | 文件 | PR | 状态 | 代码坐实锚点 |
|---|---|---|---|---|
| **A** seal-once + results key | `docs/superpowers/specs/2026-06-25-chart-maker-seal-once-and-results-key-uniqueness-spec.md` | **#199** `ccaf17d9` | ✅ merged | `seal_handoff_tools.py:680/704`(force 门)、`run_chart_plan_tool.py:297`(force=True)、`:353/367`(results index key)、`:201`(chart_meta list) |
| **B** render decouple | `docs/superpowers/specs/2026-06-25-chart-render-decouple-subprocess-spec.md` | **#197** `3ed58c5a` | ⏸️ **暂缓**(实测 P1 FAILS,不实施) | 无运行时代码改动;spec 状态回写「暂缓·前置未坐实」 |
| **C** read_file + code-executor | `docs/superpowers/specs/2026-06-25-read-file-line-range-bug-and-code-executor-no-read-plan-spec.md` | **#206** `29584236` | ✅ merged | `sandbox/tools.py:1421-1429`(越界 fail-loud + 单边 start/end_line)、`code_executor.py:59-65`(不 read 大 plan prompt) |
| **D** thinking 语言 | `docs/superpowers/specs/2026-06-25-subagent-thinking-language-constraint-spec.md` | **#209** `1bc3dbf1` | ✅ merged | 4 subagent system_prompt 点名 thinking、`lead_agent/prompt.py:540`(same language) |
| **前端** grouping | `docs/superpowers/specs/2026-06-25-frontend-streaming-orphan-tool-message-grouping-spec.md` | **#204** `63d237ed` | ✅ merged | `utils.ts:79-95`(processing 组兜底)、去掉 `console.error + drop` |

> ⑦(run_metric_plan 中断)无独立 spec——根因回填进 **spec C §七**(它是 ④ 的下游,修 C 即根治),`297c0ad9` 提交。次生缺陷(max_turns break 掐断未返回 tool → 进程池白跑)记录但不单独修(改终止条件高危)。

---

## 四、4 个问题的根因关系图(接手必读)

```
④ read_file 越界 bug + code-executor 策略错  ← 根源
   ├─→ code-executor read_file 死循环 20+ 轮(观感硬伤)
   └─→ 耗尽 max_turns=20 → ⑦ run_metric_plan 在 turn 20 被 break 掐断
                            → seal-resume 补轮重跑(进程池白跑一次)
   【spec C 一箭双雕:修 ④ 即同时消除 ⑦】

① double-seal 覆盖 ── 独立 ── spec A(封存只允许一次门)
② results id 覆盖  ── 与 ① 同族(later seal 覆盖确定性产物)── spec A(R1+R2 必须同 PR)
⑤ thinking 语言    ── 独立 ── spec D
⑥ 前端孤儿 grouping── 独立(流式时序)── 前端 spec
③ Gateway 压力     ── 实测痛点不存在 ── spec B 暂缓
```

**关键洞察**:表面 4 个独立问题,其中 ④+⑦ 是同一根因链。spec C 是这次最高杠杆的修复(治死循环观感 + 消除中断 + 省进程池白跑)。

---

## 五、剩余唯一动作:dogfood 验收(代码已 merge ≠ 运行时生效)

守铁律「merge≠生效」「代码有修复≠现象消除」(`feedback_code_has_fix_not_equal_bug_eliminated_seal_react_floor`)。**全部 merge 不代表 dogfood 现象消失,需重跑核实**:

1. **核进程加载新代码**:`make stop && make dev`,验 uvicorn 进程启动时间晚于最后 commit(`ps -eo pid,lstart,cmd | grep uvicorn`) + `inspect.getsource` 验运行时模块含改动。
2. **重跑 EPM dogfood**(`/noldus-insight-e2e ~/DemoData/real_data/Raw\ data-EPM-Xuhui-28`),逐项验收:
   - **spec A**:chart-maker handoff `sealed_by=**run_plan**`(不是 model);chart_files=112;chart-maker trace **不再手调 seal_chart_maker_handoff**(撞门被拒)。
   - **spec C**:code-executor trace **不再 read_file 死循环**(应直接 ls→run_metric_plan);不撞 max_turns;run_metric_plan **一次跑完不中断**(⑦ 消除)。
   - **spec D**:lead + subagent thinking **中文占比大幅提升**(不强求 100%,thinking 是模型倾向无法上确定性门)。
   - **前端**:浏览器 console **无** `Unexpected tool message`。
3. **若验收发现现象仍在**:别只信「PR merged」,按各 spec 的「风险/注意事项」段排查(如 dogfood 前是否真重启进程、是否撞 test isolation 污染)。

---

## 六、本会话沉淀的 memory(接手可 recall)

- `feedback_chart_reconcile_loop_key_must_be_unique_not_chart_id` — 核盘/计数循环 key 必须唯一,per_subject 不能用 chart id
- `feedback_deterministic_seal_and_llm_self_report_dual_track_masks_count_bug` — 确定性封存+LLM自报双轨,后者掩盖前者计数 bug;修封存轨前必查计数轨
- `feedback_subagent_tool_interrupted_may_be_maxturns_break_not_tool_crash` — subagent「工具中断」未必工具崩,可能 max_turns hard-limit break;诊断先核 turn 数

> 另:本会话沉淀了 PR#197 的 memory `reference_sync_tools_run_off_event_loop_via_run_in_executor`(sync @tool 经 ainvoke 跑默认 ThreadPoolExecutor 不阻塞 loop)——避免未来对任何 sync 工具重复争论「会不会拖垮 Gateway」。

---

## 七、风险与注意事项(容易踩的坑)

1. **别把 B 当遗留待办又去做** —— B 暂缓 = 实测证明「Gateway 被画图拖垮」不存在(sync 工具跑 worker 线程)。重启条件三选一(sync 工具被强制跑 loop 线程 / 真实并发打满默认 executor / fork 成本被 live dogfood 证明是瓶颈),目前都不成立。
2. **spec A 的 R1+R2 必须理解为同族** —— 当时 dogfood handoff 112 张正确是靠 chart-maker 手调 seal 自己 ls 补全(R1 的 LLM 自报「救」了 R2 的 results id 覆盖)。两者同 PR 修(#199 已同修),缺一即回归。
3. **⑦ 中断的次生缺陷潜伏** —— max_turns break 掐断未返回 tool 会让进程池白跑。修 C 后基本不触发(正常 subagent 不会在 turn 20 才调首个重工具),但若未来别的 subagent 烧 turn 仍可能。改 break 终止条件是高危,需独立 spec + 充分回归。
4. **dogfood 验收 thinking 语言别强求 100%** —— thinking 是模型(GLM/deepseek)倾向,prompt 是上限,无法像 seal 上确定性门(thinking 非结构化产物)。看占比改善即可。
5. **前端 grouping 是回归高发区** —— `utils.ts` 历史多次回归(2026-05-25/06-04)。若改它必跑全部 group 类型回归。

---

## 八、下一位 Agent 的第一步建议

1. 读本文 §三 状态表 + §四 根因关系图,理解 5 份 spec 的依赖(C→D 同改 code_executor.py 已分别 merge;其余正交)。
2. **核心动作**:跑 §五 dogfood 验收(`make stop && make dev` 重启 + 重跑 EPM,逐项核 spec A/C/D/前端现象消除)。**这是「代码有修复≠现象消除」的必经验证。**
3. 若验收全绿 → 更新 milestone(见下);若有现象残留 → 按对应 spec 风险段排查,写 problem doc。
4. 别动 B(暂缓);别为 ⑦ 次生缺陷单独改 executor(高危)。

---

## milestone 建议

归入「chart-maker 确定性化 / harness 工具鲁棒性 / 交互语言一致性」三个 track 的交叉 checkpoint。本会话:「EPM 真实数据 dogfood 端到端成功,但暴露 5 类问题。全部取证写 spec + 实施 agent merge dev:① double-seal 覆盖(#199,封存只允许一次门)② results id 覆盖(#199,同 PR)③ Gateway 画图压力(#197 实测痛点不存在→暂缓)④ read_file 越界 bug + code-executor 死循环(#206,工程修+prompt 双管)⑤ thinking 语言(#209,约束扩展到 thinking channel)⑥ 前端流式孤儿 grouping(#204,processing 组兜底)⑦ run_metric_plan 中断(=④下游,max_turns break,回填 spec C)。**纠正多处误诊(核盘竞态/double-seal 良性/read_file 回绕/Gateway 被拖垮),落到真根因。剩余:dogfood 重跑验收现象消除。**」
