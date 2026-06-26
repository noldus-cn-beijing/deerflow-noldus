# 交接：chart-maker run_chart_plan 系列问题取证 + spec（三份 dogfood problem 全就绪待实施）

> **会话日期**：2026-06-24（午后～傍晚）
> **代码基线**：dev HEAD `24383de8`（#192，含 F1 argv 预解析 + F2 fail-safe warning）
> **本会话角色**：取证 + 写 spec，**零生产代码改动**（用户每次只授权调查 + 写 spec，实施交别的 agent）
> **一句话**：围绕 chart-maker 的确定性执行工具 `run_chart_plan`，连续三次 dogfood 暴露三类问题，本会话全部取证定位根因 + 写 spec。**关键贡献是纠正了诊断 doc 的两处误诊**（守 ETHO-2 教训：不跑代码不凭推断），三份 spec 待实施。

---

## 〇、给接手 agent 的一句话

本会话产出 **3 份待实施 spec + 1 份 pattern 文档 + 4 条 memory**。三份 spec 对应三次 dogfood 的三类问题，**其中两份纠正了 problem doc 的根因误判**——接手前先读本文 §四「三份 spec 现状」，别照 problem doc 原始归因实施（problem doc 的"封存门覆盖率"和"F4 改尖括号"都被取证推翻）。

---

## 一、本会话完整脉络（从最早到最新）

这是一条长链，从「写 run_chart_plan spec」到「它上线后连环暴露问题」：

1. **起点**：接 handoff `2026-06-24-chart-maker-run-chart-plan-batch-execution-handoff.md` → 取证发现 handoff 的「loop-detection 熔断诱因」**未坐实**（thread 15c9805e 画全 113 张未熔断）→ 据实写 `run_chart_plan` spec（执行确定性化，不挂熔断）。
2. **固化范式**：用户要把 batch 执行固化成 feature → 写 **pattern 文档**（不抽公共代码，守 overlap 接受重复）。
3. **run_chart_plan 上线（#190，别的 agent 实施）后第一次 dogfood**（thread 89be7344）→ problem doc `permissionerror.md` → 取证：plot 脚本 `fig.savefig(args.output)` 不 resolve `/mnt` → 写 **F1（argv 预解析）+F2（fail-safe warning）spec** → 别的 agent 实施合入 **#192**。**这暴露了我原 run_chart_plan spec 的遗漏**（误以为「直接透传 args」通用，实际依赖脚本自 resolve）。
4. **#192 merge 后第二次 dogfood**（thread 590fbbd3）→ problem doc `silent-drop-completed-handoff.md` → 取证**纠正诊断**：不是封存门覆盖率，是 **run_chart_plan 子集调用 per-call 覆盖 handoff**（make dev 没重启 F1 没生效 + chart-maker 逐个 only_chart_ids 调）→ 写 **subset-overwrite spec**。
5. **同一次 dogfood 的 UX 问题** → problem doc `intent-gate-signals-tags-leaked.md` → 取证**纠正诊断**：F4（改尖括号）会破坏后端 guardrail/lead → 写 **前端 render strip spec**（只取 F1）。

---

## 二、当前进展（✅ 已完成）

✅ **取证全部用代码坐实**（守 ETHO-2，不凭推断）：SSE trace 分析 + 进程内手段 c 复现 + git 时序拓扑 + 运行时模块源码验证。
✅ **3 份待实施 spec 写完**（见 §四）。
✅ **pattern 文档写完**：`docs/superpowers/specs/2026-06-24-deterministic-batch-execution-tool-pattern.md`（12 铁律，run_metric_plan/run_chart_plan 抽象）。
✅ **4 条 memory 沉淀**（见 §五）。
✅ **纠正了 problem doc 的两处误诊**（silent-drop 的"封存门"、intent-leak 的"F4 改尖括号"）。

❌ **三份 spec 均未实施**（用户只授权写 spec，实施交别的 agent）。
❌ run_chart_plan 的根因 B（子集覆盖）+ 前端泄露 **代码未动**，dogfood 还会复现。

---

## 三、关键上下文（项目结构 + 锚点）

- **代码位置**：harness 在 `packages/agent/backend/packages/harness/deerflow/`；ethoinsight 库在 `packages/ethoinsight/ethoinsight/`；前端 `packages/agent/frontend/src/`。
- **run_chart_plan 工具**：`tools/builtins/run_chart_plan_tool.py`（#190 上线，#192 加 F1）。范本是 `run_metric_plan_tool.py`。
- **封存对账门**：`tools/builtins/seal_handoff_tools.py:_reconcile_chart_maker_payload`(525)，单一注入点 `_seal_handoff_to_workspace`(701)。
- **chart-maker**：`subagents/builtins/chart_maker.py`（system_prompt L53-61 教 run_chart_plan + only_chart_ids）。
- **前端 strip**：`src/core/messages/utils.ts:extractContentFromMessage`(317，render 入口) + `INTERNAL_MARKER_RE`(526，只覆盖尖括号)。
- **dogfood 证据**：`/tmp/noldus-e2e-runs/20260624-161811/`（thread 590fbbd3，含 sse-0..7 + final-body + analyze）。
- **⚠️ make dev 不 hot-reload**：`serve.sh:140-147` Gateway reload 默认关（watchfiles 与 ProcessPoolExecutor fork 冲突）。**merge ≠ 运行时生效**，dogfood 前必 `make stop && make dev` + 核进程启动时间。

---

## 四、三份 spec 现状（接手重点，含对 problem doc 的纠正）

### spec 1：`2026-06-24-run-chart-plan-permissionerror-argv-resolve-spec.md` —— **已实施合入 #192，无需再做**
- F1（Step 5 argv `replace_virtual_path` 预解析）+ F2（`_cli.py:135` debug→warning）。
- 已 repro 验证 113/113。**dev `24383de8` 已含**（核：`run_chart_plan_tool.py:202` 有 `args = [replace_virtual_path(a, thread_data) for a in args]`）。

### spec 2：`2026-06-24-run-chart-plan-subset-overwrite-silent-drop-spec.md` —— **待实施（高优先）**
- **⚠️ 纠正了 problem doc 的根因归因**：problem doc 说「封存门 `_reconcile` 覆盖率不足放过静默丢图」；**真根因（SSE 铁证 summary 全 `0/1`/`1/1` 无 `X/113`）= run_chart_plan 子集调用 per-call 整体覆盖 handoff + LLM 可用 only_chart_ids 任意缩 plan** → 成功单张 `1/1 completed` 覆盖失败全量 `1/113 partial`。
- **修法**：M1（handoff 以全 plan 为对账分母 + 子集核全 plan 磁盘不覆盖，照齐 run_metric_plan）+ M2（封存门对账总数不变式兜底，用 output 不用 id 因 per_subject id 多对一）+ M3（completed/failed 互斥）。
- **已坐实**：run_metric_plan 免疫（`aggregate_metrics_to_handoff(plan,...)` 传全 plan），是 run_chart_plan 该学的，无需修 metric。
- 改 `run_chart_plan_tool.py`（Step 7/8）+ `seal_handoff_tools.py`（reconcile 加 2.3）。

### spec 3：`2026-06-24-intent-gate-signals-frontend-leak-spec.md` —— **待实施（中优先，5 行前端）**
- **⚠️ 纠正了 problem doc 的修法**：problem doc 推荐 F1+F4；**F4（方括号→尖括号）/F2/F3 会破坏后端**——`[intent]` 被 `intent_classification_provider.py:40-56` 从历史 content 读校验、`[gate_signals]` 被 lead 读决策，从 content 抽走/改语法 = guardrail/lead 瞎了。
- **修法**：只取前端 render 层 strip（`utils.ts:extractContentFromMessage` 加 `stripControlSignalLines`），content 保留供后端读。**已坐实** `extractContentFromMessage` 3 类调用方（render/导出/派生）无回传后端 → 安全全覆盖、后端零改动。

---

## 五、本会话沉淀的 memory（接手可 recall）

- `feedback_chart_maker_loop_detection_burnout_premise_refuted_by_forensics` — 熔断诱因未坐实
- `feedback_deterministic_batch_execution_tool_pattern` — batch 执行范式（12 铁律）
- `feedback_align_template_must_verify_hidden_premise_not_just_shape` — 对齐范本核隐性前提 + 实跑别改写关键输入
- `feedback_batch_tool_subset_call_must_not_overwrite_full_plan_handoff` — 子集调用必以全 plan 为对账分母 + make dev 不 reload 陷阱

---

## 六、未完成事项（按优先级）

| 优先级 | 事项 | spec |
|---|---|---|
| **高** | 实施 subset-overwrite（M1/M2/M3）——否则下次 dogfood 仍可能静默丢图伪 completed | spec 2 |
| **中** | 实施前端 intent/gate_signals strip（5 行 + 测试）——v0.1 前必清的观感硬伤 | spec 3 |
| 中 | 重新跑 dogfood 验收（F1 已生效进程）——验 run_chart_plan 全量 113 一次画完 | — |
| 低 | run_chart_plan spec / pattern 文档的 milestone 收口（chart-maker 确定性化 track） | — |

---

## 七、风险与注意事项（容易踩的坑）

1. **别照 problem doc 原始归因实施 spec 2/3**——两份的根因/修法都被本会话取证纠正了（封存门❌→子集覆盖✅；F4 改尖括号❌→前端 strip✅）。spec 文件里写的是纠正后的。
2. **dogfood 前必核进程加载目标代码**：`make stop && make dev` 后验 uvicorn 进程启动时间晚于最后 commit（`ps -eo pid,lstart,cmd | grep uvicorn`）+ `inspect.getsource(tool.func)` 验运行时模块含改动。merge ≠ 生效。
3. **取证 subagent 行为别只看 handoff 表象**：`sealed_by=run_plan` 不代表工具产了全量；`summary` 分母揭示真实 n_total。看 SSE turn 序列 + 真实工具返回。
4. **改 seal_handoff_tools.py / run_chart_plan_tool.py 后裸导入两入口**：`PYTHONPATH=. python -c "import app.gateway"` + `make_lead_agent`（harness 导入环铁律）。
5. **spec 2 的 M2 对账用 output 不用 id**：per_subject 多个 chart 共享同一 id（如 28 个 `open_arm_time_ratio_bar`），id 集合相等会误判。
6. **frontend phase0-3/4/5 spec（17:12-17:17）不是本会话产出**——是别的 agent 同时写的前端 UX 计划，别混淆。

---

## 八、下一位 Agent 的第一步建议

1. 读本文 + `docs/problems/2026-06-24-chart-maker-run-chart-plan-silent-drop-completed-handoff.md`（problem）+ `docs/superpowers/specs/2026-06-24-run-chart-plan-subset-overwrite-silent-drop-spec.md`（纠正后的 spec），对比看本会话纠正了什么。
2. 若用户要实施 spec 2：先 `git log --oneline -1` 确认基线，按 spec §九锚点改 `run_chart_plan_tool.py` Step 7/8 + `seal_handoff_tools.py` reconcile，TDD 红→绿（T1 复现「子集成功覆盖全量失败」），裸导入两入口。
3. 若用户要实施 spec 3：改 `utils.ts:extractContentFromMessage`，加 `utils.test.ts` 用例，`pnpm check`。
4. 若用户要重跑 dogfood：先核进程加载了 F1（见风险 2），再跑，验收别只信 `status=completed`（核 `outputs/*.png` 数 == plan charts 数）。

---

## milestone 建议
归入「chart-maker 确定性化 / reward hacking 治理」track。本会话 checkpoint：「run_chart_plan 上线后三次 dogfood 连环暴露：① 熔断诱因证伪（执行确定性化仍写）② argv 透传 /mnt 致 savefig 崩（F1/F2 已修 #192）③ 子集调用覆盖 handoff 伪 completed（spec 待实施）④ 控制 tag 泄露前端（spec 待实施）。三份 spec 纠正了两处 problem doc 误诊。pattern 文档固化 batch 执行范式（12 铁律）」。
