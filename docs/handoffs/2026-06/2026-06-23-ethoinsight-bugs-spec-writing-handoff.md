# 交接：ETHOINSIGHT_BUGS.md 10 条 E2E bug —— 根因已全查清，为每条写修复 spec（写了 2 份，剩 4 份待写）

> **会话日期**：2026-06-23
> **来源**：用户在 dev.ethoinsight.com 跑 2 轮 EPM E2E（28 xlsx，数据飞轮）发现的 10 个 bug，原始清单 `~/ETHOINSIGHT_BUGS.md`
> **本会话做了什么**：① 把 10 条 bug 全部做了代码级根因取证（4 个并行 Explore 子代理）；② 写成根因对照文档；③ 用户指示「不亲自改代码，每个修复写 spec」，已写 ETHO-8、ETHO-2 两份 spec，剩 ETHO-1/7+9/5/3 待写。
> **为什么交接**：上一个 agent（我）的工具调用 XML 格式持续失误（`<invoke>` 标签写错），无法可靠继续写文件。新 agent 接手即可正常工作——纯是输出格式问题，与任务无关。

---

## ⚠️ 接手第一步：核实文件状态

```bash
cd /home/wangqiuyang/noldus-insight
git log --oneline -1 origin/dev   # 应是 58588dc6（PR#174）；本地 dev 落后，需 pull
ls -la docs/superpowers/specs/2026-06-23-*.md
ls -la docs/problems/2026-06-22-ethoinsight-bugs-e2e-root-cause.md
```

**确认成功落盘的**（收到过成功回执，应完整）：
- `docs/problems/2026-06-22-ethoinsight-bugs-e2e-root-cause.md` — **10 条 bug 根因总账，最重要的产物**
- `docs/superpowers/specs/2026-06-23-template-candidate-ab-order-deterministic-spec.md` — ETHO-8 spec
- `docs/superpowers/specs/2026-06-23-lead-dispatches-illegal-bash-subagent-registry-consistency-spec.md` — ETHO-2 spec
- memory `feedback_code_has_fix_not_equal_bug_eliminated_seal_react_floor.md` + MEMORY.md 索引补一行

**未成功落盘（需重写）**：
- ETHO-5 spec（`2026-06-23-grouping-single-file-extrapolation-spec.md`）— 写了两次都没拿到成功回执，先 `ls` 确认，不在/残缺就按 §四 ETHO-5 重写。

---

## 一、本仓库当前状态（接手须知）

- **本地 dev 落后**：会话初 pull 到 `45824364`（#172），之后用户合了 **PR#174（`58588dc6`，chart-budget-ask-user = 旧 handoff 的 #2）**。接手先 `git checkout dev && git pull`。
- **工作树有大量在制品**（别误删别误提交）：A+B DB 迁移三件套（`run-db-migrations.sh` + `deploy-via-tar.sh` 改 + `deploy-via-tar-sop.md` 改，另一条 P0 待办，见 `docs/handoffs/2026-06/2026-06-22-db-migration-automation-and-chart-maker-two-bugs-handoff.md`）、本会话产物、若干 untracked spec/handoff。
- **写 spec 不碰代码**（用户明确指示），无需开 worktree；spec 直接写进主仓 `docs/superpowers/specs/`。

---

## 二、10 条 bug 根因总账（详见 `docs/problems/2026-06-22-ethoinsight-bugs-e2e-root-cause.md`）

判断标准：**「现象在当前 dev 代码（45824364/58588dc6）还会不会发生」**，不是「有没有 PR」。

| # | 状态 | 根因一句话 | spec |
|---|---|---|---|
| ETHO-1 | 🔴降级未根治 | report-writer 漏 seal：ReAct 终止时机 LLM 决定；三道防线（SealGate `_MAX_REMINDERS=2` 后放行 / seal-resume / auto-seal 前提 report.md 落盘）只降触发率 | ✅ **infra 加固**（走 middleware/deerflow） |
| ETHO-2 | 🔴仍存在 | registry「校验集合 `_SubagentLiteral` ≠ 可派集合 `BUILTIN_SUBAGENTS`」不自洽 | ✅ **已写** |
| ETHO-3 | ◐主流程消除/残留 | 133K plan_metrics 截断：report-writer 侧 #169 已修；**code-executor 分诊侧（code_executor.py:59）仍可能截断** | ✅ 残留缺口 |
| ETHO-4 | ❌设计非bug | identify 故意不用 experiment 字段，守 HITL 反问铁律 L484。**回给用户对齐，不修** | ❌ 不写 |
| ETHO-5 | 🟡仍存在 | 分组单文件外推：工具侧 per_file_grouping 全量正确，prompt L493「优先用」是软建议 | ✅ **重写**（纯 prompt） |
| ETHO-6 | ✅已消除 | chart 参数 #168 按 requires_columns 确定性裁剪 | ❌ 不写 |
| ETHO-7 | 🔴病根 | 决策点数量/形态 100% lead LLM 自由裁量，反问合并规则（prompt.py:491）是建议非强制 | ✅ 与 9 合并 |
| ETHO-8 | 🟡仍存在 | 模板 A/B 顺序：候选列表 `identify_ev19_template_tool.py:600` 无 sort | ✅ **已写** |
| ETHO-9 | 🟢仍存在 | ask_clarification 的 options 可选，lead 自由传（clarification_tool.py:17） | ✅ 与 7 合并 |
| ETHO-10 | 🟢仍可能 | chart-maker 并行 bash 被 loop_detection 频率熔断（`_DEFAULT_TOOL_FREQ_WARN=3`）partial strip | ⚪ 低优先 |

**ETHO-7/8/9 共享病根** = 交互流程决策全由 lead prompt 自由裁量、无确定性约束。ETHO-8 是纯确定性数据修复（候选 sort），已单独成 spec；ETHO-7+9 走 path_registry + guardrail。

---

## 三、用户拍板的关键决策（写剩余 spec 必须遵守）

1. **范围**：写「能真正解决的」+ ETHO-1（infra）。即 ETHO-1/2/5/7/8/9 + ETHO-3 残留。ETHO-10 低优先（可先记不写 spec）。ETHO-4 不写（是设计）。
2. **粒度**：ETHO-8 独立（已写）；**ETHO-7+9 合并一份**；ETHO-2 独立（已写）；ETHO-5 独立；ETHO-1 独立；ETHO-3 残留独立。
3. **ETHO-1 方向（用户亲自指出）**：根因「LLM 决定何时停 prompt 约束不了」**不代表 infra 约束不了**。走 middleware/deerflow/langgraph infra 解决，**不是只能降级**。
4. **ETHO-2 方向（用户拍板）**：registry filter（infra 治本）+ prompt 消除动机（双层）；**不新建批量扫描工具**（identify_ev19_template 已是批量扫描器，吃 uploaded_files 全量返回 per_file_grouping）。

---

## 四、剩余 spec 的设计要点（已调研，可直接落笔）

### ETHO-1 spec（infra 加固堵死漏 seal）—— 已做完整 infra 调研

**根因精确锚点**：
- `seal_gate_middleware.py:58` `_MAX_REMINDERS=2`，第 5 条规则「reminder ≥ cap → return None（放行）」——漏网直接原因。
- `executor.py:1400` `_validate_handoff_emitted` **只查 workspace/handoff_*.json，不查 outputs/report.md**。
- `executor.py:301` `_attempt_auto_seal_from_artifacts` 对 report-writer 前提是 **report.md 已落盘且非空**（L350-352）；「连 report.md 都没写盘」时返回 False → fall through FAILED → lead 重试（prompt.py:258/272 匹配 "terminated without emitting"）。

**关键发现**：memory `feedback_seal_missing_root_cause...` 说 SealGate「jump_to=model 拦在终止前、结构性 100%」，**但当前代码实际是「提醒 2 次就放行」**——MEMORY 与代码不符，ETHO-1 复现正暴露这个落差。

**推荐 infra 方案（子代理调研 + 已核实 after_agent hook 真实存在）**：
- **方案②（首选，高可行）**：新建 `FinalSealEnforcementMiddleware`，实现 **`after_agent` hook**（已核实：memory/token_budget/loop_detection/todo/training_data 共 5 个 middleware 都用了 after_agent，语义可靠）。agent 即将终止时检查 handoff 缺失则调 auto-seal。它是图正常终止点、不 jump、可执行重操作。注册在 `executor.py` 的 middleware 链。
- **方案③（可选辅助）**：`_attempt_auto_seal_from_artifacts` 扩展——report.md 缺失时从 final_state 最后 AIMessage 内容重建。风险：AIMessage 可能只是「报告已生成」宣言非报告本体，信息损失。**先不做，等数据证明需要。**
- **不推荐方案①**（去掉 _MAX_REMINDERS 改硬循环）：违反 langgraph 设计 + 与「第4类 thinking 超时」根因（memory `feedback_seal_fourth_root_cause_thinking_overload_turn_timeout`）冲突——turn 内超时时 after_model 永不触发，硬循环救不了。
- **验收必含**：`sealed_by` 四分类标记（gate_reminder/framework_rebuild/after_agent_gate/ai_message_reconstruction）+ 触发率可观测（memory `feedback_fallback_trigger_rate_must_be_observable`）。回归测试含「thinking 超时时 after_agent 不触发应标 TIMED_OUT」用例。
- **守 memory**：`feedback_isolate_root_cause_before_stacking_fallback_mechanisms`——方案②是「在正确的终止点补」不是「再加一层盲兜底」。

### ETHO-7+9 合并 spec（交互决策点确定性）

**根因锚点**：
- ETHO-7：`prompt.py:491`「反问合并规则」是建议非强制；`prompt.py:307` intent 分类（E2E_FULL vs E2E_FULL_ASKVIZ）凭 LLM 对「模糊程度」理解，每轮不同。现有 `IntentPostStepAskGateProvider` 只拦 task() 派遣，不约束反问决策点。
- ETHO-9：`clarification_tool.py:17` options 是可选参数，lead 自由传；prompt.py:333 只在「是否出图」样板硬编码了 options。
- **共同病根**：交互决策全 prompt 自由裁量、无确定性约束。
**修法方向**：用已有 `path_registry.PATHS` + guardrail provider 模式，把「反问顺序/合并/选项格式」从 prompt 自由裁量升级为结构化可执行约束。**注意**：PR#174（chart-budget-ask-user）已把「是否出图」决策点改成 lead 反问——写 ETHO-7 前先读 PR#174 改了什么（`58588dc6`），别和它重复/冲突。这份较大，可能需先调研 path_registry 现状再落笔。

### ETHO-3 残留缺口 spec（code-executor 分诊侧）

**根因锚点**：report-writer 侧 #169 已修（prompt 导向 `_metric_metadata.json`）；但 `code_executor.py:59`「用 read_file/ls 查证失败细节」在分诊失败场景若读 133K plan_metrics 仍触发 50K 截断（`sandbox/tools.py:1738`）。
**修法方向**：分诊侧也走 metadata（或对 plan_metrics 用 read_file 的 offset/行范围参数，`tools.py:1372-1394` 有 start_line/end_line 能力）。小改动。

### ETHO-5 spec（重写）—— 纯 prompt

**根因锚点**：`prompt.py:493-494`「lead 构造分组时**优先用** per_file_grouping…**不需要逐个** inspect」是软建议。工具侧 `identify_ev19_template_tool.py:518` `for f in uploaded_files` 全量扫描正确。
**修法**：① L493「优先用」→「**必须**先读完整 per_file_grouping、分组判定**必须基于全部文件**」；② L483 硬性反问场景段（L484 范式有「禁止猜」铁律）后补对称的「**分组判定基于全量、禁止以偏概全**：绝不用单个/少数文件 inspect 结果对全部文件下分组结论」。不碰工具。正面提示为主（memory `feedback_skill_describing_tool_output_enables_hallucination`）。

---

## 五、写 spec 的格式与纪律（照抄即可）

- **模板**：参照 `docs/superpowers/specs/2026-06-22-identify-zone-info-persist-spec.md` 骨架：标题 → 状态/日期/性质/关联(blockquote) → 〇给实施agent一句话 → 一根因(逐字节实证 file:line) → 二设计(修法+不改什么) → 三改动清单 → 四测试(TDD红→绿) → 五验收 → 六风险 → milestone建议。
- 已写的 ETHO-8/ETHO-2 spec 就是按这个模板，可直接参照风格。
- **TDD 强制**：每份 spec 的「四测试」要给红→绿用例。
- **prompt 契约测试铁律**（memory）：测 prompt 必须用 importlib 加载 worktree/被测源，否则读主仓旧 prompt 假绿。
- **import 环铁律**（CLAUDE.md）：spec 若涉及改 `subagents/`、`tools/builtins/`、`agents/` 核心（ETHO-1 改 executor/middleware、ETHO-2 改 registry），验收必含裸导入 `app.gateway` + `make_lead_agent`。
- **受保护文件**：prompt.py / registry.py / executor.py 都是 deerflow 定制面，spec 的「风险」段要提示 sync 时 surgical 守护。

---

## 六、milestone 建议
- 「EPM dogfood 图表/交互流水线打磨」track：ETHO-8（A/B 确定性）、ETHO-5（分组全量）、ETHO-7+9（决策点确定性）。
- 「subagent 派遣/生命周期 infra 加固」track：ETHO-2（registry 自洽）、ETHO-1（after_agent seal gate）。
- checkpoint = 「为 dev 站点 E2E 暴露的 10 bug 系统性根因 + 分档修复」。

---

## 七、最重要的教训（memory 已记 `feedback_code_has_fix_not_equal_bug_eliminated_seal_react_floor`）
判 bug 是否已修：**问「现象在当前代码还会不会发生」，不是「有没有 PR」**。曾凭记忆误判 ETHO-1「已根治」被用户当场推翻——修复合了、站点在跑，bug 仍复现，因为 SealGate 有放行口子、auto-seal 有前提。确定性修复（裁剪/排序/枚举）可判已消除；依赖 LLM 决策时机/自由裁量的（seal 终止/反问合并）只能判降级或仍存在。
