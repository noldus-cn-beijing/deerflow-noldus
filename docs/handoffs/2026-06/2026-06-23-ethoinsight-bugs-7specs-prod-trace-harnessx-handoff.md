# 交接：ETHOINSIGHT_BUGS 10 条 E2E bug —— 7 条修复 spec 全部写完（含 prod trace 取证 + ETHO-2 误诊纠正 + HarnessX 工程化）

> **会话日期**：2026-06-23
> **来源**：用户在 dev.ethoinsight.com 跑 EPM E2E 发现的 10 个 bug，原始清单 `~/ETHOINSIGHT_BUGS.md`；根因母文档 `docs/problems/2026-06-22-ethoinsight-bugs-e2e-root-cause.md`。
> **本会话做了什么**：① 接手上一会话（写了 ETHO-8/ETHO-2 两 spec）；② **dump 4 个生产 thread 取证**（最高证据等级），坐实 ETHO-1 真因、**推翻并纠正 ETHO-2 误诊**；③ 读 **HarnessX 报告**（arXiv 2606.14249）提炼工程化方法论；④ 用「prod trace 根因 + HarnessX 最佳实践 + DeerFlow 原生」三重输入，把 **7 条 bug 的 spec 全部写成生产级**。
> **用户两条硬指示（贯穿）**：① 写 spec 不碰代码；② **DeerFlow-first**——HarnessX 只借工程化思想，运行时机制一律用 DeerFlow/LangGraph 原生（middleware hook / GuardrailProvider / path_registry / 复用现有纯函数），不引入 HarnessX 的 processor/AEGIS 机制。

---

## ⚠️ 接手第一步：核实文件状态

```bash
cd /home/wangqiuyang/noldus-insight
git log --oneline -1 dev          # 应含 PR#175 f14d3dc0（本地 dev 已最新）
ls -la docs/superpowers/specs/2026-06-23-*.md   # 应有 6 份（本会话）+ ETHO-8 那份
```

**7 个 bug 的 spec（全部 untracked 新文件，未提交）**：
| Bug | spec 文件 | 优先级 | 状态 |
|---|---|---|---|
| ETHO-1 | `2026-06-23-etho1-seal-gate-after-agent-reconstructable-vs-cognitive-spec.md` | 🔴高 | ✅ 生产级 |
| ETHO-2 | `2026-06-23-lead-dispatches-illegal-bash-subagent-registry-consistency-spec.md` | 🔴→纠正为体验 | ✅ **已纠误诊** |
| ETHO-3 | `2026-06-23-etho3-code-executor-triage-metadata-sidecar-spec.md` | ◐残留低 | ✅ 生产级 |
| ETHO-5 | `2026-06-23-etho5-grouping-full-scan-no-single-file-extrapolation-spec.md` | 🟡中 | ✅ 轻量 |
| ETHO-7 | `2026-06-23-etho7-intent-determinism-decision-point-drift-spec.md` | 🔴病根 | ✅ 生产级 |
| ETHO-8 | `2026-06-23-template-candidate-ab-order-deterministic-spec.md` | 🟡中 | ✅（上一会话，已核实行号准） |
| ETHO-9 | `2026-06-23-etho9-ask-clarification-requires-options-path-registry-spec.md` | 🟢低 | ✅ 生产级 |

**不写 spec 的 3 条**：ETHO-4（设计裁决，回用户对齐）、ETHO-6（#168 已消除）、ETHO-10（chart-maker 并行 bash loop-detection 误伤，低优先，可后续）。

---

## 一、最重要：ETHO-2 误诊纠正（上一会话 spec 根因错了，已改写）

**上一会话的 ETHO-2 spec 根因诊断错误，本会话已推翻并改写。** 这是接手核实的最大收获，务必读懂：

- **原 spec 说**：registry「校验集合 `_SubagentLiteral` ⊋ 可派集合 `BUILTIN_SUBAGENTS`」不自洽，bash 残留在校验枚举→放行又 runtime 失败，需加自洽 filter（修法 A）。用户当时基于这个描述「拍板了 registry filter」。
- **三重实证推翻**（mock executor 破环后实跑）：
  1. `BUILTIN_SUBAGENTS` 不含 bash（`builtins/__init__.py:23-29` 只 5 个 ethoinsight subagent）。
  2. `config.yaml` `subagents.custom_agents = {}` 空 → `get_subagent_names` 两来源都不含 bash。
  3. `get_available_subagent_names()` 实跑 = 5 个，`bash in available? False`。
- **结论**：`task(subagent_type='bash')` **当前已在 Pydantic schema 层被拒**（报错 `Input should be 'chart-maker',...` 正是 schema 层 enum 错——两套报错文案对照见 spec §1.3）。**修法 A 是 no-op（修不存在的 bug）**。真因**只剩 prompt**（lead 不知 `identify_ev19_template` 已批量扫全部文件）。
- **改写后**：砍修法 A（降级为防御性回归测试守"校验集合恒=可派集合"不变式），只留修法 B（prompt 讲清批量扫描走 identify、没有 bash 这个 subagent 类型）。
- **memory**：`feedback_etho2_spec_misdiagnosed_infra_bug_already_schema_rejected`。
- **接手须知**：交付 ETHO-2 给实施 agent 时，若 reviewer 质疑"为何不加 filter"，答案在 spec §六.1。**别被旧 handoff 的"用户拍板 registry filter"误导**——那是基于错误根因的拍板。

---

## 二、生产 thread 取证（4 thread，最高证据等级）

按 `docs/sop/prod-thread-triage-sop.md` dump 4 个 prod thread（本机 `~/<tid>.json`，远端临时文件已清）：
- `3a41e483`（59 msgs，EPM）/ `83bfde49`（35）/ `e6ea7946`（57，zero_maze）/ `47e8155a`（68）

**坐实的根因**：

1. **ETHO-1（seal 漏调）**：`terminated without emitting`/`forgot to call seal` 出现在 **4 中 3**（高频）。漏调发生在 **data-analyst（e6ea7946 #38）+ report-writer（3a41e483 #49）两类**。重试 7 步即成功 = **纯收尾漏 call，非 thinking 超时**。runs 全 `success`（lead 重试有效，已从"致命无限重派"降级成"偶发重试一轮"）。
   - memory：`feedback_etho1_prod_trace_seal_miss_both_dataanalyst_and_reportwriter`。
2. **ETHO-2**：4 thread `task(subagent_type='bash')` **调用 0 次**——现象未在这批复现，佐证当前 lead 正常不碰 bash。
3. **ETHO-9（options 缺失）活体证据**：`e6ea7946 #5` 范式确认反问 `options=无`（纯文本框），同 thread 出图/出报告反问都 `options=有(2)`。
4. **ETHO-7（决策点漂移）活体证据**：4 thread 反问决策点数量/形态各不同；identify status 四态（unknown/ambiguous/ok/error）lead 据此走不同分支。

**SOP 用法**：`ssh root@39.105.231.16`（已配免密 + settings.local.json 放行，但**会话级 auto 模式分类器要用户显式确认这台 prod host**）→ `docker exec deer-flow-gateway .venv/bin/python scripts/dump_thread.py $TID`。分析脚本走 heredoc 写本机再提结构化信号，别把全文灌进 context。

---

## 三、HarnessX 报告的工程化应用（用户要求"写更生产级"）

报告：`/home/wangqiuyang/resources/parsed/2606.14249v1/vlm/2606.14249v1.md`（**HarnessX**，Darwin Agent Team，**非 MiMo**）。memory：`reference_harnessx_report_and_etho_spec_application` + `feedback_harnessx_ideas_on_deerflow_not_harnessx_mechanisms`。

**6 份 spec 统一套用的生产级元素**（从报告提炼）：
- **change manifest**：每处改动写「预期改善 + 可能回归 + 对应测试」（Evolver）。
- **确定性 gate 序列验收**：manifest 完整→smoke→红绿→回归(seesaw)→import 环→scope，第一个失败即停（Critic+Gate）。
- **三大病理自检**（§六固定段）：reward hacking / catastrophic forgetting / under-exploration（operational mirror）。
- **trace richness**：ETHO-1 的 `sealed_by` 五分类 + 触发率可观测（§7.2）。
- **Telecom 累加 reminder 禁令**（§6.6）：禁止再加 seal/options/反问合并的提醒 prompt（我们 seal 已改 4 次 prompt 打地鼠）——只动结构。
- **smoke test**：processor 实例化跑通（Evolver 要求）。

**DeerFlow-first 映射**（每份 spec 都有「框架契合声明」段）：HarnessX processor→DeerFlow LangChain middleware；8 hook→middleware hook；D7 控制安全→SealGate/GuardrailProvider；Digester→sidecar；co-evolution→训练飞轮+SkillOpt。**只借思想，机制全 DeerFlow 原生。**

---

## 四、各 spec 核心修法（实施 agent 直接看 spec，这里只give 一句话 + 关键约束）

### ETHO-1（升级 SealGate，分两类）
- 给 `SealGateMiddleware` 加 `after_agent` hook（**不能 jump**，纯副作用）。
- **report-writer/chart-maker**（产物可重建）→ after_agent 调 `_attempt_auto_seal_from_artifacts` 结构堵死 + 消除 `Task failed` 中间态。
- **data-analyst**（认知产物）→ after_agent **不伪造**，只靠 L1 逼补调+L2 补轮+L4 lead 重试，**降触发率不归零（诚实）**。
- **不改 `_MAX_REMINDERS=2`**（提高 cap 依赖 thinking-overload 线先解决——**用户已让别的 agent 处理 thinking 超时**，两线别在本 spec 冲突）。
- 关键约束：`_attempt_auto_seal_from_artifacts` 惰性 import（import 环铁律）。

### ETHO-2（纯 prompt，见 §一）
- 砍 no-op 修法 A，留 prompt 修法 B + 防御性不变式回归测试。

### ETHO-3 残留（复用 #169 sidecar）
- code-executor 分诊段指向已建 `_metric_metadata.json`（#169），不读 133K plan_metrics。
- **诚实标注：未在 4 prod thread 复现**（分诊失败场景未触发），低优先。
- 改 **system_prompt**（最高权威源）非只 SKILL.md（memory `feedback_subagent_system_prompt_higher_authority_than_skill`）。

### ETHO-5（纯 prompt 轻量）
- `prompt.py:493`「优先用 per_file_grouping」→「**必须**基于全量 + 禁止单文件外推」（正面措辞）。
- **刻意不上结构门**——工具已全量正确，缺的只是 prompt 硬度（不是 under-exploration，是结构已在）。

### ETHO-7（病根，分两子改动可独立合）
- 子改动 A（低险）：intent 分类确定化——E2E_FULL（跳出图反问）需对话有明确出图意向，否则 deny 要求 ASKVIZ（默认"不确定就问"）。
- 子改动 B（中险）：扩 `IntentPostStepAskGateProvider` 检测 ask 步骤乱序/合并。
- **诚实：降漂移非消除漂移**（HarnessX §7.3 符号空间无收敛保证）。
- ⚠️ 子改动 B 和 ETHO-9 同改 `IntentPostStepAskGateProvider`——**建议 ETHO-9 先合，ETHO-7-B 基于其上**。

### ETHO-8（上一会话，已核实）
- `identify_ev19_template_tool.py:600` 候选列表加确定性排序（推荐项恒为 A，其余 template_id 字典序）。行号已核实准。

### ETHO-9（Step 加字段 + 新 guardrail）
- `path_registry.Step` 加 `requires_options: bool=False`；PATHS 给 viz/report/four_choice 标 True。
- 新 `AskClarificationOptionsProvider` 拦 ask_clarification，requires_options 的反问漏 options → deny。
- **边界诚实**：范式确认反问（intent 分类前）不在 PATHS ask step 内，本 spec 不覆盖（归 ETHO-7/prompt 侧）。

---

## 五、关键代码锚点（已全部核实属实）

- `seal_gate_middleware.py`：`_MAX_REMINDERS=2`(L58)、`after_model`+`can_jump_to`(L120)、`_REQUIRES_SEAL`(L56 = data-analyst/chart-maker/report-writer)。
- `executor.py`：`_validate_handoff_emitted`(L176)、`_attempt_auto_seal_from_artifacts`(L301，仅 rw/cm/code-executor 可重建，data-analyst 跳过)、seal-resume→auto-seal→FAILED 链(L1400-1444)。
- `after_agent` hook：5 个 middleware 用（todo/memory/training_data/loop_detection/token_budget）**无一标 can_jump_to** → after_agent 不能 jump（memory `feedback_etho1_after_agent_cannot_jump_seal_gate_upgrade_boundary`）。
- `path_registry.py`：`PATHS`(L45 8 intent 有序 step)、`Step` frozen dataclass(L18)、`ASK_GATE_MAP`(L100)、CI 哨兵 `test_path_registry_ssot.py`。
- `intent_post_step_ask_gate_provider.py`：`evaluate`(L129) **只拦 task()**、`_VIZ_DENY_MESSAGE`(L105 PR#175 改的文案)。
- `intent_classification_provider.py`：强制 lead 声明 `[intent]` 行。
- `clarification_tool.py:17`：`options: list[str]|None=None` 可选。
- `code_executor.py`：分诊段「用 read_file/ls 查证失败细节」；`report_writer.py:181`：已指向 `_metric_metadata.json`。
- registry：`get_available_subagent_names`(L150)、`task_tool.py` `_SubagentLiteral`(L350)、兜底文案(L409)。

---

## 六、写 spec 的格式与纪律（6 份已统一，续写照此）

- **模板骨架**：标题→状态/日期/性质/关联(blockquote)→〇给实施agent一句话→〇·五框架契合声明→一根因(逐字节+prod实证)→二设计→三改动清单(manifest)→四测试(TDD红→绿+smoke)→五验收(确定性gate序列)→六风险(三大病理自检)→milestone。
- **TDD 强制** + **prompt 契约测试用 importlib 读被测源**（防主仓假绿）。
- **import 环铁律**：改 subagents/tools/agents/guardrails 核心，验收含裸导入 `app.gateway`+`make_lead_agent`。
- **三指令源**：改 subagent 行为先 grep `builtins/<name>.py` 的 system_prompt（最高权威）。
- **受保护文件 sync surgical**：prompt.py/registry.py/executor.py/seal_gate_middleware.py/path_registry.py/guardrail provider 全是 deerflow 定制面。

---

## 七、本会话新增/更新的 memory（接手先读）

- `feedback_etho2_spec_misdiagnosed_infra_bug_already_schema_rejected` — ETHO-2 误诊纠正（最重要）。
- `feedback_etho1_prod_trace_seal_miss_both_dataanalyst_and_reportwriter` — prod 实证 seal 漏调两类。
- `feedback_etho1_after_agent_cannot_jump_seal_gate_upgrade_boundary` — after_agent 不能 jump 的物理边界。
- `reference_harnessx_report_and_etho_spec_application` — HarnessX 方法论 + 对 4 spec 应用 + 三大病理自检清单。
- `feedback_harnessx_ideas_on_deerflow_not_harnessx_mechanisms` — DeerFlow-first 硬约束。

---

## 八、待办（交接给下一 agent / 用户）

1. **用户对齐 ETHO-4**：identify 故意不读 experiment 字段是**设计**（守 HITL 反问铁律），不是 bug。需回用户确认这是预期行为。
2. **用户对齐 ETHO-2 误诊**：原"registry filter 拍板"基于错误根因，已改 prompt-only。需用户知悉方向变更。
3. **ETHO-10 未写 spec**（低优先）：chart-maker 并行 bash 被 loop-detection 频率熔断 partial strip。同类病 memory `feedback_loop_detection_tool_semantics_floor_and_partial_strip`。需要时补。
4. **实施顺序建议**（按性价比 + 依赖）：ETHO-8（确定性极小）→ ETHO-9（options，ETHO-7-B 的前置）→ ETHO-5（prompt）→ ETHO-2（prompt）→ ETHO-1（infra）→ ETHO-7（病根，A 先 B 后）→ ETHO-3 残留（收尾）。
5. **7 份 spec 入库**：当前全 untracked。需 commit 进 dev（spec 是文档，可直接进；或随各自实施 PR 带入）。
6. **thinking-overload 线协调**：用户已让别的 agent 处理 thinking 超时（seal 第 4 类根因）。ETHO-1 spec 写明了「不提高 cap」依赖那条线——两线实施时需对齐，别冲突。

---

## milestone 建议
- **「subagent 派遣/生命周期 infra 加固」track**：ETHO-1（SealGate after_agent 分两类）、ETHO-2（registry/prompt）、ETHO-3 残留（分诊 metadata）。
- **「EPM dogfood 交互流水线确定性」track**：ETHO-5（分组全量）、ETHO-7（决策点病根）、ETHO-8（A/B 序）、ETHO-9（options）。
- **checkpoint**：「dev 站点 E2E 暴露的 10 bug → 系统性 prod-trace 取证 + 7 条生产级修复 spec（DeerFlow 原生 + HarnessX 工程化）」。
- **方法论资产**：HarnessX 三大病理自检清单可考虑进 CLAUDE.md/sync SOP（改 harness 前自检 reward hacking/forgetting/under-exploration）——这是本会话超出 bug 修复的长期价值。
