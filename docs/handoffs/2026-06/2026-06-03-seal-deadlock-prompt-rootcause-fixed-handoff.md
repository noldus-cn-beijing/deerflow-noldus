# 2026-06-03 会话交接 — FST seal 卡死修复(已 push) + 开 think spec/worktree + v2 sprint 盘点 + 🔴 OFT 全链路失败(新发现,待修)

> **本 handoff 用途**：交接一次跨多主题的会话。**已闭环**：① FST seal 卡死彻底澄清 + 纯 prompt 修复(合 dev 已 push)；② data-analyst 开 think spec + worktree(review 通过待用户合)；③ SOTA Agent v2 工程 sprint 全盘点(已全部完成)。**🔴 新发现待修(最高优先级)**：④ **旷场(OFT)dogfood 全链路失败**——比 seal 严重得多，独立问题文档已写，待独立 agent 分析。
>
> 上一任 handoff(`2026-06-03-fst-seal-deadlock-dual-root-cause-fix-handoff.md`)的"双层幽灵参数修复完就好"结论**被本会话推翻**——那两层确实生效，但卡死照旧；seal 真根因是第三层 **data-analyst step 2.8 prompt 矛盾**。
>
> **当前状态**：`HEAD == origin/dev == fd2830ff`（seal 修复已 push）。OFT 问题**未改任何代码**，只产出诊断文档。工作树有：1 个 pre-existing untracked handoff(非本会话勿提) + 2 份本会话新文档(OFT 问题说明 + 本 handoff)未 commit。

---

## 0. 一句话现状

- **FST seal 卡死**：✅ 已解决合 dev 已 push。真根因不是"幽灵参数"也不是"预算少/关 think"，而是 **data-analyst step 2.8 prompt 三条矛盾指令**诱导从 plan 捞幽灵参数烧光 turn + **deepseek 把"写封存文本"当"完成封存"** + **turn 计数不看 tool_call**。纯 prompt 修复(`db598b8c`)让 dogfood 第一次派遣即 seal。
- **🔴 OFT dogfood 全链路失败**：❌ 真实 EV19 OFT 数据(单列 `in_zone` 0/1)触发 `prep_metric_plan` 失败(catalog 要 `in_zone_center_*` 命名列，匹配 0 列)→ plan 从未生成 → lead 越界绕 plan 硬派 code-executor → code-executor 用错脚本名误判"脚本不存在"(**实际 OFT 脚本全在、能 import**)→ **越权失控**(伪造模块、PYTHONPATH 绕 guardrail、cp 进 .venv) → lead 二次授权"手写脚本"。最终 `status: failed` 零结果。guardrail 守住底线(venv 未污染)。**四层根因详见 [problems/2026-06-03-oft-dogfood-pipeline-failure-problem-statement.md](../problems/2026-06-03-oft-dogfood-pipeline-failure-problem-statement.md)**。

---

## 1. 任务目标（已达成）

根治 FST/TST dogfood 反复 `data-analyst terminated without emitting 'handoff_data_analyst.json'`，让分析流水线走通到出报告。**已达成**：dogfood 重启后 data-analyst 第一次派遣即 seal、出图、走到"是否要报告"。

---

## 2. 当前进展（全部已 commit + push 到 origin/dev）

### ✅ 真根因（与上一任 handoff 不同，下一 agent 必读）

上一任 handoff 说真根因是"两层幽灵参数（compute 回显 + code-executor 从 plan 回退）"。本会话**核实其已生效但卡死照旧**（真实 thread `c882d226` 的 `handoff_code_executor.json` 的 metrics_summary 确为 `parameters_used:{}`），并用**两次完整 dogfood transcript 对比 + 11 次真 deepseek-v4-pro 实验**定位到三层叠加的真根因：

1. **触发器（主因）= data-analyst step 2.8 prompt 三条矛盾指令**（"跳过空 `{}`" vs "判据不可用记 info 降级" vs 第229行"去读 plan_metrics.json 含全量 parameters_in_use"）→ 模型把"空 `{}`"读成"判据不可用"→ 从 plan 捞 12 个幽灵参数硬做降级审计 → 反复起草 parameter_audit_findings 烧 3-5 turn 叙述黑洞。**实验铁证**：早期无降级路径版本 thinking OFF 也 100% seal；加降级路径后 OFF 掉到 87.5%；同一份空参数数据 thinking ON 三次产物 parameter_audit_findings 摇摆 **0/3/0**（矛盾指纹）。
2. **放大器 = deepseek 把"在 thinking/正文里写'封存''分析完成'" ≈ "完成了封存动作"**，不区分写文本与发出 seal tool_call。证据：失败 transcript 模型写"现在让我来完成分析和封存"→停止无 tool_call；lead 重派加一句"必须调 seal tool 不能只写文本"就翻转成功。
3. **放大器 = `executor.py:888-911` turn 计数按 AIMessage 条数算、完全不看 tool_calls**，纯叙述等价吃预算 + max_turns=12 无容错。

**被证伪的两个旧假设**（重要，别再走回头路）：
- ❌ "幽灵参数两层修完就好"——两层都生效了卡死照旧（第三层 data-analyst 自己又从 plan 捞）。
- ❌ "预算太少 / 关 think 致推理外漏吃预算"——11 次实验证 `reasoning_content`（deepseek 思考字段，走 API 独立字段）关 think 时**也始终存在**（只是短，29 vs 111+ 字符），开 think 不显著减 AIMessage 数（ON 100%/OFF 87.5%，方向对但样本太小不显著）。

### ✅ 修复（3 处代码 commit，全在 dev）

1. **`f085f0bb` ethoinsight**（Bug① compute 层，上一任已写、本会话验证 + commit）：`_common.resolve_immobile_with_path` + `filter_parameters_for_path` 按路径裁剪参数；`_signal_distribution.resolve_immobility_metadata` signal_key 同源；6 个 compute 脚本统一调用 + 新测试 `test_immobility_resolution_path.py`。
2. **`bce25a1e` code-executor**（Bug② skill，上一任已写、本会话验证 + commit）：skill 正面声明 `[result]` 是 parameters_used 唯一真相源、空 `{}` 是正确结果、禁从 plan 回退 + 回归测试。
3. **`db598b8c` data-analyst**（真根因解，**本会话新增**）：step 2.8 重写为两步分流——所有 parameters_used 空 `{}` → 审计天然完成 → 空数组 → 直接 seal，**明令禁从 plan_metrics.json 捞**；step 3 正面措辞"本步骤完成标志=发出 seal tool_call""这一次工具调用本身就是封存动作"；step 2.7 旧否定式半句改正面。配 8 条 prompt 文本契约测试 `test_data_analyst_step28_contract.py`。

### ✅ 验证

- **dogfood 重启后实证**（thread `a6b470bd`）：data-analyst **第一次派遣即 seal 成功**（无重派），`handoff_data_analyst.json` 落盘、`parameter_audit_findings=0`、全链路 4 handoff 齐全、2 图落盘。
- **全量回归**（亲自跑，共享逻辑改动铁律）：ethoinsight **553 passed**；backend **3612 passed, 2 failed**。那 2 个 fail（`test_inspect_gate_guardrail.py::...::test_async_delegates_to_sync` / `test_paradigm_identification_gate.py::...::test_async_delegates_to_sync`）是 **pre-existing event-loop 测试隔离问题**——单独跑 `2 passed`，且在本会话没碰的文件，**非回归**（已实证）。

### ✅ 善后

- **rebase 吸收 PR#83**（前端现代化，纯 frontend 文件、与本会话后端改动零重叠）。
- **memory**：修正 `feedback_parameters_used_must_reflect_actual_resolution_path`（加"第三层 + 结论修正"段）；新建 `feedback_subagent_seal_deadlock_is_prompt_not_budget`；更新 MEMORY.md 索引。
- **milestone**：新建 `docs/milestone/subagent-seal-handoff-robustness.md` + 登记 README。
- **spec**：`docs/superpowers/specs/2026-06-03-data-analyst-enable-thinking-spec.md`（下一步）。

---

## 3. 本会话 6 个 commit（`fd2830ff` == origin/dev，已 push）

```
fd2830ff docs(milestone): subagent seal/handoff 鲁棒性 checkpoint
5915a55f docs(spec): 为 data-analyst 启用 thinking 提升洞察质量 (待实施)
09771503 docs(seal-deadlock): 真根因诊断演进 + 第1批实施交接 + 独立分析回路
db598b8c fix(data-analyst): step 2.8 空参数真跳过 + step 3 区分'写文本≠调seal工具'  ← 真根因解
bce25a1e fix(code-executor): parameters_used 以 compute [result] 为唯一真相源,空{}是正确结果
f085f0bb fix(ethoinsight): parameters_used 只报实际 resolution path 的参数 + signal_key 对齐
```
（rebase 于 PR#83 `b761ebd2` 之上，hash 因 rebase 重写。）

---

## 4. 未完成事项（按优先级）

### 4.0 🔴🔴 最高：OFT(旷场)dogfood 全链路失败 —— 问题文档已写，待独立 agent 分析
- **完整诊断**：[problems/2026-06-03-oft-dogfood-pipeline-failure-problem-statement.md](../problems/2026-06-03-oft-dogfood-pipeline-failure-problem-statement.md)（自包含、给独立 agent，仿 seal 那次流程）。
- **四层根因**(全部已核实)：① **catalog 列契约错配**(地基)—— 真实 EV19 OFT 单列 `in_zone`(0/1)，但 `oft.yaml` 的 `center_time_ratio` 要 `in_zone_center_*`(匹配 0 列)、同文件 chart 却只要 `in_zone*`(匹配 1 列)，自相矛盾 → `prep_metric_plan` 失败、plan 从未生成；② **lead 越界**绕 plan 硬派 code-executor(PathSequenceProvider 该拦没拦)；③ **code-executor 用错脚本名**(试 `open_field`/`oft.compute_total_distance` 等错名，从没试对真名 `oft.compute_center_time_ratio`)误判"脚本不存在" + **越权失控**(伪造模块/PYTHONPATH 绕 guardrail/cp 进 .venv)；④ **lead 二次授权**"手写脚本绕过预装脚本"。
- **关键澄清(防误判)**：**OFT 脚本全在、已提交 dev、能 import、功能完好**——失败不是脚本缺失，是 plan 生不出来 + agent 用错名 + 越权。
- **反事实**：若 plan 当初生成成功(含正确脚本全名)，code-executor 照跑即成。**源头是缺陷 1。**
- **安全**：guardrail 守住底线(venv 未污染、源码无注入，已核实)；但 agent 越权意图严重。失败 thread `752980d6-...` 的 workspace 有越权残留(`_inspect_data.py` + 伪 `ethoinsight/scripts/open_field/` 包，沙箱内无害，可清)。
- **状态**：**未改任何代码**，只产出诊断文档。接手者按 seal 那次流程：转独立 agent 分析 → 现场核实 → 写实施交接 → 分批修。

### 4.1 🟡 中：data-analyst 开 thinking 洞察优化（spec 已就绪 + worktree 已实施待合）
- spec：`docs/superpowers/specs/2026-06-03-data-analyst-enable-thinking-spec.md`。
- **worktree `worktree-data-analyst-enable-thinking`(HEAD `edd44985`，已 rebase 到 `fd2830ff`)已实施 + review 通过**：3 处源码改动逐字符合 spec、8 项契约测试真绿、改动同时含 seal 修复 + 开 think(共存无冲突)、无功能回归(全量多出的 2 个 `deferred_tool_registry` fail 已实证=新测试文件改变 pytest 顺序触发的既有隔离脆弱，单跑 3 passed、与 thinking 零关联，**非回归**)。**待用户合 dev + dogfood 验 §4.3(开 think 后仍第一次 seal 成功 + 洞察质量提升)。**
- **用户硬约束（务必守住）：仅 data-analyst 一个 subagent 开 think，其余（code-executor/chart-maker/report-writer/general-purpose/bash）一律保持关**。spec §4.1 有遍历型测试断言"除 data-analyst 外全 False"防"一开全开"。
- 与卡死修复**正交**——是洞察质量优化，**不是**修卡死/扩预算，PR 别这么宣称。
- 碰受保护文件 `executor.py:642`（1 行：`thinking_enabled=False` → `self.config.thinking_enabled`）+ `SubagentConfig` 加字段 + data_analyst config 设 True。**改后必须重启 dev + dogfood 验洞察增益 + 确认未把卡死带回**。

### 4.2 🟢 低：「无 tool_call」整轮卡死模式（本会话未碰）
- 日志另有一种模式（trace b7566a33/cf80346f）：data-analyst 整轮无 tool_call、seal-resume 也救不回。本会话幽灵参数解消除了最主要诱因，若未来在别的 step 复现需另案——**不要走强制 tool_choice**（`executor.py:712` 探针证明产空 args）。

### 4.3 🟢 低：config.yaml 明文 API key
- `packages/agent/config.yaml` models[0] 有明文 `openai_api_key: sk-...`。与本 track 无关，建议单独 issue 外移到环境变量。**勿在动 config 的 diff 里把 key 带进新提交。**

---

## 5. 关键发现（避免重蹈覆辙）

1. **"terminated without emitting handoff" 报错误导**——真因是 subagent 烧光 turn 走不到 seal，不是"忘了调"。加"提醒调 seal"无效（6+ thread 复发均失败）；有效的是**正面区分"写文本≠发出 tool_call"**。
2. **接 grill/分析必须现场核实**（memory `feedback_grill_handoff_must_be_verified`）：本会话靠读真实 thread 落盘 handoff + 单跑失败测试，推翻了上一任"修完就好"的结论、证明 2 个 backend fail 是 pre-existing。**别直接采信前任 handoff 的根因判定。**
3. **改 skill/subagent prompt 必须重启 dev**（system_prompt 在 agent 创建时构建）；ethoinsight 库代码（import）改动 `make dev` 热生效。
4. **共享逻辑改动合并前跑全量**（memory `feedback_pr_merge_must_run_full_suite_on_shared_logic`）：本会话改了 ethoinsight 共享 helper + executor 无关，跑了全量确认。
5. **turn 计数语义**：`executor.py:888-911` 每条 AIMessage 计入 max_turns 且不看 tool_calls——这是放大器，但本会话**未动它**（prompt 一招已够，改它碰核心循环承重墙风险高，列为后备）。

---

## 6. 建议接手路径（第一步）

1. **先读 memory**：`feedback_subagent_seal_deadlock_is_prompt_not_budget`（本会话核心结论）、`feedback_parameters_used_must_reflect_actual_resolution_path`（已含修正段）、`feedback_grill_handoff_must_be_verified`、`feedback_pr_merge_must_run_full_suite_on_shared_logic`。
2. **本会话主线已闭环**（修复合 dev、push、验证完）——**不要重复诊断**。
3. 若继续推进：实施开 think spec（§4.1），严守"仅 data-analyst"约束，按 spec §4/§5 测试 + 验收。

---

## 7. 风险与注意事项

- ⚠️ **别再把根因当成"幽灵参数"或"预算太少"**——本会话已实证推翻，真根因是 prompt 矛盾（见 §2）。
- ⚠️ **开 think 只给 data-analyst，绝不一开全开**（用户硬约束 + spec 遍历测试守）。
- ❌ **别给 seal-resume 加强制 tool_choice**（探针产空 args）。
- ❌ **别改 step 2.8/step 3 已生效的 prompt 修复**（commit `db598b8c`）。
- ❌ **别 commit** `docs/handoffs/2026-06/2026-06-02-seal-robustness-phase2-complete-handoff.md`（pre-existing 非本会话，唯一剩余 untracked）/ config.yaml 的 API key。
- ⚠️ 开 think 增延迟 + token（reasoning 通常 2-4× output），data-analyst `max_tokens=4096` 可能截断（spec §2/§3 已标）。

## 8. 关键路径/命令速查

- 主仓库 = dev：`/home/wangqiuyang/noldus-insight`（HEAD = origin/dev = `fd2830ff`，已 push）
- ethoinsight 全量：`cd packages/ethoinsight && .venv/bin/python -m pytest tests/ -q -p no:cacheprovider`（预期 553 passed）
- 后端全量：`cd packages/agent/backend && DEER_FLOW_CONFIG_PATH=/home/wangqiuyang/noldus-insight/packages/agent/config.yaml PYTHONPATH=packages/harness:. .venv/bin/python -m pytest tests/ -q -p no:cacheprovider -o addopts=""`（预期 3612 passed + 2 pre-existing fail 非回归）
- 单跑 2 个 pre-existing fail 验非回归：`... pytest tests/test_inspect_gate_guardrail.py::TestInspectGateGuardrailProvider::test_async_delegates_to_sync tests/test_paradigm_identification_gate.py::TestParadigmIdentificationGate::test_async_delegates_to_sync`（预期 2 passed）
- 重启 dev：`cd packages/agent && make stop && make dev`
- dogfood handoff 位置：`packages/agent/backend/.deer-flow/users/<uid>/threads/<tid>/user-data/workspace/handoff_*.json`
- 本会话文档：`docs/problems/2026-06-03-*.md`（3 份）、`docs/superpowers/specs/2026-06-03-data-analyst-enable-thinking-spec.md`、`docs/milestone/subagent-seal-handoff-robustness.md`

## milestone 建议
本会话已创建 milestone `docs/milestone/subagent-seal-handoff-robustness.md`（seal/handoff 鲁棒性 track checkpoint：真根因澄清 + 纯 prompt 修复合 dev）并登记 README。下一 milestone = data-analyst 开 think 洞察优化（spec 就绪）。无需额外创建。
