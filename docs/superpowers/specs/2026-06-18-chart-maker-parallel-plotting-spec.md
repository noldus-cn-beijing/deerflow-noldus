# Spec：chart-maker 并行绘图 + 删除禁用 bash 的 bundle 死指引

> 状态：实施 spec，可直接交付 agent 执行
> 日期：2026-06-18
> 性质：性能 + 契约修正。chart-maker 当前逐张串行跑绘图脚本（N 张图 = N 个 turn），而 guardrail **本就放行** code-executor 同款的 `bash -c "python -m ... & ... & wait"` 并行形态——chart-maker SKILL 没教它这么做。另：chart-maker / report-writer SKILL 里仍有一条「`bash cat ... > bundle.txt`」加速指引，但这些 subagent **禁用了 `bash`**（且 `cat ... >` 含 guardrail 禁字符 `>`），从来跑不通、还在 dogfood 里浪费 turn。
> 关联：
> - 正例参照：code-executor 并行 bash 实现 `packages/agent/skills/custom/ethoinsight-code/SKILL.md`「并行执行规则」段（L54-81）
> - guardrail（已支持并行，无需改）：`packages/agent/backend/packages/harness/deerflow/guardrails/script_invocation_only_provider.py`（`_PARALLEL_BASH_PATTERN` + `_validate_parallel_bash_content`）
> - 契约断裂同源教训：memory `feedback_code_executor_skill_writefile_contradicts_seal_tool`（SKILL 教用工具没有的能力）
> - 与 handoff 瘦身 spec 互补：`docs/superpowers/specs/2026-06-18-code-executor-handoff-slimming-spec.md`（那份也删 data-analyst 的 bundle 指引；本份删 chart-maker/report-writer 的绘图侧 bundle 指引）

---

## 〇、给实施 agent 的一句话

两件事：① 教 chart-maker 用 code-executor 同款的 `bash -c "python -m ...plotA & python -m ...plotB & wait"` 把**互不依赖的绘图脚本**一次并行跑完（N turn → 1 turn）；② 删掉 chart-maker / report-writer SKILL 里那条用 `bash cat ... > bundle` 拼上下文的指引——这俩 subagent 禁了 bash，该指引从来执行不了，只会让它们撞 guardrail 浪费 turn。

**核心约束**：并行只对「输入不同、输出不同、互不依赖」的绘图脚本；guardrail 已限制并行内层只能是 `python -m ethoinsight.scripts.*`（+ wait/echo），照此形态即可，无需改 guardrail。

---

## 一、根因（实证 + 能力核对）

### 1.1 chart-maker 串行跑图（性能问题，非故障）

dogfood thread `3bcbee10`：8 张图，chart-maker 一条一条 `python -m ethoinsight.scripts.epm.plot_... ` 跑，trace 显示「第 1 张成功 ✅。继续第 2 张...第 5 张成功 ✅。继续第 6 张」——**8 张图 = 8 个串行 turn**。

但 guardrail [`script_invocation_only_provider.py`](../../../packages/agent/backend/packages/harness/deerflow/guardrails/script_invocation_only_provider.py) **本来就放行并行**：

```python
_PARALLEL_BASH_PATTERN = re.compile(
    r"^\s*(?:cd\s+\S+\s*&&\s*)?bash\s+-c\s+([\"'])(.+?)\1\s*$", re.DOTALL,
)
# _validate_parallel_bash_content: 内层每段必须是 python -m ethoinsight.scripts.* 或 wait/echo
```

code-executor 早就用这个形态（[`ethoinsight-code/SKILL.md:64-72`](../../../packages/agent/skills/custom/ethoinsight-code/SKILL.md)）：

```bash
bash -c "
python -m ethoinsight.scripts.oft.compute_center_distance --input s1.xlsx --output cdist_s1.json --parameters-json '{...}' &
python -m ethoinsight.scripts.oft.compute_center_distance --input s2.xlsx --output cdist_s2.json --parameters-json '{...}' &
wait
"
```

**chart-maker 的绘图脚本满足同样的并行前提**（每张图输入不同 xlsx / 输出不同 png / 互不依赖），却没有这条指引。这是 SKILL 缺失，不是 guardrail 限制。

### 1.2 chart-maker / report-writer 的 bash bundle 死指引（契约断裂）

chart-maker（[`chart_maker.py:36`](../../../packages/agent/backend/packages/harness/deerflow/subagents/builtins/chart_maker.py)）和 report-writer（[`report_writer.py:210`](../../../packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py)）SKILL 都有：

```bash
bash cat /mnt/user-data/workspace/handoff_code_executor.json ... > ..._context_bundle.txt
```

但：
- chart-maker disallowed_tools 含 `bash`？**核查**：chart-maker 是 `_BASH_GATED_AGENTS`（受 guardrail 限制），`cat ... >` 不是 `python -m ethoinsight.scripts.*`、`>` 是 `_DANGEROUS_META` 禁字符 → **guardrail 直接拦**（dogfood trace 实证：「The guardrail blocked the bash -c approach with cat ... > ...」）。
- report-writer disallowed_tools **明确含 `bash`**（report_writer.py:302）→ 它连 bash 工具都没有，该指引更是空指令。

**结论**：两处 bundle 指引都跑不通，dogfood 里 chart-maker 已经因此撞墙浪费 turn。删除。

---

## 二、设计

### 2.1 chart-maker 绘图并行：照搬 code-executor 形态

chart-maker 拿到 plan_charts.json 的 charts[] 后，把**互不依赖**的绘图命令用一条 `bash -c "...& ...& wait"` 并行发出。判据与 code-executor 一致：
1. 脚本间无依赖（不同 chart_id / 不同 subject）。
2. 输出文件不同（每张图不同 png）。
3. 输入不同（不同 xlsx / inputs.json）。

guardrail 内层约束（不可改、照办）：每段必须是 `python -m ethoinsight.scripts.*`，以 `&` 或 `;` 分隔，结尾 `wait`/`echo`，**不得含 `| > < $ \`` 等元字符**。

### 2.2 失败排查仍按串行语义（wait 后逐个查 png）

并行后无法逐条看 stderr，按 code-executor 既有规则（SKILL.md:79）：`wait` 后检查每个预期 png 是否存在，缺失的记 `failed_charts[]`。**保留**「某脚本连续失败 ≥2 次跳过」的既有 recovery 语义——但改为「并行批跑后，对缺失的 png 单独重试一次」，避免一张图失败拖累整批。

### 2.3 删 bundle 指引，改为逐个 read_file

chart-maker / report-writer 读 handoff 上下文：直接逐个 `read_file`（它们要读的 handoff 已经/即将被瘦身 spec 压到 50K 内，单次可读全）。删除 `bash cat ... > bundle` 段。

---

## 三、改动清单

### 3.1 `chart_maker.py` SKILL —— 加并行绘图指引 + 删 bundle

- **删除** L35-40 的 `bash cat ... > cm_context_bundle.txt` 段，改为「逐个 read_file 读 handoff_code_executor.json / plan_metrics.json；本 subagent 受 guardrail 限制，bash 仅可跑 `python -m ethoinsight.scripts.*` 与 ls/cp/mv/mkdir」。
- **在执行绘图段（chart_maker.py 现「逐一执行绘图脚本」处）改为并行指引**，照搬 code-executor 措辞：
  ```
  ## 并行绘图（E2E 加速）
  charts[] 中互不依赖的绘图脚本（不同 subject/不同图、输出不同 png）用一条 bash 并行：
  bash -c "
  python -m <entry0.script> <entry0.args 拼接> &
  python -m <entry1.script> <entry1.args 拼接> &
  ... &
  wait
  "
  wait 后逐个检查 entry.output 的 png 是否存在；缺失的单独重试一次，仍失败记 failed_charts[]。
  约束：每段必须是 python -m ethoinsight.scripts.*；不得用 | > < $ 反引号（guardrail 拦截）。
  ```
- 保留 `<bash_constraints>` 但更新「绘图脚本」条目，明确「可用一条 bash -c 并行多张」。

### 3.2 `report_writer.py` SKILL —— 删 bundle 指引

删除 L209-214 的 `bash cat ... > rw_context_bundle.txt` 段（report-writer 无 bash 工具），改为逐个 read_file。

> 注意：handoff 瘦身 spec（2026-06-18）也会动 report_writer 这段。**两份 spec 若并行实施，约定本份只删 bundle 段、不碰 outlier 读取**；若瘦身 spec 先合，本份对 report_writer 的改动可能已完成，核对后跳过即可（不重复删）。

### 3.3 （可选核查）guardrail 不需要改

`script_invocation_only_provider.py` 的 `_PARALLEL_BASH_PATTERN` 已支持并行绘图——**确认无需改**。仅在测试中加一条断言锁定该能力（防回归）。

---

## 四、测试（TDD）

### 4.1 guardrail 放行并行绘图（锁定既有能力）

`packages/agent/backend/tests/test_script_invocation_guardrail.py`（或新建）：
1. **`test_parallel_plot_scripts_allowed`**：chart-maker agent_id + `bash -c "python -m ethoinsight.scripts.epm.plot_box_open_arm --output a.png & python -m ethoinsight.scripts._common.plot_trajectory --output b.png & wait"` → allow=True。
2. **`test_parallel_with_redirect_denied`**：含 `>` 的 `bash -c "cat x > y"` → allow=False（坐实 bundle 指引为何跑不通）。

### 4.2 SKILL 契约静态守护

3. **`test_chart_report_skills_no_bash_bundle`**：对 chart_maker / report_writer CONFIG，断言 system_prompt **不含** `cm_context_bundle` / `rw_context_bundle` / `bash cat /mnt/user-data/workspace/handoff`。这是与瘦身 spec §4.4 同类的契约守护（可合并为一个测试覆盖三个 subagent）。
4. **`test_chart_maker_skill_has_parallel_plot_guidance`**：断言 chart_maker system_prompt 含并行绘图形态（`bash -c` + `& ` + `wait`），确保指引落地。

### 4.3 全量回归

改 subagent SKILL（`subagents/builtins/*.py`）属核心，按 CLAUDE.md 闭环铁律裸导入两入口：
```bash
cd packages/agent/backend
PYTHONPATH=. python -c "import app.gateway"
PYTHONPATH=. python -c "from deerflow.agents import make_lead_agent"
```
+ `make test` 全量（memory `feedback_pr_merge_must_run_full_suite_on_shared_logic`），扣除已知污染（`feedback_known_full_suite_test_pollution_4_tests`）。

---

## 五、验收标准

1. dogfood EPM 复跑：chart-maker 用 1-2 个并行 bash 跑完全部图（而非 N 个串行 turn），turn 数显著下降。
2. chart-maker / report-writer 不再尝试 `bash cat ... > bundle`，不再撞 guardrail。
3. 某张图失败时只该图记 failed_charts，不拖累整批。
4. guardrail 并行能力有回归测试锁定。

---

## 六、风险与注意事项

1. **并行内层禁元字符**：guardrail 拦 `| > < $ 反引号`。entry.args 里若含这些（理论上不该有）会整批被拦——SKILL 要提示「args 照抄 plan_charts.json，不要自己加重定向」。
2. **失败可观测性下降**：并行后 stderr 混在一起。务必落实「wait 后逐 png 查存在性」的排查规则，否则失败会被并行掩盖（memory `feedback_fallback_trigger_rate_must_be_observable_acceptance_criterion` 精神：别让故障变哑）。
3. **与瘦身 spec 的 report_writer 改动协调**：见 §3.2，避免两份 spec 重复删同一段或互相冲突。实施前先 `git log`/`git diff` 看 report_writer.py 当前状态。
4. **本 spec 不修 Bug 1（plot 脚本列对齐）**：并行不解决「plot 脚本拿不到 zone 参数」——那是 `2026-06-18-plot-scripts-zone-param-alignment-spec.md` 的事。两份独立，但建议先合列对齐 spec（否则并行跑的图仍可能整批失败于同一个列对齐缺陷）。
