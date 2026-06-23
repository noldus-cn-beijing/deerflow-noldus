# Spec：Lead 派遣非法 `bash` 子代理 —— 真因是 prompt（lead 不知 identify 已批量扫），registry 已正确在 schema 层拒绝

> 状态：实施 spec，可直接交付 agent 执行
> 日期：2026-06-23（**2026-06-23 二次修订：三重代码实证 + 4 个生产 thread trace 推翻原 infra 根因**）
> 性质：**纯 prompt / 体验**。Lead 想批量扫 28 个文件的分组，尝试 `task(subagent_type='bash')` → 被拒（报 `Input should be 'chart-maker', 'code-executor', 'data-analyst', 'report-writer' or 'knowledge-assistant'`）→ 退化成逐个 `inspect_uploaded_file`。
> **⚠️ 原 spec 的 infra 根因诊断错误，本次修订推翻**：原 spec 称"校验集合 `_SubagentLiteral` ⊋ 可派集合 `BUILTIN_SUBAGENTS` 不自洽，bash 残留在校验枚举、放行又 runtime 失败，需加自洽 filter（修法 A）"。**三重实证证明 bash 从来进不了校验集合，`task(subagent_type='bash')` 当前已在 Pydantic schema 层被正确拒绝，修法 A 是 no-op**（详见 §一）。真因**只剩 prompt 一层**：lead 不知道 `identify_ev19_template` 已经批量扫了全部上传文件，才想自创 bash 去批量扫。
> 关联：
> - 来源：`~/ETHOINSIGHT_BUGS.md` ETHO-2（高）；根因对照 `docs/problems/2026-06-22-ethoinsight-bugs-e2e-root-cause.md`
> - 实证记录：memory `feedback_etho2_spec_misdiagnosed_infra_bug_already_schema_rejected`、`feedback_etho1_prod_trace_seal_miss_both_dataanalyst_and_reportwriter`（4 prod thread 中 `task(subagent_type='bash')` 调用 0 次）
> - 受保护文件：`agents/lead_agent/prompt.py` 是 deerflow 定制面，sync 时 surgical 守护（见 CLAUDE.md 同步核心规则）
> - 既有约束：runtime 兜底报错文案已符合 memory `feedback_deny_messages_must_direct`（含「请改用…」指引），本 spec 保留它作第二道防线

---

## 〇·五、框架契合声明（DeerFlow-first）

本 spec 运行时改动**零代码逻辑改动**（registry 只加注释、prompt 加指引），天然 DeerFlow 原生。引用 HarnessX 仅在「LLM 提议+确定性门定生死」思想层面——本 spec 体现为：schema 层（Pydantic `_SubagentLiteral`）是确定性门（已正确拒 bash），prompt 只负责消除 lead 想派 bash 的动机。**不引入任何 HarnessX 机制**（memory `feedback_harnessx_ideas_on_deerflow_not_harnessx_mechanisms`）。

---

## 〇、给实施 agent 的一句话

**先跑这段坐实现状，再动手**（mock executor 破环，避开已知导入环）：

```bash
cd packages/agent/backend && source .venv/bin/activate
PYTHONPATH=. python -c "
import sys; from unittest.mock import MagicMock
sys.modules['deerflow.subagents.executor'] = MagicMock()
from deerflow.subagents.registry import get_available_subagent_names
from deerflow.subagents.builtins import BUILTIN_SUBAGENTS
print('BUILTIN keys:', list(BUILTIN_SUBAGENTS.keys()))
print('available:', get_available_subagent_names())
print('bash in available?', 'bash' in get_available_subagent_names())
"
```
预期输出：`available` 是 5 个 ethoinsight subagent，`bash in available? False`。**这证明 `_SubagentLiteral` 不含 bash → `task(subagent_type='bash')` 已在 Pydantic schema 层被拒**。

**修法（只剩一层）**：改 `agents/lead_agent/prompt.py`，讲清批量扫描路径——一次拿到所有上传文件的分组用 `identify_ev19_template`（它已吃 `<uploaded_files>` 全部文件、返回 `per_file_grouping`），逐个细看用 `inspect_uploaded_file`，没有 `bash` 这个 subagent 类型。**不新建批量扫描工具**（identify 已覆盖）。**不动 registry / task_tool**（它们已正确）。

---

## 一、根因（逐字节 + 实证推翻原诊断）

### 1.1 现象（ETHO-2）

Run2，Lead 想批量扫 28 个文件的分组，调 `task(subagent_type='bash', ...)` → 系统报 `Input should be 'chart-maker', 'code-executor', 'data-analyst', 'report-writer' or 'knowledge-assistant'` → 退化成逐个 `inspect_uploaded_file`，显著拖慢、多耗 token。

### 1.2 原 spec 的 infra 根因被三重实证推翻

**实证 1 —— `BUILTIN_SUBAGENTS` 不含 bash**（[`subagents/builtins/__init__.py:23-29`](../../../packages/agent/backend/packages/harness/deerflow/subagents/builtins/__init__.py#L23)）：
```python
BUILTIN_SUBAGENTS = {
    "chart-maker": ..., "code-executor": ..., "data-analyst": ...,
    "report-writer": ..., "knowledge-assistant": ...,
}   # ← 5 个，无 bash。BASH_AGENT_CONFIG 虽 import 但没进字典。
```

**实证 2 —— `names` 的两个来源都不含 bash**（[`registry.py:133-147`](../../../packages/agent/backend/packages/harness/deerflow/subagents/registry.py#L133)）：
```python
def get_subagent_names(*, app_config=None):
    names = list(BUILTIN_SUBAGENTS.keys())          # 5 个，无 bash
    for custom_name in subagents_config.custom_agents:   # 我们 config.yaml: custom_agents = {} 空
        if custom_name not in names: names.append(custom_name)
    return names
```
实跑 `config.yaml`：`subagents.custom_agents == {}`（空）。∴ `names` **永远不含 bash**。

**实证 3 —— `get_available_subagent_names()` 那段「剔 bash」是 no-op**（[`registry.py:150-165`](../../../packages/agent/backend/packages/harness/deerflow/subagents/registry.py#L150)）：
```python
names = get_subagent_names(...)        # 不含 bash
if not host_bash_allowed:
    names = [n for n in names if n != "bash"]   # ← 对一个本就不含 bash 的列表做剔除 = no-op
return names
```
`is_host_bash_allowed(cfg)` 实跑返回 `True`（config `allow_host_bash: true`），所以这段 `if` 根本不进；即便进，剔的也是不存在的元素。

**实跑验证**（mock executor 破环后）：
```
get_available_subagent_names() = ['chart-maker','code-executor','data-analyst','report-writer','knowledge-assistant']
bash in available? False
```

**结论**：`_SubagentLiteral`（由 `get_available_subagent_names()` 在 module-load 时生成，[`task_tool.py:334-350`](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/task_tool.py#L334)）**当前就不含 bash**。`task(subagent_type='bash')` 在 **Pydantic schema 校验层**就被拒——根本到不了 runtime 的 `get_subagent_config is None` 兜底。**原 spec 修法 A（给 `get_available_subagent_names` 加自洽 filter）修的是不存在的 bug：bash 早已不在 names 里，加 filter 是 no-op。**

### 1.3 报错文案本身证明是 schema 层拒绝（两套文案对照）

系统有两套不同的失败文案，对应两条路径：

| 路径 | 触发层 | 文案 |
|---|---|---|
| ① Pydantic schema | `subagent_type: _SubagentLiteral` 烘进 JSON Schema enum，进函数体前校验 | **`Input should be 'chart-maker', 'code-executor', ...`** |
| ② runtime 兜底 | 非法名绕过 schema 进函数体，[`task_tool.py:403-411`](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/task_tool.py#L403) `get_subagent_config=None` | `Error: Unknown subagent type 'bash'. ... 请改用 task(...)`（中文） |

用户 `ETHOINSIGHT_BUGS.md` 记录的报错原文是 **`Input should be 'chart-maker', ...`** = **路径①（schema 层）文案**。这反证：bash **不在枚举**（若在，绕过 schema 进 runtime，报的该是路径②的中文文案）。∴ **bash subagent 从未真正派遣成功**——lead 的 `task(subagent_type='bash')` 在 schema 层就被拦下、没派成，lead 看到报错后退化逐个 inspect。

### 1.4 生产 trace 佐证（4 thread `bash` 调用 0 次）

2026-06-23 dump 的 4 个生产 thread（3a41e483/83bfde49/e6ea7946/47e8155a）中，`task(subagent_type='bash')` **调用次数全为 0**，lead 正常派 5 个合法 subagent。ETHO-2 现象未在这批复现——进一步说明当前代码下 lead 正常不碰 bash，残余触发面窄，且即便触发也已被 schema 层正确拦截。

### 1.5 真因：lead 不知 identify 已是批量扫描器（纯 prompt）

[`identify_ev19_template_tool.py:432`](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/identify_ev19_template_tool.py#L432) 签名 `uploaded_files: list[str]` —— 它**吃全部上传文件**，内部 [L518 `for f in uploaded_files`](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/identify_ev19_template_tool.py#L518) 逐文件 parse_header、返回 `per_file_grouping`（全量分组）。lead prompt（[`prompt.py:216-217`](../../../packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py#L216)）当 `available_names` 含 bash 时才渲染「bash 可用」行——但实证 bash 不在 available_names，那行**当前根本不渲染**。lead 想派 bash 的动机 = **想批量扫 + 不知道 `identify_ev19_template` 已覆盖**。这是 prompt 没讲清路径，纯体验问题。

---

## 二、设计

### 2.1 修法（唯一有效层）：prompt 讲清批量扫描路径

改 [`prompt.py`](../../../packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py) 派遣规则段，加一段批量扫描路径指引（正面指令，符合 memory `feedback_skill_describing_tool_output_enables_hallucination` 的正面提示原则）：

```
**批量看分组 / 扫多文件，用 identify_ev19_template，不要逐个 inspect：**
- 要一次拿到所有上传文件的分组判定 → 调 identify_ev19_template（它已吃 <uploaded_files> 全部文件，
  返回 per_file_grouping = 每个文件的分组字段）。一次调用覆盖全部文件。
- 要细看某一个文件的列 / 预览 → inspect_uploaded_file（单文件）。
- subagent 类型只有这几个 EthoInsight 专用代理（见上）；没有 'bash' 这个 subagent 类型。
  需要跑 shell 命令是 lead 自己用 bash 工具，不是 task(subagent_type='bash')。
```

放在派遣规则附近（靠近现有 subagent 列表渲染处 / `prompt.py:216` 一带）。

### 2.2 防御性回归不变式（registry 自洽，**非 ETHO-2 解药**）

原修法 A 的 filter 当前是 no-op，但「校验集合恒 = 可派集合」是个**有价值的结构性不变式**——防将来有人往 `custom_agents` 注入一个 `get_subagent_config` 查不到的名字（那会重现「schema 放行又 runtime 失败」）。处理方式**降级为一条纯防御性回归测试 + 一行注释**，**不声称它修 ETHO-2**：

- **不改 registry 代码**（filter 当前 no-op，加了徒增认知负担）。改为加**回归测试**断言不变式：`get_available_subagent_names()` 返回的每个 name 都满足 `get_subagent_config(name) is not None`。若将来有人破坏（注入不可派名字），这条测试红，提示去修。
- 在 `get_available_subagent_names` 上方加一行注释，点明这个不变式（校验集合应恒 = 可派集合），引导未来改动者。

> 这样既守住「复用优先、不叠无效兜底」（memory `feedback_isolate_root_cause_before_stacking_fallback_mechanisms`），又把原 spec 想要的结构性保护以**可观测断言**形式留下——不是塞一段永不执行的 filter。

### 2.3 不新建批量扫描工具（守「复用优先」）

`identify_ev19_template` 已是批量扫描器（§1.5）。新建 `batch_inspect` 会与 `per_file_grouping` 职责重叠、违反 CLAUDE.md「复用 deerflow 现成功能优先于自造轮子」。**不新建。**

### 2.4 不改的东西

- **不改** `registry.py` 代码（自洽 filter 是 no-op，改为回归测试守不变式）。
- **不改** `task_tool.py`（schema 层已正确拒绝；runtime 兜底文案保留作第二道防线）。
- **不改** `BUILTIN_SUBAGENTS`（5 个 ethoinsight subagent 不变；本 fork 故意不要 bash subagent）。
- **不改** `identify_ev19_template` / `inspect_uploaded_file`。
- **不删** `is_host_bash_allowed` 调和逻辑（sandbox 安全语义，正交）。

---

## 三、改动清单

### 3.1 `agents/lead_agent/prompt.py` —— 加批量扫描路径指引
- 派遣规则段（[L216](../../../packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py#L216) 一带）加 §2.1 那段。

### 3.2 `subagents/registry.py` —— 加一行不变式注释（不改逻辑）
- `get_available_subagent_names` 上方注释：校验集合应恒 = 可派集合（`get_subagent_config` 非 None）；当前由 BUILTIN_SUBAGENTS 无 bash + custom_agents 受控保证，回归测试守此不变式。

### 3.3 不改其他
- task_tool、identify_ev19_template、inspect_uploaded_file、BUILTIN_SUBAGENTS 均不动。

---

## 四、测试（TDD）

> ⚠️ prompt 契约测试必须用 importlib 加载被测源 prompt（memory `feedback_worktree_shares_main_venv...`），否则读主仓旧 prompt 假绿。

测试文件：`packages/agent/backend/tests/test_subagent_registry_self_consistency.py`（新增）+ `test_lead_prompt_batch_scan_guidance.py`（新增）。

1. **test_available_names_excludes_bash**（坐实现状，**改前已绿**）
   - mock `deerflow.subagents.executor` 破环。
   - 断言 `'bash' not in get_available_subagent_names()`。
   - 断言 `get_subagent_config('bash') is None`。
   - 说明：这条**不是红→绿**，它坐实「现状已正确」——文档化 §1.2 的实证，防回归。

2. **test_registry_self_consistency_invariant**（防御性不变式）
   - 断言 `get_available_subagent_names()` 的每个 name 都满足 `get_subagent_config(name) is not None`。
   - 这是 §2.2 的不变式守卫：将来若有人注入不可派名字，此测试红。

3. **test_subagent_literal_excludes_bash**
   - 断言 `_make_subagent_literal()` 生成的 Literal 不含 'bash'（间接验 schema 层会拒）。
   - ⚠️ 导 task_tool 触发已知环——用 conftest 既有 mock 或 importlib 局部加载，参照现有 task_tool 相关测试的导入方式。

4. **test_runtime_fallback_message_preserved**（守第二道防线）
   - 直接调 task_tool 传一个绕过 schema 的非法名（模拟 app_config 漂移），断言仍返回含「请改用」的指引文案（`task_tool.py:405-411` 不被破坏）。

5. **prompt 契约测试 test_lead_prompt_has_batch_scan_guidance**（红→绿）
   - 用 importlib 加载被测源 prompt，渲染后断言含「identify_ev19_template」批量扫描指引串 + 「没有 'bash' 这个 subagent 类型」串。
   - 改前红（prompt 无此段），改后绿。

---

## 五、验收标准

1. 测试 1/2/3/4 全绿（1/2/3/4 多为「坐实现状 + 守不变式」非红→绿）；测试 5（prompt 契约）改前红、改后绿。
2. lead prompt 含批量扫描路径指引（identify 批量扫 + 没有 bash subagent 类型）。
3. `task(subagent_type='bash')` 仍在 schema 层被拒（本就如此，本 spec 不退化它）。
4. registry / task_tool 代码**未改**（只加注释 + 测试）。
5. 裸导入 `app.gateway` / `make_lead_agent` 0 退出（虽未改核心代码，prompt 改动后仍按 import 环铁律验一次）。
6. backend prompt 相关测试邻域绿。

---

## 六、风险与注意事项

1. **不要恢复原 infra 修法 A**：它是 no-op（bash 早不在 names）。若 reviewer 质疑「为何不加 filter」，答：filter 不解决现象（现象在 prompt 层），且当前无 bug 可修；不变式由回归测试守，比塞一段永不执行的 filter 更诚实可观测。
2. **prompt 改动是受保护文件**：sync deerflow 时 surgical 守护这段批量扫描指引（CLAUDE.md 同步规则）。
3. **prompt 契约测试必须读被测源**：否则读主仓旧 prompt 假绿（memory 多次踩坑）。
4. **schema 层校验依赖 module-load 时的 `get_available_subagent_names()`**：`_SubagentLiteral` 在 import task_tool 时固化。运行期 config 漂移（custom_agents 注入）导致可用集合变时，schema 不重算 → 靠 runtime 兜底文案（L403）+ 测试 2 的不变式提前在 CI 抓。这是已有行为。
5. **若将来真要支持 lead 批量跑 shell**：那是给 lead 的 bash **工具**（已有），不是 bash **subagent**；两者别混。
6. **三大病理自检**（HarnessX operational mirror）：① reward hacking——不适用（prompt 指引+注释，无验收可 hack）；② catastrophic forgetting——加批量扫描指引会不会削弱邻近派遣规则？只加一段、diff 守邻段；③ under-exploration——本条**正确选 prompt 而非结构**（schema 门已正确拒 bash，结构没缺，缺的是 lead 动机），不是逃避结构改动。

---

## milestone 建议
- 「EPM dogfood 图表/交互流水线打磨」track：ETHO-2 修订澄清了「registry 已正确、问题在 prompt」，归到交互体验打磨；与 ETHO-5（分组全量 prompt 约束）同属「lead 不知道现成工具能力 → 自创低效路径」的 prompt 指引缺口系列。
