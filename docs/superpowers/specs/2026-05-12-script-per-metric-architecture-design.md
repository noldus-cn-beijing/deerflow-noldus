# Script-Per-Metric 架构设计

> 日期：2026-05-12
> 状态：approved by user，待 plan → implementation
> 触发：2026-05-12 e2e dogfooding（EPM 单样本分析）暴露 code-executor 不可控性 + 前端 reasoning 重复 bug

---

## 设计原则（贯穿本 spec）

> **"agent 架构由上游 deerflow 操心，我们专注数据分析洞察。"**

任何架构决策先过这条筛子：

- 凡是要动 deerflow 上游组件（agent 框架、middleware 范式、tool 协议、sandbox 协议）的方案，重新评估
- 凡是属于"产品业务层"（ethoinsight 库、skills、prompt、范式知识）的方案，优先考虑

这条原则贯穿本 spec 所有决策。

---

## 1. 背景与触发问题

### 1.1 触发的两个 bug

**2026-05-12 e2e dogfooding（EPM 单样本分析）暴露：**

**Bug 1（后端 / 关键）**：code-executor 在 `max_turns=12` 预算内 8 次都在用 `bash python -c "...help(...)"` 探测 `ethoinsight` 库 API 签名，没来得及写 `analysis.py` 就被强制终止，subagent 返回 `"Task Succeeded. Result: No text content in response"`，整个分析失败。

**Bug 2（前端）**：lead-agent 的"思考过程"在 UI 上重复显示两遍。

### 1.2 根因（已用日志和代码闭环）

**Bug 1 根因链：**

1. `code_executor.py:83` 配置 `max_turns=12`
2. `executor.py:478-481` 硬限：AI message 数 ≥ max_turns 即 break
3. 日志显示 code-executor 在 12 步内的实际工具调用序列：
   - 步骤 1-3：read SKILL.md + read paradigm md + read charts md + read experiment-context
   - 步骤 4-9：**6 次 `python -c "...help(...)" / "from ethoinsight import ..."` 探测命令**
   - 步骤 10-12：mkdir + cp 文件准备
   - 第 12 步触发上限，从未写过 `analysis.py`
4. `executor.py:523` fallback 路径返回 `"No text content in response"`

**为什么探测？** 即使 `by-paradigm/epm.md` 已有完整胶水脚本范例（含所有签名、调用方式、handoff schema），agent 仍然不信任文档，要用 `help()` 二次验证。这是「agent 拼脚本」CodeAct 范式的固有失败模式 —— agent 把 bash 当 REPL 用，而非脚本执行器。

**Bug 2 根因：**

`packages/agent/frontend/src/core/messages/utils.ts:88-121` 的 `groupMessages` 对**同一条** AI message（reasoning + content + 无 tool_calls）会**同时**分入两个 group：

- L102：`hasReasoning(message)` true → 加入 `assistant:processing` group（MessageGroup 渲染 thinking 折叠块）
- L118：`hasContent && !hasToolCalls` true → 又开 `assistant` group（MessageListItem 渲染主回答）

`MessageListItem.MessageContent_` (L138-191) 中 reasoning 在 `!rawContent` 守卫下渲染，流式过程中可能两处同时渲染。

### 1.3 本次任务承载的关键决策

不只是修两个 bug，而是借这次 dogfood **重新评估"agent 拼脚本"架构**是否适合我们的封闭科研场景，并把结论 + 未来愿景固化到本 spec。

---

## 2. 三种架构的辩论 + 调研 evidence

### 2.1 架构方向 1：Agent 拼脚本（当前架构，CodeAct 范式）

**哲学**：agent 是开发者，给它工具（bash + 文件读写）+ 文档，让它现场拼出胶水脚本并执行。

**学术背书**：
- [Executable Code Actions Elicit Better LLM Agents (CodeAct, ICML 2024)](https://arxiv.org/abs/2402.01030) —— 对 JSON function-calling 成功率高 20%、token 少 30%
- [Building agents with the Claude Agent SDK (Anthropic)](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk) —— "give your agents a computer"

**失败模式（dogfood 实证）**：

- agent 不信任文档，会用 bash 反复 `python -c help/import` 探测 API
- 浪费 turn 预算
- 拼出的代码质量取决于 LLM 个体差异，不可复现

**治标方案**：prompt 收紧 + 文档补全 + Guardrail 拦探测命令。但属"打地鼠"性质，新模型 / 新场景会冒出新失败模式。

### 2.2 架构方向 2：固定函数（agent 是参数填充器）

**哲学**：ethoinsight 库提供 `analyze_<paradigm>()` 端到端函数，agent 只填参数（文件路径、分组、可选指标）。

**业界背书**：
- [Compiled AI (arxiv 2604.05150)](https://arxiv.org/abs/2604.05150) —— 编译期生成代码，运行期 deterministic：96% 成功率 + token 减少 57x
- [Text-to-SQL for Enterprise Data Analytics (arxiv 2507.14372)](https://arxiv.org/abs/2507.14372) —— 工业实践要求 separation of generation from execution

**致命缺陷**：

- 函数签名固定 → 用户特殊需求（"只要轨迹图"、"跳过统计"）必须改函数签名或加大量可选参数
- 灵活性硬限制
- 跨范式分析（v0.2+ 愿景）无法表达

**同领域对标**：[BehaveAgent (PMC, 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12139829/) 用固定 5-stage pipeline，但每 stage 内部还是 LLM 推理 —— 不是纯函数。

### 2.3 架构方向 3：脚本即指标（**最终选择**）

**哲学**：ethoinsight 提供大量**独立可执行脚本**（每个脚本 = 一个指标 / 一张图 / 一个统计检验），agent 是**实验师 / 编排者**，看 skill md 选择跑哪些脚本、什么顺序，bash 执行后收集结果。

**设计灵感**：
- unix 哲学 —— 每个工具做好一件事，组合成 pipeline
- [Aider's repo map](https://aider.chat/) —— front-load context 思想（skill md = agent 的决策手册）
- [Hermes Agent's execute_code](https://hermes-agent.nousresearch.com/) —— "组合 RPC 调用"，把多步 collapse 成一次推理调用

**关键优势**：

| 优势 | 说明 |
|---|---|
| **不让 agent 拼代码** | 解决方向 1 的不可控性 |
| **保留 agent 编排灵活性** | 解决方向 2 的硬限制 —— agent 仍根据用户需求决定脚本组合 |
| **最大化复用 deerflow 现有架构** | code-executor 工具集（bash + read_file + write_file + ls + str_replace）一个不变；skill 渐进披露机制不变 |
| **科学严谨性 = 库代码质量** | 每个脚本调用 `ethoinsight.metrics.<paradigm>.*` 已有函数，统计方法、领域阈值都在 Python 库里、可单测、可回归 |
| **golden case 飞轮天然衔接** | 每个脚本 = 独立测试单元；专家标注"哪种实验该跑哪些脚本"可直接作为回归测试和 SFT 样本 |

### 2.4 调研 evidence 汇总

| Evidence | 来源 | 与本设计的关系 |
|---|---|---|
| **CodeAct 优势** | [arxiv 2402.01030](https://arxiv.org/abs/2402.01030) | 学术 SOTA 是"开放问题"场景，我们是"封闭科研流程"，范式错配 |
| **Compiled AI** | [arxiv 2604.05150](https://arxiv.org/abs/2604.05150) | 工业生产要求 deterministic 兜底；我们的脚本即指标就是 deterministic 层 |
| **ChatGPT 科研数据分析一致性** | [JOGH 2024](https://jogh.org/2024/jogh-14-04070) | 描述统计高度一致，复杂统计不稳；这正是为什么把统计封装到脚本而非让 agent 拼 |
| **BehaveAgent（同领域）** | [PMC 12139829, 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12139829/) | 行为学分析 SOTA agent 走"固定 pipeline + 模块内 LLM"，与我们脚本即指标同源 |
| **Aider repo map** | [aider.chat](https://aider.chat/) | front-load context 思想 → 我们的 skill md 作为决策手册 |
| **OpenCode / Cline / LangChain / LlamaFirewall** | 多个 | 程序化 tool 拦截范式 → 我们的 ScriptInvocationOnlyGuard |
| **Hermes execute_code** | [hermes-agent.nousresearch.com](https://hermes-agent.nousresearch.com/) | 组合 RPC 哲学 → 我们的脚本即指标 |

### 2.5 决策与理由

**决策：走方向 3（脚本即指标）。**

**理由**：

1. 服务"分析结论 SOTA"这个真正目的（不是"agent 架构 SOTA"）
2. 兼顾方向 1 的灵活性 + 方向 2 的可控性
3. 工程改动小：不动 deerflow，不改工具集，只是把 metrics 模块包装成可独立调用脚本 + 重写 skill md 为"脚本清单 + 决策手册"
4. 符合本 spec 顶部的设计原则（agent 架构由 deerflow 操心，我们专注数据分析洞察）

---

## 3. 脚本即指标的具体架构

### 3.1 目录结构

```
packages/ethoinsight/ethoinsight/scripts/
├── __init__.py
├── _common/
│   └── ...（多范式共用：轨迹图、距离速度等）
├── epm/
│   ├── compute_open_arm_time_ratio.py
│   ├── compute_open_arm_entry_count.py
│   ├── compute_open_arm_entry_ratio.py
│   ├── compute_open_arm_time.py
│   ├── compute_total_entry_count.py
│   ├── plot_trajectory.py
│   ├── plot_box_open_arm.py
│   ├── plot_raincloud.py
│   └── run_groupwise_stats.py
├── oft/
├── zero_maze/
├── ldb/
├── fst/
├── tst/
└── shoaling/
```

每个脚本 ~20-40 行，做 4 件事：

1. argparse 解析参数（统一接口）
2. parse 原始数据（**每个脚本独立 parse**，重复 ~100ms 成本可接受）
3. 调 `ethoinsight.metrics.<paradigm>.*` 或 `ethoinsight.charts.*` 现有函数
4. 输出 JSON / PNG，stdout 打印 `[result] {...}` 给 subagent 抓取

### 3.2 统一脚本接口约定

| 脚本类型 | 输入 | 输出 |
|---|---|---|
| `compute_*.py` | `--input <轨迹文件>`（单文件） | `--output <metric.json>`；stdout `[result] {"metric": "...", "value": ...}` |
| `plot_*.py` | `--input <轨迹文件>`（单文件，如 trajectory）或 `--inputs <JSON 数组路径>` + 可选 `--groups <JSON 路径>`（多文件聚合图） | `--output <plot.png>` |
| `run_*_stats.py` | `--inputs <JSON 数组路径>` + `--groups <JSON 路径>` | `--output <stats.json>`；stdout `[result] {...}` |

**多文件输入约定**：

- `--inputs` 接受**指向 JSON 数组的文件路径**，文件内容形如 `["/mnt/.../subject1.txt", "/mnt/.../subject2.txt"]`
- 不用逗号分隔字符串（避免文件名含逗号或特殊字符的歧义、shell 转义复杂性）
- agent 用 `write_file` 先生成这个 JSON 文件，再调脚本传入路径

**分组约定**：

- `--groups` 同理接受 JSON 文件路径，内容形如 `{"control": ["subject_1", "subject_2"], "treatment": ["subject_3", "subject_4"]}`
- 不接受 inline JSON 字符串（shell 转义复杂）

**通用术语定义**：

- **n_per_group**：每个分组的样本数（subject 数，不是观察帧数）
- **n**：总样本数（subject 数）
- **subject**：受试个体（一条数据文件 = 一个 subject）

### 3.3 调用方式

**使用 Python `-m` 模块调用**（不是绝对路径）：

```bash
python -m ethoinsight.scripts.epm.compute_open_arm_time_ratio \
  --input /mnt/user-data/uploads/subject1.txt \
  --output /mnt/user-data/workspace/outputs/epm_open_arm_time_ratio.json
```

**理由**：

- 无路径硬编码
- 无 cwd 依赖
- 可被 pytest 单测（`subprocess.run(["python", "-m", "..."])`）
- 包重命名 / 重组对 agent 无侵入（包结构由 Python import 系统隐藏）

### 3.4 skill md 的新角色

`packages/agent/skills/custom/ethoinsight-code/references/by-paradigm/<paradigm>.md` 从「胶水脚本范例 + handoff schema」变成 **「脚本清单 + 决策手册」**。

**结构**：

```markdown
# <paradigm> 范式可用脚本

## 范式简介
EV19 模板映射：...
学术范式：...

## 可用脚本清单

### 核心指标
| 脚本 | 输入 | 输出 | 含义 |
|---|---|---|---|
| python -m ethoinsight.scripts.epm.compute_open_arm_time_ratio | --input <轨迹> --output <json> | {value: float} | 开臂时间占比 |
| ...

### 可视化
| 脚本 | ... | ... | ... |

### 统计
| 脚本 | ... | ... | ... |

## 实验设计决策树
- n=1（单样本描述性）→ 跑核心指标 + plot_trajectory；跳过所有统计 + 组间对比图
- n_per_group 3-4 → 跑核心指标 + 组间箱线图；统计仅作参考
- n_per_group ≥ 5 → 全套
- 用户明确"只要 X" → 仅跑相关脚本

## handoff JSON 字段约定
- paradigm: "<key>"
- metrics: { 脚本输出 JSON 聚合 }
- charts: [ PNG 路径列表 ]
- statistics: { stats 脚本输出 }（可选）
- data_quality_warnings: [ ... ]（保留现有 schema）

## 错误处理
- 脚本 stderr 含 "ValueError: ..."  → 数据列名识别失败 → 反问 lead
- 脚本 stderr 含 "FileNotFoundError" → 输入路径问题 → 检查 ls
- ...
```

### 3.5 code-executor 工作流（新）

**system_prompt workflow 段重写为**：

```
1. read by-paradigm/<paradigm>.md   ← 看清单 + 决策树
2. 根据 lead 给的实验信息（n、分组、用户特殊需求），裁剪要跑的脚本列表
3. for script in 选中列表:
     bash python -m ethoinsight.scripts.<paradigm>.<script> --input ... --output ...
   遇到 stderr 报错：
     - 读 traceback
     - 查 by-paradigm md 的「错误处理」段
     - 决定重试 / 跳过 / 反问 lead
4. 把每个脚本的输出 JSON 收集起来 + 图表路径，写 handoff_code_executor.json
5. 输出 [gate_signals] 块给 lead
```

**工具集不变**（bash + read_file + write_file + ls + str_replace），但 bash 的使用模式从"任意 Python"变成"调脚本"。

---

## 4. ScriptInvocationOnlyGuard（白名单 Guardrail）

### 4.1 角色

把"agent 只调脚本"从**软约定**（prompt 描述）升级为**硬约束**（程序拦截）。

### 4.2 实现

`packages/agent/backend/packages/harness/deerflow/guardrails/script_invocation_only_provider.py`（新文件）：

```python
class ScriptInvocationOnlyProvider:
    """code-executor 的 bash 命令必须是：
      - python -m ethoinsight.scripts.<paradigm>.<script> ...
      - mkdir / cp / mv / ls / cat / grep / head / tail（文件操作）
    其他 bash 命令一律 deny + 提示走脚本路径。"""

    name = "script_invocation_only"

    _ALLOWED_PYTHON_PATTERN = re.compile(
        r"^\s*python\s+-m\s+ethoinsight\.scripts\.\w+\.\w+(\s|$)"
    )
    _ALLOWED_FILE_OPS = re.compile(
        r"^\s*(mkdir|cp|mv|ls|cat|grep|head|tail)(\s|$)"
    )

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        if request.tool_name != "bash":
            return GuardrailDecision(allow=True)
        if "code-executor" not in (request.agent_id or ""):
            return GuardrailDecision(allow=True)
        cmd = request.tool_input.get("command", "")
        if self._ALLOWED_PYTHON_PATTERN.match(cmd):
            return GuardrailDecision(allow=True)
        if self._ALLOWED_FILE_OPS.match(cmd):
            return GuardrailDecision(allow=True)
        return GuardrailDecision(allow=False, reasons=[
            GuardrailReason(
                code="script_invocation_only.not_a_script_call",
                message=(
                    "该 bash 命令不是脚本调用。code-executor 仅可：\n"
                    "  1. 调脚本：python -m ethoinsight.scripts.<paradigm>.<name> --input ... --output ...\n"
                    "  2. 文件操作：mkdir / cp / mv / ls / cat / grep / head / tail\n"
                    "请改用脚本调用形式。可用脚本清单见 by-paradigm/<范式>.md。"
                ),
            )
        ], policy_id="script_invocation_only")

    async def aevaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        return self.evaluate(request)
```

### 4.3 挂载位置

`subagents/executor.py:_create_agent()` 现有代码已经 append 了 `GuardrailMiddleware(HandoffIsolationProvider)`，再 append 一个 `GuardrailMiddleware(ScriptInvocationOnlyProvider)`。**零架构变化**。两个 Guard 正交：

- `HandoffIsolationProvider`：管 subagent 读 handoff 文件的隔离
- `ScriptInvocationOnlyProvider`：管 code-executor 的 bash 命令模式

### 4.4 设计决策：白名单（正向）而非黑名单（负向）

| 维度 | 黑名单（旧方案，已弃） | 白名单（本方案） |
|---|---|---|
| 拦截目标 | "禁止 `python -c help/import`" —— 列举禁忌 | "限定 `python -m ethoinsight.scripts...` + 文件操作" —— 列举允许 |
| 模型反激活风险 | 中（"禁止 X" 反而吸引模型试边界） | 低（白名单清晰，deny 反馈是"请改成 Y"而非"禁止做 X"） |
| 黑名单膨胀风险 | 高（新探测模式不断冒出） | 无（白名单稳定，加新脚本不动 Guard） |
| 维护成本 | 高 | 低 |

---

## 5. 前端 reasoning 重复修复

### 5.1 数据结构层修复（F1）

`packages/agent/frontend/src/core/messages/utils.ts:groupMessages`：

**原则**：同一条 AI message **只进一个 group**。

**新分组规则**（替换 L88-121 现有逻辑）：

| message 形态 | 归属 group |
|---|---|
| reasoning + content + 无 tool_calls（最终回答） | `assistant`（**只此一处**） |
| reasoning + tool_calls（中间步骤） | `assistant:processing` |
| tool_calls 无 reasoning（纯中间动作） | `assistant:processing` |
| 单纯 reasoning 无 content 无 tool_calls | `assistant:processing` |
| content 无 reasoning 无 tool_calls | `assistant` |
| present_files | `assistant:present-files` |
| subagent | `assistant:subagent` |

### 5.2 渲染层配套修复（message-list-item.tsx）

`MessageContent_` 当 message 有 reasoning 时，在 main content **上方**加 `<Reasoning>` 折叠块：

```tsx
return (
  <AIElementMessageContent className={className}>
    {filesList}
    {reasoningContent && (
      <Reasoning isStreaming={isLoading}>
        <ReasoningTrigger />
        <ReasoningContent>{reasoningContent}</ReasoningContent>
      </Reasoning>
    )}
    <MarkdownContent
      content={contentToDisplay}
      isLoading={isLoading}
      className="my-3"
      components={components}
    />
  </AIElementMessageContent>
);
```

删除原 `!rawContent` 特殊分支（L182-191），统一渲染路径。

**默认折叠**：与 `MessageGroup` 中间步骤 thinking 块行为一致。流式期间 `isStreaming` 自动展开，结束后自动折叠。

### 5.3 测试锁死不变量（F2）

引入 vitest（项目 frontend 之前无 test runner）：

```bash
pnpm add -D vitest @vitest/ui @testing-library/react jsdom
```

`packages/agent/frontend/src/core/messages/utils.test.ts` 用例：

| Case | 输入 | 期望 |
|---|---|---|
| 1 | 1 条 AI message：reasoning + content + 无 tool_calls | groups.length === 1，type === "assistant" |
| 2 | 1 条 AI message：reasoning + tool_calls | groups.length === 1，type === "assistant:processing" |
| 3 | 1 条 AI message：reasoning only | groups.length === 1，type === "assistant:processing" |
| 4 | 1 条 AI message：content only | groups.length === 1，type === "assistant" |
| 5 | 1 条 AI message：tool_calls only | groups.length === 1，type === "assistant:processing" |
| 6 | 连续 2 条 AI message：第 1 条 reasoning+tool_calls，第 2 条 reasoning+content | 2 个 group（processing + assistant） |

---

## 6. 本次任务范围

### 6.1 后端 —— 7 个范式全部脚本即指标重构

**范围**：`epm / oft / zero_maze / ldb / fst / tst / shoaling`（已实现指标的 7 个范式）

**改动清单**：

1. 新建 `packages/ethoinsight/ethoinsight/scripts/` 包，按 §3.1 目录结构组织（含 `_common/` 子目录用于跨范式通用脚本，如 `compute_distance_moved.py`、`compute_velocity_stats.py`、`plot_trajectory.py` 等）
2. 每个范式现有 metrics 函数 → 包装成可执行脚本，统一参数接口（§3.2）
3. `packages/ethoinsight/ethoinsight/charts.py` 的图函数 → 同样包装
4. `statistics.run_groupwise` → 包装成 `run_*_stats.py` 脚本
5. 每个脚本独立单测（`tests/scripts/test_<paradigm>_<script>.py`）：
   - 用 `subprocess.run` 调脚本，验证输出 JSON 结构和值
   - 边界情况（缺列、空数据、None 返回）
6. `packages/agent/skills/custom/ethoinsight-code/` 改造：
   - `SKILL.md` 顶部从"按 by-paradigm/<paradigm>.md 写胶水脚本"改为"按 by-paradigm/<paradigm>.md 选脚本编排"
   - `by-paradigm/<paradigm>.md` 重写为「可用脚本清单 + 决策树 + handoff 字段约定」（§3.4）
7. `code_executor.py` system_prompt workflow 段重写为 §3.5 的 3 步硬路径
8. 新建 `guardrails/script_invocation_only_provider.py`（§4），在 `subagents/executor.py` append

### 6.2 前端 —— reasoning 重复修复

1. `core/messages/utils.ts:groupMessages` 修分组规则（§5.1）
2. `components/workspace/messages/message-list-item.tsx:MessageContent_` 统一渲染（§5.2）
3. 引入 vitest + 写 `utils.test.ts`（§5.3）

### 6.3 不动的部分

- deerflow 上游架构（subagent executor、guardrails middleware 框架、skill 渐进披露）
- code-executor 工具集（bash + read_file + write_file + ls + str_replace）
- lead-agent prompt（除非 by-paradigm 重写后 lead 派发文本需同步）
- data-analyst / report-writer / knowledge-assistant 三个 subagent
- Pydantic warnings（langgraph 0.7.65 内部告警，无功能影响）

### 6.4 验收标准

1. 7 个范式合成数据 e2e 测试全绿（每个范式至少 1 个 happy path + 1 个 n=1 path）
2. 前端 dogfood：reasoning 在 UI 上只出现一次
3. vitest test 全绿
4. 旧 code-executor 胶水脚本相关代码删除，无废弃残留

---

## 7. 未来愿景 + 迁移路径

### 7.1 脚本即指标作为长期契约

这次架构选择不是临时方案，是 **v0.1 → v0.2 → v1.0 的核心契约**。它在三个维度上服务于产品演进：

**1. 范式扩展不改 agent**

当行为学同事提出新范式（review-packages 里还没实现的 13 个范式：`barnes_maze / morris_water_maze / fear_conditioning / novel_object / sociability / t_maze / y_maze / radial_8_arm / active_avoidance / 3d_swimming / aquatic_open_field / cross_maze_fish / phenotyper_homecage / insect_open_field`）：

- 加 `ethoinsight/scripts/<新范式>/` 目录 + 脚本
- 加 `skills/.../by-paradigm/<新范式>.md`（脚本清单 + 决策树）
- **不改 agent 代码、不改 prompt、不改 guardrail**

agent 自动认识新范式（通过 lead-agent 的范式识别 + skill 渐进披露）。

**2. Golden Case 飞轮天然衔接**

行为学同事的 golden case（参见 [golden-cases/SCHEMA.md](../../../golden-cases/SCHEMA.md)）天然映射到脚本即指标架构：

- golden case 的"期望分析结论" = 「在该数据上应该跑哪几个脚本，每个脚本输出应该是什么」
- **回归测试**：跑 golden case 输入 → 比对脚本输出
- **领域知识源**：哪些脚本组合才算"完整分析" = 专家定义
- **SFT 种子数据**：「场景输入 → 脚本编排决策 → 执行结果」是 fine-tune Qwen3-8B 的天然样本（CLAUDE.md 第 5 条提到的微调策略）

**3. 跨范式 / 跨指标的高级分析（v0.2+）**

用户可能提出："这只 zebrafish 的 EPM 焦虑指标和 OFT 焦虑指标是否一致？"

- 旧架构（agent 拼脚本）：agent 要现场拼跨范式的 join 代码 → 高度不可控
- **脚本即指标**：加几个跨范式聚合脚本（`scripts/cross_paradigm/anxiety_consistency.py`），agent 学会"用户问跨范式问题 → 调聚合脚本"，**仍然是编排，仍然不写代码**

这是为什么 [CLAUDE.md 第 1 条] 提到的"全生命周期行为学研究助手"（实验指导 → 数据分析 → 追问 → 知识问答 → 跨范式证据链）在脚本即指标架构下天然可达。

### 7.2 迁移路径

```
Phase 1（本次任务，~ 1 周）
├── 7 个范式脚本化（epm/oft/zero_maze/ldb/fst/tst/shoaling）
├── skill md 重写为脚本清单 + 决策树
├── code-executor prompt 改造 + ScriptInvocationOnlyGuard
├── 前端 reasoning 重复修复 + vitest 锁死
└── 合成数据 e2e 全绿

Phase 2（拿到真数据后，~ 2-3 天）
├── 真 EthoVision 数据列名 regex 调校（接续 2026-05-11-handoff.md 的待办）
├── 用真数据 + 行为学同事提供的实验 case 跑 dogfood
└── 修暴露的剩余问题

Phase 3（v0.1 9 月里程碑前，golden case 飞轮启动后）
├── 行为学同事补全 golden cases（覆盖 7 个范式各 1-2 个 case）
├── golden case → 回归测试持续运行
└── 训练数据飞轮（auto-collected JSONL）累积到 SFT 触发线

Phase 4（v0.2，~ 2026 年 11-12 月）
├── 行为学同事新增的 13 个范式逐个加 scripts/<范式>/ 目录
└── 跨范式分析脚本（focused on anxiety / locomotion / social 等主题群）

Phase 5（v1.0+）
├── Qwen3-8B SFT 训练（用累积 SFT 样本）
└── 微调模型替换 lead / code-executor 推理（按 CLAUDE.md 第 5 条计划）
```

### 7.3 未来不要走的弯路（明确否决）

1. ❌ **不要回退到"agent 拼脚本"**：即使遇到"用户特殊需求"也不要给 agent 开 `python -c` 后门。新需求 = 加新脚本，不是让 agent 现写代码。

2. ❌ **不要走"单一固定函数"**（前述方向 2）：会损失编排灵活性，且不符合上游 deerflow 的 agent-as-orchestrator 哲学。

3. ❌ **不要在 ethoinsight 库里造新 agent 框架**：跨范式 / 跨指标分析的编排逻辑应该交给 agent，不是封装成 `analyze_anxiety_across_paradigms()` 这种巨函数。

4. ❌ **不要把 skill md 写成 API reference 文档**：skill md 是「决策手册」，告诉 agent "什么场景跑什么脚本"，不是「函数签名手册」。API reference 留给 Python docstring。

---

## 8. 关键参考资料

### 项目内文档

- [CLAUDE.md](../../../CLAUDE.md) —— 项目设计原则（特别是第 5、7、8、12 条）
- [docs/handoffs/2026-05-11-sota-migration-completed-real-data-pending-handoff.md](../../handoffs/2026-05/2026-05-11-sota-migration-completed-real-data-pending-handoff.md) —— SOTA 架构迁移交接（即本 spec 要重新评估的架构）
- [docs/e2e/高架十字迷宫轨迹分析.md](../../e2e/高架十字迷宫轨迹分析.md) —— 触发本次重构的 e2e 记录
- [golden-cases/SCHEMA.md](../../../golden-cases/SCHEMA.md) —— golden case 数据结构
- [docs/sop/golden-case-sop.md](../../sop/golden-case-sop.md) —— golden case 协作 SOP
- [docs/specs/llm-finetuning-strategy.md](../../specs/llm-finetuning-strategy.md) —— 微调策略
- [packages/agent/backend/packages/harness/deerflow/guardrails/handoff_isolation_provider.py](../../../packages/agent/backend/packages/harness/deerflow/guardrails/handoff_isolation_provider.py) —— GuardrailProvider 范式参考

### 外部调研

- [Executable Code Actions Elicit Better LLM Agents (CodeAct, ICML 2024)](https://arxiv.org/abs/2402.01030)
- [Compiled AI: Deterministic Code Generation for LLM-Based Workflow Automation (arxiv 2604.05150)](https://arxiv.org/abs/2604.05150)
- [Text-to-SQL for Enterprise Data Analytics (arxiv 2507.14372)](https://arxiv.org/abs/2507.14372)
- [BehaveAgent: An autonomous AI agent for universal behavior analysis (PMC 12139829, 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12139829/)
- [Evaluating ChatGPT-4's data analytic proficiency (JOGH 2024)](https://jogh.org/2024/jogh-14-04070)
- [Building agents with the Claude Agent SDK (Anthropic)](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk)
- [Hermes Agent (Nous Research)](https://hermes-agent.nousresearch.com/)
- [Aider AI Pair Programming](https://aider.chat/)
- [OpenCode Permissions](https://opencode.ai/docs/permissions/)
- [LangChain Guardrails](https://docs.langchain.com/oss/python/langchain/guardrails)
- [LlamaFirewall (Meta)](https://ai.meta.com/research/publications/llamafirewall-an-open-source-guardrail-system-for-building-secure-ai-agents/)
