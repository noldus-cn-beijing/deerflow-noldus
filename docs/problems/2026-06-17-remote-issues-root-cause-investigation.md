# 远程 Issue 批量根因调研（2026-06-17）

> 本文档对 `noldus-cn-beijing/noldus-insight` 远程仓库当前 **17 个 open issue** 中的**用户反馈类 bug** 做根因调研。
> 调研方式：读取 issue 正文 → 在当前 `dev` 分支（`a381cb22`）代码中定位触发逻辑 → 写出问题原因与问题描述。
>
> 调研时 git 状态：工作区 `packages/agent/config.yaml` 有**未提交的本地改动**（放宽了若干阈值），见下文「⚠️ 横切根因」。

---

## ⚠️ 横切根因（影响多数 issue）：本地放宽配置从未进 git、从未部署

在调研过程中发现一个**贯穿聚类 A、B 的系统性根因**：

| 配置项 | 工作区本地值（未提交） | `config.example.yaml`（git 模板 = 线上默认形态） |
|---|---|---|
| `uploads.max_files` | **50** | **10** |
| `uploads.max_total_size` | **500 MiB** | **100 MiB** |
| `loop_detection.tool_freq_overrides.inspect_uploaded_file.hard_limit` | **100** | **（不存在该段）** |
| `loop_detection.tool_freq_hard_limit`（全局） | **50** | **（不存在该段）** |

**问题**：`packages/agent/config.yaml` 是 gitignored 的本地文件（CLAUDE.md：「配置文件存在开发者本地 `~/ethoinsight-prod/`，不进 git」）。开发者本机已把上传上限和循环检测阈值放宽到合理值，但：

1. `git log -S "inspect_uploaded_file" -- packages/agent/config.yaml` 返回空 → **放宽配置从未进入过 git 历史**；
2. `git show HEAD:packages/agent/config.yaml` 报「路径不在 HEAD 中」→ **config.yaml 整个文件不在版本控制**；
3. 线上部署用的是部署者本机的 config（或 config.example 模板），**这些放宽值大概率从未上线**。

**后果**：线上生效的仍是代码硬编码默认值（`loop_detection` 段缺失时）或 `config.example.yaml` 模板值（`uploads` 段）。issue 用户从未享受过本机的修复。

**部署链证据（`packages/agent/scripts/deploy.sh:84-107`）**：部署脚本对 config 内容**零保证**——
- `DEPLOY_CONFIG_PATH` 未设时默认指向 `$REPO_ROOT/config.yaml`（仓库根本地文件，gitignored）；
- 若该文件**不存在**，则 `cp config.example.yaml → config.yaml`（`deploy.sh:95-98`）从模板**种子化**。

即线上 config 来源是二选一：(a) 部署者本机仓库根的 config.yaml（含放宽值），或 (b) 若不存在则从模板种子（**max_files=10、无 loop_detection 段**）复制。CLAUDE.md 又说生产 config 存在 `~/ethoinsight-prod/` 经 `DEPLOY_CONFIG` env rsync——这等于一个**不可复现的手工配置流**，没有任何机制保证放宽后的 loop_detection/上传上限进入生产。模板 `config.example.yaml` 仍是严格默认值，是线上实际生效概率最高的那一档。

**建议（工程纪律）**：配置放宽要么(a) 提交进 `config.example.yaml` 模板并同步部署 SOP，要么(b) 写进部署脚本/`.env` 注入。仅改本地工作区 = 永远只修好了开发者自己的机器。

---

## 聚类 A —「强制停止 / FORCED STOP」（高频反复复发）

**涉及 issue**：[#114](https://github.com/noldus-cn-beijing/noldus-insight/issues/114)、[#96](https://github.com/noldus-cn-beijing/noldus-insight/issues/96)、[#55](https://github.com/noldus-cn-beijing/noldus-insight/issues/55)、[#52](https://github.com/noldus-cn-beijing/noldus-insight/issues/52)、[#43](https://github.com/noldus-cn-beijing/noldus-insight/issues/43)（第 3 点）

### 问题描述（用户视角）

用户上传数据后让 agent「帮我分析一下」，分析过程中突然出现一段英文系统消息并终止：

> `[FORCED STOP] Tool inspect_uploaded_file called 5 times — exceeded the per-tool safety limit. All tool_calls stripped. Produce a final text answer now summarizing what to do next...`

用户完全看不懂这段英文，只看到分析「意外停止 / 强制停止」。issue #96 用户甚至怀疑是不是自己「帮我分析一下」的命令太笼统、或卡在分组问题。黑白箱（#52）、明暗箱（#55）、旷场（#43）等不同范式都复现，是**最高频的用户痛点**。

### 问题原因（代码根因）

触发点：`packages/agent/backend/packages/harness/deerflow/agents/middlewares/loop_detection_middleware.py`

**两层检测机制**：
1. **Layer 1（hash-based）**：对每次模型响应的 tool_calls 集合做 hash，滑动窗口内同一 hash 出现 ≥ `hard_limit`(默认 **5**) 次 → 剥离所有 tool_calls，强制纯文本回答。
2. **Layer 2（per-tool-type 频率）**：不看参数，只统计**同一工具名**被调用次数，≥ `tool_freq_hard_limit`(默认 **5**) 次 → 同样强制停止。报错信息 `Tool {name} called {count} times — exceeded the per-tool safety limit` 正是 Layer 2 的 `_TOOL_FREQ_HARD_STOP_MSG`（`loop_detection_middleware.py:183-187`）。

**为什么 `inspect_uploaded_file` / `read_file` 会连调 5 次**：lead 在多文件场景（issue #39/#97 那种几十个 txt）需要逐个 inspect 文件确认范式/列结构；或在列对齐过程中反复 inspect 不同列。这些是**合法的多文件/多列检查**，不是死循环，但 Layer 2 按工具名计数会把它们一概判为「循环」而在第 5 次强杀。

**关键证据**：issue #114 报错信息明确写「**called 5 times**」—— 这正是 `_DEFAULT_TOOL_FREQ_HARD_LIMIT = 5`（`loop_detection_middleware.py:69`）的硬编码默认值。开发者本机已在 `config.yaml` 给 `inspect_uploaded_file` 设了 `hard_limit: 100` 的 override，但该配置**从未进 git、从未部署**（见上文横切根因），线上仍在用默认值 5。

### 建议
- **短期**：把 `loop_detection.tool_freq_overrides.inspect_uploaded_file = {warn: 50, hard_limit: 100}` 提进 `config.example.yaml` 模板 + 部署 SOP，让线上真正生效。同时给 `read_file` 加同款 override（`5c3d1596` 只覆盖了 `inspect_uploaded_file`，`read_file` 仍只走全局 `tool_freq_hard_limit=50`，虽有 Layer 1 的 200 行分桶保护缓解，但频率层无专门 override——#43 第 3 点撞的就是 `read_file` 变体）。
- **中长期**：Layer 2 的「按工具名裸计数」对 inspect/read 这类天然需要多次调用的工具是误判设计；考虑给工具打 `is_iterable` 标记，这类工具只走 Layer 1（hash 去重）不走 Layer 2 频率。或把强杀后的消息本地化为用户能看懂的中文（当前直接把英文 prompt 泄露给终端用户）。

### 衍生：FORCED STOP 会连带让「反馈按钮」消失（#43 第 1 点的因果链）

#43 三症状并非独立——「找不到反馈按钮」(第 1 点) 的根因就是 FORCED STOP (第 3 点)。前端反馈按钮渲染依赖 message→run_id 映射：

- 渲染条件 `message-list-item.tsx:64-66`：`const runId = messageRunIds?.get(message.id); if (!runId) return null;` —— **拿不到 run_id 就静默不渲染反馈按钮**。
- 映射建立 `hooks.ts:404-417`：靠流式 event 捕获 `event.run_id` + `output.id`，前提是 run 正常完成。

FORCED STOP 强制 strip tool_calls 提前终止 run，最终 AIMessage 的 stream event 很可能不携带 run_id 或 output.id → 映射建不起来 → 反馈按钮消失。用户「找不到反馈按钮」不是 UI 没做，是**强终止的 run 拿不到 run_id 绑定**。修复 FORCED STOP（聚类 A）后此症状自然缓解；治本则需让 run_id 绑定不依赖 run 正常完成（run 启动即建占位映射，或后端给每条 AIMessage 持久化 run_id metadata）。

---

## 聚类 B —「上传失败 / 无法上传」（大文件 / 多文件）

**涉及 issue**：[#97](https://github.com/noldus-cn-beijing/noldus-insight/issues/97)、[#39](https://github.com/noldus-cn-beijing/noldus-insight/issues/39)

### 问题描述（用户视角）

- **#97**：lin 上传「19 个 txt 文件，总共 103MB」，点确认后「上传失败」。
- **#39**：lin 上传「同一批 73 个 txt 原始数据」，叫他分析一下，「会上传失败」。
- 共同点：**文件数量多 / 总体积大**时上传失败。

### 问题原因（代码根因）

触发点：`packages/agent/backend/app/gateway/routers/uploads.py`

上传限制的代码默认值（`uploads.py:36-38`）：
```python
DEFAULT_MAX_FILES = 10
DEFAULT_MAX_FILE_SIZE = 50 * 1024 * 1024        # 50 MiB
DEFAULT_MAX_TOTAL_SIZE = 100 * 1024 * 1024       # 100 MiB
```

`upload_files()` 在 `_get_upload_limits()` 读 config，缺失时回退到上述默认；超限直接抛 **HTTP 413**：
- `len(files) > max_files` → `413 "Too many files: maximum is {max_files}"`（`uploads.py:206-207`）
- 累积大小超 `max_total_size` → `413 "Total upload size too large"`（`uploads.py:163-164`）

**对号入座**：
- **#97（19 个 / 103MB）**：同时撞 `max_files=10`（19>10）和 `max_total_size=100MiB`（103>100），任一触发即 413。
- **#39（73 个）**：`73 > 10` 远超文件数上限 → 413。

**为什么是这些值**：`config.example.yaml`（git 模板，代表线上默认）的 `uploads` 段写的就是 `max_files: 10` / `max_total_size: 100MiB`（`config.example.yaml:746-751`）。开发者本机 config.yaml 已放宽到 `max_files: 50` / `max_total_size: 500MiB`，但同样**未进 git 未部署**（见横切根因）。EthoVision XT 一次实验导出几十个 per-subject txt 是常态，10 个文件 / 100MB 的默认上限对真实科研场景严重偏低。

### 建议
- **短期**：把 `max_files: 50` / `max_total_size: 500MiB`（或更高）提进 `config.example.yaml` + 部署 SOP。
- **中长期**：上传超限时的错误信息应**前端友好化**（中文 + 明确告知「当前上限是 N 个文件，你传了 M 个，请分批或联系管理员调高」），而不是让用户只看到一个干瘪的「上传失败」。当前 413 detail 是英文且偏技术。

---

## 聚类 C — 分析逻辑类

### C-1：上传原始数据后仍要求「上传统计学数据」（[#94](https://github.com/noldus-cn-beijing/noldus-insight/issues/94)）

**问题描述**：用户上传了 EthoVision XT 导出的原始轨迹数据，agent 却仍要求用户「上传统计学数据」。

**问题原因（代码根因）**：经核查，**代码中不存在「检测数据是原始/统计后主动要求上传统计数据」的逻辑**。真实链路是：

1. lead 调 `prep_metric_plan` → catalog resolve 检查每个指标所需的 `requires_columns`；
2. 若数据缺列（实验录制时没划分析区，或用户自定义了列名如「中心区」未被自动识别），resolve 抛 `error_code="columns_missing"`（`packages/ethoinsight/ethoinsight/catalog/resolve.py`）；
3. lead 按 prompt 规则向用户转述「数据缺列，请检查实验导出设置」（`prep_metric_plan_tool.py:38-44` 的 `_ERROR_HINTS["columns_missing"]`）。

**根因是沟通错位**：agent 说的是「**缺原始数据列**（需要检查导出/对齐列名）」，用户理解成了「**要统计数据**」。`columns_missing` 的错误提示对非技术用户不够清晰，容易把「缺轨迹列」误读成「要我补统计结果」。issue 标题「要求上传统计学数据」是用户误解，但反映出**错误文案的用户友好度问题**。

**建议**：`columns_missing` 错误消息明确列出「系统期望的原始轨迹列」清单，并区分两种情况——(a)「你的数据有自定义列名，请对齐」（走 column-confirmation）vs (b)「你的导出确实漏了这列」（请重新导出）。避免笼统说「缺列」。

**补充根因视角（identify 失败 status 的 prompt 指引缺口）**：除上述「`columns_missing` 文案沟通错位」，#94 还暴露了 identify 阶段的 prompt 缺口。`identify_ev19_template` 有 5 种返回 status（ok/ambiguous/unknown/unsupported/error），其中 `parse_failed`(error) 和 `unknown` 两路**不给结构化 `clarification_question`**（`identify_ev19_template_tool.py:608-613`），而 lead prompt「硬性反问场景」（`prompt.py:466-471`）只覆盖「范式推断失败/分组无法推断」，**没覆盖「文件解析失败/数据格式不对」该说什么**。全仓搜 `统计学数据/统计数据` 在 lead prompt + 所有 skill 源文件零命中——「请上传统计学数据」是 LLM 在这个空白处**自由发挥的幻觉话术**（把「parse 失败」误读为「该上传别的数据=统计学数据」）。整个 ethoinsight 流水线假设输入永远是原始 EV19 轨迹，系统根本不存在「原始 vs 统计学数据」二分。治本：(1) identify 在 `parse_failed`/`unknown` 路径也返回结构化 `clarification_question`（「文件无法解析为 EV19 格式，请确认导出方式」）让 lead 有现成文案；(2) prompt 补一条覆盖该 status 的反问指引（用正面指令）。注：thread `33d81da4` 在生产无本地轨迹，本结论为静态分析，待生产 trace 坐实。

---

### C-2：旷场中心区列名告知后不采纳（[#43](https://github.com/noldus-cn-beijing/noldus-insight/issues/43) 第 2 点）

**问题描述**：用户做旷场实验（OFT），EthoInsight 说「找不到旷场中心区」。用户把中心区对应的列名告诉了 agent，agent 依然不采纳，反而要求用户重新导出或设置变量名。

**问题原因（代码根因）—— 当前 dev 已修复**：

issue 创建于 2026-05-25，当时的链路存在**三个级联 bug**，已在 **PR#143（2026-06-17 合入 dev）** 全部修复：

1. **列语义对齐通道断链**（`experiment_context.py`）：当时 lead 调 `set_experiment_paradigm(column_semantics={...})` 做独立的列对齐反问时**不带 paradigm 字段**，工具的校验逻辑要求 paradigm 必填 → `column_semantics` **根本没写盘** → prep_metric_plan 读不到列别名 → 自定义列仍被判 `columns_missing` → lead 死循环反问。修复：新增 `column_semantics` **独立写入通道**（`experiment_context.py:469-511`），不依赖 paradigm。
2. **GuardrailMiddleware 误锁**（`ev19_template_provider.py`）：模板锁原本对「非 acknowledge_quality 字段」一律判 `template_already_set` 拦截，把正交的 `column_semantics` 调用也误锁。修复：泛化放行逻辑——「没传 ev19_template / paradigm = 不改模板 = 放行」（`ev19_template_provider.py:62-79`）。
3. **Gate 1 覆盖丢失**：原 Gate 1 从零重建 context 会丢掉已对齐的列语义。修复：改为 `base = dict(existing)` 继承既有字段（`experiment_context.py:583-586`）。

**当前 dev 状态**：三个 bug 均已闭环，列语义对齐链路完整可用（已验证 `experiment_context.py` 中独立通道与 `column_aliases` 派生逻辑存在）。**#43 的这一子问题在当前 dev 不再复现**。

> 注意：#43 还包含「找不到反馈按钮」和聚类 A 的 FORCED STOP 两点，这两点不随 PR#143 解决。

---

## 聚类 D — Agent 行为类（无崩溃 trace，但影响体验）

### D-1：lead 同一 AIMessage 内重复发 tool_calls（[#79](https://github.com/noldus-cn-beijing/noldus-insight/issues/79)）

**问题描述**：lead 在一条 AIMessage 里先输出 thinking，然后发一组 tool_calls（`identify_ev19_template` + `inspect_uploaded_file`），还没等结果就又输出文字 + 再发一组**同样的** tool_calls——即模型在一条消息里把这两个工具各调了两次。报告者判断是 deepseek 在「想好要调工具」和「实际发对 tool_calls」之间的试探性抖动（先发一轮、又补一轮）。

**问题原因（代码根因）**：经核查，**当前系统对「同一 AIMessage 内重复 tool_calls」完全没有任何去重/合并机制**：
- `SubagentLimitMiddleware`（`subagent_limit_middleware.py:41-72`）只截断超出并发数的 `task` 工具，按索引丢弃，不做重复识别；
- `LoopDetectionMiddleware` 的 Layer 1（hash）和 Layer 2（per-tool 频率）都是**消息间**统计，hash 是上一条 vs 本条的整体指纹、频率是滑动窗口跨消息计数，**都无法检测单一 AIMessage 内的重复**；
- `DanglingToolCallMiddleware` 只补缺响应的 tool_call，不涉及去重；
- `tool_call_metadata.py` 是纯元数据同步器，无去重。

**为什么这次没炸**：`identify_ev19_template` 和 `inspect_uploaded_file` 都是**只读幂等**工具（header/sheet 解析，零写入），重复调用结果相同、成本极低。issue 报告者准确识别了这点。

**与 viz race fix 同模式**：2026-05-29 的 `project_2026-05-29_dogfood_viz_race_fix`（同一 AIMessage 并行发 `set_viz_choice`+`task` 致 guardrail 竞态）是**同模式但不同触发点**——那次是「同消息不同工具」的时序竞态（已修：`_current_batch_tool_names()` 读最后 AIMessage 的 tool_calls，同批内有 setter 视为 in-flight 不误拦）；#79 是「同消息同工具重复」，**完全没有对应防护**。

**隐患**：模式上的脆弱点——一旦混入非幂等工具（`write_file` / `task()` / 有副作用工具），同消息重复会写两次文件、派遣两个相同 subagent、token 翻倍。**建议（P2 防御性）**：在 `after_model` 加一步按 `(name, json.dumps(args, sort_keys=True))` 对 `last_msg.tool_calls` 去重，仅保留首次出现，记 WARNING 日志便于诊断 deepseek 抖动。当前不是 bug，属「及时修能省未来麻烦」。

---

### D-2：thinking 过度隐藏，用户希望更多可见反馈（[#82](https://github.com/noldus-cn-beijing/noldus-insight/issues/82)）

**问题描述**：端到端测试中，subagent 的 reasoning 部分被过度藏到 think tab 里，用户看不到。用户期望 agent 有时给一些直接反馈，而不是全部藏进 thinking。

**问题原因**：这不是 bug，是 **deepseek 模型行为 + 产品可见性设计**的权衡。issue 报告者自己也判断「似乎因为 deepseek 本身就是这样，没有什么更好的解决方案」。机制上：模型把中间推理放进 reasoning/thinking 字段，前端默认折叠到 think tab，正文只留最终结论。用户体感是「agent 在黑箱里想、不给我过程反馈」。

**建议**：属产品设计层面——可考虑（a）前端在关键节点（如「正在识别范式…」「正在计算指标…」用 tool_call 事件流而非 thinking 文本给用户可见进度反馈；（b）对部分高价值 reasoning 片段做选择性披露。无代码层 bug 可修。

**补充定位（前端折叠策略 + prompt 矛盾，可工程干预的放大因素）**：

1. **前端默认折叠激进**：`message-group.tsx:59-61` 的 `showLastThinking` 初始值绑定 `isLoading`——subagent 步骤结束 / 回看历史时（isLoading=false）默认折叠；叠加 `ai-elements/reasoning.tsx:82-93` 流式结束后 `setTimeout` 自动 `setIsOpen(false)`。reasoning「流式时可见、一结束就收」是用户「过度隐藏」的直接体感。缓解：subagent 最近一步 reasoning 默认展开、放宽 auto-close。
2. **后端 prompt 自相矛盾**：`lead_agent/prompt.py:580` `<response_style>` 段 `Action-Oriented: Focus on delivering results, not explaining processes`（专注结果不解释过程）**主动鼓励藏过程**，与 `prompt.py:652`（thinking 对用户可见、须用用户语言）矛盾。deepseek 本就倾向把面向用户的话写进 `<think>`，line 580 进一步强化 → 用户连进度反馈都看不到。缓解：改写 line 580 为正面指令（「对关键决策点和进度节点给出简短可见说明，不要把面向用户的反馈写进 thinking」），形成「正文给反馈 + thinking 给推理」双通道（守 memory `feedback_skill_describing_tool_output_enables_hallucination`：deepseek 用正面提示）。

### D-3：subagent 输出可见但 think 看不到，等待时以为卡住（[#158](https://github.com/noldus-cn-beijing/noldus-insight/issues/158)，babysit 监控新发现 2026-06-18）

> 与 #82 同族（前端可见性），但聚焦 subagent 卡片的展开/折叠架构。

**问题描述**：用户能看到 subagent 输出，但看不到 subagent 的 think/reasoning。等 data-analyst 这类 subagent 时，前端无可见进度，用户以为卡住。诉求：不用手动展开「子代理」就能看到 subagent 输出，且能打开折叠 tab 看 think，同时 lead/subagent 的输出和 think 要可区分。

**根因**：subagent 卡片折叠状态 `subtask-card.tsx:56` —— `const [collapsed, setCollapsed] = useState(task.status !== "in_progress")`：**仅 subagent 运行中（in_progress）默认展开，完成（completed）即默认折叠**。subagent 的输出 `task.result`（line 149-155）和 reasoning（嵌套 `ChainOfThoughtContent` line 158-189）都在卡片内，折叠后用户看不到。运行中虽展开，但 reasoning 在嵌套折叠区，用户分不清「这是在跑」还是「卡住」。

与 #82 / `message-group.tsx:59` 同模式：**运行时展开、完成后折叠**，subagent 卡片更激进（连输出都藏在折叠卡里）。

**建议**：(a) subagent 完成后仍默认展示输出摘要（`task.result` 首行 line 132-134 已有 summary，应常驻可见而非折叠）；(b) reasoning 用独立可折叠 tab（默认收起但显式有「查看思考」入口），与输出区分；(c) 运行中给明确的「正在思考…」进度态（区别于卡住）。lead/subagent 用角色徽章区分（subtask-card 已有，保持）。

### D-4：subagent 输出/reasoning 不跟随用户交互语言（[#159](https://github.com/noldus-cn-beijing/noldus-insight/issues/159)，babysit 监控新发现 2026-06-18）

**问题描述**：用户说中文时 subagent 的输出和 reasoning 仍是英文；说英文时偶尔出现中文。

**根因**：subagent prompt **有**语言锁定指引（如 `data_analyst.py:13-20` `<语言>` 段，明确「输出语言必须与用户语言一致；lead 会在 prompt 开头声明语言；未声明则从任务描述推断」）。lead prompt（`prompt.py:655-660`）也要求 lead 派 subagent 时「在 prompt 开头明确声明用户语言」。**但这是两层都靠 LLM 自觉执行，无 harness 强制注入**——`subagents/executor.py` 派发时不注入语言（grep `language/语言` 无注入逻辑）。当 lead 忘了写语言声明句（deepseek 抖动），subagent 回退到「从任务描述推断」，而任务描述常含英文术语/字段名 → subagent 误判用英文输出；reasoning 更易脱缰（deepseek R1 推理默认英文）。

**建议**（治本，强制注入）：lead 的 thread 已锁定用户语言（`prompt.py:648-652` 的首条消息语言检测），应在 `subagents/executor.py` 派发 subagent 时**自动把用户语言注入 subagent 的 system/human prompt 前缀**，而非依赖 lead 手写。这样不依赖 LLM 自觉，从结构上保证 subagent 语言跟随用户（守 memory `feedback_seal_missing_root_cause_is_react_no_toolcall_exit_gate_not_fallback` 思路：靠 prompt 提醒=打地鼠，harness 结构性注入才治本）。

---

## 聚类 E — xlsx in_zone 列匹配 + chart-maker 重试（[#152](https://github.com/noldus-cn-beijing/noldus-insight/issues/152)，babysit 监控新发现 2026-06-17）

> 监控轮（2026-06-17 ~20:45）发现的新 issue。两个子问题：① catalog 层不支持 xlsx 格式的 in_zone 列匹配；② chart-maker 重试时遇到系统错误。箱线图/柱状图未生成。

### E-1：xlsx 的 in_zone 列未被 catalog 匹配（columns_missing）

**问题描述**：用户上传 xlsx 格式数据（含分析区 in_zone 列），catalog 层报该列缺失，箱线图/柱状图无法生成。用户表述为「catalog 层不支持 xlsx 格式的 in_zone 列匹配」。

**根因方向（代码层静态分析，待生产 trace 坐实）**：

经核查，列名归一化 `normalize_column_name`（`ethoinsight/utils.py:151-187`）与列名匹配是**纯字符串正则/glob 操作，与文件格式无关**——xlsx 和 txt 走同一归一化路径（`_parse_header_xlsx` 与 txt 版 `parse_header` 都调用 `normalize_columns`，见 `parse/_core.py:168` vs `_core.py:222`）。所以「不支持 xlsx」字面不成立。

真实根因更可能是 **in_zone 列命名格式未被归一化正则/前缀白名单覆盖**：
- zone 检测依赖精确模式（`utils.py:51-56`）：`In zone(...)` / `分析区中(...)` / `分析区中`，以及 zone 前缀白名单（`utils.py:137-140`）：`开放臂/封闭臂/闭合臂/中央区/Open arms/Closed arms/Center`。
- 若用户 xlsx 导出的 in_zone 列用了**不在此白名单的命名**（如 EthoVision 不同版本/语言环境的 `In zone ×`、`在区域内`、`中心点`、自定义区名直接做列名等），normalize 落到 fallback `_slugify(name)`，**不产生 `in_zone` 前缀** → catalog 的 `in_zone*` glob（如 `ldb.yaml:18` 的 `in_zone_light*`）匹配失败 → `resolve_metrics` 抛 `columns_missing`（`resolve.py:242-253`）。
- 用户恰好用 xlsx，所以表述成「不支持 xlsx」，本质是「该列名格式未被正则覆盖」。

**与 #43/#94 的关系**：同属 Gate 1 列对齐家族——理想路径应走 column-confirmation 闭环（用户确认列映射 → `column_aliases` → resolve 用别名匹配），而非直接报缺失。但若归一化没识别出「这是 zone 列」，column-confirmation 也无从提示对齐。

**建议**：
1. 扩展 `_ZONE_PREFIX_TOKENS` / `_DYNAMIC_PATTERNS` 覆盖更多 EthoVision in_zone 命名变体（需生产 xlsx 真实列名样本）。
2. column-confirmation 在检测到「疑似区归属列但 normalize 后非 in_zone 前缀」时主动反问（守 memory `feedback_oft_single_zone_must_ask_not_guess`）。
3. `columns_missing` 文案明确列出期望的 in_zone 列名模式 + 已检测到的疑似列，引导用户对齐而非笼统报缺。

### E-2：chart-maker 重试遇到系统错误

**问题描述**：chart-maker 在重试时遇到系统错误（具体错误信息未贴）。

**根因方向**：与已知 chart-maker guardrail/prerequisite 问题（`docs/handoffs/2026-06/2026-06-09-chart-maker-prerequisites-fix-handoff.md` 记录的 path_sequence guardrail 误拦）同源——E-1 的列缺失导致 chart 数据文件未生成，chart-maker 重试时前置 handoff（`handoff_code_executor.json` 或 metrics 文件）缺失 → 重试撞 runtime error（`executor.py` 的 `RuntimeError` 路径，line 746/752/1409）。具体系统错误文本未贴，待生产 trace `d58adb40` 坐实。治本在修 E-1（列匹配通了，chart 数据有了，chart-maker 不再空转重试）。

**注**：thread `d58adb40-0d84-44c8-bb36-43f1b76ac059` 在生产，本轮 babysit 无本地轨迹，E-1/E-2 均为代码层静态分析 + 因果推断，待生产对话轨迹/真实 xlsx 列名坐实（守 memory `feedback_grill_handoff_must_be_verified`）。

---

## 其余 open issue（非用户反馈 bug，简述）

| Issue | 类型 | 状态/说明 |
|---|---|---|
| [#98](https://github.com/noldus-cn-beijing/noldus-insight/issues/98) | 方法论 enhancement | 卡行为学同事：范式自定义列的系统化识别/对齐/聚合语义待产出（结构聚合 Sprint 2）。非工程 bug。 |
| [#90](https://github.com/noldus-cn-beijing/noldus-insight/issues/90) | enhancement | 行为学专家待办：填 6 范式 Golden Cases。非工程 bug。 |
| [#82](https://github.com/noldus-cn-beijing/noldus-insight/issues/82) / [#79](https://github.com/noldus-cn-beijing/noldus-insight/issues/79) | 行为观察 | → 见上方「聚类 D」正式调研段（#79 同消息重复 tool_calls 无去重机制；#82 thinking 过度隐藏）。 |
| [#72](https://github.com/noldus-cn-beijing/noldus-insight/issues/72) / [#63](https://github.com/noldus-cn-beijing/noldus-insight/issues/63) | documentation | TST 上线确认项 / 调参指南待补。非 bug。 |
| [#27](https://github.com/noldus-cn-beijing/noldus-insight/issues/27) / [#20](https://github.com/noldus-cn-beijing/noldus-insight/issues/20) | enhancement | 后台记录 thread↔账号关联 / 自然语言改图。功能请求，非 bug。 |

---

## 优先级建议

1. **P0（影响面最大、修复成本最低）**：把 `uploads` 上限和 `loop_detection` 阈值的放宽配置**提交进 `config.example.yaml` 并同步部署 SOP**（横切根因 + 聚类 A + 聚类 B 一次解决）。本机已调好，缺的只是「让它上线」这一步。
2. **P1**：`columns_missing` 错误文案用户友好化（聚类 C-1 的 #94）。
3. **P2**：FORCED STOP 强杀消息本地化为中文，避免英文 prompt 泄露给终端用户（聚类 A 的体验问题）。同 P2：`after_model` 加同消息重复 tool_calls 去重（聚类 D-1 的 #79，防御性，防未来非幂等工具踩雷）。
4. **已闭环**：#43 列名不采纳（PR#143），可考虑在 issue 下评论关闭并指向 PR。
5. **产品设计层（非 bug）**：#82 thinking 过度隐藏——前端用 tool_call 事件流给可见进度反馈，而非依赖 thinking 文本。

---

*调研基线：dev 分支 `a381cb22`（2026-06-17）。工作区 config.yaml 有未提交改动，已在「横切根因」段单独标注。*
