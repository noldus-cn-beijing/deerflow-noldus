# Spec B — 澄清答案 write-through 进工作记忆 + 每轮回灌（resolved facts）

> 日期：2026-06-10 ｜ 状态：待实施 ｜ 层级：harness ｜ 独立于行为学同事方法论阻塞
> 前置 review brief：`docs/handoffs/2026-06/2026-06-10-oft-dogfood-two-failures-seal-replan-review-brief.md`
> 评审：Fable 5（项目目录外）三条判断已吸收——尤其 facts 复用的**三个隔离条件** + SSOT 派生方向
> 主仓库 `/home/wangqiuyang/noldus-insight`，dev HEAD `1d4a9d25`
> harness 根：`packages/agent/backend/packages/harness/deerflow/`（下文 file:line 相对此根）

---

## 0. 一句话目标

lead 在输入对齐阶段连问三轮澄清、**每轮重读输入文件再问下一歧义**（`inspect_uploaded_file` ≈9 次、触发 loop-detection），根因是**已解析答案没有即时持久化、也从不回灌进 system_prompt**。修复 = 复用 DeerFlow **已有的 memory `facts[]` 存储 + `apply_prompt_template` 每轮回灌钩子**，让澄清答案**在回合边界即时 write-through** 并**每轮渲染成独立 `<resolved_task_facts>` 块注入**（自带消费规则）。

**不做**：不新建独立状态文件（复用 facts 通道）；不改退出协议；**不削弱 loop-detection**（它是症状探测器但也是真实安全网，削弱是反模式）。

---

## 1. 根因（已逐字核实，2026-06-10）

持久化机制存在但闭环没闭：

- **状态文件缺字段**：`agents/middlewares/experiment_context.py:385-404` 写 `experiment-context.json` 只含 `paradigm` / `category` / `subject` / `ev19_template` / `gate_completed` / `parameter_overrides` / `column_semantics` 等，**全文 grep `groups` = 0 命中**——不存分组/输入映射这类答案。
- **答案落盘太晚**：`groups.json` 由链路**最后一个工具** `tools/builtins/prep_metric_plan_tool.py:225-262`（:245 `groups_path = ... / "groups.json"`）才写。中间几轮答案**无处即时寄存**。
- **已落盘字段下轮不回灌**：`agents/lead_agent/agent.py:523-540` 读 `read_context` 后**只取 `paradigm`（:532）+ `gate_completed`（:533）**，且 `paradigm` 仅用于 reasoning_effort 降级（:534-538），其余字段读出即丢；`agents/lead_agent/prompt.py` grep `experiment_context` / `resolved_facts` / `<experiment_context` = **0 命中**，无注入。
- **inspect 输出瞬态**：从不持久化，再用只能重调。
- **prompt 缺复用规则**：`lead_agent/prompt.py:455-489` 有"已发 ask_clarification 未答时静默等待"（:489），但**无"历史已答=既定事实，禁止重读文件重新验证、禁止重问"**的复用规则（grep `既定事实`/`已答` 复用类 = 0）。

**与 6-04 TodoMiddleware 修复的关系（已核实，关键）**：`agents/middlewares/todo_middleware.py:54-77` `_is_awaiting_clarification` + :328-329 `after_model` 短路（commit `d3026f31`）**已在位且生效**——它治的是**同一轮内**被催办逼着连续重问（thread `ca86744f` "要不要画图"症状）。本故障 B 是**跨轮重规划**残留，是 6-04 修复**明确未覆盖**的部分。
→ **铁律**：B 不是 6-04 回归，**不许再动 TodoMiddleware / loop-detection** 来"治"B；B 走 write-through + 回灌。

---

## 2. 复用的 DeerFlow 现成机制（已核实，2026-06-10）

**不自造**——以下都已存在：

- **离散事实存储**：`agents/memory/storage.py:62-190` `FileMemoryStorage`，schema 含 `facts: [{id, content, category, confidence, source, createdAt}]`（:26-40），append-only 语义、带 `source` 字段。
- **每轮回灌钩子**：`agents/lead_agent/prompt.py:929-1048` `apply_prompt_template()` **每轮**调 `_get_memory_context()`（:679-715，渲染 `<memory>` 块）+ `_get_prior_corrections_context()`（:718-745，渲染 `<prior_corrections>` 块）注入 system_prompt。**后者是 `_get_resolved_facts_context()` 的现成同形模板**。
- **跨轮累积 reducer**：`agents/thread_state.py:22-96` `Annotated[list, merge_*]`（LangGraph checkpoint 自动持久化），新字段照抄 `merge_artifacts`（:22-29）。
- **中间件 before_model 注入范式**：`agents/middlewares/dynamic_context_middleware.py:206` / `view_image_middleware.py:101-120` 已在做"每轮把动态上下文塞进去"。

---

## 3. Fable 的三个隔离条件（facts 复用不埋坑的前提，必须全补）

复用 facts 的**存储形状 + 注入钩子（机制）**，**不复用**渲染块 + 消费规则（语义）。三条件：

### C1 作用域隔离（防跨实验毒化）
长期记忆 / 历史纠正是**跨会话**的；用户澄清答案绑定**这一次任务**。若 facts[] 不分池，上个实验的"分组是 A/B/C"会毒化下个实验。
- **改法**：澄清答案写入时 `source="user_clarification"` + 绑 `thread_id`（或当前任务 ID）；`_get_resolved_facts_context()` 渲染时**按 scope 过滤**（只取本 thread 的 `user_clarification` facts）。

### C2 独立渲染块 + 独立消费规则（闭环复用规则）
澄清答案渲染成**单独 `<resolved_task_facts>` 块**，**不混进 `<memory>`**——因为附着指令完全不同。块内自带消费规则（正面措辞，CLAUDE.md §6）：
```
<resolved_task_facts>
（以下是本次任务中用户已明确回答的既定事实，按既定事实处理：）
- {fact.content}
...
（规则：这些是本任务的既定事实。直接复用，无需重读输入文件重新验证；
 与你从文件推断的结论冲突时，以此处为准；已答过的不再 ask_clarification 重问。）
</resolved_task_facts>
```
→ **根因里"prompt 缺复用规则"靠注入块自带规则闭环**，光注入不带规则仍会重读。

### C3 改口语义（防矛盾事实并存）
facts[] 是 append-only，但用户会改口（"分组其实是 X/Y"）。旧 fact 不标失效 → 注入块同时出现两条矛盾事实，**比不注入更糟**。
- **改法**：渲染时按 **fact key 做 last-writer-wins**（同一 `key` 取 `createdAt` 最新），或给 facts 加 `supersede` 标记。
- **接上** `project_2026-06-06_context_memory_four_gap_analysis` 的"改口检测→自动移账本"——同一问题，本 spec 落地其 write 侧。

---

## 4. SSOT 派生方向（必须显式定，防两份真相）

澄清答案有两个潜在落点：**facts[]**（注入投影）与**权威上下文文件**（worker 侧消费）。按硬原则 `feedback_single_source_of_truth` 必须指定派生：

- **决定**：lead 写 fact 的那个动作**同时 write-through 到 `experiment-context.json` 对应字段**（字段存在的话，如新增 `groups`）。
- **权威归属**：`experiment-context.json` 仍是 **worker 侧权威**（code-executor / data-analyst 消费）；`facts[]` 定位为**注入投影**（只喂 lead 的下一轮 LLM）。
- **`groups.json` 不变**：仍由 `prep_metric_plan_tool` 写、是 **plan 的派生产物**（catalog resolve 用）；它与"用户答过分组是什么"语义不同——facts/context 存"用户说试验1=实验组"，groups.json 存"plan 用的 {file: group} 映射"，**不是双存**。
- **一个动作、两个落点、一个指定派生**：write fact ⇒ 同时落 context 文件字段 ⇒ facts[] 是投影，context 文件是权威。不留两份真相。

---

## 5. 实施方案

### B1（HARNESS，最高杠杆，必做）— 每轮回灌 `<resolved_task_facts>`
仿 `_get_prior_corrections_context()`（`prompt.py:718-745`）新增 `_get_resolved_facts_context(thread_id)`：
1. 从 `FileMemoryStorage` 读 facts，按 C1 scope 过滤（本 thread + `source=user_clarification`）。
2. 按 C3 last-writer-wins 去矛盾。
3. 渲染成 C2 的 `<resolved_task_facts>` 块（自带消费规则）。
4. `apply_prompt_template()`（:929-1048）调它，注入 `SYSTEM_PROMPT_TEMPLATE`（:1034-1046），和 `<memory>` / `<prior_corrections>` 并列。
- 空 facts 时返回空串（不注入空块）。

### B2（HARNESS / schema，必做）— 澄清答案即时 write-through（边界对账）
**边界事件**：Fable 观察——lead 发 `ask_clarification` 后**下一条用户消息到达**就是"一轮澄清结束"边界，框架可观测（与 Spec A 的"subagent 退出"同构）。
- **落点 1（持久化动作）**：让 lead 把已解析答案落库。两条路选其一（实施者按现有工具面定，**倾向复用而非加新工具**）：
  - **α**：扩 `set_experiment_paradigm`（`experiment_context.py:385-404`）接收 `groups` 等已解析字段 → 同时写 `experiment-context.json`（§4 权威）+ 投影一条 `user_clarification` fact。
  - **β**：加轻量 `set_resolved_fact(key, value)` 工具 → 写 fact + write-through context 字段。
- **落点 2（边界对账，可选加固）**：在边界事件（澄清答案到达）加一层薄对账——"答案刚到 → 检查 fact 是否已写 → 没写就提示 lead 落库"。**与 Spec A 的 `_attempt_auto_seal` 同构**（框架在可观测边界对账规范持久化是否发生）。**MVP 可先只做落点 1**，落点 2 作为 follow-up（避免一次改动面过大）。
- **红线**：write-through 不破 SSOT（§4）；**不双存**（facts 是投影，context 文件是权威，groups.json 是 plan 派生）。

### B3（PROMPT，便宜，必做）— 正向复用规则
`lead_agent/prompt.py:455-489` 澄清段加（正面措辞，CLAUDE.md §6）：
> "历史中用户已回答的澄清是**既定事实**：先读 `<resolved_task_facts>` 块复用，再决定下一步；已答过的信息直接采用，无需重读输入文件去重新验证，也无需重复 ask_clarification 重问。收到新答案立即调持久化工具落库，再处理下一项歧义。"

注意与 :489"已发未答时静默等待"**不冲突**（那条管"问了没答"，这条管"答了别重问"）。

### B4（不做）— 不动 loop-detection / TodoMiddleware
- `loop_detection_middleware.py:171` + `config/loop_detection_config.py:31-35`（`warn_threshold=3`）**不改**——它是症状探测器但也是真实安全网，削弱它是反模式（review brief §B + Fable 确认）。
- TodoMiddleware 6-04 修复（`todo_middleware.py:54-77/328-329`）**已在位、不许再动**（§1 铁律）。
- write-through + 回灌让 lead 不再每轮从头重规划 → loop-detection 自然不再触发，**治本不治标**。

---

## 6. 红 / 绿测试锚点（TDD 强制）

放 `packages/agent/backend/tests/`：

1. **`test_resolved_facts_context_renders_scoped_block`**：写 2 条本 thread `user_clarification` fact + 1 条他 thread/他 source fact → `_get_resolved_facts_context` 只渲染本 thread 2 条（**C1 作用域隔离红锚点**）。
2. **`test_resolved_facts_last_writer_wins_on_conflict`**：同 key 写两次（"分组 A/B" → "分组 X/Y"）→ 渲染只出最新（**C3 改口红锚点**）。
3. **`test_resolved_facts_block_carries_reuse_rule`**：渲染块含"既定事实/无需重读/不重问"消费规则文案（**C2 红锚点**）。
4. **`test_resolved_facts_block_separate_from_memory`**：`<resolved_task_facts>` 与 `<memory>` 是独立块、不混入。
5. **`test_clarification_answer_write_through_dual_landing`**：调持久化工具 → `experiment-context.json` 对应字段写入（权威）+ facts[] 投影一条（**§4 SSOT 派生红锚点**，断言 context 文件是权威、facts 是投影、groups.json 不被本动作写）。
6. **`test_apply_prompt_template_injects_resolved_facts`**：`apply_prompt_template` 输出含 `<resolved_task_facts>`（非空 facts 时）/ 不含（空 facts 时）。
7. **`test_loop_detection_unchanged` / `test_todo_middleware_awaiting_clarification_unchanged`**：B4 不回归——`warn_threshold` 仍 3、`_is_awaiting_clarification` 短路仍在。

**验收项（写进 spec、非备注）**：
- **V1 dogfood OFT**：复现原"三轮重问 + inspect×9"场景 → 修复后**每轮答案即时回灌、lead 不重读输入文件、ask_clarification 各歧义只问一次**、loop-detection 不再触发。
- **V2 SSOT 不破**：grep 确认 facts/context/groups.json 三处无重复真相（facts 投影、context 权威、groups 派生），符 `feedback_single_source_of_truth`。

---

## 7. 实施顺序与验证

1. 起 worktree：`git worktree add <path> -b spec-b-resolved-facts dev`（**显式基于 dev**）。
2. B1（`_get_resolved_facts_context` + apply_prompt_template）→ B2 落点1（持久化 write-through）→ B3（prompt 复用规则）。先红测试再实现。
3. 跑 `make test` **全量**（改了共享 prompt.py / experiment_context.py / 可能的 thread_state，不能只跑新测试，`feedback_pr_merge_must_run_full_suite_on_shared_logic`）。
4. **裸导入验证**（改了 `agents/**`，`feedback_conftest_mock_hides_circular_import`）：
   ```bash
   cd packages/agent/backend
   PYTHONPATH=. python -c "import app.gateway"
   PYTHONPATH=. python -c "from deerflow.agents import make_lead_agent"
   ```
5. 已知 5 个基线红非本改动引入。
6. dogfood OFT（§6 V1）。

---

## 8. 与 Spec A 的关系（同构本质，修复形状不同）

| | 正确机制（已存在） | prompt 指向的错误终结动作 | harness 只事后检测的层 | 修复形状 |
|---|---|---|---|---|
| **A（提交结果）** | seal 工具 + 原子写 manifest | write_file / 输出 gate_signals 叙述 | `_validate_handoff_emitted`（事后报错） | 退出边界对账 + 从文件重建 |
| **B（保存状态）** | memory facts + apply_prompt_template 回灌 | clarify-first / 从对话历史重推导 | loop-detection（事后告警） | 回合边界 write-through + **回灌** |

**Fable 精确化（采纳）**：
- **write 侧同一原语**：框架在**它能观测的边界**对账规范持久化是否发生、没发生就确定性补上。A 边界=进程退出；B 边界=问答回合（ask_clarification 后下条用户消息到达）。
- **read 侧只 B 需要**：A 消费者是框架机器（不需回灌）；B 消费者是下一轮 LLM（必须注入回灌）。→ 这是两修复形状差异的**唯一**来源。
- **"同构"是诊断透镜、不是单一解法**；但 write 侧确有统一原语（边界对账），不止"两个不相干修复"。
- **不造 `BoundaryReconciler` 基类**：n=2 抽象早熟，共享一个概念即可（合"智能在节点内、约束在节点间"）。

详见 `docs/superpowers/specs/2026-06-10-spec-a-code-executor-autoseal.md`。
