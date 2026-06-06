# 2026-06-02 设计 spec — 根治 lead 跳过/冒充探查工具(identify_ev19_template 零调用幻觉)

**类型**:可实施版(已对 dev HEAD + langgraph.log thread `9af3ba6d`/`81051535` + training-data 三重核验;实施前 `git pull` 复核行号)
**对应**:2026-06-02 FST dogfood 中 lead agent 跳过 identify_ev19_template、用 skill 文档内容冒充工具返回值、直接 ask_clarification 反问模板
**估期**:~1-1.5 天(三层:skill 瘦身 + prompt 主体化 + 双兜底机制 + inspect data-preview)
**前置**:无硬前置(与 seal-robustness 阶段2 是独立线,不冲突)

> **目标(用户 2026-06-02 锁定)**:让现有探查工具(identify_ev19_template / inspect_uploaded_file)**被真实调用**,不被 LLM 在 thinking 里脑补冒充。复刻 deerflow 官方让工具被可靠调用的手法,不是新造工具。

---

## 0. 根因(git + log + training-data 三证据钉死,实施前必读)

### 现象
FST dogfood(thread `81051535`):lead 在 thinking 里规划"调 identify_ev19_template",但 output 阶段只产出 `[intent] E2E_FULL_ASKVIZ` + 一句"我先来识别…"的播报文本(**无 tool_call**),然后直接 `ask_clarification` 反问"用 PorsoltCylinder-NoZones 还是 AllZones"。

### 三重证据
| 证据源 | 内容 |
|---|---|
| training-data jsonl 记录 0/1 | `role=lead`,output 是播报文本,**`tool_calls=[]`** —— 规划了调 identify 但没真发起 |
| langgraph.log grep | `identify_ev19_template` 出现 **0 次**、`inspect_uploaded_file` **0 次**;唯一真调的是 `ask_clarification`(`Intercepted clarification`) |
| skill 文件 | `forced_swim.md:15-16` 写了 NoZones/AllZones 两候选 + 推荐;`identification-decision-tree.md` 写了 ambiguous 结构 + 反问话术 |

### 根因链(钉死)
**agent read 了 `ethovision-paradigm-knowledge` skill → 从 `forced_swim.md` 看到候选清单 + ambiguous 反问范式 → 在 thinking 里把"我从 skill 知道有两个候选"冒充成"identify 工具返回了 ambiguous + 两候选" → 直接拿去 ask_clarification。identify 工具全程零调用。**

用户贴的"隐藏步骤"里那个 `status: ambiguous` + 两候选,正是 agent 把 skill 文档冒充成了工具返回值。

### 为何 deerflow 原生 ask_clarification 不被冒充(对照)
官方 `<clarification_system>` prompt 段(prompt.py:379-440,60 行):
1. **只讲"何时必须调/怎么传参",绝不描述工具返回什么** → LLM 无脑补素材
2. 用 `MANDATORY`/`CRITICAL RULE`/`STRICT ENFORCEMENT` 硬词 + 5 条 `❌ DO NOT`(含 **`❌ DO NOT skip clarification for "efficiency"`** —— 官方明确预判并堵了"省事跳过工具"这个失效模式)
3. 全在 **system prompt 主体**,模型每轮都看到

我们的 identify 教学:藏在要 read 的 skill 文件、详述了输出格式、无硬指令强度 —— 三处全输。

### 关键事实(影响方案)
- `identify_ev19_template` / `inspect_uploaded_file` 是 **Noldus 自造工具**,deerflow 上游没有 → prompt/skill 改动**无上游 commit 可拉**,是改我们自己代码,参照 deerflow 写法。
- `ask_clarification` 是 **deerflow 原生**,被 harness 反复打磨。
- **`tool_choice="required"` 在 deepseek/Qwen 上不可靠**(executor.py:712 实证产空 args)→ 不走这条路。
- 现象**不是**"我们的工具写得差所以模型不调",而是"agent 在能直接问用户的工具(ask_clarification)和要它先干活的工具(identify)之间偷懒选了前者"。ask_clarification 被调用恰恰证明模型有能力精准调工具。

---

## 1. 总体方案(三层 + inspect 增强,全部复刻 deerflow 手法)

### 层 1 — skill 瘦身(釜底抽薪,移除脑补素材)
`ethovision-paradigm-knowledge` skill **不再详述 identify 工具的输出格式/候选清单/ambiguous 结构/反问话术**。改成只说"调 identify_ev19_template 拿结果,不要自己列候选"。
- **SSOT 一致性**([[feedback_single_source_of_truth]]):候选模板清单的权威源是 `identify_ev19_template` 工具(它读 `_facts.md`),不该在 skill 的 by-experiment 文档里复制一份让 agent 能抄。
- **这是治根**:没有脑补素材,agent 无法冒充工具返回。

### 层 2 — prompt 主体化(复刻 deerflow `<clarification_system>` 强度)
把 identify 的"必须先调"教学从 skill **提到 system prompt 主体**,新增 `<paradigm_identification_system>` 段,仿官方 `<clarification_system>` 写法。

### 层 3 — 机制双兜底(after_model 主动 + guardrail 被动,用户选"两者都上")
不靠模型自觉,机制层强制。

### 增强 — inspect_uploaded_file 加 data preview(用户最初提案)
现有 `inspect_uploaded_file` 只返回 columns/metadata,不返回数据前几行 → agent 退回 bash 看。补 `data_preview`(前 N 行)字段,作为 hard fact。

---

## 2. 详细设计

### 2.1 层 1 — skill 瘦身

**改动文件**(skills/custom/ethovision-paradigm-knowledge/,**非受保护**):
- `references/by-experiment/forced_swim.md`、`tail_suspension.md` 等:**删除候选清单的"列举式"描述**(NoZones/AllZones 各是什么、推荐哪个)。保留范式级解读知识(那是行为学同事维护的领域知识,不是工具输出格式)。
- `references/identification-decision-tree.md`:**删除 ambiguous 结构 + 反问话术模板**(`status=ambiguous → 用返回的 clarification_question`)。改成"调 identify 工具,按它返回的 status 走"。
- `SKILL.md`:确认引导语是"调工具拿结果",不是"自己判断候选"。

**保留**:范式判读语言、风险点、与其他范式区分(这些是领域知识 SSOT,在 review-packages,[[feedback_ssot_lives_in_review_packages]])。

**判定边界**:删的是"工具会返回什么"(输出格式),保留的是"这个范式怎么解读"(领域知识)。前者制造冒充素材,后者是 data-analyst 真需要的。

### 2.2 层 2 — prompt 主体化(`agents/lead_agent/prompt.py`,**受保护,surgical**)

新增 `<paradigm_identification_system>` 段,放在主 prompt 显眼处(靠近 `<clarification_system>`),仿官方强度:

```
<paradigm_identification_system>
**WORKFLOW: 上传数据 → 先 identify → 再决定反问/派遣**

当本轮 <uploaded_files> 有新数据文件且用户请求分析时,你 MUST 先真实调用
identify_ev19_template(uploaded_files, user_message) 工具,再做任何后续决策。

**MANDATORY**:
- identify_ev19_template 是一个 TOOL,你必须真正发起 tool call,等它返回真实
  status/candidates/clarification_question,再决定下一步。
- 工具返回 status="ambiguous" → 用它返回的 clarification_question 调 ask_clarification
- 工具返回 status="ok" → 用它返回的 ev19_template + paradigm_key 调 set_experiment_paradigm
- 工具返回 status="unknown" → ask_clarification 反问

**绝不允许(用正面指令复述 deepseek 友好)**:
- 你对 EV19 模板候选的所有判断,都必须来自 identify_ev19_template 工具的真实返回值。
- 如果你在 thinking 里想到了候选模板,那也只是假设——你必须调工具确认,工具返回才是事实。
- ask_clarification 反问模板之前,必须先有 identify_ev19_template 的真实工具调用。
- 探查文件结构用 inspect_uploaded_file 工具(它返回 columns + data preview + EV19 metadata),
  不要用 bash 自己看,不要凭文件名猜数据内容。
</paradigm_identification_system>
```

**注意**:用 deepseek 正面提示(CLAUDE.md §6)——不写"禁止脑补",写"判断必须来自工具返回值"。

**派遣硬约束补一条**(prompt.py:244 那段):
- 新增:"反问 EV19 模板前必须有真实 identify_ev19_template 工具调用 — InspectGate 拦截"

### 2.3 层 3 — 机制双兜底

**复刻范式来源**:deerflow 上游 commit **#2135**(`e4f896e9`,`fix(todo-middleware): prevent premature agent exit`)。它的 `after_model + @hook_config(can_jump_to=["model"])` 范式:检测"产出无 tool_call 的回复但工作未完成"→ 注入 reminder HumanMessage + `return {"jump_to": "model", "messages": [reminder]}` 强制再来一轮 + cap 2 次防死循环。

**⚠️ 不能直接 cherry-pick**:#2135 绑在 TodoMiddleware(plan-mode todo 完成度),直接拉会和我们 todo 逻辑冲突。**拉范式,不拉文件**,surgical 适配。

#### 3a. after_model 主动打回(新中间件 / 加到现有 gate)
新 `ParadigmIdentificationGateMiddleware`(或加到现有 GateEnforcement):
```
@hook_config(can_jump_to=["model"])
after_model(state, runtime):
  1. 本轮无上传数据 → return None(纯知识问答不管)
  2. last AIMessage 有 tool_calls 且含 identify_ev19_template → return None(正在调,放行)
  3. last AIMessage 有 tool_calls 但只含 ask_clarification/set_paradigm/task,
     且 messages 里无真实 identify ToolMessage → 注入 reminder + jump_to=model
  4. last AIMessage 无 tool_calls(纯播报想退) 且应 identify 未 identify → 同上
  5. cap:reminder 计数 ≥ 2 → return None(放行,防死循环,兜底交给 guardrail + seal-resume 同理念)
```
reminder 文案(正面指令):
```
<system_reminder>
检测到本轮有上传数据文件,但你尚未真实调用 identify_ev19_template 工具。
你对 EV19 模板的判断必须来自该工具的真实返回值。请现在调用
identify_ev19_template(uploaded_files=..., user_message=...),用它的返回再决定下一步。
</system_reminder>
```

#### 3b. guardrail 被动兜底(复用现有 GuardrailProvider 框架)
新 `InspectGateGuardrailProvider`(抄 `ev19_template_provider.py` 结构):
- 拦 `ask_clarification`
- Bridge Middleware 从 `request.state` 提取 messages + uploaded_files(抄 `Ev19WorkspaceBridgeMiddleware`)
- evaluate:本轮有上传数据 + messages 无真实 identify ToolMessage → DENY
- deny 消息([[feedback_deny_messages_must_direct]],必须含明确指令):
  ```
  检测到你要反问,但本轮有上传数据且尚未真实调用 identify_ev19_template
  (messages 中无其 ToolMessage)。请先调用 identify_ev19_template(uploaded_files,
  user_message),用工具返回的真实 status/clarification_question 再决定反问。
  不要基于推测的模板候选反问。
  ```

#### 两套机制的协调边界(避免重复触发)
- **after_model(3a)先介入**:它在模型产出后、路由前打回,是第一道。多数情况它就把 agent 拉回去调 identify 了。
- **guardrail(3b)是漏网兜底**:若 after_model 因 cap 放行、或 agent 绕过 after_model 直接发 ask_clarification tool_call,guardrail 在 wrap_tool_call 再拦一次。
- **查工具历史的统一模式**(两者共用):扫 messages 找 `ToolMessage(name="identify_ev19_template")`,见 deerflow guardrail provider 范例。
- **fail-open**:state/messages 拿不到时放行(和 ev19 provider 一致),不阻断正常流程。

### 2.4 增强 — inspect_uploaded_file 加 data preview

**改动**:`tools/builtins/inspect_uploaded_file_tool.py`(Noldus 自造,**非上游受保护**)
- 返回结构新增 `data_preview` 字段:前 N 行(默认 5)数据行的真实值
  ```
  "data_preview": {
    "columns": ["Trial time", "X center", "Y center", "Mobility state", ...],
    "rows": [[0.0, 12.3, 45.6, "Mobile", ...], ...],   # 前 5 行
    "n_rows_total": 7501
  }
  ```
- txt(UTF-16)/xlsx/csv 三路径都加(`_inspect_txt`/`_inspect_excel`/`_inspect_csv`)
- **复用现有 parse**(`ethoinsight.parse`),不自己解析

**价值**:agent 调一次 inspect 就拿到列名 + 前几行真实值 + EV19 metadata,无需 bash head,省上下文 + 可复现 + 作为 hard fact。这也配合层 2 prompt 里"用 inspect 不用 bash"的指令。

---

## 3. 实施阶段(建议单独 PR,不和 seal 阶段2 混)

### 一次性做完(改动面适中,无需拆阶段)
1. 层 1 skill 瘦身(改 markdown,无代码风险,先做)
2. 增强 inspect data_preview(独立功能,TDD)
3. 层 2 prompt 主体化(受保护文件 surgical)
4. 层 3a after_model 中间件(复刻 #2135 范式,TDD)
5. 层 3b guardrail provider(抄 ev19_template_provider,TDD)
6. 层 2/3 注册进 lead_agent/agent.py 中间件链(受保护,surgical)

---

## 4. 验收

- [ ] skill 瘦身:`forced_swim.md`/`tail_suspension.md`/`identification-decision-tree.md` 不再列举候选模板清单 + ambiguous 结构;范式领域知识保留
- [ ] prompt 主体新增 `<paradigm_identification_system>` 段,含 MANDATORY + 正面指令"判断必须来自工具返回值"
- [ ] inspect_uploaded_file 返回 data_preview(前 5 行真实值),txt/xlsx/csv 三路径都覆盖,复用 ethoinsight.parse
- [ ] after_model 中间件:有上传数据 + 应 identify 未 identify → jump_to=model + reminder;cap 2 次
- [ ] guardrail provider:拦 ask_clarification,有上传数据 + 无真实 identify ToolMessage → deny;deny 消息含明确指令;fail-open
- [ ] 两套机制不重复触发(after_model 先,guardrail 兜底)
- [ ] **🔴 复现验证(核心,langgraph.log 对照)**:用 FST 数据(thread `81051535` 同款,大鼠强迫游泳 + "帮我分析一下")跑 → identify_ev19_template **被真实调用**(log grep 命中 + training-data tool_calls 含它);agent **不再**用 skill 内容冒充工具返回
- [ ] 单测:after_model 各分支 + guardrail allow/deny + inspect data_preview 三格式 + cap 生效
- [ ] 全量 make test 不退化(改受保护 prompt + 加中间件,必跑全量,[[feedback_pr_merge_must_run_full_suite_on_shared_logic]])

---

## 5. 与其他工作的关系
- **独立于** seal-robustness 阶段2(那个补 data-analyst 数据通路;本 spec 治 lead 探查工具调用)。两条线不冲突,可并行。
- **复刻而非 cherry-pick**:identify/inspect/skill 是 Noldus 自造,无上游 commit;层 3 拉的是 #2135 的 `after_model + jump_to` **范式**,不是文件。
- **受保护文件**:`lead_agent/prompt.py` + `lead_agent/agent.py`(中间件链)→ surgical + 全量测试。

## 6. 不在范围
- ❌ tool_choice="required" 强制(deepseek/Qwen 实测产空 args)
- ❌ 重写 identify_ev19_template / inspect 工具的核心逻辑(只加 data_preview + 不改返回契约)
- ❌ 删 ask_clarification / 改其 deerflow 原生行为
- ❌ 编领域阈值(issue #63 归同事)
- ❌ 改 skill 里行为学同事维护的范式领域知识(只删工具输出格式描述)
