# Prompt.py 英文 System Prompt 中文化 + file_pattern 修复方案

## Context

**问题 1（输出语言）**: GLM-5.1 的输出语言受 system prompt 语言强烈影响。当前 `prompt.py` 中大量英文段落导致前端输出混杂英文。

**问题 2（file_pattern 丢失前缀）**: E2E 测试中 lead agent 派遣 code-executor 时，传递了 `file_pattern="/mnt/user-data/uploads/.txt"`（丢失了 `轨迹*` 前缀），导致 code-executor 无法定位数据文件，用完 8 轮 max_turns 仍未完成分析。

### 问题 2 的根因分析

排查对比了 568e6c2c（最后成功 E2E）和 5fb824d3（当前）的 prompt.py diff：

- orchestration_guide 中 Step 1 的文件路径校验指令**完全一样**（正确/错误示例对比都有）
- code_executor.py 配置也**完全没变**
- **真正的变化**：c24e575c 起引入了大量新 Noldus 规则（共享 workspace 机制、契约表、实验设计类型识别），noldus_rules 从 ~25 行膨胀到 ~80 行，总 prompt 从 39KB 增长到 43KB
- 这些新规则与 subagent_section 中仍保留的**英文 DeerFlow 框架指令**（DECOMPOSE/DELEGATE/SYNTHESIZE、CRITICAL WORKFLOW、VIOLATION 等）形成了**中英混杂的双层指令**
- GLM-5.1 在处理更长、中英混杂的 prompt 时，对 orchestration_guide 中 file_pattern 模板的遵循度下降
- 这不是确定性 bug，是概率性退化——prompt 越长越复杂，GLM 对具体格式要求的注意力越分散

**结论**：prompt 中文化 + file_pattern 强化可以一起解决。

### 中文化对其他模型的影响

| 模型类型 | 中文 prompt | 备注 |
|---------|-----------|------|
| GLM-5.1 (当前) | 显著改善 | 输出语言一致，指令遵循度提升 |
| Qwen 系列 (微调方向) | 中性偏好 | 中英文理解力相当，不影响逻辑 |
| Claude/GPT-4 | 无影响 | 顶级模型多语言能力完全够 |
| 英文开源小模型 (<14B) | 可能轻微下降 | 但不在当前技术路线内 |

中文化影响的是**输出语言一致性**，不影响逻辑理解能力。顶级模型和 Qwen 系列都能同等处理中英文指令。

**目标文件**: `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`

---

## 改写原则

1. **语义完全一致** — 只改语言，不改逻辑
2. **保留 XML tag 名称为英文** — `<subagent_system>`、`<role>` 等不改
3. **保留模板变量** — `{n}`、`{available_subagents}` 等不改
4. **保留代码示例中的代码** — `task()`、`ask_clarification()` 等函数调用不改
5. **用正面指令** — GLM 对"禁止X"会反向激活，改为"请做Y"

---

## 改动计划

### 第一批（高优先级）— 直接影响 GLM 输出语言和 file_pattern

#### 1.1 `_build_subagent_section()` 英文段落 (line 192-415)

| 行号 | 当前内容 | 改为 |
|------|---------|------|
| 192 | `general-purpose: For ANY non-trivial task...` | `general-purpose: 通用型——网络调研、代码探索、文件操作、分析等` |
| 194 | `bash: For command execution...` | `bash: 命令执行——git、构建、测试、部署等操作` |
| 287 | `🚀 SUBAGENT MODE ACTIVE - DECOMPOSE, DELEGATE, SYNTHESIZE` | `🚀 子代理模式已启用 — 分解、委派、整合` |
| 289-293 | 角色描述 (task orchestrator, DECOMPOSE/DELEGATE/SYNTHESIZE) | 中文：你是**任务调度员**：1.分解 2.委派 3.整合 |
| 294 | CORE PRINCIPLE | 核心原则：复杂任务应分解为多个子任务，分发给子代理并行执行 |
| 296-306 | HARD CONCURRENCY LIMIT 说明 | 中文化：并发硬限制说明，保留 `{n}` 变量 |
| 308 | Available Subagents | 可用子代理 |
| 311 | Your Orchestration Strategy | 你的调度策略 |
| 313 | DECOMPOSE + PARALLEL EXECUTION (Preferred Approach) | 分解 + 并行执行（首选方案） |
| 333-343 | USE Subagents when / Execute directly when | 何时使用子代理 / 何时自己执行 |
| 345-357 | CRITICAL WORKFLOW + VIOLATION | 关键工作流 + 违规说明 |
| 359-363 | How It Works | 运行机制 |
| 411 | CRITICAL 最后的提醒 | 已部分中文，补齐剩余英文 |

#### 1.2 `SYSTEM_PROMPT_TEMPLATE` 开头 (line 418-433)

| 行号 | 当前内容 | 改为 |
|------|---------|------|
| 420 | `You are {agent_name}, an open-source super agent.` | `你是 {agent_name}，一个开源超级代理。` |
| 428 | Think concisely and strategically... | 简洁且有策略地思考用户请求，行动前先思考 |
| 429 | Break down the task... | 拆解任务：哪些明确？哪些模糊？哪些缺失？ |
| 430 | PRIORITY CHECK... | **优先检查：如有不清晰、缺失或多义之处，必须先澄清——请先提问再开始工作** |
| 431 | `{subagent_thinking}` | 不改（模板变量） |
| 432 | Never write down your full final answer... | 思考过程中只列提纲，请将完整回答写在正式回复中 |
| 433 | CRITICAL: After thinking... | 关键：思考后必须提供正式回复。思考用于规划，回复用于交付。 |
| 434 | Your response must contain the actual answer... | 你的回复必须包含实际答案，请直接给出结果 |

#### 1.3 强化 file_pattern 校验（orchestration_guide Step 1）

**位置**: `apply_prompt_template()` 中的 `orchestration_guide`，约 line 830-840

**当前写法**（声明式，GLM 容易忽略）：
```
**CRITICAL: 文件路径必须使用正确的 glob 模式！**
- 正确: `/mnt/user-data/uploads/轨迹*.txt`
- 错误: `/mnt/user-data/uploads/.txt` （丢失了文件名前缀）
```

**改为**（正面指令 + 提取步骤）：
```
**文件路径构造方法**（必须按以下步骤执行）：
1. 从 <uploaded_files> 中读取完整文件名（如 "轨迹-Shoaling...Subject 1.txt"）
2. 提取文件名公共前缀（如 "轨迹"）
3. 构造 glob 模式：/mnt/user-data/uploads/<公共前缀>*.<扩展名>
4. 示例：文件名 "轨迹-Shoaling...Subject 1.txt" → /mnt/user-data/uploads/轨迹*.txt
5. 示例：文件名 "Subject 1-Trial 1.csv" → /mnt/user-data/uploads/Subject*.csv

请确认你构造的文件路径包含文件名前缀和 * 通配符。
```

同时将 **prompt 格式要求**中的 `文件路径: /mnt/user-data/uploads/<文件前缀>*.<扩展名>` 改为更具体的：
```
文件路径: /mnt/user-data/uploads/轨迹*.txt  ← 必须包含文件名前缀，从 uploaded_files 提取
```

### 第二批（中优先级）

#### 2.1 `<clarification_system>` 英文部分 (line 435-498)

| 行号 | 当前内容 | 改为 |
|------|---------|------|
| 437 | WORKFLOW PRIORITY: CLARIFY → PLAN → ACT | 工作流优先级：澄清 → 计划 → 行动 |
| 438-440 | FIRST/SECOND/THIRD 步骤说明 | 第一步/第二步/第三步 中文化 |
| 441 | CRITICAL RULE... | 关键规则：澄清永远在行动之前。请先确认需求再开始工作。 |
| 443 | MANDATORY Clarification Scenarios... | 必须澄清的场景——以下情况必须在开始工作前调用 ask_clarification |
| 445-467 | 5 个场景说明 | 中文化描述和 REQUIRED ACTION |
| 475-483 | How to Use 代码示例 | 使用方法（保留代码，注释改中文） |

#### 2.2 `<working_directory>` (line 506-521)

| 行号 | 当前内容 | 改为 |
|------|---------|------|
| 507-509 | User uploads / workspace / outputs 说明 | 用户上传 / 工作目录 / 输出文件 中文化 |
| 511 | File Management | 文件管理 |
| 512-519 | 文件管理规则 | 中文化 |

#### 2.3 `<response_style>` (line 523-527)

| 行号 | 当前内容 | 改为 |
|------|---------|------|
| 524 | Clear and Concise... | 简洁明了：除非用户要求，请保持简洁格式 |
| 525 | Natural Tone... | 自然语气：默认使用段落和散文，请按需使用列表 |
| 526 | Action-Oriented... | 行动导向：专注于交付结果，请直接展示成果 |

#### 2.4 `<citations>` (line 529-590)

| 行号 | 当前内容 | 改为 |
|------|---------|------|
| 530 | CRITICAL: Always include citations... | 关键：使用网络搜索结果时必须标注引用 |
| 532-541 | When to Use / Format / Placement / Sources | 中文化说明 |
| 543-590 | 示例和规则 | 中文化（保留 markdown 格式和代码示例） |

#### 2.5 `<critical_reminders>` (line 594-604)

| 行号 | 当前内容 | 改为 |
|------|---------|------|
| 595 | Clarification First... | **澄清优先**：请先确认不清晰/缺失/模糊的需求，再开始工作 |
| 596 | `{subagent_reminder}` | 不改（模板变量） |
| 596 | Skill First... | 技能优先：执行**复杂**任务前先加载相关技能 |
| 597 | Progressive Loading... | 渐进加载：按技能引用逐步加载资源 |
| 598 | Output Files... | 输出文件：最终交付物必须放在 `/mnt/user-data/outputs` |
| 599 | Clarity... | 简洁直接，请减少不必要的元叙述 |
| 600 | Including Images and Mermaid... | 图片和 Mermaid：欢迎在 Markdown 中使用图片和流程图 |
| 601 | Multi-task... | 并行调用：请善用并行工具调用以提升效率 |
| 602 | Language Consistency... | 语言一致：请与用户使用相同语言 |
| 603 | Always Respond... | 必须回复：思考是内部过程，你必须在思考后提供可见的回复 |

### 第三批（apply_prompt_template 中的备选英文）

#### 3.1 非 Noldus 场景的 subagent_reminder (line 792-796)

改为中文：**调度模式**：你是任务调度员——将复杂任务分解为并行子任务。**硬限制：每轮最多 {n} 个 `task` 调用。**

#### 3.2 非 Noldus 场景的 subagent_thinking (line 807-810)

改为中文：**分解检查：当前任务能否拆分为 2 个以上并行子任务？如果可以，请计数。**

#### 3.3 `_build_skill_evolution_section()` (line 154-164)

改为中文：技能自进化说明

---

## 不需要改的段落（确认）

- Noldus 调度规则 `noldus_rules` (line 208-285) — 已是中文 ✅
- Noldus subagent 描述 `noldus_descriptions` (line 182-186) — 已是中文 ✅
- `orchestration_guide` (line 818-893) — 已是中文（Step 1 的 file_pattern 部分需要强化） ✅
- 中文澄清示例 (line 486-497) — 已是中文 ✅
- 代码示例中的函数名/变量名 — 保留英文 ✅
- XML tag 名称 — 保留英文 ✅

---

## 验证步骤

1. **回归测试**: `cd packages/agent/backend && make test` — 确认所有测试通过
2. **启动服务**: `cd /home/qiuyangwang/noldus-insight && make dev`
3. **E2E 测试**: 上传 Shoaling 数据 → 确认 code-executor 收到正确的 file_pattern（含文件名前缀 `轨迹*`）
4. **前端验证**: 确认输出是中文
5. **检查模板变量**: 确认 `{n}`、`{available_subagents}` 等变量未被破坏
6. **多次测试**: 跑 3 次 E2E 确认 file_pattern 稳定正确（排除概率性失败）

---

## 风险

- prompt.py 是受保护文件，改动越大下次上游同步冲突越大（但中文化是必要的）
- 正面指令：GLM 对"不要做X"会反向激活，所有"DO NOT"/"NEVER"改为正面表述"请做Y"
- 对其他模型无负面影响：中文化只影响输出语言一致性，Qwen/Claude/GPT 的逻辑理解不受影响
