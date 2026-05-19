# Handoff: Golden-Case 对齐计划 — Task 10 待执行

**生成时间**：2026-04-27
**前一会话主题**：执行 `docs/plans/2026-04-24-golden-case-alignment-plan.md` 的 Task 1-9（subagent-driven-development 模式）
**Git 分支**：`dev`（CLAUDE.md 写的是 `feature/etho-skills`，但实际工作流已迁到 `dev`，本会话所有 commit 都在 `dev` 上）
**最新 commit**：`1b4c9aa`（Task 9）
**模型**：Claude Opus 4.7（带 thinking）

---

## 1. 当前任务目标

执行 [docs/plans/2026-04-24-golden-case-alignment-plan.md](../plans/2026-04-24-golden-case-alignment-plan.md) 中的 **Task 10：端到端验证**。

**完成标准**：
1. 启动 `make dev` 完整应用栈（LangGraph + Gateway + Frontend + Nginx）
2. 用浏览器上传 `golden-cases/case-001-shoaling-baseline/raw-data/` 下 5 个轨迹文件触发 agent 分析
3. 验证 agent 在 planning 阶段会主动追问"处理描述"（Task 6 新增的 ask_clarification）
4. 完成完整流水线：planning → code-executor → data-analyst → report-writer
5. 检查生成的 `report.md`，对照计划 Task 10 表格里 7 个验证项逐条核对
6. 跑 `golden-case` should_not_contain 反例校验，无命中 16 条禁词
7. 写 e2e 文档 `docs/e2e/2026-04-24-斑马鱼鱼群轨迹分析-post-alignment.md`
8. Commit + `make stop`

---

## 2. 当前进展

### Task 1-9 全部完成（subagent-driven-development 流程）

13 个 commit，11 个文件改动。每个 task 都走完了 implementer → spec reviewer → code quality reviewer 三段式审核。

| Commit | Task | 改动概要 |
|---|---|---|
| `1b4c9aa` | 9 | golden-case yaml: outlier metrics 收窄到 [mean_nnd] + should_not_contain 新增 10 条（共 16 条） |
| `363ce05` | 8 | lead_agent/prompt.py 清理 18 处 "APA" 字眼（受保护文件，用户授权改动） |
| `95a41ba` | 8 | report_writer.py 转义 bug 修复 + handoff_schemas/test/lead_prompt:285 删除 references_used |
| `26dd69f` | 8 | report_writer.py system_prompt 重写为 6 段骨架（不再是 APA 论文模板） |
| `c30e5bb` | 7 | quality-gates.md 删除 distance_moved 警告 + 新增 shoaling 专属警告 |
| `4ea1fbe` | 6 | planning/SKILL.md Step 2 新增"处理描述"必问项 + group_semantics 字段 |
| `a058f39` | 5 | apa-reporting-format.md → report.md 重命名 + 整份重写 + SKILL.md:42 引用更新 |
| `31585f2` | 4 fix | effect-size-guide 加入"组间对比"和"不用绝对阈值"框架声明 |
| `1355081` | 4 | effect-size-guide 表型确认段重写（删 activity control 第 2 条） |
| `3f98efc` | 3 | confound-checklist 运动量差异不再作污染源 |
| `7a4d762` | 2 fix | paradigm-interpretation 统一"组间对比"术语（4 处） |
| `f0b6cb7` | 2 | paradigm-interpretation 整份重写（去 6 范式阈值表，扩 shoaling 段） |
| `1d0d1c9` | 1 | ethoinsight/SKILL.md:30 措辞对齐（删"正常范围"引导词） |

### 已通过的回归测试

- `tests/test_subagent_contracts.py` 14 passed（包括 `TestReportWriterHandoffSchema` 的 2 个用例，验证 references_used 删除后无破坏）
- `tests/test_lead_agent_prompt.py` 6 passed（验证 lead_agent prompt.py 18 处 APA 替换无副作用）
- `validate_golden_case.py golden-cases/case-001-shoaling-baseline/` PASS

---

## 3. 关键上下文

### 计划本身的核心目标

把 `golden-cases/case-001-shoaling-baseline/notes.md` 中行为学同事确认的 **6 条 ANSWER 规则**，端到端落实到 4 个 custom skill 文件、report-writer 的 system_prompt、lead_agent prompt 路由表、golden-case 回归 yaml：

1. 不用常模/baseline，组间对比为唯一判据
2. 离群判据用 mean_nnd / 象限分布，不用 distance_moved
3. 不主动建议"排除"个体（排除是生物学判断，不是统计判断）
4. Control/Treatment 裸 label 必须追问具体处理描述
5. Result 段（§3）和 Discussion 段（§4）必须分离
6. IID/NND 在 EthoVision 中是 JS Continuous 自定义变量，raw data 不一定包含

### 用户在会话中明确的关键决策

1. **GLM-5.1 已停用**——不再使用 GLM-5.1，新模型对负面措辞鲁棒。Memory 已更新（[feedback_positive_prompting.md](/home/qiuyangwang/.claude/projects/-home-qiuyangwang/memory/feedback_positive_prompting.md)），原"正面提示"规则降级为可选。Code quality reviewer 在审核中遇到的"GLM-5.1 反向激活"风险全部按 Suggestion 处理不阻塞。
2. **当前工作分支是 `dev`**——CLAUDE.md 写的 `feature/etho-skills` 是过期文档，所有 13 个 commit 都在 `dev` 上。
3. **lead_agent/prompt.py 是 CLAUDE.md 列出的受保护文件**——但用户**明确授权**在 Task 8 范围内修改它（Task 8 的 C1.c 和 I2 修复都动了它）。
4. **Subagent-driven-development 流程**——用户选择此模式而非"我直接干"。每个 task 派 fresh implementer + 两段 review（spec compliance → code quality）。

### 项目级铁律（来自 CLAUDE.md）

- 第 9 条：**"判读哲学：组间比较，不用绝对阈值"**——这是本计划所有改动的统一目标
- 第 8 条：Golden-case 是专家知识注入的正式途径
- 受保护文件清单：`lead_agent/prompt.py`、`subagents/builtins/__init__.py`、`mcp/tools.py`、`sandbox/tools.py`（本计划只动了 `lead_agent/prompt.py`）

---

## 4. 关键发现

### 计划文档的笔误（已记录在 commit 决策中）

1. **Task 2 验证命令笔误**：`grep -c "组间对比" ≥ 2`，但模板里"组间对比"只出现 1 次。Subagent 严格按模板写入正确，后续修复 `7a4d762` 把"组间比较"改"组间对比"统一术语，最终 `grep -c "组间对比" = 4`。
2. **Task 4 grep 验证笔误**：`grep -c "排除运动障碍" = 0`，但新文本里"排除运动障碍"作为反例引用出现 1 次。Subagent 按实际写入，验证 = 1，正确。
3. **Task 7 行号笔误**：计划说"第 30 行"，实际是第 29-30 行。Subagent 用 grep 定位不受影响。
4. **Task 9 数量笔误**：计划说"新增 9 条 = 总 15 条"，实际新增 10 条 = 总 16 条（4 离群越界 + 4 绝对阈值 + 1 APA 论文腔 + 1 distance_moved）。Subagent 按列表写入 16 条，正确。
5. **Task 8 计划范围漏洞**：原计划只让改 `report_writer.py` 的 system_prompt，但 code quality reviewer 发现 `handoff_schemas.py` / `test_subagent_contracts.py` / `lead_agent/prompt.py`（18 处 APA）都需要同步改才能端到端一致。用户授权扩展范围，commit `95a41ba` + `363ce05` 完成。

### 计划自身的 self-review 表

计划 Task 10 列出了 7 个验证项（行 1054-1062），post-alignment 应满足：

| # | 检查项 | 期望改动后行为 |
|---|---|---|
| 1 | 是否追问处理描述 | ✅ planning 阶段追问一次（Task 6 加的 ask_clarification） |
| 2 | 是否用 distance_moved 判离群 | ❌ 只引 mean_nnd（Task 2/3/7/8/9 共同保证） |
| 3 | 是否建议"排除 Subject 3" | ❌ 改为"建议单独标注并检查生物学依据"（Task 5/8 措辞 + Task 9 should_not_contain） |
| 4 | 是否引用"正常范围"或"典型值" | ❌ 不引用（Task 1/2/4 删除阈值表 + Task 9 should_not_contain） |
| 5 | 报告是否有 §3 §4 清晰分离 | ✅ 严格按 6 段骨架（Task 5/8 强约束） |
| 6 | 是否用 APA 句式 "t(10) = 2.34, p = .031" | ❌ 直接列数值（Task 5/8 禁止 APA 句式） |
| 7 | 是否包含 §6 "下一步建议" | ✅ 明确"可考虑的方向"（Task 5/8 6 段骨架第 6 段） |

---

## 5. 未完成事项

### 高优先级 — Task 10 主线

- [ ] **启动应用**：`cd packages/agent && make stop && make dev`（已在 dev 分支）
- [ ] **健康检查**：`curl http://localhost:2026/api/health`
- [ ] **前端交互**：浏览器打开 `http://localhost:2026`，新建 thread
- [ ] **上传数据**：`golden-cases/case-001-shoaling-baseline/raw-data/` 下 5 个 .txt 轨迹文件
- [ ] **触发分析**：发送 "请分析这批斑马鱼 shoaling 数据"
- [ ] **验证 planning 追问**：agent 应在第一次响应里调用 `ask_clarification` 追问"处理描述"
- [ ] **回答追问**：建议沿用 "Control 1,2 / Treatment 3,4,5；为演示分组，无实际药物处理"
- [ ] **流水线推进**：触发完整 code-executor → data-analyst → report-writer
- [ ] **读 report.md**：`backend/.deer-flow/threads/<new-thread-id>/user-data/outputs/report.md`
- [ ] **逐条核对 7 个验证项**（见上面"关键发现"表）
- [ ] **反例 grep 校验**：
  ```bash
  grep -E "建议排除|排除后重新分析|作为离群值排除|将其剔除|正常范围|典型值|高于常模|低于常模|As shown in Figure|总运动距离仅为" \
      <thread>/user-data/outputs/report.md
  ```
  Expected: 0 命中
- [ ] **写 e2e 文档**：`docs/e2e/2026-04-24-斑马鱼鱼群轨迹分析-post-alignment.md`，按 `docs/e2e/斑马鱼鱼群轨迹分析-deepseek-fix.md` 同样格式 + 末尾加"与 deepseek-fix 版本对比"小结表
- [ ] **Commit + `make stop`**：用计划 Task 10 Step 7 提供的 commit message 模板

### 低优先级 — 改进空间（非阻塞）

- 多个 reviewer 提到的 Suggestion 暂未处理（不阻塞 e2e）：
  - paradigm-interpretation.md S3：Y-maze 22.2% 缺简短解释（2/9 推导）
  - paradigm-interpretation.md S4：硬编码行号"第 55-320 行"易失效
  - paradigm-interpretation.md S5：EPM 缺次级指标
  - report.md S5：§3.4 图表语法描述过简
  - planning/SKILL.md S1：通用 label 清单可扩展（vehicle/sham/WT 等）
  - planning/SKILL.md S2：session 内去重缺少"用户改主意"覆盖路径
  - report_writer.py I3：§4 群体指标只覆盖 shoaling，未给非 shoaling 范式的 fallback
  - report_writer.py S1：§6 recommendations 为空时的 fallback 措辞
  - 跨 skill 术语"组间对比"vs"组间比较"统一仅在 paradigm-interpretation.md 内部统一，跨文件（report_writer.py / report.md）仍混用——reviewer 评估这是有意区分（动作 vs 方法论），不阻塞
- 上面这些建议如果 e2e 暴露问题，再单独修；e2e 通过则可推到 v0.1 后

---

## 6. 建议接手路径

### 第一步：先看本会话的最终状态

```bash
cd /home/qiuyangwang/noldus-insight
git log --oneline -15            # 看到 13 个 task commit
git status                        # 应该是 clean
git branch --show-current         # 应该是 dev
```

### 第二步：读关键文档

按这个顺序读，把 context 建起来：

1. **本文档**——你现在读的这份
2. [docs/plans/2026-04-24-golden-case-alignment-plan.md](../plans/2026-04-24-golden-case-alignment-plan.md) Task 10 段（行 1014-1099）——e2e 详细步骤
3. [golden-cases/case-001-shoaling-baseline/notes.md](../../golden-cases/case-001-shoaling-baseline/notes.md)——行为学同事 6 条 ANSWER 规则的原始来源
4. [golden-cases/case-001-shoaling-baseline/expected-analysis.yaml](../../golden-cases/case-001-shoaling-baseline/expected-analysis.yaml)——刚更新的回归锚点
5. [docs/e2e/斑马鱼鱼群轨迹分析-deepseek-fix.md](../e2e/斑马鱼鱼群轨迹分析-deepseek-fix.md)（如果存在）——pre-alignment 版本，作为对比基线

### 第三步：跑 e2e

按"未完成事项"高优先级清单逐项执行。

### 第四步：写 e2e 文档 + 提交

按计划 Task 10 Step 6-7 的模板。

---

## 7. 风险与注意事项

### 容易跑偏的地方

1. **不要切换分支到 `feature/etho-skills`**——CLAUDE.md 写的是过期信息，所有改动在 `dev` 上。如果新 agent 看 CLAUDE.md 后想"切回正确分支"，**那是错的**。13 个 commit 都在 dev，切走会丢上下文。
2. **不要为 GLM-5.1 重写措辞**——GLM-5.1 已停用。Code quality review 中所有"反向激活"相关的 Suggestion 都是基于停用前的旧规则，按当前规则不阻塞。Memory 已更新但不要再改 prompt 文件去掉负面措辞。
3. **不要碰其他受保护文件**——`mcp/tools.py`、`sandbox/tools.py`、`subagents/builtins/__init__.py` 在本次都没动。e2e 跑出问题不要去那些文件里"修"；先看是不是 prompt/skill 配置问题。
4. **report_writer.py 第 171 行的转义**：当前 Python 源码层面是 `\\"`（两个反斜杠+双引号），加载后字符串是 `\"` 两字符（backslash + quote），LLM 看到的是 JSON 转义示范。如果新 agent 看到这行觉得"少了个反斜杠"，**不要改**——这是已经确认正确的状态（commit `95a41ba` 修复后用 importlib + repr 验证过）。
5. **`should_not_contain` 列表是 16 条不是 15 条**——计划文档"新增 9 条"是笔误，实际新增 10 条。e2e 校验时按列表的 16 条全部检查，不要为了对齐计划数字而删一条。

### 已经验证过、不要重复劳动

- 13 个 commit 的 spec + code quality review 全部走完，每个 commit 都对照计划逐字核对过 + 关键测试已跑通
- backend pytest（test_subagent_contracts + test_lead_agent_prompt）已通过
- golden-case schema 校验通过

### e2e 可能遇到的真实问题

1. **NewAPI 网关 thinking 错误**：会话开始时用户提到过的 `400 The content[].thinking in the thinking mode must be passed back to the API` 错误。如果 e2e 触发 LLM 调用时复现这个错误，那是 agent 后端 / NewAPI 网关层的问题，**不是本计划改动引入的**。处理方向：检查 `packages/agent/backend/` 下 `langchain_anthropic:ChatAnthropic` 调用是否正确传递 thinking 块；或临时关闭 thinking 模式跑 e2e 看核心流程是否对。
2. **Agent 不追问"处理描述"**：planning skill SKILL.md Step 2 的 ask_clarification 是 LLM 主动调用（不是硬编码触发），可能存在 LLM 不按 skill 走的情况。如果不追问，Task 6 的改动其实没真正生效，是 e2e 暴露的真问题——记录在 e2e 文档里。
3. **Agent 仍说"建议排除 Subject 3"**：触发 should_not_contain 的"建议排除"条目。也是真问题，记录。

---

## 8. 下一位 Agent 的第一步建议

打开 `/home/qiuyangwang/noldus-insight` 工作目录，运行：

```bash
git log --oneline -15
git status
git branch --show-current
```

确认状态干净（应为 dev 分支、最新 commit `1b4c9aa`、working tree clean）。

然后**直接启动应用**：

```bash
cd /home/qiuyangwang/noldus-insight/packages/agent
make stop 2>&1 | tail -5
make dev 2>&1 | tail -10 &
sleep 15
curl -s http://localhost:2026/api/health 2>&1 | head -3
```

服务起来后**让用户在浏览器里操作前端**（Task 10 是 user-in-the-loop，不是纯 backend 自动化），上传 `golden-cases/case-001-shoaling-baseline/raw-data/` 5 个文件。

观察第一次 agent 响应：是否追问"处理描述"？

- **追问了**：Task 6 改动生效，继续走完整流水线
- **没追问**：第一个真问题，记录在 e2e 文档

后面就照计划 Task 10 表格逐条对比 7 个验证项。

**预计 e2e 时间：30-60 分钟**（agent 流水线 5-10 分钟 + 写 e2e 文档 20-30 分钟）。
