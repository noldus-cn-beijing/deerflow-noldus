# 审批说明：Lead Agent 工作流增强实施计划

**计划文件**: `/home/qiuyangwang/.claude/plans/dynamic-petting-petal-implementation.md`
**日期**: 2026-04-27
**请求**: 审查实施计划，指出遗漏、风险、不一致之处，批准后即可开始实施

---

## 背景

为 EthoInsight 的 lead agent 增加两个能力：

1. **实验类型识别** — 用户上传 EV19 数据后，lead agent 通过两级 `ask_clarification`（先选 7 大类 → 再选 2-6 个细分范式）让用户确认实验类型，而非靠数据推断（raw data 列结构都一样）
2. **Human-in-the-Loop** — 三 Gate 确认关键决策点，通过 `workflow_mode` 开关支持 manual（飞轮期）和 auto（post-training 后）两种模式

## 范围

| 维度 | 数值 |
|------|------|
| 代码增量 | ~470 行 |
| 单测增量 | 18 个 |
| 文件修改 | 13 个（5 新文件，8 修改） |
| 新前端组件 | **0** |
| 新 API 路由 | **0** |
| 实施分层 | 7 Layer，每层独立可测可回滚 |

## 核心设计决策

1. **所有 Gate 复用 `ask_clarification`** — DeerFlow 已有 tool → 中断 → 按钮 → 用户回复闭环，TrainingDataMiddleware 自动录制
2. **`workflow_mode` 照抄 `is_plan_mode` 模式** — config.configurable 注入 → prompt 分支 → middleware 条件启用，零架构风险
3. **Gate 间状态写文件** — `experiment-context.json` 持久化，不被 summarization 丢弃。文件不存在 → None → free-form fallback（老 thread 兼容）
4. **GateEnforcementMiddleware 强制 Gate 顺序** — 不依赖 prompt 遵守，tool call 层拦截
5. **PARADIGMS registry 作单一事实源** — `ethoinsight/templates/__init__.py` 统一管理。基于 62 个 EV19 模板分析得出 18 种行为学范式

## 与设计计划的对应

| 设计（dynamic-petting-petal.md） | 实施计划 Task |
|------|------|
| Layer 1: 单一事实源 | Task 1 |
| Layer 2: 状态存储 | Task 2 |
| Layer 3: Enforcement | Task 3, 3b |
| Layer 4: Prompt | Task 5 |
| Layer 5: 前端 | Task 6, 8 |
| Layer 6: 飞轮回归 | Task 7 |
| Layer 7: E2E 验证 | Task 10, 11 |
| quality-gates.md 更新 | Task 9 |

## 审查重点

请特别关注以下方面：

1. **PARADIGMS 注册表的 18 个范式是否完整？** 对照 EV19 模板列表（62 个），确认没有遗漏常用行为学范式
2. **7 个分类是否合理？** "空间学习记忆迷宫"下有 6 个范式（最多），是否存在更合理的分组方式？
3. **GateEnforcementMiddleware 的放行白名单是否安全？** 允许 `write_file`/`bash`/`ls`/`read_file` 放行，是否会存在绕过路径？
4. **experiment-context.json 的读写竞争** — lead agent 写，code-executor 读，中间没有锁。在 sandbox 本地文件系统下是否可接受？
5. **auto 模式下只保留 Gate 1** — 后续 post-training 后，模型能否稳定走通端到端（这是 v0.1 后的验证项，不是当前范围）
6. **前端 mode-switcher 的 flywheel 选项** — 是否与现有的 flash/thinking/pro/ultra 模式在语义上存在冲突？flywheel 更像 workflow 而非推理模式

## 已知不在此范围的

- 16 种范式的实际模板实现（目前只有 shoaling ready）
- 范式知识（zone 阈值、判读逻辑）— 走 golden-cases 流程
- auto 模式端到端测试 — v0.1 后
- 国际化 — 当前中文为主

## 前置验证

```bash
# 基线测试必须全绿
cd /home/qiuyangwang/noldus-insight/packages/agent/backend && source .venv/bin/activate && make test
cd /home/qiuyangwang/noldus-insight/packages/ethoinsight && uv run pytest tests/ -v
```

---

**批准后下一步**: 按 Task 1-11 顺序实施，推荐 subagent-driven（每 Task 独立 subagent，Task 间 review gate）
