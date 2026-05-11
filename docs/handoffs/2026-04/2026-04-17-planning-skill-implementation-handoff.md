# EthoInsight 会话交接文档

> 日期: 2026-04-17
> 上一份交接: `docs/handoffs/2026-04-17-roadmap-and-strategy-handoff.md`
> 本次任务: ethoinsight-planning skill 实装

---

## 1. 当前任务目标

本次会话完成了两件事：

1. **修复 2 个 pre-existing 测试** — `tests/test_client.py` 中 `TestClientInit::test_default_params` 和 `TestEnsureAgent::test_reuses_agent_same_config`（DeerFlowClient 默认参数变化导致的断言过时）
2. **实装 ethoinsight-planning skill**（三层协同的数据分析任务规划机制）

---

## 2. 当前进展

| 编号 | 任务 | 状态 |
|------|------|------|
| 1 | 修复 test_default_params（_subagent_enabled 默认 True）| ✅ |
| 2 | 修复 test_reuses_agent_same_config（config_key 第 4 位改 True）| ✅ |
| 3 | 项目根创建 CLAUDE.md | ✅ |
| 4 | Brainstorming 收敛规划 skill 设计方向 | ✅ |
| 5 | Plan 写入 `docs/plans/2026-04-17-ethoinsight-planning-skill-implementation.md` | ✅ |
| 6 | 创建 SKILL.md + 5 个 references 文件 | ✅ |
| 7 | 注册到 `extensions_config.json` | ✅ |
| 8 | lead_agent prompt 加入"规划先于派遣"章节 | ✅ |
| 9 | 单元测试（skill 加载 4 个 + prompt 3 个） | ✅ |
| 10 | 全量 1546 个测试通过 | ✅ |
| 11 | 更新 CLAUDE.md 和 docs/roadmap.md | ✅ |
| 12 | E2E 手动验证（4 个场景） | ❌ **用户自行测试** |

**分支**: `feature/etho-skills`（未提交，等待用户 review 后提交）

---

## 3. 本次改动的文件

### 修改（在 git 中的文件）

| 文件 | 改动 |
|------|------|
| [packages/agent/backend/tests/test_client.py](packages/agent/backend/tests/test_client.py) | `_subagent_enabled` 断言改为 True，`_agent_config_key` 第 4 位改为 True |
| [packages/agent/extensions_config.json](packages/agent/extensions_config.json) | 注册 `ethoinsight-planning` skill（enabled=true）。**注意：此文件 gitignored，改动不会入 git。部署时需手动同步。** |
| [packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py](packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py) | 在 `orchestration_guide` 开头加 20 行"规划先于派遣"章节 |
| [CLAUDE.md](CLAUDE.md) | skills/custom 注解从 3 个 skill 更新为 4 个 |
| [docs/roadmap.md](docs/roadmap.md) | M0.1 里程碑加入 ethoinsight-planning skill 条目 |

### 新增（在 git 中的文件）

| 文件 | 内容 |
|------|------|
| [CLAUDE.md](CLAUDE.md) | 项目根 CLAUDE.md（本次会话新建） |
| [docs/plans/2026-04-17-ethoinsight-planning-skill-draft.md](docs/plans/2026-04-17-ethoinsight-planning-skill-draft.md) | Skill 初版草案（brainstorming 前） |
| [docs/plans/2026-04-17-ethoinsight-planning-skill-implementation.md](docs/plans/2026-04-17-ethoinsight-planning-skill-implementation.md) | 14 task 实施计划 |
| [packages/agent/backend/tests/test_ethoinsight_planning_skill.py](packages/agent/backend/tests/test_ethoinsight_planning_skill.py) | 4 个 skill 加载/配置测试 |
| [packages/agent/backend/tests/test_lead_agent_planning_prompt.py](packages/agent/backend/tests/test_lead_agent_planning_prompt.py) | 3 个 prompt 规划指令测试 |

### 新增（**在 git 中** — 上一任交接误标为 gitignored，实际 skills/custom/ 进 git）

```
packages/agent/skills/custom/ethoinsight-planning/
├── SKILL.md                                 # 6 步规划流程主文档
└── references/
    ├── intent-classification.md             # 意图分类决策树
    ├── planning-templates.md                # 各范式规划模板
    ├── design-type-keywords.md              # 实验设计关键词表
    ├── quality-gates.md                     # 5 个质量门控点
    └── failure-recovery.md                  # 5 种失败降级策略
```

**重要更正**：与上一任交接文档不同，`skills/custom/` **不是** gitignored。`ethoinsight`、`ethoinsight-analysis` 也在 git 中（可用 `git ls-files packages/agent/skills/custom/` 确认）。本次新增的 `ethoinsight-planning/` 也应当 commit 入 git。

**gitignored 的是**：
- `packages/agent/extensions_config.json`（注意：这意味着 Task 8 的改动不会被 git 跟踪——下次克隆仓库的人需要手动配置）
- `packages/agent/config.yaml`

---

## 4. 关键架构决策（brainstorming 收敛结果）

### 三层协同架构

```
Layer 1: Prompt 核心原则（20 行，硬编码在 prompt.py）
  ↓ 引导模型调用
Layer 2: ethoinsight-planning Skill（SKILL.md + 5 references/）
  ↓ 规划产出
Layer 3: write_todos Tool（DeerFlow 原生，用户手动开启 Plan Mode 激活）
```

### 7 个关键设计决定

| 维度 | 决定 |
|------|------|
| 核心角色 | Lead agent 的思考框架（主）+ 用户透明契约（辅）+ 可追踪状态机（辅） |
| 思考形式 | 固定模板 + 检查清单 + 缺信息反问（human-in-the-loop） |
| 反问范围 | **低召回——仅问必须** |
| 规划前必问 | 范式推断失败、分组无法推断（仅这 2 项） |
| 执行中必问 | 小样本量（<3/组）、数据质量警告、执行失败 |
| 计划可见度 | **单行摘要**（"将对 <范式> 数据执行 <操作>，约 X 分钟"） |
| 状态机实现 | DeerFlow 原生 `write_todos`（用户手动开启 Plan Mode） |

### 为什么不改 DeerFlow 框架代码自动开启 Plan Mode

为减少对"受保护文件"（[client.py](packages/agent/backend/packages/harness/deerflow/client.py)）的修改，Plan Mode 保持用户手动开启。Phase 1 微调模型上线后可重新评估。

---

## 5. 验证结果

### 单元测试

| 测试文件 | 通过数 |
|---------|--------|
| `test_ethoinsight_planning_skill.py` | 4/4 |
| `test_lead_agent_planning_prompt.py` | 3/3 |
| `test_client.py`（修复后） | 129/129 |
| **全量测试** | **1546 passed, 14 skipped, 0 failed** |

### Lint 检查

`make lint` 有 2 个 errors，**都是 pre-existing**（与本次改动无关）：
- `prompt.py:207` F841 `direct_execution_example` unused variable
- `prompt.py:302` E501 line too long (246 > 240)

这些在本次改动之前就存在，不阻塞提交。

---

## 6. 未完成事项（用户需自行操作）

### 高优先级

1. **E2E 手动验证 4 个场景**（详见 plan 文档 Task 14 之后的验收章节）：
   - 场景 1: 标准端到端分析（shoaling 或 EPM 数据 + "帮我分析"）
   - 场景 2: 范式推断失败（文件名无范式信息 + "分析一下"）
   - 场景 3: 小样本量提醒（2 个文件 + 分析请求）
   - 场景 4: 纯知识问答（无文件 + "什么是 NND？"）
   - 场景 5（可选）: 手动开启 Plan Mode 验证 TodoList 可见

2. **Git 提交**：
   - `git add` 5 个修改的文件 + 新增的 2 个测试文件 + 新增的 CLAUDE.md + 2 个 plan 文档 + 本交接文档
   - 不要 add skills/custom/（会被 .gitignore 忽略，但 habit）
   - 建议 commit 拆分：
     - commit 1: "fix: 修复 test_client.py 2 个 pre-existing 测试"
     - commit 2: "feat: ethoinsight-planning skill 三层实装"
     - commit 3: "docs: 添加项目根 CLAUDE.md 和规划 skill 文档"

### 中优先级

3. **观察 GLM-5.1 对 skill 的遵从率** — 如果 E2E 验证发现模型经常不调用 skill，需要在 prompt 里加更强的指令（如 reminder 中强调）

4. **精简 `orchestration_guide` 其他段落** — 现在 prompt.py:864-941 的派遣流程详情有大量与 skill 重复的内容，可以精简 30%+（plan 中 Task 13 提到）

### 低优先级

5. **补全更多范式模板**（EPM/OFT/novel_object 等）到 `references/planning-templates.md`

---

## 7. 建议接手路径

### 若要 E2E 验证

```bash
cd /home/qiuyangwang/noldus-insight/packages/agent
make dev
# 打开 localhost:2026
# 按 plan 文档 Task 14 之后的 4 个场景测试
```

### 若 E2E 发现问题（模型不遵从 skill）

1. 查看 agent 日志确认模型是否调用了 `read_file("/mnt/skills/ethoinsight-planning/SKILL.md")`
2. 若没调用 → 加强 prompt 中的 reminder
3. 若调用但输出不符合模板 → 在 SKILL.md 中加更具体的输出示例

### 若要提交

```bash
cd /home/qiuyangwang/noldus-insight
git add packages/agent/backend/tests/test_client.py
git commit -m "fix: 修复 test_client.py 的 2 个 pre-existing 测试"

# 注: packages/agent/extensions_config.json 是 gitignored，不会被 add
git add packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py \
        packages/agent/backend/tests/test_ethoinsight_planning_skill.py \
        packages/agent/backend/tests/test_lead_agent_planning_prompt.py \
        packages/agent/skills/custom/ethoinsight-planning/
git commit -m "feat: ethoinsight-planning skill 三层实装"

git add CLAUDE.md docs/
git commit -m "docs: 添加项目 CLAUDE.md 和规划 skill 设计/实施文档"
```

---

## 8. 风险与注意事项

1. **GLM-5.1 是否稳定调用 skill 未验证** — 本次实装纯代码/配置，未做 E2E 测试。GLM-5.1 对"必须调用 X"类指令的遵从率是未知数。如果遵从率低，需要微调 prompt。

2. **skill 文件 gitignored** — `ethoinsight-planning/` 整个目录不在 git 中。部署/换机时需要手动恢复或从其他环境拷贝。

3. **`orchestration_guide` 现在有内容重复** — prompt.py 中既有"规划先于派遣"又有"Step 0-4 派遣流程详情"，部分重复。建议后续迭代精简。

4. **Plan Mode 默认仍是关** — 用户如果想看 TodoList，需要手动在前端 UI 切换 Plan Mode。本次实装没改变这个默认行为（有意为之，减少框架改动）。

5. **规划本身会增加延迟** — 每次分析前多一轮 LLM 输出（规划摘要），约 +3-5s。对研究员用户影响不大，但要监控。

6. **Pre-existing lint errors** — `prompt.py:207/302` 有 2 个 errors，与本次改动无关。建议作为 TODO 另找时间修。

---

## 9. 下一位 Agent 的第一步建议

1. 读本文档了解全貌
2. 若要继续 Phase 0 其他任务（EPM 范式、OFT 范式、429 重试），参考 [docs/roadmap.md](docs/roadmap.md) 的 M0.2-M0.4
3. 若要优化规划 skill，先跑 E2E 验证收集数据，再决定改进方向
4. 若要提交本次改动，按第 7 节的 3 个 commit 拆分
