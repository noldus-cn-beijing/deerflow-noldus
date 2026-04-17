# EthoInsight 会话交接文档

> 日期: 2026-04-17
> 上一份交接: `docs/handoffs/2026-04-17-planning-skill-implementation-handoff.md`
> 模型切换: Opus → Sonnet（本文档由 Opus 在会话结束前生成）

---

## 1. 当前任务目标

本次会话完成了以下工作（**均未提交**，等待用户 review）：

1. **修复 2 个 pre-existing 测试** — `test_client.py` 中断言过时问题
2. **创建项目根 CLAUDE.md** — 给 Claude Code 的项目总览文档
3. **brainstorming + plan** — 设计并计划 ethoinsight-planning skill
4. **ethoinsight-planning skill 三层实装** — 完整实现，1546 个测试全部通过

---

## 2. 当前进展

| 编号 | 任务 | 状态 |
|------|------|------|
| 1 | 修复 test_default_params（_subagent_enabled 默认 True）| ✅ |
| 2 | 修复 test_reuses_agent_same_config（config_key 第 4 位改 True）| ✅ |
| 3 | 项目根 CLAUDE.md 创建 | ✅ |
| 4 | ethoinsight-planning skill（SKILL.md + 5 references） | ✅ |
| 5 | extensions_config.json 注册 ethoinsight-planning | ✅ |
| 6 | lead_agent prompt 加入"规划先于派遣"章节 | ✅ |
| 7 | 单元测试（7 个新增，全部通过） | ✅ |
| 8 | 全量 1546 个测试通过，0 失败 | ✅ |
| 9 | 更新 CLAUDE.md 和 docs/roadmap.md | ✅ |
| 10 | **Git 提交** | ❌ 未提交，等用户操作 |
| 11 | **E2E 手动验证** | ❌ 未做，需用户手动测试 |

---

## 3. 本次改动的文件

### 已修改（在 git tracking 中）

| 文件 | 改动 |
|------|------|
| `packages/agent/backend/tests/test_client.py` | `_subagent_enabled` 断言改为 True，`_agent_config_key` 第 4 位改为 True |
| `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` | orchestration_guide 开头加 20 行"规划先于派遣"章节（引导模型调用 ethoinsight-planning skill） |
| `docs/roadmap.md` | M0.1 里程碑加入 ethoinsight-planning 条目 |

### 新增未追踪文件

| 文件 | 内容 |
|------|------|
| `CLAUDE.md` | 项目根 CLAUDE.md（架构、命令、规范总览） |
| `docs/handoffs/2026-04-17-planning-skill-implementation-handoff.md` | 本次实装的详细交接 |
| `docs/handoffs/2026-04-17-roadmap-and-strategy-handoff.md` | 之前会话的 roadmap 交接 |
| `docs/plans/2026-04-17-ethoinsight-planning-skill-implementation.md` | 14 task 实施计划 |
| `docs/plans/2026-04-17-ethoinsight-planning-skill-draft.md` | Skill 初版草案 |
| `packages/agent/backend/tests/test_ethoinsight_planning_skill.py` | 4 个 skill 加载/配置测试 |
| `packages/agent/backend/tests/test_lead_agent_planning_prompt.py` | 3 个 prompt 规划指令测试 |
| `packages/agent/skills/custom/ethoinsight-planning/` | skill 目录（SKILL.md + 5 references/） |

### 未入 git（gitignored）的改动

- `packages/agent/extensions_config.json` — 注册了 `ethoinsight-planning`（`"enabled": true`）

---

## 4. 关键架构决策

### ethoinsight-planning skill 三层架构

```
Layer 1: lead_agent prompt 核心原则（20 行）
  → 让模型"记起要规划"，触发 read_file skill
Layer 2: ethoinsight-planning skill（SKILL.md + 5 references）
  → 6 步规划流程、决策树、各范式模板
Layer 3: write_todos tool（DeerFlow 原生 Plan Mode）
  → 用户手动开启，规划产物的状态机
```

### 设计决策（brainstorming 确定）

| 维度 | 决定 |
|------|------|
| 反问范围 | 仅问：范式推断失败 / 分组无法推断（低召回原则）|
| 计划可见度 | 单行摘要（"将对 <范式> 执行 <操作>，约 X 分钟"）|
| Plan Mode 开启 | 用户手动（不改框架代码）|
| 状态机 | DeerFlow 原生 write_todos |

### 重要更正（上一任交接文档有误）

- `skills/custom/` **不是 gitignored**，这 4 个 skill 都在 git 中
- `extensions_config.json` **是 gitignored**（存储在 packages/agent/.gitignore 中）

---

## 5. 未完成事项

### 高优先级（立即可做）

1. **Git 提交**（建议 3 个 commit）：
   ```bash
   cd /home/qiuyangwang/noldus-insight
   git add packages/agent/backend/tests/test_client.py
   git commit -m "fix: 修复 test_client.py 的 2 个 pre-existing 测试"

   git add packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py \
           packages/agent/backend/tests/test_ethoinsight_planning_skill.py \
           packages/agent/backend/tests/test_lead_agent_planning_prompt.py \
           packages/agent/skills/custom/ethoinsight-planning/
   git commit -m "feat: ethoinsight-planning skill 三层实装"

   git add CLAUDE.md docs/
   git commit -m "docs: 添加项目 CLAUDE.md 和规划 skill 文档"
   ```

2. **E2E 手动验证**（用户自行测试，`make dev` 后测试以下场景）：
   - 场景 1: shoaling 数据 + "帮我分析" → 期望：单行计划摘要 → 流水线执行
   - 场景 2: 无范式文件名 + "分析" → 期望：ask_clarification 询问范式
   - 场景 3: 2 个文件 + 分析 → 期望：小样本量警告 + ask_clarification
   - 场景 4: 无文件 + "什么是 NND" → 期望：直接 knowledge-assistant，不规划

### 中优先级（Phase 0 剩余）

3. **EPM 范式补全** — `templates/epm.py` + `metrics.py` 6 个函数 + `assess.py` 阈值
4. **Open Field 范式补全** — `templates/open_field.py` + 指标 + 阈值
5. **429 重试策略** — `llm_error_handling_middleware.py` 改为 5s/15s/30s
6. **read_file UTF-16 fallback** — `local_sandbox.py:337` 加 BOM 检测

### 低优先级

7. **恢复 noldus-kb** — 等 `180.184.84.124:7001` 恢复后改 `extensions_config.json`
8. **精简 orchestration_guide** — 与 skill 内容有重复，可精简 30%+
9. **pre-existing lint errors** — `prompt.py:207/302` 有 F841 和 E501

---

## 6. 建议接手路径

### 如果要提交本次改动

直接按第 5 节的 commit 命令操作。

### 如果要继续 Phase 0（EPM 范式补全）

```bash
cd /home/qiuyangwang/noldus-insight

# 1. 确认测试状态
source packages/agent/backend/.venv/bin/activate
cd packages/agent/backend && make test 2>&1 | tail -3
# 期望：1546 passed

# 2. 参考现有模板
cat packages/ethoinsight/ethoinsight/templates/shoaling.py

# 3. 查看 EPM 已有代码
grep -n "epm\|open_arm\|closed_arm" packages/ethoinsight/ethoinsight/metrics.py

# 4. 参考行为学需求
cat docs/problems/大鼠高架十字迷宫实验.md
```

### 如果 E2E 发现 GLM-5.1 不遵从 skill

在 `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` 的 `<critical_reminders>` 段落加强提示，例如：
```
- **分析数据前必须规划**: 检测到新上传数据 + 分析请求时，必须先 read_file ethoinsight-planning SKILL.md
```

---

## 7. 风险与注意事项

1. **GLM-5.1 遵从率未验证** — skill 实装后未做 E2E 测试，不清楚模型是否稳定调用
2. **extensions_config.json gitignored** — 部署新环境需手动添加 `"ethoinsight-planning": { "enabled": true }`
3. **pre-existing lint errors** — 两个 errors 在 prompt.py，提交前不影响功能
4. **Plan Mode 默认关** — 用户如需 TodoList 可见需手动切换前端 UI 的 Plan Mode 开关
5. **noldus-kb 仍然禁用** — `extensions_config.json` 中 `noldus-kb` `"enabled": false`，不要改为 true（服务器未恢复）

---

## 8. 下一位 Agent 的第一步建议

1. 读本文档了解全貌
2. 读 [CLAUDE.md](CLAUDE.md) 了解项目架构（本次会话新建的）
3. 读 [docs/roadmap.md](docs/roadmap.md) 了解 Phase 0 待办
4. 根据用户意图选择：
   - 提交 → 直接执行第 5 节的 git commit 命令
   - 继续 Phase 0 → 开始 EPM 范式补全
   - 验证效果 → `make dev` 后手动 E2E 测试
