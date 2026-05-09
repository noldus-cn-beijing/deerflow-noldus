# 2026-05-08 EV19 模板识别地基 — 实施完成

## TL;DR

按 spec [docs/superpowers/specs/2026-05-08-ev19-template-skill-foundation-design.md](../specs/2026-05-08-ev19-template-skill-foundation-design.md) 完成 D1-D10：
- 新建 `ethovision-paradigm-knowledge` skill（4 篇 references + 20+20 篇按大类/实验组织的 markdown）
- 新增 `ethoinsight.ev19_facts` 模块（62 变体白名单 + 范式兼容性映射）
- `set_experiment_paradigm` 工具升级 — 加 `ev19_template` 必填参数 + 白名单校验 + 兼容性 warning
- 新增 `Ev19TemplateGuardrailProvider` — 通过 deerflow GuardrailMiddleware 框架拦截 `task("code-executor")` 当 ev19_template 缺失；含锁定逻辑（防中途改模板）
- 删除 lead_agent/prompt.py 旧的「7 大类 18 范式」段，引导 agent 走新 skill
- ethoinsight templates 加 ev19_template 软门（`_gate.py` + shoaling.py 集成示范）
- ethoinsight-planning quality-gates.md 更新为 EV19 模板体系

## 改动清单

| Commit | 内容 |
|---|---|
| `d2dbdbd5` | 新增 ethovision-paradigm-knowledge skill 骨架并搬入 review 包 markdown |
| `7c9e0e02` | 编写 ethovision-paradigm-knowledge SKILL.md 并启用 skill |
| `3105ae29` | 生成 _facts.md（62 变体事实表，人类可读版） |
| `2d9591ec` | 新增 ev19_facts.py: 62 变体白名单 + 范式兼容性映射 + 单元测试 |
| `ba77cf28` | 补充 is_paradigm_template_compatible 测试覆盖 |
| `37848343` | set_experiment_paradigm 工具加 ev19_template 必填参数 + 白名单校验 |
| `1e68d919` | 新增 default-template-fallback.md: 范式→默认 EV19 模板降级表 |
| `553a0094` | 新增 identification-decision-tree.md: agent 决策流程 + 反问质量准则 |
| `a67e69cf` | 新增 Ev19TemplateGuardrailProvider: 拒绝 ev19_template=null 时的 code-executor 派遣 |
| `bda8ad3b` | 在 lead agent 中间件链注册 Ev19TemplateGuardrailProvider |
| `e0e5ec46` | 删除 lead agent prompt 旧的 7 大类 18 范式段，引导走新 EV19 skill |
| `421dbf1e` | ethoinsight templates 加 ev19_template 软门 + shoaling.py 集成示范 |
| `a05f2de1` | Ev19TemplateGuardrailProvider 加 ev19_template 锁定 — 防止中途切换模板 |
| `e332b0b5` | ethoinsight-planning quality-gates.md 引用 ev19_template 字段 |

## 新增文件

| 路径 | 职责 |
|---|---|
| `packages/agent/skills/custom/ethovision-paradigm-knowledge/SKILL.md` | skill 入口：决策树 + 大类索引 + 何时反问指引 |
| `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/_facts.md` | 62 变体事实表（人类可读） |
| `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/identification-decision-tree.md` | agent 决策流程 + 反问质量准则 |
| `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/default-template-fallback.md` | 范式 → 默认变体降级表 |
| `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-template/*.md` | 20 大类知识（从 review 包搬入） |
| `packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-experiment/*.md` | 20 实验范式知识（从 review 包搬入） |
| `packages/ethoinsight/ethoinsight/ev19_facts.py` | Python 模块：62 变体白名单 + 范式兼容性映射 |
| `packages/ethoinsight/ethoinsight/templates/_gate.py` | 软门 helper：require_ev19_template() |
| `packages/agent/backend/packages/harness/deerflow/guardrails/ev19_template_provider.py` | Ev19TemplateGuardrailProvider + Ev19WorkspaceBridgeMiddleware |
| `packages/agent/backend/tests/test_ev19_template_guardrail_provider.py` | provider 单元测试（9 tests） |
| `packages/agent/backend/tests/test_set_experiment_paradigm_ev19.py` | set_experiment_paradigm 工具测试（3 tests） |
| `packages/ethoinsight/tests/test_ev19_facts.py` | 事实表单元测试（14 tests） |
| `packages/ethoinsight/tests/test_template_soft_gate.py` | 软门单元测试（4 tests） |

## 修改文件

| 路径 | 修改点 |
|---|---|
| `packages/agent/config.yaml` | 启用 `guardrails.enabled=true`, `fail_closed=false` |
| `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py` | 注册 GuardrailMiddleware + Ev19TemplateGuardrailProvider |
| `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py` | 删除旧 18 范式表（~85 行），替换为 EV19 skill 引导段 |
| `packages/agent/backend/packages/harness/deerflow/agents/middlewares/experiment_context.py` | set_experiment_paradigm 加 ev19_template 必填 + 白名单校验 |
| `packages/ethoinsight/ethoinsight/templates/shoaling.py` | main() 入口加软门检查 |
| `packages/agent/skills/custom/ethoinsight-planning/references/quality-gates.md` | Gate 0 更新为 EV19 模板识别体系 |

## 测试结果

- Agent backend: 2046 passed, 14 skipped（2 pre-existing failures：extensions_config.json 不存在、prompt 旧字符串已移除）
- Ethoinsight: 30 passed（新测试全部通过）
- 无回归

## 后续 / 不在本次范围（明天起）

- E1: 同事 review PR 进来后，更新 skill `references/by-template/*.md` 和 `references/by-experiment/*.md`
- E2: 6 个 PRD 范式分析模板补全（templates/epm.py 等）— 依赖同事 PR；软门模式已在 shoaling.py + _gate.py 中示范，新模板直接复制
- E3: shoaling golden-case 校验
- E4: 抽象 templates/_base.py 基类

## 已知遗留

- 手工 e2e 未执行（需要启动完整服务 + 准备测试数据）
- LDB 默认变体 (`OpenFieldRectangle-Subdivided2x2`) 是临时兜底，等行为学同事 PR 修正
- LoopDetectionMiddleware 对 ask_clarification 的 hash 区分度未验证（D8d）
- ev19_template 锁定逻辑中的 workspace 解析依赖 ContextVar 桥接（`Ev19WorkspaceBridgeMiddleware`），中间件链顺序已确保 Bridge 在 Guardrail 之前
