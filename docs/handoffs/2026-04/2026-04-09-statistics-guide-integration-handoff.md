# EthoInsight 统计方法指南集成 — 交接文档

> 日期: 2026-04-09
> 上一份交接: `docs/handoffs/2026-04-09-llm-strategy-review-handoff.md`

---

## 1. 当前任务目标

**问题**: 行为学同事提供了统计方法文档，需要将其整合进 EthoInsight agent 系统，提升所有 subagent 选择统计方法的能力。

**预期产出**: (A) 改 agent 的 skill/prompt 让 agent 知道怎么选统计方法；(B) 扩展 `statistics.py` 加入缺失的统计检验方法。

**完成标准**: 所有 subagent 能根据实验设计类型选择正确的统计方法，`statistics.py` 覆盖 Levene、Hedges' g、Omega-squared、Tukey HSD。

---

## 2. 当前进展

### 全部完成 ✅

| 编号 | 任务 | 状态 |
|------|------|------|
| 1 | ethoinsight SKILL.md 新增"统计方法选择指南" | ✅ |
| 2 | statistics.py 新增 `test_homogeneity()` (Levene) | ✅ |
| 3 | statistics.py 新增 `compute_hedges_g()` | ✅ |
| 4 | statistics.py 新增 `compute_omega_squared()` | ✅ |
| 5 | `compare_two_groups()` 增强：Levene → independent-t/Welch-t 分支 + hedges_g 输出 | ✅ |
| 6 | `compare_groups()` 增强：方差齐 → Tukey HSD 事后检验 + omega_squared | ✅ |
| 7 | code-executor system prompt 新增"统计方法选择"指南 | ✅ |
| 8 | data-analyst system prompt 新增 Step 4"统计方法验证" | ✅ |
| 9 | lead agent prompt 新增"识别实验设计类型"路由规则 | ✅ |
| 10 | report-writer system prompt 新增方法选择说明模板 | ✅ |
| 11 | 全量测试回归验证 — 32/32 passed | ✅ |

---

## 3. 关键上下文

### 同事提供的统计方法文档

路径: `/home/qiuyangwang/noldus-insight/docs/EthoInsight-技术文档/动物行为学常用统计检验方法.md`

内容: 370 行，涵盖差异比较（两组/多组/重复测量/分类数据）、回归分析、聚类分析、关联与相关、生存分析、降维，附有范式×方法速查表、正态性检验、效应量。

**核心价值**: 指导 agent 的统计方法决策逻辑（什么实验设计 → 什么方法），不是让我们手动实现所有方法。

### 改动的文件

| 文件 | 改动类型 |
|------|---------|
| `packages/agent/skills/custom/ethoinsight/SKILL.md` | 新增 ~70 行：决策流程、事后比较选择、范式×方法速查、效应量补充 |
| `packages/ethoinsight/ethoinsight/statistics.py` | 新增 3 个函数 + 增强 2 个现有函数 |
| `packages/ethoinsight/tests/test_statistics.py` | 新增 13 个测试用例（32 total） |
| `packages/agent/backend/.../code_executor.py` | system_prompt 新增统计方法选择 section |
| `packages/agent/backend/.../data_analyst.py` | workflow 新增 Step 4 方法验证 + principles 新增方法学把关 |
| `packages/agent/backend/.../lead_agent/prompt.py` | noldus_rules 新增设计类型识别表 |
| `packages/agent/backend/.../report_writer.py` | workflow + formatting 新增方法选择说明 |

### statistics.py 新增函数

| 函数 | 功能 | 返回 |
|------|------|------|
| `test_homogeneity(*groups)` | Levene 方差齐性检验 | `{"is_homogeneous", "p_value", "statistic"}` |
| `compute_hedges_g(g1, g2)` | 小样本偏差校正 Cohen's d | `{"g", "magnitude"}` |
| `compute_omega_squared(groups)` | 无偏 ANOVA 效应量 | `{"omega_squared", "magnitude"}` |

### statistics.py 增强的函数

| 函数 | 变化 |
|------|------|
| `compare_two_groups()` | 正态+非配对时先 Levene → 方差齐用 `independent-t-test`(equal_var=True)，不齐用 `welch-t-test`；输出新增 `effect_size_hedges_g` 和 `variance_homogeneity` |
| `compare_multiple_groups()` | 新增 `variance_homogeneity` 和 `effect_size_omega_squared` 输出 |
| `compare_groups()` (dispatcher) | ANOVA 显著+方差齐 → Tukey HSD 事后检验；否则 fallback 到 pairwise |

### 设计文档仍在此处

`~/.gstack/projects/noldus-cn-beijing-noldus-insight/qiuyangwang-feature-etho-skills-design-20260409-151642.md`

---

## 4. 关键发现

### 同事文档 vs spec 表的关系

- 同事给的是**通用统计方法选择指南**（什么情况用什么方法）
- spec 表 (`docs/specs/paradigm-analysis-tools-spec.md`) 需要的是**范式特定信息**（每个范式的指标、EthoVision 列名、参考阈值、图表选择）
- **spec 表仍然是空的**，需要同事另外填写
- 统计方法文档已经转化为 skill 知识 + statistics.py 增强

### noldus-kb 是外部只读服务

- `http://180.184.84.124:7001/mcp` — 无法从代码端添加文档
- 统计方法文档无法直接导入 noldus-kb
- 已通过 SKILL.md 注入解决（skill 内容会注入所有 subagent prompt）

### test_used 字段变化（向后兼容注意）

- 以前: 正态两组总是返回 `"welch-t-test"`
- 现在: 方差齐返回 `"independent-t-test"`，不齐返回 `"welch-t-test"`
- 所有现有测试已通过，但如果有其他代码硬编码检查 `test_used == "welch-t-test"`，需要注意

### Tukey HSD 依赖 scipy >= 1.8

- `scipy.stats.tukey_hsd` 在 scipy 1.8 引入
- 有 try/except fallback：如果不可用，退回到 pairwise 比较
- 当前环境 scipy 版本已支持

---

## 5. 未完成事项

### 高优先级

1. **提交代码** — 所有改动尚未 git commit（用户未要求提交）
2. **CEO 评审剩余部分** — /plan-ceo-review Sections 1-11 未完成（从上一份交接继承）
3. **Phase 0c 实现** — 稳定 agent 架构、E2E 测试（设计评审后的下一步）

### 中优先级

4. **spec 表填写** — `docs/specs/paradigm-analysis-tools-spec.md` 仍空，需要行为学同事填写范式特定信息
5. **`run_paradigm_analysis` 工具实现** — code-executor redesign handoff 中的设计目标（单一工具替代 get_analysis_template + 8 步 checklist）
6. **更多模板** — 目前只有 `shoaling.py`，OFT/EPM/MWM 模板还没写

### 低优先级

7. **RM-ANOVA 实现** — statistics.py 中未实现 RM-ANOVA（需要 pingouin 或 statsmodels），当前靠 code-executor 在生成代码时自行使用
8. **Games-Howell 事后检验** — 方差不齐时的最佳选择，需要 pingouin.pairwise_gameshowell
9. **noldus-kb 质量负责人** — Pre-Phase 1 交付物

---

## 6. 建议接手路径

### 如果要继续评审流程

```bash
# 读取设计文档
cat ~/.gstack/projects/noldus-cn-beijing-noldus-insight/qiuyangwang-feature-etho-skills-design-20260409-151642.md

# 读取 CEO 评审日志
export PATH="$HOME/.bun/bin:$PATH"
~/.claude/skills/gstack/bin/gstack-review-read
```

从 /plan-ceo-review Section 1 开始。

### 如果要开始 Phase 0c 实现

```bash
cd /home/qiuyangwang/noldus-insight

# 查看改动状态
git status

# 查看当前 agent 配置
cat packages/agent/config.yaml | head -30

# 运行统计测试确认一切正常
cd packages/ethoinsight && uv run python -m pytest tests/test_statistics.py -v

# 查看 code-executor redesign 交接
cat docs/handoffs/2026-04-09-code-executor-redesign-handoff.md
```

Phase 0c 的下一步:
1. 先提交本次改动
2. 定义 E2E 测试用例（基于 DeerFlowClient + pytest）
3. 实现 `run_paradigm_analysis` 工具
4. 让 OFT/EPM/MWM 三个范式端到端跑通

---

## 7. 风险与注意事项

1. **代码尚未提交**: 所有改动在工作区中，还没有 git commit。接手后根据用户意愿决定是否提交。

2. **用户偏好正面指令**: GLM-5.1 对"禁止X"会反向激活（memory: `feedback_positive_prompting.md`）。

3. **用户是工程师思维**: 更关注"能不能跑通"而非评审文档完善度。

4. **CWD 问题**: 用户的 CWD 是 `/home/qiuyangwang`，不是 `/home/qiuyangwang/noldus-insight`。ethoinsight 的 pytest 需要在 `packages/ethoinsight/` 目录下用 `uv run python -m pytest` 运行。

5. **bun 路径**: gstack 工具需要 `export PATH="$HOME/.bun/bin:$PATH"`。

---

## 下一位 Agent 的第一步建议

1. 读取本文档了解全貌
2. 问用户: 先提交代码，还是继续做下一步？
3. 如果提交: `cd /home/qiuyangwang/noldus-insight && git add -A && git status` 看改动范围
4. 如果继续开发: 读取 `docs/handoffs/2026-04-09-code-executor-redesign-handoff.md`，开始 Phase 0c（`run_paradigm_analysis` 工具实现）
