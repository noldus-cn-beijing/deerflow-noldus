# Handoff — EPM dogfood 失败诊断 + column_aliases/loopdetection 修复实施

> 日期：2026-06-06 → 2026-06-08 ｜ 分支：`dev`（主仓库）+ `worktree-zone-concept-params-loopdetection-fix`（实施分支）
> 这是给下一位 AI Agent 的交接，不是给用户的总结。

---

## 1. 本会话任务链

从「EPM Xuhui 28 文件 dogfood」出发，完成了一条完整链路：

**诊断 → spec 编写 → Opus review（2 轮）→ spec 修订 → 实施 → Opus review 实施 → 次要问题修复**

核心成果：**column_aliases → compute 参数注入通用化 + LoopDetection subagent 配置修复已完整实施，等待合入 dev。**

---

## 2. 本会话已完成

1. ✅ **EPM dogfood（28 文件 Xuhui 数据）** — Playwright 操作全流程，发现全部 140 条指标 value=None
2. ✅ **根因诊断** — Opus agent 分析所有证据（training JSONL + checkpoints + catalog + resolve + metrics + executor），判定 **100% 工程 bug，不需等行为学专家**
3. ✅ **Spec 编写** — `docs/superpowers/specs/2026-06-06-zone-concept-params-and-loopdetection-fix-spec.md`
4. ✅ **Opus spec review 第 1 轮** — NO-GO，发现 3 个 P0（closed_arm_zones 不存在于 metrics 层 / loader 不解析 / null default 被拒）+ 2 个 P1
5. ✅ **Spec v2 修订** — 修复全部 P0/P1
6. ✅ **Opus spec review 第 2 轮** — GO-WITH-FIXES（仅剩签名/声明对齐的局部修订）
7. ✅ **Spec v3 定稿** — 按 metric 精确声明参数子集 + loader 健壮校验 + 完整验收标准
8. ✅ **Worktree 实施** — `worktree-zone-concept-params-loopdetection-fix`（commit `ca61303d`），11 文件 468 行
9. ✅ **Opus 实施 review** — **GO**。全部测试绿：ethoinsight 722/0 + loop 67/0 + guardrail 96/0 + executor 150/0
10. ✅ **次要问题修复** — prompt 引号示例 + 恒真断言（commit `cff6a518`）

---

## 3. 两个根因（Bug A + Bug B）

### Bug A：column_aliases → compute 参数注入断在 EPM

`_build_zone_aliases_overrides`（resolve.py）只在范式有 `anonymous_zone_override` 时工作。EPM 无此字段 → 永远返回 `{}` → `parameters_in_use={}` → compute 脚本拿不到 zone 列名 → autodiscovery 正则也匹配不到用户列 `open`/`closed` → 全部 None。

**修复**：新增范式级 `zone_concept_params` YAML 字段（概念→参数显式映射），`_build_zone_aliases_overrides` 改为从 `zone_concept_params` 读取映射（新增），`anonymous_zone_override.target_param` 继续作为 fallback（保留）。EPM 的 `open_arm_entry_ratio`/`total_entry_count` 需关闭列 info → 给 `_get_closed_zone_cols`/`compute_total_entry_count`/`compute_open_arm_entry_ratio` 补 `closed_arm_zones` 参数。

### Bug B：LoopDetection 裸构造掐死 subagent

`executor.py:655` 用 `LoopDetectionMiddleware()` 裸构造 → 模块常量 `tool_freq_hard_limit=5`。140 次合法 bash → 第 5 次 hard-stop。主 agent 走 `from_config` 拿到 50，subagent 没拿到。

**修复**：subagent 局部传参 `LoopDetectionMiddleware(tool_freq_warn=30, tool_freq_hard_limit=50)`，**不动全局常量**（3/5 是 Noldus fork 对 lead agent 的有意防护）。+ batch bash prompt 降 bash 次数（140→~5-8）。

---

## 4. 当前仓库状态

- **dev HEAD**：`b440b1d5`（已含 PR #103 column_aliases 修复）
- **实施分支**：`worktree-zone-concept-params-loopdetection-fix`，2 commits：
  - `ca61303d` — 完整实施（11 文件，468 行）
  - `cff6a518` — 次要问题修复（prompt 引号 + 恒真断言）
- **远程**：已推送到 GitHub `origin/worktree-zone-concept-params-loopdetection-fix`
- **Worktree**：`/home/wangqiuyang/noldus-insight/.claude/worktrees/zone-concept-params-loopdetection-fix`（保留）

---

## 5. 下一步（按优先级）

### 立即（PR 合入）

1. **为 `worktree-zone-concept-params-loopdetection-fix` 建 PR 合入 dev**
   - 分支领先 dev 1 个实施 commit + 1 个修复 commit
   - 所有测试绿，Opus 审查 GO
   - 合入后跑一次全量 `pytest` 确认

2. **重新 dogfood EPM 28 文件**
   - `make dev` 跑新代码
   - 上传 `/home/wangqiuyang/DemoData/real_data/Raw data-EPM-Xuhui-28/` 下 28 个 xlsx
   - 确认指标 value 非 null → data-analyst 可消费 → 端到端通

### 后续（不阻塞）

3. **P2：解 checkpoints msgpack 确认 Bug B 根因** — Opus 建议解一条真实 EPM dogfood transcript，确认是 `FORCED_STOP` 还是 seal prompt 黑洞，最终定性后再定兜底力度。关联 memory：`feedback_subagent_seal_deadlock_is_prompt_not_budget.md`

4. **前端内部术语包装** — `[intent] E2E_FULL_ASKVIZ` 等后端枚举值裸露给用户，加前端映射表包装为人类可读文案

5. **S5 Experiment Log** — 方法论 §8 标"设计已就绪经 Opus review 待实施"

### 阻塞（等行为学专家）

6. **Issue #98** — Layer 1/2/3 概念清单 + zone 空间重叠性（决定 OR vs sum 聚合语义）
7. **#90 Golden Cases** / **#72 TST** / **#63 调参指南**

---

## 6. 关键文件（下一位 agent 必读）

| 文件 | 说明 |
|------|------|
| `docs/superpowers/specs/2026-06-06-zone-concept-params-and-loopdetection-fix-spec.md` | 实施 spec（v3 定稿，Opus 2 轮 review 通过） |
| `packages/ethoinsight/ethoinsight/catalog/resolve.py` | `_build_zone_aliases_overrides` 多 concept 路由（核心改动） |
| `packages/ethoinsight/ethoinsight/catalog/epm.yaml` | `zone_concept_params` + metric 参数声明 |
| `packages/ethoinsight/ethoinsight/catalog/schema.py` | `ZoneConceptParam` dataclass |
| `packages/ethoinsight/ethoinsight/catalog/loader.py` | `zone_concept_params` 解析 + CatalogError |
| `packages/ethoinsight/ethoinsight/metrics/epm.py` | `closed_arm_zones` 参数补齐 |
| `packages/ethoinsight/tests/test_column_semantics.py` | `TestEPMZoneConceptParams` + `TestLoaderZoneConceptParams` |
| `packages/agent/backend/packages/harness/deerflow/subagents/executor.py` | L655 局部传参 30/50 |
| `packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py` | batch bash prompt |
| `packages/agent/backend/packages/harness/deerflow/agents/middlewares/loop_detection_middleware.py` | docstring 修正（不改常量） |
| `docs/design/2026-06-06-data-processing-methodology-design.md` | 方法论三层框架，Issue #98 关联 |

---

## 7. 重要决策记录

- **100% 工程 bug，不等行为学专家**：EPM 的概念清单/参数名/requires_columns 全部已在 catalog 里写死自洽，坏的是 plumbing。专家只挡 Layer 3（多子区聚合），与本次 1:1 场景正交。
- **zone_concept_params 显式声明而非 convention 推导**：显式 > 隐式，可审计，不脆弱。OFT/LDB/Zero Maze 有 azo 的不动，EPM 新增。
- **LoopDetection 局部传参不动全局常量**：3/5 是 Noldus fork 对 lead agent 的有意防护，subagent 只需自己宽松。
- **YAML 参数声明严格对齐函数签名**：`open_arm_time_ratio`/`open_arm_time`/`open_arm_entry_count` 只声明 `open_arm_zones`；`open_arm_entry_ratio`/`total_entry_count` 声明两者。通过 `_compute_parameters_in_use` 的 replace-only 机制保证不泄漏。
- **default 用空字符串 `""` 不用 null**：loader 的 ParamSpec 校验拒绝 None。

---

## 8. 测试命令

```bash
# ethoinsight 全量
cd packages/ethoinsight && uv run pytest tests/ -q

# 新增 column_semantics 测试
uv run pytest tests/test_column_semantics.py -v

# loop detection
cd packages/agent/backend && DEER_FLOW_CONFIG_PATH=/home/wangqiuyang/noldus-insight/packages/agent/config.yaml uv run pytest tests/test_loop_detection_middleware.py -q

# 后端全量
cd packages/agent/backend && make test
```

---

## 9. 下一位 Agent 的第一步建议

1. 读 spec：`docs/superpowers/specs/2026-06-06-zone-concept-params-and-loopdetection-fix-spec.md`
2. 进 worktree：`cd /home/wangqiuyang/noldus-insight/.claude/worktrees/zone-concept-params-loopdetection-fix`
3. 确认测试全绿：ethoinsight 全量 + loop detection + guardrail + executor
4. 建 PR 合入 dev
5. 重新 dogfood EPM 28 文件验证端到端
