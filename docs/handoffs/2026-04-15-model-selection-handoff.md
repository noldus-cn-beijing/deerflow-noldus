# EthoInsight 会话交接文档

> 日期: 2026-04-15
> 上一份交接: `docs/handoffs/2026-04-14-upstream-sync-handoff.md`

---

## 1. 当前任务目标

**本次会话完成了四件事**：

1. **诊断 E2E 测试"卡住"问题** — 找到两个根因并修复
2. **确认 subagent_enabled 默认值** — 被 linter 还原为 False，已改回 True
3. **前端 hydration mismatch 修复** — CommandDialog 的 DialogHeader 位置问题
4. **微调选型和数据采集规划** — 确定 Qwen3-8B + Fireworks.ai，写了完整数据 checklist

---

## 2. 当前进展

| 编号 | 任务 | 状态 |
|------|------|------|
| 1 | 诊断 gateway WatchFiles 重启问题 | ✅ `serve.sh` 修复 reload-exclude glob |
| 2 | 修复 `subagent_enabled` 默认值为 True | ✅ 4 处已改 |
| 3 | 修复前端 hydration mismatch | ✅ `command.tsx` DialogHeader 移入 DialogContent |
| 4 | E2E 测试验证 | ✅ 流水线跑通，code-executor → data-analyst → report-writer |
| 5 | 微调基座模型选型 | ✅ 确定 Qwen3-8B (Dense, Apache 2.0) |
| 6 | 微调平台选型 | ✅ 确定 Fireworks.ai (LoRA, <16B 免费) |
| 7 | 微调数据采集 checklist | ✅ `docs/plans/2026-04-15-fine-tuning-data-checklist.md` |
| 8 | 架构图文档 | ✅ `docs/architecture-diagram.md` (Mermaid，双层) |
| 9 | 提交代码 | ❌ 未提交（含上次 43 个文件 + 本次修改） |
| 10 | 429 限流重试策略优化 | ❌ 未开始 |

---

## 3. 本次改动的文件

### 3.1 Bug 修复

| 文件 | 改动 |
|------|------|
| `packages/agent/scripts/serve.sh:138` | `--reload-exclude='sandbox/'` → `'sandbox/*'`，`--reload-exclude='.deer-flow/'` → `'.deer-flow/*'`。修复 uvicorn WatchFiles 的 fnmatch glob 不匹配子目录文件的问题 |
| `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py:257,285` | `subagent_enabled` 默认值从 `False` 改为 `True` |
| `packages/agent/backend/packages/harness/deerflow/client.py:120,226` | 同上，`subagent_enabled` 默认值改为 `True` |
| `packages/agent/frontend/src/components/ui/command.tsx:46-59` | `DialogHeader` 从 `DialogContent` 外移到内部，修复 Radix useId() hydration mismatch |

### 3.2 新增文档

| 文件 | 用途 |
|------|------|
| `docs/architecture-diagram.md` | 双层架构图（非技术人员 + 技术人员），含 Mermaid 图和数据流水线详解 |
| `docs/plans/2026-04-15-fine-tuning-data-checklist.md` | 微调数据采集完整 checklist，A-H 八个来源，含时间线和交付物清单 |

---

## 4. 关键架构决策

### 微调方案确定

| 决策项 | 选择 | 原因 |
|--------|------|------|
| 基座模型 | **Qwen3-8B** (Dense) | Tool calling 原生最强(BFCL排名靠前)，中文+代码双强(HumanEval 76.0)，Apache 2.0，微调生态最成熟 |
| 训练平台 | **Fireworks.ai** | <16B 模型免费微调，支持 LoRA + DPO + thinking traces，100 个 LoRA 同时部署不加钱 |
| 训练策略 | **SFT → DPO** | SFT ~1800 条(带 CoT thinking traces) → DPO ~300 对(专家偏好标注) |
| 数据格式 | **JSONL + `<think>` 标签** | Fireworks 原生支持 Qwen3 thinking traces 训练 |
| 推理显卡 | **RTX 4090 (24GB)** | BF16 全精度 ~16GB，微调 LoRA ~16GB，一卡双用 |

### 排除的模型及原因

| 模型 | 排除原因 |
|------|---------|
| MiniMax 系列 | 无 7B-14B 开源 dense 模型，全是 200B+ MoE |
| Kimi/月之暗面 | 无开源小模型 |
| DeepSeek-R1-Distill | Tool calling 不稳定（原生不支持），7B/14B 已 deprecated |
| GLM-4-9B | Agent 能力弱于 Qwen（智谱 0414 更新明确说 9B 只优化了批处理） |
| Qwen3.5-9B | MoE 架构，LoRA 微调不成熟，Fireworks 明确不支持 Qwen3 MoE 微调 |

---

## 5. 关键发现

### E2E 测试结果

两次 E2E 测试都跑通了 subagent 流水线（code-executor → data-analyst → report-writer），但 GLM-5.1 API 稳定性是主要瓶颈：
- **429 Rate Limit**: 密集出现在 subagent 完成后 lead agent 做总结时
- **500 Internal Server Error**: 偶发，约每次运行 2-3 次
- **180s 超时**: subagent 执行期间偶发
- 当前重试策略（1s/2s/放弃）对 GLM 限流窗口太激进

### LoRA + CoT 微调策略

DeepSeek-R1-0528-Qwen3-8B 的实验证明：用大模型的 CoT 推理数据微调 Qwen3-8B，AIME 上超过 Qwen3-235B。启示：SFT 数据应包含 `<think>` 推理步骤，Qwen3-8B 原生支持 thinking/non-thinking 双模式。

### MoE vs Dense 微调

小数据量（~1800 条）+ 垂直领域 + tool calling 密集型 = Dense 更合适。MoE 的 expert 路由需要足够多样的数据，1800 条分散到多个 expert 上效果存疑。

---

## 6. 未完成事项

### 高优先级

1. **提交代码** — 上次 43 个文件 + 本次 5 个文件改动均未提交
2. **429 重试策略优化** — 增加退避时间（当前 1s/2s 太短），或增加 circuit breaker 等待窗口
3. **恢复 noldus-kb** — 等 `180.184.84.124:7001` 恢复后改 `extensions_config.json` 为 `"enabled": true`

### 中优先级

4. **微调数据采集启动** — 按 `docs/plans/2026-04-15-fine-tuning-data-checklist.md` 的 A-B 部分，找产品团队要资料，找行为学专家写解读指南
5. **TrainingDataMiddleware 实现** — E2E 日志钩子，自动录制对话为 Fireworks JSONL 格式
6. **数据转化脚本** — `generate_stats_qa.py`, `generate_skill_qa.py`, `generate_synthetic_data.py`

### 低优先级

7. **范式模板验证** — open_field 和 epm
8. **spec 表填写** — `docs/specs/paradigm-analysis-tools-spec.md`
9. **更新微调设计文档** — `docs/plans/2026-04-13-fine-tuning-small-model-design.md` 中基座模型锁定 Qwen3-8B，平台锁定 Fireworks.ai

---

## 7. 建议接手路径

### 如果要提交代码

```bash
cd /home/qiuyangwang/noldus-insight

# 1. 确认 subagent_enabled 已改为 True
grep "subagent_enabled.*True\|subagent_enabled.*False" \
  packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py \
  packages/agent/backend/packages/harness/deerflow/client.py

# 2. 跑测试确认
cd packages/agent/backend && make test

# 3. 提交
cd /home/qiuyangwang/noldus-insight
git add packages/agent/scripts/serve.sh
git add packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py
git add packages/agent/backend/packages/harness/deerflow/client.py
git add packages/agent/frontend/src/components/ui/command.tsx
git add docs/architecture-diagram.md
git add docs/plans/2026-04-15-fine-tuning-data-checklist.md
git commit -m "fix: gateway reload-exclude glob + subagent_enabled default + hydration mismatch

- serve.sh: change --reload-exclude from 'dir/' to 'dir/*' (fnmatch requires glob)
- agent.py + client.py: subagent_enabled default False → True (4 places)
- command.tsx: move DialogHeader inside DialogContent (Radix useId hydration fix)
- Add architecture diagram and fine-tuning data checklist docs"
```

### 如果要优化 429 重试策略

```bash
# 查看当前重试配置
grep -n "retry\|backoff\|wait\|429\|rate_limit" \
  packages/agent/backend/packages/harness/deerflow/agents/middlewares/llm_error_handling_middleware.py
```

关键位置: `llm_error_handling_middleware.py` 中的 `awrap_model_call`，当前 3 次重试间隔 1s/2s/放弃。建议改为 5s/15s/30s 或指数退避。

### 如果要启动微调数据采集

```bash
# 读取数据采集 checklist
cat docs/plans/2026-04-15-fine-tuning-data-checklist.md
```

第一步：联系产品团队获取 EthoVision XT Reference Manual + 各范式 demo 数据。

---

## 8. 风险与注意事项

1. **代码仍未提交** — 上次 43 个文件 + 本次修改全在工作区
2. **noldus-kb 临时禁用** — `extensions_config.json` 中 `"enabled": false`，不要提交
3. **skills/custom/ 是 gitignored** — skill 文件不在 git 中
4. **GLM-5.1 API 不稳定** — 429 限流和 500 错误频繁，E2E 测试可能需要多跑几次
5. **Fireworks.ai 不支持 MoE 微调** — 不要尝试在 Fireworks 上微调 Qwen3.5-9B

---

## 下一位 Agent 的第一步建议

1. 读取本文档了解全貌
2. `git status` 查看未提交的改动
3. 跑 `make test` 确认测试通过
4. 提交代码
5. 根据优先级选择下一个任务：429 重试优化 or 微调数据采集
