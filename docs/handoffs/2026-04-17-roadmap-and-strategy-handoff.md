# EthoInsight 会话交接文档

> 日期: 2026-04-17
> 上一份交接: `docs/handoffs/2026-04-16-agent-robustness-handoff.md`

---

## 1. 当前任务目标

**本次会话完成了三件事**：

1. **Agent 分析架构梳理** — 完整梳理了 EthoInsight 的数据分析流水线、领域知识注入机制、知识问答能力现状
2. **产品 Roadmap 创建** — 编写了 12 个月产品路线图，以 2026 年 9 月 v0.1 可用版本为关键里程碑
3. **Fireworks.ai 邮件草稿** — 为回复 Fireworks sales 的微调咨询邮件准备了详细回复草稿

---

## 2. 当前进展

| 编号 | 任务 | 状态 |
|------|------|------|
| 1 | 梳理 agent 数据分析流水线架构 | ✅ |
| 2 | 评估知识问答能力现状和短板 | ✅ |
| 3 | 评估现有架构对"全能助手"愿景的支撑度 | ✅ |
| 4 | 创建 12 个月产品 Roadmap（`docs/roadmap.md`） | ✅ 已提交 |
| 5 | 调整 Roadmap 时间线（9 月 v0.1 交付） | ✅ 已提交 |
| 6 | Fireworks.ai 回复邮件草稿 | ✅ 草稿已给出（未发送，需用户确认后发送） |
| 7 | EPM 范式补全 | ❌ 未开始 |
| 8 | 人工 E2E 验证 | ❌ 未做 |

**提交**: `334ace78` ("roadmap")，branch: `feature/etho-skills`

---

## 3. 本次改动的文件

| 文件 | 改动 |
|------|------|
| `docs/roadmap.md` | 新建。12 个月产品路线图，6 个 Phase + v0.1 里程碑。面向内部开发团队。 |

---

## 4. 关键架构决策

### 产品愿景确认

> **全生命周期行为学研究助手**：实验指导 → 数据分析 → 结果追问 → 知识问答 → 跨范式证据链

### 架构评估结论

> **架构骨架够用，不需要重写。** DeerFlow 的 orchestrator + subagent + skill + MCP 模式天然支持多场景扩展。需要的是填充"肌肉"（知识）和"大脑"（微调模型）。

### Roadmap 时间线（压缩版）

| Phase | 时间 | 核心交付 |
|-------|------|---------|
| 0 稳固根基 | 4 月中 - 5 月中（4 周） | 3 范式 + 鲁棒 agent |
| 1 微调上线 | 5 月中 - 6 月底（6 周） | SFT 版 Qwen3-8B 替代 GLM-5.1（DPO 推迟） |
| 2 知识升级 | 7 月 - 8 月底（8 周） | 能推理的知识助手 + 5 范式 |
| ★ **v0.1** | **9 月初** | **可用版本交付** |
| 3 实验指导 | 9-11 月（8 周） | experiment-advisor + DPO + 7 范式 |
| 4 跨范式 | 11-1 月（8 周） | 证据链整合 + 11 范式 |
| 5 部署 | 1-3 月（8 周） | 多用户 + 本地部署 |

### 知识问答能力评估

| 场景 | 评分 | 说明 |
|------|------|------|
| 追问分析结果 | 7/10 | 有具体数据支撑，回答质量可以 |
| 纯领域知识问答 | 4/10 | noldus-kb 禁用后只有 5 个 skill reference 文件 |
| 深度方法学讨论 | 2/10 | 不推理，只查表 |

### 微调的核心价值定位

> 微调不是为了替代 RAG，而是让模型**内化行为学专家的推理模式**。具体包括：
> 1. Lead agent 的意图分类和多意图拆解能力
> 2. Knowledge-assistant 的推理能力（结合上下文回答，不是查表）
> 3. 领域术语理解和报告生成质量

---

## 5. 关键发现

### Agent 分析流水线（完整梳理）

```
Lead Agent（路由判断：有数据→分析，无数据→知识）
    ↓
code-executor（解析→指标→统计→图表，通过 run_paradigm_analysis 一步完成）
    ↓
data-analyst（审核统计方法、排查混杂因素、发现洞察）
    ↓
report-writer（APA 报告 + 文献引用）
```

核心"专家思维"在三层：
1. **自动统计决策树**（`statistics.py`）— Shapiro-Wilk → 参数/非参数自动选择
2. **领域知识驱动解读**（Skills + `assess.py`）— 表型推断、混杂因素排查、效应量判断
3. **质量审核关卡**（data-analyst）— 统计方法适配性检查、异常检测

### 知识注入三层机制

1. **System prompt 注入**（静态）— skill reference 文件直接在 context 中
2. **Knowledge-assistant 专用 prompt** — 优先用 skill 知识，其次调 MCP
3. **noldus-kb MCP**（当前禁用）— 6200+ 论文，是深度知识的来源

### 未来路由复杂性预警

当前路由是简单 if-else（有文件→分析，没文件→知识）。愿景中需要区分 5+ 种意图（实验指导/数据分析/追问/知识问答/产品操作），且用户可能混合意图。**建议在 Phase 1 的 SFT 数据中提前加入意图分类样本**。

---

## 6. 未完成事项

### 高优先级（Phase 0 — 立即开始）

1. **人工 E2E 验证** — 启动 agent，上传 EPM 数据，验证循环修复后的优雅降级
2. **EPM 范式补全** — `templates/epm.py` + `metrics.py` 6 个函数 + `assess.py` 阈值
3. **Open Field 范式补全** — `templates/open_field.py` + 指标 + 阈值
4. **429 重试策略优化** — `llm_error_handling_middleware.py` 改为 5s/15s/30s
5. **修复 2 个 pre-existing 测试** — `test_client.py` 中 `subagent_enabled` 断言
6. **恢复 noldus-kb** — 等 `180.184.84.124:7001` 恢复后改 `extensions_config.json`
7. **read_file UTF-16 fallback** — `local_sandbox.py:337` 加 BOM 检测

### 中优先级（Phase 1 准备 — Phase 0 期间同步启动）

8. **联系产品团队获取资料** — EthoVision XT Reference Manual + 各范式 demo 数据
9. **微调数据采集脚本开发** — `generate_stats_qa.py`, `generate_skill_qa.py`, `generate_synthetic_data.py`
10. **Fireworks.ai 邮件回复** — 草稿已在对话中，用户确认后发送

### 低优先级

11. **更新微调设计文档** — 锁定 Qwen3-8B + Fireworks.ai
12. **更多范式模板** — forced_swim, morris_water_maze（Phase 2）

---

## 7. 建议接手路径

### 如果要继续 Phase 0

```bash
cd /home/qiuyangwang/noldus-insight

# 1. 确认当前状态
git log --oneline -3
# 应看到 334ace78 "roadmap" 和 6bf51adc "修复了循环问题"

# 2. 读取 Roadmap 了解全盘规划
cat docs/roadmap.md

# 3. 人工 E2E 验证（最高优先级）
make dev
# 上传 demo-data/DemoData/高架十字迷宫/ 下的文件
# 发送 "帮我分析这个高架十字迷宫数据"
# 预期：agent 用 ask_clarification 告知 EPM 暂不支持

# 4. 开始 EPM 范式补全
cat packages/ethoinsight/ethoinsight/templates/shoaling.py  # 参考模板
grep -n "epm\|open_arm" packages/ethoinsight/ethoinsight/metrics.py  # 已有 EPM 代码
```

### 如果要回复 Fireworks 邮件

对话中已给出完整英文草稿，涵盖：
- Use case（行为学分析 agent，orchestrator 架构）
- Model & Training Plan（Qwen3-8B Dense，SFT → DPO）
- Latency（P50 TTFT < 500ms，P90 < 1s，streaming 30+ tok/s）
- Cost（<16B 免费微调吸引点 + 未来本地部署需求）

用户需要确认后发送。关键注意：不要提 GLM-5.1 的不稳定问题，不要提 MoE 不支持的事。

---

## 8. 风险与注意事项

1. **Phase 0 是 v0.1 的关键路径** — 4 周内需完成 EPM + OFT 范式 + 鲁棒性验证 + 基础设施修复，时间紧
2. **Phase 1 的产品资料依赖外部** — 需要产品团队配合，建议 Phase 0 期间同步启动资料收集
3. **DPO 被推迟到 Phase 3** — v0.1 版本只有 SFT，报告生成质量可能不及预期，这是有意的 scope 取舍
4. **noldus-kb 仍然禁用** — `extensions_config.json` 中 `"enabled": false`，不要提交
5. **skills/custom/ 是 gitignored** — skill 文件不在 git 中
6. **roadmap.md 已提交** — 如需调整直接编辑并提交即可
7. **Fireworks 邮件未发送** — 草稿在对话记录中，用户需要确认措辞后自行发送

---

## 下一位 Agent 的第一步建议

1. 读取本文档了解全貌
2. `cat docs/roadmap.md` 了解 12 个月规划和 v0.1 里程碑
3. 根据 Phase 0 优先级开始工作：E2E 验证 → EPM 范式补全 → OFT 范式补全
4. 如果用户提到 Fireworks 邮件，草稿在上一次对话中，核心要点在本文档第 7 节
