# 教师模型 / 高并发 API 接入选型 Spec

> **Date**: 2026-05-14
> **Author**: Claude Code（调研对话产物）
> **Decision owner**: 王秋阳
> **Status**: Draft，待团队 review
> **Trigger**: 下周内部上线，10 用户并发测试。当前 NewAPI 网关 + DeepSeek-V4-Pro Preview 并发上限不够，需要换更稳的接入通道。同时为未来蒸馏到 Qwen3-30B-A3B 准备教师模型 + CoT 数据飞轮。

---

## 1. 决策摘要（TL;DR）

### 最终结论

| 角色 | 选型 | 通道 |
|---|---|---|
| **线上服务主模型 = 教师模型** | `deepseek-v4-pro` | 阿里云百炼 OpenAI 兼容 endpoint |
| **线上服务备模型（撞 429 兜底）** | `deepseek-v3.2` | 同上 |
| **开发/测试通道** | `deepseek-v4-pro` | 保留 NewAPI 网关（Anthropic 协议）|
| **未来蒸馏学生模型** | `Qwen3-30B-A3B-Thinking-2507` | Fireworks（已有方案） |

### 一句话决策依据

**DeepSeek-V4-Pro 是 MIT License 开源模型**——这一条事实让"切到 Qwen 才能蒸馏"的论据不再成立。继续用已经为 V4-Pro 调教好的 prompt + 把通道从 NewAPI 换成百炼，是零代码改动 + 零法务风险 + 教师能力最强的最优解。

### 不做的事

- ❌ 不切到 Qwen3-235B 当教师（阿里 ToS 明确禁止蒸馏到自家小模型 + 切换成本高）
- ❌ 不用 DeepSeek-R1 当主模型（推理强但 agent / tool calling 弱）
- ❌ 不用豆包 Seed 2.0 Pro 当教师（火山 ToS §7 风险 + 闭源不可自部署兜底）
- ❌ 不删 NewAPI 网关（保留为开发链路，生产走百炼）

---

## 2. 背景与约束

### 2.1 项目现状

- EthoInsight 行为学数据分析 agent 产品，下周内部上线
- 当前主模型：DeepSeek-V4-Pro Preview，经 NewAPI 网关（Anthropic 协议代理）调用
- Agent 框架：DeerFlow fork，LangGraph + 多 subagent（code-executor / data-analyst / report-writer / knowledge-assistant）
- 当前框架对 DeepSeek 的耦合：**低**——`config.yaml` 走 `langchain_anthropic:ChatAnthropic` + 中转网关，换模型只是改 config

### 2.2 业务约束

- **下周必须上线**：留给"切换模型"的窗口不超过 1-2 天
- **10 用户并发测试**：估算高峰 100-300 RPM，500K-2M TPM
- **数据飞轮目标**：内部上线期间收集 CoT 数据，为未来蒸馏 Qwen3-30B-A3B 做训练数据
- **法务**：必须能合法用于训练自家模型（蒸馏目标 = Qwen3-30B-A3B）

### 2.3 关键发现（颠覆性）

| 发现 | 影响 |
|---|---|
| **DeepSeek-V4-Pro 是 MIT License 开源模型** | 蒸馏零法务风险，权重在 HuggingFace 上公开下载 |
| 阿里云百炼 ToS §"竞品训练"明确禁止用 Qwen 输出训练自家模型 | Qwen3-235B 不能直接做教师；必须走 Fireworks 中转（Apache 2.0 模型 + 第三方托管） |
| 火山方舟 ToS §7 同样禁止训练"竞品"模型 | 豆包系列做教师有风险 |
| 阿里云百炼已正式上架 `deepseek-v4-pro`（1.6T MoE / MIT）| V4-Pro 不只在 NewAPI 中转，有官方企业 SLA 通道 |
| DeepSeek-V4-Pro `enable_thinking` 默认开启 + `reasoning_content` 透传 | CoT 蒸馏数据自然采集，无需额外工程 |
| DeepSeek 官方就用 R1 蒸馏出 R1-Distill-Qwen-32B | "DeepSeek → Qwen 跨家族蒸馏" 是行业最知名成功案例之一，可行性已验证 |

---

## 3. 候选方案对比

调研覆盖：阿里云百炼、火山方舟、Fireworks、腾讯云 LKEAP、硅基流动、DeepSeek 官方、OpenRouter。

### 3.1 教师模型候选对比

| 模型 | License | Agent benchmark | 蒸馏 ToS | 中文 | 你产品适配 |
|---|---|---|---|---|---|
| **DeepSeek-V4-Pro** | **MIT 开源** | SWE-Verified **80.6** / Codeforces **3206** / MCPAtlas 73.6 | ✅ 零限制 | 强 | ✅ Prompt 已调教 |
| Qwen3-235B-Thinking-2507 | Apache 2.0 | BFCL-v3 70.9 / TAU2-Retail 74.6 / IFEval 88.7 | ❌ 百炼 ToS 禁止；Fireworks 中转可绕 | 极强 | ⚠️ 需 prompt 回归测试 |
| Qwen3.6-Max-Preview | 闭源 | SWE-Bench Pro 第一 / Terminal-Bench 2.0 第一 | ❌ 阿里 ToS 禁止 | 极强 | ⚠️ Preview 不稳 |
| Doubao-Seed-2.0-Pro | 闭源 | AIME25 98.3 / SuperCLUE-VLM 90.66 榜首 | ⚠️ 火山 ToS §7 风险 | 极强 | ❌ 不能自部署兜底 |
| DeepSeek-R1 / R1-0528 | MIT 开源 | 推理强 / agent 弱 | ✅ | 中 | ❌ Tool calling 差，handoff JSON 不稳 |

### 3.2 通道对比（DeepSeek-V4-Pro）

| 通道 | TPM | SLA | 价格（输入未命中/输出，元/M tokens） | 国内延迟 |
|---|---|---|---|---|
| DeepSeek 官方 | 动态限流，~300 RPM 实测 | ❌ 无 | 12 / 24（原价）；2.5 折 = 3 / 6 | 优 |
| **阿里云百炼** | 默认 1.2M + 可自助提额 + PTU 预留 | ✅ 企业级 | 同上 + 75% 折扣到 2026-05-31 | **最优**（北京区） |
| 火山方舟 | 5M 起步 | ✅ 企业级 | V4 系列**接入点待确认** | 优 |
| 硅基流动 | 企业版多租户 | ✅ | 同官方 | 优 |
| NewAPI 网关（现状）| 未公开 | ❌ | 未公开 | 加一跳 |

→ **百炼是当前最稳的选择**：V4-Pro 已正式上架 + 75% 折扣 + 企业 SLA + 国内延迟最优。

---

## 4. 选定方案：DeepSeek-V4-Pro 走百炼

### 4.1 模型规格

- **Model ID**: `deepseek-v4-pro`
- **架构**: 1.6T 总参 / 49B 激活 MoE（全球最大开源 MoE）
- **License**: MIT
- **Context**: 1M tokens 输入 + 384K tokens 输出
- **Thinking**: 默认开启，可控参数 `enable_thinking` + `reasoning_effort`（`high` / `max`）
- **Function Calling**: ✅
- **JSON 输出 / 上下文缓存**: ✅ 默认启用
- **Endpoint**: `https://dashscope.aliyuncs.com/compatible-mode/v1`（OpenAI 兼容）
- **SLA**: 企业级（百炼平台保障）
- **价格**（2026-05-31 前 2.5 折活动）:
  - 输入 cache miss：¥3 / M tokens（原价 ¥12）
  - 输入 cache hit：¥0.1 → 叠加再降至 **¥0.025** / M tokens
  - 输出：¥6 / M tokens（原价 ¥24）
  - Batch（异步）：实时价 × 50%

### 4.2 成本测算

**假设**：10 用户 × 每天 5 thread × 每 thread 50K input + 20K output（含 CoT thinking）

| 阶段 | 日 input | 日 output | 日成本 | 月成本 |
|---|---|---|---|---|
| 5 月底前促销 + 60% cache 命中 | 350 万 (140 万 hit + 210 万 miss) | 140 万 | ¥630 (hit) + ¥630 (miss) + ¥840 = ¥2.1K | **~¥8-10K** |
| 6 月起 4 折 + 60% cache | 同上，单价上调 | 同上 | ~¥3K | **~¥15-20K** |

→ 完全在小公司预算内。

### 4.3 配置改动（零代码）

在 `packages/agent/config.yaml` 的 `models` 段新增：

```yaml
models:
  # === 生产主模型：DeepSeek-V4-Pro via 阿里云百炼 ===
  - name: deepseek-v4-pro
    display_name: deepseek
    use: langchain_openai:ChatOpenAI
    model: deepseek-v4-pro
    openai_api_base: https://dashscope.aliyuncs.com/compatible-mode/v1
    openai_api_key: $DASHSCOPE_API_KEY
    timeout: 180.0
    max_retries: 2
    max_tokens: 8192
    temperature: 0.3
    supports_thinking: true
    extra_body:
      enable_thinking: true
      reasoning_effort: high  # 复杂 agent 任务可改 max；省成本可关 thinking

  # === 生产备模型：DeepSeek-V3.2 撞 429 兜底 ===
  - name: deepseek-v3.2-fallback
    display_name: deepseek-fallback
    use: langchain_openai:ChatOpenAI
    model: deepseek-v3.2
    openai_api_base: https://dashscope.aliyuncs.com/compatible-mode/v1
    openai_api_key: $DASHSCOPE_API_KEY
    timeout: 180.0
    max_retries: 2
    max_tokens: 4096
    temperature: 0.3
    supports_thinking: false  # 省成本

  # === 摘要模型：复用 V4-Pro 但关 thinking ===
  - name: deepseek-v4-pro-summary
    display_name: deepseek-summary
    use: langchain_openai:ChatOpenAI
    model: deepseek-v4-pro
    openai_api_base: https://dashscope.aliyuncs.com/compatible-mode/v1
    openai_api_key: $DASHSCOPE_API_KEY
    timeout: 180.0
    max_retries: 2
    max_tokens: 4096
    temperature: 0.3
    supports_thinking: false
    extra_body:
      enable_thinking: false

  # === 开发/测试链路（保留 NewAPI Anthropic 通道）===
  # 旧 config 不删，方便回滚和本地调试
  - name: deepseek-v4-pro-newapi
    use: langchain_anthropic:ChatAnthropic
    model: deepseek-v4-pro
    anthropic_api_url: https://newapi.noldusapi.com
    anthropic_api_key: $NEWAPI_KEY  # 从环境变量读，原 key 保留作为 dev key
    # ... 其余同现状
```

**环境变量**：在 `.env` 或部署环境中设置 `DASHSCOPE_API_KEY=sk-xxx`（从百炼控制台获取）。

### 4.4 ThinkTagMiddleware 兼容性

`packages/agent/backend/packages/harness/deerflow/agents/middlewares/think_tag_middleware.py` 已经在注释里明确支持："DeepSeek-R1, minimax, Qwen reasoning variants, and Claude's extended thinking"。OpenAI 兼容协议下 `reasoning_content` 通过 `additional_kwargs` 透传，middleware 无需改动。

### 4.5 训练数据飞轮兼容性

`TrainingDataMiddleware` 已经在记录 `reasoning_content`（见 `CLAUDE.md` 第 7 条 + `packages/agent/backend/CLAUDE.md` middleware 章节）。切换到百炼后 CoT 数据**自动落到** `.deer-flow/training-data/auto-collected/<thread_id>.jsonl`，无需额外工程。

---

## 5. 上线 Checklist

### 阶段 1：账户准备（即可执行，~1 小时）

- [ ] 在阿里云百炼控制台开通 deepseek-v4-pro 模型（如未开通）
- [ ] 创建 API Key，记录到密钥管理（不入 git）
- [ ] 在百炼控制台「限流提额」自助申请 TPM 提升（默认 1.2M → 申请 3-5M），30 天有效
- [ ] 评估是否需要采购 PTU 预置吞吐（长期 SLA 保障，可选）

### 阶段 2：配置改动（~30 分钟）

- [ ] 按 §4.3 修改 `packages/agent/config.yaml`
- [ ] 设置 `DASHSCOPE_API_KEY` 环境变量
- [ ] 启动服务确认无 import 错误
- [ ] 用 curl 直接打一次 `dashscope.aliyuncs.com/compatible-mode/v1/chat/completions` 验证 endpoint 可达

### 阶段 3：回归测试（~半天）

- [ ] `make test` 全绿（agent backend + ethoinsight）
- [ ] 端到端跑 1 个 EPM golden case，确认 lead → code-executor → data-analyst → report-writer 链路完整
- [ ] 检查 `.deer-flow/training-data/auto-collected/<thread_id>.jsonl` 里有 `reasoning_content` 字段
- [ ] 检查 ThinkTagMiddleware 正确路由 `<think>` 块（前端 Reasoning 折叠区可见）
- [ ] Gate 反问机制（GuardrailMiddleware / GateEnforcementMiddleware）行为不变
- [ ] handoff JSON 文件格式不变（subagent 间数据传递）
- [ ] 中文输出风格符合 APA 报告规范

### 阶段 4：灰度（~半天，上线第 1 天）

- [ ] 内部 1-2 个用户先跑半天，观察 latency / 错误率 / cache 命中率
- [ ] 监控 token usage，确认 cache 命中率符合预期（目标 >50%）
- [ ] 全量开放给 10 用户

### 阶段 5：监控（持续）

- [ ] 在阿里云百炼控制台监控：
  - TPM / RPM 使用率（接近 80% 上限时申请提额）
  - 429 错误率（>0.5% 触发 v3.2-fallback 降级策略）
  - 平均 latency
- [ ] 每周看一次 `.deer-flow/training-data/auto-collected/` 飞轮累积量（参考 `make training-stats`）

---

## 6. 风险登记 + 应对

| 风险 | 概率 | 影响 | 应对 |
|---|---|---|---|
| V4-Pro 算力紧张撞 429 | 中 | 中 | config 已带 v3.2 fallback；框架支持模型切换 |
| 5 月底促销结束，6 月起价格 ×1.6 | 高（必然） | 低 | 提前规划月预算 ¥15-20K；评估 PTU 预付 |
| 阿里云百炼对 DeepSeek 输出做"数据回流"训练 | 低 | 中 | 百炼默认不训练（AES-256 加密）；**不勾选** "Coding Plan" / "协作奖励计划"（这两个会授权数据） |
| ThinkTagMiddleware 在 OpenAI 协议下 `reasoning_content` 解析不对 | 低 | 中 | 回归测试明确 cover；如果有问题，调整 middleware 的 OpenAI 路径 |
| 阿里云 deepseek-v4-pro 临时下架或改名 | 极低 | 高 | 备用通道：火山方舟（待 V4 接入点确认）、硅基流动、DeepSeek 官方 |
| NewAPI 网关同时保留导致密钥泄漏面增大 | 低 | 低 | 旧 key 保留为开发用，生产 key 单独管理 + 短期 rotate |

---

## 7. 未来蒸馏 Roadmap（不在本 spec 实施范围）

**预期时间线**：v0.1 后（2026-09 后）启动蒸馏。本 spec 只确保**数据飞轮在内部上线期间正常采集**。

```
阶段 A（now → 2026-09）：数据采集
  - DeepSeek-V4-Pro 跑线上服务
  - TrainingDataMiddleware 自动记录 (input, reasoning_content, output) 三元组
  - 行为学同事走 feedback API 标注 verdict + revised_text
  - 目标累积：~500-1000 高质量 thread

阶段 B（2026-09+）：蒸馏训练
  - 教师数据：累积的 V4-Pro reasoning trace
  - 学生模型：Qwen3-30B-A3B-Thinking-2507（同模式，保留 CoT）
  - 训练平台：Fireworks（已锁定，详见 docs/plans/2026-05-14-fireworks-meeting-questions.md）
  - 跨家族蒸馏路径：DeepSeek → Qwen
    - Tokenizer 重新对齐（按 Qwen 重 tokenize 即可，无损）
    - Chat template 对齐（都是 <think>...</think>，无需改）
    - 风格漂移：SFT 数据后处理统一为中文学术风
  - 参考案例：DeepSeek-R1-Distill-Qwen-32B（行业最知名跨家族蒸馏成功案例）

阶段 C（蒸馏后）：本地部署
  - 学生模型：Qwen3-30B-A3B-Thinking-2507 + LoRA / NVFP4 量化
  - 目标硬件：RTX 5090 32GB（详见 docs/plans/2026-05-13-base-model-decision-memo.md）
  - 部署形态：on-premise，研究员工作站本地推理
```

**学生模型为什么选 Thinking 而不是 Instruct**：
- 蒸馏目标是让学生**学习思维过程**，不只是答案
- 行为学场景需要展示"为什么选 Mann-Whitney U test 而不是 t-test"这种统计决策推理
- 同模式（Thinking-Thinking）蒸馏比跨模式（Thinking-Instruct）保留 CoT 质量更高

---

## 8. 决策依据：为什么不切到 Qwen3-235B

调研过程中最反复讨论的点。最终不切的理由：

| 维度 | DeepSeek-V4-Pro | Qwen3-235B-Thinking-2507 |
|---|---|---|
| License | **MIT** | Apache 2.0 |
| 通过官方 API 蒸馏的 ToS | ✅ MIT 零限制 | ❌ 阿里云百炼明确禁止 |
| 通过第三方（Fireworks）API 蒸馏 | ✅ 也可（MIT） | ⚠️ 灰区，需法务确认 |
| Agent benchmark | SWE 80.6 / Codeforces 3206 / MCPAtlas 73.6 | BFCL 70.9 / TAU2-Retail 74.6 |
| 你产品 prompt 适配度 | ✅ 已调教 | ⚠️ 需 1-2 天回归 |
| 切换框架代码改动 | 0 | 0（架构已支持）|
| TAU2-Telecom（多步工具编排，最接近你产品） | 数据未公开 | **32.5（弱）** |
| Verbose 程度 | 正常 | **1.85x**，handoff JSON 易污染 |
| 蒸馏到 Qwen3-30B-A3B 的工程损失 | 跨家族：tokenizer/template 重对齐（无损 + 风格漂移）| 同家族：直接训 |

**核心权衡**：Qwen3-235B 唯一硬优势是"同家族蒸馏顺滑"，但 V4-Pro 是 MIT + 已调教 + agent benchmark 全面胜出。**跨家族蒸馏的损失被 V4-Pro 自身的优势完全抵消**——这是 DeepSeek 官方蒸 R1-Distill-Qwen-32B 已经验证过的事实。

---

## 9. Open Questions（提交团队 review 时确认）

1. **百炼账户开通流程**：公司是否已有阿里云企业账号？如未开通，预计 1-3 天流程，会影响下周一上线时间窗。
2. **5 月底促销价 vs 6 月正常价**的预算决策：是否需要在 5 月内做 PTU 预付锁价？
3. **NewAPI 网关是否保留**：当前提案是"生产走百炼 + 开发保留 NewAPI"，是否需要更激进地全删？
4. **TPM 提额申请**：默认 1.2M 够不够 10 用户高峰？是否首日就申请提升到 3-5M？

---

## 10. 相关文档

- [CLAUDE.md](../../../CLAUDE.md) — 项目总览（第 5、6、7 条与本 spec 直接相关）
- [docs/plans/2026-05-13-base-model-decision-memo.md](../../plans/2026-05-13-base-model-decision-memo.md) — Qwen3-30B-A3B 基座升级提议（v0.1 后蒸馏目标）
- [docs/plans/2026-05-14-fireworks-meeting-questions.md](../../plans/2026-05-14-fireworks-meeting-questions.md) — Fireworks 平台问询（蒸馏训练平台）
- [docs/specs/llm-finetuning-strategy.md](../../specs/llm-finetuning-strategy.md) — 微调策略
- [docs/sop/training-data-flywheel-sop.md](../../sop/training-data-flywheel-sop.md) — 训练数据飞轮 SOP
