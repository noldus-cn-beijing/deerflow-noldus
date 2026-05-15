# 训练数据飞轮 — 行为学同事使用 SOP

> 2026-04-23 启动。目标：两个月内攒够 800 条 SFT + 300 对 DPO。

> **与 Golden-case 的关系**：飞轮采集**真实使用中**的 agent 输入输出 + 三按钮反馈（量大、质量不均）；[golden-case](golden-case-sop.md) 是**离线人工标注**的黄金标准（量少、质量高）。飞轮训练规模，golden-case 校准质量。两套系统并存，不重复。

## 一句话说明

**你正常使用 EthoInsight 就在贡献训练数据**。每次对话系统自动录制。你在 assistant 回复下打 ✅/⚠️/❌，就在帮我们教未来的自研模型。

## 使用步骤

1. 打开 http://localhost:2026
2. 新建对话，上传你的 EthoVision 数据，像平时一样做分析
3. Agent 每输出一段 assistant 回复或 subtask 卡片，下面都会有三个按钮：
   - **✅ 正确** — 回复没问题，一键提交
   - **⚠️ 需修正** — 基本对但某些点需要改，点开后写出修正版
   - **❌ 错误** — 整体错了，写出正确版本
4. 一次会话结束，所有反馈自动存档

## 你不用做的事

- 不用手动导出任何数据
- 不用记录任何元信息
- 不用担心"反馈不够专业"—即使简单的 ✅ 也是有价值的信号

## 隐私承诺

- 所有数据存在 `packages/agent/backend/.deer-flow/training-data/` 本地目录
- 不上传任何外部服务
- 你可以随时删除某次会话的录制（删除 `auto-collected/<thread_id>.jsonl`）

## 飞轮状态查看（给工程）

```bash
cd packages/agent/backend
make training-stats
```

显示累计样本数、DPO 对数、反馈率、距离目标进度。

## 反馈聚合时机

每周一上午工程跑一次 `scripts/extract_e2e_sessions.py`，合成当周的 SFT/DPO 数据集。届时会发周报给所有同事。

## 推荐节奏

每位同事每周做 2-3 次完整分析，每次打 5-10 条反馈。3-4 位同事 × 2 个月 = 达标。

## 技术说明（给工程）

| 组件 | 位置 |
|------|------|
| 录制中间件 | `packages/agent/backend/packages/harness/deerflow/agents/middlewares/training_data_middleware.py` |
| 反馈 API | `GET/POST /api/threads/{tid}/runs/{rid}/feedback`（SQLite 后端，verdict 三分类 + revised_text） |
| 原始录制 | `.deer-flow/training-data/auto-collected/<thread_id>.jsonl` |
| 反馈数据 | SQLite `feedback` 表（schema 见 `packages/agent/backend/packages/harness/deerflow/persistence/feedback/model.py`），离线导出脚本 `scripts/export_feedback_jsonl.py` 待实现 |
| 后处理脚本 | `scripts/extract_e2e_sessions.py` |
| 进度命令 | `make training-stats` |

### 反馈 ↔ 样本 Join 规则

- 录制样本写入 `message_id`：lead 消息用 AIMessage 的 LangGraph run ID，subagent 用 `subtask-{tool_call_id}`
- 前端发反馈时 `message_id` 与上面两种格式严格对齐
- `extract_e2e_sessions.py` 按 `message_id` 精确匹配；同一 `message_id` 若被打过多次反馈，最新 `submitted_at` 的胜出
- 没有匹配反馈的样本仍然进 SFT（作为无反馈样本），不会被丢弃
- 2026-04-23 之前录制的旧样本没有 `message_id` 字段 → 在新逻辑下等同于无反馈样本，不会误应用其他反馈

