# DeepSeek 切换 + 前端 Hydration 修复 交接文档

**日期**: 2026-04-24
**上一会话主题**: agent 模型从 Sonnet 切到 DeepSeek V4 Pro 后，前端报 hydration 错误；上游 sync；前后端修复

---

## 当前任务目标

用户把 `noldus-insight/packages/agent/config.yaml` 里 agent 模型从 Claude Sonnet 切换到 **DeepSeek V4 Pro**（经 NewAPI 的 Anthropic 协议路由）。跑通后发现前端报 React 19 hydration 错误：

```
In HTML, <div> cannot be a descendant of <p>.
```

并且 AI 消息内容被渲染成每个 token 一行。目标：定位根因、修复、拉取上游相关 commit、为上游写 issue 草稿。

---

## 当前进展 ✅

### 已完成 commits（按时间倒序）

| Commit | 主题 | 备注 |
|---|---|---|
| `6dd8bf59` | 用户整合的最终 commit：**合并了 sync 上游 + 修复 + docs + 去掉 langchain-deepseek 依赖** | 已推送/打包 |
| `34996c9b` | (被用户 squash) fix: 修复 DeepSeek 切换后的前端 hydration 错误和消息碎片化 | 已含在 `6dd8bf59` |
| `4872a7b4` | (被用户 squash) sync: 合入 DeerFlow 上游安全改动 (-> 80a7446) | 已含在 `6dd8bf59` |

### 具体改动

**前端 3 处**：
1. [packages/agent/frontend/src/core/messages/utils.ts:139,192](packages/agent/frontend/src/core/messages/utils.ts) — `extractContentFromMessage` / `extractTextFromMessage` 的 content 数组拼接从 `.join("\n")` 改为 `.join("")`；image block 前后加 `\n`
2. [packages/agent/frontend/src/core/streamdown/plugins.ts](packages/agent/frontend/src/core/streamdown/plugins.ts) — 去掉 `streamdownPlugins` 里的 `rehypeRaw`，新增 `reasoningPlugins` 别名（=`streamdownPlugins`）
3. [packages/agent/frontend/src/components/ai-elements/reasoning.tsx:180](packages/agent/frontend/src/components/ai-elements/reasoning.tsx) — `<Streamdown {...props}>` → `<Streamdown {...reasoningPlugins}>`（对齐上游 #2321）

**后端 1 处**：
4. [packages/agent/backend/packages/harness/deerflow/agents/middlewares/archiving_summarization.py:74-79](packages/agent/backend/packages/harness/deerflow/agents/middlewares/archiving_summarization.py) — `__init__` 尾部加一行：
   ```python
   self.model = self.model.with_config(tags=["nostream"])
   ```

**上游 sync**：
- 42 个安全文件从 `deerflow/main@80a7446` 直接合入
- 3 个上游新增文件：`agents/memory/message_processing.py`、`agents/memory/summarization_hook.py`、`agents/middlewares/summarization_middleware.py`
- 7 个受保护文件**未合并**（见下方遗留）

**依赖清理**：
- 删除了 `packages/agent/backend/pyproject.toml` 里的 `langchain-deepseek>=1.0.1`（从未被代码 import，config.yaml 里 DeepSeek 用的是 `langchain_anthropic:ChatAnthropic`）
- `uv sync` 后 `uv.lock` 自动清理，归零

**文档**：
- [docs/problems/2026-04-24-deerflow-upstream-issues.md](docs/problems/2026-04-24-deerflow-upstream-issues.md) — 给上游 bytedance/deer-flow 提 issue 的完整草稿，包含 3 个独立 issue 和建议顺序

---

## 关键上下文

### 架构约束（来自 memory）
- **模型路由统一走 NewAPI + Anthropic 协议**：所有 LLM 调用都用 `langchain_anthropic:ChatAnthropic`，通过 `https://newapi.noldusapi.com` 转发。切换底层模型（Sonnet/DeepSeek/Qwen）只改 config.yaml。
- **飞轮必须吞异常**：录制/反馈代码失败不能拖垮 agent（已有 `TrainingDataMiddleware` 内部 swallow）。
- **GLM 已不用**：memory 里的 GLM 正面提示经验对 DeepSeek 不必须适用；切换模型后要重新验证 prompt 效果。
- **DeerFlow 是 subtree fork**：上游 `deerflow/main` 在 `https://github.com/noldus-cn-beijing/deerflow-noldus.git`（Noldus 维护的 fork，基于 bytedance/deer-flow）。同步用 `scripts/sync-deerflow.sh`。

### 根因链（一定要理解）

切换到 DeepSeek 暴露了 Sonnet 时期被掩盖的 **3 个独立 bug**：

1. **前端 `utils.ts:139,192` 用 `\n` 拼接 content array**：Anthropic content array 相邻 text block 本是连续文本。Sonnet 通常只有 1 个 text block，无事；DeepSeek delta 粒度细，产生几百个小 text block，`\n` 让每 token 一行 → markdown 解析崩溃。
2. **前端 `streamdownPlugins` 含 `rehypeRaw`**：LLM 幻觉的 HTML tag（`<dt>`、`<ee>`）被渲染为真 DOM，块级元素出现在 `<p>` 内 → hydration 错误。上游 #2321 已经意识到此坑，给 reasoning 内容加了 `reasoningPlugins`，但 AI 消息主体仍保留 `rehypeRaw`。
3. **后端 `ArchivingSummarizationMiddleware` 内部 LLM 调用被广播**：`_create_summary` 里 `self.model.invoke()` 没有 nostream tag，LangGraph 的 `stream_mode="messages"` handler 会捕获其 token 流并广播到前端。DeepSeek delta 粒度细 → 前端 SDK 把它累积成一条"幽灵 AI 消息"，内容是整份压缩摘要（被按 token 切碎）。

### 关键文件导航

- `config.yaml`: DeepSeek 模型配置在 lines 15-43（主模型 + summary 模型，都走 Anthropic 协议）
- `docs/problems/2026-04-24-deerflow-upstream-issues.md`: 3 个上游 issue 草稿
- `docs/e2e/斑马鱼鱼群轨迹分析-deepseek.md`: 上次 DeepSeek 跑出问题的对话导出（第 294-624 行能看到碎片化证据）
- `packages/agent/logs/langgraph.log`: 上次跑的日志
- `packages/agent/backend/.deer-flow/checkpoints.db`: thread 的 SQLite checkpoint，用 `langgraph.checkpoint.serde.jsonplus.JsonPlusSerializer` 解码

### 已验证的诊断证据

- NewAPI 返回符合 Anthropic 流式规范（`curl` 测过：`content_block_delta` 的 `index:0` 保持一致，delta 累加）
- 后端 checkpoint 里**没有**那条幽灵摘要消息 → 确认是流广播问题，不是持久化问题
- 最大的单条 AI text block 是完好的 markdown（从 checkpointer 解出 1258 字符、32 行、avg 38 字符/行）
- `langgraph.constants.TAG_NOSTREAM = "nostream"` 是官方 opt-out 机制，`langgraph/pregel/_messages.py:134` 处生效

---

## 未完成事项

### 高优先级

**P1 — 端到端验证修复生效**
做法：
1. `cd packages/agent && make dev`
2. 在前端跑一次完整的斑马鱼分析（长对话，触发 archiving summarization）
3. 检查：
   - DevTools Console 没有 hydration 错误
   - AI 消息不再是每字一行
   - 导出的 markdown 不再有碎片化单字
   - DeepSeek agent 输出连贯

### 中优先级

**P2 — 给上游 bytedance/deer-flow 提 3 个 issue**
草稿在 [docs/problems/2026-04-24-deerflow-upstream-issues.md](docs/problems/2026-04-24-deerflow-upstream-issues.md)。建议顺序：
1. Issue 3（`nostream` tag，一行修复，受益面广）
2. Issue 2（去掉 AI 消息主体的 rehypeRaw，和 #2321 同源）
3. Issue 1（`.join("\n")` → `.join("")`，讨论可能争议）

**P3 — 处理 sync 遗留的 10 个失败测试**
都是上游 API 变化导致，不是本次改动引入。具体：
- `tests/test_memory_upload_filtering.py` — import `_filter_messages_for_memory` 失败（上游 `898f4e8a` 移除了该内部函数）
- `tests/test_memory_updater.py` 7 个失败 — 上游改了 updater 签名（`898f4e8a` + `07fc25d2`）
- `tests/test_skills_parser.py` 2 个失败 — 上游改了 skill YAML 解析（`6dce26a5`）
- `tests/test_client.py::TestClientInit::test_default_params` — `DeerFlowClient` 的某个 default 变了
- `tests/test_client.py::TestEnsureAgent::test_reuses_agent_same_config` — agent 创建/缓存逻辑变了

### 低优先级

**P4 — 处理 7 个受保护文件的上游 sync**
各自带独立 feature，建议单独 session 处理。diff 报告在 `/tmp/deerflow-sync-report/`（会被系统清理，重新跑 `scripts/sync-deerflow.sh --dry-run` 重生成）：
- `agents/lead_agent/agent.py` (142 行) — `55474011` subagent tool_groups 继承、`2176b2bb` bootstrap 名校验、`4ba3167f` summarization flush memory、`eef0a6e2` setup wizard
- `agents/lead_agent/prompt.py` (780 行) — `30d619de` per-subagent skill loading（大 feature）、`5ba1dacf` present_file→present_files 重命名
- `agents/middlewares/llm_error_handling_middleware.py` — `a62ca5dd` httpx.ReadError 捕获、`4d4ddb3d` 断路器
- `sandbox/local/local_sandbox.py` — `dc50a7fd` read/write_file 路径解析
- `sandbox/tools.py` — `ca1b7d5f` ls_tool 输出路径脱敏
- `subagents/executor.py` — `30d619de`、`e5b14906` event loop 冲突
- `tools/builtins/task_tool.py` — `30d619de`、`55474011`

**P5 — 考虑迁移到 `DeerFlowSummarizationMiddleware`**
上游 `4ba3167f` 引入了 `DeerFlowSummarizationMiddleware`（在 `agents/middlewares/summarization_middleware.py` 新文件里），提供 `before_summarization` hook 机制。Noldus 自定义的 `ArchivingSummarizationMiddleware` 目前还直接继承 langchain 原版，迁移到新 API 更干净。**但不急迫**——当前实现正常工作。

---

## 建议接手路径

### 如果接手"验证修复"（P1）

1. 先读这个 handoff 全文
2. 看提交 `git show 6dd8bf59 --stat` 了解改了啥
3. `cd packages/agent && make dev` 启动所有服务
4. 按 P1 步骤验证
5. 如果有问题，看 `packages/agent/logs/langgraph.log`，参考上次调查思路（checkpointer 解码、`git show deerflow/main:...`、`curl` NewAPI 确认协议合规）

### 如果接手"给上游提 issue"（P2）

1. 读 [docs/problems/2026-04-24-deerflow-upstream-issues.md](docs/problems/2026-04-24-deerflow-upstream-issues.md)
2. 到 https://github.com/bytedance/deer-flow/issues 看有没有已存在的相关讨论
3. 按建议顺序提 issue，每个都自包含复现步骤和修复建议

### 如果接手"处理遗留测试/受保护文件"（P3/P4）

1. 跑 `cd packages/agent/backend && make test` 看失败测试
2. 读上游对应 commit 的 diff 理解 API 变化
3. 要么跟进 API 修测试，要么判断 Noldus 自己的实现要不要迁移

---

## 风险与注意事项

### ⚠️ 不要做的事

- **不要重新加 `langchain-deepseek` 依赖**。项目用 `langchain_anthropic:ChatAnthropic` + NewAPI 协议转发路由所有模型（包括 DeepSeek、Qwen）。加 deepseek 包是之前切换时的误操作，已清理。
- **不要把 summarization 改成 `streaming=False`**。底层 `langchain_anthropic` 即便 `invoke()` 也可能走流式 HTTP，改 streaming 参数治标不治本。正确做法是用 `nostream` tag（已做）。
- **不要改 `streamdownPluginsWithWordAnimation` 或 `humanMessagePlugins`**。它们本来就没 `rehypeRaw`，是对的。
- **不要试图修复 `memory_upload_filtering` 测试的 `_filter_messages_for_memory` import**。这个函数被上游 `898f4e8a` 移除了，Noldus 的测试也要同步更新或迁移。
- **不要把 7 个受保护文件直接 `git show deerflow/main:... > 本地`**。会覆盖 Noldus 的定制改动。必须逐个 diff 手动合并。

### ⚠️ 容易搞混的点

- **前端消息是"实时流累积"vs"从 checkpoint 加载"两条路径**。落盘到 checkpointer 的消息是合并好的（单 text block）；但实时流通过 `@langchain/langgraph-sdk/react` 的 `useStream` 累积时可能变多 text block（DeepSeek 粒度细）。导出功能用的是前端 state，不是 checkpoint。
- **LangGraph 的 `stream_mode="messages"` 会捕获所有内部 LLM 调用**，不只是顶层 agent model。middleware 里的 `model.invoke()` 也会被广播，这是我们遇到 summarization 污染的根因。
- **DeerFlow 的 fork 是 `noldus-cn-beijing/deerflow-noldus`，不是 bytedance/deer-flow**。但大多数 commit 都来自上游 bytedance。提 issue 要去 bytedance 仓。

### ⚠️ 已验证失败的方向

- 试图在后端 `archiving_summarization.py` 里 override `_create_summary` 手动传 `config={"callbacks": []}` —— 不需要，`with_config(tags=["nostream"])` 一行即可。
- 试图修 langchain 源码里的 `SummarizationMiddleware._create_summary` —— 不现实，是第三方包。我们在继承层 override `self.model` 才是正道。

---

## 下一位 Agent 的第一步建议

```bash
# 1. 读这份 handoff + 上游 issue 草稿
cat /home/qiuyangwang/noldus-insight/docs/handoffs/2026-04-24-deepseek-hydration-fix-handoff.md
cat /home/qiuyangwang/noldus-insight/docs/problems/2026-04-24-deerflow-upstream-issues.md

# 2. 看最终 commit 的改动
cd /home/qiuyangwang/noldus-insight
git show 6dd8bf59 --stat
git show 6dd8bf59 -- packages/agent/frontend/src/core/messages/utils.ts
git show 6dd8bf59 -- packages/agent/frontend/src/core/streamdown/plugins.ts
git show 6dd8bf59 -- packages/agent/backend/packages/harness/deerflow/agents/middlewares/archiving_summarization.py

# 3. 端到端验证（如果接 P1）
cd packages/agent
make dev
# 然后在浏览器 localhost:2026 跑一次长对话，看 DevTools Console 有没有 hydration 错误
```

**问用户的第一个问题**：想接哪个？P1 验证修复 / P2 给上游提 issue / P3 修 sync 遗留测试 / P4 处理 7 个受保护文件 / P5 迁移 summarization middleware。

---

## 附：本次 session 关键技术发现

1. **NewAPI 的 Anthropic 协议转换是合规的**，问题全在 DeerFlow 自身 —— 只是 Sonnet 粗粒度 delta 掩盖了 3 个 bug。
2. **`langgraph.constants.TAG_NOSTREAM = "nostream"`** 是 middleware 内部 LLM 调用的官方 opt-out 机制，社区普遍知识度不高但记录在 `pregel/_messages.py:134`。
3. **前端 `@langchain/langgraph-sdk/react` 的 `useStream` 会把 `stream_mode="messages"` 的所有事件累积为前端 state 消息**，不区分"顶层 agent 输出"还是"middleware 内部调用"。这是架构层问题，不只是 DeerFlow 的 bug。
