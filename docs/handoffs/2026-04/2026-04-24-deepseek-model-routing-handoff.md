# 2026-04-24 DeepSeek 模型接入与 NewAPI 协议路由决策交接

## 当前任务目标

把 DeerFlow agent 的主模型从 Sonnet（商务暂停）切换到通过 NewAPI 中转的 DeepSeek V4 Pro，同时**保留 thinking 模式**，并让以后切换 Sonnet / GLM / DeepSeek 时尽可能丝滑（只改少量字段）。

## 当前进展

- ✅ 配置跑通：[packages/agent/config.yaml](../../packages/agent/config.yaml) 已改成 DeepSeek V4 Pro 走 NewAPI Anthropic 协议
- ✅ 主模型和摘要模型都用 `use: langchain_anthropic:ChatAnthropic`
- ✅ thinking 模式正常工作，`reasoning_content` 相关 400 错误已消除
- ✅ 验证 NewAPI 的 Anthropic 格式渠道能把 DeepSeek 的 `reasoning_content` 正确映射成 Anthropic `thinking` block（带 `signature`）
- ✅ Memory 已更新：`project_model_routing.md` 记录了这套路由决策
- ✅ CLAUDE.md 里 GLM-5.1 的正面提示规则仍有效（见 memory `feedback_positive_prompting.md`）

## 关键上下文

### 最终可用的 config.yaml 模型段

```yaml
models:
  - name: deepseek-v4-pro
    display_name: deepseek
    use: langchain_anthropic:ChatAnthropic
    model: deepseek-v4-pro
    anthropic_api_url: https://newapi.noldusapi.com   # 注意：不带 /v1
    anthropic_api_key: sk-TAM2iGsQg4IBb0WhZZewlLrgRLWxDqBehzxbIjiSfDCVXnay
    timeout: 180.0   # 注意：不是 request_timeout
    max_retries: 2
    max_tokens: 4096
    temperature: 0.3
    supports_thinking: true
    supports_vision: false
    thinking:
      type: enabled
      budget_tokens: 2048

  - name: deepseek-v4-pro-summary
    # ... 同上，但 supports_thinking: false，无 thinking block
```

同时也把 `summarization.model_name` 从旧的 `claude-sonnet-4-6-summary` 改成了 `deepseek-v4-pro-summary`，两个 model 条目 `name` 不再重名。

### NewAPI 的相关设置（重要）

- NewAPI 地址：`https://newapi.noldusapi.com`
- 有多个渠道，截图确认至少三个：`glm` / `anthropic` / `deepseek-openai`，分别对应不同上游类型
- 走 Anthropic 协议时：**NewAPI 会自动把 DeepSeek 的 `reasoning_content` 映射成 `thinking` block**，所以下面这几个 NewAPI 开关**必须保持关闭**：
  - 强制格式化
  - 思考内容转换（将 reasoning_content 转为 `<think>` 标签）
  - 透传请求体

  开启"思考内容转换"开关过的，会把 thinking 污染到普通 content，导致 title 生成时 thinking 泄漏到用户可见输出。

## 关键发现（踩过的坑）

排查顺序和为什么每个方案没走通：

1. **直接用 `langchain_openai:ChatOpenAI` 打 NewAPI OpenAI 协议渠道** → 400 `Invalid schema for function 'run_statistics': null is not of type "array"`。原因其实是当时主模型还拿 Sonnet 的命名（`claude-sonnet-4-6-summary`），schema 推断混乱。这个错跟协议本身无关，排查时走了弯路。

2. **`langchain_openai:ChatOpenAI` + NewAPI 开"思考内容转换"** → 表面看 thinking 内容被拼进 content（`<think>` 标签），实际仍然 400：`The reasoning_content in the thinking mode must be passed back to the API`。原因：**NewAPI 只转下行响应，不解决 DeepSeek 上游对下一轮请求的 `reasoning_content` 回传校验**。

3. **`deerflow.models.patched_deepseek:PatchedChatDeepSeek`**（装了 `langchain-deepseek` 包）→ 401 Authentication Fails。原因：`ChatDeepSeek` **字段是 `api_base` 不是 `base_url`**，且我们 key 是 NewAPI key，若字段名错会打到 `api.deepseek.com` 官方 endpoint。即便改对 `api_base`，DeepSeek 官方 + NewAPI key 组合本身就不通。这条方案可行，但和 NewAPI 中转叠加的兼容性有隐患，不优雅。

4. **`langchain_anthropic:ChatAnthropic` + NewAPI Anthropic 协议渠道** → **✅ 通了**。验证 curl：

    ```bash
    curl https://newapi.noldusapi.com/v1/messages \
      -H "Content-Type: application/json" \
      -H "x-api-key: $KEY" \
      -H "anthropic-version: 2023-06-01" \
      -d '{"model":"deepseek-v4-pro","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}'
    ```

    响应是标准 Anthropic 格式（content blocks 含 `thinking` + `text`，带 `signature`）。

5. 其他字段细节：
   - `ChatAnthropic` 不认 `request_timeout`，只认 `timeout`
   - `anthropic_api_url` **不带 `/v1` 后缀**，客户端自己拼 `/v1/messages`
   - `ChatDeepSeek` 的 key 环境变量是 `DEEPSEEK_API_KEY`（我们没用）

## 未完成事项

按优先级：

1. **低优先级：切回 Sonnet 商务恢复后**
   - 只需改 `model: deepseek-v4-pro` → `model: claude-sonnet-4-6`
   - `use` / `anthropic_api_url` / `anthropic_api_key` 保持不变
   - 需在 NewAPI 侧确认 Claude 渠道可用

2. **可选：评估是否上 GLM-4/5 作为备选**
   - 同样把 `model` 字段改成 GLM 的模型名
   - 需 NewAPI 侧支持 GLM 的 Anthropic 协议渠道（截图中有 `glm` 渠道但不确定协议类型）
   - 切换前做一次和 curl 类似的协议验证

3. **可选：用户是否想保留多个模型同时可选**
   - 可以在 `models` 数组里并列写多个条目，每个 `name` 不同
   - 前端 model picker 能切换；DeerFlow agent 默认用第一个

## 建议接手路径

**下一位 Agent 的第一步：**

1. **读** [project_model_routing.md](/home/qiuyangwang/.claude/projects/-home-qiuyangwang/memory/project_model_routing.md) memory 确认路由决策
2. **不要重新验证 DeepSeek 接入**——已经跑通了，不要折腾
3. 如果用户要切模型：
   - 只改 `config.yaml` 里的 `model` 字段
   - 重启 `make dev`
   - 不要碰 `use` / `anthropic_api_url` / `anthropic_api_key`
4. 如果用户遇到 thinking 相关报错：
   - 先检查 NewAPI 侧是否误开了"思考内容转换"开关
   - 不要改回 `langchain_openai`——会重新踩坑
   - 不要打开"思考内容转换"——只转响应不解决上游校验

## 风险与注意事项

1. **❌ 不要**改回 `langchain_openai:ChatOpenAI` —— 会重新碰到 `reasoning_content must be passed back` 400
2. **❌ 不要**在 NewAPI 侧开启"思考内容转换"开关 —— 会污染 content 字段，thinking 泄漏到用户可见输出
3. **❌ 不要**用 `anthropic_api_url: https://newapi.noldusapi.com/v1` —— ChatAnthropic 自己拼 `/v1/messages`，多加 `/v1` 会 404
4. **❌ 不要**用 `request_timeout: 180` —— ChatAnthropic 不认，会报 `unexpected keyword argument`
5. **⚠️ 注意** `models[]` 里 `name` 必须唯一（旧 config 两个都叫 `deepseek-v4-pro` 导致 summarization 报错）
6. **⚠️ API key 会过期** —— `sk-TAM2iGsQg4IBb0WhZZewlLrgRLWxDqBehzxbIjiSfDCVXnay` 是当前 NewAPI key，商务变化时需要同步更新
7. **⚠️ Anthropic thinking budget_tokens 要小于 max_tokens** —— 当前 `max_tokens: 4096` / `budget_tokens: 2048`，够 thinking 用且有余量产出最终回答

## 下一位 Agent 的第一步建议

如果用户继续聊这个话题，按场景：

- **要切模型**：只改 [config.yaml:18](../../packages/agent/config.yaml#L18) 的 `model` 字段，重启验证
- **报 `reasoning_content` 错**：去 NewAPI 关掉"思考内容转换"开关，不要改代码
- **报 401/404**：检查 `anthropic_api_url` 有没有错加 `/v1`，检查 key 是不是 NewAPI 的而不是 DeepSeek 官方的
- **要加新模型**：先用 curl 验证 NewAPI 的对应渠道能返 Anthropic 格式，再改 `models[]`
- **用户要验证 thinking 是否在工作**：看 `langgraph.log` 里 DeepSeek 响应 content 是否有 `thinking` block，或 agent 回复是否质量提升
