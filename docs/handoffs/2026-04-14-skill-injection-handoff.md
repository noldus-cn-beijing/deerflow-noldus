# Subagent Skill 注入 + Lead Agent Skill Bug 修复 — 交接文档

> 日期: 2026-04-14
> 上一份交接: `docs/handoffs/2026-04-14-upstream-sync-handoff.md`

---

## 1. 当前任务目标

**本次会话完成了四件事**：

1. **诊断并修复 code-executor 浪费轮次 pip install 的问题** — bash 工具 docstring 误导 subagent 创建 venv
2. **诊断 GLM-5.1 API 返回空响应问题** — 确认是 `newapi.noldusapi.com` 代理 API key 问题，非代码 bug
3. **发现并修复 lead agent skill 注入为空的 bug** — `_get_enabled_skills()` 缓存未就绪时返回空（同 GitHub issue bytedance/deer-flow#2143）
4. **实现 SubagentConfig skill 注入框架** — subagent 可声明 `skills` 字段，executor 自动将 skill 内容内联注入 system prompt

---

## 2. 当前进展

| 编号 | 任务 | 状态 |
|------|------|------|
| 1 | bash 工具 docstring 优化（不再引导 pip install） | ✅ 已提交 (commit a85752bc) |
| 2 | code-executor prompt 添加 `<environment>` 块 | ✅ 已提交 (commit a85752bc) |
| 3 | 修复 `_get_enabled_skills()` 返回空 bug | ✅ 代码完成，未提交 |
| 4 | SubagentConfig 加 `skills` 字段 | ✅ 代码完成，未提交 |
| 5 | executor.py skill 内联注入逻辑 | ✅ 代码完成，未提交 |
| 6 | code-executor 声明 `skills=["ethoinsight-analysis"]` | ✅ 代码完成，未提交 |
| 7 | 清理 debug 日志 + 修复测试断言 | ✅ 代码完成，未提交 |
| 8 | 全量测试通过 | ✅ 1539 passed, 0 failed |
| 9 | 提交代码 | ❌ 未提交 |
| 10 | prompt.py 回退英文版 | ✅ 已回退（中文化不是空响应根因，但保持英文） |
| 11 | E2E 测试验证 skill 注入效果 | ❌ 未开始 |

---

## 3. 改动的文件（未提交）

共 6 个文件改动：

### 3.1 Lead Agent Skill Bug 修复

| 文件 | 改动 |
|------|------|
| `agents/lead_agent/prompt.py` | `_get_enabled_skills()` 缓存未就绪时阻塞等待（最多 5s），而非返回空列表 `[]` |

### 3.2 Subagent Skill 注入框架（新功能）

| 文件 | 改动 |
|------|------|
| `subagents/config.py` | 新增 `skills: list[str] \| None = None` 字段 |
| `subagents/executor.py` | 新增 `_build_system_prompt()` 方法 + `_load_skill_contents()` 函数；`_create_agent()` 使用 `_build_system_prompt()` |
| `subagents/builtins/code_executor.py` | 新增 `skills=["ethoinsight-analysis"]`，prompt 改为"按注入的 skill 执行" |

### 3.3 清理 + 测试修复

| 文件 | 改动 |
|------|------|
| `agents/middlewares/token_usage_middleware.py` | 移除 ZERO-TOKEN DEBUG 临时日志 |
| `tests/test_lead_agent_skills.py` | 断言从中文"技能自进化"改为英文"Skill Self-Evolution"（匹配回退后的 prompt） |

---

## 4. 关键架构决策

### Subagent Skill 注入：内联 vs Progressive Loading

- **Lead agent** 用 progressive loading（列出 skill 路径，agent 用 `read_file` 读取）— 适合 max_turns 充裕的场景
- **Subagent** 用内联注入（skill 内容直接拼入 system prompt）— 因为 max_turns 有限（code-executor 只有 8 轮），不该浪费在 read_file 上
- `executor.py._build_system_prompt()` 读取 SKILL.md、去掉 YAML frontmatter、用 `<skill>` XML 标签包裹后追加到 system_prompt 尾部

### Lead Agent Skill 缓存 Bug

- 根因：`_get_enabled_skills()` 是非阻塞的，缓存未就绪直接返回 `[]`
- 启动时 `prime_enabled_skills_cache()` 启动后台线程加载，但第一个请求可能在线程完成前到达
- 修复：缓存未就绪时调用 `event.wait(timeout=5s)` 阻塞等待
- 这与 GitHub issue bytedance/deer-flow#2143 是同一个问题

### GLM-5.1 空响应问题

- `newapi.noldusapi.com` API 代理不稳定，GLM-5.1 间歇性返回 `input=0 output=0` 空消息
- 用户切换为直连 `open.bigmodel.cn` API key 后恢复正常
- 不是代码问题，不是 prompt 中文化问题（回退英文后仍复现，换 key 后解决）

### Prompt 中文化状态

- prompt.py 已回退到英文版本（commit 5fb824d3 的状态）
- 中文化不是空响应根因，但暂时保持英文版本
- 如需重新中文化，需在 API 稳定后单独进行

---

## 5. 关键发现

### code-executor 不调用 run_paradigm_analysis

E2E 测试中，code-executor 完全忽略 `run_paradigm_analysis` 工具（即使该工具支持 shoaling 范式），转而用 bash 手动分析数据文件，8 轮 max_turns 全部浪费。原因：
1. code-executor 的 system_prompt 过于简短（4 行），GLM-5.1 不遵循
2. skill 文件 `ethoinsight-analysis/SKILL.md` 存在但无法注入到 subagent（框架限制）
3. 新增的 skill 注入框架解决了这个问题——skill 内容内联后 code-executor 有完整的执行指南

### `run_paradigm_analysis` 支持的范式

| 范式 | 模板状态 | metrics 支持 |
|------|----------|-------------|
| shoaling | ✅ 有模板 | ✅ distance_moved, mean_iid, mean_nnd, mean_polarity |
| open_field | ❌ 无模板 | ✅ distance_moved, center_time_ratio, thigmotaxis_index |
| epm | ❌ 无模板 | ✅ distance_moved, open_arm_time_ratio |

模板文件位于 `packages/ethoinsight/ethoinsight/templates/`。

---

## 6. 未完成事项

### 高优先级

1. **提交代码** — 6 个文件改动未提交
2. **E2E 测试** — 重启 `make dev`，上传 shoaling 数据，验证：
   - lead agent prompt 是否包含 `<skill_system>` section
   - code-executor 是否调用 `run_paradigm_analysis` 而非手动 bash
   - handoff_code_executor.json 是否生成
3. **确认 API key** — 当前用的是 `open.bigmodel.cn` 直连 key，确认 config.yaml 中的 base_url 和 api_key 正确

### 中优先级

4. **Prompt 中文化** — 如需重新中文化 prompt.py，确保在 API 稳定后进行
5. **恢复 noldus-kb** — 等 `180.184.84.124:7001` 恢复后改 `extensions_config.json` 为 `"enabled": true`
6. **open_field / epm 模板** — 在 `packages/ethoinsight/ethoinsight/templates/` 下创建

### 低优先级

7. **上游贡献** — 可将 `_get_enabled_skills()` 修复和 SubagentConfig skills 功能提 PR 到 bytedance/deer-flow
8. **SubagentConfig skills 测试** — 为 `executor.py._load_skill_contents()` 添加单元测试

---

## 7. 建议接手路径

### 如果要提交代码

```bash
cd /home/qiuyangwang/noldus-insight

# 1. 确认测试通过
cd packages/agent/backend && make test

# 2. 查看改动
git diff --stat HEAD

# 3. 提交
cd /home/qiuyangwang/noldus-insight
git add packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py \
       packages/agent/backend/packages/harness/deerflow/agents/middlewares/token_usage_middleware.py \
       packages/agent/backend/packages/harness/deerflow/subagents/config.py \
       packages/agent/backend/packages/harness/deerflow/subagents/executor.py \
       packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py \
       packages/agent/backend/tests/test_lead_agent_skills.py
git commit -m "feat: subagent skill injection + fix lead agent skill cache bug"
```

### 如果要继续 E2E 测试

```bash
cd /home/qiuyangwang/noldus-insight

# 确认 config.yaml 中的 API key 和 base_url
grep -A 5 'glm-5.1' packages/agent/config.yaml

# 启动服务
make dev

# 上传 shoaling demo 数据，观察 langgraph.log：
# 1. 搜索 "skill_system" 确认 lead agent skill 注入
# 2. 搜索 "run_paradigm_analysis" 确认 code-executor 调用了工具
# 3. 搜索 "handoff_code_executor.json" 确认输出文件生成
```

---

## 8. 风险与注意事项

1. **6 个文件未提交** — 需要尽快提交
2. **prompt.py 已回退英文** — 如果前端或其他地方依赖中文 prompt，需注意
3. **API key 已切换** — 从 `newapi.noldusapi.com` 切到 `open.bigmodel.cn`，确认 config.yaml 正确
4. **skills/custom/ 是 gitignored** — skill 文件不在 git 中，换环境需重新部署
5. **_get_enabled_skills() 加了阻塞等待** — 最多等 5 秒，理论上不影响性能（第一次请求后缓存命中），但极端情况下文件系统很慢可能增加首次请求延迟

---

## 下一位 Agent 的第一步建议

1. 读取本文档了解全貌
2. `cd packages/agent/backend && make test` 确认 1539 passed
3. 提交代码
4. 重启 `make dev` 进行 E2E 测试，重点观察 code-executor 是否调用 `run_paradigm_analysis`
5. 如果 code-executor 仍然不调用工具，检查 `executor.py._build_system_prompt()` 的日志输出，确认 skill 内容确实被注入
