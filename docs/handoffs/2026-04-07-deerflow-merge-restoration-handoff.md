# 交接文档：DeerFlow 上游 Merge 后恢复 Noldus 专有功能

> 写给下一位接手的 AI Agent。假设你无法访问之前的对话上下文。

## 1. 当前任务目标

**核心问题**：执行 `git subtree pull` 从 deerflow-noldus 上游拉取了 232 个文件的更新，覆盖了大量 Noldus 专有的代码定制。需要把被覆盖的内容重新加回来。

**预期产出**：
- 所有 Noldus 专有功能恢复正常
- `make dev` → 新建 thread → 上传斑马鱼数据 → 完整的分析流程跑通
- 前端 Noldus 品牌标识恢复

## 2. 当前进展

### ✅ 已修复
- ethoinsight 包加入 uv workspace（`pyproject.toml`）→ `uv sync` 不再清掉它
- ethoinsight symlink 恢复（`backend/packages/ethoinsight` → `../../../ethoinsight`）
- 三个自定义 subagent 文件恢复 + 注册到 `__init__.py`
- `--allow-blocking` 加到 Makefile 解决 matplotlib 阻塞问题
- `.gitattributes` 保护 Noldus 专有文件
- config.yaml 保留了我们的 GLM-5 配置（`request_timeout: 120.0`）

### ❌ 被覆盖、需要恢复的内容

## 3. 需要恢复的文件清单

### P0 — 阻塞 agent 核心流程

#### 3.1 `prompt.py` — Lead Agent Orchestration Guide（最关键）
**文件**: `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`
**问题**: 上游版本是通用的 subagent prompt，没有 Noldus 的 EthoVision 数据分析派遣流程。
**影响**: lead agent 不知道要派发 code-executor → data-analyst → report-writer，而是自己直接用 bash 执行命令。

**需要恢复的内容**（完整内容在旧版 git commit `39dafb5` 的第 260-380 行）：

1. **`_build_subagent_section()` 函数中的自定义 agent 描述**（旧版第 24-29 行）：
   ```python
   if name == "code-executor":
       agent_lines.append("- **code-executor**: 执行 Python 数据分析代码（使用 ethoinsight 库）")
   elif name == "data-analyst":
       agent_lines.append("- **data-analyst**: 解读分析结果，应用行为学领域知识")
   elif name == "report-writer":
       agent_lines.append("- **report-writer**: 撰写 APA 格式的科学报告")
   ```

2. **`<orchestration_guide>` XML 块**（旧版第 260-376 行）：
   - Step 0: 确认需求（范式推断、分组定义）
   - Step 1: 派遣 code-executor（含 CRITICAL 文件路径 glob 模式规则）
   - Step 2: 读 handoff → 派遣 data-analyst
   - Step 3: 读 handoff → 派遣 report-writer
   - Step 4: 整合 → present_files
   - 可用范式模板列表

3. **agent 禁令规则**（旧版第 48-50 行）：
   ```
   - code-executor MUST NOT: explore files, check encodings — only write script + execute
   - data-analyst MUST NOT: run code or produce charts — only interpret existing outputs
   - report-writer MUST NOT: run code or re-analyze data — only write the report
   ```

**恢复方法**：
```bash
# 提取旧版完整内容
cd /home/qiuyangwang/noldus-insight
git show 39dafb5:packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py > /tmp/old_prompt.py
```
然后对比新版 prompt.py（724 行），把上述三块内容注入到新版的对应位置。新版结构有变化，需要仔细阅读后决定插入点。

#### 3.2 `sandbox/tools.py` — 虚拟路径环境变量注入
**文件**: `packages/agent/backend/packages/harness/deerflow/sandbox/tools.py`
**问题**: `_build_path_env()` 函数和 `extra_env` 参数被上游覆盖掉了。
**影响**: Python 脚本内的 `glob.glob("/mnt/user-data/uploads/...")` 无法解析虚拟路径，分析脚本执行失败。

**需要恢复的内容**（旧版 `39dafb5`）：
1. `_build_path_env(thread_data)` 辅助函数（旧版第 332-345 行）：把虚拟路径映射为 `DEERFLOW_PATH_*` 环境变量
2. 在 `bash_tool` 中调用 `_build_path_env()` 并传给 `sandbox.execute_command(command, extra_env=path_env)`（旧版第 861-862 行）

#### 3.3 `sandbox/local/local_sandbox.py` — `extra_env` 参数
**文件**: `packages/agent/backend/packages/harness/deerflow/sandbox/local/local_sandbox.py`
**问题**: `execute_command` 方法的 `extra_env` 参数被去掉了。
**影响**: 即使 tools.py 传了环境变量，local_sandbox 也不接收。

**需要恢复的内容**：
1. `execute_command` 签名加 `extra_env: dict[str, str] | None = None`
2. 在 subprocess 调用时合并 `extra_env` 到 `env`

#### 3.4 `sandbox/sandbox.py` — 基类签名
**文件**: `packages/agent/backend/packages/harness/deerflow/sandbox/sandbox.py`
**问题**: 基类 `Sandbox.execute_command` 签名缺少 `extra_env`。

### P1 — 前端定制

#### 3.5 `artifact-file-list.tsx` — 图片内联预览
**文件**: `packages/agent/frontend/src/components/workspace/artifacts/artifact-file-list.tsx`
**问题**: 我们添加的图片内联预览（`IMAGE_EXTENSIONS`、`isImageFile`、img 网格渲染）被上游覆盖。
**影响**: present_files 返回的图片不会在 chat 中显示预览，只显示文件卡片。

**需要恢复的内容**：完整内容在 git commit `39dafb5` 的该文件中。核心是：
- `IMAGE_EXTENSIONS` Set 和 `isImageFile()` 函数
- 图片文件渲染为 `<img>` + 2 列网格布局
- 非图片文件保持原有卡片样式

#### 3.6 前端 Noldus 品牌
**问题**: logo、品牌名、中文翻译等被 DeerFlow 默认值覆盖。
**影响**: 前端显示 "DeerFlow" 而非 "Noldus Insight"。
**涉及文件**: landing 组件、settings/about、layout 等。
**恢复方法**: `git diff 39dafb5..HEAD -- packages/agent/frontend/` 查看具体差异。

### P2 — 可选改进

#### 3.7 `.gitattributes` 补充
当前 `.gitattributes` 保护了 subagent 文件和 config.yaml，但遗漏了：
- `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`
- `packages/agent/backend/packages/harness/deerflow/sandbox/tools.py`
- `packages/agent/backend/packages/harness/deerflow/sandbox/local/local_sandbox.py`
- `packages/agent/backend/packages/harness/deerflow/sandbox/sandbox.py`
- `packages/agent/frontend/src/components/workspace/artifacts/artifact-file-list.tsx`

## 4. 关键上下文

### 项目架构
- **noldus-insight** monorepo，DeerFlow 在 `packages/agent/` 下通过 git subtree 管理
- **ethoinsight** 是 Noldus 自有的行为数据分析 Python 库，在 `packages/ethoinsight/`
- **上游 remote**: `deerflow` → `https://github.com/noldus-cn-beijing/deerflow-noldus.git`
- **subtree 拉取命令**: `git subtree pull --prefix=packages/agent deerflow main --squash`

### 虚拟路径系统（两层翻译）
1. **bash 命令层**: `replace_virtual_paths_in_command()` 替换命令字符串中的虚拟路径
2. **Python 进程内**: `DEERFLOW_PATH_*` 环境变量 + 模板脚本的 `_resolve_path()` → 解决 `glob.glob()`, `open()` 等 Python API 的路径问题

### 环境变量命名规则
```
/mnt/user-data          → DEERFLOW_PATH_MNT_USER_DATA
/mnt/user-data/uploads  → DEERFLOW_PATH_MNT_USER_DATA_UPLOADS
/mnt/user-data/outputs  → DEERFLOW_PATH_MNT_USER_DATA_OUTPUTS
/mnt/user-data/workspace → DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE
```

### GLM-5 API 不稳定
- 服务端间歇性响应慢（单次请求 2-13 分钟）
- `request_timeout: 120.0` + `max_retries: 3` 已配置
- 测试建议在非高峰时段（凌晨/深夜）进行

### 关键决策
- 上游 DeerFlow 已实现了 cancel_event 机制（在 SubagentResult 中），比我们之前的 threading.Event 方式更好
- 上游 registry.py 已支持 `get_max_turns_for()`，可以在 config.yaml 中覆盖 per-agent max_turns
- 上游 executor.py 结构有较大变化，不要简单恢复旧版

## 5. 恢复旧版内容的方法

所有旧版文件可以从 git commit `39dafb5` 提取：
```bash
cd /home/qiuyangwang/noldus-insight
# 提取某个文件的旧版
git show 39dafb5:packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py > /tmp/old_prompt.py
git show 39dafb5:packages/agent/backend/packages/harness/deerflow/sandbox/tools.py > /tmp/old_tools.py
git show 39dafb5:packages/agent/backend/packages/harness/deerflow/sandbox/local/local_sandbox.py > /tmp/old_local_sandbox.py
git show 39dafb5:packages/agent/backend/packages/harness/deerflow/sandbox/sandbox.py > /tmp/old_sandbox.py
git show 39dafb5:packages/agent/frontend/src/components/workspace/artifacts/artifact-file-list.tsx > /tmp/old_artifact_file_list.tsx
```

**注意**：不要直接用旧版覆盖新版！上游有很多有价值的改进（新的 middleware、vLLM 支持、skill manager、search tools 等）。正确做法是**对比新旧版本，只把 Noldus 专有的部分注入到新版中**。

## 6. 优先查看的文件

| 文件 | 作用 | 恢复难度 |
|------|------|----------|
| `agents/lead_agent/prompt.py` | orchestration guide | 高 — 新版结构变化大，需要仔细定位注入点 |
| `sandbox/tools.py` | _build_path_env + extra_env | 中 — 添加函数 + 修改 bash_tool 调用 |
| `sandbox/local/local_sandbox.py` | execute_command extra_env 参数 | 低 — 只改签名和 env 合并 |
| `sandbox/sandbox.py` | 基类签名 | 低 — 只改签名 |
| `artifact-file-list.tsx` | 图片内联预览 | 低 — 对比后注入图片渲染逻辑 |

## 7. 验证方式

恢复完成后：
1. `make dev` 重启
2. 清理 checkpoints: `rm -f backend/.deer-flow/checkpoints.db* && rm -rf backend/.deer-flow/threads/*`
3. 新建 thread → 上传 5 个斑马鱼文件 → 发送分析请求
4. 检查 `logs/langgraph.log`：
   - lead agent 是否派发了 code-executor subagent？（而非自己 bash 执行）
   - code-executor 是否调用了 `get_analysis_template`？
   - `DEERFLOW_PATH_*` 环境变量是否注入？
   - analysis.py 是否一次执行成功？
   - 图片是否在前端 chat 中内联显示？

## 8. 风险与注意事项

1. **不要简单覆盖新版文件** — 上游有很多有价值的改进，需要 merge 而非 replace
2. **prompt.py 是最复杂的** — 新版 724 行 vs 旧版 535 行，结构有变化
3. **测试前必须用新 thread** — 旧 thread 有 dirty checkpoint
4. **GLM-5 不稳定** — 如果 API 持续超时，不是代码问题，等服务端恢复
5. **每次 subtree pull 后** — 检查 `.gitattributes` 保护的文件是否被覆盖，检查 ethoinsight symlink
6. **sandbox/tools.py 变化很大** — 上游新增了 grep/glob 搜索工具，`_build_path_env` 要找到正确的注入位置

## 下一位 Agent 的第一步建议

1. 读取本文档
2. 读取 `packages/agent/backend/CLAUDE.md` 了解项目全貌
3. 提取旧版文件到 /tmp（见第 5 节命令）
4. **先处理 prompt.py**（P0.1）— 这是 lead agent 不派发 subagent 的直接原因
5. 再处理 sandbox env var 注入（P0.2-P0.4）
6. 最后处理前端（P1）
7. 更新 `.gitattributes` 保护列表（P2）
8. 重启测试验证
