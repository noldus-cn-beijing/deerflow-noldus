# 交接文档：DeerFlow Merge 后 Noldus 专有功能恢复 — 第二阶段完成

> 写给下一位接手的 AI Agent。假设你无法访问之前的对话上下文。

## 1. 当前任务目标

**核心问题**：执行 `git subtree pull` 从 deerflow-noldus 上游拉取了 232 个文件的更新，覆盖了大量 Noldus 专有的代码定制。需要把被覆盖的内容重新加回来。

**预期产出**：
- 所有 Noldus 专有功能恢复正常
- `make dev` → 新建 thread → 上传斑马鱼数据 → 完整的分析流程跑通
- 前端 Noldus 品牌标识恢复

## 2. 当前进展

### ✅ 已完成（前一个 session）
- ethoinsight 包加入 uv workspace（`pyproject.toml`）→ `uv sync` 不再清掉它
- ethoinsight symlink 恢复（`backend/packages/ethoinsight` → `../../../ethoinsight`）
- 三个自定义 subagent 文件恢复 + 注册到 `__init__.py`
- `--allow-blocking` 加到 Makefile 解决 matplotlib 阻塞问题
- `.gitattributes` 保护 Noldus 专有文件
- config.yaml 保留了我们的 GLM-5 配置

### ✅ 已完成（本 session，尚未 commit）

以下 12 个文件已修改完成，**全部通过语法检查**，但尚未 git commit：

#### P0 — Agent 核心流程

1. **`packages/agent/backend/packages/harness/deerflow/sandbox/sandbox.py`**
   - `execute_command` 基类签名添加 `extra_env: dict[str, str] | None = None` 参数

2. **`packages/agent/backend/packages/harness/deerflow/sandbox/local/local_sandbox.py`**
   - `execute_command` 签名添加 `extra_env` 参数
   - 注入 venv PATH、`DEERFLOW_PATH_*` 环境变量、合并 `extra_env`
   - 两个 `subprocess.run()` 调用都加了 `env=env`

3. **`packages/agent/backend/packages/harness/deerflow/sandbox/tools.py`**
   - 添加 `_build_path_env(thread_data)` 函数（在 `_thread_actual_to_virtual_mappings` 之后）
   - `bash_tool` local sandbox 分支改为 `sandbox.execute_command(command, extra_env=path_env)`

4. **`packages/agent/backend/packages/harness/deerflow/agents/lead_agent/prompt.py`**（最大改动 +162/-）
   - `_build_subagent_section()`: 动态查询 subagent 名称，为 code-executor/data-analyst/report-writer 添加中文描述
   - 添加 Noldus 责任边界规则（MUST NOT 规则、正确/错误派遣模式）
   - `SYSTEM_PROMPT_TEMPLATE` 添加 `{orchestration_guide}` 占位符（在 `</citations>` 和 `<critical_reminders>` 之间）
   - `apply_prompt_template()`:
     - 构建 `orchestration_guide`（Step 0-4 EthoVision 分析流程、范式模板列表）
     - `subagent_reminder` / `subagent_thinking` 有 Noldus 中文版本（当 `has_noldus_agents` 时）
     - agent 默认名 `"DeerFlow 2.0"` → `"EthoInsight"`

#### P1 — 前端

5. **`packages/agent/frontend/src/components/workspace/artifacts/artifact-file-list.tsx`**
   - 添加 `IMAGE_EXTENSIONS` Set 和 `isImageFile()` 函数
   - 渲染逻辑分为 imageFiles（2列网格 `<img>` 预览）和 otherFiles（Card 列表）
   - 保留新版 CardHeader grid layout

6. **前端品牌恢复**（6 个文件）：
   - `layout.tsx`: title `"DeerFlow"` → `"Noldus"`, description 更新
   - `workspace-header.tsx`: 折叠态 `DF` → Noldus emblem SVG, 展开态 `DeerFlow` → `Noldus`
   - `header.tsx`: `DeerFlow` → `Noldus Insight`
   - `hero.tsx`: `with DeerFlow` → `with Noldus Insight`
   - `footer.tsx`: `DeerFlow` → `Noldus`
   - `about-content.ts`: 整体恢复为 Noldus Insight 版本（noldus.com.cn 链接等）

#### P2 — 保护

7. **`.gitattributes`**
   - 新增 5 个文件的 `merge=ours` 保护：prompt.py, tools.py, local_sandbox.py, sandbox.py, artifact-file-list.tsx

### ✅ 已验证完好（无需修改）
- 三个自定义 subagent（code_executor.py, data_analyst.py, report_writer.py）+ `__init__.py` 注册
- `get_analysis_template` tool（ethoinsight/templates/tool.py）
- ethoinsight 包（symlink、workspace 注册、可 import）
- EthoInsight skill（skills/custom/ethoinsight/SKILL.md, enabled）
- config.yaml（GLM-5、ethoinsight tool group、subagents enabled）
- noldus-kb MCP 配置（extensions_config.json，当前 disabled）

## 3. 未完成事项

### 必须做
- **Git commit**: 12 个已修改文件尚未提交。建议一次性提交或按功能分 2-3 个 commit。

### 建议做
- **端到端验证**: `make dev` → 新建 thread → 上传斑马鱼数据 → 验证完整分析流程
  ```bash
  cd /home/qiuyangwang/noldus-insight
  make dev
  # 清理旧 checkpoint
  rm -f packages/agent/backend/.deer-flow/checkpoints.db*
  rm -rf packages/agent/backend/.deer-flow/threads/*
  ```
  然后在前端上传数据测试，检查 `logs/langgraph.log`：
  - lead agent 是否派发 code-executor（而非自己 bash 执行）
  - `DEERFLOW_PATH_*` 环境变量是否注入
  - 图片是否在前端 chat 中内联显示

### 可选
- 检查 `noldus-emblem.svg` 是否在 `packages/agent/frontend/public/images/` 中（workspace-header 引用了它）
- 考虑是否需要给更多前端文件加 `.gitattributes` 保护（landing 页面组件等）

## 4. 关键上下文

### 旧版参考
所有旧版文件可从 git commit `39dafb5` 提取：
```bash
git show 39dafb5:<文件路径> > /tmp/old_<文件名>
```

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
- 测试建议在非高峰时段进行

### 关键设计决策
- prompt.py 中的 Noldus 定制是**条件注入**的：只在 `get_available_subagent_names()` 返回包含 code-executor/data-analyst/report-writer 时才生效
- 上游的通用 subagent 逻辑（general-purpose/bash、并行分解策略）作为 fallback 保留
- `_build_path_env()` 只在 local sandbox 分支注入，不影响 Docker sandbox

## 5. 实施计划文档

详细实施计划在：`/home/qiuyangwang/.claude/plans/glimmering-booping-meteor.md`

## 6. 下一位 Agent 的第一步建议

1. 读取本文档
2. **先 commit 已修改的 12 个文件**
3. 运行 `make dev` 重启
4. 清理 checkpoints 后做端到端测试
5. 如果测试通过，考虑 push 到 remote
