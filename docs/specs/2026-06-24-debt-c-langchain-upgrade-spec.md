# 债 C：langchain / langgraph 生态版本升级（独立可执行 spec）

> **母 spec**：`docs/specs/2026-06-24-deerflow-sync-historical-debt.md` 的「债 C」小节（§C.1–C.3）。本文自包含、可直接交给 agent 执行。
>
> **解锁目标**：移除 `tests/test_sandbox_middleware.py::test_default_lazy_tool_acquisition_uses_async_provider` 的 `@pytest.mark.skip`，10 passed 0 skipped。**（该测试不在 5 个回退测试文件之列，收益最低但风险最高。）**
>
> **规模**：中—高（有 blast radius，非「小」）。无源码改动，只改 `packages/harness/pyproject.toml`（版本下界）+ `uv.lock`（重解析）。
>
> **⚠️ 本债风险最高，spec 强烈要求：最后做、独立 PR、独立验证、本地 `make dev` 起一次确认运行时。**

---

## 〇、给实施 agent 的一句话

把本地 langchain/langgraph 生态版本下界对齐上游（langchain 1.2.3→1.2.15、langgraph 1.0.x→1.1.x），让本地 `ToolRuntime` 接受 `tools=` kwarg，解锁那个被 skip 的 sandbox 中间件测试。

---

## ⚠️ 给执行 agent 的前置须知（务必先读）

1. **本债是版本升级，blast radius 真实存在**：langgraph 跨 **1.0→1.1 minor**，harness 中间件 / GuardrailProvider / checkpointer 全站在 langgraph API 上。升级后中间件/guardrail/checkpointer 相关测试可能新红。**判断新红时必须区分「升级破坏」vs「baseline 污染」**。
2. **必须在 A/B/D 全合入 dev 后再做**：否则全量回归的红数会被 A/B/D 的改动与 C 的库版本变化混在一起，无法干净对账「C 升级引入了什么」。
3. **commit SHA 不可信**。看 `git show deerflow/main:backend/packages/harness/pyproject.toml` 当前态。
4. **测试解释器 + env**：`cd packages/agent/backend`，`.venv/bin/python -m pytest`，`PYTHONPATH=packages/harness:.`，worktree 必带 `DEER_FLOW_CONFIG_PATH=<abs config.yaml>`。
5. **`make dev` 运行时验证不可省**：langgraph minor 升级可能让 Gateway 起不来（import 期不崩不代表运行时不崩）。升级生效后必须 `make dev` 起一次、跑一轮对话。

---

## 一、真实根因（2026-06-24 已坐实，推翻旧版 spec 三处错）

旧版 spec 说「`ToolRuntime` 上游接受 `tools`、本地不接受，17 个测试调用点加 `tools=None`，规模小」——**三处错**：

1. **方向反了**：`ToolRuntime` 来自 `langchain.tools`（**库**，不是 deerflow 源码）。本地装 `langchain==1.2.3` 的 `ToolRuntime.__init__` 签名是 `(state, context, config, stream_writer, tool_call_id, store)`——**不接受 `tools`**。是**上游升级到 `langchain>=1.2.15` 后** `ToolRuntime` 新增了 `tools` 字段。**是「本地库太旧」，不是「本地代码缺参数」。**（已 `inspect.signature` 坐实：本地签名无 `tools`。）
2. **不是 17 处、是 1 处**：本地真正受影响的只有 `tests/test_sandbox_middleware.py::test_default_lazy_tool_acquisition_uses_async_provider`（唯一传 `tools=[]` 的测试），而且**已被显式 skip**（行 159，reason 标 langchain 1.2.3）。**5 个回退测试文件无一依赖债 C。**
3. **规模不小**：本地 vs 上游是**整个 langchain/langgraph 生态的版本差**（见下表），不是单个库。langgraph `>=1.0.6,<1.0.10` 上界**主动挡住**新栈，升 langchain 到 1.2.15+ 很可能连带要放开 langgraph 上界到 1.1.x——**minor 跳跃**。

### 本地 vs 上游 pyproject.toml 下界（已 grep 坐实）

| 包 | 本地下界 | 上游下界 |
|---|---|---|
| `langchain` | `>=1.2.3` | `>=1.2.15` |
| `langchain-anthropic` | `>=1.3.4` | `>=1.4.1` |
| `langchain-openai` | `>=1.1.7` | `>=1.2.1` |
| `langchain-mcp-adapters` | `>=0.1.0` | `>=0.2.2` |
| `langgraph` | `>=1.0.6,**<1.0.10**` | `>=1.1.9`（**跨 1.0→1.1 minor**） |
| `langgraph-api` | `>=0.7.0,<0.8.0` | `>=0.8.1` |
| `langgraph-cli` | `>=0.4.14` | `>=0.4.24` |
| `langgraph-runtime-inmem` | `>=0.22.1` | `>=0.28.0` |
| `langchain-deepseek` | `>=1.0.1` | `>=1.0.1`（同） |
| `langchain-google-genai` | `>=4.2.1` | `>=4.2.1`（同） |
| `langgraph-checkpoint-sqlite` | `>=3.0.3` | `>=3.0.3`（同） |
| `langgraph-sdk` | `>=0.1.51` | `>=0.1.51`（同） |
| `langgraph-checkpoint-postgres` | `>=3.0.5` | `>=3.0.5`（同） |

---

## 二、执行步骤

### 决策对齐版本（先试最小升级，解析失败再跟齐）

- **最小升级**：只把 `langchain`/`langchain-core` 升到能用 `tools=`，langgraph 尽量不动。blast radius 小，但 langchain 1.2.15 的依赖约束可能本身就要求 langgraph 1.1（需 `uv` 解析时才知道）。
- **跟齐上游**：langchain+langgraph 全家对齐上游下界。更彻底、消除未来 sync 冲突（符合「全量跟随上游底座」策略），但要吃 langgraph 1.0→1.1 的 API 变更。

**先试最小升级，解析失败再跟齐。**

### 步骤
1. 改 `packages/agent/backend/packages/harness/pyproject.toml` 对应下界（上表「本地」→「上游」列）。
2. 重解析锁：
   ```bash
   cd packages/agent/backend
   uv lock --upgrade-package langchain --upgrade-package langchain-core  # 最小升级
   # 若解析报 langgraph 冲突，放开 langgraph 上界并：
   # uv lock --upgrade-package langgraph --upgrade-package langgraph-api --upgrade-package langgraph-cli --upgrade-package langgraph-runtime-inmem \
   #   --upgrade-package langchain-anthropic --upgrade-package langchain-openai --upgrade-package langchain-mcp-adapters
   uv sync
   ```
3. **回归整个 harness**（langgraph 升级高危）：
   ```bash
   PYTHONPATH=packages/harness:. python -c "import app.gateway"
   PYTHONPATH=packages/harness:. python -c "from deerflow.agents import make_lead_agent"
   PYTHONPATH=packages/harness:. DEER_FLOW_CONFIG_PATH=<abs> make test 2>&1 | tail -40
   ```
   重点看中间件 / guardrail / checkpointer 相关测试有无新红。
4. 升级生效后**移除那条 skip**并跑：
   ```bash
   # 删掉 test_sandbox_middleware.py 里 test_default_lazy_tool_acquisition_uses_async_provider 上的 @pytest.mark.skip
   PYTHONPATH=packages/harness:. DEER_FLOW_CONFIG_PATH=<abs> python -m pytest tests/test_sandbox_middleware.py -q
   # 期望 10 passed, 0 skipped
   ```
5. **务必本地 `make dev` 起一次**（在 `packages/agent/` 目录），确认 langgraph 升级没破运行时：Gateway 起得来、能跑一轮对话、SSE 正常。
6. 在 PR 注明实际落定的版本组合 + langgraph 是否跨 minor。

---

## 三、测试

1. `test_sandbox_middleware.py` 全绿 10 passed 0 skipped（skip 移除后）。
2. 全量回归：中间件 / guardrail / checkpointer / memory / SSE 相关测试无新红。
3. `make dev` 起得来 + 跑通一轮对话（langgraph minor 升级的运行时验证）。

---

## 四、验收标准

1. ✅ `make test` 全量 ≤ 开工 baseline 红数（**严格 ≤**：langgraph 升级不允许新增任何红，新红即升级破坏）。
2. ✅ `test_sandbox_middleware.py` 的 skip 移除后 10 passed 0 skipped。
3. ✅ `make dev` 起得来，能完成一轮对话（langgraph minor 升级的运行时验证）。
4. ✅ PR 注明实际落定版本组合 + langgraph 是否跨 minor + 若跟齐上游列明吃的 API 变更。

---

## 五、风险与注意事项

1. **最高风险债**：langgraph 1.0→1.1 是 minor 跳跃。已知站在 langgraph API 上的本地组件（**改前 grep 确认调用面**）：
   - 自定义中间件：`ArchivingSummarizationMiddleware` / `ThinkTagMiddleware` / `TrainingDataMiddleware` / `GateEnforcementMiddleware` / `SealGateMiddleware` / `DegradationCircuitBreakerMiddleware` / `LoopDetectionMiddleware` / `ClarificationMiddleware` / `ParadigmIdentificationGateMiddleware`（`agents/middlewares/`）。
   - `GuardrailProvider` 协议（`guardrails/`）。
   - checkpointer（`runtime/checkpointer` / `persistence/`）。
   - `create_agent` / `AgentMiddleware` / `Command(goto=...)` / `jump_to`（LangChain 1.2 + langgraph 1.1 API）。
2. **若升级后某 Noldus 中间件因 langgraph 1.1 API 变更而崩**：surgical 适配该中间件（守受保护），别回退升级。若适配代价过大，记录并降级（保留 skip + 在 PR/交接注明 langgraph 1.1 与 Noldus 某中间件不兼容，留 follow-up）。
3. **`uv.lock` 冲突**：若 langchain 1.2.15 硬性要求 langgraph 1.1+，最小升级自动失败，必须跟齐。**别为保 langgraph 1.0 而强行钉 langchain 1.2.3**（那等于不做本债）。
4. **不要碰 5 个回退测试文件**：本债与它们无关（债 A/B/D 负责）。
5. **顺序硬约束**：A/B/D 全合入 dev 后才做本债。否则全量红数对账无法区分「C 升级破坏」vs「A/B/D 改动」。
6. **`make dev` 必跑**：import 期不崩 ≠ 运行时不崩。langgraph minor 升级最易在运行时（checkpointer 序列化、中间件 hook 签名、SSE 事件）暴露。

---

## 附：核验命令速查（开工前复跑，防代码漂移）

```bash
cd /home/wangqiuyang/noldus-insight
# 当前已装版本
packages/agent/backend/.venv/bin/python -c "import langchain, langchain_core; print('langchain', langchain.__version__, 'langchain_core', langchain_core.__version__)"
packages/agent/backend/.venv/bin/python -c "from importlib.metadata import version; print('langgraph', version('langgraph'))"
# pyproject 下界对比
diff <(grep -nE 'langchain|langgraph' packages/agent/backend/packages/harness/pyproject.toml) \
     <(git show deerflow/main:backend/packages/harness/pyproject.toml | grep -nE 'langchain|langgraph')
# skip 标记现状
grep -nB1 'tools=\[\]' packages/agent/backend/tests/test_sandbox_middleware.py
# ToolRuntime 签名（升级后应多出 tools 字段）
packages/agent/backend/.venv/bin/python -c "import inspect; from langchain.tools import ToolRuntime; print(inspect.signature(ToolRuntime.__init__))"
# Noldus 中间件调用面（升级后重点回归这些）
grep -rln 'AgentMiddleware\|after_model\|before_model\|jump_to\|Command(goto' packages/agent/backend/packages/harness/deerflow/agents/middlewares/
```
