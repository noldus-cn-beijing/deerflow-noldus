# Spec S1: 修 loop_detection 配置未接线（config.yaml 静默失效跑硬编码 5）

> 日期：2026-06-12
> 顺序：第 1 份（共 4 份，按序实施）。最小、最低风险，**正阻塞端到端 dogfood**，先解卡。
> 来源：dogfood thread `a5b97c00` 端到端卡死实证 + 代码坐实
> 实施方式：新开 worktree 基于最新 `dev`，单 PR

---

## 0. 背景与根因（实证，非推测）

端到端 dogfood thread `a5b97c00`：lead 应用户「你帮我看下数据 group 信息决定」逐个 `inspect_uploaded_file` 探查 28 文件分组，调到**第 5 次被强制中止**：

```
[FORCED STOP] Tool inspect_uploaded_file called 5 times — exceeded the per-tool safety limit.
```

整个分析卡死。

**真根因（代码坐实）**：`LoopDetectionMiddleware` 的 Layer 2 per-tool-type frequency 硬上限是 `_DEFAULT_TOOL_FREQ_HARD_LIMIT=5`（[loop_detection_middleware.py:69](packages/agent/backend/packages/harness/deerflow/agents/middlewares/loop_detection_middleware.py)）。两处实例化**都没接 `app_config.loop_detection`**：

| 位置 | 现状 | 后果 |
|---|---|---|
| [agent.py:292](packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py)（**lead 唯一生效路径**，make_lead_agent 自组装 middlewares） | 裸 `LoopDetectionMiddleware()` 空参 | 用硬编码默认 hard_limit=5 |
| [factory.py:283](packages/agent/backend/packages/harness/deerflow/agents/factory.py)（create_deerflow_agent 通用工厂，lead 不走但同 bug） | `from_config(LoopDetectionConfig())` 传**全新默认实例** | 同样无视 config.yaml |

`config.yaml:114-125` 明明配了 `tool_freq_warn:30 / tool_freq_hard_limit:50` + `write_todos` override，`app_config.py:119` 也有 `loop_detection: LoopDetectionConfig` 字段从 config.yaml 解析——**但两处实例化都不读它**→ 你 config.yaml 的 50 是**死配置**，实跑硬编码 5。

`inspect_uploaded_file` 没在 `tool_freq_overrides`，逐个查 28 文件分组边界本就 >5 次 → 必然第 5 次死。这是 loop-detection **误报**：把合法的「多文件密集探查」当死循环。

> **教训已落 memory**：`feedback_loop_detection_config_not_wired_runs_hardcoded_default.md`（config 字段存在 + from_config 存在，但实例化点绕过它们用默认值 = 配置静默失效的哑故障；遇「改 config.yaml 没效果」先 grep 实例化点是否真读了它）。

> **注**：本 spec 只修「config 没接线」+「inspect 加 override」。分组探查为何走单文件逐个试探（而非一次批量提取）是**第二层根因**，归 Spec S2（identify 工具返回分组字段）。两者正交、按序实施。

---

## 1. 修复 A：两处实例化接 `app_config.loop_detection`

### A1. lead 路径（必须，解 dogfood 卡死）
[agent.py:292](packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py)：
```python
# 改前
loop_detection = LoopDetectionMiddleware()
# 改后
loop_detection = LoopDetectionMiddleware.from_config(get_app_config().loop_detection)
```
- `get_app_config` 已在 [agent.py:44](packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py) import、line 288 已在用，直接可调。
- `from_config`（[loop_detection_middleware.py:251](packages/agent/backend/packages/harness/deerflow/agents/middlewares/loop_detection_middleware.py)）接受 `LoopDetectionConfig`，`app_config.loop_detection` 正是该类型（[app_config.py:119](packages/agent/backend/packages/harness/deerflow/config/app_config.py)）。
- 改后 `loop_detection` 实例仍按原样 `:348` append、`:297` 传给 summarization，**不动其余逻辑**。

### A2. 通用工厂路径（顺手修，同 bug）
[factory.py:283](packages/agent/backend/packages/harness/deerflow/agents/factory.py)：
```python
# 改前
chain.append(LoopDetectionMiddleware.from_config(LoopDetectionConfig()))
# 改后
from deerflow.config.app_config import get_app_config
chain.append(LoopDetectionMiddleware.from_config(get_app_config().loop_detection))
```
- 删掉 `from deerflow.config.loop_detection_config import LoopDetectionConfig`（不再需要 new 默认实例）。
- **惰性 import**：`get_app_config` 放在该分支函数体内 import（factory.py 是核心模块，遵「import 别放顶层防闭环」铁律，见 backend/CLAUDE.md 同步规则末尾）。

---

## 2. 修复 B：`inspect_uploaded_file` 加 per-tool override（双保险）

即使 A 让全局 hard_limit 变 50，仍给 `inspect_uploaded_file` 显式高阈值——它是「多文件合法密集调用」工具（同 `read_file`），不该受通用频率限制误杀。

在 `config.yaml:115` 的 `loop_detection.tool_freq_overrides` 下加：
```yaml
loop_detection:
  tool_freq_overrides:
    write_todos:
      warn: 2
      hard_limit: 4
    inspect_uploaded_file:        # 新增：多文件分组探查合法，不限频
      warn: 50
      hard_limit: 100
```
- 阈值取 100：远超任何真实数据集文件数（28 是当前最大批次），但仍有上界防真死循环。
- override 经 `from_config`（A 修复后）真正生效——A 是 B 生效的前提，故 A+B 同一 PR。

> **可选**：`read_file` 当前也没单独 override（走全局），A 修复后全局变 50 够用；若行为学数据集将来 >50 文件，再给 read_file 加 override。本次不做，记一句即可。

---

## 3. 测试（TDD）

### 单元测试（harness）— 新建 `tests/test_loop_detection_config_wiring.py`
1. **接线测试（红→绿）**：构造一个 `app_config.loop_detection.tool_freq_hard_limit=50` 的配置，断言 `make_lead_agent` 链里的 `LoopDetectionMiddleware` 实例的 `tool_freq_hard_limit == 50`（不是 5）。
   - 改前红（实例是空参 default 5），改后绿。
   - 取实例方式：可 mock `get_app_config` 返回自定义 config，调组装函数后从 middlewares 列表里捞 `LoopDetectionMiddleware` 实例查其属性。
2. **override 生效测试**：断言 `inspect_uploaded_file` 在 `_tool_freq_overrides` 里且 hard_limit==100（经 config 解析后）。
3. **factory 路径测试**：`create_deerflow_agent` 组装的链里 `LoopDetectionMiddleware` 也接了 app_config（不是 default）。

### 行为测试（复现 dogfood）
4. 构造一个连续调 `inspect_uploaded_file` 6 次（不同文件参数）的消息序列，喂给 `LoopDetectionMiddleware._track_and_check`（用接了 config 的实例），断言**第 6 次不触发 FORCED STOP**（override hard_limit=100）。改前（hard_limit=5）第 5 次就 stop。

### 回归
5. 现有 loop_detection 测试全绿（`pytest tests/ -k loop_detection`）——确认接 config 不破坏 Layer 1 hash 检测、`task:subagent` 分桶、`write_todos` override 等既有行为。

---

## 4. 验证（端到端）
```bash
cd /home/wangqiuyang/noldus-insight/packages/agent/backend
# 接线 + 行为测试
PYTHONPATH=. .venv/bin/python -m pytest tests/test_loop_detection_config_wiring.py -q
# loop_detection 全套回归
PYTHONPATH=. .venv/bin/python -m pytest tests/ -k "loop_detection" -q
# 改了 agents/ + factory 核心 → 裸导入两生产入口（conftest mock 藏循环导入）
PYTHONPATH=. .venv/bin/python -c "import app.gateway"
PYTHONPATH=. .venv/bin/python -c "from deerflow.agents import make_lead_agent"
# 全量回归（基线 3-6 failed 已知债，勿归因本次）
PYTHONPATH=. .venv/bin/python -m pytest -q
```
最终：复跑端到端 dogfood，lead 探查分组时 `inspect_uploaded_file` 可调到远超 5 次不被 FORCED STOP。

---

## 5. 关键文件
- `packages/agent/backend/packages/harness/deerflow/agents/lead_agent/agent.py`（:292 接 config）
- `packages/agent/backend/packages/harness/deerflow/agents/factory.py`（:283 接 config + 惰性 import）
- `packages/agent/config.yaml`（loop_detection.tool_freq_overrides 加 inspect_uploaded_file）
- `packages/agent/backend/tests/test_loop_detection_config_wiring.py`（新建，红→绿）

## 6. 红线（勿违）
- `from_config` 接 `app_config.loop_detection`，**不**新建 `LoopDetectionConfig()` 默认实例。
- factory.py 的 `get_app_config` import 放**函数体内**（防顶层 import 闭环，backend/CLAUDE.md 铁律）。
- 改 agents/ + factory 后必跑裸导入两生产入口（conftest mock 藏循环导入，pytest 假绿）。
- 只动 loop_detection 接线 + inspect override，**不**改 loop_detection 检测逻辑本身、**不**碰分组探查路径（归 S2）。
- 全量 3-6 failed 是已知基线债，勿归因本次。
