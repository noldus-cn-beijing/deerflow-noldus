# Spec：chart-maker 渲染执行解耦 —— run_chart_plan spawn 轻子进程跑批量入口（Gateway fork-free）

> 状态：**实施 spec，但含「实测前置」——必须先坐实 Gateway 被画图拖垮再动架构**
> 日期：2026-06-25
> 代码基线：dev HEAD `51f00191`
> 性质：🟡 中 · 执行架构优化（非 bug 修复）。把"画 N 张图"从「Gateway 重进程内 fork N 个 worker」改为「Gateway spawn 1 个轻子进程，子进程内开进程池」。**目的是 Gateway event loop 解耦 + 崩溃隔离强化，不是提速**。
> **方向（用户拍板 2026-06-25）**：用户提「一次画几百张图对（Gateway）进程压力太大，能否用 celery 类似手段把画图变成 bash 一次性执行、别的程序程序化工程化出图」。调研结论：**不上 celery**（broker 三件套对单机批量画图是杀鸡牛刀），改用「spawn 轻子进程跑批量入口脚本 + 子进程内进程池」= 方案 A。**但前置：先实测坐实 Gateway 真被拖垮**（当前实测 112 图仅 5s，速度非瓶颈——若 Gateway event loop 没被阻塞/其他请求没被拖慢，本 spec 不该实施）。
> **范式归属**：本 spec 是「[确定性批量执行工具 pattern](2026-06-24-deterministic-batch-execution-tool-pattern.md)」第 2 个实例（run_chart_plan）的**执行层架构演进**。pattern 的 12 铁律不变，只改「执行器宿主」——从「主进程内 ProcessPoolExecutor」演进为「主进程 spawn 子进程 + 子进程内 ProcessPoolExecutor」。
> 受保护文件（sync surgical）：`tools/builtins/run_chart_plan_tool.py`、`ethoinsight/scripts/__init__.py`（新增批量入口）。

---

## ⚠️ 〇、为什么不是 celery / 线程池 / 协程（方案选型，回应用户调研）

用户问「celery 类似手段」。下表是完整选型，**结论=方案 A（spawn 轻子进程），不上 celery**：

| 方案 | 多核并行 | Gateway 解耦 | 崩溃隔离 | 新依赖 | 部署复杂度 | 结论 |
|---|---|---|---|---|---|---|
| **当前**：Gateway 内 ProcessPoolExecutor | ✅ | ❌（fork 重父进程）| worker 进程 | 0 | 不变 | 现状，痛点=fork 重父进程 |
| **方案 A**：spawn 轻子进程 + 子进程内进程池 | ✅ | ✅（Gateway fork-free）| 子进程（多一层）| 0 | 不变 | **✅ 推荐** |
| celery（broker + worker + result backend）| ✅ | ✅ | worker 进程 | redis/rabbitmq+celery | +broker 一套 | ❌ 杀鸡牛刀 |
| 线程池（ThreadPoolExecutor）| ❌（GIL）| 部分 | ❌（segfault 带走 Gateway）| 0 | 不变 | ❌ GIL+matplotlib |
| 协程池（asyncio）| ❌（单线程）| ❌（阻塞 loop）| ❌ | 0 | 不变 | ❌ CPU 密集死路 |
| 112 个独立进程（每图一个）| ✅ | ❌ | 进程 | 0 | 不变 | ❌ 创建开销爆炸 |

### 为什么不上 celery（核心反驳）
celery 解决的是「**分布式/跨机器/高吞吐任务队列**」——broker（redis/rabbitmq）+ worker 池 + result backend。本场景是「**单机一次性批量画 ~100 张图**」：
- 引入 broker 三件套 = +1 个常驻服务（redis）+ celery 依赖 + 任务序列化/反序列化 + 失败重试/任务丢失处理，**故障面全涨**。
- 当前痛点是「同进程 fork 重 worker 压力」，不是「需要分布式队列」。celery 解的病不对。
- 违背 CLAUDE.md「复用 deerflow 现成能力优先、确定性结构优先、避免新依赖」。
- celery 的 worker 也是进程池形态——**它内部还是进程池**，只是隔了个 broker。方案 A 直接拿进程池、去掉 broker，等价且零依赖。

### 为什么必须是进程池（不是单进程/线程/协程）
- **GIL**：CPython 同进程内同一时刻只有一个线程跑字节码。matplotlib savefig 是 CPU 密集（numpy→像素→png 编码），线程池拿不到多核。
- **C 扩展释放 GIL 的细节**：numpy 底层 C 在纯数值计算时 `Py_BEGIN_ALLOW_THREADS` 释放 GIL，但 matplotlib savefig 夹大量 Python 层胶水（figure 状态/坐标轴/文字布局）在 GIL 下；且交互式 backend 非主线程 savefig 崩。**线程/协程对画图是死路**。
- **崩溃隔离**：绘图脚本 C 层 segfault 时，进程池杀 worker 进程不影响父进程；线程池一个线程 segfault 带走整个 Gateway。
- **进程池 vs 单进程**：单进程串行画 112 图 ≈ 4.5s 没用多核；进程池 N worker（N=核数）并行吃满多核。
- **进程池 vs 112 进程**：进程创建有成本（fork 页表/spawn pickle+冷启 import matplotlib），池复用固定 N 个 worker 省「建 112 个进程」的开销。

**结论**：池类型选进程池（不变），改的是「池开在哪个进程里」——从 Gateway 重进程，挪到 spawn 的轻子进程。

### 当前实测数据（2026-06-25，真实 plan 112 图）
| 指标 | 当前架构（Gateway 内进程池）| 备注 |
|---|---|---|
| 112 图渲染耗时 | **5.0s**（0.04s/图）| 速度**非瓶颈** |
| parent peak RSS | 84MB（heavy 模拟 Gateway 预 import fastapi/langgraph）| |
| child peak RSS | 260MB | worker 内 matplotlib 内存 |
| MAX_WORKERS | `os.sched_getaffinity` 核数 | 受控，非 112 |

**关键判读**：当前架构**速度没问题**（5s）。本 spec 的价值不在提速，在 **(a) Gateway event loop 不被画图占住 (b) fork 重页表成本消失 (c) 崩溃隔离多一层**。这三点是否值得动架构，**取决于 Gateway 是否真被拖垮**——见 §一 实测前置。

---

## ⚠️ 一、实测前置（必须先做，决定 spec 是否实施）

按 [[feedback_align_template_must_verify_hidden_premise_not_just_shape]] + [[feedback_isolate_root_cause_before_stacking_fallback_mechanisms]]：**本 spec 的前提「一次画几百张图对 Gateway 压力太大」尚未坐实**。当前实测 5s 跑完，无证据 Gateway event loop 被阻塞。**实施前必须先跑下列实测**，若全部通过才实施本 spec；任一不成立则 spec 转入「暂缓」并记原因。

### 前置实测 P1：Gateway event loop 阻塞（核心判据）
在 `make dev` 真实 Gateway 下，跑一次 112 图 dogfood，**同时**发一个轻量健康请求（如 `GET /health` 或 `GET /api/models`），量：
- 画图期间健康请求的 **P99 延迟**。若从 ~10ms 飙到 >1s → event loop 被阻塞 → 前置成立。
- 当前 `run_chart_plan` 是**同步工具**（非 async），它在 Gateway 的某个线程跑。需确认：它是跑在 event loop 线程（阻塞 loop）还是 thread pool 线程（不阻塞 loop）。grep `run_chart_plan` 的调度路径确认。

### 前置实测 P2：fork 重父进程的真实成本
对比「Gateway-loaded 进程 fork worker」vs「轻子进程 fork worker」的：
- fork 单次延迟（`time` 包 `pool.submit` 首次冷启）。
- 峰值内存（Gateway RSS 在画图期间涨幅）。
- 若 Gateway RSS 涨幅 <50MB 且 fork 延迟 <100ms → fork 成本可接受 → 前置弱化。

### 前置实测 P3：多 thread 并发场景
Gateway 多线程并发跑多 thread subagent。若两个 thread 同时画图（两个进程池抢核）→ 实测是否互相拖慢/内存爆。若并发画图是真实场景且会拖垮 → 前置强化（方案 A 让每个画图在独立子进程，天然隔离）。

### 前置结论判据
- P1 成立（event loop 阻塞）→ **强实施信号**，本 spec 实施。
- P1 不成立 + P3 成立（并发拖垮）→ **中实施信号**，本 spec 实施（解耦价值在并发隔离）。
- P1/P2/P3 全不成立 → **暂缓**，当前架构已够，本 spec 转待办，记「实测未坐实 Gateway 被拖垮」。

> **接手 agent 第一步就是跑 P1/P2/P3 并把结果填回本节**，再决定是否继续 §二之后的实施。

---

## 二、方案 A 设计（前置通过后实施）

### 架构

```
Gateway 进程（重：FastAPI+langgraph+模型 client，1 个 GIL，跑 event loop）
  │  run_chart_plan 工具内：
  │  asyncio.create_subprocess_exec / subprocess.run（同步工具用后者）
  ▼
render_plan 子进程（轻：只 import ethoinsight 入口，无 FastAPI/langgraph）
  │  开 ProcessPoolExecutor（fork 自这个轻进程 → worker 页表小）
  ▼
N 个 worker（N=核数，复用跑 112 个任务，matplotlib 只在 worker import）
```

### 新增：批量渲染入口脚本

`packages/ethoinsight/ethoinsight/scripts/_render_plan.py`（新）：

```python
"""批量渲染入口：读 plan_charts.json，进程池并行跑全部绘图脚本。
供 run_chart_plan 工具 subprocess 调用（Gateway fork-free）。
DEERFLOW_PATH_* env 由父进程（run_chart_plan 工具）透传，本脚本开池前设进 os.environ。"""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor

def main(argv: list[str]) -> int:
    plan_path = argv[1]
    results_path = argv[2]   # 子进程把每个 task 的 (rc, err) 写这里，供 Gateway 核盘
    plan = json.loads(open(plan_path).read())
    charts = plan["charts"]
    # 开池前确保 env 已设（父进程透传的 DEERFLOW_PATH_* + MPLBACKEND）
    from ethoinsight.scripts._common_runner import _run_chart_task, _worker_init  # 复用现有
    path_env = {k: os.environ[k] for k in os.environ if k.startswith("DEERFLOW_PATH")}
    pool = ProcessPoolExecutor(max_workers=int(os.environ.get("RENDER_MAX_WORKERS","0")) or _available_cpus(),
                               initializer=_worker_init, initargs=(path_env,))
    results = {}
    fut_map = {pool.submit(_run_chart_task, c["script"], _pre_resolved_args(c), c.get("id","")): idx
               for idx, c in enumerate(charts)}
    for fut, idx in fut_map.items():
        try:
            tid, rc, err = fut.result(timeout=120)
            results[idx] = {"tid": tid, "rc": rc, "err": err}
        except Exception as e:
            results[idx] = {"tid": "?", "rc": 1, "err": str(e)}
    pool.shutdown(wait=True)
    json.dump(results, open(results_path,"w"))
    return 0 if all(r["rc"]==0 for r in results.values()) else 1
```

> **复用现有 worker**：`_run_chart_task` / `_worker_init` 从 `run_chart_plan_tool.py` 抽到 `ethoinsight/scripts/_common_runner.py`（模块级，pickle 友好）。`run_chart_plan_tool.py` 改为 `subprocess.run` 调本入口，不再自己开池。

### 改 `run_chart_plan_tool.py`（执行层换宿主）

Step 6 `_execute_tasks` 生产路径改为 subprocess 调入口脚本：

```python
# Step 6（改）：spawn 轻子进程跑批量入口（Gateway fork-free）
import subprocess, tempfile
results_path = workspace / ".render_results.json"
env = {**os.environ, **path_env, "MPLBACKEND": "Agg",
       "RENDER_MAX_WORKERS": str(MAX_WORKERS),
       "PYTHONPATH": ethoinsight_py_root + os.pathsep + os.environ.get("PYTHONPATH","")}
proc = subprocess.run(
    [sys.executable, "-m", "ethoinsight.scripts._render_plan",
     str(plan_real_path), str(results_path)],
    env=env, capture_output=True, timeout=PER_BATCH_TIMEOUT_SECONDS,
)
results = json.loads(results_path.read_text())   # {idx: {tid, rc, err}}
```

**核盘（Step 7）+ 封存（Step 9）不变**——仍在 Gateway 进程读 `results` + 磁盘 exists 核盘 + `_seal_handoff_to_workspace` 封存。**保留"磁盘是真相"+"封存只允许一次"（见 spec A M1）两条核心不变式**。

### 关键不变式（必须保留）
1. **核盘靠磁盘**：Step 7 仍逐个 `replace_virtual_path(output).exists()`，不信子进程自报。
2. **封存在 Gateway**：`_seal_handoff_to_workspace` 仍在 Gateway 进程调（sealed_by=run_plan），子进程不封存。
3. **results key 唯一**（spec A M2）：入口脚本 results 按 index 存，不按 id。

---

## 三、风险（重点：用户担心的「存储位置不对等」）

### 风险 1（最高）：`/mnt` 虚拟路径在子进程里解析不了 —— 用户的「存储位置不对等」直觉正确

**这是本 spec 最大风险**。当前 `run_chart_plan` 能 resolve `/mnt/user-data/...` 是靠 `_worker_init` 把 `DEERFLOW_PATH_*` env 注入每个 worker。换 subprocess 后：

- spawn 的子进程**继承父进程 env**（`subprocess.run(env=...)` 显式传）✅，但子进程**内部再 fork 的 worker** 需要子进程先把 env 设进 `os.environ`，initializer 才能拿到。
- 若 `DEERFLOW_PATH_*` 没透传到子进程 → worker 调 `resolve_sandbox_path("/mnt/...")` 静默退化 → **112 张图全 FileNotFoundError**（实测脚本首次跑 light 模式正是此错）。
- 这正是 memory 铁律 [[feedback_run_metric_plan_inprocess_scripts_dont_resolve_sandbox_path]] + [[feedback_run_metric_plan_step8_validation_not_scoped_path_env]]：**argv 虚拟路径谁解析？新进程不调 resolve 就静默退化**。

**修法（写死在 spec）**：
1. `run_chart_plan` 工具构造 `env = {**os.environ, **path_env}`（`path_env` = `_build_path_env(thread_data)`），显式传给 `subprocess.run(env=env)`。
2. 入口脚本 `_render_plan.py` 开池**前** `os.environ.update(path_env)`（从自己 env 取 DEERFLOW_PATH_* 设回 os.environ，供 initializer）。
3. **F1 预解析不变**：argv 仍由 `run_chart_plan` 工具 `replace_virtual_path` 预解析成物理路径再传（见 [L202](../../../packages/agent/backend/packages/harness/deerflow/tools/builtins/run_chart_plan_tool.py#L202)），子进程拿到的 args 已是物理路径——**双保险**（即使 env 透传漏了，argv 已物理化也能跑）。
4. **TDD 覆盖**：构造「子进程 env 无 DEERFLOW_PATH」场景，断言画图仍成功（靠 F1 argv 预解析）。

### 风险 2：fork 多线程子进程的死锁
入口脚本若预 import 了带线程的库（numpy OpenMP 后端、logging），fork worker 可能死锁。
**缓解**：入口脚本保持极简（只 import ethoinsight 入口），用 `forkserver` 或保证单线程。`multiprocessing.set_start_method("forkserver")` 在入口 `__main__` 设。

### 风险 3：进程残留/僵尸
子进程崩了没 wait → 僵尸。`subprocess.run`（同步 wait）或 `timeout=` + 捕获 `subprocess.TimeoutExpired` kill。run_chart_plan 工具捕获后转 partial。

### 风险 4：IPC 开销（results 文件）
子进程把 results 写 `.render_results.json`，Gateway 读。112 entry 的 JSON 几十 KB，可忽略。比 celery 的 broker IPC 轻得多。

### 风险 5：subprocess 冷启 import 成本
`python -m ethoinsight.scripts._render_plan` 冷启要 import ethoinsight + matplotlib（~500ms-1s）。当前进程内 fork worker 无冷启（fork 复用父 import）。**方案 A 多付一次冷启**，但只在每次 run_chart_plan 调用付一次（不是每图）。实测前置 P2 要量这个冷启成本是否可接受。

### 风险 6：ethoinsight 必须是 backend workspace 依赖
入口脚本是 `ethoinsight.scripts._render_plan`，subprocess 用 `sys.executable` + `PYTHONPATH` 调。确认 backend venv 装了 ethoinsight（editable）。守 memory [[feedback_ethoinsight_must_be_backend_workspace_dep]]。

---

## 四、TDD（前置通过后）

测试文件：扩 `test_run_chart_plan.py`。

### S1：子进程路径仍正确 resolve `/mnt`（风险 1 核心回归）
```python
def test_s1_subprocess_resolves_virtual_paths(ws_and_outputs, monkeypatch):
    # run_chart_plan 改 subprocess 后，plan args 的 /mnt 仍能 resolve 画图成功
    # 双保险：env 透传 + F1 argv 预解析
```

### S2：Gateway 不再 fork worker（解耦验证）
```python
def test_s2_gateway_no_direct_pool(monkeypatch):
    # mock subprocess.run，断言 run_chart_plan 走 subprocess 而非 ProcessPoolExecutor.submit
    # 即 Gateway 进程内不再直接开画图池
```

### S3：子进程崩溃 Gateway 不倒（隔离验证）
```python
def test_s3_subprocess_crash_isolated(ws_and_outputs, monkeypatch):
    # 入口脚本 sys.exit(1) / segfault 模拟 → run_chart_plan 返回 partial/failed，
    # Gateway 进程（测试进程）不崩
```

### S4：results 仍按 index 唯一（兼容 spec A M2）
```python
def test_s4_subprocess_results_index_keyed(...):
    # 子进程写 .render_results.json 按 index，Gateway 读后核盘 112 张全核
```

### S5：裸导入两入口
改 `run_chart_plan_tool.py` 后 `import app.gateway` + `make_lead_agent`。

---

## 五、实施步骤（前置通过后）

1. **前置实测**（§一 P1/P2/P3），填回结果，判定是否实施。
2. 抽 `_run_chart_task` / `_worker_init` 到 `ethoinsight/scripts/_common_runner.py`（模块级）。
3. 新增 `ethoinsight/scripts/_render_plan.py` 入口。
4. 改 `run_chart_plan_tool.py` Step 6 为 subprocess 调入口；Step 7/9 核盘封存不变。
5. TDD S1-S5 绿；裸导入两入口。
6. dogfood 验收：核进程加载新代码，画 112 图，验收 Gateway 画图期间 `/health` P99 不飙升（P1 前置的反向验证）。

---

## 六、暂缓判据与回退

- 若 §一 前置实测 P1/P2/P3 全不成立 → **本 spec 暂缓**，在本文顶部状态改「暂缓·前置未坐实」，记实测数据。当前架构（Gateway 内进程池）保留。
- 若实施后 dogfood 发现 subprocess 冷启成本（风险 5）让画图从 5s 涨到 >15s 且 P1 前置本就不成立 → **回退**到当前架构（git revert 本 spec PR）。
- **回退安全**：本 spec 不改 seal/核盘/封存不变式（都在 spec A），只改执行宿主。回退不影响 spec A 的封存只允许一次门。

---

## 七、milestone 建议

归入「chart-maker 确定性化 / 执行架构演进」track。本 spec checkpoint：「用户提『画几百张图对 Gateway 压力大，能否 celery 化』→ 调研否决 celery（broker 对单机批量画图是杀鸡牛刀）→ 选方案 A（spawn 轻子进程 + 子进程内进程池）= Gateway fork-free + 崩溃隔离强化。**但设实测前置 P1/P2/P3**：当前实测 112 图仅 5s，速度非瓶颈，必须先坐实 Gateway event loop 真被阻塞才实施（守『先坐实根因再上结构改动』）。**不是提速优化，是稳定性/解耦优化**。最大风险=`/mnt` 虚拟路径在子进程解析不了（用户『存储位置不对等』直觉对），修法=env 透传 + F1 argv 预解析双保险。」
