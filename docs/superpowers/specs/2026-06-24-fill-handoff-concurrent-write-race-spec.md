# Spec：#187 fill 工具读-改-写并发竞态 —— 同消息多 fill 抢固定 tmp 路径致 handoff JSON 损坏

> 状态：**实施 spec，可直接交付 agent 执行**
> 日期：2026-06-24
> 性质：🔴 高 · #187（data-analyst 分步填模板）的第二个实施缺陷（生产坐实）。fill 工具「读-改-写」无锁 + 固定 tmp 路径，data-analyst 同一条 AIMessage 并行发多个 fill tool_call → 抢同一个 `handoff_data_analyst.json.tmp` → 竞态 → handoff JSON 损坏/空 → fill 反复失败 → data-analyst 卡死。
> 来源：2026-06-24 EPM 28-subject dogfood（修了 #187 漏注册 BUILTIN_TOOLS 的 `6a4e4bf9` 之后，fill 真跑起来才暴露这第二层缺陷）。
> 关联：
> - 前序：`6a4e4bf9`（#187 第一个缺陷：fill/finalize 漏注册 tools.py:BUILTIN_TOOLS）。本 spec 是同次 dogfood 暴露的**第二个独立缺陷**。
> - spec 母体：`docs/superpowers/specs/2026-06-23-data-analyst-seal-stepwise-fill-template-spec.md`（分步填模板设计，本竞态是「把一次写拆成 N 次读-改-写」转换时漏掉的并发安全一环）。
> - 现成防护参照：`sandbox/tools.py:1885` 的 write_file/str_replace 已用 `get_file_operation_lock(sandbox, path)` 串行化同路径写——**fill 工具裸奔没用任何锁**。
> - 取证铁律：memory `feedback_diagnose_subagent_behavior_replay_subagent_not_dump_lead`（看 subagent 内部行为必须独立复现，不能只 dump lead）。
> - 受保护文件：`tools/builtins/seal_handoff_tools.py` 是 deerflow 定制面，sync surgical。

---

## 〇、给实施 agent 的一句话

data-analyst 的 LLM 在一条消息里**并行发多个 `fill_data_analyst_*` tool_call**（prompt 教它逐字段填，LLM 倾向批量发），它们都走「`_load_da_payload` 读 → 改 payload → `_write_da_payload`（写固定 `handoff_data_analyst.json.tmp` → `os.rename`）」，**无锁 + tmp 路径固定共享** → 竞态：`A.write_tmp → B.write_tmp(覆盖) → A.rename(移走 tmp) → B.rename(💥 FileNotFoundError: tmp 不存在)`；rename 瞬间另一 fill 的 `_load` 读到 0 字节 → `Expecting value: line 1 column 1 (char 0)`。**修法：fill/finalize 的读-改-写整体用按 workspace-path 的模块级 `threading.Lock` 串行化（fill 是 sync 工具，用 threading 不用 asyncio；fill 只有 runtime 拿不到 sandbox，不能直接复用 `get_file_operation_lock`，按 workspace/filename 自建锁）。** 必须独立复现坐实（手段 c：触发同消息多 fill 并行）。

---

## 一、根因（代码坐实 + dogfood 实证）

### 现象（2026-06-24 dogfood 实证）
```
ValueError: handoff_data_analyst.json is unreadable/invalid JSON:
            Expecting value: line 1 column 1 (char 0)      ← 读到 0 字节
FileNotFoundError: ...handoff_data_analyst.json.tmp -> handoff_data_analyst.json
            ← os.rename 时 tmp 已被另一个 fill 移走
```
data-analyst AI message #4→#11 反复 fill 失败、挣扎，**无 invalid_tool_calls、无 max_tokens 腰斩**（与 #187 第一个缺陷、与原 seal 腰斩根因都不同——这是纯并发竞态）。

### 根因（逐处核实）
1. **fill 工具读-改-写无锁**（`seal_handoff_tools.py` fill 工具体，约 L867-960）：
   ```python
   workspace = _resolve_workspace(runtime)
   payload = _load_da_payload(workspace)      # 读
   payload[field] = ...                        # 改
   _write_da_payload(workspace, payload)       # 写：tmp.write_bytes → os.rename
   ```
   三步之间无任何串行化保护。
2. **tmp 路径固定共享**（`_write_da_payload` L835 / `preset` L798）：`tmp_path = workspace / f"{_DATA_ANALYST_HANDOFF_FILENAME}.tmp"`——所有 fill 共用**同一个** `handoff_data_analyst.json.tmp`。
3. **LangChain 默认并行执行同一 AIMessage 的多个 tool_call**。data-analyst prompt 教逐字段填（key_findings / outlier_findings / method_warnings / recommendations / gate_signals…），LLM 自然在一条消息批量发多个 fill → 并行进入读-改-写。
4. **竞态序列**（两 fill A/B 并行）：
   ```
   A: tmp.write_bytes(payloadA)      # 写 tmp
   B: tmp.write_bytes(payloadB)      # 覆盖同一 tmp
   A: os.rename(tmp, final)          # 移走 tmp，final=payloadA（丢了 B 的改动）
   B: os.rename(tmp, final)          # 💥 FileNotFoundError：tmp 已不存在
   ```
   且 A.rename 与 B._load 交错时，B 读 final 读到空/半截 → `Expecting value char 0`。
   即使没崩，**也丢字段**（A 的 payload 不含 B 刚填的字段，rename 后 B 的改动被覆盖）。

### 为什么前面没复现（盲区分析）
- **单元测试**：串行调 fill，从不并发 → 竞态不触发。
- **第一次 dogfood**：#187 漏注册 BUILTIN_TOOLS（`6a4e4bf9` 前），fill 根本没被调用 → 竞态没机会暴露。修注册后 fill 真跑，**第二层缺陷才浮现**。
- **本质**：分步填模板把「一次 seal 写」拆成「N 次读-改-写」，引入了原 seal（一次性终结调用、不与自己并发）没有的并发面。设计转换时漏了并发安全。

---

## 二、方案（按 workspace-path 的模块级锁串行化读-改-写）

### 核心
> data-analyst handoff 的**所有读-改-写操作**（3 个 fill + finalize + preset 的写）按 `(workspace, filename)` 用模块级 `threading.Lock` 串行化。同一时刻只有一个操作能进读-改-写临界区 → 无竞态、无丢字段、无 tmp 抢占。

### 为什么用模块级 threading.Lock，不复用 get_file_operation_lock
- `get_file_operation_lock(sandbox, path)`（`sandbox/file_operation_lock.py`）按 `(sandbox_id, path)` 建 threading.Lock，**但要 sandbox 对象**。fill 工具只有 `runtime`，经 `_resolve_workspace(runtime)` 拿 workspace path，**拿不到 sandbox** → 不能直接复用。
- fill 是 **sync 工具**（`def`，非 `async def`）→ 用 `threading.Lock`（不是 asyncio.Lock）。
- 自建一个同款的按 path 锁注册表（WeakValueDictionary + guard lock，照 `file_operation_lock.py` 模式），key = `str(workspace / filename)`。

### 实现要点
1. 在 `seal_handoff_tools.py` 加模块级锁注册表（照 `file_operation_lock.py` 抄）：
   ```python
   import threading, weakref
   _DA_HANDOFF_LOCKS: weakref.WeakValueDictionary[str, threading.Lock] = weakref.WeakValueDictionary()
   _DA_HANDOFF_LOCKS_GUARD = threading.Lock()

   def _get_da_handoff_lock(workspace: Path) -> threading.Lock:
       key = str(workspace / _DATA_ANALYST_HANDOFF_FILENAME)
       with _DA_HANDOFF_LOCKS_GUARD:
           lock = _DA_HANDOFF_LOCKS.get(key)
           if lock is None:
               lock = threading.Lock()
               _DA_HANDOFF_LOCKS[key] = lock
           return lock
   ```
2. fill 3 个工具 + finalize 的**读-改-写整段**包进 `with _get_da_handoff_lock(workspace):`：
   ```python
   workspace = _resolve_workspace(runtime)
   with _get_da_handoff_lock(workspace):
       payload = _load_da_payload(workspace)
       payload[field] = ...
       _write_da_payload(workspace, payload)
       progress = _da_progress(workspace)   # 读回也在锁内，保证进度一致
   return f"OK: ... progress={progress}"
   ```
3. **preset 写也进锁**（防 preset 与首个 fill 竞态——preset 在派遣路径、fill 在 subagent turn，理论上有先后但显式加锁更稳）。
4. **finalize 进锁**（finalize 读模板改 status 写——与末尾 fill 可能竞态）。

### 加固（可选但推荐）：tmp 路径加唯一后缀
- 即使有锁，给 `_write_da_payload` / `preset` 的 tmp 路径加进程内唯一后缀（如 `f"{filename}.{os.getpid()}.{threading.get_ident()}.tmp"`）作纵深——锁保证不并发，唯一 tmp 保证万一锁失效也不抢同一 tmp。**注意：用 `os.getpid()`/`threading.get_ident()`，不用 `Date.now()`/随机数**（确定性、可测）。
- 鲁棒性边界：唯一 tmp 的清理——`os.rename` 成功即消费 tmp，无残留；异常路径 tmp 残留无害（下次覆盖）。

### 边界
- **不影响其他 3 个 subagent 的 seal**：它们是一次性终结调用、各写各的 handoff 文件，不与自己并发——本锁只加 data-analyst handoff 路径，零影响。
- **锁粒度**：按 `(workspace, filename)` → 不同 thread（不同 subagent）写不同文件不互相阻塞；同一 data-analyst 内多 fill 串行（正是要的）。
- **死锁风险**：临界区内不调用任何会再取同一锁的函数（`_load`/`_write`/`_da_progress` 都不取锁）→ 无重入死锁。

---

## 三、改动清单（change manifest）

| # | 文件:锚点 | 改动 | 预期改善 | 可能回归 | 测试 |
|---|---|---|---|---|---|
| 1 | `seal_handoff_tools.py` 模块级 | 加 `_DA_HANDOFF_LOCKS` 注册表 + `_get_da_handoff_lock` | 提供按 path 锁 | — | T1 |
| 2 | `seal_handoff_tools.py` fill ×3 + finalize | 读-改-写整段包 `with _get_da_handoff_lock` | 串行化消除竞态 | 锁内异常未释放（with 自动释放，不会）| T2/T3 |
| 3 | `preset_data_analyst_template_to_workspace` | 写进锁 | preset 与首 fill 不竞态 | — | T4 |
| 4（可选）| `_write_da_payload` / preset tmp 路径 | tmp 加 pid+tid 唯一后缀 | 纵深：万一锁失效不抢 tmp | tmp 残留（无害）| T5 |

---

## 四、测试清单（TDD 红→绿，并发是重点）
1. **T1 锁注册表**：`_get_da_handoff_lock(同 workspace)` 两次返回**同一** Lock 对象；不同 workspace 返回不同。
2. **T2 并发 fill 无损（核心红→绿）**：多线程（≥8）并发对同一 workspace 各 `fill_data_analyst_text_list(不同 field)` → 全部成功、最终 handoff 含**所有** field 的值（无丢字段）、无 FileNotFoundError、无 unreadable JSON。**改前红**（无锁丢字段/崩）、**改后绿**。
3. **T3 并发 append 累加**：多线程并发 `append` 同一 field → 最终条数 = 所有 append 之和（无覆盖丢失）。
4. **T4 preset + fill 不竞态**：preset 写模板后立即多线程 fill → 模板不被读到半截。
5. **T5（若做唯一 tmp）**：并发写不抢同一 tmp 文件名（断言 tmp 名含 pid/tid）。
6. **T6 单线程不回归**：串行 fill→finalize 流程（既有 `test_da_seal_stepwise_fill.py` 全部）仍绿。
7. **T7 import 环**：改 seal_handoff_tools.py 后裸导入 `app.gateway` + `make_lead_agent` 0 退出。

> **并发测试写法**：用 `threading.Thread` + `concurrent.futures.ThreadPoolExecutor` 起 N 个线程同时调 fill，`barrier`/`event` 对齐起跑增加竞态概率；断言最终 handoff 完整。守 memory：fill 是 sync 工具，threading 复现真实并发路径。

---

## 五、验收（确定性 gate 序列）
1. manifest 完整（改动全覆盖测试）。
2. smoke：`_get_da_handoff_lock` + fill 实例化跑通。
3. 红→绿：**T2/T3 改前必红**（无锁并发丢字段/崩，复现 dogfood `Expecting value char 0` + `tmp FileNotFoundError`）、改后绿。这是本 spec 成败判据。
4. 回归（seesaw）：`test_da_seal_stepwise_fill.py` 全部 + seal_handoff_tools 邻域 + tools 邻域；backend 全量（守已知污染基线）。
5. import 环：T7 两入口 0 退出。
6. **独立复现坐实（取证铁律）**：手段 c 起一个真实 data-analyst（或直接多线程压 fill），复现 dogfood 的并发损坏 → 加锁后消失。守 memory `feedback_diagnose_subagent_behavior_replay_subagent_not_dump_lead`：看 subagent 内部必须独立复现，不能只凭 dump 推断竞态。
7. **端到端 dogfood**：同份 28-subject EPM 复跑，data-analyst 一次派遣内 fill 全成功、finalize 落盘、handoff 字段完整、lead 搬运判读（非降级、非卡死）。

---

## 六、风险与三大病理自检
1. **Reward hacking**：本 spec 不涉 LLM 行为，纯并发安全；finalize gate 仍核 key_findings 非空（不松动）。
2. **Catastrophic forgetting**：加锁不改 fill/finalize 的数据语义（读-改-写逻辑不变，只串行化）；`test_da_seal_stepwise_fill.py` 全量守不回归。
3. **Under-exploration**：这是结构修复（加锁），不是 prompt 规则——合规。

### 与 #187 两个缺陷的同源教训
#187 实施暴露**两个独立缺陷**：① fill/finalize 漏注册 BUILTIN_TOOLS（`6a4e4bf9` 修，装配链）；② 本 spec 的读-改-写并发竞态（设计转换漏并发安全）。**共性**：分步填模板把「一次性 seal」拆成「多工具、多次读-改-写」，引入了原设计没有的两个面（工具装配面 + 并发面），实施时都漏了。教训：**把单步操作拆成多步/多工具时，显式检查新引入的装配面与并发面**（memory `feedback_tool_definition_export_not_equal_registered_in_builtin_tools` + 本 spec）。

---

## 七、守的铁律
- import 环：改 seal_handoff_tools.py 后裸导入两入口 0 退出。
- 确定性可测：tmp 唯一后缀用 pid/tid，不用 `Date.now()`/随机数。
- 不引入 HarnessX 机制：threading.Lock 是 Python 原生 + 照 deerflow 现有 `file_operation_lock.py` 模式。
- 受保护文件 sync surgical：seal_handoff_tools.py。
- TDD 强制 + 并发红验证（T2/T3 改前必红=复现 dogfood）。

---

## 八、关键代码锚点
- `tools/builtins/seal_handoff_tools.py`：fill 3 工具(867/910/961)、finalize(986)、`_load_da_payload`(806)、`_write_da_payload`(826，tmp 固定路径 L835)、`preset_data_analyst_template_to_workspace`(756，tmp 固定路径 L798)、`_da_progress`(842)、`_DATA_ANALYST_HANDOFF_FILENAME`(729)
- `sandbox/file_operation_lock.py`：现成按 path 锁的**抄写模板**（WeakValueDictionary + guard lock）
- `sandbox/tools.py:1885`：write_file 用 `get_file_operation_lock` 串行化的参照（证明「同路径写要锁」是仓库既有纪律，fill 漏了）
- `subagents/executor.py:1683`：`MAX_CONCURRENT_SUBAGENTS=3`（subagent 间并行——但各写各 handoff，本竞态是单 data-analyst 内多 fill 并行，非跨 subagent）
- dogfood 证据：`Expecting value char 0` + `.tmp -> .json FileNotFoundError`

---

## milestone 建议
「data-analyst seal 分步填模板（#187）实施收口」：① 漏注册 BUILTIN_TOOLS（`6a4e4bf9` 已修）② 读-改-写并发竞态（本 spec）。两个都是「单步拆多步」转换时漏掉的新增面。checkpoint：「#187 设计正确（旧腰斩已灭、gate 守住），但实施漏装配链 + 漏并发安全两处，dogfood 逐个坐实逐个修 → 分步填模板真正可用」。
