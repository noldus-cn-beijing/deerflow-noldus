# Spec D — validate_catalog 虚拟路径解析修复（plan 内路径字符串不经 bash 替换）

> 日期：2026-06-08 ｜ 目标分支：从 `dev` 新建 worktree（独立 spec）
> 来源：EPM dogfood 三轮都出现的 `result_file_unreadable` 误报（最新 thread `7d4d9b8e`）
> 性质：ethoinsight 库 bug 修复（路径解析）。根因已完全确证。
> 这是给执行 agent 的施工单，不是给用户的总结。

---

## 0. 一句话目标

`validate_catalog` 把指标输出文件误报为 `result_file_unreadable`（5/5 全报，三轮 dogfood 都出现），**不是文件权限问题**（Spec1 的 chmod 0o644 已让文件变 644，已核实），而是 **validate_catalog 从 plan JSON 内部读出的 `/mnt/user-data/workspace/m_*.json` 虚拟路径字符串没被解析成真实路径，直接 `Path("/mnt/...").read_text()` → OSError**。

**修复**：新增"虚拟路径 → 真实路径"解析 helper（用 sandbox 注入的 `DEERFLOW_PATH_*` 环境变量），让 validate_catalog 读 plan 内路径时先解析。

---

## 1. 根因（已完全确证，附证据）

### 1.1 为什么 644 文件还报 unreadable

- `validate_catalog.py:362-363`：
  ```python
  output_path = entry.get("output", "")   # 从 plan JSON 读出的字符串
  data = json.loads(Path(output_path).read_text(encoding="utf-8"))  # 直接当真实路径开
  ```
- plan_metrics.json 里 `output` = `/mnt/user-data/workspace/m_open_arm_time.json`（**虚拟沙箱路径**，已核实）。
- **`/mnt/user-data` 在宿主机不存在**（`ls /mnt/user-data` → No such file，已核实）。local sandbox 是**靠路径替换**模拟虚拟路径，不是真挂载。
- 所以 `Path("/mnt/user-data/workspace/m_*.json").read_text()` → FileNotFoundError(OSError) → `except (json.JSONDecodeError, OSError)` → 报 `result_file_unreadable`。

### 1.2 为什么 compute 脚本 / read_file 不中招，唯独 validate_catalog 中招

| 路径来源 | 是否经 bash 路径替换 | 结果 |
|---------|---------------------|------|
| compute 脚本的 `--output /mnt/...`（命令行参数） | ✅ bash 工具 `replace_virtual_path`（tools.py:475）替换命令行里的 /mnt | 脚本拿到**真实路径**，写成功 |
| code-executor `read_file /mnt/...`（工具调用） | ✅ 工具层替换 | 读成功 |
| **validate_catalog 从 plan JSON 内部读出的 `output` 字符串** | ❌ **不经任何替换**（Python 代码内部读 JSON 拿到的值，不是命令行参数） | `Path("/mnt/...")` 指向不存在的宿主机路径 → **OSError** |

**这是 D 的精确边界**：只有"从 JSON 内部读出路径字符串、再在进程内 `Path()` 打开"的代码中招；命令行 `--xxx /mnt/...` 参数不中招（被 bash 替换了）。

### 1.3 ⚠️ 更广的系统不一致（扩大 D 范围）

核实 workspace 里的路径风格，发现**同一 workspace 不同 JSON 用不同路径风格**：
- `raw_files.json`：**宿主机绝对路径**（`/home/wangqiuyang/.../uploads/...`）→ `read_inputs_json` 从中读出来能直接 Path() 打开，**碰巧不中招**
- `groups.json`：**虚拟路径**（`/mnt/user-data/uploads/...`）→ `read_groups_json` 从中读出 key 若用于打开文件，**会中招**
- `plan_metrics.json` 的 `output`：**虚拟路径** → validate_catalog **确证中招**

**结论**：这不是 validate_catalog 一家的 bug，是"plan/JSON 内路径字符串风格不统一 + 消费方不解析虚拟路径"的**系统问题**。raw_files.json 碰巧用真实路径躲过；哪天写它的代码改用虚拟路径，`read_inputs_json` 也炸。**正确修法是统一的虚拟路径解析 helper，不是只给 validate_catalog 打补丁。**

### 1.4 sandbox 已注入解析所需的 env（修复依据）

`local_sandbox.py:337-339`：sandbox 运行脚本时注入环境变量，**正是为了让脚本运行时解析虚拟路径**：
```python
# Expose path mappings as environment variables so Python scripts
# can resolve virtual paths at runtime (e.g. glob, open, etc.)
for mapping in self.path_mappings:
    env_key = "DEERFLOW_PATH_" + mapping.container_path.strip("/").replace("/", "_").replace("-", "_").upper()
    env[env_key] = mapping.local_path
```
- 即：`/mnt/user-data/workspace` → 环境变量 `DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE=<真实路径>`（注意 container_path 是 `/mnt/user-data/workspace`，strip("/") 后 `mnt/user-data/workspace` → `MNT_USER_DATA_WORKSPACE`）。
- **validate_catalog 当前没用这个 env**，所以解析不了。

> 执行 agent 务必先核实 env_key 的确切拼法：用一次真实 dogfood 或加日志打印 `os.environ` 里所有 `DEERFLOW_PATH_*`，确认 `/mnt/user-data/workspace` 对应的确切 key（是 `DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE` 还是 `DEERFLOW_PATH_USER_DATA_WORKSPACE`——取决于 container_path 是否含 `/mnt` 前缀）。这是修复成败的关键，**别凭注释猜**。

---

## 2. 改动清单

### 改动 D1：新增虚拟路径解析 helper（`_cli.py`）

**文件**：`packages/ethoinsight/ethoinsight/scripts/_cli.py`

新增一个 helper，把 `/mnt/...` 虚拟路径用 `DEERFLOW_PATH_*` env 解析成真实路径；非虚拟路径（已是真实路径，如 raw_files.json 那种）原样返回：

```python
import os

def resolve_sandbox_path(path: str | Path) -> Path:
    """把 /mnt/<x>/... 虚拟沙箱路径解析成真实路径（用 sandbox 注入的 DEERFLOW_PATH_* env）。

    设计依据：local sandbox 给脚本进程注入 DEERFLOW_PATH_<CONTAINER_PATH> 环境变量
    （见 local_sandbox.py）。命令行参数里的 /mnt 路径由 bash 工具替换，但【从 JSON
    内部读出的路径字符串】不经替换——本函数补这个解析。

    - 输入是 /mnt/... 且能匹配到 DEERFLOW_PATH_* env → 返回真实路径。
    - 输入已是真实路径（不以 /mnt 开头）→ 原样返回（Path）。
    - 输入是 /mnt 但匹配不到 env（如非沙箱环境/直接跑测试）→ 原样返回（fail-safe，
      行为与修复前一致，不引入新失败模式）。
    """
    p = str(path)
    if not p.startswith("/mnt/"):
        return Path(p)
    # 尝试从最长匹配的 container_path 前缀解析（/mnt/user-data/workspace 优先于 /mnt/user-data）
    # 遍历 DEERFLOW_PATH_* env，找 container_path 是 p 前缀的，最长优先。
    best_key = None
    best_prefix = ""
    for key, real in os.environ.items():
        if not key.startswith("DEERFLOW_PATH_"):
            continue
        # 还原 container_path：DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE → /mnt/user-data/workspace
        # ⚠️ 这个还原要和 local_sandbox.py 的生成规则【精确对称】，见 §1.4 核实
        container = "/" + key[len("DEERFLOW_PATH_"):].lower().replace("_", "/")
        if p.startswith(container) and len(container) > len(best_prefix):
            best_prefix = container
            best_key = key
    if best_key is not None:
        real_base = os.environ[best_key]
        suffix = p[len(best_prefix):].lstrip("/")
        return Path(real_base) / suffix
    return Path(p)  # 匹配不到 → 原样（fail-safe）
```

> ⚠️ **关键风险点**：`container = "/" + key[...].lower().replace("_", "/")` 这个**还原**必须和 `local_sandbox.py:338` 的**生成**精确对称。生成是 `container_path.strip("/").replace("/", "_").replace("-", "_").upper()`——注意它把 `-` 也替换成 `_`，所以**还原时无法区分原本是 `-` 还是 `/`**（信息有损）。
> - **更稳妥的方案**（推荐执行 agent 采用）：不要反向还原 container_path，而是**正向比对**——sandbox 也应能提供"container→local"映射的权威来源。若拿不到，则 helper 改为**只解析已知的固定前缀**（`/mnt/user-data/workspace`、`/mnt/user-data/uploads`、`/mnt/user-data/outputs`、`/mnt/shared`），用对应的固定 env key 直查，避免有损还原。这几个前缀是稳定契约（local_sandbox_provider.py:20 `_USER_DATA_VIRTUAL_PREFIX`）。
> - 执行 agent 二选一，但**必须实测 env key 拼法**（§1.4）。倾向固定前缀方案（确定性、不依赖有损还原）。

### 改动 D2：validate_catalog 读 plan 内路径时走 helper

**文件**：`packages/ethoinsight/ethoinsight/validate_catalog.py`

line 362-363 改为：
```python
from ethoinsight.scripts._cli import resolve_sandbox_path  # 或就近 import
...
        try:
            data = json.loads(resolve_sandbox_path(output_path).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            ...
```
即把 `Path(output_path)` 换成 `resolve_sandbox_path(output_path)`。

> 同时检查 validate_catalog 里**其他**读 plan 内路径的地方（如 line 416 读 `--plan` 本身——那个是命令行参数，经 bash 替换，**不用改**；只改"从 plan 内部读出的路径"）。

### 改动 D3（建议，治本）：read_inputs_json / read_groups_json 内的路径也走 helper

**文件**：`packages/ethoinsight/ethoinsight/scripts/_cli.py`

`read_inputs_json`（返回路径列表）和 `read_groups_json`（返回 {路径: 组名}）读出的路径若后续被打开，同样应解析。但要谨慎：
- 这两个 helper 当前只返回字符串，**解析时机**取决于调用方何时 `Path().open()`。
- **保守做法**：让 `read_inputs_json` 返回的每个路径、`read_groups_json` 返回的每个 key，都经 `resolve_sandbox_path` 解析后返回。这样调用方拿到的就是真实路径。
- **但**：raw_files.json 当前是真实路径（不以 /mnt 开头）→ helper 原样返回，无影响。groups.json 是 /mnt → 被正确解析。**向前兼容两种风格**。

> ⚠️ D3 改动面比 D1/D2 大（影响所有 compute/plot 脚本的输入读取）。执行 agent 评估：若时间紧，**D1+D2 先修 validate_catalog（确证的 bug），D3 作为同 spec 的"系统一致性"改动一起做但充分测试**。两者都做才治本（否则 groups.json 的 /mnt 路径仍是潜在雷）。

### 改动 D4：code-executor 不应把 validator 误报编码成 critical warning

**文件**：`packages/agent/backend/packages/harness/deerflow/subagents/builtins/code_executor.py`（或相关 prompt）

本次 code-executor 把 5 条 `result_file_unreadable` 编码成 `SAMPLE.FILE_UNREADABLE` critical warning ×5，污染 gate_signals（critical_count=5）。修了 D1-D3 后这些误报会消失。但**防御性**：code-executor 不该在"文件其实可读（自己刚 read_file 成功）"时仍报 critical。
> 这部分**可选**——修好 D1-D3 后 validator 不再误报，根因消除。但若想更稳，可在 code-executor prompt 提示"validate_catalog 的 result_file_unreadable 若与你已成功 read_file 的文件矛盾，按 info 而非 critical 记账"。**优先级低于 D1-D3，执行 agent 酌情**。

---

## 3. 测试（TDD，先 red）

放 `packages/ethoinsight/tests/`，新建 `test_resolve_sandbox_path.py`：

```python
import json, os
from pathlib import Path
from ethoinsight.scripts._cli import resolve_sandbox_path

class TestResolveSandboxPath:
    def test_virtual_workspace_path_resolved(self, tmp_path, monkeypatch):
        real_ws = tmp_path / "real_workspace"; real_ws.mkdir()
        # 模拟 sandbox 注入的 env（执行 agent 按【实测】的确切 key 名改）
        monkeypatch.setenv("DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE", str(real_ws))
        resolved = resolve_sandbox_path("/mnt/user-data/workspace/m_x.json")
        assert resolved == real_ws / "m_x.json"

    def test_real_path_passthrough(self):
        # 已是真实路径（raw_files.json 风格）→ 原样
        p = "/home/user/data/file.json"
        assert resolve_sandbox_path(p) == Path(p)

    def test_virtual_path_no_env_failsafe(self, monkeypatch):
        # /mnt 但无对应 env（非沙箱/测试）→ 原样返回，不抛
        for k in list(os.environ):
            if k.startswith("DEERFLOW_PATH_"):
                monkeypatch.delenv(k)
        assert resolve_sandbox_path("/mnt/user-data/workspace/x.json") == Path("/mnt/user-data/workspace/x.json")

class TestValidateCatalogReadsResolvedPath:
    def test_validator_reads_via_resolved_path(self, tmp_path, monkeypatch):
        """red 锚点：修复前 validator 读 /mnt 字符串 → unreadable；修复后经 helper 读到真实文件。"""
        real_ws = tmp_path / "ws"; real_ws.mkdir()
        # 造一个真实指标文件
        (real_ws / "m_x.json").write_text(json.dumps({"metric": "x", "value": 0.5}), encoding="utf-8")
        monkeypatch.setenv("DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE", str(real_ws))
        # 造一个 plan，output 用 /mnt 虚拟路径
        plan = {"metrics": [{"id": "x", "output": "/mnt/user-data/workspace/m_x.json",
                             "output_unit": "ratio", "subject_index": 0}]}
        # 调 validate_catalog 的核心校验函数（执行 agent 按真实函数名调）
        from ethoinsight.validate_catalog import <核心校验函数>
        violations = <核心校验函数>(plan)
        # 修复前：会有 result_file_unreadable；修复后：无（文件经 helper 读到了）
        assert not any(v["issue"] == "result_file_unreadable" for v in violations)
```
> 执行 agent：先读 validate_catalog 的真实函数结构（CLI 入口 vs 可单测的核心函数），按实际写。**务必实测 env key 拼法**后再定测试里的 `setenv` key。

---

## 4. 验收标准

1. red：修复前 `test_validator_reads_via_resolved_path` 有 `result_file_unreadable`。
2. 修复后：
   - `cd packages/ethoinsight && uv run pytest tests/test_resolve_sandbox_path.py -v` 全绿。
   - 全量：`uv run pytest tests/ -q`（确认没破坏现有 compute/plot 脚本的路径处理——D3 改了输入读取，重点回归）。
3. **dogfood 验证（关键）**：重跑 EPM，code-executor 的 gate_signals **不再有 5 条 `result_file_unreadable`**，`critical_count` 归 0（或只剩真实的）。validate_catalog 正常完成范围校验。
4. **env key 拼法实测确认**（§1.4）——这是最大风险点，必须先实测再写死。

---

## 5. 影响面与风险

- **最大风险 = env key 还原的有损性**（§2 D1 警告）。缓解：用固定前缀直查方案（确定性），不用反向还原。
- **D3 改输入读取**影响所有脚本——但 helper 对真实路径 passthrough、对 /mnt 才解析，向前兼容。充分跑 ethoinsight 全量。
- **fail-safe 设计**：匹配不到 env 时原样返回（行为同修复前），不引入新失败模式。非沙箱环境（如本地直接跑脚本）不受影响。
- 与其他 spec 正交（纯 ethoinsight 库改动）。

---

## 6. 提交

- worktree 名建议：`worktree-validate-catalog-virtual-path-fix`
- commit message（中文）：`fix(ethoinsight): validate_catalog 解析 plan 内 /mnt 虚拟路径，修 result_file_unreadable 误报`
- 全量绿 + env key 实测确认后建 PR 合入 dev。

---

## 7. 关联

- memory：`project_2026-06-08_epm_dogfood_routing_and_constitution_leak.md`（"其他观察"段，本 spec 的根因来源）
- Spec1（chmod 0o644，修了 600 但不是全部根因，本 spec 补第二根因）：`2026-06-08-handoff-status-partial-and-file-perms-spec.md`
- 路径替换机制：`sandbox/tools.py:replace_virtual_path`（475）、`local_sandbox.py` env 注入（337）、`local_sandbox_provider.py:20` 虚拟前缀契约
- 同批 spec A/B/C/E
