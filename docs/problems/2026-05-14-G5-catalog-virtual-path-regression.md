# 2026-05-14 G5 回归根因诊断：catalog.resolve 物理路径泄漏

> **状态**：systematic-debugging Phase 1 (root cause) 完成；Phase 4 fix 等用户拍板方向
> **诊断 agent**：本会话 Claude
> **关联**：dogfood 测试 thread `8ff3be6d-43b5-4724-ab09-60ce23db6f2e` 复测发现的 G5 ❌

---

## 症状

`packages/agent/backend/.deer-flow/users/.../threads/8ff3be6d.../user-data/workspace/metric_plan.json` 里 `metrics[*].output` 字段是**物理路径**：

```json
{
  "metrics": [
    {
      "output": "/home/wangqiuyang/noldus-insight/packages/agent/backend/.deer-flow/users/359636ac-.../threads/8ff3be6d-.../user-data/workspace/m_open_arm_time_ratio.json"
    }
  ]
}
```

期望：`/mnt/user-data/workspace/m_open_arm_time_ratio.json`（虚拟路径）。

commit `2eb1532a` 自带的单测 `test_catalog_resolve_paths.py` 两条全部通过：
- `test_resolve_outputs_only_virtual_paths` ✅
- `test_resolve_outputs_use_virtual_workspace_with_explicit_kwarg` ✅

所以**单测和生产环境脱节**。

---

## 根因

**Sandbox 在 bash 命令字符串层面对所有 `/mnt/user-data/...` 字面量做物理路径替换，包括 `--virtual-workspace-dir` 这个参数的值。**

### 证据链

**1. Lead 调用 CLI 时确实传了正确的虚拟路径参数**（来自 langgraph.log 实际 SandboxAudit）：

```
python -m ethoinsight.catalog.resolve --paradigm epm \
  --columns-file "/mnt/user-data/workspace/columns.json" \
  --raw-files-json "/mnt/user-data/workspace/raw_files.json" \
  --workspace-dir "/mnt/user-data/workspace" \
  --virtual-workspace-dir "/mnt/user-data/workspace"     ← 字面字符串
  --groups-file "/mnt/user-data/workspace/groups.json" \
  --output "/mnt/user-data/workspace/metric_plan.json" \
  --ev19-template PlusMaze-FewZones
```

**2. 但 sandbox 在执行命令前，跑了 `replace_virtual_paths_in_command`**（[`sandbox/tools.py:959-1004`](../../packages/agent/backend/packages/harness/deerflow/sandbox/tools.py)）：

```python
def replace_virtual_paths_in_command(command: str, thread_data) -> str:
    if VIRTUAL_PATH_PREFIX in result and thread_data is not None:
        pattern = re.compile(rf"{re.escape(VIRTUAL_PATH_PREFIX)}(/[^\s\"';&|<>()]*)?")
        def replace_user_data_match(match) -> str:
            return replace_virtual_path(match.group(0), thread_data)
        result = pattern.sub(replace_user_data_match, result)
    return result
```

**这个正则会把命令字符串中所有出现的 `/mnt/user-data/...` 子串替换成物理路径**。包括 `--virtual-workspace-dir` 后面那一坨"本应是虚拟字面量"的字符串。

**3. 所以 Python CLI 实际收到的命令行参数是**：

```
--workspace-dir          "/home/wangqiuyang/.../workspace"  ← 物理
--virtual-workspace-dir  "/home/wangqiuyang/.../workspace"  ← 也变物理了！
```

**4. resolve.py 的逻辑没问题**——`virtual_workspace_dir or workspace_dir` 兜底没用上，因为 `virtual_workspace_dir` 不是 None，但其值已经被外部"翻译"过了：

```python
def _metric_to_plan(...):
    effective_workspace = virtual_workspace_dir or workspace_dir
    output_path = str(Path(effective_workspace) / f"m_{m.id}.json")  # 拼成物理路径
```

**5. 单测能过的原因**：

`test_catalog_resolve_paths.py` 直接调 Python `resolve()` 函数，传 `virtual_workspace_dir="/mnt/user-data/workspace"`（字符串）。Python 函数边界不经过 sandbox 替换，参数原样保留——所以单测看到虚拟路径并通过。

**这是经典的"单测 mock 了一层，恰好把 bug 藏起来"反模式。** 单测覆盖了 resolve() 的 Python 逻辑，但生产路径是 `lead → bash → sandbox replace → CLI → resolve()`——单测没覆盖 sandbox replace 这一层。

---

## 受影响范围

**任何通过 bash 调用 CLI 且参数语义"必须保持虚拟"的场景都受影响。** G5 不是孤例。

类似的潜在风险点：
- `--output` 参数本身：如果某 CLI 期望接收的是"虚拟路径"用于写入 metadata（如 `metric_plan.json` 里），都会踩同样的坑
- 未来的 catalog modifier（spec §6）：如果 modifier args 含路径，会再次踩
- 任何让 lead/subagent 传递"虚拟路径作为数据而非文件操作目标"的命令，都会被替换

**核心矛盾**：sandbox 现在的 `replace_virtual_paths_in_command` 假设"命令字符串里出现的所有 `/mnt/user-data/...` 都是要访问的文件路径"。这个假设对 **文件操作参数** 是对的，对 **作为 metadata / 字符串值** 的虚拟路径是错的。

---

## 修复方向（候选，待用户拍板）

| 方案 | 改动位置 | 利弊 |
|---|---|---|
| **A. CLI 加 `--virtual-workspace-dir-env` 改用环境变量** | `ethoinsight/catalog/cli.py` + sandbox 已注入的 `DEERFLOW_PATH_MNT_USER_DATA_WORKSPACE` env var | 利：复用 sandbox 已有 env var 注入机制（[`tools.py:507-520`](../../packages/agent/backend/packages/harness/deerflow/sandbox/tools.py)），干净。弊：CLI 接口变。需要更新 lead prompt 引导其不再传 `--virtual-workspace-dir`，改为完全依赖 env。 |
| **B. 在 CLI 出口 reverse-map 物理路径回虚拟路径** | `ethoinsight/catalog/cli.py` 写出 plan.json 前用 env var 反向替换 | 利：CLI 接口不变，lead prompt 不动。弊：CLI 实现里有"知道 sandbox 物理路径"的耦合（虽然只用 env var），轻微违反"ethoinsight 不知道 deerflow 存在"边界 |
| **C. 修 sandbox `replace_virtual_paths_in_command` 让它跳过被引号包裹的参数值** | `sandbox/tools.py:959-1004` | 利：根治所有同类问题。弊：bash 解析很复杂，shell 引号识别不可靠（已有 `_split_shell_tokens` 用 `shlex` 不识别 heredoc，类似问题），改动深 |
| **D. 让 lead 写 `metric_plan.json` 前后用 sed 反替换** | lead prompt 加一步后处理 | 弊：违反"lead 不操心物理细节"哲学，prompt 复杂度增加。**不推荐** |
| **E. 让 catalog.resolve 直接用环境变量构造 output，根本不接受 `--virtual-workspace-dir` 参数** | `ethoinsight/catalog/resolve.py` + `cli.py` | 利：最简单。CLI 必须运行在 sandbox 注入了 env var 的环境下；非 sandbox 跑（如直接命令行）需 fallback。弊：catalog 模块假设了运行环境，可单测性下降 |

**我的初步偏好**：**方案 A**——它和 spec §10.3.1 的"未来允许 subagent 自由写脚本时改用 `DEERFLOW_PATH_*` env var"路径一致，是同一套修复哲学的早期落地。修一个用例，下次再来类似 bug 用同样模板修。

但方案 A 有一个细节风险：**lead prompt 里 `python -m ethoinsight.catalog.resolve` 的示例命令需要改**——不再要求 lead 传 `--virtual-workspace-dir`。如果不改 prompt，lead 还是会传一个"虚拟路径字面量"参数，被 sandbox 替换成物理路径——但 CLI 现在不读这个参数了，所以无害。**保留参数兼容也行**。

---

## 单测设计的教训

修复时必须加一条**通过 bash sandbox 真实路径**的单测：

```python
def test_resolve_via_sandbox_bash_preserves_virtual_output(...):
    # 用真实 LocalSandbox + bash_tool 调 catalog.resolve CLI
    # 不直接调 Python 函数
    # 断言：metric_plan.json 的 output 字段是 /mnt/user-data/workspace/... 而非物理路径
```

这条单测在 G5 修复时**必须先红后绿**——证明它能复现 dogfood 看到的故障；不能再像 `test_catalog_resolve_paths.py` 那样只覆盖 Python 函数层。

---

## 与 spec 的关系

- spec §10.3.1 已论证 sandbox 路径不对称问题"在 lead 路径堵死后不可触发"——**G5 暴露的是 catalog.resolve 这条 CLI 路径上的同源问题**，spec §10.3.1 的判断**不完整**。lead 调 ethoinsight.parse / ethoinsight.catalog 的合法 bash 仍然走 sandbox replace，所以仍然会泄漏物理路径。
- 修复 G5 后建议补 spec §10.3.1 一段："CLI 接受字符串路径参数时，必须把字符串视为 env var 注入而非命令行参数——否则被 sandbox replace 后语义丢失"。

---

## 工作量估算

- 方案 A：~2-3 小时（含 cli.py 改、prompt 改、新增 1 条 e2e 单测、跑 dogfood 复测）
- 方案 B：~1-2 小时（最小改动，但有边界质疑）
- 方案 C：~1-2 天（涉及 shell parser 重写，风险高）

---

## 待用户决定

1. 修复方向选 A / B / C / E 中哪一个
2. 是否同步修补 spec §10.3.1 的论证缺陷
3. 是否在本 plan 里同时处理 G4（阶段播报失效）？G4 是 prompt 层 bug，与 G5 路径无关，可并行也可分开
