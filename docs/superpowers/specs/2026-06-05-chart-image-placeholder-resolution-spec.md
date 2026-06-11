# Spec: 图表图片占位符解析系统

**日期**: 2026-06-05
**状态**: Draft — 待实施
**作者**: Opus 4.8 审核 / Plan agent 初稿

## 1. 问题描述

### 现象

远程部署环境（ECS Docker）中，report.md 引用的图表图片全部 404，本地开发正常。

### 根因

报告撰写 subagent（report-writer）在生成 `report.md` 时，LLM 会**自行发明描述性文件名**来引用图表，而非使用 `handoff_chart_maker.json.chart_files` 中的实际文件名。

| 报告中引用的（LLM 编的） | 磁盘上实际的文件 |
|---|---|
| `epm_trajectory_control.png` | `plot_trajectory_s0.png` |
| `epm_trajectory_treatment.png` | `plot_trajectory_s1.png` |
| `epm_boxplot_open_arm_time.png` | `plot_open_arm_time_ratio_bar_s0.png` |

- 同一台 ECS 上，有 thread 碰巧用了正确文件名→200 OK；有 thread 用了发明文件名→404
- 这是 LLM 非确定性行为，prompt-only 修复不可靠
- 路径解析链（Sandbox→前端 URL→Nginx→Gateway Artifact API→文件系统）验证正确

## 2. 设计决策

### 2.1 占位符语法：`{{img:<basename>}}`

**选择 basename 而非 `chart_id:subject_index`。**

理由：
- basename 是 `handoff_chart_maker.json.chart_files` 中已存在的物理字段——模型只需**原样复制**路径的最后一段，无需反向推导命名规则
- 用 `chart_id:subject_index` 需要模型理解 catalog 命名约定（`plot_{id}_s{idx}.png`），是把一个错误源（发明名字）换成另一个更难排查的错误源（发明映射规则）
- basename 在 `/mnt/user-data/outputs/` 下全局唯一（catalog 生成 `plot_{id}{suffix}.png`，无碰撞）
- 解析器变成纯 dict lookup——不重实现 catalog 命名规则，遵循 SSOT 原则
- 与代码库中已有的 `{{handoff://X}}` 占位符约定风格一致（`handoff_placeholder_injection_middleware.py`）

### 2.2 解析位置：`seal_report_writer_handoff` 内部

**在 `_normalize_report_image_paths` 之前执行。**

理由：
- **唯一满足全部四个约束**：确定性/非 LLM/已有 workspace 访问/report-writer 的 bash 被 disallowed
- CLI 方案不可行：subagent 不能调 bash，lead 不能跨边界调 subagent 的 bash（已有记忆记录此类故障）
- Lead post-processing 不可行：重新引入非确定性，"lead 不读 handoff" 问题会再次出现
- `seal_report_writer_handoff` 本身就已有 workspace → outputs 的路径解析能力（`_resolve_workspace`），无需新增

### 2.3 错误处理：可见错误文本替换，seal 继续

- unmatched placeholder → 替换为 `[图表 'xxx' 未找到；可用: a, b, c]`
- 不中断 seal：中断会导致 handoff 缺失 → 误标 FAILED + 浪费重派遣
- 可见错误文本比静默 broken-image 提供更好的调试体验
- 匹配现有 `_normalize_report_image_paths` 的 "log + continue" 契约

## 3. 架构设计

### 3.1 两层防御

```
Layer 1（新增）: _resolve_report_image_placeholders
  → 将 {{img:<basename>}} 解析为正确的 mnt/user-data/outputs/<basename>
  → 修复根因（LLM 发明文件名）

Layer 2（保留）: _normalize_report_image_paths
  → 修正路径前缀（outputs/ → mnt/user-data/outputs/）
  → 兜底：即使 LLM 忽略新 prompt 继续写字面路径，前缀依然正确
```

两层是互补的，不可去掉 Layer 2——它是兜底机制。

### 3.2 数据流

```
chart-maker 产出
  ↓
handoff_chart_maker.json {chart_files: ["/mnt/user-data/outputs/plot_X.png", ...]}
  ↓
report-writer 读取 chart_files, 原样取 basename → 写入 report.md:
  ![轨迹图]({{img:plot_trajectory_s0.png}})
  ↓
seal_report_writer_handoff 被调用
  → _resolve_report_image_placeholders:
    1. 读 handoff_chart_maker.json → {basename: full_path} map
    2. 扫 report.md → 替换 {{img:...}} → mnt/user-data/outputs/xxx.png
  → _normalize_report_image_paths (idempotent, 已是正确格式时无操作)
  → _seal_handoff(...)
  ↓
前端 artifacts API 返回 200 OK
```

## 4. 详细实现

### 4.1 文件修改清单

| 文件 | 操作 | 描述 |
|------|------|------|
| `packages/agent/backend/packages/harness/deerflow/tools/builtins/seal_handoff_tools.py` | MODIFY | 新增 `_IMG_PLACEHOLDER_RE`、`_load_chart_files_map()`、`_resolve_report_image_placeholders()`；在 `seal_report_writer_handoff` 的 `_normalize_report_image_paths` 调用前插入 placeholder 解析 |
| `packages/agent/backend/packages/harness/deerflow/subagents/builtins/report_writer.py` | MODIFY | 三处 prompt 更新：`<contract>` 段、`§3.4 图表` 段、`<optional_chart_handoff>` 段 |
| `packages/agent/backend/tests/test_seal_handoff_tools.py` | MODIFY | 新增 `TestResolveReportImagePlaceholders` 测试类 |

### 4.2 新增函数

#### `_IMG_PLACEHOLDER_RE`

```python
_IMG_PLACEHOLDER_RE = re.compile(r"\{\{img:([^}]+)\}\}")
```

放在 `_BAD_IMG_PATH_RE` 旁边（当前 line 146）。

#### `_load_chart_files_map(workspace: Path) -> dict[str, str]`

```python
def _load_chart_files_map(workspace: Path) -> dict[str, str]:
    """Return {basename: full_virtual_path_lstrip_slash} from handoff_chart_maker.json.

    Returns empty dict when file absent, unparseable, or chart_files empty.
    The value has its leading '/' stripped — artifacts API requires no leading slash.
    """
    chart_handoff = workspace / "handoff_chart_maker.json"
    if not chart_handoff.exists():
        return {}
    try:
        data = json.loads(chart_handoff.read_text(encoding="utf-8"))
        chart_files = data.get("chart_files", [])
        if not isinstance(chart_files, list):
            return {}
        result: dict[str, str] = {}
        for f in chart_files:
            if isinstance(f, str):
                result[Path(f).name] = f.lstrip("/")
        return result
    except Exception:
        return {}
```

**注意**: value 需要 `lstrip("/")` —— `chart_files` 条目以 `/mnt/...` 开头，但 artifacts API 要求无前导斜杠。

#### `_resolve_report_image_placeholders(report_host_path: Path, workspace: Path) -> None`

```python
def _resolve_report_image_placeholders(
    report_host_path: Path,
    workspace: Path,
) -> None:
    """Resolve {{img:<basename>}} placeholders in report.md.

    Reads handoff_chart_maker.json from workspace, builds a {basename: full_path}
    mapping, then replaces every {{img:<basename>}} placeholder in the report
    with the canonical virtual path.

    Unmatched basenames → replaced with visible error stub listing available files.
    Missing/empty chart_files → no-op (placeholders survive for human diagnosis).
    """
    if not report_host_path.is_file():
        return

    chart_files_map = _load_chart_files_map(workspace)

    try:
        original = report_host_path.read_text(encoding="utf-8")

        def _replace(match: re.Match[str]) -> str:
            basename = match.group(1).strip()
            if not chart_files_map:
                return match.group(0)          # 无映射时保留原样
            if basename in chart_files_map:
                return chart_files_map[basename]
            # 不匹配 → 可见错误文本
            available = ", ".join(
                sorted(chart_files_map.keys())[:5]
            )
            suffix = f"；可用: {available}" if available else ""
            return f"[图表 '{basename}' 未找到{suffix}]"

        resolved = _IMG_PLACEHOLDER_RE.sub(_replace, original)

        if resolved != original:
            report_host_path.write_text(resolved, encoding="utf-8")
            logger.info(
                "seal_report_writer_handoff: resolved image placeholders in %s",
                report_host_path,
            )
    except Exception:
        logger.warning(
            "seal_report_writer_handoff: image placeholder resolution skipped",
            exc_info=True,
        )
```

### 4.3 Call site 修改

在 `seal_report_writer_handoff()` 中（当前 line 489），在 `_normalize_report_image_paths` 之前插入：

```python
# 0. 解析 {{img:...}} 占位符（chart image placeholder system）
try:
    _ws = _resolve_workspace(runtime)
    _report_host = _ws.parent / "outputs" / Path(report_path).name
    _resolve_report_image_placeholders(_report_host, _ws)
except Exception as _e:
    logger.warning("seal_report_writer_handoff: image placeholder resolution failed: %s", _e)

# 1. 规范化图片路径前缀（现有逻辑，保留）
try:
    _ws = _resolve_workspace(runtime)
    _report_host = _ws.parent / "outputs" / Path(report_path).name
    _normalize_report_image_paths(_report_host)
except Exception as _e:
    logger.warning("seal_report_writer_handoff: image normalisation pre-step failed: %s", _e)
```

### 4.4 Prompt 修改

#### `<contract>` 段（当前 line ~52）

将图表路径说明替换为占位符语法说明：

```
图表已由 chart-maker 生成。在 report.md 中引用图表时，必须使用**占位符语法**
`{{img:<文件名>}}`。文件名必须从 handoff_chart_maker.json 的 chart_files 数组中
逐条原样复制路径的最后一段（最后一个 `/` 之后的部分），不得自己编造文件名。
系统会在封存 handoff 时自动将占位符解析为正确的前端路径。
```

#### `§3.4 图表` 段（当前 line ~95）

替换为占位符用法示例：

```
#### 3.4 图表

图表引用使用**占位符语法** `{{img:<文件名>}}`。从 handoff_chart_maker.json 的
chart_files 数组中取每个路径的文件名部分，原样填入占位符。

操作步骤：
1. read_file /mnt/user-data/workspace/handoff_chart_maker.json
2. 看 chart_files 数组，提取文件名（路径最后一段）
3. 在 report.md 中写成 `{{img:<文件名>}}`

✅ 正确：`![Figure 1: 轨迹图]({{img:plot_trajectory_s0.png}})`
✅ 正确：`![Figure 2: 箱线图]({{img:plot_box_open_arm.png}})`
❌ 错误：`![Figure 1: 轨迹图](mnt/user-data/outputs/epm_trajectory_control.png)` ← 文件名是编的
❌ 错误：`![Figure 1: 轨迹图](outputs/plot_trajectory_s0.png)` ← 不要自己写路径
```

#### `<optional_chart_handoff>` 段（当前 line ~217）

```
<optional_chart_handoff>
如果 lead 在 task prompt 中包含 handoff_chart_maker.json，read_file 读取。
chart_files 中是 /mnt/user-data/outputs/ 开头的虚拟路径。

**在 report.md 中引用图表时，必须使用 {{img:<文件名>}} 占位符**——取每个路径的文件名
部分原样填入。系统会在封存时自动将占位符解析为正确的路径。

若 handoff_chart_maker.json 不存在或 chart_files 为空，Figures section 写
"(无可视化输出)"，不报错。
</optional_chart_handoff>
```

## 5. 边界情况

| 边界情况 | 处理 |
|----------|------|
| 占位符含空格: `{{img: plot_X.png }}` | `.strip()` 在解析器中处理 |
| 文件名含 `}` | 正则 `[^}]+` 在首个 `}` 处停止；实际文件名不含 `}` |
| 两个 chart_files 同名（不同目录） | 不会发生——全部在同一 `/mnt/user-data/outputs/` 前缀下 |
| 非 ASCII 文件名 | `Path(f).name` 正确处理 Unicode |
| `chart_files` 不是 list | 类型守卫返回 `{}` |
| handoff JSON 损坏 | try/except 返回 `{}` |
| 占位符在代码块内 | 仍然解析（可接受——代码块不应含图片占位符） |
| 占位符不存在但模型写了字面路径 | Layer 2 `_normalize_report_image_paths` 兜底修前缀 |
| report.md 由 write_file chunked 写入 | 不相关——占位符解析在 seal 时一次性处理完整文件 |
| 现有已正确路径的报告 | 零影响——`_IMG_PLACEHOLDER_RE` 不匹配字面路径，`_normalize` idempotent |
| 文件已存在但路径错误（chart-maker bug） | 不检查文件存在性——若 sealed path 不指向实际文件，那是 chart-maker 的 bug |

## 6. 测试策略

### 6.1 单元测试（`test_seal_handoff_tools.py`）

新增 `TestResolveReportImagePlaceholders` 类：

| 测试 | 描述 |
|------|------|
| `test_resolves_valid_basename` | 单个占位符匹配→替换为正确虚拟路径 |
| `test_resolves_multiple_placeholders` | 多个占位符全部正确解析 |
| `test_unmatched_basename_stub` | 未知文件名→替换为错误提示，列出可用文件 |
| `test_missing_handoff_file_noop` | 无 handoff 文件→占位符保留原样 |
| `test_empty_chart_files_noop` | chart_files=[]→占位符保留原样 |
| `test_missing_report_file_noop` | report.md 不存在→静默跳过 |
| `test_basename_exact_match_only` | 部分匹配不解析（"trajectory.png"≠"plot_trajectory_s0.png"） |
| `test_idempotent_on_already_resolved` | 已正确路径不变 |
| `test_leading_slash_stripped` | `/mnt/...` → `mnt/...`（API 要求） |
| `test_whitespace_in_placeholder` | `{{img: plot_X.png }}` 正常解析 |
| `test_mixed_placeholders_and_literal` | 占位符+字面路径混合，各自正确处理 |
| `test_corrupt_handoff_json` | 损坏 JSON→保留占位符，不抛异常 |
| `test_no_placeholders_in_content` | 无意外的正则副作用 |

### 6.2 集成测试

在 `seal_report_writer_handoff` 的现有集成测试中增加场景：设置 `handoff_chart_maker.json`，写入含 `{{img:...}}` 的 report.md，调用 seal，验证 report.md 中占位符已被解析。

### 6.3 E2E 验证

1. 用测试 EPM 数据跑完整 chart-maker + report-writer 流水线
2. 验证 report.md 不含 `{{img:...}}` 残留
3. 验证所有图片路径指向实际存在的文件
4. 通过 artifacts API 访问每个图片 → 确认 HTTP 200

## 7. 关键注意事项

### 7.1 受保护文件

- `seal_handoff_tools.py` — **这是所有 subagent 共享的 handoff 密封基础设施**，修改后必须跑**全量测试**（`make test`），不能只跑新测试
- `report_writer.py` — 受保护文件（2026-06-02 锁定），sync 上游时需 surgical merge

### 7.2 回归风险

- 对不使用占位符的报告零影响（二层防御设计）
- `_normalize_report_image_paths` 行为不变
- `seal_report_writer_handoff` 接口不变
- 所有 failure path 都是非致命的

### 7.3 与现有约定的兼容性

- `{{img:...}}` 与 `{{handoff://...}}` 风格一致
- `chart_files` schema 不变
- 前端 artifacts API 路径格式不变

## 8. 实施顺序

1. `seal_handoff_tools.py`：新增函数和 call site
2. `test_seal_handoff_tools.py`：红阶段——先写测试确认全部红
3. `test_seal_handoff_tools.py`：绿阶段——实现后验证全部绿
4. `report_writer.py`：三处 prompt 更新
5. 全量 `make test` 确认无回归
6. E2E dogfood 验证
