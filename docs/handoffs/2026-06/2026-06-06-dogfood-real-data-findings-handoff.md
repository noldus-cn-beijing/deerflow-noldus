# Handoff — Dogfood 真实数据 6 范式测试报告

> 日期：2026-06-06 ｜ 测试者：Claude (deepseek-v4-pro)
> 测试范围：`/home/wangqiuyang/DemoData/real_data/` 下全部 6 个实验数据集
> 测试方法：Playwright 浏览器自动化，端到端操作 EthoInsight 应用 (localhost:2026)
> 测试账号：qiuyang.wang@noldus.com

---

## 1. 测试概况

| # | 数据集 | 范式 | 文件格式 | 文件数 | 状态 | 结论 |
|---|--------|------|----------|--------|------|------|
| 1 | `Raw data-OFT-Xuhui-34` | OFT 旷场 | XLSX | 34 | 🔴 阻塞 | format_unrecognized |
| 2 | `Raw data-EPM-Xuhui-28` | EPM 高架十字迷宫 | XLSX | 28 | 🔴 未测(同因) | 全 XLSX，同 bug |
| 3 | `原始数据-高架十字迷宫实验` | EPM | XLSX | 2 | 🔴 未测(同因) | 全 XLSX，同 bug |
| 4 | `原始数据-悬尾实验` | TST 悬尾 | XLSX | 4 | 🔴 未测(同因) | 全 XLSX，同 bug |
| 5 | `o迷宫` | EZM 零迷宫 | XLSX | 12 | 🔴 未测(同因) | 全 XLSX，同 bug |
| 6 | `原始数据-强迫游泳实验` | FST 强迫游泳 | XLSX | 1 | 🔴 未测(同因) | 全 XLSX，同 bug |

**所有 6 个数据集均为 XLSX 格式，全部受同一 bug 阻塞，无法完成端到端分析。**

---

## 2. BUG #1 (CRITICAL, Blocking): `detect_ethovision` 不识别真实 XLSX 导出

### 症状

`identify_ev19_template` 返回 `format_unrecognized`，导致 `prep_metric_plan` 失败 → guardrail 拦截 code-executor → 管道卡死。

### 根因

`packages/ethoinsight/ethoinsight/parse/_core.py:80-93`:

```python
def _detect_ethovision_xlsx(path, sheet_name=0):
    df = pd.read_excel(path, sheet_name=sheet_name, header=None, nrows=1)
    first_cell = str(df.iloc[0, 0])
    return "标题行数" in first_cell   # ← 只查 A1 单元格
```

**真实 OFT XLSX 文件的 A1 单元格不含"标题行数"**。这些文件是从 EthoVision XT 的"按分析区输出"功能导出的 XLSX 格式——数据列直接放在第一行作为表头（`trial_time`, `recording_time`, `x_center`, `y_center`, ...），没有 EthoVision 标准 TXT 格式的 header 段。

但 `parse_header` 是能解析这些文件的（`inspect_uploaded_file` 成功返回了完整列结构），说明 **parse 层支持这种格式，但 detect 层不认**——两个函数对 XLSX 格式的标准不一致。

### 影响范围

- **所有 6 个真实数据集**无法分析
- 任何从 EthoVision XT "按分析区输出"导出的 XLSX 文件都会触发此 bug
- 这可能是用户最常见的上传格式（XLSX 比 TXT 更方便）

### 修复方向

1. **方案 A（推荐）**: 扩展 `_detect_ethovision_xlsx` 的检测逻辑——不仅检查 A1 的"标题行数"，也检查表头行是否包含 EthoVision 特征列（`trial_time`/`recording_time`/`x_center`/`y_center` 等的任一组合）
2. **方案 B**: `identify_ev19_template` 在 `format_unrecognized` 时不要立即返回 error，而是把 `inspect_uploaded_file` 的列结构传给 ev19_facts 尝试匹配
3. **方案 C**: 让 `detect_ethovision` 对 XLSX 更宽松（仅检查列名特征），把格式确认放到 `identify_ev19_template` 的后续步骤

### 证据

截图为证：`/home/wangqiuyang/dogfood-oft-blocked.png`

Agent 日志：
```
identify_ev19_template → "format_unrecognized": "Not an EthoVision export: Raw data-OFT-Xuhui-Trial 1.xlsx"
prep_metric_plan → error (format not recognized)
Guardrail: "code-executor 需要 plan_metrics.json 作为施工单"
```

---

## 3. Sprint 1 Column Semantics: 核心逻辑工作正常，但被 BUG #1 阻塞

### ✅ 确认正常的部分

| 检查项 | 状态 | 证据 |
|--------|------|------|
| `inspect_uploaded_file` 返回 `column_assessment` | ✅ | 正确识别 `center`, `边缘区`, `边缘区到center` 为自定义 zone 列 |
| `column_assessment` 含证据（0/1 binary zone） | ✅ | Agent 报告 "binary zone，取值 0/1" |
| `open_questions` 非空触发 HITL | ✅ | Agent 主动反问 3 个自定义列的含义 |
| 反问合并（一次问完不分开） | ✅ | 格式+分组+列语义三合一 |
| 预填推测+可否决（D13/D15） | ✅ | "是否表示中心区？" 带证据 |
| 列名证据来自取值分布不来自字面 | ✅ | "center（binary zone，取值 0/1）— 是否表示中心区？" |
| 标准固定列不被反问 | ✅ | `trial_time`, `x_center` 等未出现在 open_questions |
| `set_experiment_paradigm` 接受 column_semantics | ✅ | Agent 写了 `open_field · OpenFieldRectangle-AllZones` |
| 用户确认后 agent 正确映射 | ✅ | center→中心区, 边缘区→外周区, 边缘区到center→忽略 |

### ❌ 未验证的部分（被 BUG #1 阻塞）

| 检查项 | 状态 | 原因 |
|--------|------|------|
| `column_aliases` 从 column_semantics 派生 | ❓ | prep_metric_plan 在 format 检测就失败了 |
| `_apply_aliases` 重映射列名 | ❓ | 同上 |
| `resolve_metrics` 用 aliases 算出指标 | ❓ | 同上 |
| OFT center_time_ratio 等 5 metric 正确计算 | ❓ | 同上 |
| guardrail 放行 open_questions=[] 的标准路径 | ❓ | 没有标准命名数据可测 |
| `anonymous_zone_is` 旧机制不受影响 | ❓ | 同上 |
| chart-maker CLI 路径 column-aliases-file | ❓ | 同上 |

### 关键判断

**Sprint 1 的 HITL 交互层（inspect → column_assessment → 反问 → set_experiment_paradigm）是正确的。** 但 resolve/alias 的代码路径从未被执行到，无法验证集成正确性。

---

## 4. 其他观察

### 4.1 Agent 对 `format_unrecognized` 的处理

Agent 在收到 `format_unrecognized` 后：
1. 先尝试 `identify_ev19_template` 两次（相同结果）
2. 正确识别列结构仍然可用
3. 尝试绕过调用 `set_experiment_paradigm` + `prep_metric_plan`
4. prep 失败后给用户 A/B 选项（重新导出 vs 绕过）
5. 用户选 B 后尝试直接派 code-executor → guardrail 拦截
6. code-executor 运行了 Python 检查 xlsx 结构（能看到列数据）

Agent 的行为是合理的，没有陷入死循环（和之前的 seal deadlock 不同）。

### 4.2 XLSX 列名归一化

`normalize_column_name` 对中文列名的处理：
- `中心区` → `center`（zone-prefix detection 导致，handoff §4.1 警告过）
- `边缘区` → `边缘区`（保持不变）
- `边缘区到中心区` → `边缘区到center`

handoff 警告的"中心区归一化成 center"已确认——`_apply_aliases` 的两边都过 normalize 的鲁棒匹配（handoff §4.1）将在这个场景被考验。

### 4.3 全自动模式未影响流程

Agent 在"全自动"模式下仍然正确触发了 HITL（ask_clarification），因为 column_semantics 的问询是必需的 Gate 1 信息收集，不是可跳过的优化。

---

## 5. 推荐行动

### P0（阻塞修复，立即）

修复 BUG #1：`_detect_ethovision_xlsx` 支持无"标题行数"标记的 XLSX 文件。

建议实现：
```python
def _detect_ethovision_xlsx(path, sheet_name=0):
    df = pd.read_excel(path, sheet_name=sheet_name, header=None, nrows=2)
    # 检查 A1 是否有 "标题行数"
    first_cell = str(df.iloc[0, 0])
    if "标题行数" in first_cell:
        return True
    # 备选：检查表头行是否包含 EthoVision 特征列
    # 某些 XLSX 导出没有 header 段，数据直接从第一行开始
    second_row = [str(c).lower() for c in df.iloc[min(1, len(df)-1)] if pd.notna(c)]
    ethovision_markers = {'trial_time', 'recording_time', 'x_center', 'y_center', 
                          'distance_moved', 'velocity', 'in_zone', 'trial time',
                          'recording time', 'x center', 'y center'}
    if ethovision_markers & set(second_row):
        return True
    return False
```

合并前必跑：`cd packages/ethoinsight && pytest`（全量，改的是 parse 承重墙）。

### P1（修复后重测）

修复 BUG #1 后，重新 dogfood OFT-Xuhui-34，验证：
1. `prep_metric_plan` 能否读取 `column_aliases`
2. `_apply_aliases` 能否正确映射 `center` → `in_zone_center`
3. OFT metric 能否算出（center_time_ratio 等）
4. 整条管道跑通到 report-writer

### P2（补齐测试矩阵）

修复后对其他 5 个数据集各跑一次，确认：
- EPM/TST/EZM/FST 的标准路径不受影响
- 有自定义列的也被正确对齐
- `anonymous_zone_is` 兜底路径仍工作

---

## 6. 文件清单

- 截图：`/home/wangqiuyang/dogfood-oft-blocked.png`（agent 被 format_unrecognized 阻塞的完整页面）
- OFT dogfood 对话 ID：`abf26ac3-566c-42d1-a439-a948930f8251`
- 相关代码：
  - `packages/ethoinsight/ethoinsight/parse/_core.py:80-93` — `_detect_ethovision_xlsx`（root cause）
  - `packages/agent/backend/packages/harness/deerflow/tools/builtins/identify_ev19_template_tool.py:430-434` — 调用 detect_ethovision 处

## milestone 建议

无需新建 milestone。这是 BUG #1 的发现 + Sprint 1 的部分验证。修复 BUG #1 后应重新 dogfood 并更新本 handoff。
