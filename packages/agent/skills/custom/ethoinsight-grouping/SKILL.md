---
name: ethoinsight-grouping
description: >
  EthoInsight 分组（control vs treatment）链路单一权威。
  Lead 收集用户分组答案 → prep_metric_plan(groups=...) → workspace/groups.json
  → plan_metrics.json.inputs.groups_file → code-executor 读 groups 聚合统计。
  本 skill 解决两件事:(1) lead 怎么把"第一个是实验组"翻译成 dict;
  (2) code-executor 拿不到 groups 时禁止幻觉脚本去探测 drug 列。
version: 1.0.0
author: noldus-insight
---

# 分组链路单一权威

## 何时使用本 skill

- **Lead**: 收到上传数据 / 准备调 `prep_metric_plan` 时 → 必读 [§Lead 落地](references/lead-translates-answer.md)
- **Code-executor**: 跑指标计算前读 `plan.inputs.groups_file` 时 → 必读 [§Code-executor 行为](references/code-executor-handles-groups.md)

## 整链路(2026-05-28 更新:加入 inspect_uploaded_file)

```
[用户] 上传文件 + "第一组是实验组" 之类的自然语言
   │
   ▼
[Lead] 调 inspect_uploaded_file 读 EV19 元数据
   │  → 多数情况直接拿到 Treatment 字段 (无需反问用户)
   ▼
[Lead] 翻译成 {file_path: group_name} dict
   │  (优先用 EV19 Treatment 字段;无则反问用户对应关系)
   ▼
[prep_metric_plan(groups=...)] 写 workspace/groups.json
   │                           + plan_metrics.json.inputs.groups_file
   ▼
[code-executor] read plan → read groups.json → 按 group_name 聚合 metrics_summary
   │
   ▼
[seal_code_executor_handoff] handoff 落盘
```

## 关键设计:EV19 头部内置 Treatment 字段

Noldus EthoVision XT 导出的 txt / xlsx 在头部 metadata 区域**直接写明 Treatment / Dose / Animal ID**。例如:

```
"Treatment";"Drug"
"Dose";"5 mg/L"
"Animal ID";"1"
```

这意味着**多数 dogfood 场景 lead 根本不需要反问用户分组** — `inspect_uploaded_file` 一调,Treatment 字段就回来了。

例外:用户用了非 Noldus 标准导出 / 头部缺 Treatment 字段 → 降级到反问用户对应关系。详见 [lead-translates-answer.md §场景 B](references/lead-translates-answer.md)。

## 单一权威原则

`plan.inputs.groups_file` 是分组信息的唯一权威:

| 阶段 | 角色 | 做什么 / 不做什么 |
|---|---|---|
| 探查 | Lead | **必须**调 inspect_uploaded_file 看每个文件的 ev19_metadata;**禁止**派 general-purpose 写 Python 探查 |
| Gate 1 反问 | Lead | EV19 Treatment 缺失时才反问;有 Treatment 时可省略或仅做对应确认 |
| 翻译落地 | Lead | **必须**把分组结论翻译成 `{path: group_name}` dict 传 prep_metric_plan(groups=...) |
| 落盘 | prep_metric_plan tool | 写 workspace/groups.json + plan_metrics.json.inputs.groups_file 字段 |
| 消费 | Code-executor | **必须**从 plan.inputs.groups_file 读;**禁止**自己从原始数据探测 drug / treatment 列 |
| 聚合 | Code-executor → seal | 按 group_name 算 mean/std/n;groups_file 为 null 时按"无分组"处理 |

## 反模式(实证踩过的坑)

### ❌ Lead 派 general-purpose subagent 写 Python pandas 探查

实证 2026-05-28 thread 复现:lead 试图 `task(subagent_type='general-purpose', prompt='用 pandas 读 xlsx 列出 drug 唯一值')` → **subagent 类型不存在** → 报错。Noldus fork 只有 5 个 subagent: chart-maker / code-executor / data-analyst / report-writer / knowledge-assistant。

**正确**:调 `inspect_uploaded_file` 工具。

### ❌ Code-executor 幻觉脚本去读 drug 列

实证 2026-05-28 thread 485a899d:lead 派 code-executor 时 plan.inputs.groups_file 为 null,code-executor LLM 尝试调 `inspect_drug_column` / `dump_subject_metadata`(都不存在) / `python -c "..."`(被 sandbox 拦)→ 用户卡 6+ 分钟。

**根因**:lead 没把分组翻译成 groups dict。

### ❌ Lead 死循环反问"drug 列具体值是什么"

实证 2026-05-28 thread:lead 知道要拿 groups dict 但没工具看 xlsx,反复问用户 drug 列值,**用户也不知道**(普通用户不会 pandas)。

**根因**:没有 inspect_uploaded_file 工具,本次修复正在补这个缺口。

### ❌ 派遣 prompt 用自然语言告知分组

不要这样:

```
派遣 code-executor: "分析数据, drug 列第一个值是 treatment, 第二个是 control"
```

LLM 不知道"drug 列第一个值"具体是什么字符串。**正确做法**:lead 在调 prep_metric_plan 之前先 inspect_uploaded_file,然后把信息固化进 groups dict。

### ❌ 按 uploaded_files 顺序硬编码

不要这样:

```python
groups = {uploaded_files[0]: "treatment", uploaded_files[1]: "control"}
```

uploaded_files 顺序由 frontend 上传决定。**正确做法**:用 inspect 拿到的 EV19 Treatment 字段,或反问用户确认对应关系。

## 参考文件

- [Lead 落地分组答案 — 三种典型场景](references/lead-translates-answer.md)
- [Code-executor 处理 groups](references/code-executor-handles-groups.md)
- [不存在脚本的应对](references/missing-script-recovery.md)
