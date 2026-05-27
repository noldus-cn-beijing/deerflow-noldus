# EV19 模板识别决策流程

> **lead agent 不需要自己执行此流程中的文件读取步骤。**
> 改为调 `identify_ev19_template` 工具（1 次 tool call），工具内部完成
> Step 1-5 的所有证据收集和候选过滤。此文档描述工具的内部逻辑，供维护参考。

## 简化流程（lead agent 执行）

```
用户上传文件 + 提问
    ↓
调 identify_ev19_template(uploaded_files, user_message)
    ↓
    ├─ status="ok" (候选=1) → 直接调 set_experiment_paradigm(...)
    ├─ status="ambiguous" (候选2-3) → 用返回的 clarification_question 调 ask_clarification
    │      ↓ 用户回复后 → 调 set_experiment_paradigm(...)
    └─ status="unknown" → 用返回的 clarification_question 调 ask_clarification
    ↓
set_experiment_paradigm 成功后 → 调 prep_metric_plan → 派遣 subagent
```

## 工具内部逻辑（供参考，lead 不要手动执行）

```
Step 1: 从 user_message + 文件名提取范式 hint
Step 2: parse_header → 检测列结构 (AllZones/NoZones/NovObjZones)
Step 3: 查 EV19_TEMPLATE_PARADIGM_MAP → 候选模板列表
Step 4: 按 zone 证据 + subject hint 交叉排除
Step 5: 读 by-experiment/<paradigm>.md + by-template/<Category>.md
Step 6: 候选=1 → ok; 候选2-3 → ambiguous + 反问话术; 候选=0 → unknown
```

## 反问质量准则

### 反例（不精准）

> "请问您用的是 EthoVision 哪个模板？"

用户懵：他根本不熟 EthoVision 模板表。

### 正例（精准）

> 我从您的数据看到：
> - 实验 = 高架十字迷宫 (EPM) — 文件名含 "EPM"
> - 数据有 zone 列（开臂/闭臂标记）
>
> 您用的是哪个 EV19 模板？
> A. **PlusMaze-AllZones**（推荐，含开闭臂 + 头探出区，90%+ EPM 实验用这个）
> B. **PlusMaze-FewZones**（只有开闭臂，无头探出区）
> C. **PlusMaze-NoZones**（仅坐标无 zone — 但您的数据有 zone 列，可能不是这个）
>
> 如果不确定，选 A。

要点：
1. **先告知 agent 已经收集到的证据**（让用户知道你不是瞎问）
2. **结构化选项 ≤3 个**
3. **推荐项放第一位 + 解释为什么推荐**
4. **每个选项标差异**
5. **兜底说明**（让用户能在 5 秒内回答）

## 反问后的处理

| 用户回复 | agent 行为 |
|---|---|
| 选了某个选项（A/B/C）| 直接 set_experiment_paradigm |
| "我不知道" / "随便" / "你定" | 查 default-template-fallback.md，告知用户默认值后 set |
| 提供了表里没有的模板名 | 用 suggest_nearby_templates 函数（在工具返回的 candidates 字段）反问澄清 |
| 完全跑题（用户开始说别的事） | 把模板识别挂起，回应用户当前问题；下次需要 set 时回到此流程 |

## 不要做的事

- ❌ 一上来不读 raw txt 直接反问（候选不缩小，烂问）
- ❌ 反问 ≥2 次模板相关问题（LoopDetectionMiddleware 会拦截）
- ❌ 给开放性问题（"您的实验是什么类型？"）
- ❌ 自己拼 ev19_template 字符串（必须从白名单选）
- ❌ 在用户没明确同意时使用默认值（默认值要在用户面前说出来，不要默不作声）
