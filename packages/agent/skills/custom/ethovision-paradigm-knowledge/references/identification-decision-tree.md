# EV19 模板识别决策流程

## 决策树（agent 严格按此流程执行）

```
用户上传文件 + 提问
    ↓
Step 1: 收集证据（不调任何工具，只看上下文）
    - 用户消息中是否提到学术范式名（"高架十字迷宫" / "EPM" / "焦虑测试" / "斑马鱼" 等）？
    - 上传文件名中是否带范式简写（"轨迹-EPM-Trial 1...txt" 等）？
    - 上传文件数量？Subject 数？
    ↓
Step 2: 推测候选 paradigm_key（snake_case 英文）
    - 命中明确范式 → paradigm_key 确定（如 "epm"）
    - 模糊（"焦虑测试"）→ paradigm_key 多候选（["epm", "open_field", "zero_maze", "light_dark_box"]）
    - 完全没线索 → 进入 Step 6（必反问）
    ↓
Step 3: 读 by-experiment/<paradigm_key>.md（已确定单一 paradigm 时）
    - 看 "适用模板" 字段，得到 ev19_template 候选列表
    - 候选 ≥ 2 → 进入 Step 4
    - 候选 = 1 → 进入 Step 7（直接 set）
    ↓
Step 4: 读用户上传的第一个 raw txt（前 50 行）
    - 用 read_file 看 meta + 列结构
    - 单位（"毫米"/"厘米"）→ 区分鱼/啮齿
    - 追踪点（仅 X 中心 = 单点 / X 鼻点 + X 中心 + X 尾 = 三点）→ 区分鱼/啮齿
    - 列名是否含 "In zone(...)" → 区分 AllZones / NoZones
    - 列名是否含 "Nose within object zone(...)" → NovObj 模板
    - 据此把候选缩到 ≤ 3 个
    ↓
Step 5: 候选数判断
    - 候选 = 1 → 进入 Step 7（直接 set）
    - 候选 2-3 → 进入 Step 6（结构化反问）
    ↓
Step 6: ask_clarification（最多 1 次）
    - 必须给 ≤3 个结构化选项
    - 推荐项放第一位，标 "(推荐)"
    - 每个选项标差异（"含开闭臂 + 头探出区，最常见"）
    - 兜底说明（"如不确定，选 A，绝大多数 EPM 用这个"）
    - 用户回复后 → 进入 Step 7
    - 用户答 "不知道" / "随便" → 查 default-template-fallback.md → 进入 Step 7
    ↓
Step 7: 调 set_experiment_paradigm(paradigm=<key>, ev19_template=<选定>, ...)
    - 工具校验白名单
    - 通过 → 写 experiment-context.json，进入分析
    - 失败（白名单不通过）→ 看错误的 candidates 字段重选 → 重调（这一步算入 LoopDetection 计数，避免反复）
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
