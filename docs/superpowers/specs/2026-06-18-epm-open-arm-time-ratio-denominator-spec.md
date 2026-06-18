# ~~Spec：EPM `open_arm_time_ratio` 分母改为「开臂+闭臂」（排除中央区）~~

> ## ⛔ 此 Spec 前提已撤回，请勿执行
>
> **2026-06-18 专家最终裁定（实施过程中多轮更正后落点）**：
> `open_arm_time_ratio = 开臂帧 / 总帧（含中央区）`——即**当前实现就是正确的**，分母无需改动。
>
> 本 spec §1.2 的目标公式 `open/(open+closed)` 已被专家明确否决，理由：
> - OFT 只有 center 列（无独立外围区列），与 EPM 数据拓扑不同，若「排除中央区」则 OFT 无法对应实现；
> - 专家最终在「分母=总帧含中央区 vs 臂内分母」的明确二选一中选定前者。
>
> **代码（`epm.py:compute_open_arm_time_ratio` 的 `combined.mean()`）、catalog one_liner、skill 文档均无需修改。**
>
> 以下原始 spec 内容仅作历史存档。

---

# Spec：EPM `open_arm_time_ratio` 分母改为「开臂+闭臂」（排除中央区）

## 〇、给实施 agent 的一句话

行为学专家（用户）裁定：EPM 的 `open_arm_time_ratio` 应为 **开臂时间 / (开臂时间 + 闭臂时间)**，**排除中央区**——而非现实现的 **开臂时间 / 总记录时间**（分母含中央区）。这是 **SSOT 级语义裁决**：专家明确说"SSOT（catalog/skill 现有口径）写错了，以专家口径为准"。本 spec 把这一改动的代码点、SSOT 联动点、历史结果影响、跨范式一致性风险一次说清，供实施 agent 直接执行。

> ⚠️ 关键边界：本改动**只动 `open_arm_time_ratio` 一个指标的分母**。`open_arm_time`（绝对秒数，无分母）、`open_arm_entry_ratio`（次数比，分母是入臂次数非时间）、`open_arm_entry_count`、`total_entry_count` **全部不动**。

---

## 一、现状（逐字节实证，HEAD `4300f510`）

### 1.1 当前公式 = 开臂帧 / 总帧（含中央区）

`packages/ethoinsight/ethoinsight/metrics/epm.py:83`：

```python
def compute_open_arm_time_ratio(df, open_arm_zones=None) -> float | None:
    cols = [c for c in open_arm_zones if c in df.columns] if open_arm_zones else _get_open_zone_cols(df)
    if not cols:
        return None
    combined = df[cols].max(axis=1).dropna()   # 每帧：在任一开臂列=1 则 1
    if combined.empty:
        return None
    return float(combined.mean())              # = 开臂帧数 / 总帧数（分母含中央区帧）
```

`combined.mean()` 的分母是 `len(combined)` = **整个 trial 的所有帧**（开臂 + 闭臂 + 中央区）。这是当前实现的事实。

### 1.2 专家裁定的目标公式

```
open_arm_time_ratio = 开臂帧数 / (开臂帧数 + 闭臂帧数)        # 排除中央区帧
```

行为学依据（用户原话转述）：这一次实验的「全部 open / (全部 open + 全部 close)」——分母只算动物在臂内（开或闭）的时间，中央区是过渡区不计入。

### 1.3 现有可复用件（降低实施风险）

- `_get_closed_zone_cols(df, closed_arm_zones=None)` **已存在**（epm.py:54）——闭臂列查找现成。
- `compute_open_arm_time_ratio` 经 `compute_open_arm_time_ratio(df, **parameters)` 调用（`scripts/epm/compute_open_arm_time_ratio.py:35`），`parameters = parse_parameters(args)`。**只要函数签名加 `closed_arm_zones` 参数 + catalog 声明它，HITL `column_aliases` 注入链路自动流通**——无需改 compute 脚本本身。
- catalog 里 `open_arm_entry_ratio` / `total_entry_count` **已经**声明了 `closed_arm_zones` 参数（`catalog/epm.yaml:78/103`），注入链路成熟，本指标照搬即可。

---

## 二、改动清单（4 处，按依赖序）

### 2.1 代码：`compute_open_arm_time_ratio` 加 `closed_arm_zones` 参数 + 改分母

`packages/ethoinsight/ethoinsight/metrics/epm.py`：

```python
def compute_open_arm_time_ratio(
    df: pd.DataFrame,
    open_arm_zones: list[str] | None = None,
    closed_arm_zones: list[str] | None = None,
) -> float | None:
    """Ratio of open-arm time over arm time (open + closed), EXCLUDING the centre.

    分母 = 开臂帧 + 闭臂帧（不含中央区），与行为学口径「全部 open / (全部 open + 全部
    close)」一致（2026-06-18 专家裁定）。中央区是过渡区，不计入分母。
    """
    open_cols = ([c for c in open_arm_zones if c in df.columns]
                 if open_arm_zones else _get_open_zone_cols(df))
    closed_cols = _get_closed_zone_cols(df, closed_arm_zones)
    if not open_cols:
        return None

    open_in = df[open_cols].max(axis=1)
    closed_in = df[closed_cols].max(axis=1) if closed_cols else 0
    # 逐帧：开臂帧数、臂内帧数（开∪闭）
    open_frames = float(open_in.fillna(0).sum())
    if closed_cols:
        arm_frames = float(((open_in.fillna(0).astype(bool)) | (closed_in.fillna(0).astype(bool))).sum())
    else:
        arm_frames = open_frames  # 无闭臂列时退化（见 §3.2 降级）
    if arm_frames == 0:
        return None
    return open_frames / arm_frames
```

> 实施注意：上面是**示意**，实施 agent 要按 epm.py 现有代码风格写（`df[cols].max(axis=1)` 的 dropna/fillna 处理要与同文件其他函数一致；逐帧并集用 bool OR）。**关键不变量**：分母 = 在开臂或闭臂的帧数，分子 = 在开臂的帧数（开/闭重叠帧理论上不存在，EthoVision 互斥分区）。

### 2.2 catalog：声明 `closed_arm_zones` 参数 + 更新 one_liner

`packages/ethoinsight/ethoinsight/catalog/epm.yaml` 的 `open_arm_time_ratio` 条目：

- `requires_columns`：加 `in_zone_closed_arms_*`（现在只有 `in_zone_open_arms_*`）。
- `parameters`：增加 `closed_arm_zones`（照抄 `open_arm_entry_ratio` 的 `closed_arm_zones` 声明，`tunable_by_user: false`，HITL 注入）。
- `one_liner`：从「占总时长的比例」改为「占臂内（开+闭）时长的比例，排除中央区」。

### 2.3 skill 口径文档：`by-experiment/epm.md`

`packages/agent/skills/custom/ethovision-paradigm-knowledge/references/by-experiment/epm.md:47`：
「开臂停留时间百分比」的口径注释要点明**分母是臂内时间（开+闭），不含中央区**。该文件第 43 行写「指标公式以 catalog 为准（SSOT）」——所以**主改 catalog 的 one_liner**，epm.md 若有具体口径描述同步，无则不强加。

### 2.4 output-constitution / 判读口径

`skills/custom/ethoinsight/references/output-constitution.md`：grep 是否有「开臂时间比例 = …/总时间」之类的口径描述；有则改，无则不动。判读方向（越低越焦虑）不变。

---

## 三、影响与风险（实施前必读）

### 3.1 ⚠️ 所有历史结果会变化

改分母后，**每个 subject 的 `open_arm_time_ratio` 数值都会变大**（分母从「总帧」缩小到「臂内帧」，中央区时间被剔除）。中央区停留越多的个体，变化越大。具体到本轮 dogfood 数据：
- Trial 3/12（total_entry_count=1、几乎全程停一处）这类极端个体的 ratio 变化方向取决于它停在开臂还是中央。
- 组间显著性（p 值）、效应量、报告结论**全部会重算**。已产出的 golden case / 回归基线若锚定旧公式的数值，**需要重新生成**。

→ 实施时必须：① 跑 ethoinsight 全量，预期 `test_*open_arm_time_ratio*` 的数值断言会红，**逐个核对新值是否符合新公式**后更新断言（不是盲改）；② 若有 golden case 锚定该指标，标记待行为学同事重新标注。

### 3.2 降级：无闭臂列时怎么办

EthoVision 数据若**只标了开臂、没标闭臂列**（本轮 dogfood 的 `open`/`closed` 两列都有，但不能假设永远有），分母 (open+close) 无法算。降级策略二选一，**需在 spec 落地时确认**：
- (A) 回退到旧口径（open/总时间）并在结果里标注「无闭臂列，分母用总时间」——保证有值、可追溯。
- (B) 返回 None + data_quality_warning「open_arm_time_ratio 需闭臂列，缺失」——响亮失败、不静默换口径。

倾向 (B)（不静默换语义），但 (A) 对"只有开臂数据"的老数据更友好。**这是实施前要和用户确认的降级点。**

### 3.3 ⚠️ 跨范式一致性风险（必须向用户点明）

**OFT 的 `center_time_ratio` 当前也是 center/总时间**（`metrics/oft.py:71` `series.mean()`，分母含全部帧）。改 EPM 的 `open_arm_time_ratio` 为「臂内分母」后，**EPM 的 ratio 口径与 OFT 的 ratio 口径不再一致**（EPM 排除中央区/非臂区，OFT 不排除）。

按用户的 SSOT 纪律（同一类知识不双存、口径要统一），这里有两种可能，**实施 agent 不得自行决定，须问用户**：
- 仅 EPM 改，OFT 保持现状（接受两范式 ratio 口径不同——EPM 有明确的「臂 vs 中央」结构，OFT 的中心/边缘是连续空间，本就不同质，分母不同可接受）。
- OFT 也要同步改（若专家认为所有 ratio 都该排除过渡区）。

本 spec **只规划 EPM**；OFT 是否联动留作 §5 待确认项。

---

## 四、验收

1. `compute_open_arm_time_ratio(df, open_arm_zones=[...], closed_arm_zones=[...])` 返回 开臂帧/(开臂帧+闭臂帧)，手算一个小 fixture 核对。
2. 新增/更新单测：`test_open_arm_time_ratio_excludes_center`（构造已知开/闭/中央帧数的 df，断言分母不含中央）。
3. catalog `closed_arm_zones` 注入链路通：跑一遍 HITL column_aliases → plan_metrics 带 closed_arm_zones → compute 脚本收到。
4. ethoinsight 全量绿（旧数值断言已按新公式逐个核对更新，非盲改）。
5. 降级路径（§3.2）按用户确认的 A/B 实现并各有单测。

---

## 五、待用户确认（实施前）

1. **§3.2 降级**：无闭臂列时走 (A) 回退旧口径+标注，还是 (B) 返回 None+warning？
2. **§3.3 跨范式**：OFT 的 `center_time_ratio` 是否同步改「排除过渡区」，还是仅 EPM 改？
3. **历史结果/golden case**：是否有锚定旧 `open_arm_time_ratio` 数值的 golden case 需重标？

---

## 六、本 spec 不做的事

- 不碰 `open_arm_time`（绝对秒，无分母）、`open_arm_entry_ratio`（次数比）、其他范式（除非 §5.2 用户要 OFT 联动）。
- 不碰判读方向（越低越焦虑不变）、不碰 reward YAML。
- 不在 epm.md 重复公式（公式 SSOT 在 catalog，epm.md 只描述口径语义）。
