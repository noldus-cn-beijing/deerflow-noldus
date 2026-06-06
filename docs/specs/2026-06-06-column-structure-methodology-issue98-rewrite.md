# [方法论] 范式确定后，如何系统性地识别、对齐、聚合用户自定义数据列

**类型**：方法论设计 + 专家输入需求
**关联**：Sprint 1（列语义对齐 1:1 名称映射，已完成）+ Sprint 2（结构聚合 N:1，本 issue）
**设计文档**：[列语义对齐 v2 设计](../design/2026-06-05-column-semantics-hitl-design-v2.md) | [Sprint 1 spec](../superpowers/specs/2026-06-05-column-semantics-alignment-sprint1-spec.md)

---

## 一、问题

真实 EV19 raw data 面向用户完全开放：分析区列名 100% 自定义，分区粒度也随用户实验设计变化。当前 agent 面对自定义列只有两个极端：

1. 列名完全匹配 catalog 预期 → 直接算
2. 列名不匹配 → HITL 反问（Sprint 1，1:1 名称映射）

但缺少一个**系统性的数据处理方法论**来覆盖中间地带：用户数据有哪些列、哪些是 EV19 固定列不需要问、哪些是自定义分析区需要对齐、哪些是同逻辑区的子区需要先聚合再算。

这不只是"请专家确认聚合规则"的数据收集——而是需要先建立**agent 面对任意用户数据的通用决策框架**，专家输入只是框架中的一个环节。

---

## 二、三层列处理框架

### Layer 1: EV19 Template 固定列（确定性识别，0 HITL）

每个 EV19 template 导出的数据文件包含一组**跨用户不变的固定列**。这些列名不受用户自定义影响，可以直接 grep 白名单匹配：

| 固定列 | 说明 | 确定性？ |
|--------|------|---------|
| 试用时间 | Trial time | 需专家确认 |
| 录制时间 | Recording time | 需专家确认 |
| X 中心 | X center coordinate | 需专家确认 |
| Y 中心 | Y center coordinate | 需专家确认 |
| 区域 | Area | 需专家确认 |
| 面积变化 | Area change | 需专家确认 |
| 伸长 | Elongation | 需专家确认 |
| 移动距离 | Distance moved | 需专家确认 |
| 速度 | Velocity | 需专家确认 |
| Result 1 | Result 1 block | 部分 template 无此列 |

Layer 1 的实现方案：
```
agent 调 bash head -1 /mnt/user-data/uploads/*.txt | tr ',' '\n'
→ 秒出所有列名（确定性，0 turn LLM 推理，不走 tool 往返）
→ 白名单 grep 匹配固定列 → 剩余列进入 Layer 2
→ 仅当需要看数据值（如推断某列的值分布、确认分组标签）时才调 inspect_uploaded_file
```

**需要专家输入**：
- 确认上述固定列清单是否完整
- 不同 template 的固定列有无差异（某些 template 可能多/少列，如 Result 1 在某些 template 中不存在）
- 是否有跨语言差异（中文/英文版 EV19 导出的列名不同？）

### Layer 2: 自定义分析区列 → 1:1 概念映射（Sprint 1 已完成，但需要专家验证完整性）

Layer 1 匹配后剩余的分析区列，用户命名无法穷举（"中心区"、"Center"、"zone_A"……）。每个范式只有固定的几个**逻辑分析区概念**：

| 范式 | 逻辑分析区概念 | 已确认？ |
|------|--------------|---------|
| EPM | open_arms, closed_arms | 需专家确认 center 是否需要 |
| OFT | center, border | 需专家确认 corner 是否是独立区 |
| LDB | light, dark | 需专家确认 |
| Zero Maze | open, closed | 需专家确认 |
| FST | 无自定义分析区 | 已确认 |
| TST | 无自定义分析区 | 已确认 |

Sprint 1 已经实现了 agent 的反问机制：读取 catalog 合法概念菜单 → 根据列值分布预填最佳猜测 → 合并反问用户确认。这一步不需要额外工程，但需要专家验证**每个范式的逻辑区概念列表是否完整**。

**需要专家输入**：
- 每个范式的逻辑分析区概念列表是否完整？是否有缺失的概念？
- 某些范式是否有"可选分析区"（某些实验有、某些没有）？

### Layer 3: 分析区结构聚合 N:1（Sprint 2 — 当前缺失，本 issue 重点）

用户可能将**一个逻辑分析区拆成多个物理子区**。检测方式：Layer 1+2 匹配后，用户列数 > catalog 声明的逻辑区数。

```
EPM 示例：
  标准 2 区：Open arm (1 列), Close arm (1 列)
  真实 4 区：Open arm1, Open arm2, Close arm1, Close arm2
  聚合：   Open arm = Open arm1 ∪ Open arm2（时间/距离/进入次数均为 sum）
          Close arm = Close arm1 ∪ Close arm2
```

类似场景在 OFT 可能出现（中心区 = 中心子区A + 中心子区B），LDB 可能出现（明箱 = 明箱左 + 明箱右）。

agent 需要的决策流程：
1. **检测**：用户分析区列数 > catalog 逻辑区数 → 可能存在子区拆分
2. **模式匹配**：对用户列名做相似度聚类（"Open arm1" 和 "Open arm2" 共享前缀 → 可能是同一逻辑区的子区）
3. **HITL 确认**："您的数据有 Open arm1, Open arm2, Close arm1, Close arm2 共 4 区。本范式通常按 2 区（开臂/闭臂）计算。请确认：Open arm1 + Open arm2 → 开臂？Close arm1 + Close arm2 → 闭臂？"
4. **聚合执行**：按指标语义选择聚合函数——时间 = sum、距离 = sum、进入次数 = sum、比率 = 聚合后再算（不能直接用子区比率的平均值）
5. **产出**：聚合后的列映射表，code-executor 直接以此计算指标

**需要专家输入（核心）**：对 EPM / OFT / LDB / FST / Zero Maze / TST 六个范式，逐个回答：

1. **该范式的标准逻辑分析区有哪些？** 
2. **真实数据中，一个逻辑区是否常被拆成多个物理子区？**
   - 如果是，常见的拆分模式和**列命名惯例**是什么？（编号后缀 Open arm1/2？字母后缀？地理位置描述？）
   - 每个逻辑区常见的子区数是几个？
3. **正确的聚合语义是什么？**
   - 简单 sum（时间、距离、进入次数各子区相加）？
   - 需要加权（如按子区面积加权）？
   - 是否存在"绝对不能简单合并"的情况（如某些行为学解读依赖于子区间的差异）？
4. **有没有 DEMO 数据 vs 真实数据的对照样例**可以提供给工程侧作为测试 fixture？
5. **除了子区拆分，有没有其他结构差异模式？**（如用户把多个逻辑区合并为一个、用户定义了与范式无关的分析区需要忽略等）

---

## 三、完整数据处理流水线（方法论）

```
用户上传数据
    ↓
Step 0: identify_ev19_template（tool）
  → 确定 ev19_template 和 paradigm_key
  → 注意：此 tool 内部会解析列结构用于模板匹配，但它的职责是"识别模板"不是"列出所有列名"
    ↓
Step 1: bash head -1 /mnt/user-data/uploads/*.txt | tr ',' '\n'（grep 提速）
  → 秒出所有列名（确定性，0 turn LLM 推理）
  → 比 tool 往返更快、更可靠，且不消耗 LLM 推理预算
  → 仅在需要看数据值推断分组时，才额外调 inspect_uploaded_file（看数据预览行）
    ↓
Step 2: Layer 1 — grep 白名单匹配固定列
  → 匹配的：直接标记为 known_fixed_columns，不需再问
  → 不匹配的：进入 Step 3
    ↓
Step 3: Layer 2 — 识别分析区列 + Layer 3 — 检测子区拆分
  → 读 catalog：该范式有几个逻辑分析区概念？
  → 子区检测：用户分析区列数 > catalog 逻辑区数？
    - 否 → 仅需 Layer 2 名称映射（Sprint 1 流程）
    - 是 → 需要 Layer 3 结构聚合
  → 预填推测（基于列名模式 + catalog 合法菜单，不来自字面猜测）
  → 如需数据值辅助推断（如某列的值分布像中心区），调 inspect_uploaded_file 获取数据预览
  → HITL 合并反问（所有问题一次问清）
    ↓
Step 4: 用户确认后，执行聚合（如需要）
  → 产出：column_semantics + aggregation_rules → experiment-context.json
    ↓
Step 5: code-executor 按常规流程计算指标
  → 聚合后的数据已匹配 catalog 期望，直接可算
```

---

## 四、期望交付

**从专家侧**：每个范式一份结构声明（可放在 `review-packages/` 下，或直接回复本 issue），包含：
- 逻辑分析区概念列表
- 常见的子区拆分模式 + 列命名惯例
- 聚合语义（sum / weighted / 不可聚合）
- 样例数据对照（如果有）

**从工程侧**：
1. 新增 `bash head -1 | tr ',' '\n'` 列名提取 step — 替代当前完全依赖 tool（identify_ev19_template / inspect_uploaded_file）返回列结构的做法。两个 tool 各司其职：identify 管模板识别、inspect 管数据预览；列名提取是更基础的机械操作，bash 更快更确定
2. 新增 `references/column-structure-methodology.md`（agent 在执行前 read 的方法论参考文档，教 agent 三层框架 + grep 提速 + inspect 的使用时机）
3. 按范式增量落地 Layer 3 聚合逻辑（根据专家确认的子区模式 + 聚合语义）
4. catalog 扩展 `logical_zones` 声明（可选——如果子区模式足够规律，可以声明在 catalog 中做确定性匹配；如果不规律，走 HITL）

**工程侧不会在专家确认聚合语义前自行实现任何聚合逻辑**（避免猜错算错）。
