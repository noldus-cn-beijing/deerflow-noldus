# 设计 spec：C3 指标表选区追问（Gemini 式数据追问）（2026-06-30）

> 生成式 UX 路线图 C 系列第三步。让研究员在 C1 的**指标结果表**上框选一块数据，把它凝成一个结构化引用胶囊喂回主输入框，再自拟问题对这块数据追问。
>
> 来源：生成式 UX 路线图（`2026-06-30-generative-ux-roadmap-and-a1-event-track-foundation-design.md:29`）C3 = 「产物框选→喂回输入框追问（Gemini 式数据追问）」。本 spec 只设计 C3，交别的 agent 实施。
>
> **实施前置 = C1 已 merge dev**（接手时校准：C1 当时在 worktree `worktree-c1-metrics-table-export` 实施中、未 commit）。C3 spec 针对 C1 spec 已冻结的 `metrics_table.json` 契约编写，故 spec 现在可写完；实施排在 C1 落地之后。

---

## 目标

研究员在指标结果表上选中一个单元格 / 一整行（subject）/ 一整列（指标）/ 多选矩形区，点「追问此选区」→ 选区凝成一个不可编辑胶囊插进主输入框 → 研究员打字提问 → 发送时胶囊确定性展开成结构化文本前缀拼进消息。**纯前端、零后端/prompt/sync 改动**：agent 收到的就是一条带数据前缀的普通用户消息。

## 五个承重决策（brainstorm 锁定）

1. **可框选产物范围**：v0.1 **仅 C1 指标结果表**（结构化选区，能精确带出 subject/group/指标名+值）。图表 PNG / 报告 HTML 不在 v0.1，留后续。
2. **选区去向**：选区 → 「追问此选区」按钮 → 凝成胶囊喂回**主输入框** → 用户自拟问题 → 发送。**不**在选区处自带迷你输入框，**不**直接触发分析。
3. **选区载荷**：胶囊在**发送时**展开成结构化**文本前缀**拼进用户消息。**纯前端**——选区从不进网络层 / 后端 schema / prompt。
4. **选区粒度**：单元格 + 行（subject）+ 列（指标）+ 多选（shift/拖拽连续矩形区），Excel 式。
5. **胶囊形态**：不可编辑胶囊，可点 × 移除，可多个并存；用户不直接编辑选区内容，只能加/删整个胶囊。复用现有 attachments 胶囊视觉语言。

## 架构方案（方案 A：复用现有 composer 胶囊范式）

C3 的选区引用**不走文件上传通道**，而是新增一个与现有 `followups` 并列的轻量前端状态 `selectionRefs`，渲染成胶囊堆在输入框上方（复用 `StackedAttachments` 视觉），发送时确定性序列化成文本前缀。

**否决的替代方案**：
- **方案 B（选区处自带 popover 输入框）**：违背决策 2（明确要喂回主输入框）；新造一套发送路径，与主 composer 的草稿/附件/加急逻辑割裂。否决。
- **方案 C（全局选区 store 跨组件共享）**：当前 YAGNI 违例（v0.1 只有表一类产物，选区→输入框是单向一跳，无需跨组件共享可变态）。降级 v1.0。

选方案 A 的理由：完全坐在已验证的 composer 胶囊地基（`followups` / `pendingSuggestion` / `StackedAttachments` 都已在 `input-box.tsx`）上，新增面最小，且把唯一承重逻辑「选区→文本前缀」收敛成一个确定性可测纯函数。

## 三个单元（单一职责、接口清晰、可独立测试）

### 单元 1：`MetricsTable` 选区层（在 C1 的指标表渲染组件上）

- **做什么**：把 C1 的 `metrics_table.json` 渲成可选区表格。点单元格选一格；点行头选整行（一个 subject）；点列头选整列（一个指标）；shift/拖拽选连续矩形区。维护本地 `selection` 状态（一组 `{rowKey, colKey}` 坐标）。选中后在表格角落浮出「追问此选区」按钮。
- **接口**：父层传入 `tableData`（来自 C1 JSON）+ 回调 `onAskSelection(selection: SelectionRef)`。点按钮 → 调回调 → 清本地选区高亮。
- **依赖（C1 已落地 #255，下为实际产出形状，2026-06-30 复核更正）**：C1 的 `metrics_table.json` =
  `{paradigm, metric_names: string[], groups: [{group, n, metrics: {<指标>:{mean,std,n}}}], per_subject: [{subject, group, values: {<指标>:值|null}, outlier_flags: {<指标>:bool}}]}`。
  **⚠️ 关键修正**：每行的指标值**嵌在 `values` 子对象**下（`row.values[指标]`），**不是**平铺在行上（早期 spec 误写 `{subject, group, <指标>:值}`）。离群标记同样在 `outlier_flags` 子对象。前端已建的 `metrics-table-card.tsx`（`SubjectRow` 类型 + `r.values?.[m]` 读值，`metrics-table-card.tsx:49-53/204`）就是 C3 选区层要挂载/复用的组件。**这是 C3 对 C1 的唯一硬依赖点，已按 #255 实际产出冻结。**

### 单元 2：选区序列化纯函数 `serializeSelection(ref, tableMeta): string`

- **做什么**：把结构化选区 ref 确定性转成人类可读文本前缀。例：
  `[选中数据 · open_arm_time_ratio] Control 组: subj01=0.32, subj02=0.28; Treatment 组: subj07=0.51 …`
  多指标 / 多组按**稳定顺序**（列名字典序 → 组名 → subject）展开，保证同一选区每次序列化字节一致。
- **接口**：输入 = 选区 ref + 表元数据（列名、组名映射）；输出 = 字符串。**无副作用、无 React、纯数据。**
- **依赖**：无（纯 TS）。
- **这是 C3 唯一承重逻辑，是测试重心。**

### 单元 3：输入框选区胶囊状态（在 `input-box.tsx`）

- **做什么**：新增与现有 `followups` 并列的轻量状态 `selectionRefs: SelectionRef[]`。渲染成胶囊（复用 `StackedAttachments` 视觉，标签如 `📊 Control×open_arm_ratio · 3 值`），可点 × 移除。`onSend` 时把每个 ref 经单元 2 序列化、拼到消息文本最前面，发完清空。
- **接口**：表格的 `onAskSelection` 经页面层路由到这里的 `appendSelectionRef(ref)`。
- **依赖**：单元 2（序列化）；现有 composer 的 `onSend` / 草稿状态。

## 数据流（一条单向链，无回路）

```
用户在 MetricsTable 框选 (单元1)
  → 点「追问此选区」→ onAskSelection(ref)
  → 页面层路由 ref 到 input-box (单元3) appendSelectionRef
  → 胶囊显示在输入框上方（可×移除、可再选再加）
  → 用户打字 + 发送
  → onSend: ref 经 serializeSelection (单元2) 展开成文本前缀
  → 前缀 + 用户文本 拼成最终消息 → 走现有发送通道
  → selectionRefs 清空
```

**关键边界**：选区 ref 从不进网络层、不进后端 schema、不进 prompt——它只在前端存活到 `onSend` 那一刻被展开成普通文本。后端 / agent 收到的就是一条带结构化数据前缀的普通用户消息。

**跨组件接线**：表格（产物面板/画廊侧）与输入框（对话侧）是兄弟组件。`onAskSelection` 由共同父层（`workspace-content.tsx` 或对应 workspace 页面）持有一个轻量 handler 往下传给输入框。这是方案 A 唯一的跨组件接线，且是**单向一次性传递**（不是共享可变态——这正是不选方案 C 的原因）。

## 错误处理 / 边界（只处理真实会发生的）

| 场景 | 处理 |
|---|---|
| C1 数据未就绪 / 表为空 | 无可选区表格 → C3 入口自然不出现（无表无选区无按钮）。不专门报错。 |
| 选区为空就发送 | `selectionRefs` 为空 → `onSend` 序列化跳过、不加前缀，等同普通发送。不报错。 |
| 多 chip + 多指标多组 | `serializeSelection` 按稳定顺序展开，保证字节一致（可快照测）。 |
| 选区含 outlier / NaN 值 | 照 C1 JSON 值原样透传（C1 已做 outlier_flags + 清洗）；C3 不重新判断。某单元格值缺失 → 序列化成 `subjXX=—` 占位符，不崩。 |
| 同格重复选 / 重复加 chip | 加 chip 时按 ref 内容去重（同坐标集只留一个胶囊）。 |
| 切 thread / 表数据变更后旧 chip 还在 | 切 thread / 表数据变更时清空 `selectionRefs`（陈旧引用不跨上下文存活）。 |

## 测试策略（守 CLAUDE.md TDD 强制 + 防 vacuous 铁律）

**核心 = 单元 2 序列化纯函数（C3 唯一承重逻辑，无 React 依赖）**：
1. 单格选区 → 断言精确文本前缀。
2. 整行（subject）选区 → 断言带该 subject 全部指标。
3. 整列（指标）选区 → 断言带该指标全部 subject + 组归属。
4. 多选矩形区 + 多 chip → 断言稳定顺序拼接。
5. **防 vacuous 探针**：删掉序列化里「带组名」那行 → 对应断言必须变红（证测试真在测组名透传，非恒真）。
6. 缺失值 → 断言降级成占位符不抛。

**组件层（vitest + testing-library，项目已配）**：
7. `MetricsTable`：点单元格/行头/列头 → 断言 `selection` 状态正确；点「追问」→ 断言 `onAskSelection` 收到正确 ref。
8. `input-box`：`appendSelectionRef` → 断言胶囊渲染；点 × → 断言移除；`onSend` → 断言最终消息含序列化前缀且发后 `selectionRefs` 清空。

**回归护栏**：C1 的指标表渲染若已有测试，C3 加选区层后断言原渲染不回归（选区是叠加层，不改原显示）。

## sync 友好性（守「全量跟随上游 + surgical 守 Noldus」）

C3 **100% 落在 deerflow 子树外、纯前端**：
- 改/建文件全在 `packages/agent/frontend/src/components/workspace/` 下。
- **零改** `prompt.py` / 消息 schema / 任何 `packages/harness/deerflow/` 子树文件（决策 3 的直接红利）。
- 不碰 `ai-elements/` / `ui/` registry 结构（只在其上组合，不改其 API）。
- `sync-deerflow.sh` 只门控 deerflow 子树 → C3 完全不在门控面内。

## 依赖时序

- **C3 对 C1 的唯一硬依赖** = 单元 1 消费的 `metrics_table.json` 形状。**✅ 已对 C1 实际落地（#255 / `5a45d865`）复核完成**：契约按真实产出冻结（见单元 1 依赖块）。复核发现一处 schema 漂移并已更正——指标值嵌在 `per_subject[].values` 子对象下，非平铺行键（早期 spec 误写）。源：`metrics_table_export.py:159-176`（导出）+ `metrics-table-card.tsx:49-53/204`（前端类型与读值）。
- **C3 实施前置 = C1 已 merge dev**：✅ 已满足（#255 在 dev）。实施时单元 1/2 按 `row.values[指标]` 读值、按 `row.outlier_flags[指标]` 读离群标记，复用/挂载到已存在的 `metrics-table-card.tsx`（不要从零另建表组件，除非 C2 重排了画廊布局）。
- 路线图依赖（`roadmap:29`）C3 = A2 + C2：A2（前端分轨渲染）+ C2（画廊布局）就绪后，指标表在画廊中的呈现位置才稳定。实施时按 C1/C2 实际落地的表组件名对齐选区层挂载点。

## 文件影响清单（实施 plan 精确化）

**新建**：
- 在 C1 已落地的 `packages/agent/frontend/src/components/workspace/artifacts/metrics-table-card.tsx` **上加选区层**（不另建表组件）；若选区逻辑较重可抽到同目录 `metrics-table-selection.ts`，但渲染挂载点是这个已存在的卡。
- 选区序列化纯函数 `.../selection/serialize-selection.ts` + 类型 `SelectionRef`
- 3 个测试文件（序列化纯函数 + `MetricsTable` + `input-box` 选区路径）

**修改**：
- `input-box.tsx`：加 `selectionRefs` 状态 + 胶囊渲染 + `onSend` 序列化拼接 + 切 thread 清空
- workspace 共同父层（`workspace-content.tsx` 或对应页面）：持有 `onAskSelection` handler 向下接线
- 可能扩 `StackedAttachments` 或并列加一个选区胶囊渲染（实施时定，优先复用）

**零改**：后端、prompt、消息 schema、deerflow 子树、ai-elements / ui registry。

## 明确不做（YAGNI 边界）

- ❌ 图表 PNG / 报告 HTML 框选（决策 1：v0.1 仅表）→ C2 / 后续单独立项。
- ❌ 选区结构化传后端 + prompt 消费（决策 3：纯前端文本前缀）→ 若未来要 agent 精确回指某选区再说。
- ❌ 全局选区 store / 跨组件共享（方案 C）→ v1.0。
- ❌ 选区直接触发分析、不经输入框（违决策 2）。
- ❌ 选区处自带迷你输入框（方案 B）。

## 验收标准

1. 指标表上单元格/行/列/多选可选区，选中浮出「追问此选区」。
2. 点按钮 → 选区胶囊出现在主输入框上方，可 × 移除、可多个并存、内容去重。
3. 打字 + 发送 → 最终消息含确定性结构化数据前缀（subject/group/指标/值），发后胶囊清空。
4. 切 thread / 表数据变更 → 旧胶囊清空。
5. 序列化纯函数全测过，含防 vacuous 探针（删字段断言变红）。
6. C1 指标表原渲染不回归。
7. 后端 / prompt / deerflow 子树零改动（sync 友好）。
