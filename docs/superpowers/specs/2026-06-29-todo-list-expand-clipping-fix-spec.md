# 实施 spec：To-dos 面板展开后内容被压扁/裁切（2026-06-29）

> 实施级文档。修 PR#201 引入的 To-dos 展开面板高度回归。单文件、小改。

---

## 〇、现象与根因（已取证，确定性 bug）

**现象**（用户截图）：对话页 "To-dos" 折叠面板，标题行 "To-dos" 正常，但展开后下面的 todo 条目（如 "data-analyst …统计…"）被压扁/裁切、像被一条横线压着只露半行，看不全。

**根因**：[todo-list.tsx:68-73](../../../packages/agent/frontend/src/components/workspace/todo-list.tsx) 的 `<main>`：

```tsx
<main
  className={cn(
    "bg-accent flex grow px-2 transition-[height] duration-slow ease-brand-in-out",
    collapsed ? "h-0 pb-3" : "h-28 pb-4",   // ← 展开态固定 h-28 = 112px
  )}
>
  <QueueList className="bg-background mt-0 w-full rounded-t-xl">  {/* 内部 max-h-40 = 160px */}
```

三个因素叠加：
1. 外层容器 [todo-list.tsx:42](../../../packages/agent/frontend/src/components/workspace/todo-list.tsx) 有 `overflow-hidden`。
2. `<main>` 展开态固定 `h-28`（112px），但内部 `QueueList` 的 `ScrollArea` 是 `max-h-40`（160px）。
3. 当 todo 条目实际高度 > 112px 时，超出部分被 `overflow-hidden` 裁掉 → 「被横线压着只露半行」。

**引入来源**：PR#201（动效 token 化，commit `4d847228`，2026-06-25）把这里从 `transition-all duration-300`（高度可随内容自适应）改成 `transition-[height] duration-slow` + 固定 `h-28`。固定高度 + 只过渡 height 是回归点。

**分类**：CSS 高度塌陷 / 固定高度裁切。读代码即可定论，不依赖现场。

---

## 一、目标与验收

### 目标
To-dos 展开后完整显示 todo 条目（不被裁），保留折叠动画顺滑，保留长列表滚动（`max-h-40` 滚动语义合理，多条 todo 时滚动而非无限撑高）。

### 验收标准
1. 展开 To-dos：todo 条目完整可读、不被横线/边界压住、不与标题重叠。
2. 条目数超过可视高度时，内部 `ScrollArea` 滚动（不撑爆整个面板）。
3. 折叠/展开动画仍顺滑（不要求保留 height 过渡，但展开不能瞬跳得突兀）。
4. `pnpm check` 通过。
5. Playwright 截图核：展开态截图，条目完整；折叠态截图，面板收起。

---

## 二、改动（单文件）

[todo-list.tsx](../../../packages/agent/frontend/src/components/workspace/todo-list.tsx) `<main>`（68-73），去掉固定 `h-28`，让高度随内容自适应、由内部 `ScrollArea` 的 `max-h-40` 兜住上限。

**推荐改法（grid 行模板过渡，动画+自适应兼得）**：

```tsx
<main
  className={cn(
    "bg-accent grid px-2 transition-[grid-template-rows] duration-slow ease-brand-in-out",
    collapsed ? "grid-rows-[0fr] pb-0" : "grid-rows-[1fr] pb-4",
  )}
>
  <div className="overflow-hidden">
    <QueueList className="bg-background mt-0 w-full rounded-t-xl">
      {/* …既有 todos.map 原样… */}
    </QueueList>
  </div>
</main>
```

要点：
- `grid-rows-[0fr]→[1fr]` 是「高度自适应 + 可过渡」的标准做法：展开态高度=内容自然高（由内部 `max-h-40` 封顶并滚动），折叠态塌成 0，且 `transition-[grid-template-rows]` 让它平滑。
- 内层包一个 `overflow-hidden` 的 div，让折叠中间态裁切发生在「内容包裹层」而非定高的 main——内容完整时不会被裁。
- 去掉 `flex grow` 改 `grid`（main 不再需要 flex 撑子项；若 `grow` 是为在父 flex 里占位，保留 `grow`：`className="... grid grow ..."`，grid 与 grow 不冲突）。

**备选改法（最小改动，但动画退化）**：直接把 `h-28` 改成 `h-auto`，并把 `transition-[height]` 改 `transition-[opacity]` 或去过渡：

```tsx
collapsed ? "h-0 overflow-hidden pb-3" : "h-auto pb-4",
```

- 缺点：`h-auto` 无法 height 过渡（CSS 不能从 0 过渡到 auto），展开会瞬跳。若你接受瞬跳、要最小改动，用这个。**推荐用 grid 法**保动画。

> 实施 agent 二选一，**默认用推荐的 grid 法**（保动画 + 自适应，符合本项目「动效曲线讲究」的前端基调，见 memory `feedback_frontend_design_japanese_minimal_motion_craft`）。

---

## 三、不做什么

- ❌ 不动外层容器（42 行）的 `overflow-hidden`——它管的是 `rounded-t-xl` 圆角裁切，去掉会漏角。
- ❌ 不动 `header`（47-67）。
- ❌ 不改 `QueueList` / `QueueItem` / `max-h-40`（滚动上限语义合理，保留）。
- ❌ 不动 `line-clamp-1`（条目单行截断是设计意图，不是本 bug；本 bug 是高度裁切不是文本截断）。

---

## 四、风险 / 注意

1. **别误判成文本截断**：现象是「高度被裁」（h-28 固定），不是 `line-clamp-1`（那是单行省略，设计意图）。改 `line-clamp` 是错方向。
2. **grid 法的 flex 依赖**：若 `<main>` 的 `flex grow` 是父布局必需（grow 在父 flex 容器里占剩余空间），grid 改造保留 `grow`。改后目测面板在对话流里位置正常。
3. **动画曲线 token**：`duration-slow` / `ease-brand-in-out` 是项目动效 token，保留沿用（别换成裸 ms）。

---

## 五、实施顺序

1. 改 `<main>`（grid 法）。
2. `pnpm check`。
3. dev 起 → 进有 todos 的 thread（或 mock todos）→ Playwright 截图核展开/折叠两态。
4. 精确路径 commit。
