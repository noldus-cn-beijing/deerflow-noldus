# Spec：画廊单样本图折叠提示增强（让用户明白可展开）

> 状态：**实施 spec，可直接交付 agent 执行**
> 日期：2026-06-26
> 代码基线：dev HEAD `6ec47f78`
> 性质：🟢 低 · 前端交互可发现性（折叠是对的，只是提示不够明显），独立可做
> 取证：E2E（thread `bd7ca7f7`）—— E2E agent 看到"单样本图 (112)"分组**没点开就以为没渲染**，说明折叠入口不够明显

---

## 〇、问题（E2E 暴露）

画廊页 per_subject 分组（112 张个体图）**默认折叠、点击标题展开才渲染**——这是**正确的交互设计**（防图墙、一屏一主角、aggregate 代表图直接显示）。`artifact-gallery.tsx:31` `perSubjectExpanded = useState(false)` + `:133` `{perSubjectExpanded && <GalleryGrid .../>}` 坐实。

**但 E2E agent 滚动 8 次都没发现要点击展开**，误判成"112 张不渲染"。说明当前折叠入口（一个小 chevron + "单样本图 (112)"文字）**可发现性不足**——真实研究员也可能不知道这 112 张图要点开看。

**目标（用户拍板）**：**保持默认折叠**（不改成展开，防图墙），但**加明显的展开提示**，让用户一眼明白可点击查看全部。

---

## 一、现状（读码坐实）

`artifact-gallery.tsx:117-133`：
```jsx
<button className="mb-2 flex items-center gap-1 text-sm font-medium" onClick={...} aria-expanded={perSubjectExpanded}>
  {perSubjectExpanded ? <ChevronDownIcon/> : <ChevronRightIcon/>}
  {g.perSubject} ({perSubjectMetas.length})   {/* 如"单样本图 (112)" */}
</button>
{perSubjectExpanded && <GalleryGrid items={perSubjectMetas} .../>}
```
问题：折叠态只有 `▶ 单样本图 (112)`——小箭头 + 朴素文字，不像可点击的"展开 112 张图"。

---

## 二、修复：增强折叠入口可发现性（不改默认折叠态）

在**不改 `useState(false)` 默认折叠**的前提下，让折叠入口明显可点：

1. **文案明确动作**：折叠态文字从 `单样本图 (112)` 改成带动作提示，如 `▶ 展开查看 112 张个体图` / `单样本图 (112) · 点击展开`（i18n，`g.perSubject` 加变体或新 key）。展开态可回到 `▼ 单样本图 (112) · 点击收起`。
2. **视觉上像可点击区**：折叠态整行做成更明显的可点 affordance——加 hover 背景、border 或 button 样式（`--shadow-rest`/边框），不只是裸文字 + chevron。当前 `text-sm font-medium` 太轻。
3. **可选：折叠态露缩略图提示**：折叠时在标题右侧放 2-3 张 per_subject 缩略图的小预览叠层（类似"还有 112 张"的视觉暗示），点击展开全部。**这是更强的可发现性**，但改动稍大；第一版做 1+2 即可。
4. **保留**：aggregate 代表图仍直接展示（:86-107，不动）；per_subject 仍默认折叠（防图墙）。

> 设计纪律（守 spec#6 + 母方案）：折叠入口用 spec#1/#6 token（hover 用 `--ease-brand-out` + `--dur-fast`，可点区用 `--shadow-rest`/边框），日式克制不喧宾夺主——是"明确可点"，不是"花哨大按钮"。

---

## 三、验证

1. `pnpm check`。
2. **真机/E2E**：打开有 per_subject 的画廊 → 折叠态一眼能看出"可点击展开 112 张"（文案 + 可点 affordance）→ 点击展开渲染全部 112 张缩略图 → 再点收起。
3. **可发现性测试**：让没看过的人（或 E2E agent 重跑）打开画廊，确认**不滚动也能注意到展开入口**（不再误判"没渲染"）。
4. **回归**：aggregate 代表图仍直接显示；展开后 GalleryGrid 虚拟化正常（112 张不卡）；折叠态默认收起（防图墙）。

---

## 四、关键文件

- `packages/agent/frontend/src/components/workspace/artifacts/gallery/artifact-gallery.tsx:117-135`（per_subject 折叠入口）
- `packages/agent/frontend/src/core/i18n/locales/*.ts`（`gallery.perSubject` 文案，加"点击展开"变体）
- token：spec#1 `--ease-brand-out`/`--dur-fast` + spec#6 `--shadow-rest`（hover/可点 affordance）

---

*依据：E2E（thread `bd7ca7f7`）agent 未发现展开入口误判"112 张不渲染" + 读码坐实 `perSubjectExpanded=useState(false)` 默认折叠（设计正确）。用户拍板：保持折叠 + 加明显提示。纯前端交互可发现性，不改默认折叠态。*
