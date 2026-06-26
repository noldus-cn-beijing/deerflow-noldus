# Spec：消息流附件复用 #8 堆叠 —— 修发送后附件平铺淹没

> 状态：**实施 spec，可直接交付 agent 执行**
> 日期：2026-06-26
> 代码基线：dev HEAD `5616a73f`
> 性质：🟡 中 · #8（堆叠上传）范围补全 + HCI 体验一致性
> 承继：[2026-06-24-frontend-phase0-8-stacked-upload-attachments-spec.md](2026-06-24-frontend-phase0-8-stacked-upload-attachments-spec.md)（#8 只覆盖输入框，本 spec 补发送后的消息流）

---

## 〇、问题（实测坐实）

用户上传几十~几百份 EPM 文件，**输入框已由 #8 堆叠收纳（发送前 OK）**，但**按回车发送后，消息流里的附件仍平铺成一整面**（实测截图：几十个 XLSX 卡片 3 列铺满整屏，把对话内容挤没）。

**根因**：`RichFilesList`（[message-list-item.tsx:303](../../packages/agent/frontend/src/components/workspace/messages/message-list-item.tsx#L303)）是无脑全平铺——与 #8 修复前的输入框同病：
```jsx
<div className="mb-2 flex flex-wrap justify-end gap-2">
  {files.map((file, index) => <RichFileCard .../>)}   // ← 全量 map，几十个就铺满
</div>
```
**#8 范围盲区**：#8 spec 只改 `input-box.tsx`（发送前，数据源 `usePromptInputAttachments()` store）；发送后的消息附件是**另一个组件 `RichFilesList`** + **另一个数据源** `message.additional_kwargs.files`（`FileInMessage[]`），#8 完全没覆盖。

**HCI 问题（用户指出）**：输入框堆叠了、发送瞬间却炸开成一整面——堆叠的收纳价值发送后立刻消失，体验断裂。

**目标**：发送后的消息附件**复用 #8 同一套堆叠+扇开**，与输入框体验连贯一致（堆叠→hover/点击扇开）。

---

## 一、复用可行性（已读码坐实）

| #8 资产 | 复用性 | 说明 |
|---|---|---|
| `partitionAttachments<T>`（partition-attachments.ts） | ✅ **零改动直接用** | 纯泛型函数（`<T>` 无 React 依赖），`FileInMessage[]` 直接传入，阈值/堆叠逻辑完全复用 |
| `AttachmentStack`（attachment-stack.tsx） | ✅ 直接复用 | `children` 是任意 ReactNode，不强依赖删除；`--shadow-overlap` 堆叠视觉 + hover/tap 扇开（Radix HoverCard）已就绪 |
| `FanOutList` / `AttachmentChip` | ⚠️ 需适配类型 | 当前吃 `Attachment = PromptInputFilePart & {id}`（`{id,url,mediaType,filename}`），发送后是 `FileInMessage`（`{filename,path,status}`）——字段不同，需 map 适配 |

**关键差异（消息流场景）**：
1. **数据源**：发送后是 `FileInMessage[]`（`core/messages/utils.ts:510`），非 attachments store。
2. **只读**：消息已发出，**不能删文件** → `onRemove` 应去掉（传 no-op 或让组件支持可选）。
3. **URL 解析**：`FileInMessage.path` 经 `resolveArtifactURL(path, threadId)` 取 URL（`RichFileCard` 现有逻辑），非 store 的 `file.url`。

---

## 二、改动方案

### 1. 适配函数：`FileInMessage` → `Attachment` 形状

在消息流侧把 `FileInMessage` map 成 `AttachmentChip` 能吃的形状（复用现有展示组件，不重写卡片）：
```typescript
// message-list-item.tsx 内或新建 messages/message-attachments.tsx
function toAttachment(file: FileInMessage, threadId: string): Attachment {
  return {
    id: `${file.filename}`,   // 消息内文件名唯一即可（同名极少；必要时加 index）
    filename: file.filename,
    url: file.path ? resolveArtifactURL(file.path, threadId) : undefined,
    mediaType: /* 按扩展名推 image/* 让 AttachmentChip 走缩略图分支，否则普通卡 */,
  };
}
```

### 2. 新建 `MessageAttachments`（只读堆叠，替 `RichFilesList`）

```tsx
// packages/agent/frontend/src/components/workspace/messages/message-attachments.tsx
function MessageAttachments({ files, threadId }: { files: FileInMessage[]; threadId: string }) {
  const [open, setOpen] = useState(false);
  if (files.length === 0) return null;

  // uploading 态的文件保持原 RichFileCard 的 spinner 展示（见 §三）；已上传的走堆叠
  const attachments = files.map((f) => toAttachment(f, threadId));
  const { flat, stacked, stackedCount } = partitionAttachments(attachments);

  return (
    <div className="mb-2 flex w-full flex-wrap justify-end gap-2">
      {flat.map((a) => <AttachmentChip data={a} key={a.id} />)}  {/* 无 onRemove = 只读 */}
      {stackedCount > 0 && (
        <AttachmentStack count={stackedCount} open={open} onOpenChange={setOpen}>
          <FanOutList items={stacked} onRemove={() => {}} />  {/* no-op：消息已发不可删 */}
        </AttachmentStack>
      )}
    </div>
  );
}
```
- **`onRemove` 的处理**：`AttachmentChip` 的 `onRemove` 是可选（`onRemove?:`），只读时**不传**即不渲染删除按钮（读 attachment-chip.tsx:71 `{onRemove && (...)}`）。`FanOutList` 的 `onRemove` 当前必传 → **改成可选**（`onRemove?:`），或传 no-op；推荐把 `FanOutList`/`FanOutRow` 的 `onRemove` 改可选（小改，让两端共用，删除按钮仅 store 端渲染）。

### 3. 替换 `RichFilesList` 调用

`message-list-item.tsx:174-176`：
```tsx
const filesList =
  files && files.length > 0 && thread_id ? (
    <MessageAttachments files={files} threadId={thread_id} />   // ← 替 RichFilesList
  ) : null;
```

---

## 三、必须保留的现有行为（别回归）

1. **uploading 态**：`RichFileCard` 现有 `isUploading` 分支（spinner + "上传中"，message-list-item.tsx 上传 mock 消息用）。**堆叠不能吞掉上传进度**——上传中的文件应仍可见进度。方案：uploading 态文件不进堆叠（始终 flat 展示），或 `AttachmentStack` 的 `progressLabel` 聚合显示（#8 已有该 prop）。**推荐**：上传中全 flat + stack 的 `progressLabel` 显示聚合进度（复用 #8 能力），上传完再归入堆叠。
2. **图片附件缩略图**：`RichFileCard` 的 `isImage` 分支渲染图片预览。`AttachmentChip` 已有 image 分支（`data.mediaType?.startsWith("image/") && data.url`）——`toAttachment` 正确推 `mediaType` 即复用。
3. **`justify-end`**：用户消息附件靠右对齐（现有 `RichFilesList` 是 `justify-end`），保留。
4. **向后兼容**：`<uploaded_files>` tag 解析路径（`parseUploadedFiles`）产出的也是 `FileInMessage[]`，同样走 `MessageAttachments`，不破。

---

## 四、设计纪律（守 #8 + 母方案）

- **复用不重写**：堆叠/扇开视觉全用 #8 的 `AttachmentStack`/`FanOutList`/`AttachmentChip`/`partitionAttachments`，不另造一套（SSOT，与输入框体验严格一致）。
- **阈值一致**：`STACK_THRESHOLD=5`（#8 已定，partition-attachments.ts），消息流沿用同值。
- **扇开列表 >50 虚拟化**：`FanOutList` 已用 `@tanstack/react-virtual`（#8），消息流复用即得（几百文件扇开不卡）。
- **token**：堆叠用 `--shadow-overlap`、扇开用 spec#1 曲线（#8 已接，复用即合规）。
- **不碰流式核心 / generated**：只改 `message-list-item.tsx`（替渲染）+ 新建 `message-attachments.tsx` + `FanOutList` onRemove 改可选；不动 `core/`、不动 ai-elements。

---

## 五、验证

1. `pnpm check`（lint + tsc）。
2. **真机**（`make dev`，上传几十份 EPM XLSX 发送）：
   - 发送后消息气泡里附件**堆叠成一叠 + "+N"**，不再平铺满屏。
   - hover（桌面）/ 点击（触屏）**扇开**列表，与输入框体验一致。
   - 上传中文件进度可见（不被堆叠吞）。
   - 图片附件缩略图正常。
   - ≤5 个文件仍平铺（阈值内不堆叠）。
3. **回归**：单文件 / 0 文件 / `<uploaded_files>` tag 老消息渲染不破。

---

## 六、关键文件

- `packages/agent/frontend/src/components/workspace/messages/message-list-item.tsx`（替 `RichFilesList` 调用为 `MessageAttachments`；`RichFileCard` 的 image/uploading 分支逻辑可挪进新组件或保留供 flat 用）
- **新建** `packages/agent/frontend/src/components/workspace/messages/message-attachments.tsx`（只读堆叠，复用 #8 组件）
- `packages/agent/frontend/src/components/workspace/attachments/fan-out-list.tsx`（`onRemove` 改可选，让只读端共用）
- **复用（不改）**：`attachments/partition-attachments.ts`、`attachments/attachment-stack.tsx`、`attachments/attachment-chip.tsx`
- 参考数据形状：`core/messages/utils.ts:510`（`FileInMessage`）

---

*依据：实测发送后附件平铺截图 + 读码坐实 `RichFilesList`(message-list-item.tsx:303) 无脑 flex-wrap + #8 组件复用性分析（partitionAttachments 纯泛型零改动、AttachmentStack children 不依赖删除、FanOutList onRemove 需改可选）。承继 #8。*
