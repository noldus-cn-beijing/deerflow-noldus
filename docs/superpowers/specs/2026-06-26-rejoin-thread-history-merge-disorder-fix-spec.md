# Spec B：重进 thread 历史合并乱序/消息消失（前端流式核心 · 🔴 红线区）

> 状态：**诊断 spec + 修复方向**（⚠️ 改动落在踩坑沉淀的流式核心，**不是可随手实施的 spec**——见 §四 红线纪律）
> 日期：2026-06-26
> 代码基线：dev HEAD `5616a73f`
> 性质：🔴 高风险 · 触及 `mergeMessages`/`dedupeMessagesByIdentity`/optimistic（memory 反复标红"重写必复发"）
> 取证：dogfood thread `e9837b33`——用户重新点进 thread 后，input 数据消息跑到中间、之前部分输出消失（非 summarization 压缩，该 thread log 无归档触发）

---

## ⚠️ 〇、红线警告（实施前必读）

本 spec 触及的 `hooks.ts` 的 `mergeMessages`/`messageIdentity`/`dedupeMessagesByIdentity`/optimistic 是 **CLAUDE.md + memory 反复标红的流式核心**：
> "不动 `useStream` / `mergeMessages` / `dedupeMessagesByIdentity` / optimistic / summarization——踩坑沉淀，重写必复发。"（`project_2026-06-24_frontend_generative_ux_upgrade_plan`）

**本 spec 是诊断 + 修复方向，不是"照着改"的实施单**。任何改动必须：① 先复现（拿到该 thread 原始 history API 响应）② grill 改动对所有合并场景的影响 ③ 全量跑 `utils.test.ts` 的 groupMessages/streaming-continuity 测试 + 新增乱序回归测试。**别因为"看起来是个排序 bug"就随手改 slice/sort——这块每一个分支都是踩坑沉淀。**

---

## 一、现象（dogfood 实测）

用户跑完一个 EPM 端到端分析后，**重新点进该 thread**，前端消息流：
1. **顺序乱**：最开始的 input（28 个 XLSX 上传）消息跑到了**中间**，不在最前。
2. **之前输出消失**：一部分之前的输出不见了（用户疑似"被压缩"）。

**排除 summarization**：该 thread 的 gateway.log **无** summariz/archiv 触发记录 → "消失"不是归档压缩导致，是消息合并逻辑切掉的。

---

## 二、根因诊断（读码定位，待原始数据坐实）

重进 thread 时走**历史加载 + 实时合并**：`hooks.ts:119` `mergeMessages(historyMessages, threadMessages, optimisticMessages)`：

```js
// hooks.ts:134-152
// overlap = history 尾部与 thread 重叠的连续后缀；从尾部扫，shrink cutoff，遇到不重叠就 break
let cutoff = historyMessages.length;
for (let i = historyMessages.length - 1; i >= 0; i--) {
  const identity = messageIdentity(historyMessages[i]);
  if (identity && threadMessageIds.has(identity)) { cutoff = i; }
  else { break; }   // ← 提前 break 的脆弱点
}
return dedupeMessagesByIdentity([...historyMessages.slice(0, cutoff), ...threadMessages, ...optimistic]);
```

**脆弱点 1 — `messageIdentity` 对无 id 消息返回 `undefined`**（`hooks.ts:59-71`）：身份键只认 `tool:${tool_call_id}` 或 `message:${id}`，**两者都没有 → `undefined`**。overlap 循环里遇到 `identity===undefined` → `threadMessageIds.has(undefined)` 为 false → **`break` 提前触发**，`cutoff` 停在错误位置。

**脆弱点 2 — 提前 break 的双重后果**：
- `cutoff` 停早 → `history.slice(0, cutoff)` **切掉了本该保留的 history 段** → **"之前输出消失"**。
- 或重叠判断错位 → history 段与 thread 段拼接顺序错 → **"input 跑到中间"**（input 消息若 identity 不稳定，没被正确识别为"最前/重叠"，被插错位置）。

**脆弱点 3 — `dedupeMessagesByIdentity` 用 `lastIndexByIdentity` 保留最后一次**（`hooks.ts:73-83`）：无 id 消息每条 identity 都 `undefined`，dedupe 对它们的处理（保留/丢弃）取决于实现细节，可能导致同内容消息重复或丢失。

> **核心假设（待坐实）**：input/上传类消息或某些流式中间消息**缺稳定 identity**（无 `id` 也无 `tool_call_id`），在 history/live 边界上触发 overlap 提前 break + dedupe 错判，产出乱序 + 消失。

---

## 三、坐实步骤（实施前必做，不可跳）

1. **拿原始数据**：用 owner 账号调 `GET /api/threads/{tid}/runs/{rid}/messages`（hooks.ts:196 的端点）拿该 thread 的**原始 history 响应**，检查：
   - history 里 input/上传消息有没有稳定 `id`？顺序对不对（后端返回就乱，还是前端合并搞乱）？
   - 有多少消息 `messageIdentity` 返回 `undefined`？
2. **隔离复现**：把原始 history + 一段 live messages 喂给 `mergeMessages`（纯函数，可单测），断言输出顺序——复现"input 在中间 + 消失"。这一步把"线上现象"变成"可调试的失败测试"。
3. **区分根因层**：① 后端 `/messages` 返回顺序就乱（改后端排序）② 前端 `mergeMessages` 合并搞乱（改前端，红线）③ `messageIdentity` 覆盖不全（给 input 消息补稳定 id）。**先坐实是哪一层，别假设是前端。**

---

## 四、修复方向（坐实后择一，按风险从低到高）

> 优先选**不碰核心合并算法**的方向。

- **方向1（最低风险，首选）— 后端保证 history 有序 + 消息带稳定 id**：若坐实是后端 `/messages` 返回顺序乱或 input 消息缺 id → 改后端：`/messages` 端点按 `seq`/`created_at` 严格排序（`runs.py` 已有 `before_seq`/`after_seq` 分页，确认排序键）；input/上传消息持久化时带稳定 `id`。**不碰前端红线。**
- **方向2（中风险）— `messageIdentity` 扩展覆盖**：给无 id/无 tool_call_id 的消息一个**稳定 fallback identity**（如基于 content hash + 角色 + 序号），让 overlap 检测不再因 `undefined` 提前 break。**改 `messageIdentity` 是红线内较局部的改动**，但必须全量回归 + 新增乱序测试。
- **方向3（最高风险，仅当 1/2 不解决）— 重构 overlap 检测**：overlap 不靠"从尾 break"，改成基于稳定排序键的归并。**这是动核心算法，需最高强度 grill + 全量测试 + 多场景验证**（首条/末条/中间 break/全重叠/无重叠/optimistic 在途）。

**通用纪律**：任何方向都必须保留 `utils.test.ts` 现有的 groupMessages + streaming-continuity（no-flicker）测试全绿，且**新增"重进 thread 历史乱序"专项回归测试**（把 §三步骤2 的失败用例固化）。

---

## 五、验证

1. **§三坐实先行**：拿到原始 history 数据 + 隔离复现失败测试——**没复现不准改**。
2. 修复后：该失败测试转绿 + `utils.test.ts` 全量绿 + `pnpm check`。
3. **dogfood**：重进多个跑完的 thread（含本 thread `e9837b33` 等存量），确认 input 在最前、无消息丢失、顺序正确。
4. **多场景手测**：流式进行中刷新 / 跑完刷新 / 多轮对话刷新 / optimistic 在途时刷新——都不乱不丢。
5. **回归**：别破坏正常流式（实时消息追加无闪烁、无重复）。

---

## 六、关键文件

- `packages/agent/frontend/src/core/threads/hooks.ts:59`（messageIdentity）/ `:73`（dedupe）/ `:119-160`（mergeMessages）— **红线核心**
- `packages/agent/frontend/src/core/messages/utils.ts:152`（groupMessages）/ `utils.test.ts`（必守测试）
- `packages/agent/backend/app/gateway/routers/runs.py`（/messages 端点排序 — 方向1）
- 红线依据：memory `project_2026-06-24_frontend_generative_ux_upgrade_plan`（不动流式核心）

---

*依据：dogfood thread `e9837b33` 重进乱序现象 + 读码定位 mergeMessages overlap 提前 break（messageIdentity 对无 id 消息返 undefined）+ 排除 summarization（log 无归档）。**本 spec 触流式红线，是诊断+方向，坐实原始数据后方可改，改动需 grill + 全量回归。***
