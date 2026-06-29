/**
 * G4 方案 C: 业务语义播报识别
 *
 * 把 tool_call event 的"被动渲染"升级为"语义化状态条"。
 * UI 调本 module 函数把 tool_call 翻译成用户友好的中文文案：
 *
 *   tool_call: task(subagent_type="data-analyst", ...)
 *   → "🔬 指标已完成，正在请专家解读…"
 *
 *   tool_call: bash("python -m ethoinsight.catalog.resolve ...")
 *   → "📋 正在生成指标计划…"
 *
 * 设计原则：
 *   1. 单一识别入口——不在 SubtaskCard / ToolCall 散落识别逻辑
 *   2. fallback 友好——未匹配的 subagent_type / bash 命令返回通用文案，不抛错
 *   3. 文案走 i18n——本 module 不硬编码中文，所有文案从 t.toolCalls.stageBroadcast.* 取
 */

import type { Translations } from "@/core/i18n";

/**
 * 这两个函数只读 `t.toolCalls.stageBroadcast.*`。参数类型用窄的 Pick，既兼容现有调用方
 * （传完整 Translations 仍满足 Pick），又让只持有 Translations 子集的消费者能直接复用，
 * 不必强转（LSP 安全）。
 */
type StageBroadcastTranslations = Pick<Translations, "toolCalls">;

/**
 * Subagent type → 状态播报文案。
 * 未知 subagent_type 走通用 "正在派遣 <type>…" fallback。
 */
export function getStageBroadcastForSubagent(
  subagentType: string,
  t: StageBroadcastTranslations,
): string {
  return t.toolCalls.stageBroadcast.dispatchSubagent(subagentType);
}

/**
 * 识别 EthoInsight CLI command pattern。
 * 仅对带 `python -m ethoinsight.<module>.<script>` 模式的命令返回非 null。
 */
export function detectEthoinsightCli(
  command: string,
): "parse" | "catalog" | "scripts" | null {
  // 容忍前导空白、python/python3
  const match = /^python3?\s+-m\s+ethoinsight\.(parse|catalog|scripts)\.(\w+)/.exec(
    command.trim(),
  );
  if (!match) return null;
  return match[1] as "parse" | "catalog" | "scripts";
}

/**
 * bash command → 状态播报文案（仅识别 EthoInsight CLI，其他返回 null 表示"用通用文案"）。
 */
export function getStageBroadcastForBash(
  command: string,
  t: StageBroadcastTranslations,
): string | null {
  const cliKind = detectEthoinsightCli(command);
  if (cliKind === "parse") {
    return t.toolCalls.stageBroadcast.parseHeaders;
  }
  if (cliKind === "catalog") {
    return t.toolCalls.stageBroadcast.resolveCatalog;
  }
  if (cliKind === "scripts") {
    // 提取脚本名 (ethoinsight.scripts.epm.compute_open_arm_time_ratio → compute_open_arm_time_ratio)
    const scriptMatch = /ethoinsight\.scripts\.\w+\.(\w+)/.exec(command);
    const scriptName = scriptMatch?.[1] ?? "script";
    return t.toolCalls.stageBroadcast.runScript(scriptName);
  }
  return null;
}
