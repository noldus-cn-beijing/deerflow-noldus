/**
 * Threshold-based partitioning for the stacked-attachments component (Phase0#8).
 *
 * Researchers routinely upload dozens–hundreds of EPM trial files. Flat-wrapping
 * every chip floods the input area (spec §1.1). Instead, files beyond a fixed
 * threshold collapse into a single stack with a "+N" badge; the flat prefix
 * stays a constant size regardless of total count (spec §3.1: "平铺数固定不随
 * 总数涨").
 *
 * This module is a pure function (no React) so the partition contract is unit
 * tested independently of rendering. Reused in `stacked-attachments.tsx`.
 */

export const STACK_THRESHOLD = 5;

export interface PartitionResult<T> {
  /** Files shown as normal chips. Constant length once stacking kicks in. */
  flat: T[];
  /** Files collapsed into the "+N" stack. Empty while total ≤ threshold. */
  stacked: T[];
  /** Count behind the "+N" badge (stacked.length, surfaced for readability). */
  stackedCount: number;
}

/**
 * Split `files` into a fixed-size flat prefix + an overflow stack.
 *
 * Contract:
 * - total ≤ threshold  → all flat, empty stack.
 * - total >  threshold  → first `threshold - 1` flat, remainder stacked.
 *
 * "threshold - 1" (not "threshold") so the stack slot itself occupies one
 * visible position — the flat row width stays stable as the stack grows.
 */
export function partitionAttachments<T>(
  files: readonly T[],
  threshold: number = STACK_THRESHOLD,
): PartitionResult<T> {
  const count = files.length;
  if (count <= threshold) {
    return { flat: [...files], stacked: [], stackedCount: 0 };
  }
  const flatCount = threshold - 1;
  return {
    flat: files.slice(0, flatCount),
    stacked: files.slice(flatCount),
    stackedCount: count - flatCount,
  };
}
