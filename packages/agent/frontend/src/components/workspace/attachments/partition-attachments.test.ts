import { describe, expect, it } from "vitest";

import {
  partitionAttachments,
  STACK_THRESHOLD,
} from "@/components/workspace/attachments/partition-attachments";

interface FileLike {
  id: string;
  filename: string;
}

function makeFiles(n: number): FileLike[] {
  return Array.from({ length: n }, (_, i) => ({
    id: `f${i}`,
    filename: `Trial${i}.xlsx`,
  }));
}

describe("partitionAttachments", () => {
  it("exposes a STACK_THRESHOLD constant defaulting to 5", () => {
    expect(STACK_THRESHOLD).toBe(5);
  });

  it("puts all files in the flat group and leaves the stack empty when count <= threshold", () => {
    for (const n of [0, 1, 3, 5]) {
      const { flat, stacked } = partitionAttachments(makeFiles(n));
      expect(flat).toHaveLength(n);
      expect(stacked).toHaveLength(0);
    }
  });

  it("collapses the overflow into the stack once count exceeds the threshold", () => {
    const { flat, stacked } = partitionAttachments(makeFiles(10));
    // Flat count is fixed (threshold - 1 = 4) and does NOT grow with total.
    expect(flat).toHaveLength(4);
    expect(stacked).toHaveLength(6);
  });

  it("keeps the flat count fixed regardless of how large the total grows", () => {
    const small = partitionAttachments(makeFiles(50));
    const large = partitionAttachments(makeFiles(200));
    expect(small.flat).toHaveLength(4);
    expect(large.flat).toHaveLength(4);
    expect(small.stacked).toHaveLength(46);
    expect(large.stacked).toHaveLength(196);
  });

  it("preserves order: flat holds the first files, stacked holds the trailing ones contiguously", () => {
    const { flat, stacked } = partitionAttachments(makeFiles(8));
    expect(flat.map((f) => f.id)).toEqual(["f0", "f1", "f2", "f3"]);
    expect(stacked.map((f) => f.id)).toEqual(["f4", "f5", "f6", "f7"]);
  });

  it("accepts a custom threshold", () => {
    const { flat, stacked } = partitionAttachments(makeFiles(10), 3);
    expect(flat).toHaveLength(2);
    expect(stacked).toHaveLength(8);
  });

  it("returns the stacked count for the +M badge", () => {
    // 7 files, threshold 5 → flat=4 (threshold-1), stacked=3.
    expect(partitionAttachments(makeFiles(7)).stackedCount).toBe(3);
    expect(partitionAttachments(makeFiles(5)).stackedCount).toBe(0);
  });
});
