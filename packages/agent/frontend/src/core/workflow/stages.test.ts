import { describe, expect, it } from "vitest";

import {
  COLUMN_ALIGNMENT_HINTS,
  WORKFLOW_STAGE_IDS,
  WORKFLOW_STAGES,
  isColumnAlignmentClarification,
} from "./stages";

describe("WORKFLOW_STAGES (SSOT)", () => {
  it("defines exactly the 7 domain stages in fixed order", () => {
    expect(WORKFLOW_STAGES).toHaveLength(7);
    expect(WORKFLOW_STAGES.map((s) => s.id)).toEqual([
      "upload",
      "paradigm",
      "align",
      "compute",
      "qc",
      "interpret",
      "report",
    ]);
  });

  it("each stage has a stable id, i18n name key, icon, and a stage color token", () => {
    for (const stage of WORKFLOW_STAGES) {
      expect(stage.id).toBeTruthy();
      expect(stage.nameKey).toBeTruthy();
      expect(stage.icon).toBeTruthy();
      // token must be one of the spec#1 --color-stage-* variables
      expect(stage.colorToken).toMatch(/^var\(--color-stage-[a-z]+\)$/);
    }
  });

  it("WORKFLOW_STAGE_IDS mirrors the stage order (for O(1) lookup by id)", () => {
    expect(WORKFLOW_STAGE_IDS.upload).toBe(0);
    expect(WORKFLOW_STAGE_IDS.report).toBe(6);
    expect(WORKFLOW_STAGE_IDS.align).toBe(2);
  });

  it("stage ids are unique", () => {
    const ids = WORKFLOW_STAGES.map((s) => s.id);
    expect(new Set(ids).size).toBe(ids.length);
  });
});

describe("isColumnAlignmentClarification (stage ③ signal heuristic)", () => {
  it("matches when clarification_type hints at column / zone semantics", () => {
    expect(
      isColumnAlignmentClarification({
        clarification_type: "ambiguous_requirement",
        question: "中心区和边缘区分别对应哪两列？",
        options: ["A列/B列", "C列/D列"],
      }),
    ).toBe(true);
  });

  it("matches on zone/arena keywords in the question even without an explicit type", () => {
    expect(
      isColumnAlignmentClarification({
        question: "Which column is the Open arena zone?",
      }),
    ).toBe(true);
  });

  it("matches English column / center / periphery keywords", () => {
    expect(
      isColumnAlignmentClarification({
        question: "Select the center and periphery columns",
      }),
    ).toBe(true);
  });

  it("does NOT match an unrelated clarification (e.g. paradigm grouping choice)", () => {
    expect(
      isColumnAlignmentClarification({
        clarification_type: "approach_choice",
        question: "Which statistical test do you prefer?",
        options: ["t-test", "Mann-Whitney"],
      }),
    ).toBe(false);
  });

  it("falls back to false on empty input (conservative — never a false 'done')", () => {
    expect(isColumnAlignmentClarification({})).toBe(false);
    expect(isColumnAlignmentClarification({ question: "" })).toBe(false);
  });

  it("exposes the hint keyword list (documented, not magic)", () => {
    // Chinese domain terms used by ethoinsight-column-confirmation skill +
    // English equivalents. Asserting presence keeps the heuristic honest.
    expect(COLUMN_ALIGNMENT_HINTS).toContain("分析区");
    expect(COLUMN_ALIGNMENT_HINTS).toContain("列");
    expect(COLUMN_ALIGNMENT_HINTS).toContain("zone");
  });
});
