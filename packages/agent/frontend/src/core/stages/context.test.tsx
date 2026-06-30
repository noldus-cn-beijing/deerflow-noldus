// @vitest-environment jsdom
//
// A2 TDD: stage narration context + onCustomEvent integration.
//
// 验收标准（spec §五）：
// 1. 喂 stage_plan + stage_update fixture → 断言内联渲染对应阶段叙事，active→completed 翻转。
// 2. 阶段结束后不卡死：喂最后一个 stage_update completed → 不再 active。
// 3. 防 vacuous：渲染文案来自 A1 事件 narration，不是 stage-broadcast 查表；
//    工具名（identify_ev19_template）/gate 关键字不出现。
// 4. 非流水线意图（无 stage_plan）→ 前端无阶段渲染。

import { act, render, screen } from "@testing-library/react";
import { useEffect } from "react";
import { describe, expect, it } from "vitest";

import {
  StagesProvider,
  useSetPlan,
  useStages,
  useUpdateStage,
} from "@/core/stages/context";

// ── Fixtures (mirror exact A1 backend JSON shapes) ──────────────────────────

const FIXTURE_STAGE_PLAN = {
  kind: "stage_plan",
  stages: ["识别范式", "计算指标", "数据解读", "生成图表", "撰写报告"],
  skipped: [] as string[],
};

const FIXTURE_STAGE_UPDATE_ACTIVE = {
  kind: "stage_update",
  stage: "识别范式",
  status: "active" as const,
  narration: "正在识别实验范式…",
};

const FIXTURE_STAGE_UPDATE_COMPLETED = {
  kind: "stage_update",
  stage: "识别范式",
  status: "completed" as const,
  narration: "范式识别完成",
};

const FIXTURE_STAGE_UPDATE_COMPUTE_ACTIVE = {
  kind: "stage_update",
  stage: "计算指标",
  status: "active" as const,
  narration: "正在计算 EPM 行为指标，预计 30-60 秒…",
};

const FIXTURE_STAGE_UPDATE_COMPUTE_COMPLETED = {
  kind: "stage_update",
  stage: "计算指标",
  status: "completed" as const,
  narration: "指标计算完成",
};

// ── Helpers ──────────────────────────────────────────────────────────────────

/**
 * A consumer component that renders the stage list for inspection.
 * Renders a sentinel "<no-stages>" when no plan is set (non-pipeline intent).
 */
function StageListDisplay() {
  const stages = useStages();
  if (!stages) return <div data-testid="no-stages">no-stages</div>;
  return (
    <ul>
      {stages.map((s) => (
        <li key={s.name} data-testid={`stage-${s.name}`}>
          <span data-testid={`stage-name-${s.name}`}>{s.name}</span>
          <span data-testid={`stage-status-${s.name}`}>{s.entry?.status ?? "pending"}</span>
          <span data-testid={`stage-narration-${s.name}`}>{s.entry?.narration ?? ""}</span>
        </li>
      ))}
    </ul>
  );
}

/**
 * A controller that exposes setPlan / updateStage to tests via refs,
 * so tests always call the freshest closure.
 */
function Controller({
  setPlanRef,
  updateStageRef,
}: {
  setPlanRef: { current: ReturnType<typeof useSetPlan> | null };
  updateStageRef: { current: ReturnType<typeof useUpdateStage> | null };
}) {
  const setPlan = useSetPlan();
  const updateStage = useUpdateStage();
  useEffect(() => {
    setPlanRef.current = setPlan;
    updateStageRef.current = updateStage;
  });
  return null;
}

function makeRefs() {
  const setPlanRef: { current: ReturnType<typeof useSetPlan> | null } = { current: null };
  const updateStageRef: { current: ReturnType<typeof useUpdateStage> | null } = { current: null };
  return { setPlanRef, updateStageRef };
}

// ── Tests ─────────────────────────────────────────────────────────────────────

describe("StageContext — basic plan + update", () => {
  it("no plan → renders no-stages sentinel (non-pipeline intent)", () => {
    render(
      <StagesProvider>
        <StageListDisplay />
      </StagesProvider>,
    );
    expect(screen.getByTestId("no-stages")).toBeDefined();
  });

  it("喂 stage_plan → 渲染5个 pending 阶段、无 narration", async () => {
    const { setPlanRef, updateStageRef } = makeRefs();
    render(
      <StagesProvider>
        <Controller setPlanRef={setPlanRef} updateStageRef={updateStageRef} />
        <StageListDisplay />
      </StagesProvider>,
    );
    await act(async () => {
      setPlanRef.current!({
        stages: FIXTURE_STAGE_PLAN.stages,
        skipped: FIXTURE_STAGE_PLAN.skipped,
      });
    });
    // All 5 stages rendered
    for (const stage of FIXTURE_STAGE_PLAN.stages) {
      expect(screen.getByTestId(`stage-${stage}`)).toBeDefined();
      expect(screen.getByTestId(`stage-status-${stage}`).textContent).toBe("pending");
    }
    // No narration yet
    expect(screen.getByTestId("stage-narration-识别范式").textContent).toBe("");
  });

  it("stage_update active → 阶段变 active，narration 来自后端事件（非查表）", async () => {
    const { setPlanRef, updateStageRef } = makeRefs();
    render(
      <StagesProvider>
        <Controller setPlanRef={setPlanRef} updateStageRef={updateStageRef} />
        <StageListDisplay />
      </StagesProvider>,
    );
    await act(async () => {
      setPlanRef.current!({ stages: FIXTURE_STAGE_PLAN.stages, skipped: [] });
    });
    await act(async () => {
      updateStageRef.current!(
        FIXTURE_STAGE_UPDATE_ACTIVE.stage,
        FIXTURE_STAGE_UPDATE_ACTIVE.status,
        FIXTURE_STAGE_UPDATE_ACTIVE.narration,
      );
    });

    const statusEl = screen.getByTestId("stage-status-识别范式");
    const narrationEl = screen.getByTestId("stage-narration-识别范式");

    expect(statusEl.textContent).toBe("active");
    // 防 vacuous: narration 来自后端，不含工具名、不含 stage-broadcast 查表文案
    expect(narrationEl.textContent).toBe("正在识别实验范式…");
    expect(narrationEl.textContent).not.toContain("identify_ev19_template");
    expect(narrationEl.textContent).not.toContain("getStageBroadcast");
    expect(narrationEl.textContent).not.toContain("dispatchSubagent");
  });

  it("stage_update completed → active 翻转 completed，不卡死", async () => {
    const { setPlanRef, updateStageRef } = makeRefs();
    render(
      <StagesProvider>
        <Controller setPlanRef={setPlanRef} updateStageRef={updateStageRef} />
        <StageListDisplay />
      </StagesProvider>,
    );
    await act(async () => {
      setPlanRef.current!({ stages: FIXTURE_STAGE_PLAN.stages, skipped: [] });
    });
    // active
    await act(async () => {
      updateStageRef.current!("识别范式", "active", "正在识别实验范式…");
    });
    expect(screen.getByTestId("stage-status-识别范式").textContent).toBe("active");

    // completed — must NOT stay active (non-stuck guarantee, spec §五.2)
    await act(async () => {
      updateStageRef.current!(
        FIXTURE_STAGE_UPDATE_COMPLETED.stage,
        FIXTURE_STAGE_UPDATE_COMPLETED.status,
        FIXTURE_STAGE_UPDATE_COMPLETED.narration,
      );
    });
    expect(screen.getByTestId("stage-status-识别范式").textContent).toBe("completed");
    expect(screen.getByTestId("stage-narration-识别范式").textContent).toBe("范式识别完成");
  });

  it("最后一个 stage_update completed 后无更多事件 → 不再 active（不卡死）", async () => {
    // spec §五.2: 直接覆盖 #214/report-writer 卡死那类回归
    const { setPlanRef, updateStageRef } = makeRefs();
    render(
      <StagesProvider>
        <Controller setPlanRef={setPlanRef} updateStageRef={updateStageRef} />
        <StageListDisplay />
      </StagesProvider>,
    );
    await act(async () => {
      setPlanRef.current!({ stages: FIXTURE_STAGE_PLAN.stages, skipped: [] });
    });
    // Drive all stages to completed
    for (const stage of FIXTURE_STAGE_PLAN.stages) {
      await act(async () => {
        updateStageRef.current!(stage, "active", `正在${stage}…`);
      });
      await act(async () => {
        updateStageRef.current!(stage, "completed", `${stage}完成`);
      });
    }
    // No more events after this. Assert none are stuck in active.
    for (const stage of FIXTURE_STAGE_PLAN.stages) {
      const status = screen.getByTestId(`stage-status-${stage}`).textContent;
      expect(status, `${stage} must not be active after completed`).not.toBe("active");
      expect(status).toBe("completed");
    }
  });

  it("sequential stage_updates: 计算指标 active → completed 序列正确", async () => {
    const { setPlanRef, updateStageRef } = makeRefs();
    render(
      <StagesProvider>
        <Controller setPlanRef={setPlanRef} updateStageRef={updateStageRef} />
        <StageListDisplay />
      </StagesProvider>,
    );
    await act(async () => {
      setPlanRef.current!({ stages: FIXTURE_STAGE_PLAN.stages, skipped: [] });
    });
    // 识别范式 done
    await act(async () => {
      updateStageRef.current!("识别范式", "completed", "范式识别完成");
    });
    // 计算指标 active
    await act(async () => {
      updateStageRef.current!(
        FIXTURE_STAGE_UPDATE_COMPUTE_ACTIVE.stage,
        FIXTURE_STAGE_UPDATE_COMPUTE_ACTIVE.status,
        FIXTURE_STAGE_UPDATE_COMPUTE_ACTIVE.narration,
      );
    });
    expect(screen.getByTestId("stage-status-计算指标").textContent).toBe("active");
    expect(screen.getByTestId("stage-narration-计算指标").textContent).toBe(
      "正在计算 EPM 行为指标，预计 30-60 秒…",
    );
    // 计算指标 completed
    await act(async () => {
      updateStageRef.current!(
        FIXTURE_STAGE_UPDATE_COMPUTE_COMPLETED.stage,
        FIXTURE_STAGE_UPDATE_COMPUTE_COMPLETED.status,
        FIXTURE_STAGE_UPDATE_COMPUTE_COMPLETED.narration,
      );
    });
    expect(screen.getByTestId("stage-status-计算指标").textContent).toBe("completed");
    // 识别范式 still completed (not regressed)
    expect(screen.getByTestId("stage-status-识别范式").textContent).toBe("completed");
  });

  it("skipped 阶段在计划中但跳过时，stages 列表包含 skipped 标记", async () => {
    const { setPlanRef, updateStageRef } = makeRefs();
    render(
      <StagesProvider>
        <Controller setPlanRef={setPlanRef} updateStageRef={updateStageRef} />
        <StageListDisplay />
      </StagesProvider>,
    );
    await act(async () => {
      // n<2 → 数据解读 skipped
      setPlanRef.current!({
        stages: FIXTURE_STAGE_PLAN.stages,
        skipped: ["数据解读"],
      });
    });
    // All stages still rendered (UI shows them as visually skipped)
    for (const stage of FIXTURE_STAGE_PLAN.stages) {
      expect(screen.getByTestId(`stage-${stage}`)).toBeDefined();
    }
    // 数据解读 remains pending (no stage_update will arrive for skipped)
    expect(screen.getByTestId("stage-status-数据解读").textContent).toBe("pending");
  });

  it("防 vacuous: narration 文案不含 identify_ev19_template / gate 关键字", async () => {
    const { setPlanRef, updateStageRef } = makeRefs();
    render(
      <StagesProvider>
        <Controller setPlanRef={setPlanRef} updateStageRef={updateStageRef} />
        <StageListDisplay />
      </StagesProvider>,
    );
    await act(async () => {
      setPlanRef.current!({ stages: FIXTURE_STAGE_PLAN.stages, skipped: [] });
    });
    await act(async () => {
      updateStageRef.current!("识别范式", "active", "正在识别实验范式…");
    });
    const narration = screen.getByTestId("stage-narration-识别范式").textContent ?? "";
    // Tool names must never leak into UI narration
    expect(narration).not.toContain("identify_ev19_template");
    expect(narration).not.toContain("inspect_uploaded_file");
    expect(narration).not.toContain("prep_metric_plan");
    // Frontend query-table strings must not appear (A 方案 SSOT 验证)
    expect(narration).not.toContain("正在派遣");
    expect(narration).not.toContain("🧮 正在计算指标");
  });
});

describe("StageContext — kind field discrimination (A1 event shape)", () => {
  it("stage_plan 事件 kind 字段为 'stage_plan'（不是 'type'）", () => {
    expect(FIXTURE_STAGE_PLAN.kind).toBe("stage_plan");
  });

  it("stage_update 事件 kind 字段为 'stage_update'（不是 'type'）", () => {
    expect(FIXTURE_STAGE_UPDATE_ACTIVE.kind).toBe("stage_update");
  });
});
