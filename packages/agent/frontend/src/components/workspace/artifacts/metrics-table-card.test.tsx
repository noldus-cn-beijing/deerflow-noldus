// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { MetricsTableCard } from "./metrics-table-card";

/**
 * spec 2026-06-30 C1 模块3：指标结果表卡 overview-first。
 *
 * 用 mock 的 fetcher + i18n 隔离网络/翻译。重点钉死**性能红线**：
 * - 折叠态 DOM 不含 subject 行（懒 fetch + 条件渲染，非 CSS hide）。
 * - 展开后默认仍只渲染组综述；某组展开才渲染该组 subject 行。
 */

const fetcherMock = vi.fn();

vi.mock("@/core/api/fetcher", () => ({
  fetch: (...args: unknown[]) => fetcherMock(...args),
}));

const tGallery = {
  metricsTableTitle: "指标结果表",
  metricsTableExpand: "查看指标表",
  metricsTableGroupN: (n: number) => `n=${n}`,
  outlierFlag: "离群",
  downloadCsv: "下载 CSV",
  metricsTableEmpty: "暂无指标结果",
  metricsTableLoading: "加载中…",
};

vi.mock("@/core/i18n/hooks", () => ({
  useI18n: () => ({ t: { gallery: tGallery }, locale: "zh-CN" }),
}));

beforeEach(() => {
  fetcherMock.mockReset();
  const noop = () => undefined;
  window.matchMedia =
    window.matchMedia ?? ((_q: unknown) => ({ matches: false, addEventListener: noop, removeEventListener: noop }));
});
afterEach(() => cleanup());

const TABLE_JSON = {
  paradigm: "epm",
  metric_names: ["open_arm_time_ratio"],
  groups: [
    { group: "Control", n: 2, metrics: { open_arm_time_ratio: { mean: 0.35, std: 0.07, n: 2 } } },
    { group: "Treatment", n: 2, metrics: { open_arm_time_ratio: { mean: 0.6, std: 0.2, n: 2 } } },
  ],
  per_subject: [
    { subject: "ctrl_01", group: "Control", values: { open_arm_time_ratio: 0.3 }, outlier_flags: { open_arm_time_ratio: false } },
    { subject: "ctrl_02", group: "Control", values: { open_arm_time_ratio: 0.4 }, outlier_flags: { open_arm_time_ratio: false } },
    { subject: "drug_01", group: "Treatment", values: { open_arm_time_ratio: 0.5 }, outlier_flags: { open_arm_time_ratio: false } },
    { subject: "drug_02", group: "Treatment", values: { open_arm_time_ratio: 0.9 }, outlier_flags: { open_arm_time_ratio: true } },
  ],
};

function makeResponse(json: unknown) {
  return {
    ok: true,
    status: 200,
    json: async () => json,
  };
}

function makeMeta() {
  return { path: "/mnt/user-data/outputs/metrics_table.json", kind: "data" as const, filename: "metrics_table.json" };
}

describe("MetricsTableCard collapsed (overview-first red line)", () => {
  it("renders only the header when collapsed; fetch NOT called on mount (lazy)", () => {
    render(<MetricsTableCard meta={makeMeta()} threadId="t1" />);
    // 卡头标题在（meta.filename 优先 → metrics_table.json）
    expect(screen.getByText("metrics_table.json")).toBeInTheDocument();
    // 未 fetch（懒加载，挂载不拉）
    expect(fetcherMock).not.toHaveBeenCalled();
  });

  it("DOM does not contain subject rows when collapsed (performance red line)", () => {
    const { container } = render(<MetricsTableCard meta={makeMeta()} threadId="t1" />);
    // 任何 subject stem 都不该出现在 DOM 里
    expect(container.textContent).not.toContain("ctrl_01");
    expect(container.textContent).not.toContain("drug_02");
    expect(container.querySelectorAll('[data-testid^="metrics-group-rows-"]')).toHaveLength(0);
  });
});

describe("MetricsTableCard expand + group expand", () => {
  it("expanding fetches the metrics-table JSON once and renders group summary", async () => {
    fetcherMock.mockResolvedValueOnce(makeResponse(TABLE_JSON));
    const { container } = render(<MetricsTableCard meta={makeMeta()} threadId="t1" />);

    fireEvent.click(screen.getByText("查看指标表"));

    await waitFor(() => {
      expect(container.textContent).toContain("Control");
      expect(container.textContent).toContain("Treatment");
    });
    // fetch 一次，命中 metrics-table 端点
    expect(fetcherMock).toHaveBeenCalledTimes(1);
    expect(String(fetcherMock.mock.calls[0]?.[0])).toContain("/api/threads/t1/artifacts/metrics-table");
  });

  it("DOM still has no subject rows until a group is expanded (conditional render)", async () => {
    fetcherMock.mockResolvedValueOnce(makeResponse(TABLE_JSON));
    const { container } = render(<MetricsTableCard meta={makeMeta()} threadId="t1" />);
    fireEvent.click(screen.getByText("查看指标表"));

    // 等组综述渲染
    await waitFor(() => expect(screen.getByTestId("metrics-group-Control")).toBeInTheDocument());
    // 展开了卡，但没展开组 → 仍无 subject 行
    expect(container.textContent).not.toContain("ctrl_01");
    expect(container.querySelectorAll('[data-testid="metrics-group-rows-Control"]')).toHaveLength(0);

    // 展开某组 → 该组 subject 行条件渲染出现
    fireEvent.click(screen.getByTestId("metrics-group-Control"));
    await waitFor(() => {
      expect(container.textContent).toContain("ctrl_01");
      expect(container.textContent).toContain("ctrl_02");
    });
    expect(container.querySelectorAll('[data-testid="metrics-group-rows-Control"]')).toHaveLength(1);
    // 别的组未展开 → 仍无该组 subject 行
    expect(container.textContent).not.toContain("drug_01");
    expect(container.querySelectorAll('[data-testid="metrics-group-rows-Treatment"]')).toHaveLength(0);
  });
});

describe("MetricsTableCard outlier marking + CSV download", () => {
  it("outlier row has the marker and full opacity; non-outlier has reduced opacity", async () => {
    fetcherMock.mockResolvedValueOnce(makeResponse(TABLE_JSON));
    const { container } = render(<MetricsTableCard meta={makeMeta()} threadId="t1" />);
    fireEvent.click(screen.getByText("查看指标表"));
    await waitFor(() => screen.getByTestId("metrics-group-Treatment"));
    fireEvent.click(screen.getByTestId("metrics-group-Treatment"));
    await waitFor(() => expect(container.textContent).toContain("drug_02"));

    // 离群行（drug_02）有一个 outlier-marker
    const markers = container.querySelectorAll('[data-testid="outlier-marker"]');
    expect(markers.length).toBe(1);
  });

  it("download CSV anchor points at the data-table endpoint", () => {
    render(<MetricsTableCard meta={makeMeta()} threadId="t1" />);
    const link = screen.getByRole("link", { name: /下载 CSV/ });
    expect(link.getAttribute("href")).toContain("/api/threads/t1/artifacts/data-table");
  });
});
