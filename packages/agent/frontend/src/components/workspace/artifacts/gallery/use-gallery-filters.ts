"use client";

import { useMemo, useState } from "react";

import type { ArtifactMeta } from "@/core/artifacts/types";

export interface GalleryFilters {
  paradigm: string;
  chartType: string;
  outputMode: "all" | "aggregate" | "per_subject";
  group: string;
  subject: string;
  search: string;
}

export interface FacetOptions {
  paradigms: string[];
  chartTypes: string[];
  groups: string[];
  subjects: string[];
}

function buildFacetOptions(metas: ArtifactMeta[]): FacetOptions {
  const paradigms = [...new Set(metas.map((m) => m.paradigm ?? "").filter(Boolean))].sort();
  const chartTypes = [...new Set(metas.map((m) => m.chart_type ?? "").filter(Boolean))].sort();
  const groups = [...new Set(metas.map((m) => m.group ?? "").filter(Boolean))].sort();
  const subjects = [...new Set(metas.map((m) => m.subject ?? "").filter(Boolean))].sort();
  return { paradigms, chartTypes, groups, subjects };
}

function applyFilters(metas: ArtifactMeta[], filters: GalleryFilters): ArtifactMeta[] {
  return metas.filter((m) => {
    if (filters.paradigm && m.paradigm !== filters.paradigm) return false;
    if (filters.chartType && m.chart_type !== filters.chartType) return false;
    if (filters.outputMode !== "all") {
      if (filters.outputMode === "aggregate" && m.output_mode !== "aggregate") return false;
      if (filters.outputMode === "per_subject" && m.output_mode !== "per_subject") return false;
    }
    if (filters.group && m.group !== filters.group) return false;
    if (filters.subject && m.subject !== filters.subject) return false;
    if (filters.search) {
      const needle = filters.search.toLowerCase();
      const haystack = [m.path, m.chart_id, m.metric, m.subject, m.paradigm]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      if (!haystack.includes(needle)) return false;
    }
    return true;
  });
}

const DEFAULT_FILTERS: GalleryFilters = {
  paradigm: "",
  chartType: "",
  outputMode: "all",
  group: "",
  subject: "",
  search: "",
};

export function useGalleryFilters(metas: ArtifactMeta[]) {
  const [filters, setFilters] = useState<GalleryFilters>(DEFAULT_FILTERS);

  const facetOptions = useMemo(() => buildFacetOptions(metas), [metas]);
  const filteredMetas = useMemo(() => applyFilters(metas, filters), [metas, filters]);

  function setFilter(key: keyof GalleryFilters, value: string) {
    setFilters((prev) => ({ ...prev, [key]: value }));
  }

  return { filters, setFilter, filteredMetas, facetOptions };
}
