"use client";

import { DownloadIcon, XIcon } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { archiveArtifactsURL } from "@/core/artifacts/utils";
import { useI18n } from "@/core/i18n/hooks";
import { cn } from "@/lib/utils";

import type { FacetOptions, GalleryFilters } from "./use-gallery-filters";

interface GalleryFacetBarProps {
  filters: GalleryFilters;
  facetOptions: FacetOptions;
  totalCount: number;
  onFilter: (key: keyof GalleryFilters, value: string) => void;
  threadId: string;
  className?: string;
}

export function GalleryFacetBar({
  filters,
  facetOptions,
  totalCount,
  onFilter,
  threadId,
  className,
}: GalleryFacetBarProps) {
  const { t } = useI18n();
  const g = t.gallery;

  const activeFilters: { key: keyof GalleryFilters; label: string; value: string }[] = [];
  if (filters.paradigm) activeFilters.push({ key: "paradigm", label: g.filterParadigm, value: filters.paradigm });
  if (filters.chartType) activeFilters.push({ key: "chartType", label: g.filterChartType, value: filters.chartType });
  if (filters.outputMode !== "all") activeFilters.push({ key: "outputMode", label: g.filterMode, value: filters.outputMode });
  if (filters.group) activeFilters.push({ key: "group", label: g.filterGroup, value: filters.group });
  if (filters.subject) activeFilters.push({ key: "subject", label: g.filterSubject, value: filters.subject });
  if (filters.search) activeFilters.push({ key: "search", label: g.searchPlaceholder, value: filters.search });

  return (
    <div className={cn("flex flex-col gap-3", className)}>
      <div className="flex flex-wrap items-center gap-2">
        {facetOptions.paradigms.length > 0 && (
          <Select
            value={filters.paradigm || "__all__"}
            onValueChange={(v) => onFilter("paradigm", v === "__all__" ? "" : v)}
          >
            <SelectTrigger className="h-8 w-36 text-sm">
              <SelectValue placeholder={g.allParadigms} />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__all__">{g.allParadigms}</SelectItem>
              {facetOptions.paradigms.map((p) => (
                <SelectItem key={p} value={p}>{p}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}

        {facetOptions.chartTypes.length > 0 && (
          <Select
            value={filters.chartType || "__all__"}
            onValueChange={(v) => onFilter("chartType", v === "__all__" ? "" : v)}
          >
            <SelectTrigger className="h-8 w-32 text-sm">
              <SelectValue placeholder={g.allTypes} />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__all__">{g.allTypes}</SelectItem>
              {facetOptions.chartTypes.map((ct) => (
                <SelectItem key={ct} value={ct}>{ct}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}

        <Select
          value={filters.outputMode}
          onValueChange={(v) => onFilter("outputMode", v)}
        >
          <SelectTrigger className="h-8 w-28 text-sm">
            <SelectValue placeholder={g.allModes} />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">{g.allModes}</SelectItem>
            <SelectItem value="aggregate">{g.aggregate}</SelectItem>
            <SelectItem value="per_subject">{g.perSubject}</SelectItem>
          </SelectContent>
        </Select>

        {facetOptions.groups.length > 0 && (
          <Select
            value={filters.group || "__all__"}
            onValueChange={(v) => onFilter("group", v === "__all__" ? "" : v)}
          >
            <SelectTrigger className="h-8 w-32 text-sm">
              <SelectValue placeholder={g.allGroups} />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__all__">{g.allGroups}</SelectItem>
              {facetOptions.groups.map((gr) => (
                <SelectItem key={gr} value={gr}>{gr}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}

        {facetOptions.subjects.length > 0 && (
          <Select
            value={filters.subject || "__all__"}
            onValueChange={(v) => onFilter("subject", v === "__all__" ? "" : v)}
          >
            <SelectTrigger className="h-8 w-32 text-sm">
              <SelectValue placeholder={g.allSubjects} />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__all__">{g.allSubjects}</SelectItem>
              {facetOptions.subjects.map((s) => (
                <SelectItem key={s} value={s}>{s}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}

        <Input
          className="h-8 w-40 text-sm"
          placeholder={g.searchPlaceholder}
          value={filters.search}
          onChange={(e) => onFilter("search", e.target.value)}
        />

        <div className="ml-auto">
          <Button size="sm" asChild>
            <a href={archiveArtifactsURL(threadId)} aria-label={t.gallery.downloadAll(totalCount)}>
              <DownloadIcon className="size-4" />
              {t.gallery.downloadAllShort}
            </a>
          </Button>
        </div>
      </div>

      {activeFilters.length > 0 && (
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-muted-foreground text-xs">{g.nFiltered(totalCount)}</span>
          {activeFilters.map((af) => (
            <Badge
              key={af.key}
              variant="secondary"
              className="cursor-pointer gap-1"
              onClick={() => onFilter(af.key, af.key === "outputMode" ? "all" : "")}
            >
              {af.label}: {af.value}
              <XIcon className="size-3" />
            </Badge>
          ))}
          <button
            className="text-muted-foreground hover:text-foreground text-xs underline"
            onClick={() => {
              onFilter("paradigm", "");
              onFilter("chartType", "");
              onFilter("outputMode", "all");
              onFilter("group", "");
              onFilter("subject", "");
              onFilter("search", "");
            }}
          >
            {g.clearFilters}
          </button>
        </div>
      )}
    </div>
  );
}
