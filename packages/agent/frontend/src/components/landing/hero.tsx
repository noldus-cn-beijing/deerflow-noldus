"use client";

import { ChevronRightIcon } from "lucide-react";
import Link from "next/link";

import { Button } from "@/components/ui/button";
import { WordRotate } from "@/components/ui/word-rotate";
import { cn } from "@/lib/utils";

import { PulseGrid } from "./pulse-grid";

export function Hero({ className }: { className?: string }) {
  return (
    <div
      className={cn(
        "flex size-full flex-col items-center justify-center",
        className,
      )}
    >
      <PulseGrid />
      <div className="container-md relative z-10 mx-auto flex h-screen flex-col items-center justify-center">
        <h1 className="flex items-center gap-2 text-4xl font-bold md:text-6xl text-foreground">
          <WordRotate
            words={[
              "EthoVision",
              "行为分析",
              "统计检验",
              "APA 报告",
              "数据可视化",
              "Experiment Design",
              "Deep Research",
              "Vibe Coding",
            ]}
          />{" "}
          <div>with EthoInsight</div>
        </h1>
        <p
          className="mt-8 scale-105 text-center text-2xl"
          style={{ color: "#50615C" }}
        >
          An open-source SuperAgent harness that researches, codes, and creates.
          With
          <br />
          the help of sandboxes, memories, tools, skills and subagents, it
          handles
          <br />
          different levels of tasks that could take minutes to hours.
        </p>
        <Link href="/workspace">
          <Button className="size-lg mt-8 scale-108" size="lg" variant="brand">
            <span className="text-md">Get Started with 2.0</span>
            <ChevronRightIcon className="size-4" />
          </Button>
        </Link>
      </div>
    </div>
  );
}
