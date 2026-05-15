"use client";

import dynamic from "next/dynamic";

import { cn } from "@/lib/utils";

import { Section } from "../section";

const ProgressiveSkillsAnimation = dynamic(
  () => import("../progressive-skills-animation"),
  { ssr: false, loading: () => <div className="h-[400px] glass-card rounded-2xl" /> },
);

export function SkillsSection({ className }: { className?: string }) {
  return (
    <Section
      className={cn("h-[calc(100vh-64px)] w-full", className)}
      title="Agent Skills"
      subtitle={
        <div>
          Agent Skills are loaded progressively — only what&apos;s needed, when
          it&apos;s needed.
          <br />
          Extend DeerFlow with your own skill files, or use our built-in
          library.
        </div>
      }
    >
      <div className="relative overflow-hidden">
        <ProgressiveSkillsAnimation />
      </div>
    </Section>
  );
}
