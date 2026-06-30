"use client";

import { PromptInputProvider } from "@/components/ai-elements/prompt-input";
import { ArtifactsProvider } from "@/components/workspace/artifacts";
import { StagesProvider } from "@/core/stages/context";
import { SubtasksProvider } from "@/core/tasks/context";

export default function AgentChatLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <SubtasksProvider>
      <StagesProvider>
        <ArtifactsProvider>
          <PromptInputProvider>{children}</PromptInputProvider>
        </ArtifactsProvider>
      </StagesProvider>
    </SubtasksProvider>
  );
}
