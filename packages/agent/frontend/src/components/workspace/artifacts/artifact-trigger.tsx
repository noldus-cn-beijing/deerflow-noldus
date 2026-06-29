import { FilesIcon } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Tooltip } from "@/components/workspace/tooltip";
import { useI18n } from "@/core/i18n/hooks";

import { useArtifacts } from "./context";

export const ArtifactTrigger = () => {
  const { t } = useI18n();
  const { setOpen: setArtifactsOpen } = useArtifacts();

  // 始终渲染入口（不再据 state.artifacts.length 隐藏）。产物面板自身从磁盘端点取图/报告并
  // 处理空态——入口隐藏曾依赖 state 冒泡（恒不全），导致「画完图入口还不出现/时有时无」。
  return (
    <Tooltip content="Show artifacts of this conversation">
      <Button
        className="text-muted-foreground hover:text-foreground"
        variant="ghost"
        onClick={() => {
          setArtifactsOpen(true);
        }}
      >
        <FilesIcon />
        {t.common.artifacts}
      </Button>
    </Tooltip>
  );
};
