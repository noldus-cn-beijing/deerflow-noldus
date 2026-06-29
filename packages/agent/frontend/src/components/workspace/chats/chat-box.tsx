import { XIcon } from "lucide-react";
import { usePathname } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import type { GroupImperativeHandle } from "react-resizable-panels";

import { Button } from "@/components/ui/button";
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
} from "@/components/ui/resizable";
import { normalizeArtifact, normalizeArtifacts } from "@/core/artifacts/types";
import { env } from "@/env";
import { cn } from "@/lib/utils";

import { ThreadAssetsPanel, useArtifacts } from "../artifacts";
import { useThread } from "../messages/context";

const CLOSE_MODE = { chat: 100, artifacts: 0 };
const OPEN_MODE = { chat: 60, artifacts: 40 };

const ChatBox: React.FC<{ children: React.ReactNode; threadId: string }> = ({
  children,
  threadId,
}) => {
  const { thread } = useThread();
  const pathname = usePathname();
  const threadIdRef = useRef(threadId);
  const layoutRef = useRef<GroupImperativeHandle>(null);

  const {
    artifacts,
    open: artifactsOpen,
    setOpen: setArtifactsOpen,
    setArtifacts,
    select: selectArtifact,
    deselect,
  } = useArtifacts();

  // 资产面板 refetch 信号：run 完成（thread.isLoading true→false）后递增 → ThreadAssetsPanel
  // 重拉磁盘端点补全量产物（图/报告陆续落盘）。与 streaming 解耦，只在 run 边界触发一次。
  const wasLoadingRef = useRef(false);
  const [assetsRefetchSignal, setAssetsRefetchSignal] = useState(0);
  useEffect(() => {
    if (wasLoadingRef.current && !thread.isLoading) {
      setAssetsRefetchSignal((n) => n + 1);
    }
    wasLoadingRef.current = thread.isLoading;
  }, [thread.isLoading]);

  const [autoSelectFirstArtifact, setAutoSelectFirstArtifact] = useState(true);
  useEffect(() => {
    if (threadIdRef.current !== threadId) {
      threadIdRef.current = threadId;
      deselect();
    }

    // Update artifacts from the current thread（normalize 兜底裸 string）
    setArtifacts(normalizeArtifacts(thread.values.artifacts));

    // DO NOT automatically deselect the artifact when switching threads, because the artifacts auto discovering is not work now.
    // if (
    //   selectedArtifact &&
    //   !thread.values.artifacts?.includes(selectedArtifact)
    // ) {
    //   deselect();
    // }

    if (
      env.NEXT_PUBLIC_STATIC_WEBSITE_ONLY === "true" &&
      autoSelectFirstArtifact
    ) {
      if (thread?.values?.artifacts?.length > 0) {
        setAutoSelectFirstArtifact(false);
        selectArtifact(normalizeArtifact(thread.values.artifacts[0]!).path);
      }
    }
  }, [
    threadId,
    autoSelectFirstArtifact,
    deselect,
    selectArtifact,
    setArtifacts,
    thread.values.artifacts,
  ]);

  const artifactPanelOpen = useMemo(() => {
    if (env.NEXT_PUBLIC_STATIC_WEBSITE_ONLY === "true") {
      return artifactsOpen && artifacts?.length > 0;
    }
    return artifactsOpen;
  }, [artifactsOpen, artifacts]);

  const resizableIdBase = useMemo(() => {
    return pathname.replace(/[^a-zA-Z0-9_-]+/g, "-").replace(/^-+|-+$/g, "");
  }, [pathname]);

  useEffect(() => {
    if (layoutRef.current) {
      if (artifactPanelOpen) {
        layoutRef.current.setLayout(OPEN_MODE);
      } else {
        layoutRef.current.setLayout(CLOSE_MODE);
      }
    }
  }, [artifactPanelOpen]);

  return (
    <ResizablePanelGroup
      id={`${resizableIdBase}-panels`}
      orientation="horizontal"
      defaultLayout={{ chat: 100, artifacts: 0 }}
      groupRef={layoutRef}
    >
      {/* #227 把 chat-root 从 size-full 改成 flex-1 min-h-0 以支持内部滚动，但 flex-1 只在
          flex 父容器内生效——react-resizable-panels 的 Panel 默认是普通 block（仅做横向
          flex 尺寸），不是 flex 容器，导致 chat-root 高度塌成内容高（198px）、输入框被
          translate 推出视口、对话流空白。这里给 Panel 补 flex flex-col，让 chat-root 的
          flex-1 解析到 Panel 全高（= group stretch 得到的 SidebarInset 全高）。chats + agents
          两路由共用本修复。 */}
      <ResizablePanel
        className="relative flex flex-col"
        defaultSize={100}
        id="chat"
      >
        {children}
      </ResizablePanel>
      <ResizableHandle
        id={`${resizableIdBase}-separator`}
        className={cn(
          "opacity-33 hover:opacity-100",
          !artifactPanelOpen && "pointer-events-none opacity-0",
        )}
      />
      <ResizablePanel
        className={cn(
          "transition-opacity duration-slow ease-brand-in-out",
          !artifactsOpen && "opacity-0",
        )}
        id="artifacts"
      >
        <div
          className={cn(
            "h-full transition-transform duration-slow ease-brand-in-out",
            artifactPanelOpen ? "translate-x-0" : "translate-x-full",
          )}
        >
          <div className="relative size-full">
            <div className="absolute top-2 right-2 z-30">
              <Button
                size="icon-sm"
                variant="ghost"
                onClick={() => {
                  setArtifactsOpen(false);
                }}
              >
                <XIcon />
              </Button>
            </div>
            {/* thread 级资产面板：图 + 报告全部从磁盘端点取，与 streaming 解耦（稳定不漂移）。
                run 完成（isLoading true→false）后递增 refetchSignal 补拉全量。 */}
            <ThreadAssetsPanel
              threadId={threadId}
              chartsStatus={thread.values.charts_status}
              refetchSignal={assetsRefetchSignal}
            />
          </div>
        </div>
      </ResizablePanel>
    </ResizablePanelGroup>
  );
};

export { ChatBox };
