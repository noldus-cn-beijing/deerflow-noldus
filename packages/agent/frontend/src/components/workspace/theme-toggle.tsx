"use client";

import { MoonIcon, SunIcon } from "lucide-react";
import { useTheme } from "next-themes";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { useI18n } from "@/core/i18n/hooks";

export function ThemeToggle({ className }: { className?: string }) {
  const { resolvedTheme, setTheme } = useTheme();
  const { t } = useI18n();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // 防 hydration 闪烁：未 mount 时渲染占位（与现有 workspace-nav-menu 的 mounted 模式一致）
  if (!mounted) {
    return (
      <Button variant="ghost" size="icon" className={className} aria-hidden tabIndex={-1} />
    );
  }

  const isDark = resolvedTheme === "dark";
  return (
    <Button
      type="button"
      variant="ghost"
      size="icon"
      className={className}
      aria-label={t.workspace.toggleTheme}
      title={t.workspace.toggleTheme}
      onClick={() => setTheme(isDark ? "light" : "dark")}
    >
      {isDark ? <SunIcon className="size-4" /> : <MoonIcon className="size-4" />}
    </Button>
  );
}
