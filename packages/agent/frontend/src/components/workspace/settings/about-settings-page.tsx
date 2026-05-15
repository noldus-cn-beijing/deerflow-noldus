"use client";

import { Streamdown } from "streamdown";

import { aboutMarkdown } from "./about-content";

export function AboutSettingsPage() {
  return (
    <>
      <Streamdown>{aboutMarkdown}</Streamdown>
      <section className="mt-8 pt-6 border-t border-border">
        <p className="text-xs text-muted-foreground leading-relaxed">
          本产品使用 <strong className="text-foreground">OPPO Sans 4.0</strong> 字体,
          版权归广东欧加移动通信有限公司所有 (Copyright 2024 Guangdong OPPO Mobile
          Telecommunications Corp., Ltd.)。
          OPPO Sans Fonts 在 OPPO Sans Fonts License Agreement 下使用。
        </p>
      </section>
    </>
  );
}
