# Noldus 新视觉方案总结

> 来源：[Noldus Branding Assets](https://array-serifs-54382319.figma.site/)

---

## 1. Logo

Noldus 新视觉方案提供了两个版本的 Logo：

| 版本 | 说明 |
|------|------|
| **深色版** | 用于浅色背景 |
| **白色版** | 用于深色/彩色背景 |

Logo 下载地址（SharePoint）：[Go to download page](https://nolduscorp-my.sharepoint.com/:f:/g/personal/martijn_verhoog_noldus_com/IgDoe0X8RQHzTooIv6ZSBBqNASacL9OQZM6VVn7Z_Z62CkU?e=kE8s3G)

---

## 2. 色彩体系

### 2.1 主色板（Main Colors）

色彩调色板在专业感与活力之间取得平衡。深沉稳重的色调传达可靠性与专业感，明亮的强调色则引入能量与动感。整体色彩系统兼顾了 Noldus 核心的技术精确性与有机属性。

| 色彩名称 | 色块预览 | HEX | RGB | CMYK | RAL |
|----------|---------|-----|-----|------|-----|
| **Black** | `#282828` | #282828 | 40, 40, 40 | 0, 0, 0, 84.31 | RAL 9004 |
| **Forest Green** | `#20564E` | #20564E | 32, 86, 78 | 63, 0, 9, 66 | RAL 6016 |
| **Lime Green** | `#10DD8B` | #10DD8B | 16, 221, 139 | 93, 0, 38, 13 | — |
| **White** | `#FFFFFF` | #FFFFFF | 255, 255, 255 | 0, 0, 0, 0 | RAL 9003 |

### 2.2 主题配色（Themes）

色彩可以组合为主题色板，以支持不同的品牌语境。每个主题通过一致的方式使用主色、辅色和强调色，传达预期的氛围或焦点，同时保持整体品牌识别的一致性。

#### 主题 1：White（白色主题）

| 元素 | 配色 |
|------|------|
| 背景 | White `#FFFFFF` |
| 边框 | Black `#282828` |
| 文字 | Black `#282828` |
| 主按钮 | Lime Green 底 + Black 字，hover `#0EC97D` |
| 辅按钮 | Forest Green 底 + Lime Green 字，hover `#1A4840` |

#### 主题 2：Forest Green（森林绿主题）

| 元素 | 配色 |
|------|------|
| 背景 | Forest Green `#20564E` |
| 边框 | Black `#282828` |
| 文字 | White `#FFFFFF` |
| 主按钮 | Lime Green 底 + Black 字，hover `#0EC97D` |
| 辅按钮 | White 底 + Black 字，hover gray-100 |

#### 主题 3：Lime Green（青绿主题）

| 元素 | 配色 |
|------|------|
| 背景 | Lime Green `#10DD8B` |
| 边框 | Black `#282828` |
| 文字 | Black `#282828` |
| 主按钮 | White 底 + Black 字，hover gray-100 |
| 辅按钮 | Forest Green 底 + Lime Green 字，hover `#1A4840` |

#### 主题 4：Black（黑色主题）

| 元素 | 配色 |
|------|------|
| 背景 | Black `#282828` |
| 边框 | Black `#282828` |
| 文字 | White `#FFFFFF` |
| 主按钮 | Lime Green 底 + Black 字，hover `#0EC97D` |
| 辅按钮 | Forest Green 底 + Lime Green 字，hover `#1A4840` |

### 2.3 按钮规范

| 按钮类型 | 背景色 | 文字色 | Hover 背景色 | 圆角 |
|----------|-------|-------|-------------|------|
| Primary（主按钮） | `#10DD8B` | `#282828` | `#0EC97D` | `border-radius: 1rem` (rounded-2xl) |
| Secondary（辅按钮-A） | `#20564E` | `#10DD8B` | `#1A4840` | `border-radius: 1rem` |
| Secondary（辅按钮-B） | `#FFFFFF` | `#282828` | `gray-100` | `border-radius: 1rem` |

按钮内边距：`px-6 py-4`（水平 24px，垂直 16px）

### 2.4 卡片规范

- 圆角：`border-radius: 40px`
- 内边距：`p-8 md:p-10`（32px，中屏以上 40px）
- 高度：`h-[280px]`（桌面端）

---

## 3. 字体排版

### 3.1 字体选择

**Inter** —— 主字体，因其清晰性、通用性和现代感而被选用。适用于数字媒体和印刷品，为品牌视觉识别提供干净、易读的结构。应统一用于标题、正文和 UI 元素。

Google Fonts 地址：[Inter](https://fonts.google.com/specimen/Inter)

### 3.2 字体层级（Type Scale）

#### Display 层级（大标题）

| 层级 | 字重 | 字号 | 行高 |
|------|------|------|------|
| Display Large | Semi Bold (600) | 4.072 rem (~65.15px) | 100% |
| Display Medium | Semi Bold (600) | 3.217 rem (~51.47px) | 100% |
| Display Small | Semi Bold (600) | 2.572 rem (~41.15px) | 100% |

#### Headline 层级（标题）

| 层级 | 字重 | 字号 | 行高 |
|------|------|------|------|
| Headline Large | Semi Bold (600) | 2.036 rem (~32.58px) | 100% |
| Headline Medium | Semi Bold (600) | 1.619 rem (~25.90px) | 100% |
| Headline Small | Semi Bold (600) | 1.302 rem (~20.83px) | 100% |

#### Title 层级（小标题）

| 层级 | 字重 | 字号 | 行高 |
|------|------|------|------|
| Title Large | Medium (500) | 1.302 rem (~20.83px) | 140% |
| Title Medium | Medium (500) | 1.016 rem (~16.26px) | 140% |
| Title Small | Medium (500) | 0.794 rem (~12.70px) | 140% |

#### Label 层级（标签）

| 层级 | 字重 | 字号 | 行高 |
|------|------|------|------|
| Label Large | Medium (500) | 1.016 rem (~16.26px) | 140% |
| Label Medium | Medium (500) | 0.857 rem (~13.71px) | 140% |
| Label Small | Medium (500) | 0.698 rem (~11.17px) | 140% |

#### Body 层级（正文）

| 层级 | 字重 | 字号 | 行高 |
|------|------|------|------|
| Body Large | Regular (400) | 1.016 rem (~16.26px) | 150% |
| Body Medium | Regular (400) | 0.857 rem (~13.71px) | 150% |
| Body Small | Regular (400) | 0.698 rem (~11.17px) | 150% |

### 3.3 字重使用规则

| 字重 | 用途 |
|------|------|
| **Semi Bold (600)** | Display、Headline 层级 |
| **Medium (500)** | Title、Label 层级 |
| **Regular (400)** | Body 正文层级 |

---

## 4. 设计原则总结

1. **层次分明**：通过字重（Semi Bold → Medium → Regular）和字号（4.072rem → 0.698rem）建立清晰的视觉层级
2. **配色克制**：以四色（Black、Forest Green、Lime Green、White）构建灵活的主题系统
3. **圆润友好**：大圆角卡片（40px）和按钮（16px）传递亲和感
4. **双色系对比**：深沉色调（Black、Forest Green）负责稳重与专业，明亮色调（Lime Green）注入活力
5. **统一字体**：全场景统一使用 Inter，通过字重和字号变化创造节奏

---

*文档生成日期：2026-05-13*
