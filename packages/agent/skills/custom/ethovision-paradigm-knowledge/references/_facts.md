# EV19 模板事实表（62 变体）

**来源**: `demodata/ev19 templates/ + templateMetaData.xml`
**变体总数**: 62（20 大类）
**生成方式**: 从 `templateMetaData.xml` 自动抽取，请勿手工修改

## 字段说明

- `template_id`: 完整变体 ID（用作 `ev19_template` 字段值）
- `category`: 大类
- `arena_template`: EthoVision 软件中的 arena 类型
- `zone_template`: zone 配置
- `inferred_subject_hint`: 推测的动物种类
- `inferred_zone_config`: 推测的 zone 配置缩写
- `inferred_array_size`: 阵列规模（Single / 96w / 16x / Quad / 1cubicle / 4cubicles）

## 变体清单

| template_id | category | arena_template | zone_template | subject | zone_config | array_size |
|---|---|---|---|---|---|---|
| `AquariumTrack3D` | AquariumTrack3D | Aquarium for Track3D | No zone template | not specified by name | Default | Single |
| `BarnesMaze-20Holes` | BarnesMaze | Barnes maze | 20 hole zones | not specified by name | 20Holes | Single |
| `BarnesMaze-NoZones` | BarnesMaze | Barnes maze | No zone template | not specified by name | NoZones | Single |
| `Cross Maze-Fish-AllZones` | Cross Maze-Fish | Cross maze | Start box, center, arms and goal zones | fish | AllZones | Single |
| `Cross Maze-Fish-NoZones` | Cross Maze-Fish | Cross maze | No zone template | fish | NoZones | Single |
| `DanioVision DVOC 004x-96w-circ` | DanioVision DVOC 004x | DanioVision DVOC 004x, 96 round wells | No zone template | not specified by name | Default | 96w |
| `FlightChamberTrack3D` | FlightChamberTrack3D | Flight chamber for Track3D | No zone template | not specified by name | Default | Single |
| `MWM-AFewZones` | MWM | Morris water maze | Platform, quadrants | not specified by name | AFewZones | Single |
| `MWM-AllZones` | MWM | Morris water maze | Platform, quadrants, corridors, border | not specified by name | AllZones | Single |
| `MWM-NoZones` | MWM | Morris water maze | No zone template | not specified by name | NoZones | Single |
| `NoTemplate` | NoTemplate | No arena template | No zone template | not specified by name | Default | Single |
| `OpenFieldCircle-AllZones` | OpenFieldCircle | Open field, round | Border, center, quadrants | not specified by name | AllZones | Single |
| `OpenFieldCircle-NoZones-Fish` | OpenFieldCircle | Open field, round | No zone template | fish | NoZones | Single |
| `OpenFieldCircle-NoZones-Insects` | OpenFieldCircle | Open field, round | No zone template | insect | NoZones | Single |
| `OpenFieldCircle-NoZones-Rodents-Other` | OpenFieldCircle | Open field, round | No zone template | rodent | NoZones | Single |
| `OpenFieldCircle-NovObjZones` | OpenFieldCircle | Open field, round | Novel object zones | not specified by name | NovObjZones | Single |
| `OpenFieldRectangle-AllZones` | OpenFieldRectangle | Open field, square | Center, border, corners | not specified by name | AllZones | Single |
| `OpenFieldRectangle-NoZones` | OpenFieldRectangle | Open field, square | No zone template | not specified by name | NoZones | Single |
| `OpenFieldRectangle-NoZonesFishInsects` | OpenFieldRectangle | Open field, square | No zone template | fish, insect | NoZones | Single |
| `OpenFieldRectangle-NovObjZones` | OpenFieldRectangle | Open field, square | Novel object zones | not specified by name | NovObjZones | Single |
| `OpenFieldRectangle-Subdivided2x2` | OpenFieldRectangle | Open field, square | Subdivided arena, 2x2 | not specified by name | Subdivided2x2 | Single |
| `OpenFieldRectangle-Subdivided3x3` | OpenFieldRectangle | Open field, square | Subdivided arena, 3x3 | not specified by name | Subdivided3x3 | Single |
| `OpenFieldRectangle-Subdivided4x4` | OpenFieldRectangle | Open field, square | Subdivided arena, 4x4 | not specified by name | Subdivided4x4 | Single |
| `PhenoTyper-16x-AllZonesMice` | PhenoTyper | PhenoTyper, 16x | Feeding, spot light and shelter zones | mice | AllZones | 16x |
| `PhenoTyper-16x-AllZonesRatOther` | PhenoTyper | PhenoTyper, 16x | Feeding, spot light and shelter zones | rat_or_other | AllZones | 16x |
| `PhenoTyper-16x-FeedingShelterMice` | PhenoTyper | PhenoTyper, 16x | Feeding and shelter zones | mice | FeedingShelter | 16x |
| `PhenoTyper-16x-FeedingShelterRatOther` | PhenoTyper | PhenoTyper, 16x | Feeding and shelter zones | rat_or_other | FeedingShelter | 16x |
| `PhenoTyper-16x-NoZones` | PhenoTyper | PhenoTyper, 16x | No zone template | not specified by name | NoZones | 16x |
| `PhenoTyper-AllZonesMice` | PhenoTyper | PhenoTyper | Feeding, spot light and shelter zones | mice | AllZones | Single |
| `PhenoTyper-AllZonesRatOther` | PhenoTyper | PhenoTyper | Feeding, spot light and shelter zones | rat_or_other | AllZones | Single |
| `PhenoTyper-FeedingShelterMice` | PhenoTyper | PhenoTyper | Feeding and shelter zones | mice | FeedingShelter | Single |
| `PhenoTyper-FeedingShelterRatOther` | PhenoTyper | PhenoTyper | Feeding and shelter zones | rat_or_other | FeedingShelter | Single |
| `PhenoTyper-NoZones` | PhenoTyper | PhenoTyper | No zone template | not specified by name | NoZones | Single |
| `PhenoTyper-Quad-AllZonesMice` | PhenoTyper | PhenoTyper, 4x | Feeding, spot light and shelter zones | mice | AllZones | Quad |
| `PhenoTyper-Quad-AllZonesRatOther` | PhenoTyper | PhenoTyper, 4x | Feeding, spot light and shelter zones | rat_or_other | AllZones | Quad |
| `PhenoTyper-Quad-FeedingShelterMice` | PhenoTyper | PhenoTyper, 4x | Feeding and shelter zones | mice | FeedingShelter | Quad |
| `PhenoTyper-Quad-FeedingShelterRatOther` | PhenoTyper | PhenoTyper, 4x | Feeding and shelter zones | rat_or_other | FeedingShelter | Quad |
| `PhenoTyper-Quad-NoZones` | PhenoTyper | PhenoTyper, 4x | No zone template | not specified by name | NoZones | Quad |
| `PlusMaze-AllZones` | PlusMaze | Elevated plus maze | Closed-, open arms, head dip zone | not specified by name | AllZones | Single |
| `PlusMaze-FewZones` | PlusMaze | Elevated plus maze | Closed and open arms | not specified by name | FewZones | Single |
| `PlusMaze-NoZones` | PlusMaze | Elevated plus maze | No zone template | not specified by name | NoZones | Single |
| `PorsoltCylinder-AllZones` | PorsoltCylinder | Porsolt cylinder | Diving zone | not specified by name | AllZones | Single |
| `PorsoltCylinder-NoZones` | PorsoltCylinder | Porsolt cylinder | No zone template | not specified by name | NoZones | Single |
| `Radial-8-arm-AllZones` | Radial-8-arm | Radial 8-arm maze | Center, arm and goal zones | not specified by name | AllZones | Single |
| `Radial-8-arm-NoZones` | Radial-8-arm | Radial 8-arm maze | No zone template | not specified by name | NoZones | Single |
| `Sociability-AllZones` | Sociability | Sociability chamber | Chamber and cage zones | not specified by name | AllZones | Single |
| `Sociability-NoZones` | Sociability | Sociability chamber | No zone template | not specified by name | NoZones | Single |
| `T-Maze-Fish-AllZones` | T-Maze | T-maze | Start box, center, arms and goal zones | fish | AllZones | Single |
| `T-Maze-Fish-NoZones` | T-Maze | T-maze | No zone template | fish | NoZones | Single |
| `T-Maze-Rodents-Other-AllZones` | T-Maze | T-maze | Start box, center, arms and goal zones | rodent | AllZones | Single |
| `T-Maze-Rodents-Other-NoZones` | T-Maze | T-maze | No zone template | rodent | NoZones | Single |
| `UgoBasileActiveAvoidance` | UgoBasileActiveAvoidance | Ugo Basile Active Avoidance | Left and Right Zones | not specified by name | Default | Single |
| `UgoBasileFCS-1cubicle` | UgoBasileFCS | Ugo Basile FCS 1 cubicle | No zone template | not specified by name | Default | 1cubicle |
| `UgoBasileFCS-4cubicles` | UgoBasileFCS | Ugo Basile FCS 4 cubicles | No zone template | not specified by name | Default | 4cubicles |
| `WellPlate-Circle-AllZones` | WellPlate | Well plate, round wells | Border, center zone | not specified by name | AllZones | Single |
| `WellPlate-Circle-NoZones` | WellPlate | Well plate, round wells | No zone template | not specified by name | NoZones | Single |
| `WellPlate-Rectangle-AllZones` | WellPlate | Well plate, square wells | Border, center zone | not specified by name | AllZones | Single |
| `WellPlate-Rectangle-NoZones` | WellPlate | Well plate, square wells | No zone template | not specified by name | NoZones | Single |
| `Y-Maze-AllZones` | Y-Maze | Y-maze | Center and arm zones | not specified by name | AllZones | Single |
| `Y-Maze-NoZones` | Y-Maze | Y-maze | No zone template | not specified by name | NoZones | Single |
| `ZeroMaze-AllZones` | ZeroMaze | O-maze | Closed and open arms | not specified by name | AllZones | Single |
| `ZeroMaze-NoZones` | ZeroMaze | O-maze | No zone template | not specified by name | NoZones | Single |
