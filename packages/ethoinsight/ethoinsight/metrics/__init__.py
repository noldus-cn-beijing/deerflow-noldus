"""按学术范式组织的指标函数子模块。

物理路径:
  - _common.py: 范式无关（distance / velocity / zone 列查找 / csv 导出）
  - shoaling.py: 群体游动指标
  - oft.py: Open Field 指标
  - epm.py: 高架十字迷宫指标
  - zero_maze.py: Zero Maze 环形迷宫指标
  - dispatcher.py: compute_paradigm_metrics() 派发入口

代码外部仍可 `from ethoinsight.metrics import compute_open_arm_time` 等历史导入路径
（通过 metrics/__init__.py 重导出实现）。新代码推荐显式导入：
  from ethoinsight.metrics.epm import compute_open_arm_time
"""

from ethoinsight.metrics._common import (
    compute_distance_moved,
    compute_velocity_stats,
    save_to_csv,
)
from ethoinsight.metrics.shoaling import (
    compute_inter_individual_distance,
    compute_nearest_neighbor_distance,
    compute_group_polarity,
)
from ethoinsight.metrics.oft import (
    compute_center_time_ratio,
    compute_thigmotaxis_index,
    compute_center_distance_ratio,
    compute_center_entry_count,
)
from ethoinsight.metrics.epm import (
    compute_open_arm_time_ratio,
    compute_open_arm_entry_count,
    compute_open_arm_entry_ratio,
    compute_open_arm_time,
    compute_total_entry_count,
)
from ethoinsight.metrics.zero_maze import (
    compute_open_zone_time_ratio,
    compute_open_zone_time,
    compute_open_zone_distance,
    compute_hesitation_count,
)
from ethoinsight.metrics.ldb import (
    compute_light_time_ratio,
    compute_transition_count,
    compute_light_latency,
)
from ethoinsight.metrics.dispatcher import compute_paradigm_metrics

__all__ = [
    "compute_distance_moved",
    "compute_velocity_stats",
    "save_to_csv",
    "compute_inter_individual_distance",
    "compute_nearest_neighbor_distance",
    "compute_group_polarity",
    "compute_center_time_ratio",
    "compute_thigmotaxis_index",
    "compute_center_distance_ratio",
    "compute_center_entry_count",
    "compute_open_arm_time_ratio",
    "compute_open_arm_entry_count",
    "compute_open_arm_entry_ratio",
    "compute_open_arm_time",
    "compute_total_entry_count",
    "compute_open_zone_time_ratio",
    "compute_open_zone_time",
    "compute_open_zone_distance",
    "compute_hesitation_count",
    "compute_light_time_ratio",
    "compute_transition_count",
    "compute_light_latency",
    "compute_paradigm_metrics",
]
