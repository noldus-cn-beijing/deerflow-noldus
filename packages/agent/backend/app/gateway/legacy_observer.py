"""Legacy thread 启动观测（spec 2026-06-26 §任务5）。

调研一度判 legacy 路径「无审计」，实读发现 ``user_dir`` 已有迁移 + 审计日志
（config/paths.py:181-203）。故本模块**不动路径逻辑**，仅补一条启动观测：
统计 ``threads_meta.user_id IS NULL`` 行数，有遗留则日志告警，便于运维判断
是否还有老部署（auth 引入前）的无归属 thread 待清理/认领。

设计要点：
- best-effort：观测失败/无 SQL backend（memory 模式）→ 静默跳过，绝不阻断启动。
- 不上结构门：spec §五 反向自检——结构层（路径迁移）已对，缺的只是观测，故只加日志。
- 调用点：gateway lifespan 启动阶段（app.py），persistence engine 已 init 之后。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


async def observe_legacy_threads(*, session_factory: Any) -> int | None:
    """统计并日志告警 legacy 无归属 thread 数量。返回计数（无法统计时 None）。

    Args:
        session_factory: SQLAlchemy ``async_sessionmaker``。memory backend 无 SQL
            engine → 传 None，本函数静默 no-op（观测是 best-effort，不阻断启动）。

    Returns:
        legacy 行数；``None`` 表示无 SQL backend / 观测失败（已记 debug）。
    """
    if session_factory is None:
        logger.debug("legacy_thread_observer: no SQL session factory (memory backend); skip")
        return None

    try:
        from deerflow.persistence.thread_meta.sql import ThreadMetaRepository

        repo = ThreadMetaRepository(session_factory)
        count = await repo.count_legacy_orphans()
    except Exception:
        # best-effort：观测本身决不阻断 gateway 启动。
        logger.debug("legacy_thread_observer: count failed (non-fatal)", exc_info=True)
        return None

    if count and count > 0:
        logger.warning(
            "legacy_thread_count=%d threads_meta rows have NULL user_id — legacy "
            "data from before auth was introduced. user_dir path migration is in "
            "place (paths.py); these rows are orphan metadata awaiting owner "
            "re-assignment or cleanup.",
            count,
        )
    return count
