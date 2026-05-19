"""Tests for the configurable -> user_context bridge in make_lead_agent.

Verifies that LangGraph's ``langgraph_auth_user_id`` (set by
``langgraph_api/models/run.py:253`` from the auth context) is copied into
the deerflow ``user_context`` ContextVar at the start of agent
construction. This is required because ``make_lead_agent`` runs on the
bg-loop asyncio task, while ``authenticate()`` and ``@auth.on`` run on
the request thread — ContextVars set in the latter never reach the
former.

These tests bypass the heavy ``make_lead_agent`` machinery (model
creation, tool loading) by exercising only the user_id-copy lines via
the same ``set_current_user`` / ``cfg.get`` calls.
"""

from __future__ import annotations

import pytest

from deerflow.agents.lead_agent.agent import _AuthUser
from deerflow.runtime.user_context import (
    DEFAULT_USER_ID,
    get_effective_user_id,
    reset_current_user,
    set_current_user,
)


@pytest.mark.no_auto_user
def test_auth_user_satisfies_current_user_protocol():
    """``_AuthUser`` is a duck-typed CurrentUser (has ``.id`` attribute)."""
    from deerflow.runtime.user_context import CurrentUser

    user = _AuthUser("u-abc-123")
    assert isinstance(user, CurrentUser)
    assert user.id == "u-abc-123"


@pytest.mark.no_auto_user
def test_auth_user_id_present_sets_contextvar():
    """When configurable['langgraph_auth_user_id'] is set, the
    ContextVar mirrors it and ``get_effective_user_id()`` returns it.

    Reproduces the read path used in ``make_lead_agent``.
    """
    cfg = {"langgraph_auth_user_id": "cd95effa-d595-441a-bc44-29db0f3e259d"}

    auth_user_id = cfg.get("langgraph_auth_user_id")
    assert auth_user_id  # precondition

    token = set_current_user(_AuthUser(str(auth_user_id)))
    try:
        assert get_effective_user_id() == "cd95effa-d595-441a-bc44-29db0f3e259d"
    finally:
        reset_current_user(token)


@pytest.mark.no_auto_user
def test_auth_user_id_missing_falls_back_to_default():
    """When configurable lacks ``langgraph_auth_user_id`` (e.g. unauth
    Studio request), ``make_lead_agent`` skips set_current_user, so
    ``get_effective_user_id()`` returns DEFAULT_USER_ID ("default")."""
    cfg: dict = {}

    auth_user_id = cfg.get("langgraph_auth_user_id")
    if auth_user_id:  # mirrors the make_lead_agent guard
        set_current_user(_AuthUser(str(auth_user_id)))

    assert get_effective_user_id() == DEFAULT_USER_ID


@pytest.mark.no_auto_user
def test_auth_user_id_coerced_to_str():
    """Defensive: even if a UUID slips through (unlikely — LangGraph
    stores it as str), coercion keeps the ContextVar API contract."""
    import uuid

    uid = uuid.uuid4()
    cfg = {"langgraph_auth_user_id": uid}

    auth_user_id = cfg.get("langgraph_auth_user_id")
    token = set_current_user(_AuthUser(str(auth_user_id)))
    try:
        assert get_effective_user_id() == str(uid)
    finally:
        reset_current_user(token)
