from __future__ import annotations

import asyncio

import pytest
from langchain.agents.middleware import AgentMiddleware
from langchain.tools import ToolRuntime
from langgraph.runtime import Runtime

from deerflow.sandbox.middleware import SandboxMiddleware
from deerflow.sandbox.sandbox import Sandbox
from deerflow.sandbox.sandbox_provider import SandboxProvider, reset_sandbox_provider, set_sandbox_provider
from deerflow.sandbox.search import GrepMatch
from deerflow.sandbox.tools import ls_tool


class _SyncProvider(SandboxProvider):
    def __init__(self) -> None:
        self.thread_ids: list[str | None] = []
        self.user_ids: list[str | None] = []

    def acquire(self, thread_id: str | None = None, *, user_id: str | None = None) -> str:
        self.thread_ids.append(thread_id)
        self.user_ids.append(user_id)
        return "sync-sandbox"

    def get(self, sandbox_id: str) -> Sandbox | None:
        return None

    def release(self, sandbox_id: str) -> None:
        return None


class _SandboxStub(Sandbox):
    def execute_command(self, command: str) -> str:
        return "OK"

    def read_file(self, path: str) -> str:
        return "content"

    def download_file(self, path: str) -> bytes:
        return b"content"

    def list_dir(self, path: str, max_depth: int = 2) -> list[str]:
        return ["/mnt/user-data/workspace/file.txt"]

    def write_file(self, path: str, content: str, append: bool = False) -> None:
        return None

    def glob(self, path: str, pattern: str, *, include_dirs: bool = False, max_results: int = 200) -> tuple[list[str], bool]:
        return [], False

    def grep(
        self,
        path: str,
        pattern: str,
        *,
        glob: str | None = None,
        literal: bool = False,
        case_sensitive: bool = False,
        max_results: int = 100,
    ) -> tuple[list[GrepMatch], bool]:
        return [], False

    def update_file(self, path: str, content: bytes) -> None:
        return None


class _AsyncOnlyProvider(SandboxProvider):
    def __init__(self) -> None:
        self.thread_ids: list[str | None] = []
        self.user_ids: list[str | None] = []
        self.released_ids: list[str] = []
        self.sandbox = _SandboxStub("async-sandbox")

    def acquire(self, thread_id: str | None = None, *, user_id: str | None = None) -> str:
        del user_id
        raise AssertionError("async middleware should not call sync acquire")

    async def acquire_async(self, thread_id: str | None = None, *, user_id: str | None = None) -> str:
        self.thread_ids.append(thread_id)
        self.user_ids.append(user_id)
        return "async-sandbox"

    def get(self, sandbox_id: str) -> Sandbox | None:
        if sandbox_id == "async-sandbox":
            return self.sandbox
        return None

    def release(self, sandbox_id: str) -> None:
        self.released_ids.append(sandbox_id)
        return None


@pytest.mark.anyio
async def test_provider_default_acquire_async_offloads_sync_acquire(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = _SyncProvider()
    calls: list[tuple[object, tuple[object, ...]]] = []

    async def fake_to_thread(func, /, *args, **kwargs):
        calls.append((func, args, kwargs))
        return func(*args, **kwargs)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)

    sandbox_id = await provider.acquire_async("thread-1")

    assert sandbox_id == "sync-sandbox"
    assert provider.thread_ids == ["thread-1"]
    assert provider.user_ids == [None]
    assert calls == [(provider.acquire, ("thread-1",), {"user_id": None})]


@pytest.mark.anyio
async def test_abefore_agent_uses_async_provider_acquire() -> None:
    provider = _AsyncOnlyProvider()
    set_sandbox_provider(provider)
    try:
        middleware = SandboxMiddleware(lazy_init=False)

        result = await middleware.abefore_agent({}, Runtime(context={"thread_id": "thread-2"}))
    finally:
        reset_sandbox_provider()

    assert result == {"sandbox": {"sandbox_id": "async-sandbox"}}
    assert provider.thread_ids == ["thread-2"]


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("middleware", "state", "runtime"),
    [
        (SandboxMiddleware(lazy_init=True), {}, Runtime(context={"thread_id": "thread-lazy"})),
        (SandboxMiddleware(lazy_init=False), {}, Runtime(context={})),
        (SandboxMiddleware(lazy_init=False), {"sandbox": {"sandbox_id": "existing"}}, Runtime(context={"thread_id": "thread-existing"})),
    ],
)
async def test_abefore_agent_delegates_to_super_when_not_acquiring(
    monkeypatch: pytest.MonkeyPatch,
    middleware: SandboxMiddleware,
    state: dict,
    runtime: Runtime,
) -> None:
    calls: list[tuple[dict, Runtime]] = []

    async def fake_super_abefore_agent(self, state_arg, runtime_arg):
        calls.append((state_arg, runtime_arg))
        return {"delegated": True}

    monkeypatch.setattr(AgentMiddleware, "abefore_agent", fake_super_abefore_agent)

    result = await middleware.abefore_agent(state, runtime)

    assert result == {"delegated": True}
    assert calls == [(state, runtime)]


@pytest.mark.anyio
async def test_default_lazy_tool_acquisition_uses_async_provider() -> None:
    provider = _AsyncOnlyProvider()
    set_sandbox_provider(provider)
    try:
        runtime = ToolRuntime(
            state={},
            context={"thread_id": "thread-lazy"},
            config={"configurable": {}},
            stream_writer=lambda _: None,
            tools=[],
            tool_call_id="call-1",
            store=None,
        )

        result = await ls_tool.ainvoke({"runtime": runtime, "description": "list workspace", "path": "/mnt/user-data/workspace"})
    finally:
        reset_sandbox_provider()

    assert result == "/mnt/user-data/workspace/file.txt"
    assert provider.thread_ids == ["thread-lazy"]
    assert runtime.state["sandbox"] == {"sandbox_id": "async-sandbox"}
    assert runtime.context["sandbox_id"] == "async-sandbox"


@pytest.mark.anyio
@pytest.mark.parametrize(
    ("state", "runtime", "expected_sandbox_id"),
    [
        ({"sandbox": {"sandbox_id": "state-sandbox"}}, Runtime(context={}), "state-sandbox"),
        ({}, Runtime(context={"sandbox_id": "context-sandbox"}), "context-sandbox"),
    ],
)
async def test_aafter_agent_releases_sandbox_off_thread(
    monkeypatch: pytest.MonkeyPatch,
    state: dict,
    runtime: Runtime,
    expected_sandbox_id: str,
) -> None:
    provider = _AsyncOnlyProvider()
    to_thread_calls: list[tuple[object, tuple[object, ...]]] = []

    async def fake_to_thread(func, /, *args):
        to_thread_calls.append((func, args))
        return func(*args)

    monkeypatch.setattr(asyncio, "to_thread", fake_to_thread)
    set_sandbox_provider(provider)
    try:
        result = await SandboxMiddleware().aafter_agent(state, runtime)
    finally:
        reset_sandbox_provider()

    assert result is None
    assert provider.released_ids == [expected_sandbox_id]
    assert to_thread_calls == [(provider.release, (expected_sandbox_id,))]


@pytest.mark.anyio
async def test_aafter_agent_delegates_to_super_when_no_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[dict, Runtime]] = []

    async def fake_super_aafter_agent(self, state_arg, runtime_arg):
        calls.append((state_arg, runtime_arg))
        return {"delegated": True}

    monkeypatch.setattr(AgentMiddleware, "aafter_agent", fake_super_aafter_agent)

    state = {}
    runtime = Runtime(context={})
    result = await SandboxMiddleware().aafter_agent(state, runtime)

    assert result == {"delegated": True}
    assert calls == [(state, runtime)]


# ============================================================================
# 根因回归（2026-06-18 第四轮 EPM dogfood 并行派遣崩溃）：
# SandboxMiddlewareState.sandbox 必须带 merge_sandbox reducer，使编译出的 sandbox
# channel 恒为 BinaryOperatorAggregate，不依赖"靠链里别的中间件（SandboxAuditMiddleware
# 的 ThreadState）碰巧救回"。否则并行派遣两个 subagent 时某 super-step 两个 sandbox 写入
# 落到 LastValue channel → InvalidUpdateError: "At key 'sandbox': Can receive only one
# value per step"。
# 红→绿实证（git stash 验证）：修复前 SandboxMiddlewareState-only → LastValue；修复后
# → BinaryOperatorAggregate。
# ============================================================================


def test_sandbox_middleware_state_sandbox_has_reducer():
    """SandboxMiddlewareState.sandbox 必须挂 merge_sandbox（不是裸 LastValue 声明）。

    这是本根因修复的**确定性守卫**：注解直接带 merge_sandbox（pre-fix 无、post-fix 有，
    git stash 已证 channel 随之 LastValue→BinaryOperatorAggregate），order-independent、
    不受 create_agent channel 推断在全量测试里被前序状态污染的影响。

    （注：未给本修复加"并发 astream 跑通"的行为测试——简化复现还原不出生产崩溃所需的
    "lead tool 节点 fan-out 两 subagent、同 super-step 两 sandbox 写"条件，pre-fix 也
    不炸，故非真 red→green 守卫；channel-type 字符串断言又会被全量污染。结构断言是唯一
    可靠的回归锚点。）
    """
    import typing

    from deerflow.agents.thread_state import merge_sandbox
    from deerflow.sandbox.middleware import SandboxMiddlewareState

    hints = typing.get_type_hints(SandboxMiddlewareState, include_extras=True)
    sb = hints["sandbox"]
    assert hasattr(sb, "__metadata__"), "sandbox 必须是 Annotated（带 reducer），不能裸 NotRequired"
    assert merge_sandbox in sb.__metadata__, "sandbox 的 reducer 必须是 merge_sandbox（与 ThreadState 一致）"
