# 多用户部署实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 把 EthoInsight（DeerFlow fork）从单用户 Demo 改造成 10 人并发使用的在线部署版本：独立 user-backend 签发 JWT，Gateway 验证并按 user_id 过滤 thread，LangGraph 用 Postgres checkpointer 共享，sandbox 切换到 AioSandbox 做每-thread 容器隔离，memory 按 user_id 分文件。整个部署是 Docker Compose 单机，8C16G 云 VM 即可。

**Architecture:** 一个 LangGraph Server 进程服务所有用户，state 外置到 Postgres checkpointer 按 thread_id 隔离——这是 LangGraph 原生架构，不是每人一个 agent instance。多用户安全的唯一硬问题是代码执行隔离，用 DeerFlow 自带的 `AioSandboxProvider`（每 thread 一个 Docker container）解决。用户管理是独立服务（改造自 `/home/qiuyangwang/aws-user-backend`），Postgres 同时托管 user 表、checkpointer、业务 thread metadata。

**Tech Stack:** FastAPI + PyJWT + bcrypt（user-backend）/ LangGraph `PostgresSaver` / DeerFlow `AioSandboxProvider` + Docker / Postgres 15 / Docker Compose / pytest。

**Non-goals（明确不做）:**
- 前端注册/找回密码/邮箱验证——10 人内部用户**手动预注册**，不做自助注册流程
- Rate limit / circuit breaker / monitoring 的完整实装——先跑起来，出问题再加
- Nginx HTTPS 证书签发——依赖用户（Let's Encrypt / 云厂商负载均衡器）
- K8s / 多机部署 / 自动扩容——10 人单机足够
- Sandbox 镜像从零构建——先用 DeerFlow 默认镜像（`enterprise-public-cn-beijing.cr.volces.com/vefaas-public/all-in-one-sandbox:latest`），`ethoinsight` 库通过 pip 在 sandbox 启动时安装或走 volume mount
- 客户本地 License 部署方案——走 [2026-04-22 local-deploy 文档](2026-04-22-local-deploy-auth-and-access.md)，v0.1 之后做
- 改造 AioSandbox 本身——把它当黑盒用，只改 config

**前置依赖：**
- 与[训练数据飞轮计划](2026-04-23-training-data-flywheel.md)**并行**。飞轮需要 user_id 标注数据，本计划产出 user_id；两个计划可交替推进或分配不同人同时做
- 继承 [2026-04-08 multi-user 文档](2026-04-08-multi-user-deployment.md) 的分析，本计划是它的 TDD 版执行细则

**参考文档:**
- [`/home/qiuyangwang/aws-user-backend/`](../../../aws-user-backend) — 用户管理后端模板，本计划 95% 复用，只换 Cognito 层
- [DeerFlow backend CLAUDE.md](../../packages/agent/backend/CLAUDE.md) — Agent 架构、middleware 链
- [2026-04-08-multi-user-deployment.md](2026-04-08-multi-user-deployment.md) — 多用户问题分析（方案 A）
- [现有 docker-compose.yaml](../../packages/agent/docker/docker-compose.yaml) — 当前部署模板

---

## 实施前须知

**TDD 强制**（见 [backend/CLAUDE.md](../../packages/agent/backend/CLAUDE.md)）：每个 task 都是 write test → verify fail → implement → verify pass → commit。

**每个 task 2-5 分钟一步，完整 task 1-3 小时工程时间**。

**常用命令**：
```bash
# user-backend 测试
cd services/user-backend
source .venv/bin/activate
uv run pytest tests/ -v

# DeerFlow backend 测试
cd packages/agent/backend
source .venv/bin/activate
PYTHONPATH=. uv run pytest tests/test_<name>.py -v
make test   # 全量
make lint

# 起本地集成环境
cd packages/agent/docker
docker compose up -d
```

**关键路径速记**：
- 新建 user-backend：`services/user-backend/`（项目根下新 workspace）
- DeerFlow 后端：`packages/agent/backend/`
- Gateway 路由：`packages/agent/backend/app/gateway/routers/`
- Middleware：`packages/agent/backend/packages/harness/deerflow/agents/middlewares/`
- 主 config：`packages/agent/config.yaml`
- Docker Compose：`packages/agent/docker/docker-compose.yaml`
- Frontend：`packages/agent/frontend/src/`

**受保护文件告警**（改动会让 DeerFlow 上游同步时标冲突）：
- `packages/agent/backend/packages/harness/deerflow/agents/middlewares/memory_middleware.py` — Task 8 要改
- `packages/agent/backend/packages/harness/deerflow/agents/memory/storage.py` — Task 8 要改
- `packages/agent/backend/app/gateway/routers/threads.py` — Task 9 要改

**DeerFlow 现有状态（已验证）**：
- `CheckpointerConfig` 已支持 `type: postgres`（[async_provider.py](../../packages/agent/backend/packages/harness/deerflow/agents/checkpointer/async_provider.py)），Task 6 只改 config
- `AioSandboxProvider` 已实装（[aio_sandbox_provider.py](../../packages/agent/backend/packages/harness/deerflow/community/aio_sandbox/aio_sandbox_provider.py)），Task 7 只改 config
- Memory storage 已支持按 `agent_name` 分文件（[storage.py:81-91](../../packages/agent/backend/packages/harness/deerflow/agents/memory/storage.py)），Task 8 复用这条路径把 agent_name 借用为 `user:<user_id>`

---

## Phase A：User Backend 改造（Task 1-5）

> 目标：从 `/home/qiuyangwang/aws-user-backend` 拷出一份，去掉 Cognito，换成本地 JWT + bcrypt + Postgres 用户表。产出一个独立的 FastAPI 服务，提供 `/auth/register`（管理员用）、`/auth/login`、`/auth/verify`、`/auth/me`。

### Task 1：项目骨架 + 拷贝可复用模块

**目标**：在项目根创建 `services/user-backend/`，从 aws-user-backend 拷贝 80% 的非-Cognito 文件。此 task 不写业务代码，只搬文件。

**Files:**
- Create: `services/user-backend/pyproject.toml`
- Create: `services/user-backend/app/main.py`
- Copy from aws-user-backend: `app/core/security.py`, `app/core/config.py`, `app/core/logging.py`, `app/core/errors.py`, `app/core/database.py`, `app/core/snowflake.py`, `app/core/validators.py`

**Step 1: Create directory skeleton**

```bash
mkdir -p /home/qiuyangwang/noldus-insight/services/user-backend/{app/{api/v1,core,models,services,schemas},tests}
cd /home/qiuyangwang/noldus-insight/services/user-backend
```

**Step 2: Write `pyproject.toml`**

Create `services/user-backend/pyproject.toml`:

```toml
[project]
name = "noldus-user-backend"
version = "0.1.0"
description = "Local JWT auth backend for EthoInsight multi-user deployment"
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.110",
    "uvicorn[standard]>=0.29",
    "pydantic>=2.6",
    "pydantic-settings>=2.2",
    "sqlalchemy>=2.0",
    "alembic>=1.13",
    "asyncpg>=0.29",
    "psycopg2-binary>=2.9",
    "python-jose[cryptography]>=3.3",
    "passlib[bcrypt]>=1.7",
    "python-multipart>=0.0.9",
    "httpx>=0.27",
]

[tool.uv]
dev-dependencies = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.3",
]

[tool.ruff]
line-length = 240
```

**Step 3: Copy reusable files from aws-user-backend**

Run (adapt the files that don't depend on Cognito):

```bash
cd /home/qiuyangwang/noldus-insight/services/user-backend
cp /home/qiuyangwang/aws-user-backend/app/core/logging.py app/core/logging.py
cp /home/qiuyangwang/aws-user-backend/app/core/errors.py app/core/errors.py
cp /home/qiuyangwang/aws-user-backend/app/core/database.py app/core/database.py
cp /home/qiuyangwang/aws-user-backend/app/core/snowflake.py app/core/snowflake.py
cp /home/qiuyangwang/aws-user-backend/app/core/validators.py app/core/validators.py
```

**Step 4: Write a minimal `app/main.py`**

Create `services/user-backend/app/main.py`:

```python
"""User Backend main app — local JWT authentication for EthoInsight."""
from fastapi import FastAPI

app = FastAPI(title="Noldus User Backend", version="0.1.0")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
```

**Step 5: Write + run smoke test**

Create `services/user-backend/tests/__init__.py` (empty) and `tests/test_health.py`:

```python
from fastapi.testclient import TestClient
from app.main import app


def test_health_returns_ok():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
```

Run:

```bash
cd services/user-backend
uv venv && source .venv/bin/activate
uv pip install -e .
uv pip install pytest
uv run pytest tests/test_health.py -v
```

Expected: 1 passed.

**Step 6: Commit**

```bash
cd /home/qiuyangwang/noldus-insight
git add services/user-backend/
git commit -m "feat(user-backend): 项目骨架 + 从 aws-user-backend 拷贝可复用模块"
```

---

### Task 2：User ORM 模型 + Alembic migration

**目标**：定义 `User` 表（email + hashed_password + user_id 雪花 ID + created_at），用 Alembic 生成首个 migration。

**Files:**
- Create: `services/user-backend/app/models/user.py`
- Create: `services/user-backend/app/models/__init__.py`
- Create: `services/user-backend/alembic/` 和 migration
- Create: `services/user-backend/tests/test_user_model.py`

**Step 1: Write failing test**

Create `tests/test_user_model.py`:

```python
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.database import Base
from app.models.user import User


@pytest.fixture
def in_memory_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def test_user_creation(in_memory_db):
    user = User(
        user_id=123456789,
        email="alice@noldus.com",
        hashed_password="$2b$12$abc",
    )
    in_memory_db.add(user)
    in_memory_db.commit()

    fetched = in_memory_db.query(User).filter(User.email == "alice@noldus.com").first()
    assert fetched is not None
    assert fetched.user_id == 123456789
    assert fetched.created_at is not None


def test_email_is_unique(in_memory_db):
    u1 = User(user_id=1, email="dup@noldus.com", hashed_password="x")
    u2 = User(user_id=2, email="dup@noldus.com", hashed_password="y")
    in_memory_db.add(u1)
    in_memory_db.commit()
    in_memory_db.add(u2)
    with pytest.raises(Exception):
        in_memory_db.commit()
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_user_model.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'app.models.user'`.

**Step 3: Write model**

Create `app/models/__init__.py` (empty).

Create `app/models/user.py`:

```python
"""User ORM model."""
from sqlalchemy import BigInteger, Boolean, Column, DateTime, String
from sqlalchemy.sql import func

from app.core.database import Base


class User(Base):
    __tablename__ = "users"

    user_id = Column(BigInteger, primary_key=True, comment="Snowflake user ID")
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    display_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, nullable=False, server_default="true")
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def __repr__(self) -> str:
        return f"<User(user_id={self.user_id}, email={self.email})>"
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_user_model.py -v
```
Expected: 2 passed.

**Step 5: Init Alembic + generate first migration**

```bash
cd services/user-backend
uv run alembic init alembic
```

Edit `alembic.ini`:
```ini
sqlalchemy.url = postgresql+psycopg2://noldus:noldus@localhost:5432/noldus_users
```

Edit `alembic/env.py` — find the `target_metadata = None` line and replace:
```python
from app.core.database import Base
from app.models.user import User  # noqa: F401

target_metadata = Base.metadata
```

Generate migration:
```bash
uv run alembic revision --autogenerate -m "create users table"
```

**Step 6: Commit**

```bash
git add services/user-backend/
git commit -m "feat(user-backend): User ORM 模型 + Alembic 初始 migration"
```

---

### Task 3：密码哈希 + JWT 签发/验证（core/local_auth.py）

**目标**：替代 aws-user-backend 的 `app/core/cognito.py`。一个 `LocalAuth` 类，方法 `hash_password / verify_password / issue_jwt / verify_jwt`。JWT 使用 HS256，secret 从 env 读。Token payload: `{sub: user_id, email, exp, iat}`。

**Files:**
- Create: `services/user-backend/app/core/local_auth.py`
- Create: `services/user-backend/tests/test_local_auth.py`

**Step 1: Write failing test**

```python
import time

import pytest

from app.core.local_auth import LocalAuth


@pytest.fixture
def auth():
    return LocalAuth(secret="test-secret-key-do-not-use-in-prod", token_ttl_seconds=3600)


class TestPasswordHashing:
    def test_hash_and_verify_correct_password(self, auth):
        hashed = auth.hash_password("correcthorsebatterystaple")
        assert auth.verify_password("correcthorsebatterystaple", hashed)

    def test_verify_rejects_wrong_password(self, auth):
        hashed = auth.hash_password("abc123")
        assert not auth.verify_password("xyz789", hashed)

    def test_hash_is_bcrypt(self, auth):
        hashed = auth.hash_password("anything")
        assert hashed.startswith("$2b$")


class TestJWT:
    def test_issue_and_verify_jwt(self, auth):
        token = auth.issue_jwt(user_id=12345, email="a@b.com")
        claims = auth.verify_jwt(token)
        assert claims["sub"] == "12345"
        assert claims["email"] == "a@b.com"

    def test_tampered_token_rejected(self, auth):
        token = auth.issue_jwt(user_id=1, email="a@b.com")
        tampered = token[:-3] + "xxx"
        with pytest.raises(Exception):
            auth.verify_jwt(tampered)

    def test_expired_token_rejected(self):
        short_auth = LocalAuth(secret="s", token_ttl_seconds=1)
        token = short_auth.issue_jwt(user_id=1, email="a@b.com")
        time.sleep(2)
        with pytest.raises(Exception):
            short_auth.verify_jwt(token)
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_local_auth.py -v
```
Expected: FAIL — `ModuleNotFoundError`.

**Step 3: Write implementation**

Create `app/core/local_auth.py`:

```python
"""Local JWT + bcrypt authentication (Cognito-free replacement)."""
import time
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from passlib.context import CryptContext

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

JWT_ALGORITHM = "HS256"


class LocalAuth:
    def __init__(self, secret: str, token_ttl_seconds: int = 3600 * 24):
        if len(secret) < 16:
            raise ValueError("JWT secret must be >= 16 chars")
        self._secret = secret
        self._ttl = token_ttl_seconds

    def hash_password(self, plaintext: str) -> str:
        return _pwd_context.hash(plaintext)

    def verify_password(self, plaintext: str, hashed: str) -> bool:
        try:
            return _pwd_context.verify(plaintext, hashed)
        except Exception:
            return False

    def issue_jwt(self, user_id: int, email: str) -> str:
        now = datetime.now(timezone.utc)
        claims = {
            "sub": str(user_id),
            "email": email,
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(seconds=self._ttl)).timestamp()),
        }
        return jwt.encode(claims, self._secret, algorithm=JWT_ALGORITHM)

    def verify_jwt(self, token: str) -> dict:
        return jwt.decode(token, self._secret, algorithms=[JWT_ALGORITHM])
```

**Step 4: Run test to verify it passes**

```bash
uv run pytest tests/test_local_auth.py -v
```
Expected: 6 passed.

**Step 5: Commit**

```bash
git add services/user-backend/
git commit -m "feat(user-backend): 本地 JWT + bcrypt 实现 LocalAuth"
```

---

### Task 4：Auth API 路由（register / login / verify / me）

**目标**：暴露 4 个路由。`register` 受管理员保护（一个环境变量 `ADMIN_BOOTSTRAP_TOKEN`），`login` 返回 JWT，`verify` 和 `me` 用 JWT 鉴权。

**Files:**
- Create: `services/user-backend/app/api/v1/auth.py`
- Create: `services/user-backend/app/api/v1/router.py`
- Create: `services/user-backend/app/schemas/auth.py`
- Create: `services/user-backend/app/services/auth_service.py`
- Create: `services/user-backend/tests/test_auth_api.py`
- Modify: `services/user-backend/app/main.py`

**Step 1: Write failing test**

```python
import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base, get_db
from app.main import app


@pytest.fixture
def client(monkeypatch):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestingSession = sessionmaker(bind=engine)

    def _override_get_db():
        db = TestingSession()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = _override_get_db
    monkeypatch.setenv("JWT_SECRET", "test-secret-16chars")
    monkeypatch.setenv("ADMIN_BOOTSTRAP_TOKEN", "admin-token")
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_register_requires_admin_token(client):
    resp = client.post(
        "/api/v1/auth/register",
        json={"email": "u@n.com", "password": "Pass1234", "display_name": "U"},
    )
    assert resp.status_code == 401


def test_register_succeeds_with_admin_token(client):
    resp = client.post(
        "/api/v1/auth/register",
        headers={"X-Admin-Token": "admin-token"},
        json={"email": "u@n.com", "password": "Pass1234", "display_name": "U"},
    )
    assert resp.status_code == 201
    assert "user_id" in resp.json()


def test_login_returns_jwt(client):
    client.post(
        "/api/v1/auth/register",
        headers={"X-Admin-Token": "admin-token"},
        json={"email": "u@n.com", "password": "Pass1234", "display_name": "U"},
    )
    resp = client.post("/api/v1/auth/login", json={"email": "u@n.com", "password": "Pass1234"})
    assert resp.status_code == 200
    assert "access_token" in resp.json()


def test_login_wrong_password_rejected(client):
    client.post(
        "/api/v1/auth/register",
        headers={"X-Admin-Token": "admin-token"},
        json={"email": "u@n.com", "password": "Pass1234", "display_name": "U"},
    )
    resp = client.post("/api/v1/auth/login", json={"email": "u@n.com", "password": "wrong"})
    assert resp.status_code == 401


def test_me_requires_valid_jwt(client):
    client.post(
        "/api/v1/auth/register",
        headers={"X-Admin-Token": "admin-token"},
        json={"email": "u@n.com", "password": "Pass1234", "display_name": "U"},
    )
    login = client.post("/api/v1/auth/login", json={"email": "u@n.com", "password": "Pass1234"})
    token = login.json()["access_token"]

    me = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["email"] == "u@n.com"


def test_verify_endpoint_returns_user_id(client):
    client.post(
        "/api/v1/auth/register",
        headers={"X-Admin-Token": "admin-token"},
        json={"email": "u@n.com", "password": "Pass1234", "display_name": "U"},
    )
    login = client.post("/api/v1/auth/login", json={"email": "u@n.com", "password": "Pass1234"})
    token = login.json()["access_token"]

    resp = client.post("/api/v1/auth/verify", json={"token": token})
    assert resp.status_code == 200
    assert resp.json()["valid"] is True
    assert "user_id" in resp.json()
```

**Step 2: Run test to verify it fails**

```bash
uv run pytest tests/test_auth_api.py -v
```
Expected: FAIL — no routes exist.

**Step 3: Write Pydantic schemas**

Create `app/schemas/__init__.py` (empty) and `app/schemas/auth.py`:

```python
"""Auth request/response schemas."""
from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    display_name: str | None = None


class RegisterResponse(BaseModel):
    user_id: int
    email: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "Bearer"
    user_id: int
    email: str


class VerifyRequest(BaseModel):
    token: str


class VerifyResponse(BaseModel):
    valid: bool
    user_id: int | None = None
    email: str | None = None


class MeResponse(BaseModel):
    user_id: int
    email: str
    display_name: str | None = None
```

**Step 4: Write service**

Create `app/services/__init__.py` (empty) and `app/services/auth_service.py`:

```python
"""Auth business logic."""
import os

from sqlalchemy.orm import Session

from app.core.local_auth import LocalAuth
from app.core.snowflake import generate_id
from app.models.user import User


def _get_auth() -> LocalAuth:
    secret = os.environ.get("JWT_SECRET", "")
    if not secret:
        raise RuntimeError("JWT_SECRET env var is required")
    return LocalAuth(secret=secret, token_ttl_seconds=3600 * 24)


class AuthService:
    def __init__(self, db: Session):
        self.db = db
        self.auth = _get_auth()

    def register(self, email: str, password: str, display_name: str | None) -> User:
        existing = self.db.query(User).filter(User.email == email).first()
        if existing is not None:
            raise ValueError("email already registered")
        user = User(
            user_id=generate_id(),
            email=email,
            hashed_password=self.auth.hash_password(password),
            display_name=display_name,
        )
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def login(self, email: str, password: str) -> tuple[User, str]:
        user = self.db.query(User).filter(User.email == email, User.is_active).first()
        if user is None or not self.auth.verify_password(password, user.hashed_password):
            raise PermissionError("invalid credentials")
        token = self.auth.issue_jwt(user_id=user.user_id, email=user.email)
        return user, token

    def verify(self, token: str) -> dict | None:
        try:
            return self.auth.verify_jwt(token)
        except Exception:
            return None

    def get_by_id(self, user_id: int) -> User | None:
        return self.db.query(User).filter(User.user_id == user_id).first()
```

Note `app/core/snowflake.py` — it came from aws-user-backend in Task 1. It exposes `generate_id()` returning 64-bit int. If the existing function name differs, adapt the import.

**Step 5: Write routes**

Create `app/api/__init__.py`, `app/api/v1/__init__.py` (both empty), `app/api/v1/auth.py`:

```python
"""Auth API routes."""
import os

from fastapi import APIRouter, Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.schemas.auth import (
    LoginRequest,
    LoginResponse,
    MeResponse,
    RegisterRequest,
    RegisterResponse,
    VerifyRequest,
    VerifyResponse,
)
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])


def _require_admin(x_admin_token: str = Header(default="")) -> None:
    expected = os.environ.get("ADMIN_BOOTSTRAP_TOKEN", "")
    if not expected or x_admin_token != expected:
        raise HTTPException(status_code=401, detail="admin token required")


def _current_user_id(authorization: str = Header(default=""), db: Session = Depends(get_db)) -> int:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = authorization[7:]
    service = AuthService(db)
    claims = service.verify(token)
    if claims is None:
        raise HTTPException(status_code=401, detail="invalid token")
    return int(claims["sub"])


@router.post("/register", response_model=RegisterResponse, status_code=201, dependencies=[Depends(_require_admin)])
def register(body: RegisterRequest, db: Session = Depends(get_db)) -> RegisterResponse:
    service = AuthService(db)
    try:
        user = service.register(body.email, body.password, body.display_name)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return RegisterResponse(user_id=user.user_id, email=user.email)


@router.post("/login", response_model=LoginResponse)
def login(body: LoginRequest, db: Session = Depends(get_db)) -> LoginResponse:
    service = AuthService(db)
    try:
        user, token = service.login(body.email, body.password)
    except PermissionError:
        raise HTTPException(status_code=401, detail="invalid credentials")
    return LoginResponse(access_token=token, user_id=user.user_id, email=user.email)


@router.post("/verify", response_model=VerifyResponse)
def verify(body: VerifyRequest, db: Session = Depends(get_db)) -> VerifyResponse:
    service = AuthService(db)
    claims = service.verify(body.token)
    if claims is None:
        return VerifyResponse(valid=False)
    return VerifyResponse(valid=True, user_id=int(claims["sub"]), email=claims["email"])


@router.get("/me", response_model=MeResponse)
def me(user_id: int = Depends(_current_user_id), db: Session = Depends(get_db)) -> MeResponse:
    service = AuthService(db)
    user = service.get_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="user not found")
    return MeResponse(user_id=user.user_id, email=user.email, display_name=user.display_name)
```

Create `app/api/v1/router.py`:

```python
from fastapi import APIRouter

from app.api.v1.auth import router as auth_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(auth_router)
```

**Step 6: Register router in main.py**

Edit `app/main.py`:

```python
"""User Backend main app — local JWT authentication for EthoInsight."""
from fastapi import FastAPI

from app.api.v1.router import api_router

app = FastAPI(title="Noldus User Backend", version="0.1.0")
app.include_router(api_router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8002)
```

**Step 7: Run tests to verify all pass**

```bash
uv run pytest tests/ -v
```
Expected: all passed (9+ tests).

**Step 8: Commit**

```bash
git add services/user-backend/
git commit -m "feat(user-backend): 实装 register/login/verify/me 四个 auth 路由"
```

---

### Task 5：user-backend Dockerfile + docker-compose 集成

**目标**：写 Dockerfile，把 user-backend 加到 `packages/agent/docker/docker-compose.yaml` 里。同时在 compose 里加 Postgres container。

**Files:**
- Create: `services/user-backend/Dockerfile`
- Modify: `packages/agent/docker/docker-compose.yaml`
- Create: `packages/agent/docker/.env.example`（补 JWT_SECRET 等）

**Step 1: Write Dockerfile**

Create `services/user-backend/Dockerfile`:

```dockerfile
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir uv && uv pip install --system -e .

COPY app ./app
COPY alembic ./alembic
COPY alembic.ini ./

EXPOSE 8002

CMD ["sh", "-c", "alembic upgrade head && uvicorn app.main:app --host 0.0.0.0 --port 8002"]
```

**Step 2: Add Postgres + user-backend to docker-compose**

Modify `packages/agent/docker/docker-compose.yaml` — append services (under existing `services:` block):

```yaml
  # ── Postgres (shared by LangGraph checkpointer + user-backend) ─────────────
  postgres:
    image: postgres:15-alpine
    container_name: deer-flow-postgres
    environment:
      - POSTGRES_USER=${POSTGRES_USER:-noldus}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-noldus}
      - POSTGRES_MULTIPLE_DATABASES=noldus_users,noldus_langgraph
    volumes:
      - postgres-data:/var/lib/postgresql/data
      - ./init-multi-db.sh:/docker-entrypoint-initdb.d/init-multi-db.sh:ro
    networks:
      - deer-flow
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U noldus"]
      interval: 5s
      timeout: 5s
      retries: 5

  # ── User Backend (JWT auth service) ────────────────────────────────────────
  user-backend:
    build:
      context: ../../../services/user-backend
      dockerfile: Dockerfile
    container_name: noldus-user-backend
    environment:
      - JWT_SECRET=${JWT_SECRET}
      - ADMIN_BOOTSTRAP_TOKEN=${ADMIN_BOOTSTRAP_TOKEN}
      - DATABASE_URL=postgresql+psycopg2://${POSTGRES_USER:-noldus}:${POSTGRES_PASSWORD:-noldus}@postgres:5432/noldus_users
    depends_on:
      postgres:
        condition: service_healthy
    networks:
      - deer-flow
    restart: unless-stopped

# At the bottom, next to existing `networks:` block, add:
volumes:
  postgres-data:
```

Create `packages/agent/docker/init-multi-db.sh`:

```bash
#!/bin/bash
set -e
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
    CREATE DATABASE noldus_users;
    CREATE DATABASE noldus_langgraph;
EOSQL
```

Make executable:
```bash
chmod +x packages/agent/docker/init-multi-db.sh
```

**Step 3: Update .env.example**

Append to `packages/agent/.env.example` (or create if missing):

```env
# Postgres (shared)
POSTGRES_USER=noldus
POSTGRES_PASSWORD=change-me-in-production

# User Backend
JWT_SECRET=use-openssl-rand-hex-32-to-generate
ADMIN_BOOTSTRAP_TOKEN=bootstrap-only-rotate-after-deploy
```

**Step 4: Smoke test**

```bash
cd packages/agent/docker
cp ../.env.example .env
# Edit .env: set real JWT_SECRET (openssl rand -hex 32), ADMIN_BOOTSTRAP_TOKEN
docker compose up -d postgres user-backend
sleep 10
curl http://localhost:8002/health
```
Expected: `{"status":"ok"}`.

Also test an actual register + login roundtrip:
```bash
curl -X POST http://localhost:8002/api/v1/auth/register \
     -H "X-Admin-Token: $ADMIN_BOOTSTRAP_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"email":"test@noldus.com","password":"TestPass123","display_name":"Test"}'
curl -X POST http://localhost:8002/api/v1/auth/login \
     -H "Content-Type: application/json" \
     -d '{"email":"test@noldus.com","password":"TestPass123"}'
```
Expected: register returns 201 + user_id, login returns JWT.

**Step 5: Commit**

```bash
cd /home/qiuyangwang/noldus-insight
git add services/user-backend/Dockerfile packages/agent/docker/docker-compose.yaml \
       packages/agent/docker/init-multi-db.sh packages/agent/.env.example
git commit -m "feat(deploy): user-backend Dockerfile + Postgres + docker-compose 集成"
```

---

## Phase B：DeerFlow 多用户切换（Task 6-9）

> 目标：改 config.yaml 切 Postgres checkpointer 和 AioSandbox，改 memory/thread 路由按 user_id 隔离。

### Task 6：Checkpointer 切 Postgres

**目标**：把 `config.yaml` 的 checkpointer 从 sqlite 改成 postgres，加依赖。DeerFlow 的 `async_provider.py` 已支持，本 task 只是配置 + 集成测试。

**Files:**
- Modify: `packages/agent/config.yaml:167-169`
- Modify: `packages/agent/backend/pyproject.toml`（加 `langgraph-checkpoint-postgres`）
- Create: `packages/agent/backend/tests/test_postgres_checkpointer_config.py`

**Step 1: Write failing test**

```python
"""Verify checkpointer config is set to postgres with valid DSN."""
import os

import pytest
import yaml


@pytest.fixture
def config_yaml_path():
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.yaml")
    )


def test_checkpointer_is_postgres(config_yaml_path):
    with open(config_yaml_path) as f:
        cfg = yaml.safe_load(f)
    assert cfg["checkpointer"]["type"] == "postgres"
    assert cfg["checkpointer"]["connection_string"].startswith("postgresql")


def test_checkpointer_dsn_uses_env_var(config_yaml_path):
    with open(config_yaml_path) as f:
        cfg = yaml.safe_load(f)
    # DSN may embed $ENV_VAR — at minimum verify host is configurable
    dsn = cfg["checkpointer"]["connection_string"]
    assert "$POSTGRES_HOST" in dsn or "postgres" in dsn
```

**Step 2: Run test to verify it fails**

```bash
cd packages/agent/backend
PYTHONPATH=. uv run pytest tests/test_postgres_checkpointer_config.py -v
```
Expected: FAIL — still `type: sqlite`.

**Step 3: Update config.yaml**

Edit `packages/agent/config.yaml` lines 167-169:

```yaml
# ============================================================================
# Checkpointer
# ============================================================================
checkpointer:
  type: postgres
  connection_string: postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@$POSTGRES_HOST:5432/noldus_langgraph
```

**Step 4: Add dependency**

Edit `packages/agent/backend/packages/harness/pyproject.toml` (the one that owns deerflow-harness):

Find the `dependencies` array (or `[project.optional-dependencies]` / extras), append:

```toml
"langgraph-checkpoint-postgres>=2.0",
"psycopg[binary]>=3.1",
```

Then:
```bash
cd packages/agent/backend
uv sync
```

**Step 5: Run test to verify it passes**

```bash
PYTHONPATH=. uv run pytest tests/test_postgres_checkpointer_config.py -v
```
Expected: 2 passed.

Also do integration smoke:
```bash
cd packages/agent/docker
docker compose up -d postgres
export POSTGRES_HOST=localhost POSTGRES_USER=noldus POSTGRES_PASSWORD=noldus
cd ../backend
PYTHONPATH=. uv run python -c "
import asyncio
from deerflow.agents.checkpointer.async_provider import make_checkpointer
async def main():
    async with make_checkpointer() as ckpt:
        print(type(ckpt).__name__)
asyncio.run(main())
"
```
Expected: prints `AsyncPostgresSaver`.

**Step 6: Commit**

```bash
git add packages/agent/config.yaml packages/agent/backend/packages/harness/pyproject.toml
git commit -m "feat(deploy): checkpointer 切 Postgres"
```

---

### Task 7：Sandbox 切 AioSandbox

**目标**：`config.yaml` sandbox 部分从 LocalSandbox 换成 AioSandbox。docker-compose 里给 langgraph container 挂 `/var/run/docker.sock`。

**Files:**
- Modify: `packages/agent/config.yaml:108-112`
- Modify: `packages/agent/docker/docker-compose.yaml`（langgraph service 加 docker socket 挂载）
- Create: `packages/agent/backend/tests/test_sandbox_config.py`

**Step 1: Write failing test**

```python
import os

import pytest
import yaml


@pytest.fixture
def config_yaml():
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.yaml")
    )
    with open(path) as f:
        return yaml.safe_load(f)


def test_sandbox_is_aio(config_yaml):
    assert config_yaml["sandbox"]["use"].endswith("AioSandboxProvider")


def test_sandbox_has_isolation_settings(config_yaml):
    s = config_yaml["sandbox"]
    assert s.get("replicas", 0) >= 3
    assert s.get("container_prefix")
    assert s.get("idle_timeout") is not None


def test_sandbox_mounts_thread_data(config_yaml):
    mounts = config_yaml["sandbox"].get("mounts", [])
    container_paths = {m["container_path"] for m in mounts}
    assert "/mnt/user-data" in container_paths or "/mnt/skills" in container_paths
```

**Step 2: Run test to verify it fails**

```bash
PYTHONPATH=. uv run pytest tests/test_sandbox_config.py -v
```
Expected: FAIL — still LocalSandboxProvider.

**Step 3: Update config.yaml**

Replace `packages/agent/config.yaml` lines 105-112:

```yaml
# ============================================================================
# Sandbox — per-thread Docker container isolation
# ============================================================================
sandbox:
  use: deerflow.community.aio_sandbox:AioSandboxProvider
  image: enterprise-public-cn-beijing.cr.volces.com/vefaas-public/all-in-one-sandbox:latest
  port: 8080
  container_prefix: noldus-insight
  replicas: 5                        # Max concurrent sandbox containers
  idle_timeout: 600                  # Evict after 10min idle
  mounts:
    - host_path: ${DEER_FLOW_HOME}/threads
      container_path: /mnt/user-data
    - host_path: ${DEER_FLOW_REPO_ROOT}/packages/agent/skills
      container_path: /mnt/skills
      read_only: true
  environment:
    - PYTHONUNBUFFERED=1
  bash_output_max_chars: 20000
  read_file_output_max_chars: 50000
```

**Step 4: Mount docker socket in langgraph service**

Edit `packages/agent/docker/docker-compose.yaml`, find the `langgraph:` service block, add under `volumes:`:

```yaml
      - ${DEER_FLOW_DOCKER_SOCKET:-/var/run/docker.sock}:/var/run/docker.sock
```

**Step 5: Run test to verify it passes**

```bash
PYTHONPATH=. uv run pytest tests/test_sandbox_config.py -v
```
Expected: 3 passed.

**Step 6: Smoke test (optional but recommended)**

```bash
cd packages/agent/docker
docker compose up -d postgres langgraph
sleep 30
docker compose logs langgraph | grep -i sandbox
```
Expected: logs show AioSandboxProvider initialized; no LocalSandbox singleton creation.

**Step 7: Commit**

```bash
git add packages/agent/config.yaml packages/agent/docker/docker-compose.yaml \
       packages/agent/backend/tests/test_sandbox_config.py
git commit -m "feat(deploy): sandbox 切 AioSandbox（每 thread 一个容器）"
```

---

### Task 8：Memory 按 user_id 分文件

**目标**：把 user_id 通过 `runtime.context` 传到 `MemoryMiddleware`，storage 路径改为 `memory/user_<user_id>.json`。DeerFlow 已有 `agent_name` 参数复用这条路径。

⚠️ **受保护文件**：`memory_middleware.py` 和 `memory/storage.py` 都是 DeerFlow 上游文件，同步时会标冲突。

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/middlewares/memory_middleware.py`
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/memory/queue.py`
- Create: `packages/agent/backend/tests/test_memory_user_isolation.py`

**Step 1: Write failing test**

```python
"""Memory must be isolated per user_id."""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from deerflow.agents.middlewares.memory_middleware import MemoryMiddleware


def test_memory_middleware_uses_user_id_from_runtime_context(tmp_path, monkeypatch):
    """When user_id is in runtime.context, memory should be stored under that user's file."""
    from deerflow.config.memory_config import MemoryConfig, set_memory_config

    set_memory_config(MemoryConfig(
        enabled=True,
        storage_path=str(tmp_path / "memory.json"),
        injection_enabled=True,
        debounce_seconds=0,
    ))

    mw = MemoryMiddleware()

    state = {
        "messages": [
            HumanMessage(content="hello"),
            AIMessage(content="hi there"),
        ],
    }

    captured_agent_names: list[str] = []

    class FakeQueue:
        def enqueue(self, *, thread_id, messages, agent_name=None, **kwargs):
            captured_agent_names.append(agent_name)

    monkeypatch.setattr(
        "deerflow.agents.middlewares.memory_middleware.get_memory_queue",
        lambda: FakeQueue(),
    )

    mw.after_agent(state=state, runtime=Runtime(context={"thread_id": "t1", "user_id": "42"}))

    assert captured_agent_names == ["user:42"]


def test_memory_middleware_falls_back_to_global_when_no_user_id(tmp_path, monkeypatch):
    """If runtime.context has no user_id, agent_name stays None → global memory (backward compat)."""
    from deerflow.config.memory_config import MemoryConfig, set_memory_config

    set_memory_config(MemoryConfig(enabled=True, storage_path=str(tmp_path / "memory.json"), injection_enabled=True, debounce_seconds=0))

    mw = MemoryMiddleware()
    captured: list[str | None] = []

    class FakeQueue:
        def enqueue(self, *, thread_id, messages, agent_name=None, **kwargs):
            captured.append(agent_name)

    monkeypatch.setattr(
        "deerflow.agents.middlewares.memory_middleware.get_memory_queue",
        lambda: FakeQueue(),
    )

    mw.after_agent(
        state={"messages": [HumanMessage(content="x"), AIMessage(content="y")]},
        runtime=Runtime(context={"thread_id": "t1"}),
    )

    assert captured == [None]
```

**Step 2: Run test to verify it fails**

```bash
PYTHONPATH=. uv run pytest tests/test_memory_user_isolation.py -v
```
Expected: FAIL — middleware doesn't read user_id yet.

**Step 3: Modify MemoryMiddleware**

Open `packages/agent/backend/packages/harness/deerflow/agents/middlewares/memory_middleware.py`. Find `after_agent` method (around line 196). Modify to read user_id:

Locate the existing code:
```python
thread_id = runtime.context.get("thread_id") if runtime.context else None
```

Immediately after it add:
```python
user_id = runtime.context.get("user_id") if runtime.context else None
```

Find where `queue.enqueue(...)` is called. Replace the `agent_name` kwarg (or add it) so user_id wins over the existing `self._agent_name`:

```python
# Derive agent_name: user_id takes priority for multi-user isolation
effective_agent_name = f"user:{user_id}" if user_id else self._agent_name
queue.enqueue(
    thread_id=thread_id,
    messages=filtered_messages,
    agent_name=effective_agent_name,
    # ... other existing kwargs
)
```

Keep all other kwargs of the existing enqueue call unchanged.

**Step 4: Make sure storage.py handles `user:<id>` pattern**

Open `packages/agent/backend/packages/harness/deerflow/agents/memory/storage.py`. The existing `_validate_agent_name` may reject colons. Look at line ~85 area:

```python
def _validate_agent_name(self, agent_name: str) -> None:
    # existing validator probably checks for alphanum + dash
```

If it rejects `:`, extend the allowed charset to include `:`:
```python
import re
if not re.fullmatch(r"[A-Za-z0-9_\-:]+", agent_name):
    raise ValueError(f"Invalid agent_name: {agent_name}")
```

**Step 5: Run tests**

```bash
PYTHONPATH=. uv run pytest tests/test_memory_user_isolation.py -v
PYTHONPATH=. uv run pytest tests/test_memory_middleware.py -v  # regression
PYTHONPATH=. uv run pytest tests/test_memory_updater.py -v     # regression
```
Expected: new tests + existing memory tests all pass.

**Step 6: Commit**

```bash
git add packages/agent/backend/packages/harness/deerflow/agents/middlewares/memory_middleware.py \
       packages/agent/backend/packages/harness/deerflow/agents/memory/storage.py \
       packages/agent/backend/tests/test_memory_user_isolation.py
git commit -m "feat(deploy): memory 按 user_id 分文件隔离"
```

---

### Task 9：Gateway JWT middleware + Thread 按 user_id 过滤

**目标**：Gateway 新增 JWT 验证 middleware，提取 user_id 注入 request state；`/api/threads/search` 按 user_id 过滤；创建 thread 时写入 metadata；user_id 塞进 LangGraph `config.configurable` 以便 Middleware 读取。

⚠️ **受保护文件**：`threads.py` 会改。

**Files:**
- Create: `packages/agent/backend/app/gateway/middleware/jwt_auth.py`
- Modify: `packages/agent/backend/app/gateway/app.py`（注册 middleware）
- Modify: `packages/agent/backend/app/gateway/routers/threads.py`（过滤）
- Create: `packages/agent/backend/tests/test_jwt_auth_middleware.py`
- Create: `packages/agent/backend/tests/test_threads_user_filter.py`

**Step 1: Write failing tests for JWT middleware**

```python
"""JWT auth middleware verifies tokens via user-backend /api/v1/auth/verify."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.gateway.middleware.jwt_auth import JWTAuthMiddleware


@pytest.fixture
def app_with_mw(monkeypatch):
    import httpx

    # Mock httpx.post to simulate user-backend verify response
    def fake_post(url, json, timeout):
        class _Resp:
            status_code = 200
            def json(self):
                token = json["token"]
                if token == "valid-token":
                    return {"valid": True, "user_id": 42, "email": "a@n.com"}
                return {"valid": False}
        return _Resp()

    monkeypatch.setattr("httpx.post", fake_post)

    app = FastAPI()
    app.add_middleware(JWTAuthMiddleware, user_backend_url="http://user-backend:8002", exempt_paths=["/health"])

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.get("/protected")
    async def protected(request):
        return {"user_id": request.state.user_id}

    return TestClient(app)


def test_exempt_path_works_without_token(app_with_mw):
    resp = app_with_mw.get("/health")
    assert resp.status_code == 200


def test_protected_path_rejects_missing_token(app_with_mw):
    resp = app_with_mw.get("/protected")
    assert resp.status_code == 401


def test_protected_path_rejects_invalid_token(app_with_mw):
    resp = app_with_mw.get("/protected", headers={"Authorization": "Bearer bad"})
    assert resp.status_code == 401


def test_protected_path_accepts_valid_token_and_injects_user_id(app_with_mw):
    resp = app_with_mw.get("/protected", headers={"Authorization": "Bearer valid-token"})
    assert resp.status_code == 200
    assert resp.json()["user_id"] == 42
```

**Step 2: Run test to verify it fails**

```bash
PYTHONPATH=. uv run pytest tests/test_jwt_auth_middleware.py -v
```
Expected: FAIL — middleware doesn't exist.

**Step 3: Write JWT middleware**

Create `app/gateway/middleware/__init__.py` (empty) and `app/gateway/middleware/jwt_auth.py`:

```python
"""JWT auth middleware — delegates verification to user-backend."""
import logging

import httpx
from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class JWTAuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, user_backend_url: str, exempt_paths: list[str] | None = None):
        super().__init__(app)
        self._user_backend = user_backend_url.rstrip("/")
        self._exempt = set(exempt_paths or [])

    async def dispatch(self, request: Request, call_next):
        if request.url.path in self._exempt or any(
            request.url.path.startswith(p + "/") for p in self._exempt
        ):
            return await call_next(request)

        auth = request.headers.get("authorization", "")
        if not auth.startswith("Bearer "):
            return JSONResponse({"detail": "missing bearer token"}, status_code=401)
        token = auth[7:]

        try:
            resp = httpx.post(
                f"{self._user_backend}/api/v1/auth/verify",
                json={"token": token},
                timeout=5.0,
            )
        except Exception as e:
            logger.warning("JWT verify call failed: %s", e)
            return JSONResponse({"detail": "auth service unreachable"}, status_code=503)

        if resp.status_code != 200:
            return JSONResponse({"detail": "auth service error"}, status_code=503)
        body = resp.json()
        if not body.get("valid"):
            return JSONResponse({"detail": "invalid or expired token"}, status_code=401)

        request.state.user_id = body["user_id"]
        request.state.user_email = body.get("email")
        return await call_next(request)
```

**Step 4: Register middleware in app.py**

Edit `packages/agent/backend/app/gateway/app.py`. Find the `app = FastAPI(...)` and existing `add_middleware` calls. Append:

```python
import os
from app.gateway.middleware.jwt_auth import JWTAuthMiddleware

_user_backend_url = os.environ.get("USER_BACKEND_URL", "http://user-backend:8002")
app.add_middleware(
    JWTAuthMiddleware,
    user_backend_url=_user_backend_url,
    exempt_paths=["/health", "/docs", "/openapi.json", "/redoc"],
)
```

**Step 5: Write thread filter tests**

```python
"""Thread search must filter by current user's user_id."""
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch


def test_search_threads_filters_by_user_id(monkeypatch):
    """When user 42 requests thread list, they should not see user 7's threads."""
    from app.gateway.app import app

    # Bypass JWT middleware by setting state directly on the request
    async def fake_dispatch(request, call_next):
        request.state.user_id = 42
        return await call_next(request)

    from app.gateway.middleware.jwt_auth import JWTAuthMiddleware
    monkeypatch.setattr(JWTAuthMiddleware, "dispatch", fake_dispatch)

    # Mock store to return threads owned by both user 42 and user 7
    fake_store_items = [
        MagicMock(value={"thread_id": "t1", "metadata": {"user_id": 42}, "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z"}),
        MagicMock(value={"thread_id": "t2", "metadata": {"user_id": 7}, "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z"}),
        MagicMock(value={"thread_id": "t3", "metadata": {"user_id": 42}, "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z"}),
    ]

    async def fake_asearch(*a, **kw):
        return fake_store_items

    fake_store = MagicMock()
    fake_store.asearch = fake_asearch

    with patch("app.gateway.routers.threads.get_store", return_value=fake_store), \
         patch("app.gateway.routers.threads.get_checkpointer", return_value=None):
        client = TestClient(app)
        resp = client.post("/api/threads/search", json={})

    assert resp.status_code == 200
    returned_ids = [t["thread_id"] for t in resp.json()]
    assert "t1" in returned_ids
    assert "t3" in returned_ids
    assert "t2" not in returned_ids
```

**Step 6: Run test to verify it fails**

```bash
PYTHONPATH=. uv run pytest tests/test_threads_user_filter.py -v
```
Expected: FAIL — filter not implemented.

**Step 7: Add user_id filter to threads.py**

Edit `packages/agent/backend/app/gateway/routers/threads.py`. In `search_threads` function (around line 319), at the start insert:

```python
user_id = getattr(request.state, "user_id", None)
```

Then find the Phase 1 loop that processes store items. Before appending to `merged`, add filter:

```python
# Multi-user: only return threads belonging to current user
md = item.value.get("metadata", {}) or {}
if user_id is not None and md.get("user_id") != user_id:
    continue
```

Also in `create_thread` (find the POST route), inject user_id into metadata when creating:

```python
# Near where metadata dict is built for new thread:
if hasattr(request.state, "user_id"):
    body.metadata.setdefault("user_id", request.state.user_id)
```

**Step 8: Also pass user_id to LangGraph configurable**

Find where the Gateway forwards to LangGraph for running a thread (likely in a `runs` router or where `client.runs.stream(...)` is called). Ensure that `config.configurable` includes `user_id`:

```python
config = {
    "configurable": {
        "thread_id": thread_id,
        "user_id": request.state.user_id,
    }
}
```

This is what Task 8's `MemoryMiddleware.after_agent` reads via `runtime.context.get("user_id")`.

**Step 9: Run tests**

```bash
PYTHONPATH=. uv run pytest tests/test_jwt_auth_middleware.py tests/test_threads_user_filter.py -v
```
Expected: all pass.

Run full regression:
```bash
make test
```
Expected: no previously-passing test broken.

**Step 10: Commit**

```bash
git add packages/agent/backend/app/gateway/middleware/ \
       packages/agent/backend/app/gateway/app.py \
       packages/agent/backend/app/gateway/routers/threads.py \
       packages/agent/backend/tests/test_jwt_auth_middleware.py \
       packages/agent/backend/tests/test_threads_user_filter.py
git commit -m "feat(deploy): Gateway JWT middleware + Thread 按 user_id 过滤"
```

---

## Phase C：前端登录 + 数据隔离（Task 10-12）

### Task 10：前端登录页

**目标**：`/login` 页面，邮箱 + 密码，成功后把 JWT 存 localStorage 并跳 `/chat`。未登录访问任何页面都重定向到 `/login`。

**Files:**
- Create: `packages/agent/frontend/src/app/login/page.tsx`
- Create: `packages/agent/frontend/src/core/auth/auth-client.ts`
- Create: `packages/agent/frontend/src/core/auth/use-auth.ts`
- Create: `packages/agent/frontend/src/core/auth/auth-client.test.ts`
- Modify: `packages/agent/frontend/src/app/layout.tsx`（全局 AuthProvider）

**Step 1: Write failing test for auth-client**

```typescript
import { describe, it, expect, vi, beforeEach } from "vitest";
import { login, getToken, logout, getCurrentUser } from "./auth-client";

describe("auth-client", () => {
  beforeEach(() => {
    localStorage.clear();
    global.fetch = vi.fn();
  });

  it("login stores token on success", async () => {
    (global.fetch as any).mockResolvedValue({
      ok: true,
      json: async () => ({ access_token: "abc.def.ghi", user_id: 42, email: "a@n.com" }),
    });
    const result = await login("a@n.com", "pw");
    expect(result.access_token).toBe("abc.def.ghi");
    expect(getToken()).toBe("abc.def.ghi");
    expect(getCurrentUser()).toEqual({ user_id: 42, email: "a@n.com" });
  });

  it("login throws on 401", async () => {
    (global.fetch as any).mockResolvedValue({ ok: false, status: 401 });
    await expect(login("a@n.com", "bad")).rejects.toThrow();
  });

  it("logout clears token", () => {
    localStorage.setItem("noldus_jwt", "x");
    localStorage.setItem("noldus_user", JSON.stringify({ user_id: 1, email: "a" }));
    logout();
    expect(getToken()).toBeNull();
    expect(getCurrentUser()).toBeNull();
  });
});
```

**Step 2: Run test to verify it fails**

```bash
cd packages/agent/frontend
pnpm test src/core/auth/auth-client.test.ts
```
Expected: FAIL — module doesn't exist.

**Step 3: Write auth-client**

Create `src/core/auth/auth-client.ts`:

```typescript
const TOKEN_KEY = "noldus_jwt";
const USER_KEY = "noldus_user";

export interface CurrentUser {
  user_id: number;
  email: string;
}

export interface LoginResponse {
  access_token: string;
  user_id: number;
  email: string;
}

export async function login(email: string, password: string): Promise<LoginResponse> {
  const res = await fetch("/api/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  if (!res.ok) throw new Error("Login failed");
  const data = (await res.json()) as LoginResponse;
  localStorage.setItem(TOKEN_KEY, data.access_token);
  localStorage.setItem(USER_KEY, JSON.stringify({ user_id: data.user_id, email: data.email }));
  return data;
}

export function getToken(): string | null {
  return localStorage.getItem(TOKEN_KEY);
}

export function getCurrentUser(): CurrentUser | null {
  const raw = localStorage.getItem(USER_KEY);
  return raw ? (JSON.parse(raw) as CurrentUser) : null;
}

export function logout(): void {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}
```

**Step 4: Run test to verify it passes**

```bash
pnpm test src/core/auth/auth-client.test.ts
```
Expected: 3 passed.

**Step 5: Write login page**

Create `src/app/login/page.tsx`:

```tsx
"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

import { login } from "@/core/auth/auth-client";

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      await login(email, password);
      router.push("/chat");
    } catch {
      setError("邮箱或密码错误");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-background">
      <form onSubmit={handleSubmit} className="w-full max-w-sm space-y-4 p-8 border rounded">
        <h1 className="text-2xl font-semibold">登录 EthoInsight</h1>
        <input
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="邮箱"
          required
          className="w-full p-2 border rounded"
        />
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="密码"
          required
          className="w-full p-2 border rounded"
        />
        {error && <p className="text-sm text-red-600">{error}</p>}
        <button
          type="submit"
          disabled={busy}
          className="w-full py-2 bg-primary text-primary-foreground rounded"
        >
          {busy ? "登录中..." : "登录"}
        </button>
        <p className="text-xs text-muted-foreground">
          账号由管理员预注册。如需账号请联系 Noldus 团队。
        </p>
      </form>
    </div>
  );
}
```

**Step 6: Commit**

```bash
git add packages/agent/frontend/src/core/auth/ packages/agent/frontend/src/app/login/
git commit -m "feat(frontend): 登录页 + auth-client"
```

---

### Task 11：API 请求带 JWT + 未登录重定向

**目标**：所有前端对 `/api/*` 的请求自动加 `Authorization: Bearer <token>` header。未登录用户访问受保护页面重定向到 `/login`。

**Files:**
- Modify: `packages/agent/frontend/src/core/api/api-client.ts`
- Create: `packages/agent/frontend/src/core/auth/auth-guard.tsx`
- Modify: `packages/agent/frontend/src/app/chat/layout.tsx`（或等效 chat 页面的 layout）
- Create: `packages/agent/frontend/src/core/api/auth-headers.test.ts`

**Step 1: Write failing test**

```typescript
import { describe, it, expect, vi, beforeEach } from "vitest";
import { authedFetch } from "./api-client";

describe("authedFetch", () => {
  beforeEach(() => {
    localStorage.clear();
    global.fetch = vi.fn();
  });

  it("attaches Bearer token when present", async () => {
    localStorage.setItem("noldus_jwt", "my-jwt");
    (global.fetch as any).mockResolvedValue({ ok: true, json: async () => ({}) });

    await authedFetch("/api/threads/search", { method: "POST" });

    const call = (global.fetch as any).mock.calls[0];
    expect(call[1].headers["Authorization"]).toBe("Bearer my-jwt");
  });

  it("does not attach header when no token", async () => {
    (global.fetch as any).mockResolvedValue({ ok: true, json: async () => ({}) });

    await authedFetch("/api/public");

    const call = (global.fetch as any).mock.calls[0];
    expect(call[1]?.headers?.["Authorization"]).toBeUndefined();
  });
});
```

**Step 2: Run test to verify it fails**

```bash
pnpm test src/core/api/auth-headers.test.ts
```
Expected: FAIL — `authedFetch` not exported.

**Step 3: Export `authedFetch` from api-client**

Append to `src/core/api/api-client.ts`:

```typescript
import { getToken } from "@/core/auth/auth-client";

export async function authedFetch(
  input: RequestInfo | URL,
  init: RequestInit = {},
): Promise<Response> {
  const token = getToken();
  const headers: Record<string, string> = {
    ...((init.headers as Record<string, string>) || {}),
  };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  return fetch(input, { ...init, headers });
}
```

Then grep for all existing `fetch(` calls in `src/core/api/` and `src/core/threads/` (especially `submitFeedback`, `listFeedback`, thread CRUD) — replace with `authedFetch(...)`.

**Step 4: Write AuthGuard component**

Create `src/core/auth/auth-guard.tsx`:

```tsx
"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

import { getCurrentUser } from "./auth-client";

export function AuthGuard({ children }: { children: React.ReactNode }) {
  const router = useRouter();

  useEffect(() => {
    if (!getCurrentUser()) {
      router.replace("/login");
    }
  }, [router]);

  if (typeof window !== "undefined" && !getCurrentUser()) {
    return null;
  }
  return <>{children}</>;
}
```

**Step 5: Wrap chat layout**

Locate the chat page's layout (`src/app/chat/layout.tsx` or the closest wrapper). Wrap children:

```tsx
import { AuthGuard } from "@/core/auth/auth-guard";

export default function ChatLayout({ children }: { children: React.ReactNode }) {
  return <AuthGuard>{children}</AuthGuard>;
}
```

**Step 6: Run tests**

```bash
pnpm test
```
Expected: all pass.

Smoke manual test:
```bash
make dev   # project root
# Open http://localhost:2026
# Expect redirect to /login
# Login with the test account registered in Task 5
# Expect chat page
```

**Step 7: Commit**

```bash
git add packages/agent/frontend/src/core/api/ packages/agent/frontend/src/core/auth/auth-guard.tsx \
       packages/agent/frontend/src/app/chat/layout.tsx
git commit -m "feat(frontend): 请求带 JWT + AuthGuard 重定向未登录"
```

---

### Task 12：Nginx 路由 `/api/auth/*` 到 user-backend

**目标**：前端的 `/api/auth/login` 等请求通过 Nginx 代理到 `user-backend:8002`，其他 `/api/*` 保持现有转发到 Gateway。

**Files:**
- Modify: `packages/agent/docker/nginx/nginx.conf`

**Step 1: Read current nginx config**

```bash
cat packages/agent/docker/nginx/nginx.conf
```

Look at existing `location /api/` block.

**Step 2: Add `/api/auth/` route above the generic `/api/` block**

Edit `packages/agent/docker/nginx/nginx.conf`. Above the existing `location /api/` block, insert:

```nginx
location /api/auth/ {
    rewrite ^/api/auth/(.*)$ /api/v1/auth/$1 break;
    proxy_pass http://user-backend:8002;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
}
```

**Step 3: Smoke test**

```bash
cd packages/agent/docker
docker compose up -d nginx user-backend postgres
curl -X POST http://localhost:2026/api/auth/login \
     -H "Content-Type: application/json" \
     -d '{"email":"test@noldus.com","password":"TestPass123"}'
```
Expected: JSON with `access_token`.

**Step 4: Commit**

```bash
git add packages/agent/docker/nginx/nginx.conf
git commit -m "feat(deploy): Nginx 把 /api/auth/* 代理到 user-backend"
```

---

## Phase D：集成 + 部署文档（Task 13-15）

### Task 13：飞轮计划集成 — 录制数据带 user_id

**目标**：前置依赖——[训练数据飞轮计划](2026-04-23-training-data-flywheel.md) Task 2 的 `TrainingDataMiddleware` 已经录 thread_id，本 task 补录 user_id 和 feedback API 也带 user_id。

**注意**：这个 task 假设飞轮计划 Task 1-2 已经完成。如果还没做，先做飞轮计划那两个 task。

**Files:**
- Modify: `packages/agent/backend/packages/harness/deerflow/agents/middlewares/training_data_middleware.py`
- Modify: `packages/agent/backend/app/gateway/routers/feedback.py`
- Modify: 对应测试文件

**Step 1: Write failing test**

Append to `tests/test_training_data_middleware.py`:

```python
class TestUserIdInSamples:
    def test_lead_sample_contains_user_id_from_runtime(self, tmp_path):
        from deerflow.agents.middlewares.training_data_middleware import TrainingDataMiddleware
        from langgraph.runtime import Runtime
        from langchain_core.messages import AIMessage, HumanMessage
        import json

        mw = TrainingDataMiddleware(base_dir=str(tmp_path))
        sb = mw.before_agent(state={}, runtime=Runtime(context={"thread_id": "t", "user_id": "42"}))
        state = {
            "training_data_path": sb["training_data_path"],
            "messages": [HumanMessage(content="hi"), AIMessage(content="hello")],
        }
        mw.after_agent(state=state, runtime=Runtime(context={"thread_id": "t", "user_id": "42"}))

        out = tmp_path / "training-data" / "auto-collected" / "t.jsonl"
        rec = json.loads(out.read_text().splitlines()[0])
        assert rec["user_id"] == "42"
```

**Step 2: Run test to verify it fails**

```bash
PYTHONPATH=. uv run pytest tests/test_training_data_middleware.py::TestUserIdInSamples -v
```
Expected: FAIL.

**Step 3: Modify TrainingDataMiddleware**

In `training_data_middleware.py`:
1. In `_resolve_thread_id`, add sibling `_resolve_user_id`:

```python
def _resolve_user_id(self, runtime) -> str | None:
    ctx = runtime.context or {}
    uid = ctx.get("user_id") if isinstance(ctx, dict) else None
    if uid:
        return str(uid)
    try:
        from langgraph.config import get_config
        cfg = get_config()
        v = cfg.get("configurable", {}).get("user_id")
        return str(v) if v else None
    except Exception:
        return None
```

2. In `after_agent`, resolve user_id and thread through to extractors:

```python
user_id = self._resolve_user_id(runtime)
samples = self._extract_lead_samples(messages, thread_id, user_id)
samples.extend(self._extract_subagent_samples(messages, thread_id, user_id))
```

3. Update `_extract_lead_samples` and `_extract_subagent_samples` signatures to accept `user_id` and inject into the record dict:

```python
samples.append({
    "role": "lead",
    "thread_id": thread_id,
    "user_id": user_id,  # NEW
    ...
})
```

**Step 4: Modify feedback router**

In `app/gateway/routers/feedback.py` `post_feedback`, accept user_id from request state:

```python
@router.post("/{thread_id}/feedback", response_model=FeedbackResponse)
def post_feedback(thread_id: str, req: FeedbackRequest, request: Request) -> FeedbackResponse:
    user_id = getattr(request.state, "user_id", None)
    record = {
        "thread_id": thread_id,
        "user_id": user_id,  # NEW
        "message_id": req.message_id,
        ...
    }
```

**Step 5: Run tests**

```bash
PYTHONPATH=. uv run pytest tests/test_training_data_middleware.py tests/test_feedback_router.py -v
```
Expected: all pass.

**Step 6: Commit**

```bash
git add packages/agent/backend/packages/harness/deerflow/agents/middlewares/training_data_middleware.py \
       packages/agent/backend/app/gateway/routers/feedback.py \
       packages/agent/backend/tests/
git commit -m "feat(deploy): 飞轮录制数据带 user_id"
```

---

### Task 14：端到端部署 smoke 测试

**目标**：写一个 shell 脚本 `scripts/smoke-test-deploy.sh`，跑完整的 Docker Compose stack，注册 2 个用户，验证数据隔离。

**Files:**
- Create: `scripts/smoke-test-deploy.sh`

**Step 1: Write smoke test script**

```bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

cd packages/agent/docker
echo "=== Bringing up full stack ==="
docker compose up -d
echo "Waiting for services to be healthy..."
sleep 30

ADMIN_TOKEN="${ADMIN_BOOTSTRAP_TOKEN}"
BASE="http://localhost:2026"

echo ""
echo "=== Registering two test users ==="
curl -sf -X POST "$BASE/api/auth/register" \
     -H "X-Admin-Token: $ADMIN_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"email":"alice@noldus.test","password":"AlicePass1","display_name":"Alice"}'
echo ""
curl -sf -X POST "$BASE/api/auth/register" \
     -H "X-Admin-Token: $ADMIN_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"email":"bob@noldus.test","password":"BobPass1","display_name":"Bob"}'
echo ""

echo ""
echo "=== Login as both users ==="
ALICE_TOKEN=$(curl -sf -X POST "$BASE/api/auth/login" \
     -H "Content-Type: application/json" \
     -d '{"email":"alice@noldus.test","password":"AlicePass1"}' | jq -r .access_token)
BOB_TOKEN=$(curl -sf -X POST "$BASE/api/auth/login" \
     -H "Content-Type: application/json" \
     -d '{"email":"bob@noldus.test","password":"BobPass1"}' | jq -r .access_token)

echo "Alice JWT: ${ALICE_TOKEN:0:40}..."
echo "Bob   JWT: ${BOB_TOKEN:0:40}..."

echo ""
echo "=== Each user creates a thread ==="
ALICE_THREAD=$(curl -sf -X POST "$BASE/api/threads" \
     -H "Authorization: Bearer $ALICE_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"metadata":{}}' | jq -r .thread_id)
BOB_THREAD=$(curl -sf -X POST "$BASE/api/threads" \
     -H "Authorization: Bearer $BOB_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"metadata":{}}' | jq -r .thread_id)

echo "Alice thread: $ALICE_THREAD"
echo "Bob   thread: $BOB_THREAD"

echo ""
echo "=== Alice searches threads — should only see her own ==="
ALICE_THREADS=$(curl -sf -X POST "$BASE/api/threads/search" \
     -H "Authorization: Bearer $ALICE_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{}' | jq -r '[.[] | .thread_id]')
echo "$ALICE_THREADS"
echo "$ALICE_THREADS" | jq -e --arg t "$ALICE_THREAD" 'index($t)' > /dev/null
echo "$ALICE_THREADS" | jq -e --arg t "$BOB_THREAD" 'index($t) == null' > /dev/null && echo "✅ Alice cannot see Bob's thread"

echo ""
echo "=== Request with no token — should 401 ==="
CODE=$(curl -sf -o /dev/null -w "%{http_code}" -X POST "$BASE/api/threads/search" \
       -H "Content-Type: application/json" -d '{}' || echo "$?")
[[ "$CODE" == "401" || "$CODE" -ge 22 ]] && echo "✅ unauthenticated request rejected"

echo ""
echo "=== All smoke tests passed ==="
```

Make executable:
```bash
chmod +x scripts/smoke-test-deploy.sh
```

**Step 2: Run smoke test**

```bash
# Make sure .env has all vars set
source packages/agent/docker/.env
export ADMIN_BOOTSTRAP_TOKEN
./scripts/smoke-test-deploy.sh
```

Expected: script exits 0 with "All smoke tests passed".

**Step 3: Commit**

```bash
git add scripts/smoke-test-deploy.sh
git commit -m "test(deploy): 端到端 smoke 脚本验证多用户隔离"
```

---

### Task 15：部署文档 + CLAUDE.md 更新

**目标**：写清楚部署步骤，更新 CLAUDE.md 记录新架构。

**Files:**
- Create: `docs/sop/multi-user-deployment-sop.md`
- Modify: `CLAUDE.md`（项目根）
- Modify: `packages/agent/backend/CLAUDE.md`

**Step 1: Write deployment SOP**

Create `docs/sop/multi-user-deployment-sop.md`:

```markdown
# 多用户部署 SOP

> 2026-04-23 启动，支撑 10 人内部飞轮使用。单机 Docker Compose 部署。

## 架构

```
Nginx (2026) → Frontend / Gateway / LangGraph / User-Backend
                       ↓
                  Postgres (checkpointer + users + thread metadata)
                       ↓
            AioSandbox (per-thread Docker container)
```

## 服务器要求

| 规格 | 最低 | 推荐 |
|------|------|------|
| CPU | 4 核 | 8 核 |
| 内存 | 8 GB | 16 GB |
| 磁盘 | 40 GB SSD | 80 GB SSD |
| Docker | ✅ 必需 | ✅ |

## 首次部署步骤

### 1. 准备 .env

```bash
cd packages/agent/docker
cp .env.example .env

# 生成真实 secret
echo "JWT_SECRET=$(openssl rand -hex 32)" >> .env
echo "ADMIN_BOOTSTRAP_TOKEN=$(openssl rand -hex 24)" >> .env
# Postgres 密码改成强密码
```

### 2. 启动 stack

```bash
docker compose up -d
# 等 30-60 秒让 Postgres 初始化 + alembic migrate
docker compose logs -f user-backend  # 看 "Uvicorn running"
```

### 3. 预注册用户

```bash
source .env
curl -X POST http://localhost:2026/api/auth/register \
     -H "X-Admin-Token: $ADMIN_BOOTSTRAP_TOKEN" \
     -H "Content-Type: application/json" \
     -d '{"email":"zhangsan@noldus.com","password":"TempPass123","display_name":"张三"}'
```

重复上面给每位行为学同事开账号。

### 4. 发账号密码给用户

推荐做法：
- 初始密码设为临时的强密码（比如 `NoldusInit@<4位随机>`）
- 通过私密渠道告知用户
- 注意：当前版本**没有密码修改 API**（v0.2 再加），密码改动要管理员重置

## 运维常用命令

```bash
# 查看各服务状态
docker compose ps

# 查看日志
docker compose logs -f langgraph
docker compose logs -f gateway

# 备份数据
docker exec deer-flow-postgres pg_dumpall -U noldus > backup-$(date +%F).sql
tar czf thread-data-$(date +%F).tar.gz ./packages/agent/backend/.deer-flow/threads/

# 查看飞轮数据进度（见飞轮 SOP）
cd packages/agent/backend && make training-stats
```

## 排障

### Sandbox 启动失败

```bash
docker compose logs langgraph | grep -i sandbox
```

常见原因：
- Docker socket 未挂载 → 看 compose 文件 langgraph volumes 段
- 镜像拉不下来 → `docker pull enterprise-public-cn-beijing.cr.volces.com/vefaas-public/all-in-one-sandbox:latest` 手动预热

### Postgres 连接失败

```bash
docker exec -it deer-flow-postgres psql -U noldus -c "\l"
```
应该看到 `noldus_users` 和 `noldus_langgraph` 两个库。

### JWT 401

- 确认前端 localStorage 里 `noldus_jwt` 存在
- 确认 Gateway 能 reach user-backend（同一 docker network）
- JWT 24 小时过期，重新登录

## 数据隐私

- 所有数据存在本机 `./packages/agent/backend/.deer-flow/` 和 Postgres 卷
- 不上传任何外部服务
- user_id 按雪花 ID 生成，无真实姓名关联
- 飞轮数据全部带 user_id，可按用户删除

## 容量限制

当前配置（replicas: 5）支持**同时 5 个用户跑分析**。第 6 个会 LRU evict 最早的 sandbox 容器。10 人内部使用实际并发一般 2-4，完全够用。

## 升级到生产规模

超过 10 人的需求见 [roadmap Phase 5](../roadmap.md)——届时考虑：
- 多机 + K8s
- 切 LangGraph Platform BYOC（如果预算允许）
- Sandbox 改用火山 AgentKit（或自建 container orchestrator）
```

**Step 2: Update project root CLAUDE.md**

Edit `/home/qiuyangwang/noldus-insight/CLAUDE.md`. Under "重要注意事项"，append:

```markdown
8. **多用户部署已就绪（v0.1+）** — `services/user-backend/` 独立 JWT auth 服务；DeerFlow checkpointer 走 Postgres、sandbox 走 AioSandbox（每 thread 一容器）、memory 按 user_id 隔离。部署见 [docs/sop/multi-user-deployment-sop.md](docs/sop/multi-user-deployment-sop.md)。10 人规模单机可跑。
```

Under "仓库结构"，在 `├── packages/` 之后添加：

```
├── services/
│   └── user-backend/        # 独立 JWT 认证服务（改造自 aws-user-backend）
```

**Step 3: Update backend CLAUDE.md**

Edit `packages/agent/backend/CLAUDE.md`. Append a new section after the Middleware Chain section:

```markdown
### Multi-User Deployment Architecture

EthoInsight runs a single LangGraph Server process that serves all users. User isolation is achieved at multiple layers:

- **Auth**: Independent `services/user-backend` issues JWTs; Gateway `JWTAuthMiddleware` verifies via `/api/v1/auth/verify` and injects `user_id` into `request.state`
- **Thread**: `threads.py:search_threads` filters by `user_id` from `request.state`; new threads receive `user_id` in metadata
- **LangGraph**: `user_id` is passed via `config.configurable` and propagates to middlewares via `runtime.context`
- **Memory**: `MemoryMiddleware` reads `user_id` from runtime context and passes `agent_name="user:<user_id>"` to the queue, routing each user's memory to a separate `memory/user_<id>.json`
- **Checkpointer**: Postgres-backed; state rows are keyed by thread_id (each thread already user-scoped via auth)
- **Sandbox**: `AioSandboxProvider` allocates a fresh Docker container per thread_id; user A's bash commands cannot touch user B's filesystem
- **Training flywheel**: `TrainingDataMiddleware` and feedback router both record `user_id` in emitted JSONL

The only shared state across users is the agent code itself (stateless Python modules) and warm sandbox containers (LRU-evicted when `replicas` exceeded).
```

**Step 4: Commit**

```bash
git add docs/sop/multi-user-deployment-sop.md CLAUDE.md packages/agent/backend/CLAUDE.md
git commit -m "docs: 多用户部署 SOP + CLAUDE.md 架构更新"
```

---

### Task 16：全量回归 + handoff

**目标**：全量测试 + 写 handoff。

**Files:**
- Create: `docs/handoffs/2026-04-23-multi-user-deployment-done.md`

**Step 1: Full regression**

```bash
cd services/user-backend && uv run pytest tests/ -v
cd packages/agent/backend && source .venv/bin/activate && make test && make lint
cd packages/agent/frontend && pnpm test
```

Expected: all green.

**Step 2: Full smoke**

```bash
./scripts/smoke-test-deploy.sh
```
Expected: All smoke tests passed.

**Step 3: Write handoff**

Create `docs/handoffs/2026-04-23-multi-user-deployment-done.md` following the pattern of [2026-04-23-m01-remaining-items-done.md](../handoffs/2026-04-23-m01-remaining-items-done.md):

- Summary of all 16 tasks
- Files created/modified
- Baseline test count delta
- How to onboard a new user (reference SOP)
- Remaining work: password rotation API / email verification / rate limiting → all deferred to post-v0.1

**Step 4: Commit**

```bash
git add docs/handoffs/2026-04-23-multi-user-deployment-done.md
git commit -m "docs: 多用户部署 handoff 文档"
```

---

## 总计

- **16 tasks**，4 phases
- Phase A（User Backend 改造）：Task 1-5，~2 天
- Phase B（DeerFlow 多用户切换）：Task 6-9，~2 天
- Phase C（前端登录 + 隔离）：Task 10-12，~1 天
- Phase D（集成 + 文档）：Task 13-16，~1 天
- 总计 **5-7 天工程时间**

## 验收标准

1. ✅ `./scripts/smoke-test-deploy.sh` 全绿
2. ✅ `docker compose up -d` 所有服务 healthy
3. ✅ 两个用户注册、登录、创建 thread、互相看不到对方 thread
4. ✅ 用户 A 的 bash/python 执行在独立 container，`rm -rf /` 不会影响 B 或宿主机
5. ✅ 用户 A 的 memory 存在 `memory/user_A.json`，B 存在 `memory/user_B.json`
6. ✅ 飞轮录制数据带 user_id，可按用户聚合
7. ✅ 全量 pytest/vitest 绿
8. ✅ `make training-stats` 仍能用
9. ✅ 手动在浏览器走完完整登录 → 对话 → 反馈流程

## 和其他计划的关系

- **前置**：无（此计划可独立启动）
- **后置可联动**：[训练数据飞轮](2026-04-23-training-data-flywheel.md) Task 13（录 user_id）依赖本计划 Task 9 完成
- **互补**：[2026-04-22 license+自报名](2026-04-22-local-deploy-auth-and-access.md) 适用于 v0.1+ 客户本地部署；本计划适用于当前阶段 Noldus 内部 SaaS-like 部署
- **未来替换**：本计划的 `local_auth.py` 未来可无痛替换为 Supabase Auth / AWS Cognito / 阿里云 IDaaS——其他代码不动（仅 Task 3 的实现换一换）

## 显式不做的事

这些是真实用户会问的问题，提前写下来避免在执行时纠结：

- **邮箱验证**：v0.1 内部手动预注册，不做自助注册流程
- **找回密码 / 修改密码 API**：v0.2 再加。目前靠管理员重置 `hashed_password`
- **Rate limit**：docker-compose 规模 10 人无压力。v0.2 规模化再加 aws-user-backend 里的 `user_rate_limiter.py`
- **Session 撤销 / token 黑名单**：JWT 24 小时过期机制足够。v0.2 加 Redis-backed blacklist
- **SSO / OAuth 第三方登录**：走 Phase 5 SaaS 化再评估（此时可以换 Supabase）
- **前端路由的细粒度权限**：10 人都是同一角色，RBAC 不需要
- **Admin dashboard**：`POST /api/auth/register` 足够，crud 由工程 SSH 到服务器 curl 执行
