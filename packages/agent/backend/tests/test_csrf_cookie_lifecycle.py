"""CSRF cookie 生命周期回归测试。

根因（2026-06-11 线上 dev 403 实证）：csrf_token 曾是 session cookie（无 max_age），
而 access_token 持久（token_expiry_days）。二者生命周期不对称 → 用户用一阵后
（关标签页 / 浏览器回收 session cookie）csrf_token 失效但 access_token 仍在 →
前端 readCsrfCookie() 返回 null → POST 不带 X-CSRF-Token → 403。

修复：csrf cookie 与 access_token 完全对齐——
- https: max_age = token_expiry_days * 24 * 3600（持久化，二者同生命周期）
- http : max_age = None（session cookie，与 access_token 在 http 下一致）
- samesite = "lax"（与 access_token 一致；strict 在跨页导航进入时不发 cookie，是第二诱因）

这两点都以 access_token 的 _set_session_cookie（routers/auth.py）为对齐基准。
"""

from __future__ import annotations

from http.cookies import SimpleCookie

from app.gateway.auth.config import get_auth_config
from app.gateway.csrf_middleware import CSRF_COOKIE_NAME, _set_csrf_cookie
from fastapi import Request, Response


def _make_request(scheme: str = "https", host: str = "dev.ethoinsight.com") -> Request:
    """构造一个最小 ASGI scope 的 Request，scheme 经 x-forwarded-proto 控制。"""
    headers = [(b"host", host.encode()), (b"x-forwarded-proto", scheme.encode())]
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/models",
        "headers": headers,
        "scheme": scheme,
        "server": (host, 443 if scheme == "https" else 80),
        "query_string": b"",
    }
    return Request(scope)


def _extract_csrf_morsel(response: Response):
    """从 response 的 set-cookie 头里取出 csrf_token morsel。"""
    jar: SimpleCookie = SimpleCookie()
    for header_key, header_val in response.raw_headers:
        if header_key == b"set-cookie":
            jar.load(header_val.decode())
    assert CSRF_COOKIE_NAME in jar, f"未设置 {CSRF_COOKIE_NAME} cookie"
    return jar[CSRF_COOKIE_NAME]


def test_csrf_cookie_persistent_over_https():
    """https 下 csrf cookie 必须持久化，max_age 与 access_token 同（token_expiry_days）。"""
    response = Response()
    _set_csrf_cookie(response, _make_request(scheme="https"))
    morsel = _extract_csrf_morsel(response)

    expected_max_age = get_auth_config().token_expiry_days * 24 * 3600
    assert morsel["max-age"] == str(expected_max_age), (
        f"csrf cookie max-age 应为 {expected_max_age}（与 access_token 对齐），"
        f"实际 {morsel['max-age']!r}。session cookie（空 max-age）会随浏览器关闭失效，"
        f"导致 access_token 仍在但 csrf 丢失 → 403。"
    )


def test_csrf_cookie_samesite_lax():
    """csrf cookie 的 SameSite 应为 lax，与 access_token 一致（strict 跨页导航不发 cookie）。"""
    response = Response()
    _set_csrf_cookie(response, _make_request(scheme="https"))
    morsel = _extract_csrf_morsel(response)
    assert morsel["samesite"].lower() == "lax", (
        f"csrf cookie samesite 应为 lax，实际 {morsel['samesite']!r}"
    )


def test_csrf_cookie_secure_over_https():
    """https 下 csrf cookie 必须带 Secure 标志。"""
    response = Response()
    _set_csrf_cookie(response, _make_request(scheme="https"))
    morsel = _extract_csrf_morsel(response)
    assert morsel["secure"], "https 下 csrf cookie 应带 Secure"


def test_csrf_cookie_session_over_http():
    """http 下 csrf cookie 为 session cookie（max_age=None），与 access_token 在 http 下语义一致。

    本地 http dev 下 access_token 也是 session cookie（max_age=None if not is_https），
    csrf 必须同步，避免 http/https 行为分叉。
    """
    response = Response()
    _set_csrf_cookie(response, _make_request(scheme="http"))
    morsel = _extract_csrf_morsel(response)
    # http 下不应持久化（max-age 为空），也不应带 Secure
    assert not morsel["max-age"], f"http 下 csrf cookie 不应有 max-age，实际 {morsel['max-age']!r}"
    assert not morsel["secure"], "http 下 csrf cookie 不应带 Secure"


def test_csrf_cookie_js_readable():
    """csrf cookie 必须 JS 可读（非 httponly），Double Submit Cookie 模式要求前端能读取。"""
    response = Response()
    _set_csrf_cookie(response, _make_request(scheme="https"))
    morsel = _extract_csrf_morsel(response)
    assert not morsel["httponly"], "csrf cookie 不能是 httponly（前端需 readCsrfCookie 读取注入 X-CSRF-Token）"
