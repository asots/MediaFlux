"""FastAPI 认证与首次初始化路由。"""
from __future__ import annotations

import hmac

from fastapi import APIRouter, Form, Request
from fastapi.responses import RedirectResponse

from app.config import web_credentials
from app.logger import get_logger
from app.modules import first_run
from app.modules.login_wallpaper import get_login_wallpaper
from app.security import (
    clear_login_failures,
    login_rate_limited,
    record_login_failure,
)
from app.web import render_template

logger = get_logger(__name__)
router = APIRouter()

def _render_login(request: Request, *, status_code: int = 200, error: str = ""):
    return render_template(
        request,
        "login.html",
        status_code=status_code,
        error=error,
        login_wallpaper=get_login_wallpaper(),
        setup_mode=False,
    )


def render_setup(
    request: Request,
    *,
    status_code: int = 200,
    error: str = "",
    username: str = "",
):
    """复用登录布局渲染首启页；故意不请求 TMDB wallpaper。"""
    return render_template(
        request,
        "login.html",
        status_code=status_code,
        error=error,
        login_wallpaper=None,
        setup_mode=True,
        username=username,
    )


@router.get("/login", name="auth.login")
def login_page(request: Request):
    if first_run.needs_initialization():
        return RedirectResponse("/setup", status_code=302)
    if request.session.get("logged_in"):
        return RedirectResponse("/", status_code=302)
    return _render_login(request)


@router.post("/login", name="auth.login_submit")
def login_submit(
    request: Request,
    username: str = Form(default=""),
    password: str = Form(default=""),
):
    if first_run.needs_initialization():
        return RedirectResponse("/setup", status_code=302)
    if request.session.get("logged_in"):
        return RedirectResponse("/", status_code=302)
    identity = request.client.host if request.client else "unknown"
    if login_rate_limited(identity):
        return _render_login(
            request,
            status_code=429,
            error="登录失败次数过多，请稍后再试",
        )
    username = username[:128]
    password = password[:512]
    ok_user, ok_pass = web_credentials()
    credentials_ready = bool(ok_user and ok_pass)
    valid_user = credentials_ready and hmac.compare_digest(
        username.encode("utf-8"), ok_user.encode("utf-8")
    )
    valid_password = credentials_ready and hmac.compare_digest(
        password.encode("utf-8"), ok_pass.encode("utf-8")
    )
    valid = valid_user and valid_password
    if valid:
        clear_login_failures(identity)
        request.session.clear()
        request.session["logged_in"] = True
        logger.info(f"用户登录成功: {username}")
        return RedirectResponse("/", status_code=302)
    record_login_failure(identity)
    return _render_login(
        request,
        status_code=401,
        error="用户名或密码错误",
    )


@router.get("/setup", name="auth.setup")
def setup_page(request: Request):
    if not first_run.needs_initialization():
        return RedirectResponse("/login", status_code=302)
    return render_setup(request, error=first_run.initialization_error())



@router.post("/setup", name="auth.setup_submit")
def setup_submit(
    request: Request,
    username: str = Form(default=""),
    password: str = Form(default=""),
    password_confirm: str = Form(default=""),
):
    if not first_run.needs_initialization():
        return RedirectResponse("/login", status_code=302)

    from fastapi.responses import PlainTextResponse
    return PlainTextResponse("停止服务并显式运行 initialize；完成后重启登录", status_code=403)


@router.post("/logout", name="auth.logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/login", status_code=302)
