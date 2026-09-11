# Server-rendered pages.
#
# Jinja renders the shell and the first paint; app.js then keeps the numbers alive over SSE. That
# split means a cold page load is never blank, and it keeps the whole front-end to two static
# files with no build step - no bundler, no node_modules, nothing to rebuild on the box.
#
# Page routes redirect when unauthenticated; API routes 401. Hiding a nav link is presentation.
# The control is app/auth/deps.py, applied at the router.
from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.responses import Response

from app.auth import ratelimit, users
from app.auth.deps import current_session, get_db, get_settings
from app.auth.passwords import MIN_PASSWORD_LEN, PasswordError, verify_password
from app.auth.sessions import CSRF_HEADER, Session, create_session, drop_session, load_session
from app.db import audit
from app.db.bootstrap import connect_sync
from app.settings import Settings

router = APIRouter(tags=["pages"])

TEMPLATES_DIR = Path(__file__).parent / "templates"

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _render(request: Request, name: str, context: dict[str, Any], status_code: int = 200):
    session: Session | None = context.get("session")
    settings: Settings = request.app.state.settings
    services = getattr(request.app.state, "services", None)
    return templates.TemplateResponse(
        name,
        {
            **context,
            "request": request,
            "session": session,
            "is_admin": bool(session and session.is_admin),
            "csrf_header": CSRF_HEADER,
            "csrf_token": session.csrf if session else "",
            "settings": settings,
            "docker_ok": bool(services and services.docker_ok),
            "docker_reason": services.docker_reason if services else "",
            "min_password_len": MIN_PASSWORD_LEN,
            "unreadable_admin_count": context.get("unreadable_admin_count", 0),
        },
        status_code=status_code,
    )


@router.get("/login")
async def login_page(
    request: Request,
    session: Session | None = Depends(current_session),
):
    if session is not None:
        return RedirectResponse("/", status_code=303)
    return _render(request, "login.html", {"session": None, "error": ""})


@router.post("/login")
async def login(
    request: Request,
    db=Depends(get_db),
    settings: Settings = Depends(get_settings),
    username: str = Form(""),
    password: str = Form(""),
):
    """Form post rather than JSON, so the login page needs no JavaScript at all."""
    limiter: ratelimit.RateLimiter = request.app.state.login_limiter
    client_ip = request.client.host if request.client else "?"
    username = (username or "").strip().lower()
    ip_key, user_key = f"ip:{client_ip}", f"user:{username or '-'}"
    allowed = limiter.allow(ip_key) and limiter.allow(user_key)

    error = ""
    if not allowed:
        # Checked before the DB lookup on purpose: a locked-out client must not get a valid
        # credential checked, or rate limiting becomes a way to probe passwords.
        wait = max(limiter.retry_after_s(ip_key), limiter.retry_after_s(user_key))
        error = f"too many attempts. Try again in {wait}s."
    else:
        row = await users.by_username(db, username) if username else None
        if row is None or not row["disabled"]:
            # Run the hash check even for an unknown username, so response time does not reveal
            # whether an account exists.
            if row is not None and password:
                try:
                    ok = verify_password(password, row["pass_hash"])
                except PasswordError:
                    ok = False
            else:
                ok = False
            if not ok:
                error = "wrong username or password."
            else:
                if row["disabled"]:
                    error = "that account is disabled."
                else:
                    session_id = await create_session(
                        db, user_id=row["id"], ip=client_ip, ttl_hours=settings.session_ttl_hours
                    )
                    await users.touch_login(db, row["id"])
                    await users.upgrade_hash_if_needed(db, row["id"], password, row["pass_hash"])
                    await audit.record(db, actor=username, action="login", ok=True, detail=client_ip)
                    limiter.clear(user_key)
                    response = RedirectResponse("/", status_code=303)
                    response.set_cookie(
                        "llmrunner_session",
                        session_id,
                        httponly=True,
                        samesite="lax",
                        path="/",
                        max_age=settings.session_ttl_hours * 3600,
                        # No `secure`: the default bind is loopback and TLS is expected to terminate
                        # at a reverse proxy. Set it if you ever serve plain HTTP on a LAN.
                    )
                    return response
        else:
            error = "wrong username or password."

    await audit.record(db, actor=username or "?", action="login", ok=False, detail=error)
    return _render(
        request, "login.html", {"session": None, "error": error}, status_code=401
    )


@router.post("/logout")
async def logout(
    request: Request,
    db=Depends(get_db),
):
    session: Session | None = await current_session(request, db, request.app.state.settings)
    if session is not None:
        await drop_session(db, session.id)
        await audit.record(db, actor=session.username, action="logout", ok=True)
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie("llmrunner_session", path="/")
    return response


@router.get("/")
async def dashboard(
    request: Request,
    session: Session | None = Depends(current_session),
):
    if session is None:
        return RedirectResponse("/login", status_code=303)
    return _render(request, "dashboard.html", {"session": session, "active": "dashboard"})


@router.get("/models")
async def models_page(
    request: Request,
    session: Session | None = Depends(current_session),
):
    if session is None:
        return RedirectResponse("/login", status_code=303)
    from app.db import catalog as catalogue

    entries = await catalogue.list_all(request.app.state.db)
    return _render(
        request, "models.html",
        {"session": session, "active": "models", "entries": entries},
    )


@router.get("/models/{entry_id}")
async def model_detail(
    entry_id: int,
    request: Request,
    session: Session | None = Depends(current_session),
):
    if session is None:
        return RedirectResponse("/login", status_code=303)
    from app.db import catalog as catalogue

    entry = await catalogue.get(request.app.state.db, entry_id)
    if entry is None:
        return HTMLResponse("<h1>404</h1><p>No such model.</p>", status_code=404)
    return _render(
        request, "model_detail.html",
        {"session": session, "active": "models", "entry": entry},
    )


@router.get("/containers")
async def containers_page(
    request: Request,
    session: Session | None = Depends(current_session),
):
    if session is None:
        return RedirectResponse("/login", status_code=303)
    return _render(request, "containers.html", {"session": session, "active": "containers"})


@router.get("/admin")
async def admin_page(
    request: Request,
    session: Session | None = Depends(current_session),
):
    if session is None:
        return RedirectResponse("/login", status_code=303)
    if not session.is_admin:
        return _render(
            request, "forbidden.html",
            {"session": session, "active": "admin", "what": "the admin area"},
            status_code=403,
        )
    from app.db import audit as audit_log

    return _render(
        request, "admin.html",
        {
            "session": session,
            "active": "admin",
            "users": await users.list_users(request.app.state.db),
            "audit": (await audit_log.recent(request.app.state.db, limit=60))[:60],
        },
    )


@router.get("/system")
async def system_page(
    request: Request,
    session: Session | None = Depends(current_session),
):
    """Probe report and recent audit trail. Read-only for any signed-in user."""
    if session is None:
        return RedirectResponse("/login", status_code=303)
    from app.db import audit as audit_log

    return _render(
        request, "system.html",
        {
            "session": session,
            "active": "system",
            "audit": (await audit_log.recent(request.app.state.db, limit=40))[:40],
        },
    )
