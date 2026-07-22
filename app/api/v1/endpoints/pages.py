"""
NovaShield - HTML Page Routes (Jinja2 Templates)
Serves the futuristic frontend via FastAPI
"""

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import os

router = APIRouter()

# Resolve templates directory relative to this file's location
_HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.normpath(os.path.join(_HERE, "../../../../templates"))
templates = Jinja2Templates(directory=TEMPLATES_DIR)


def _resp(request: Request, name: str, ctx: dict = None):
    """Compatibility wrapper for Starlette 1.x TemplateResponse."""
    context = ctx or {}
    # Starlette >= 0.36 / 1.x: positional (request, name, context)
    try:
        return templates.TemplateResponse(request, name, context)
    except TypeError:
        # Fallback for older Starlette: TemplateResponse(name, {"request":...})
        context["request"] = request
        return templates.TemplateResponse(name, context)


# ── Public Pages ───────────────────────────────────────────────────────────────

@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def landing(request: Request):
    return _resp(request, "landing.html")


@router.get("/login", response_class=HTMLResponse, include_in_schema=False)
async def login_page(request: Request):
    return _resp(request, "login.html")


@router.get("/register", response_class=HTMLResponse, include_in_schema=False)
async def register_page(request: Request):
    return _resp(request, "register.html")


@router.get("/complaint", response_class=HTMLResponse, include_in_schema=False)
async def complaint_page(request: Request):
    return _resp(request, "complaint.html")


@router.get("/track", response_class=HTMLResponse, include_in_schema=False)
async def track_page(request: Request):
    return _resp(request, "track.html")


@router.get("/analytics", response_class=HTMLResponse, include_in_schema=False)
async def analytics_page(request: Request):
    return _resp(request, "analytics.html")


@router.get("/fir", response_class=HTMLResponse, include_in_schema=False)
async def fir_page(request: Request):
    return _resp(request, "fir.html")


# ── Protected Pages (auth checked client-side via /api/v1/auth/me) ─────────────

@router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
async def dashboard_page(request: Request):
    return _resp(request, "dashboard.html")


@router.get("/dashboard/detect", response_class=HTMLResponse, include_in_schema=False)
async def detect_page(request: Request):
    return _resp(request, "detect.html")


@router.get("/dashboard/history", response_class=HTMLResponse, include_in_schema=False)
async def history_page(request: Request):
    return _resp(request, "history.html")


@router.get("/dashboard/admin", response_class=HTMLResponse, include_in_schema=False)
async def admin_page(request: Request):
    return _resp(request, "admin.html")
