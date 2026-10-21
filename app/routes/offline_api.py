"""离线规则只读视图；下载提交统一进入 download_requests/dispatcher。"""
from __future__ import annotations

from fastapi import APIRouter, Request

from app.modules.offline import rules_summary
from app.web import require_api_login

router = APIRouter(prefix="/api/offline")


@router.get("/rules")
def get_rules(request: Request):
    require_api_login(request)
    return rules_summary()
