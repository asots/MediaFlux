"""发布格式教学的 Agent 薄动作适配。"""
from __future__ import annotations

from typing import Any

from app.agent.errors import AgentToolError
from app.agent.models import ToolResult
from app.agent.public_safety import sanitize_public_text, sanitize_resource_title
from app.modules.recognition import formats

_AGENT_FIELDS = {"draft", "examples", "filenames"}


def _safe(value: object, fallback: str, limit: int = 240) -> str:
    return sanitize_public_text(value, limit=limit) or fallback


def teaching_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise AgentToolError("发布格式教学参数必须是对象")
    if set(arguments) - _AGENT_FIELDS:
        raise AgentToolError(
            "请只提供格式草稿、标注样本和待预览文件；确认信息由系统处理"
        )
    try:
        return formats.normalize_request(arguments)
    except (TypeError, ValueError) as exc:
        raise AgentToolError(_safe(str(exc), "发布格式教学参数无效")) from exc


def _resource(row: dict[str, Any]) -> dict[str, Any]:
    def point(value: dict[str, Any]) -> tuple[dict[str, Any], str]:
        title = sanitize_resource_title(value["title"], limit=180) or "未识别标题"
        season, episode = value["season"], value["episode"]
        if season is not None and episode is not None:
            label = f"第{season}季第{episode}集"
        elif episode is not None:
            label = f"第{episode}集"
        elif season is not None:
            label = f"第{season}季，集号未识别"
        else:
            label = "季集未识别"
        return {"title": title, "season": season, "episode": episode}, label

    filename = sanitize_resource_title(row["filename"], limit=255) or "未命名样本文件"
    before, before_label = point(row["before"])
    after, after_label = point(row["after"])
    return {
        # 各字段已经净化且有界；不再用工具名过滤器重扫文件名，或截掉末尾的集号。
        "title": (f"季集 {before_label} → {after_label}；标题《{before['title']}》→《{after['title']}》；"
                  f"样本文件「{filename}」"),
        "filename": filename,
        "before": before,
        "after": after,
        "status": _safe(row["status"], "未确定", 40),
        "passed": bool(row["passed"]),
    }


def _public_preview(preview: dict[str, Any], confirmation: bool = False) -> ToolResult:
    summary, examples, rows = preview["summary"], preview["examples"], preview["rows"]
    batch_total, sample_count = summary["total"], len(examples)
    display_total = batch_total or sample_count
    matched, unmatched = summary["matched"], summary["unmatched"]
    blocked, conflicts = summary["blocked"], summary["conflicts"]
    review_required = unmatched + blocked + conflicts
    scope = "仅此目录" if preview["draft"]["scope"] == "directory" else "跨目录发布范围"
    batch_effect = (f"批量核对 {batch_total} 个文件：命中 {matched} 个，未匹配 {unmatched} 个，特别篇/受保护 {blocked} 个，格式冲突 {conflicts} 个；这些文件仍走原有识别流程。"
                    if batch_total else
                    f"已核对 {sample_count} 个标注样本；未提供额外批量文件，特别篇/受保护 {blocked} 个、格式冲突 {conflicts} 个仍走原有识别流程。")
    data = {
        "resources": [_resource(row) for row in examples[:8]],
        "effects": [f"适用范围：{scope}；只影响以后识别。", batch_effect, "不移动文件、不绑定 TMDB、不偏移季集编号。"],
        "total": display_total, "count": display_total, "batch_total": batch_total,
        "sample_count": sample_count, "matched": matched, "unmatched": unmatched,
        "review_required": review_required, "can_save": preview["can_save"],
        "summary": {**summary, "total": display_total, "batch_total": batch_total, "sample_count": sample_count},
    }
    if not preview["can_save"]:
        suggestions = [_safe(item, "请核对样本和适用范围") for item in preview["warnings"][:4]] or ["请核对样本、标题、季集和适用范围后重新预览。"]
        status, text = "attention", (f"发布格式预览完成，但有 {review_required} 个批量结果需人工核对"
                                     if review_required else "发布格式预览完成，但标注样本尚未通过保存前置校验")
    elif confirmation:
        suggestions = ["请核对样本前后标题与季集；确认后才会保存规则。"]
        status, text = "confirmation_required", (f"确认后保存发布格式教学规则，批量核对 {batch_total} 个文件"
                                                  if batch_total else f"确认后保存发布格式教学规则，已核对 {sample_count} 个标注样本")
    else:
        suggestions = ["这是只读预览；如需记住此格式，请另行请求保存并确认。"]
        status, text = "preview", (f"发布格式预览完成：命中 {matched} 个，未匹配 {unmatched} 个"
                                    if batch_total else f"发布格式预览完成：已核对 {sample_count} 个标注样本")
    return ToolResult(ok=True, status=status, summary=text, data=data, model_data={
        "draft": preview["draft"], "examples": examples, "rows": rows,
        "summary": summary, "warnings": preview["warnings"], "can_save": preview["can_save"],
    }, suggestions=suggestions)


def preview_release_format(arguments: dict[str, Any]) -> ToolResult:
    return _public_preview(formats.preview(teaching_arguments(arguments)))


def prepare_release_format(arguments: dict[str, Any]) -> tuple[ToolResult, str]:
    preview = formats.preview(teaching_arguments(arguments))
    if not preview["can_save"]:
        raise AgentToolError(_safe(preview["warnings"][0] if preview["warnings"] else "发布格式预览未通过，不能创建保存确认", "发布格式预览未通过，不能创建保存确认"), code="precondition_failed")
    token = preview["preview_token"]
    if not isinstance(token, str) or not token:
        raise AgentToolError("发布格式预览未生成有效确认票据", code="precondition_failed")
    return _public_preview(preview, True), token


def save_release_format_confirmed(arguments: dict[str, Any], token: str) -> ToolResult:
    normalized = teaching_arguments(arguments)
    if not isinstance(token, str) or not token:
        raise AgentToolError("发布格式确认票据无效，请重新预检", code="confirmation_stale")
    try:
        item, created = formats.save({**normalized, "confirmed": True, "preview_token": token})
    except formats.FormatConflict as exc:
        raise AgentToolError(_safe(str(exc), "发布格式预览已失效，请重新预览"), code="confirmation_stale") from exc
    except (TypeError, ValueError) as exc:
        raise AgentToolError(_safe(str(exc), "发布格式确认无法执行"), code="precondition_failed") from exc
    disabled = bool(item["disabled"])
    enabled, duplicate = not disabled, not created
    if created:
        text = "发布格式教学规则已保存并启用" if enabled else "发布格式教学规则已保存，但当前未启用"
    elif disabled:
        text = "相同发布格式规则已存在且已停用，本次未重新启用"
    else:
        text = "相同发布格式规则已存在且保持启用，未重复创建"
    data = {"resources": [], "effects": ["只保存以后识别使用的字段规则。", "不移动文件、不绑定 TMDB、不偏移季集编号。"],
            "total": 1, "count": 1, "created": bool(created), "duplicate": duplicate,
            "enabled": enabled, "review_required": 0,
            "summary": {"created": bool(created), "duplicate": duplicate, "enabled": enabled}}
    return ToolResult(True, "success", text, data=data,
                      model_data={"draft": normalized["draft"], "examples": normalized["examples"], **data})
