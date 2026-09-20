"""local_media 领域的 Agent 原子工具声明。"""

from __future__ import annotations

from app.agent.local_media_actions import (
    diagnose_local_media,
    local_media_diagnosis_arguments,
    local_media_history_arguments,
    local_media_review_queue_arguments,
    summarize_local_media_history,
    summarize_local_media_review_queue,
)
from app.agent.local_media_scan_actions import (
    local_media_scan_arguments,
    prepare_scan_local_media_sources,
    scan_local_media_sources_confirmed,
)
from app.agent.local_media_source_actions import (
    get_local_media_source_summary,
    list_local_media_source_summaries,
    local_media_source_summaries_arguments,
    local_media_source_summary_arguments,
    local_media_source_trigger_arguments,
    prepare_set_local_media_source_trigger_enabled,
    set_local_media_source_trigger_enabled_confirmed,
)
from app.agent.local_media_task_actions import (
    inspect_local_media_task,
    list_local_media_task_summaries,
    local_media_inspection_arguments,
    local_media_retry_arguments,
    local_media_task_number_arguments,
    local_media_task_summaries_arguments,
    prepare_refresh_local_media_task_library,
    prepare_retry_local_media_task,
    preview_local_media_task,
    refresh_local_media_task_library_confirmed,
    retry_local_media_task_confirmed,
    verify_local_media_task_library_visibility,
)
from app.agent.models import (
    RiskLevel,
    ToolSpec,
)
from app.modules.local_media_models import LOCAL_TASK_STATUSES


def register_specs(
    registry, *, resource_store, active_ingest_store, ingest_actions
) -> None:
    registry.register(
        ToolSpec(
            name="local_media.diagnose",
            description="只读汇总本地媒体来源、整理任务与调度器状态，不扫描文件系统、不访问外部服务且不返回路径或业务标识。",
            risk=RiskLevel.READ,
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            handler=diagnose_local_media,
            validator=local_media_diagnosis_arguments,
            examples=(
                "本地媒体来源和整理调度正常吗",
                "检查本地媒体自动化状态",
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="local_media.source_summaries",
            description="只读列出本地媒体来源的公开序号、触发状态和安全配置摘要，不返回名称、路径、媒体库标识或凭据。",
            risk=RiskLevel.READ,
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            handler=list_local_media_source_summaries,
            validator=local_media_source_summaries_arguments,
        )
    )
    registry.register(
        ToolSpec(
            name="local_media.get_source_summary",
            description="只读查看一个公开序号对应的本地媒体来源触发状态与安全摘要，不返回名称、路径、媒体库标识或凭据。",
            risk=RiskLevel.READ,
            parameters={
                "type": "object",
                "required": ["source_number"],
                "properties": {"source_number": {"type": "integer", "minimum": 1}},
                "additionalProperties": False,
            },
            handler=get_local_media_source_summary,
            validator=local_media_source_summary_arguments,
        )
    )
    registry.register(
        ToolSpec(
            name="local_media.set_source_trigger_enabled",
            description="确认后精确启停一个本地媒体来源的 qB 下载完成自动接管；不修改目录、规则、目标或凭据。",
            risk=RiskLevel.LOW_WRITE,
            parameters={
                "type": "object",
                "required": ["source_number", "trigger", "enabled"],
                "properties": {
                    "source_number": {"type": "integer", "minimum": 1},
                    "trigger": {"type": "string", "enum": ["qb_completed"]},
                    "enabled": {"type": "boolean"},
                },
                "additionalProperties": False,
            },
            validator=local_media_source_trigger_arguments,
            requires_confirmation=True,
            context_confirmed_handler=ToolSpec.context_free_confirmed_handler(
                set_local_media_source_trigger_enabled_confirmed
            ),
            context_confirmation_preparer=ToolSpec.context_free_confirmation_preparer(
                prepare_set_local_media_source_trigger_enabled
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="local_media.scan_sources",
            description="预检并确认后扫描全部或指定公开序号的已配置本地媒体来源，把发现的媒体加入整理队列；不接受任意路径。",
            risk=RiskLevel.LOW_WRITE,
            parameters={
                "type": "object",
                "properties": {
                    "source_numbers": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 1, "maximum": 10000},
                        "maxItems": 20,
                    },
                    "query": {"type": "string", "maxLength": 120},
                },
                "additionalProperties": False,
            },
            validator=local_media_scan_arguments,
            requires_confirmation=True,
            context_confirmation_preparer=ToolSpec.context_free_confirmation_preparer(
                prepare_scan_local_media_sources
            ),
            context_confirmed_handler=ToolSpec.context_free_confirmed_handler(
                scan_local_media_sources_confirmed
            ),
            domains=("local_media", "organize"),
            source_kind="system_state",
            examples=("扫描全部本地媒体来源", "扫描本地媒体来源 2"),
        )
    )
    registry.register(
        ToolSpec(
            name="local_media.review_queue_summary",
            description="只读汇总本地媒体待人工确认队列的数量、触发来源和等待时长，不返回标题、路径、任务标识或错误正文。",
            risk=RiskLevel.READ,
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            handler=summarize_local_media_review_queue,
            validator=local_media_review_queue_arguments,
        )
    )
    registry.register(
        ToolSpec(
            name="local_media.task_summaries",
            description="只读列出本地媒体任务的原文件名、时间、阶段和文件归档/冲突跳过结果。任务 completed 不等于视频入库；task_number 是本次列表短期序号，不是数据库 ID。询问整理通知里哪个完成或跳过时使用通知 scan_ref（LM 编号）或 scope=latest_scan，仅核对该批次，禁止用历史整理或 RSS 跳过记录代替。结果分页，has_more 时保留 scan_ref/scope 并用 next_offset 继续；total 仅指本页。",
            risk=RiskLevel.READ,
            parameters={
                "type": "object",
                "properties": {
                    "scope": {
                        "type": "string",
                        "enum": [
                            "all",
                            "attention",
                            "active",
                            "history",
                            "skipped",
                            "latest_scan",
                            *sorted(LOCAL_TASK_STATUSES),
                        ],
                    },
                    "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                    "scan_ref": {"type": "string", "pattern": "^(LM[1-9][0-9]{0,17}|LM-UNRECORDED)$", "maxLength": 20},
                    "offset": {"type": "integer", "minimum": 0, "maximum": 2147483647},
                },
                "additionalProperties": False,
            },
            context_handler=list_local_media_task_summaries,
            validator=local_media_task_summaries_arguments,
            workflow="local_media_task_resolution",
            workflow_stage=10,
            examples=("列出本地媒体任务", "查看失败的本地整理任务", "刚才本地整理通知里哪个完成哪个跳过", "查看本次扫描 LM12 的文件结果"),
        )
    )
    registry.register(
        ToolSpec(
            name="local_media.inspect_task",
            description="只读检查一个短期公开序号对应的待人工确认任务，生成 owner 绑定检查序号；不返回路径、错误正文或内部句柄。",
            risk=RiskLevel.READ,
            parameters={
                "type": "object",
                "required": ["task_number"],
                "properties": {"task_number": {"type": "integer", "minimum": 1}},
                "additionalProperties": False,
            },
            context_handler=inspect_local_media_task,
            validator=local_media_task_number_arguments,
        )
    )
    registry.register(
        ToolSpec(
            name="local_media.preview_task",
            description="基于 owner 绑定短期检查序号生成本地整理匹配预览；只读且不返回路径、TMDB ID、规则快照或内部检查 ID。",
            risk=RiskLevel.READ,
            parameters={
                "type": "object",
                "required": ["inspection_number"],
                "properties": {"inspection_number": {"type": "integer", "minimum": 1}},
                "additionalProperties": False,
            },
            context_handler=preview_local_media_task,
            validator=local_media_inspection_arguments,
        )
    )
    registry.register(
        ToolSpec(
            name="local_media.retry_task",
            description="预检并确认后重试 failed 或 requires_manual 的本地媒体任务。仅当任务是已选定 TMDB 剧集候选的 requires_manual 任务时，才可同时提供用户明确指定的 season 与 episode，将错误季集（如 S02E24）修正为 S02E12 后复用统一事务重新排队。若当前会话没有最新 task_number，必须先调用 local_media.task_summaries 绑定任务，禁止猜测；这不是任意文件路径改名。",
            risk=RiskLevel.LOW_WRITE,
            parameters={
                "type": "object",
                "required": ["task_number"],
                "properties": {
                    "task_number": {"type": "integer", "minimum": 1, "maximum": 100},
                    "season": {"type": "integer", "minimum": 0, "maximum": 99},
                    "episode": {"type": "integer", "minimum": 1, "maximum": 999},
                },
                "additionalProperties": False,
            },
            validator=local_media_retry_arguments,
            requires_confirmation=True,
            context_confirmation_preparer=prepare_retry_local_media_task,
            context_confirmed_handler=retry_local_media_task_confirmed,
            related_tools=(
                "local_media.task_summaries",
                "local_media.inspect_task",
                "local_media.preview_task",
            ),
            workflow="local_media_task_resolution",
            workflow_stage=20,
            examples=(
                "重试本地媒体任务 1",
                "把这个待确认任务改成 S02E12 后继续入库",
                "将本地媒体任务 1 的季集修正为第 2 季第 12 集",
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="local_media.refresh_task_library",
            description="预检并确认后，仅对已完成任务重新解析出的唯一绑定媒体服务器与媒体库执行精准路径刷新；不接受 URL、路径或内部 ID。",
            risk=RiskLevel.LOW_WRITE,
            parameters={
                "type": "object",
                "required": ["task_number"],
                "properties": {"task_number": {"type": "integer", "minimum": 1}},
                "additionalProperties": False,
            },
            validator=local_media_task_number_arguments,
            requires_confirmation=True,
            context_confirmation_preparer=prepare_refresh_local_media_task_library,
            context_confirmed_handler=refresh_local_media_task_library_confirmed,
        )
    )
    registry.register(
        ToolSpec(
            name="local_media.verify_task_library_visibility",
            description="只读核验已完成任务的媒体是否已在唯一绑定媒体库中索引，并明确标记未执行真实播放探测。",
            risk=RiskLevel.READ,
            parameters={
                "type": "object",
                "required": ["task_number"],
                "properties": {"task_number": {"type": "integer", "minimum": 1}},
                "additionalProperties": False,
            },
            context_handler=verify_local_media_task_library_visibility,
            validator=local_media_task_number_arguments,
        )
    )
    registry.register(
        ToolSpec(
            name="local_media.history_summary",
            description="只读汇总本地媒体已完成与失败历史的数量、触发来源和时间分布，不返回标题、路径、任务标识或错误正文。",
            risk=RiskLevel.READ,
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            handler=summarize_local_media_history,
            validator=local_media_history_arguments,
        )
    )
