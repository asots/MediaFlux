"""伴随元数据队列的原子工具，无独立业务路由。"""

from app.agent.models import RiskLevel, ToolSpec
from app.agent.strm_metadata_actions import (
    cancel_confirmed,
    get_metadata_status,
    no_arguments,
    policy_arguments,
    policy_confirmed,
    prepare_cancel,
    prepare_policy,
)


def register_specs(registry, **_kwargs) -> None:
    empty = {"type": "object", "properties": {}, "additionalProperties": False}
    registry.register(
        ToolSpec(
            name="strm.metadata.status",
            description="读取伴随元数据（云盘 NFO、字幕、海报同步）队列数量、开关、消费者线程和熔断状态；running=0只代表瞬时任务数，不代表不会自动工作。不是影视演员资料查询。",
            risk=RiskLevel.READ,
            parameters=empty,
            validator=no_arguments,
            handler=get_metadata_status,
            domains=("strm", "automation"),
            related_tools=("strm.metadata.set_enabled", "strm.metadata.cancel_pending"),
            examples=(
                "元数据队列为什么一直挂着",
                "伴随同步是否关闭，后台还在工作吗",
                "查看NFO同步积压",
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="strm.metadata.set_enabled",
            description="经人工确认调整已有伴随同步开关。关闭后不领取后续任务；后续启动的同步不再新增，已经运行的扫描可能沿用旧配置继续入队。保留历史队列，当前任务可完成；开启可继续历史积压。不控制Jellyfin自身刮削。",
            risk=RiskLevel.WRITE,
            parameters={
                "type": "object",
                "properties": {"enabled": {"type": "boolean"}},
                "required": ["enabled"],
                "additionalProperties": False,
            },
            validator=policy_arguments,
            requires_confirmation=True,
            context_confirmation_preparer=prepare_policy,
            context_confirmed_handler=policy_confirmed,
            domains=("strm", "automation", "config"),
            related_tools=("strm.metadata.status", "strm.metadata.cancel_pending"),
            examples=(
                "关闭NFO同步但保留积压",
                "停止元数据同步",
                "重新开启伴随元数据同步",
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="strm.metadata.cancel_pending",
            description="预览并经人工确认取消当前 queued/retry_wait 伴随元数据积压。冻结集合、事务校验，只标记取消不删除记录或文件；不影响running、已完成和预览后新任务，不开启同步、不改变开关。已关闭同步仍可清除待办。",
            risk=RiskLevel.WRITE,
            parameters=empty,
            validator=no_arguments,
            requires_confirmation=True,
            context_confirmation_preparer=prepare_cancel,
            context_confirmed_handler=cancel_confirmed,
            domains=("strm", "automation"),
            related_tools=("strm.metadata.status",),
            examples=(
                "元数据同步早关了，清除13581项积压",
                "不用重新开启同步，清空NFO待办队列",
                "取消排队元数据任务，不删已下载海报",
            ),
        )
    )
