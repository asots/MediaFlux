"""resource 领域的 Agent 原子工具声明。"""

from __future__ import annotations

from app.agent.domains.resources.candidate_actions import (
    present_candidates,
    present_candidates_arguments,
)
from app.agent.domains.resources.candidate_context import MAX_RESOURCE_CANDIDATES
from app.agent.domains.resources.indexer_actions import (
    TECHNICAL_FILTER_CHOICES,
    search_resources,
)
from app.agent.domains.resources.indexer_actions import (
    search_arguments as indexer_search_arguments,
)
from app.agent.domains.resources.ingest_actions import (
    ingest_inspect_arguments,
    ingest_status_arguments,
    ingest_submit_arguments,
)
from app.agent.domains.resources.readiness_actions import (
    diagnose_indexer_readiness,
    indexer_readiness_arguments,
)
from app.agent.models import (
    RiskLevel,
    ToolSpec,
)

_TECHNICAL_PARAMETERS = {
    key: {"type": "boolean" if key == "exclude_hdr" else "string", "enum": list(choices),
          "description": ("排除发布名含HDR/DoVi/HLG的版本；未标注仍为未知，不保证SDR。false取消排除。" if key == "exclude_hdr" else
                          "仅保留发布名明确标注此规格的候选；空值不限，未知不匹配。标题声明非媒体流核验。")}
    for key, choices in TECHNICAL_FILTER_CHOICES.items()
}


def register_specs(registry, *, ingest_actions) -> None:
    registry.register(
        ToolSpec(
            name="indexer.diagnose_readiness",
            description="只读检查多站资源索引器的本地开关、启用站点与能力声明；不访问资源站、网络或文件系统，也不返回 URL、Cookie 或凭据。",
            risk=RiskLevel.READ,
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            handler=diagnose_indexer_readiness,
            validator=indexer_readiness_arguments,
            examples=(
                "为什么没搜到资源",
                "资源站连不上怎么办",
                "检查多站资源搜索状态",
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="indexer.search_resources",
            description=(
                "按片名、季集或规格直接搜索资源，不要求媒体库中缺失。在Agent允许的多站索引中查询，不受手动页面的普通站点复选框限制；总开关和成人来源授权仍生效。只返回 opaque result_id 与公开元数据。"
                "sites 可把本次读取严格限制到指定站点且不会修改站点配置；例如 "
                'sites=["sukebei"] 只查询 Sukebei。需要近期结果时使用 '
                "sort_mode=published_desc。"
                "结果（包括空结果）返回 owner/session 绑定的 resource_candidates_ref。续查同一目标时传该引用和要改的字段，未传字段继承，返回 effective_query 核对完整条件。"
                "画质/类型空字符串、aliases/sites空数组、year/season/episode为null明确取消该条件；续查默认回到第1页，翻页须指定page。独立新搜索不传引用，必须传title。后续资源检查/提交原样使用候选引用。"
                "可用于交叉核对连载资源跟进到哪一集，但资源标题只能作为旁证，不能证明官方播出进度。"
                "technical_evidence 是发布名声明而非媒体流探测，null 表示未知；未标 HDR 不等于已核实 SDR、位深或色域。"
                "用户将画质或动态范围明确作为硬条件时传 resolution/dynamic_range，由工具过滤未知或不符项；优先等软偏好不强制过滤，不把规格塞进片名。"
                "普通搜索只供模型研究，不自动出卡；找到符合用户要求的候选后，使用 indexer.present_candidates 挑选展示，没找到不要展示。"
                "已确认缺集后找资源应使用 library.search_missing_episode_resources 或 library.search_missing_season_resources，避免把旧集当更新。"
            ),
            risk=RiskLevel.READ,
            domains=("resource_search", "official_progress"),
            source_kind="resource_index",
            freshness="realtime",
            parameters={
                "type": "object",
                "anyOf": [{"required": ["title"]}, {"required": ["resource_candidates_ref"]}],
                "properties": {
                    "resource_candidates_ref": {
                        "type": "string", "pattern": "^ref_[A-Za-z0-9_-]{16,160}$",
                        "description": "续查使用最近一次搜索/筛选返回的引用；只需传改动字段，省略字段继承，空结果也可续改。",
                    },
                    "title": {"type": "string", "minLength": 1, "maxLength": 120},
                    "original_title": {"type": "string", "maxLength": 120},
                    "english_title": {"type": "string", "maxLength": 120},
                    "aliases": {
                        "type": "array",
                        "maxItems": 8,
                        "items": {"type": "string", "minLength": 1, "maxLength": 120},
                    },
                    "year": {"type": ["integer", "null"], "minimum": 1800, "maximum": 2200},
                    "media_type": {
                        "type": "string",
                        "enum": ["", "movie", "tv", "anime"],
                    },
                    **_TECHNICAL_PARAMETERS,
                    "season": {
                        "type": ["integer", "null"],
                        "minimum": 0,
                        "maximum": 100,
                    },
                    "episode": {
                        "type": ["integer", "null"],
                        "minimum": 1,
                        "maximum": 1000,
                    },
                    "page": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 1,
                    },
                    "sort_mode": {
                        "type": "string",
                        "enum": [
                            "published_desc",
                            "relevance_desc",
                            "episode_desc",
                            "seeders_desc",
                            "size_desc",
                            "size_asc",
                        ],
                        "default": "relevance_desc",
                    },
                    "sites": {
                        "type": "array",
                        "maxItems": 16,
                        "items": {"type": "string", "pattern": "^[a-z0-9_-]{1,32}$"},
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_RESOURCE_CANDIDATES,
                        "default": 20,
                    },
                },
                "additionalProperties": False,
            },
            handler=search_resources,
            validator=indexer_search_arguments,
            related_tools=("ingest.submit", "indexer.present_candidates", "ingest.inspect"),
            examples=(
                "搜索《某片》的资源版本",
                "按片名查找指定季集的单集资源",
                "找种子或磁力资源",
                "核对某部连载动画的资源索引跟进到第几集",
                '最近有什么新出的步兵资源，仅用 sites=["sukebei"] 并按发布时间倒序搜索 uncensored',
                '查看 Sukebei 最近发布的无码资源，不查询其他索引站',
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="indexer.present_candidates",
            description=(
                "从已有 resource_candidates_ref 的完整候选快照中，按原 position 显式筛选要展示的资源候选。"
                "resolution/dynamic_range/exclude_hdr 可直接筛选已有候选，无须重新搜索；省略继承该引用的查询条件，空字符串或false取消对应条件，返回effective_query、实际positions与排除统计。严格规格条件不匹配未知项；仅排除HDR不等于已核实SDR。放宽条件须由用户明确提出；原快照不补回已排除项，应传同一引用按新条件搜索。"
                "这是纯只读展示动作，不恢复 Provider、不读取最近快照、不重新搜索、不访问云盘或索引网络；"
                "positions 为空表示本轮明确不展示候选。如需继续接入本次筛选结果，请使用本次返回的 reference_arguments.resource_candidates_ref；新引用值可能不同于输入，但仍绑定同一份候选快照，再传给 ingest.inspect 或 ingest.submit。"
            ),
            risk=RiskLevel.READ,
            parameters={
                "type": "object",
                "required": ["resource_candidates_ref", "positions"],
                "properties": {
                    "resource_candidates_ref": {
                        "type": "string",
                        "pattern": "^ref_[A-Za-z0-9_-]{16,160}$",
                    },
                    "positions": {
                        "type": "array",
                        "maxItems": MAX_RESOURCE_CANDIDATES,
                        "uniqueItems": True,
                        "items": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": MAX_RESOURCE_CANDIDATES,
                        },
                        "description": "要展示的原候选位置；只筛指定子集，不代表搜遍全部。空数组明确清除候选展示。",
                    },
                    **_TECHNICAL_PARAMETERS,
                },
                "additionalProperties": False,
            },
            handler=present_candidates,
            validator=present_candidates_arguments,
            related_tools=("ingest.submit", "ingest.inspect"),
            domains=("resource_search",),
            source_kind="resource_presentation",
            freshness="snapshot",
            examples=(
                "展示刚才第 1、3 个资源候选",
                "只展示资源候选中的第 2 个",
                "这轮不要展示资源候选",
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="ingest.inspect",
            description=(
                "统一只读检查资源接入来源：可识别光鸭官方分享、Magnet、ED2K、明确 HTTP(S) 下载直链，"
                "或读取候选快照。resource_candidates 可选 positions 仅检查指定序号，省略读取全部；保持原序号，不代表已下载。原始链接、分享令牌、file_id 与索引 result_id 只保存在"
                "owner 绑定的短期服务端快照中；普通网页链接不会创建下载任务。"
            ),
            risk=RiskLevel.READ,
            parameters={
                "type": "object",
                "properties": {
                    "source_type": {
                        "type": "string",
                        "enum": [
                            "auto",
                            "direct_url",
                            "guangya_share",
                            "resource_candidates",
                        ],
                        "default": "auto",
                    },
                    "input": {"type": "string", "maxLength": 8192, "description": "仅用于原始下载/分享链接；候选快照引用应填 resource_candidates_ref，不填 input。"},
                    "positions": {
                        "type": "array", "minItems": 1,
                        "items": {"type": "integer", "minimum": 1, "maximum": MAX_RESOURCE_CANDIDATES},
                        "description": "仅resource_candidates可用；按请求顺序去重，省略读取全部。不是集号，不会提交下载。",
                    },
                    "resource_candidates_ref": {
                        "type": "string",
                        "pattern": "^ref_[A-Za-z0-9_-]{16,160}$",
                    },
                },
                "additionalProperties": False,
                "anyOf": [
                    {"required": ["source_type"], "properties": {
                        "source_type": {"enum": ["resource_candidates"]},
                        "input": {"maxLength": 0},
                    }},
                    {"required": ["input"], "properties": {
                        "source_type": {"enum": ["auto", "direct_url", "guangya_share"]},
                        "input": {"minLength": 1},
                    }, "not": {"anyOf": [
                        {"required": ["positions"]}, {"required": ["resource_candidates_ref"]},
                    ]}},
                ],
            },
            context_handler=ingest_actions.inspect,
            validator=ingest_inspect_arguments,
            domains=("downloads", "resource_search", "cloud_files"),
            source_kind="ingest_snapshot",
            freshness="live",
            workflow="ingest_submit",
            workflow_stage=10,
            related_tools=("ingest.submit",),
            examples=(
                "解析这个光鸭分享链接",
                "检查这个磁力链接能否下载",
                "查看刚才搜索到的资源候选",
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="ingest.submit",
            description=(
                "在用户确认后统一提交最近检查的直链或光鸭分享，或按资源搜索候选序号提交。"
                "资源搜索完成后应把返回的 resource_candidates_ref 原样传入，确保同轮与后续对话均绑定"
                "同一份候选快照；直链或分享检查完成后应把 ingest_snapshot_ref 原样传入。"
                "直链和资源候选可选 qB、光鸭或两边；省略 target 或使用 preferred 时，在预检中采用用户保存的默认下载目标，没有偏好则使用光鸭；目标随计划冻结，确认期间不随偏好变化。光鸭分享始终只转存到光鸭。确认参数不包含链接、"
                "访问令牌、云端 file_id、内部 result_id 或后端任务标识。"
            ),
            risk=RiskLevel.DANGER,
            parameters={
                "type": "object",
                "required": ["source_type"],
                "properties": {
                    "source_type": {
                        "type": "string",
                        "enum": ["direct_url", "guangya_share", "resource_candidates"],
                    },
                    "target": {
                        "type": "string", "enum": ["preferred", "qb", "guangya", "both"],
                        "description": "省略或 preferred 使用保存的默认目标；显式 qb/guangya/both 优先。",
                    },
                    "positions": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 200,
                        "uniqueItems": True,
                        "items": {"type": "integer", "minimum": 1, "maximum": 200},
                    },
                    "resource_candidates_ref": {
                        "type": "string",
                        "pattern": "^ref_[A-Za-z0-9_-]{16,160}$",
                    },
                    "ingest_snapshot_ref": {
                        "type": "string",
                        "pattern": "^ref_[A-Za-z0-9_-]{16,160}$",
                    },
                },
                "additionalProperties": False,
                "anyOf": [
                    {"properties": {
                        "source_type": {"enum": ["resource_candidates"]},
                        "positions": {"maxItems": MAX_RESOURCE_CANDIDATES,
                                      "items": {"maximum": MAX_RESOURCE_CANDIDATES}},
                    }},
                    {"properties": {"source_type": {"enum": ["direct_url", "guangya_share"]}}},
                ],
            },
            validator=ingest_submit_arguments,
            requires_confirmation=True,
            context_confirmation_preparer=ingest_actions.prepare_submit,
            context_confirmed_handler=ingest_actions.execute_submit,
            related_tools=("media.preferences",),
            domains=("downloads", "resource_search", "cloud_files"),
            source_kind="ingest_snapshot",
            freshness="live",
            workflow="ingest_submit",
            workflow_stage=20,
            examples=(
                "按我的默认下载目标提交刚才的资源",
                "把刚才的磁力提交到 qB",
                "把这个光鸭分享全部转存",
                "把刚才第 1、3 个资源提交到两边，并使用搜索结果返回的 resource_candidates_ref",
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="ingest.status",
            description=(
                "按公开请求编号读取统一资源接入状态，覆盖 qB、光鸭、整理与 STRM 阶段；"
                "不返回链接、路径、哈希或后端任务标识。"
            ),
            risk=RiskLevel.READ,
            parameters={
                "type": "object",
                "required": ["request_number"],
                "properties": {"request_number": {"type": "integer", "minimum": 1}},
                "additionalProperties": False,
            },
            context_handler=ingest_actions.status,
            validator=ingest_status_arguments,
            domains=("downloads", "organize", "strm"),
            source_kind="download_request",
            freshness="live",
            examples=("查询资源请求 12 的状态", "刚才提交的资源到哪一步了"),
        )
    )
