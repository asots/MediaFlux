"""library 领域的 Agent 原子工具声明。"""

from __future__ import annotations

from app.agent.domains.library.batch_presence import (
    batch_library_presence,
    batch_presence_arguments,
)
from app.agent.domains.library.episode_audit import audit_series_episodes
from app.agent.domains.library.episode_count import (
    count_series_episodes,
    count_series_episodes_arguments,
)
from app.agent.domains.library.episode_resource_actions import (
    DEFAULT_MISSING_SEASON_EPISODES,
    MAX_MISSING_SEASON_EPISODES,
    missing_episode_resource_arguments,
    missing_season_resource_arguments,
    search_missing_episode_resources,
    search_missing_season_resources,
)
from app.agent.domains.library.library_episode_audit import audit_library_episodes
from app.agent.domains.library.missing_workflows import (
    list_missing_workflows,
    missing_workflow_arguments,
)
from app.agent.domains.library.patrol_config_actions import (
    patrol_policy_arguments,
    patrol_policy_summary_arguments,
    prepare_patrol_policy_confirmation,
    set_patrol_policy_confirmed,
    summarize_patrol_policy,
)
from app.agent.domains.library.patrol_status import (
    get_library_patrol_status,
    patrol_status_arguments,
)
from app.agent.domains.library.patrol_trigger_actions import (
    patrol_trigger_arguments,
    prepare_trigger_patrol_now,
    trigger_patrol_now_confirmed,
)
from app.agent.domains.library.search import library_search_arguments, search_library
from app.agent.domains.library.update_actions import check_library_updates
from app.agent.domains.resources.candidate_context import MAX_RESOURCE_CANDIDATES
from app.agent.domains.shared_arguments import (
    _episode_audit_arguments,
    _library_episode_audit_arguments,
    _library_update_arguments,
)
from app.agent.domains.subscriptions.preference_policy import (
    RESOURCE_PREFERENCE_PROPERTIES,
    owner_media_preferences,
)
from app.agent.domains.system.jobs_actions import (
    prepare_start_episode_audit,
    start_episode_audit_arguments,
    start_episode_audit_confirmed,
)
from app.agent.models import (
    RiskLevel,
    ToolSpec,
)


def register_specs(registry, *, missing_media_runtime) -> None:
    def missing_episode_search(arguments, context):
        result = search_missing_episode_resources(
            arguments, preferences=owner_media_preferences(context.owner),
        )
        if missing_media_runtime is not None:
            missing_media_runtime.capture_search(
                owner=context.owner,
                tool_name="library.search_missing_episode_resources",
                result=result,
            )
        return result

    def missing_season_search(arguments, context):
        result = search_missing_season_resources(
            arguments, preferences=owner_media_preferences(context.owner), context=context,
        )
        if missing_media_runtime is not None:
            missing_media_runtime.capture_search(
                owner=context.owner,
                tool_name="library.search_missing_season_resources",
                result=result,
            )
        return result

    def missing_workflows(arguments, context):
        return list_missing_workflows(
            arguments,
            context,
            repository=(
                missing_media_runtime.repository
                if missing_media_runtime is not None
                else None
            ),
        )

    registry.register(
        ToolSpec(
            name="library.search",
            description=(
                "在已配置的 Jellyfin / Emby 中按标题或演员元数据搜索、分页或统计。"
                "演员作品数量用 search_by=actor、count_only=true；只问数量无需再拉明细。可用 media_type 限定电影或剧集。"
                "total 是服务器匹配记录总数，returned 只是本次返回数，不能据它推算总量。"
                "total=null 表示无法确认；has_more/next_offset 用于继续分页，offset 按服务器应用。"
                "演员模式精确解析人物元数据，不以标题包含当演员作品；all 默认不计单集。"
                "多服务器数量按记录累加，不是去重作品数；可用时返回已校验的 open_url。"
                "已有 TMDB 身份清单核对存在性用 library.batch_presence；"
                "media_type=episode 的标题搜索查单集自身标题，不是按剧名读取分集；空结果不能证明剧或指定集不存在。"
                "按剧名查本地季集库存用 library.count_series_episodes；"
                "多部剧更新核对直接用 library.check_updates 的 queries，不要逐部重复搜索。"
            ),
            risk=RiskLevel.READ,
            domains=("library", "media_identity"),
            source_kind="local_library",
            freshness="live",
            parameters={
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string", "minLength": 1, "maxLength": 120},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "default": 8,
                    },
                    "offset": {"type": "integer", "minimum": 0},
                    "search_by": {"type": "string", "enum": ["title", "actor"]},
                    "media_type": {"type": "string", "enum": ["all", "movie", "series", "episode"]},
                    "count_only": {"type": "boolean"},
                },
                "additionalProperties": False,
            },
            handler=search_library,
            parallel_read=True,
            validator=library_search_arguments,
            examples=(
                "媒体库里有没有《某片》",
                "在 Jellyfin 或 Emby 搜索这个标题",
                "统计媒体库中某演员的电影数量",
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="library.batch_presence",
            description=(
                "一次按 TMDB ID 批量核对最多 50 部电影或剧集是否存在于已配置的 Jellyfin / "
                "Emby。人物作品表、系列片单或其他多条媒体清单必须优先使用本工具，不能逐部"
                "重复调用 library.search。"
            ),
            risk=RiskLevel.READ,
            domains=("library", "media_identity"),
            source_kind="local_library",
            freshness="live",
            parameters={
                "type": "object",
                "required": ["items"],
                "properties": {
                    "items": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 50,
                        "items": {
                            "type": "object",
                            "required": ["tmdb_id", "media_type", "title"],
                            "properties": {
                                "tmdb_id": {
                                    "type": "string",
                                    "pattern": "^[1-9][0-9]{0,9}$",
                                },
                                "media_type": {
                                    "type": "string",
                                    "enum": ["movie", "tv"],
                                },
                                "title": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 240,
                                },
                                "year": {
                                    "type": "string",
                                    "pattern": "^(?:19|20)[0-9]{2}$",
                                },
                            },
                            "additionalProperties": False,
                        },
                    }
                },
                "additionalProperties": False,
            },
            handler=batch_library_presence,
            parallel_read=True,
            validator=batch_presence_arguments,
            related_tools=("discovery.person_filmography",),
            examples=(
                "一次核对这份导演片单哪些已入库、哪些缺失",
                "批量检查这些 TMDB 电影是否在 Jellyfin",
                "比较一组剧集的媒体库收录状态",
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="library.search_missing_episode_resources",
            description="仅用于补缺；普通指定季集找资源用 indexer.search_resources，无须证明缺失。先确认指定季集属于已播缺集，再按当前身份保存的资源偏好排序；本次 preference_overrides 优先于长期偏好（空数组/0/any 可取消对应约束），不会自动下载。",
            risk=RiskLevel.READ,
            domains=("resource_search", "library"),
            source_kind="resource_index",
            freshness="realtime",
            parameters={
                "type": "object",
                "required": ["query", "season", "episode"],
                "properties": {
                    "preference_overrides": {
                        "type": "object",
                        "properties": RESOURCE_PREFERENCE_PROPERTIES,
                        "additionalProperties": False,
                        "description": "仅本次请求覆盖对应质量偏好；不保存配置。",
                    },
                    "query": {"type": "string", "minLength": 1, "maxLength": 120},
                    "tmdb_id": {"type": "string", "pattern": "^[0-9]{1,10}$"},
                    "library_name": {"type": "string", "minLength": 1, "maxLength": 80},
                    "season": {"type": "integer", "minimum": 1, "maximum": 100},
                    "episode": {"type": "integer", "minimum": 1, "maximum": 1000},
                    "as_of": {"type": "string", "format": "date"},
                    "sites": {
                        "type": "array",
                        "maxItems": 16,
                        "items": {"type": "string", "pattern": "^[a-z0-9_-]{1,32}$"},
                    },
                    "limit": {"type": "integer", "minimum": 1, "maximum": MAX_RESOURCE_CANDIDATES},
                },
                "additionalProperties": False,
            },
            context_handler=missing_episode_search,
            validator=missing_episode_resource_arguments,
            examples=(
                "搜索某剧第 2 季第 3 集的缺集资源",
                "确认 S02E03 缺失后找资源",
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="library.search_missing_season_resources",
            description="仅用于媒体库补缺；普通指定季集找资源用 indexer.search_resources，不要求缺集审计。单部按query/season核对；多部缺集找资源必须一次传items（最多12部/季）。每部默认覆盖20个已播缺集；episodes可指定最多100个集号，默认完整处理指定范围，显式max_episodes可缩小范围。一次单季共享搜索后逐集匹配，仅对无明确候选的集在总耗时/请求预算内精确补查，无需因旧三集限制让用户继续。共享查询无匹配不代表精确检索穷尽；remaining_episodes列明未完成范围，站点失败如实保留。按偏好选可核验覆盖候选，保留整批统一全局编号和一个resource_candidates_ref，随后一次ingest.submit预检整批；未知映射/未播Fin/不同作品不推荐，不自动下载。",
            risk=RiskLevel.READ,
            domains=("resource_search", "library"),
            source_kind="resource_index",
            freshness="realtime",
            parameters={
                "type": "object",
                "anyOf": [{"required": ["query", "season"]}, {"required": ["items"]}],
                "not": {"required": ["items"], "anyOf": [
                    {"required": [field]} for field in (
                        "query", "season", "tmdb_id", "library_name", "episodes"
                    )
                ]},
                "properties": {
                    "items": {"type": "array", "minItems": 1, "maxItems": 12, "items": {
                        "type": "object", "required": ["query", "season"], "additionalProperties": False,
                        "properties": {"query": {"type": "string", "minLength": 1, "maxLength": 120},
                                       "season": {"type": "integer", "minimum": 1, "maximum": 100},
                                       "episodes": {"type": "array", "minItems": 1, "maxItems": MAX_MISSING_SEASON_EPISODES, "uniqueItems": True,
                                                    "items": {"type": "integer", "minimum": 1, "maximum": 1000}},
                                       "tmdb_id": {"type": "string", "pattern": "^[0-9]{1,10}$"},
                                       "library_name": {"type": "string", "maxLength": 80}}}},
                    "preference_overrides": {
                        "type": "object",
                        "properties": RESOURCE_PREFERENCE_PROPERTIES,
                        "additionalProperties": False,
                        "description": "仅本次请求覆盖对应质量偏好；不保存配置。",
                    },
                    "query": {"type": "string", "minLength": 1, "maxLength": 120},
                    "tmdb_id": {"type": "string", "pattern": "^[0-9]{1,10}$"},
                    "library_name": {"type": "string", "minLength": 1, "maxLength": 80},
                    "season": {"type": "integer", "minimum": 1, "maximum": 100},
                    "episodes": {"type": "array", "minItems": 1, "maxItems": MAX_MISSING_SEASON_EPISODES, "uniqueItems": True,
                                 "items": {"type": "integer", "minimum": 1, "maximum": 1000}},
                    "as_of": {"type": "string", "format": "date"},
                    "sites": {
                        "type": "array",
                        "maxItems": 16,
                        "items": {"type": "string", "pattern": "^[a-z0-9_-]{1,32}$"},
                    },
                    "max_episodes": {"type": "integer", "minimum": 1, "maximum": MAX_MISSING_SEASON_EPISODES,
                                     "description": f"默认覆盖{DEFAULT_MISSING_SEASON_EPISODES}个缺集；指定episodes时默认覆盖指定范围，显式设置可缩小。"},
                    "limit_per_episode": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "additionalProperties": False,
            },
            context_handler=missing_season_search,
            validator=missing_season_resource_arguments,
        )
    )
    registry.register(
        ToolSpec(
            name="library.missing_media_workflows",
            description=(
                "查看当前用户最近缺集补库流程的安全状态；只返回剧名、季集、阶段、"
                "目标类型与是否已建立下载任务，不返回资源句柄、磁力、URL、路径或凭据。"
            ),
            risk=RiskLevel.READ,
            parameters={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 30,
                        "default": 10,
                    },
                },
                "additionalProperties": False,
            },
            validator=missing_workflow_arguments,
            context_handler=missing_workflows,
        )
    )
    registry.register(
        ToolSpec(
            name="library.check_updates",
            description=(
                "单部或一次批量核对最多20部媒体更新：单部传query，多部/片单/关注列表传queries，"
                "无需先逐部library.search或取得TMDB ID。默认刷新实际媒体库库存，‘再查一次’应沿用"
                "上下文中的片单重新调用，不得复述上次进度。剧集对照TMDB已播季集，电影核对本地存在性。"
                "逐部报告缺集、无已播缺集或待确认；集数数量不等于最新集号，不能猜测绝对集与季集映射。"
                "这不是全网资源发布或官方实时公告；需要下载候选时再检索资源，不得把单站失败当作全网无更新。"
            ),
            risk=RiskLevel.READ,
            domains=("library", "official_progress"),
            source_kind="local_library",
            freshness="live",
            parameters={
                "type": "object",
                "anyOf": [{"required": ["query"]}, {"required": ["queries"]}],
                "not": {"anyOf": [
                    {"required": ["queries", "query"]},
                    {"required": ["queries", "tmdb_id"]},
                    {"required": ["queries", "season"]},
                    {"required": ["media_type", "season"],
                     "properties": {"media_type": {"enum": ["movie"]}}},
                ]},
                "properties": {
                    "query": {"type": "string", "minLength": 1, "maxLength": 120},
                    "queries": {"type": "array", "minItems": 1, "maxItems": 20,
                                "items": {"type": "string", "minLength": 1, "maxLength": 120}},
                    "refresh": {"type": "boolean", "default": True,
                                "description": "默认重新读取当前库存；只有显式false才允许复用短时审计缓存"},
                    "media_type": {"type": "string", "enum": ["auto", "tv", "movie"]},
                    "tmdb_id": {"type": "string", "pattern": "^[0-9]{1,10}$"},
                    "season": {"type": "integer", "minimum": 1, "maximum": 100},
                    "as_of": {"type": "string", "format": "date"},
                },
                "additionalProperties": False,
            },
            context_handler=check_library_updates,
            validator=_library_update_arguments,
            related_tools=("library.search_missing_season_resources", "library.search_missing_episode_resources"),
            examples=(
                "检查《某剧》有没有更新",
                "媒体库中的这十部国漫都有更新吗",
                "再检查一下刚才那些剧的更新",
                "这部剧最新播到哪里而本地有多少",
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="library.audit_library_episodes",
            description="有界枚举已配置媒体服务器中的剧集，并按可靠 TMDB 映射巡检截至指定日期的已播缺集。",
            risk=RiskLevel.READ,
            parameters={
                "type": "object",
                "properties": {
                    "as_of": {"type": "string", "format": "date"},
                    "max_series": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                "additionalProperties": False,
            },
            handler=audit_library_episodes,
            validator=_library_episode_audit_arguments,
            examples=(
                "巡检整个媒体库有没有缺集",
                "检查全部剧集的完整性",
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="library.start_episode_audit",
            description="在用户确认后创建可恢复、可查询进度、可取消的后台全库剧集完整性检查。",
            risk=RiskLevel.LOW_WRITE,
            parameters={
                "type": "object",
                "properties": {
                    "as_of": {"type": "string", "format": "date"},
                    "max_series": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                "additionalProperties": False,
            },
            validator=start_episode_audit_arguments,
            requires_confirmation=True,
            context_confirmation_preparer=prepare_start_episode_audit,
            context_confirmed_handler=start_episode_audit_confirmed,
        )
    )
    registry.register(
        ToolSpec(
            name="library.patrol_status",
            description="查询最近一次后台全库缺集巡检的安全摘要；不会触发巡检、资源搜索或下载。",
            risk=RiskLevel.READ,
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            handler=get_library_patrol_status,
            validator=patrol_status_arguments,
        )
    )
    registry.register(
        ToolSpec(
            name="library.patrol_policy",
            description="读取后台全库缺集巡检的启用、通知、间隔和单轮检查上限，不返回其他配置。",
            risk=RiskLevel.READ,
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            handler=summarize_patrol_policy,
            validator=patrol_policy_summary_arguments,
        )
    )
    registry.register(
        ToolSpec(
            name="library.set_patrol_policy",
            description="预检并在用户确认后修改全库缺集巡检的四项白名单策略，不接受配置键、凭据、URL 或路径。",
            risk=RiskLevel.LOW_WRITE,
            parameters={
                "type": "object",
                "minProperties": 1,
                "properties": {
                    "enabled": {"type": "boolean"},
                    "notify_enabled": {"type": "boolean"},
                    "interval_hours": {"type": "integer", "minimum": 1, "maximum": 168},
                    "max_series": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                "additionalProperties": False,
            },
            validator=patrol_policy_arguments,
            requires_confirmation=True,
            context_confirmed_handler=ToolSpec.context_free_confirmed_handler(
                set_patrol_policy_confirmed
            ),
            context_confirmation_preparer=ToolSpec.context_free_confirmation_preparer(
                prepare_patrol_policy_confirmation
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="library.trigger_patrol_now",
            description="预检并在用户确认后，按当前全库缺集巡检策略把单例后台任务排到现在；不修改策略、不搜索资源、不下载。",
            risk=RiskLevel.LOW_WRITE,
            parameters={
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            validator=patrol_trigger_arguments,
            requires_confirmation=True,
            context_confirmed_handler=ToolSpec.context_free_confirmed_handler(
                trigger_patrol_now_confirmed
            ),
            context_confirmation_preparer=ToolSpec.context_free_confirmation_preparer(
                prepare_trigger_patrol_now
            ),
            examples=("按当前策略立即巡检媒体库", "现在执行一次自动缺集巡检"),
        )
    )
    registry.register(
        ToolSpec(
            name="library.count_series_episodes",
            description="按剧名或 TMDB ID 读取 Jellyfin / Emby 的本地普通集数量、季度分布及实际季集闭区间 seasons.episode_ranges，用于核对指定集是否已收录。inventory_complete=false 时未出现的集不能判为不存在。不访问 TMDB，不判断播出或应补哪些集。",
            risk=RiskLevel.READ,
            domains=("library", "episode_numbering"),
            source_kind="local_library",
            freshness="live",
            parameters={
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string", "minLength": 1, "maxLength": 120},
                    "tmdb_id": {"type": "string", "pattern": "^[0-9]{1,10}$"},
                    "library_name": {
                        "type": "string", "minLength": 1, "maxLength": 80,
                        "description": "服务器内的具体媒体库名称，如动漫、电视剧，不是服务器产品名；用户未指定具体媒体库时省略。",
                    },
                },
                "additionalProperties": False,
            },
            handler=count_series_episodes,
            parallel_read=True,
            validator=count_series_episodes_arguments,
            examples=(
                "媒体库中《某剧》一共有多少集",
                "这部剧本地有几季几集",
                "核对剧集的指定季集在 Jellyfin 里有没有收录",
            ),
        )
    )
    registry.register(
        ToolSpec(
            name="library.audit_episodes",
            description="核对媒体库剧集与 TMDB 截止日期前已播普通集，报告缺集或可更新集。",
            risk=RiskLevel.READ,
            parameters={
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {"type": "string", "minLength": 1, "maxLength": 120},
                    "tmdb_id": {"type": "string", "pattern": "^[0-9]{1,10}$"},
                    "library_name": {"type": "string", "minLength": 1, "maxLength": 80},
                    "season": {"type": "integer", "minimum": 1, "maximum": 100},
                    "target_episode": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 1000,
                    },
                    "as_of": {"type": "string", "format": "date"},
                },
                "additionalProperties": False,
            },
            handler=audit_series_episodes,
            validator=_episode_audit_arguments,
        )
    )
