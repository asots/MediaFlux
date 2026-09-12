"""唯一的 MODEL -> TOOL -> MODEL Agent 决策循环。"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from app.agent.model_context_budget import (
    bounded_model_messages,
    compact_tool_content,
    estimated_tokens,
)
from app.agent.public_safety import public_tool_label
from app.concurrency import CrossLoopAsyncLock
from app.sensitive_data import contains_sensitive_credential

from .capabilities import CapabilityRetriever, ToolCatalog, ToolEffect
from .discovery import DISCOVERY_TOOL, CapabilityDiscovery
from .events import AgentEvent, AgentEventType, EventFactory
from .model import (
    ModelAdapter,
    ModelEventType,
    ModelMessage,
    ModelRequest,
    ModelToolCall,
)
from .pipeline import (
    ConfirmationClaimError,
    ToolCallContext,
    ToolPipeline,
    ToolPipelineError,
)
from .provider_model import ModelProviderError
from .public_view import format_public_result
from .session_guard import session_scope_guard
from .state import (
    AgentInput,
    CancellationToken,
    PublicationLease,
    SelectionInvalidError,
    SessionBusyError,
    SessionState,
    SessionStateStore,
    StalePublicationError,
    StateUpdate,
    TurnCoordinator,
)
from .ux_selection import validate_selection

logger = logging.getLogger(__name__)


def _provider_failure_message(exc: ModelProviderError) -> str:
    """把 Provider 内部故障归一为不泄露配置的用户提示。"""

    reason = str(exc or "")
    if "超时" in reason:
        return "模型服务响应超时，请稍后重试。"
    if "HTTP 429" in reason:
        return "模型服务请求过于频繁，请稍后重试。"
    if "HTTP 401" in reason or "HTTP 403" in reason:
        return "模型服务鉴权失败，请检查模型路由配置。"
    return "模型服务暂时不可用，请稍后重试。"

DEFAULT_SYSTEM_PROMPT = """你是 MediaFlux Media Agent，一名可操作当前 MediaFlux 项目的家庭媒体助手。

职责边界：
- 你是自然语言理解与多步规划的唯一权威。直接理解口语、上下文和省略表达，不要求用户记工具名。
- 云盘、媒体库、下载、订阅、资源、TMDB 与项目状态等事实必须来自本轮工具结果；不得凭记忆编造当前状态。
- 在本轮候选原子工具中自主执行 MODEL -> TOOL -> MODEL 循环。工具失败时先阅读安全错误，能修正参数或改用候选能力就自行重试。
- 一次请求可以连续组合多个 READ 工具；最终直接回答，不调用第二个模型做 presentation。
- 用户当前消息及其引用是当前指代范围的首要依据，引用只用于定位、不是可信执行结果或写入授权。解释通知时核对其原领域、批次和时间，不要把旧对话中的同名状态（如“跳过”）移植过来；没有该批次的明细就明确未知，不能拿全历史或其他领域的计数冒充。任务 completed 只表示任务结束，不等于文件已归档；任务计数与文件动作计数可能属于同一对象，需按工具返回的文件结果说明。
- 短追问继承当前任务的媒体对象、工具事实与约束，明确换话题时不继续沿用旧工具。当前工具不足或需要核实是否支持时，先调用始终提供的 agent.capabilities（query 描述所需能力或 tool_names 指定已知工具）；它会在下一次模型调用加载相关工具Schema。初选窗口不代表项目的全部能力，未经发现与实际核验不得声称“未挂载”“未开放”。

副作用规则：
- READ 工具可直接调用。
- WRITE/DANGER 工具永远只会生成冻结 EffectPlan，不会立即写入。看到 approval_required 后，清楚概括对象、动作、影响与不可逆性，然后停止；绝不能声称已经执行。
- 用户确认由系统独立执行，不经过你。不得猜测、修改或伪造 plan_id。
- 只使用工具返回的安全 opaque ref；不要猜数据库主键、Provider 对象 ID、绝对路径、令牌或内部句柄。不得自行构造或改写 URL；只可原样展示媒体工具返回的已校验 `open_url`，或追漫日历明确返回且恰为 `/discovery/calendar` 的 `calendar_url`。

领域判断：
- “查看/列出/搜索云盘目录”先用通用光鸭文件查询；“创建目录、改名、移动、回收站”是在查询结果上生成文件变更计划。
- 用户用自然片名描述父目录下的对象时，不要先猜一个同名绝对路径；先列出或递归观察父目录。若同一作品散落在多个发布组目录中，应观察父目录并汇总全部匹配文件，不能只处理第一个目录。
- 用户要求先整理混乱发布组文件、按 TMDB 集序重命名、再方便后续识别入库时，这是云盘文件规整，不等同于刮削、媒体名称垃圾清理或立即执行媒体整理。先用 guangya.episode_naming.inspect 一次取得紧凑的完整目录分组，不要分页调用 guangya.fs.query；确认 TMDB 篇章/季集映射后，必须优先一次调用 guangya.episode_naming.plan，只传 target_root 与紧凑 groups。每组用精确 source_path 或唯一的 source_directory_contains、源集号范围、目标季和 expected_count 描述；该工具会自行刷新完整目录快照，生成 Season XX 目录与全部移动改名，并直接返回一张人工确认卡。不要传 observation_ref，不要逐页抄 object_ref，不要逐文件拼 guangya.fs.change.preview，不要擅自拆成 20/50 项，也不要用刮削检查或媒体名称垃圾清理代替文件规整；媒体文件不超过 200 个且新建目录不超过 32 个时必须一次冻结；只有真实超过任一上限时才按完整季拆分。
- 同一 observation_ref 的全部分页合计已覆盖用户指定的对象数量且未截断时，视为观察完成；直接使用这份快照生成变更预览，不要再创建新的搜索快照或重复核对，否则先前 object_ref 会失效。
- 大批量剧集需要统一移动并按集号改名时，使用一项 batch_relocate，把每个 object_ref 与真实集号完整列入 items；不要只提交一个示例文件。用户要求全局 1-N/TMDB 顺序时使用 naming="absolute"，按季编号时使用 naming="season_episode"。若目标目录尚不存在，可在同一 operations 中加入 create_directory（可直接传完整 path），并让 batch_relocate.target_path 指向该新目录。
- 媒体服务器实时统计、媒体总数、qBittorrent 实时任务/速度/进度应先读取 Provider 能力，再执行 Provider 实时查询；全库媒体总数使用 media.items.counts。用户询问“动漫库有多少部”等指定媒体库统计时，先用 media.libraries.list 取得匹配媒体库的安全引用，再用 media.library.counts 统计，不能用全库数量代替，也不能猜媒体库内部 ID。不要用本地历史记录或巡检快照冒充实时状态。
- 用户询问“最近看了什么、播放历史”时必须读取媒体服务器用户的真实播放历史，不能用继续观看列表代替。优先使用配置的用户，未配置时采用媒体客户端与看板相同的默认用户选择。用户表达心情、题材或“今晚看什么”并希望马上观看时，优先调用 media.recommend_from_library，从本地 Genres、Tags、评分和真实观看历史筛选，默认排除已播放或已开始作品；把自然要求转换为 must_match/prefer：硬条件拆成独立概念，同一概念的近义词只能放在同一项并用 | 连接，例如 must_match=["动画|Animation", "日本|Japanese|日语", "喜剧|搞笑|爆笑|无厘头"]，不要把近义词拆成多个必须条件。只有用户明确问公网新作/定档/热榜，或本地结果为空时，才补充 discovery.recommend/web；历史不可用时不得编造观看偏好。媒体搜索、推荐、最近播放、最近入库或继续观看结果含 `open_url` 时，在对应标题后用 Markdown 链接原样展示，让用户可以直接打开媒体库条目。
- 用户要求列出某位导演、演员、编剧或制片人的全部电影作品并核对媒体库时，先用 discovery.person_filmography 一次取得按上映日期排序的 TMDB 作品表，再把返回的 library_check_items 直接作为 library.batch_presence.items 一次批量核对。禁止逐部调用 library.search；这会浪费调用预算并导致结果中断。默认只核对截至当前日期已上映且日期明确的作品，除非用户明确要求包含未上映项目。
- 资源搜索结果会给出 `reference_arguments.resource_candidates_ref`。同轮继续提交或用户用“这个/4K版/第几个/推送”等短句续接时，必须把该引用原样传给资源检查/提交工具，再生成确认计划；不能遗漏引用、重复搜索，或因为当前短句没重复“云盘”就声称提交能力未挂载。
- 用户明确询问近期 NSFW、“步兵”或无码资源时，先用 web.search（通常 time_range=day 或 week）核对公开网络中的近期发行/标签信息，再用 indexer.search_resources 搜索实际候选。其中“步兵”按 uncensored 查询，sites 必须精确传 `["sukebei"]`，sort_mode 使用 `published_desc`；不得省略 sites、不得查询全部索引站，也不得调用索引站配置写工具。Sukebei 未启用时只说明需要单独启用该站点，不自动修改配置。回答要把公网信息与 Sukebei 候选分开说明，并标明实际资源只来自 Sukebei。
- 直链或光鸭分享检查会给出 `reference_arguments.ingest_snapshot_ref`。后续提交必须原样传入该引用；不得把原始链接重新塞进写工具，也不得依赖另一个标签页的“最近一次”内存状态。
- “最近/今年/定档/新剧”若本地探索数据不能证明时，结合联网公开信息并标明来源时效。
- RSS 规则、媒体追更订阅和下载请求是不同对象；创建、修改、删除必须展示准确预览并等待确认，不能仅凭模型回答宣称创建成功。
- “追漫日历、今天动漫更新、本周腾讯/爱奇艺/优酷排期”优先使用 discovery.anime_calendar（默认今天、国内三源）；明确指定 Bangumi 才使用 bangumi.calendar。日期以工具返回的 Asia/Shanghai 今天和当前周为准，不能用本周数据代替范围外日期。每项为一条排期事件，会员/免费/未知受众分别说明，排期不等于免费进度或免费观看。loading 表示后台获取中，本回合不要循环调用、强制刷新或换 Bangumi 冒充结果；来源不可用、旧缓存和未收录不代表当天没有更新。

回答要求：
- 先给结论，再给必要明细；明确区分实时结果、缓存结果、部分完成和未执行。
- 不重复同一错误，不输出内部链路、凭据、完整路径或无意义的“请稍后重试”。
- 若确实缺少必要对象，说明已检查什么以及只缺哪一个信息。
- 搜索摘要不等于完整详情。未查询、字段未返回、确实返回空表、请求失败是不同情况；没有演职员字段不能说官方未公布/TMDB未录入。先读取相应详情，仍不足时用已接入的web.search/web.read核实。配置关闭/缺Key/超时应按工具真实错误说明，不能统称无能力。
- 队列计数是瞬时快照；running=0不代表消费者未运行或不会自动执行。只有运行状态明确暂停/关闭时才能如此说明，未读取的状态明确未知。
- 媒体条目含 `open_url` 时使用 `[打开媒体库](原样 open_url)`；没有该字段时不要猜测链接。
- 追漫日历返回精确固定 `calendar_url` 时，Web 可显示 `[打开追漫日历](/discovery/calendar)`；Telegram 提示从 MediaFlux 网页端打开追漫日历，不猜站点地址。没有该安全字段时只用文字引导。"""


class EventJournal(Protocol):
    async def append(self, event: AgentEvent, *, owner: str) -> None: ...


class TurnAdmissionPolicy(Protocol):
    async def begin(self, agent_input: AgentInput) -> Any: ...

    async def is_current(self, token: Any, agent_input: AgentInput) -> bool: ...


class AllowAllTurnAdmission:
    async def begin(self, agent_input: AgentInput) -> None:
        del agent_input

    async def is_current(self, token: Any, agent_input: AgentInput) -> bool:
        del token, agent_input
        return True


@dataclass(frozen=True, slots=True)
class SessionLimits:
    # 复杂的只读观察 -> 批量计划通常需要 5 轮以上；预算仍有硬上限，
    # 但不能让分页本身把正常任务挤成 model_round_budget_exceeded。
    max_model_rounds: int = 12
    max_tool_calls: int = 16
    max_output_tokens: int = 6_000
    context_window_tokens: int = 128_000

    def __post_init__(self) -> None:
        if not 1 <= self.max_model_rounds <= 12:
            raise ValueError("max_model_rounds out of range")
        if not 1 <= self.max_tool_calls <= 32:
            raise ValueError("max_tool_calls out of range")
        if not 128 <= self.max_output_tokens <= 16_000:
            raise ValueError("max_output_tokens out of range")
        if not 16_384 <= self.context_window_tokens <= 2_000_000:
            raise ValueError("context_window_tokens out of range")

    @property
    def effective_output_tokens(self) -> int:
        return min(
            self.max_output_tokens,
            max(1_024, self.context_window_tokens // 4),
        )


class AgentSession:
    """领域无关、事件驱动、可暂停确认的 Agent Kernel。"""

    def __init__(
        self,
        *,
        model: ModelAdapter,
        catalog: ToolCatalog,
        retriever: CapabilityRetriever,
        pipeline: ToolPipeline,
        state_store: SessionStateStore,
        coordinator: TurnCoordinator | None = None,
        journal: EventJournal | None = None,
        limits: SessionLimits | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        turn_admission: TurnAdmissionPolicy | None = None,
    ) -> None:
        if pipeline.catalog is not catalog:
            raise ValueError("AgentSession and ToolPipeline must share one catalog")
        if pipeline.state_store is not state_store:
            raise ValueError("AgentSession and ToolPipeline must share one state store")
        self.model = model
        self.catalog = catalog
        self.retriever = retriever
        self.pipeline = pipeline
        self.state_store = state_store
        self.coordinator = coordinator or TurnCoordinator()
        self.journal = journal
        self.limits = limits or SessionLimits()
        self.system_prompt = str(system_prompt or DEFAULT_SYSTEM_PROMPT).strip()
        self.turn_admission = turn_admission or AllowAllTurnAdmission()
        # 只串行化极短的“取得 generation + 注册 active turn + 保存安全输入”窗口；
        # 已确认写操作一旦开始就不会被后续聊天抢占。
        self._start_lock = CrossLoopAsyncLock()
        # 已确认写操作脱离客户端流后仍必须持有强引用直到可信终态。
        # 普通聊天仍遵循“消费者断开即取消”，两者不能共享取消语义。
        self._detached_tasks: set[asyncio.Task[None]] = set()

    async def run(self, agent_input: AgentInput) -> AsyncIterator[AgentEvent]:
        """运行一轮并实时产生事实事件；消费者断开时取消当前回合。"""
        async for event in self._run_background(
            lambda queue: self._drive(agent_input, queue)
        ):
            yield event

    async def confirm(
        self,
        *,
        owner: str,
        session_id: str,
        plan_id: str,
        request_id: str = "",
        channel: str = "api",
    ) -> AsyncIterator[AgentEvent]:
        async for event in self._run_background(
            lambda queue: self._drive_confirmation(
                owner=str(owner or "").strip(),
                session_id=str(session_id or "").strip(),
                plan_id=str(plan_id or "").strip(),
                request_id=str(request_id or "").strip() or secrets.token_urlsafe(12),
                channel=str(channel or "api").strip().lower(),
                queue=queue,
            ),
            cancel_on_consumer_close=False,
        ):
            yield event

    async def cancel(self, *, owner: str, session_id: str) -> bool:
        return await self.coordinator.cancel(
            owner=str(owner or "").strip(),
            session_id=str(session_id or "").strip(),
            reason="user_cancelled",
        )

    async def cancel_effect(
        self,
        *,
        owner: str,
        session_id: str,
        plan_id: str,
        request_id: str = "",
    ) -> bool:
        state = await self.state_store.load(owner=owner, session_id=session_id)
        if state.generation <= 0:
            return False
        lease = PublicationLease(
            owner=owner,
            session_id=session_id,
            generation=state.generation,
            turn_id=secrets.token_urlsafe(12),
            request_id=request_id or secrets.token_urlsafe(12),
        )
        token = CancellationToken()

        async def ignore_progress(_payload: Mapping[str, Any]) -> None:
            return None

        context = ToolCallContext(
            owner=owner,
            session_id=session_id,
            request_id=lease.request_id,
            turn_id=lease.turn_id,
            lease=lease,
            cancellation=token,
            report_progress=ignore_progress,
        )
        return await self.pipeline.cancel_effect(plan_id, context=context)

    async def _run_background(
        self,
        producer: Callable[[asyncio.Queue[AgentEvent | None]], Awaitable[None]],
        *,
        cancel_on_consumer_close: bool = True,
    ) -> AsyncIterator[AgentEvent]:
        queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
        async def drive_to_completion() -> None:
            try:
                await producer(queue)
            finally:
                # 完成信号归传输边界统一拥有；初始化、审计或收尾失败也不能
                # 留下永远等待 queue.get() 的消费者。
                queue.put_nowait(None)

        task = asyncio.create_task(drive_to_completion())
        producer_finished = False
        try:
            while True:
                item = await queue.get()
                if item is None:
                    producer_finished = True
                    break
                yield item
        finally:
            if task.done() or producer_finished:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            elif cancel_on_consumer_close:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            else:
                # 用户已经确认的副作用不能由刷新、断网或关闭标签页撤销。
                # 生产者继续完成审计、状态提交和领域后置生命周期；队列会在
                # 任务结束后与任务一同释放，不再依赖已断开的流消费者。
                self._detached_tasks.add(task)
                task.add_done_callback(self._detached_task_finished)

    def _detached_task_finished(self, task: asyncio.Task[None]) -> None:
        self._detached_tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            logger.error(
                "Agent detached confirmation failed type=%s",
                type(error).__name__,
                exc_info=(type(error), error, error.__traceback__),
            )

    async def _drive(
        self,
        agent_input: AgentInput,
        queue: asyncio.Queue[AgentEvent | None],
    ) -> None:
        lease: PublicationLease | None = None
        token: CancellationToken | None = None
        factory: EventFactory | None = None
        try:
            validated_selection = None
            candidate_context = None
            selection_tool = None
            if "selection" in agent_input.metadata:
                current_state = await self.state_store.load(
                    owner=agent_input.owner, session_id=agent_input.session_id,
                )
                validated_selection = await validate_selection(
                    agent_input.metadata["selection"], state=current_state,
                    store=self.pipeline.reference_store,
                )
                try:
                    selection_tool = self.catalog.get("ingest.submit")
                    if selection_tool.effect is ToolEffect.READ:
                        raise KeyError("selection must require confirmation")
                except KeyError as exc:
                    raise ToolPipelineError("候选预览能力暂不可用", code="selection_unavailable") from exc
            if agent_input.metadata.get("candidate_context") and validated_selection is None:
                current_state = await self.state_store.load(owner=agent_input.owner, session_id=agent_input.session_id)
                candidate_view = current_state.metadata.get("ux_candidate_view") or {}
                items = candidate_view.get("items") or []
                if not items:
                    raise SelectionInvalidError()
                candidate_context = await validate_selection({
                    "ref": agent_input.metadata["candidate_context"], "positions": [items[0]["position"]], "target": "guangya",
                }, state=current_state, store=self.pipeline.reference_store, for_preview=False)
            contextual_message = self._contextual_message(agent_input)
            sensitive_input = contains_sensitive_credential(contextual_message)
            if candidate_context is not None:
                contextual_message += "\n已校验的回复批次，仅可从该引用解析候选编号：" + str(candidate_context.guard.ref)
            admission_token = await self.turn_admission.begin(agent_input)
            with session_scope_guard(agent_input.owner, agent_input.session_id):
                async with self._start_lock:
                    if await self.coordinator.has_protected_turn(
                        owner=agent_input.owner,
                        session_id=agent_input.session_id,
                    ):
                        raise SessionBusyError("confirmed effect is executing")
                    begin_options = {}
                    if validated_selection is not None:
                        begin_options["selection_guard"] = validated_selection.guard
                    elif candidate_context is not None:
                        begin_options["selection_guard"] = candidate_context.guard
                    lease, state = await self.state_store.begin_turn(
                        owner=agent_input.owner,
                        session_id=agent_input.session_id,
                        request_id=agent_input.request_id,
                        **begin_options,
                    )
                    token = await self.coordinator.begin(lease)
                    messages = self._restore_messages(state)
                    current_user_index = len(messages)
                    messages.append(ModelMessage(role="user", content=contextual_message))
                    # 输入接纳和持久化必须位于同一个 start window，不能等事件发布：
                    # journal/observer 可能挂起，此时下一条追问已取得新 generation。
                    # 这里只写通过凭据检测的用户输入，迟到的模型/工具仍不得提交。
                    if not sensitive_input:
                        await self.state_store.commit(lease, conversation=self._persisted_conversation(
                            messages, current_user_index=current_user_index,
                            original_message=agent_input.message, reply_context=agent_input.reply_context,
                            prior_conversation=state.conversation,
                        ))
            factory = EventFactory(
                session_id=agent_input.session_id,
                turn_id=lease.turn_id,
                request_id=agent_input.request_id,
            )

            async def publish(
                event_type: AgentEventType,
                payload: Mapping[str, Any] | None = None,
                **extra: Any,
            ) -> None:
                token.raise_if_cancelled()
                if not await self.coordinator.is_current(lease, token):
                    raise asyncio.CancelledError("superseded")
                if not await self.state_store.is_current(lease):
                    raise asyncio.CancelledError("stale_generation")
                if not await self.turn_admission.is_current(
                    admission_token, agent_input
                ):
                    raise asyncio.CancelledError("runtime_changed")
                event = factory.create(event_type, payload, **extra)
                if self.journal is not None:
                    await self.journal.append(event, owner=agent_input.owner)
                await queue.put(event)

            await publish(
                AgentEventType.TURN_STARTED,
                {
                    "channel": agent_input.channel,
                    "generation": lease.generation,
                },
            )
            if sensitive_input:
                await publish(
                    AgentEventType.TURN_FAILED,
                    {
                        "code": "sensitive_input",
                        "message": "消息包含疑似凭据，未发送给模型。",
                    },
                )
                return

            selection = self.retriever.retrieve(
                contextual_message,
                self.catalog,
                context={
                    "owner": agent_input.owner,
                    "session_id": agent_input.session_id,
                    "channel": agent_input.channel,
                    "reference_kinds": tuple(state.ref_kinds),
                    **self._capability_retrieval_context(state),
                    "has_current_reference": bool(agent_input.reply_context.get("text")),
                },
            )
            discovery = CapabilityDiscovery(
                self.catalog,
                context={
                    "owner": agent_input.owner, "session_id": agent_input.session_id,
                    "request_id": agent_input.request_id, "channel": agent_input.channel,
                    "generation": lease.generation, "turn_id": lease.turn_id,
                    "reference_kinds": tuple(state.ref_kinds),
                },
                maximum=getattr(self.retriever, "maximum", 10),
            )
            selected_tools = discovery.window(selection.tools)
            if validated_selection is not None:
                selected_tools = (selection_tool,)
            await publish(
                AgentEventType.CAPABILITIES_SELECTED,
                {
                    "tools": [tool.name for tool in selected_tools],
                    "count": len(selected_tools),
                },
            )
            selected_names = {tool.name for tool in selected_tools}
            selected_model_names = {tool.model_name for tool in selected_tools}
            tool_definitions = tuple(
                tool.model_definition() for tool in selected_tools
            )
            total_tool_calls = 0
            total_usage: dict[str, int] = {}

            async def progress(payload: Mapping[str, Any]) -> None:
                await publish(AgentEventType.TOOL_PROGRESS, payload)

            tool_context = ToolCallContext(
                owner=agent_input.owner,
                session_id=agent_input.session_id,
                request_id=agent_input.request_id,
                turn_id=lease.turn_id,
                lease=lease,
                cancellation=token,
                report_progress=progress,
                capability_search=discovery.search,
                selection_arguments=validated_selection.arguments if validated_selection else None,
                resource_candidate_ref=candidate_context.guard.ref if candidate_context else "",
            )
            # 新的自然语言回合会明确取代尚未确认的旧计划。若只提升
            # generation 而不撤销票据，历史卡片会永久显示“待确认”，
            # 但点击时又只能得到 stale plan，形成确认死状态。
            if state.pending_effect_plan_id:
                await self.pipeline.cancel_effect(
                    state.pending_effect_plan_id,
                    context=tool_context,
                )

            if validated_selection is not None:
                call = ModelToolCall(
                    call_id=f"selection_{lease.turn_id}", name="ingest.submit",
                    arguments=dict(validated_selection.arguments),
                )
                await publish(AgentEventType.TOOL_STARTED, {
                    "call_id": call.call_id, "tool": call.name,
                    "label": public_tool_label(call.name), "effect": selection_tool.effect.value,
                })
                result = await self.pipeline.execute(call.name, call.arguments, context=tool_context)
                messages.append(ModelMessage(role="assistant", tool_calls=(call,)))
                messages.append(ModelMessage(
                    role="tool", content=result.outcome.model_message(),
                    tool_call_id=call.call_id, tool_name=call.name,
                ))
                plan = result.effect_plan
                if plan is not None:
                    await publish(AgentEventType.EFFECT_APPROVAL_REQUIRED, {
                        "call_id": call.call_id, "tool": call.name, "label": public_tool_label(call.name),
                        "plan": plan.public_dict(), "result": dict(result.outcome.public_content),
                    })
                else:
                    await publish(AgentEventType.TOOL_COMPLETED, {
                        "call_id": call.call_id, "tool": call.name, "label": public_tool_label(call.name),
                        "elapsed_ms": result.elapsed_ms, "result": dict(result.outcome.public_content),
                    })
                answer = "" if plan else format_public_result(dict(result.outcome.public_content))
                if answer:
                    messages.append(ModelMessage(role="assistant", content=answer))
                await self.state_store.commit(lease, conversation=self._persisted_conversation(
                    messages, current_user_index=current_user_index, original_message=agent_input.message, reply_context=agent_input.reply_context,
                    prior_conversation=state.conversation,
                ))
                await publish(AgentEventType.TURN_COMPLETED, {
                    "status": "approval_required" if plan else "success", "answer": answer,
                    "plan_id": plan.plan_id if plan else "", "usage": {}, "model_calls": 0, "tool_calls": 1,
                })
                return

            for round_index in range(self.limits.max_model_rounds):
                token.raise_if_cancelled()
                await publish(AgentEventType.MODEL_STARTED, {"round": round_index + 1})
                text_parts: list[str] = []
                calls: list[ModelToolCall] = []
                finish_reason = ""
                # 只要本轮已经执行过工具，最后一次模型调用就专门用于汇总。
                # 否则第 12 轮仍执行工具后没有第 13 轮收束，会真实完成调用却
                # 对用户报 model_round_budget_exceeded，并丢失可续跑上下文。
                final_synthesis_round = (
                    round_index == self.limits.max_model_rounds - 1
                    and total_tool_calls > 0
                )
                request_tools = () if final_synthesis_round else tool_definitions
                request_system_prompt = self.system_prompt
                if final_synthesis_round:
                    request_system_prompt += (
                        "\n\n本次是最终汇总轮次：不得调用任何工具。请只基于已经取得的工具事实"
                        "给出简洁结论；若任务尚未完整完成，明确写‘部分完成’，说明未执行的"
                        "写操作，并提示用户可继续，不得声称已生成不存在的确认计划。"
                    )
                request = ModelRequest(
                    system_prompt=request_system_prompt,
                    messages=self._bounded_model_messages(
                        messages,
                        history_end=current_user_index,
                        tool_definitions=request_tools,
                        system_prompt=request_system_prompt,
                    ),
                    tools=request_tools,
                    max_output_tokens=self.limits.effective_output_tokens,
                    round_index=round_index,
                )
                async for model_event in self.model.stream(request, cancellation=token):
                    token.raise_if_cancelled()
                    if model_event.type is ModelEventType.TEXT_DELTA:
                        if model_event.text:
                            text_parts.append(model_event.text)
                            await publish(
                                AgentEventType.MODEL_DELTA,
                                {"delta": model_event.text, "round": round_index + 1},
                            )
                    elif model_event.type is ModelEventType.TOOL_CALL_COMPLETED:
                        call = model_event.tool_call
                        if call is not None:
                            calls.append(call)
                            try:
                                public_tool_name = self.catalog.get(call.name).name
                            except KeyError:
                                public_tool_name = call.name
                            await publish(
                                AgentEventType.MODEL_TOOL_CALL,
                                {
                                    "call_id": call.call_id,
                                    "tool": public_tool_name,
                                    "label": public_tool_label(public_tool_name),
                                    "argument_keys": sorted(
                                        str(key)[:80] for key in call.arguments
                                    )[:50],
                                    "round": round_index + 1,
                                },
                            )
                    elif model_event.type is ModelEventType.USAGE:
                        for key, value in model_event.usage.items():
                            try:
                                total_usage[key] = total_usage.get(key, 0) + max(
                                    0, int(value)
                                )
                            except (TypeError, ValueError):
                                continue
                    elif model_event.type is ModelEventType.FINISH:
                        finish_reason = model_event.finish_reason

                assistant_text = "".join(text_parts).strip()
                if calls and final_synthesis_round:
                    for call in calls:
                        error = ToolPipelineError(
                            "最终汇总轮次不再执行工具",
                            code="not_executed_final_round",
                        )
                        await publish(
                            AgentEventType.TOOL_FAILED,
                            {
                                "call_id": call.call_id,
                                "tool": call.name,
                                "label": public_tool_label(call.name),
                                "code": error.code,
                                "message": str(error),
                            },
                        )
                    final_text = assistant_text or (
                        "部分完成：已完成的检查结果已保留，但模型未在本轮预算内形成完整方案。"
                        "未生成确认卡的写操作均未执行；你可以回复继续，我会基于现有上下文接着处理。"
                    )
                    messages.append(ModelMessage(role="assistant", content=final_text))
                    await self.state_store.commit(
                        lease,
                        conversation=self._persisted_conversation(
                            messages,
                            current_user_index=current_user_index,
                            original_message=agent_input.message, reply_context=agent_input.reply_context,
                            prior_conversation=state.conversation,
                        ),
                    )
                    await publish(
                        AgentEventType.TURN_COMPLETED,
                        {
                            "status": "partial",
                            "answer": final_text,
                            "finish_reason": "model_round_budget_exceeded",
                            "usage": total_usage,
                            "model_calls": round_index + 1,
                            "tool_calls": total_tool_calls,
                        },
                    )
                    return
                if calls:
                    if total_tool_calls + len(calls) > self.limits.max_tool_calls:
                        raise ToolPipelineError(
                            "本轮工具调用次数超过安全上限",
                            code="tool_budget_exceeded",
                        )
                    total_tool_calls += len(calls)
                    messages.append(
                        ModelMessage(
                            role="assistant",
                            content=assistant_text,
                            tool_calls=tuple(calls),
                        )
                    )
                    for call_index, call in enumerate(calls):
                        if (
                            call.name not in selected_names
                            and call.name not in selected_model_names
                        ):
                            error = ToolPipelineError(
                                "该工具不在本轮候选能力中",
                                code="tool_not_available",
                            )
                            await publish(
                                AgentEventType.TOOL_FAILED,
                                {
                                    "call_id": call.call_id,
                                    "tool": call.name,
                                    "label": public_tool_label(call.name),
                                    "code": error.code,
                                    "message": str(error),
                                },
                            )
                            messages.append(self._tool_error_message(call, error))
                            continue
                        tool = self.catalog.get(call.name)
                        canonical_call = ModelToolCall(
                            call_id=call.call_id,
                            name=tool.name,
                            arguments=call.arguments,
                        )
                        if tool.effect is not ToolEffect.READ:
                            await publish(
                                AgentEventType.EFFECT_PREVIEW_STARTED,
                                {
                                    "call_id": call.call_id,
                                    "tool": tool.name,
                                    "label": public_tool_label(tool.name),
                                },
                            )
                        await publish(
                            AgentEventType.TOOL_STARTED,
                            {
                                "call_id": call.call_id,
                                "tool": tool.name,
                                "label": public_tool_label(tool.name),
                                "effect": tool.effect.value,
                            },
                        )
                        discovery_checkpoint = discovery.checkpoint()
                        if tool.name == DISCOVERY_TOOL:
                            current_state = await self.state_store.load(
                                owner=agent_input.owner, session_id=agent_input.session_id,
                            )
                            discovery.context["reference_kinds"] = tuple(current_state.ref_kinds)
                        try:
                            result = await self.pipeline.execute(
                                canonical_call.name,
                                canonical_call.arguments,
                                context=tool_context,
                            )
                        except ToolPipelineError as exc:
                            if tool.name == DISCOVERY_TOOL:
                                # 仅撤销本次失败的发现，不能丢掉同批之前成功的结果。
                                discovery.restore(discovery_checkpoint)
                            await publish(
                                AgentEventType.TOOL_FAILED,
                                {
                                    "call_id": call.call_id,
                                    "tool": tool.name,
                                    "label": public_tool_label(tool.name),
                                    "code": exc.code,
                                    "message": str(exc),
                                },
                            )
                            messages.append(self._tool_error_message(call, exc))
                            continue
                        if tool.name == DISCOVERY_TOOL and result.outcome.public_content.get("ok") is False:
                            discovery.restore(discovery_checkpoint)
                        if result.effect_plan is not None:
                            plan = result.effect_plan
                            await publish(
                                AgentEventType.EFFECT_APPROVAL_REQUIRED,
                                {
                                    "call_id": call.call_id,
                                    "tool": tool.name,
                                    "label": public_tool_label(tool.name),
                                    "plan": plan.public_dict(),
                                    "result": dict(result.outcome.public_content),
                                },
                            )
                            messages.append(
                                ModelMessage(
                                    role="tool",
                                    content=result.outcome.model_message(),
                                    tool_call_id=call.call_id,
                                    tool_name=call.name,
                                )
                            )
                            # Provider 已被要求禁止并行工具，但兼容服务仍可能
                            # 违规一次返回多个调用。写操作在此暂停等待人工确认，
                            # 后续调用必须明确闭合为“未执行”，不能留下缺少
                            # tool result 的无效协议历史。
                            for deferred_call in calls[call_index + 1 :]:
                                deferred_error = ToolPipelineError(
                                    "前序写操作需要人工确认，本调用未执行",
                                    code="not_executed_after_approval",
                                )
                                await publish(
                                    AgentEventType.TOOL_FAILED,
                                    {
                                        "call_id": deferred_call.call_id,
                                        "tool": deferred_call.name,
                                        "label": public_tool_label(
                                            deferred_call.name
                                        ),
                                        "code": deferred_error.code,
                                        "message": str(deferred_error),
                                    },
                                )
                                messages.append(
                                    self._tool_error_message(
                                        deferred_call, deferred_error
                                    )
                                )
                            await self.state_store.commit(
                                lease,
                                conversation=self._persisted_conversation(
                                    messages,
                                    current_user_index=current_user_index,
                                    original_message=agent_input.message, reply_context=agent_input.reply_context,
                                    prior_conversation=state.conversation,
                                ),
                            )
                            await publish(
                                AgentEventType.TURN_COMPLETED,
                                {
                                    "status": "approval_required",
                                    "plan_id": plan.plan_id,
                                    "usage": total_usage,
                                    "model_calls": round_index + 1,
                                    "tool_calls": total_tool_calls,
                                },
                            )
                            return
                        await publish(
                            AgentEventType.TOOL_COMPLETED,
                            {
                                "call_id": call.call_id,
                                "tool": tool.name,
                                "label": public_tool_label(tool.name),
                                "elapsed_ms": result.elapsed_ms,
                                "result": dict(result.outcome.public_content),
                            },
                        )
                        messages.append(
                            ModelMessage(
                                role="tool",
                                content=result.outcome.model_message(),
                                tool_call_id=call.call_id,
                                tool_name=call.name,
                            )
                        )
                    additions = discovery.consume()
                    if additions:
                        # 只在完整工具批次之后更新Schema；同一模型批次不能猜新工具名绕过初选。
                        token.raise_if_cancelled()
                        selected_tools = discovery.window(selected_tools, additions)
                        selected_names = {tool.name for tool in selected_tools}
                        selected_model_names = {tool.model_name for tool in selected_tools}
                        tool_definitions = tuple(tool.model_definition() for tool in selected_tools)
                        await publish(AgentEventType.CAPABILITIES_SELECTED, {
                            "tools": [tool.name for tool in selected_tools],
                            "count": len(selected_tools), "reason": "discovery", "round": round_index + 2,
                        })
                    continue

                final_text = assistant_text
                if not final_text:
                    raise ToolPipelineError(
                        "模型没有返回回答或工具调用",
                        code="empty_model_response",
                    )
                messages.append(ModelMessage(role="assistant", content=final_text))
                await self.state_store.commit(
                    lease,
                    conversation=self._persisted_conversation(
                        messages,
                        current_user_index=current_user_index,
                        original_message=agent_input.message, reply_context=agent_input.reply_context,
                        prior_conversation=state.conversation,
                    ),
                )
                await publish(
                    AgentEventType.TURN_COMPLETED,
                    {
                        "status": "success",
                        "answer": final_text,
                        "finish_reason": finish_reason or "stop",
                        "usage": total_usage,
                        "model_calls": round_index + 1,
                        "tool_calls": total_tool_calls,
                    },
                )
                return

            raise ToolPipelineError(
                "模型在调用轮次上限内未完成任务",
                code="model_round_budget_exceeded",
            )
        except (asyncio.CancelledError, StalePublicationError) as exc:
            if factory is not None:
                event = factory.create(
                    AgentEventType.TURN_CANCELLED,
                    {"reason": str(exc) or (token.reason if token else "cancelled")},
                )
                if self.journal is not None:
                    await self.journal.append(event, owner=agent_input.owner)
                await queue.put(event)
        except SelectionInvalidError as exc:
            failure_factory = EventFactory(
                session_id=agent_input.session_id,
                turn_id=secrets.token_urlsafe(12), request_id=agent_input.request_id,
            )
            # 拒绝发生在 begin_turn 前；不记会话事件，不污染现有确认/历史。
            await queue.put(failure_factory.create(
                AgentEventType.TURN_FAILED,
                {"code": "selection_invalid", "message": str(exc)},
            ))
        except SessionBusyError:
            busy_factory = factory or EventFactory(
                session_id=agent_input.session_id,
                turn_id=secrets.token_urlsafe(12),
                request_id=agent_input.request_id,
            )
            event = busy_factory.create(
                AgentEventType.TURN_FAILED,
                {
                    "code": "effect_in_progress",
                    "message": "已确认的写操作正在执行，请等待完成后再继续。",
                },
            )
            if self.journal is not None:
                await self.journal.append(event, owner=agent_input.owner)
            await queue.put(event)
        except ToolPipelineError as exc:
            failure_factory = factory or EventFactory(
                session_id=agent_input.session_id,
                turn_id=secrets.token_urlsafe(12),
                request_id=agent_input.request_id,
            )
            event = failure_factory.create(
                AgentEventType.TURN_FAILED,
                {"code": exc.code, "message": str(exc)},
            )
            if self.journal is not None:
                await self.journal.append(event, owner=agent_input.owner)
            await queue.put(event)
        except ModelProviderError as exc:
            logger.warning("Agent model provider failed type=%s", type(exc).__name__)
            failure_factory = factory or EventFactory(
                session_id=agent_input.session_id,
                turn_id=secrets.token_urlsafe(12),
                request_id=agent_input.request_id,
            )
            event = failure_factory.create(
                AgentEventType.TURN_FAILED,
                {
                    "code": "model_provider_error",
                    "message": _provider_failure_message(exc),
                },
            )
            if self.journal is not None:
                await self.journal.append(event, owner=agent_input.owner)
            await queue.put(event)
        except Exception as exc:  # noqa: BLE001 - final turn fault boundary
            logger.error("Agent turn failed type=%s", type(exc).__name__)
            failure_factory = factory or EventFactory(
                session_id=agent_input.session_id,
                turn_id=secrets.token_urlsafe(12),
                request_id=agent_input.request_id,
            )
            event = failure_factory.create(
                AgentEventType.TURN_FAILED,
                {"code": "internal_error", "message": "Agent 运行失败"},
            )
            if self.journal is not None:
                await self.journal.append(event, owner=agent_input.owner)
            await queue.put(event)
        finally:
            if lease is not None and token is not None:
                await self.coordinator.finish(lease, token)

    async def _drive_confirmation(
        self, *, owner: str, session_id: str, plan_id: str, request_id: str,
        channel: str, queue: asyncio.Queue[AgentEvent | None],
    ) -> None:
        if not owner or not session_id or not plan_id:
            return
        try:
            # 外层保护覆盖票据认领、远端执行、审计及 conversation 回执落盘。
            with session_scope_guard(owner, session_id, kind="effect"):
                async with self._start_lock:
                    state = await self.state_store.load(owner=owner, session_id=session_id)
                    lease = PublicationLease(
                        owner=owner,
                        session_id=session_id,
                        generation=state.generation,
                        turn_id=secrets.token_urlsafe(12),
                        request_id=request_id,
                    )
                    token = await self.coordinator.begin(lease, protected=True)
                factory = EventFactory(
                    session_id=session_id, turn_id=lease.turn_id, request_id=request_id
                )

                async def publish(
                    event_type: AgentEventType, payload: Mapping[str, Any] | None = None
                ) -> None:
                    token.raise_if_cancelled()
                    event = factory.create(event_type, payload)
                    if self.journal is not None:
                        await self.journal.append(event, owner=owner)
                    await queue.put(event)

                async def progress(payload: Mapping[str, Any]) -> None:
                    await publish(AgentEventType.TOOL_PROGRESS, payload)

                context = ToolCallContext(
                    owner=owner,
                    session_id=session_id,
                    request_id=request_id,
                    turn_id=lease.turn_id,
                    lease=lease,
                    cancellation=token,
                    report_progress=progress,
                )

                async def remember_result(
                    *,
                    tool_name: str,
                    content: str,
                    public_content: str,
                    candidate_result: Mapping[str, Any] | None = None,
                ) -> None:
                    """把确定性确认终态写回会话，供下一轮续问直接引用。"""
                    safe_content = str(content or "").strip()
                    if not safe_content:
                        return
                    conversation = [dict(item) for item in state.conversation]
                    item = ModelMessage(
                        role="assistant",
                        content=(
                            "已确认操作的可信系统结果（不是待执行计划）：\n"
                            + safe_content
                        ),
                        tool_name=tool_name,
                    ).to_dict()
                    safe_public_content = str(public_content or "").strip()
                    if safe_public_content:
                        item["public_content"] = safe_public_content
                    updates = (StateUpdate("pending_effect_plan_id", plan_id, mode="clear_if_equals"),)
                    if candidate_result:
                        item["candidate_result_ref"] = candidate_result["ref"]
                        updates += (StateUpdate("metadata.ux_candidate_result", dict(candidate_result)),)
                    conversation.append(item)
                    try:
                        await self.state_store.commit(
                            lease,
                            conversation=conversation,
                            updates=updates,
                        )
                    except StalePublicationError:
                        raise
                    except Exception as exc:  # noqa: BLE001 - 已执行副作用不得被状态写回遮蔽
                        logger.warning(
                            "Agent 确认结果写回会话失败 type=%s", type(exc).__name__
                        )
                try:
                    await publish(
                        AgentEventType.TURN_STARTED,
                        {
                            "channel": channel,
                            "generation": lease.generation,
                            "kind": "confirmation",
                        },
                    )
                    await publish(
                        AgentEventType.TOOL_STARTED,
                        {"plan_id": plan_id, "kind": "confirmed_effect"},
                    )
                    result = await self.pipeline.execute_confirmed(plan_id, context=context)
                    public_result = dict(result.outcome.public_content)
                    candidate_data = public_result.get("data")
                    candidate_items = candidate_data.get("items") if isinstance(candidate_data, Mapping) else []
                    candidate_items = candidate_items if isinstance(candidate_items, list) else []
                    await remember_result(
                        tool_name=result.tool.name,
                        content=result.outcome.model_message(),
                        public_content=format_public_result(public_result),
                        candidate_result={
                            "ref": result.arguments.get("resource_candidates_ref"), "text": format_public_result(public_result),
                            "target": result.arguments.get("target"),
                            "handled_positions": [item.get("position") for item in candidate_items if isinstance(item, dict) and type(item.get("position")) is int and item.get("status") != "failed"],
                        } if result.tool.name == "ingest.submit" and result.arguments.get("source_type") == "resource_candidates" else None,
                    )
                    if public_result.get("ok") is False:
                        code = str(public_result.get("status") or "effect_failed")[:80]
                        await publish(
                            AgentEventType.EFFECT_FAILED,
                            {
                                "plan_id": plan_id,
                                "tool": result.tool.name,
                                "code": code,
                                "message": str(
                                    public_result.get("error")
                                    or public_result.get("summary")
                                    or "执行未完成"
                                )[:500],
                                "elapsed_ms": result.elapsed_ms,
                                "result": public_result,
                            },
                        )
                        await publish(
                            AgentEventType.TURN_FAILED,
                            {"code": code, "message": "已确认操作未能完成"},
                        )
                    else:
                        await publish(
                            AgentEventType.EFFECT_COMPLETED,
                            {
                                "plan_id": plan_id,
                                "tool": result.tool.name,
                                "elapsed_ms": result.elapsed_ms,
                                "result": public_result,
                            },
                        )
                        await publish(
                            AgentEventType.TURN_COMPLETED,
                            {"status": "effect_completed", "plan_id": plan_id},
                        )
                except (asyncio.CancelledError, StalePublicationError) as exc:
                    event = factory.create(
                        AgentEventType.TURN_CANCELLED,
                        {"reason": str(exc) or token.reason},
                    )
                    if self.journal is not None:
                        await self.journal.append(event, owner=owner)
                    await queue.put(event)
                except ToolPipelineError as exc:
                    # 未领取票据的重复/失效确认不能用旧快照覆盖另一 worker 已提交的
                    # 成功回执，也不能清除仍在执行的计划。错误仅投影到本次请求。
                    if not isinstance(exc, ConfirmationClaimError):
                        await remember_result(
                            tool_name="confirmed_effect",
                            content=f"执行失败：{str(exc)[:500]}（错误码：{exc.code[:80]}）",
                            public_content=format_public_result(
                                {
                                    "ok": False,
                                    "status": exc.code,
                                    "summary": str(exc),
                                },
                                fallback="确认执行未能完成。",
                            ),
                        )
                    await publish(
                        AgentEventType.EFFECT_FAILED,
                        {"plan_id": plan_id, "code": exc.code, "message": str(exc)},
                    )
                    await publish(
                        AgentEventType.TURN_FAILED,
                        {"code": exc.code, "message": str(exc)},
                    )
                except Exception as exc:  # noqa: BLE001 - confirmed-effect fault boundary
                    logger.error("Agent confirmed effect failed type=%s", type(exc).__name__)
                    await remember_result(
                        tool_name="confirmed_effect",
                        content=(
                            "执行状态未知：确认执行发生内部错误"
                            "（错误码：internal_error），请先查询真实业务状态再决定是否重试。"
                        ),
                        public_content=(
                            "❌ 确认执行发生内部错误，请先查询真实业务状态再决定是否重试。"
                        ),
                    )
                    await publish(
                        AgentEventType.EFFECT_FAILED,
                        {
                            "plan_id": plan_id,
                            "code": "internal_error",
                            "message": "确认执行失败",
                        },
                    )
                    await publish(
                        AgentEventType.TURN_FAILED,
                        {"code": "internal_error", "message": "确认执行失败"},
                    )
                finally:
                    await self.coordinator.finish(lease, token)

        except SessionBusyError:
            event = EventFactory(
                session_id=session_id, turn_id=secrets.token_urlsafe(12), request_id=request_id,
            ).create(AgentEventType.TURN_FAILED, {
                "code": "effect_in_progress", "message": "另一项会话操作正在执行，请稍后重试。",
            })
            if self.journal is not None:
                await self.journal.append(event, owner=owner)
            await queue.put(event)

    @staticmethod
    def _estimated_tokens(value: object) -> int:
        return estimated_tokens(value)

    def _bounded_model_messages(
        self,
        messages: Sequence[ModelMessage],
        *,
        history_end: int,
        tool_definitions: Sequence[Mapping[str, Any]],
        system_prompt: str | None = None,
    ) -> tuple[ModelMessage, ...]:
        return bounded_model_messages(
            messages,
            history_end=history_end,
            tool_definitions=tool_definitions,
            system_prompt=system_prompt or self.system_prompt,
            context_window_tokens=self.limits.context_window_tokens,
            output_tokens=self.limits.effective_output_tokens,
        )

    @staticmethod
    def _compact_tool_content(content: str, *, maximum: int) -> str:
        return compact_tool_content(content, maximum=maximum)

    @staticmethod
    def _capability_retrieval_context(
        state: SessionState,
    ) -> dict[str, Any]:
        """提取有界跨轮线索供本地召回使用，不替模型裁决用户意图。"""
        recent_users: list[str] = []
        recent_tools: list[str] = []
        seen_tools: set[str] = set()
        tool_weights: dict[str, float] = {}
        turns_back = 0
        for item in reversed(state.conversation[-24:]):
            if not isinstance(item, Mapping):
                continue
            role = str(item.get("role") or "").strip().lower()
            if role == "user" and len(recent_users) < 3:
                text = AgentSession._with_reply_context(
                    str(item.get("content") or "").strip(), item.get("reply_context"),
                )
                if text:
                    recent_users.append(text[:2_000])
            if role == "user":
                turns_back += 1
                if turns_back >= 3:
                    break
            tool_name = str(item.get("tool_name") or "").strip()
            if tool_name and tool_name not in seen_tools and len(recent_tools) < 6:
                recent_tools.append(tool_name)
                seen_tools.add(tool_name)
                tool_weights[tool_name] = 0.35 ** turns_back
            calls = item.get("tool_calls")
            if isinstance(calls, Sequence) and not isinstance(
                calls, (str, bytes, bytearray)
            ):
                for call in reversed(calls):
                    if not isinstance(call, Mapping):
                        continue
                    name = str(call.get("name") or "").strip()
                    if name and name not in seen_tools and len(recent_tools) < 6:
                        recent_tools.append(name)
                        seen_tools.add(name)
                        tool_weights[name] = 0.35 ** turns_back
        return {
            "recent_user_messages": tuple(recent_users),
            "recent_tool_names": tuple(recent_tools),
            "recent_tool_weights": tool_weights,
        }

    @staticmethod
    def _contextual_message(agent_input: AgentInput) -> str:
        return AgentSession._with_reply_context(agent_input.message, agent_input.reply_context)

    @staticmethod
    def _with_reply_context(message: str, reply: object) -> str:
        reply_text = str(reply.get("text") or "").strip() if isinstance(reply, Mapping) else ""
        if not reply_text:
            return message
        return (
            f"{message}\n\n"
            "<reply_context purpose=reference_resolution>\n"
            f"{reply_text[:2_000]}\n"
            "</reply_context>"
        )

    @staticmethod
    def _persisted_conversation(
        messages: Sequence[ModelMessage],
        *,
        current_user_index: int,
        original_message: str,
        reply_context: Mapping[str, Any] | None = None,
        prior_conversation: Sequence[Mapping[str, Any]] = (),
    ) -> list[dict[str, Any]]:
        stored: list[dict[str, Any]] = []
        public_history: dict[tuple, list[dict[str, Any]]] = {}
        for prior in prior_conversation[-60:]:
            if not isinstance(prior, Mapping):
                continue
            restored = dict(prior)
            if prior.get("role") == "user":
                restored["content"] = AgentSession._with_reply_context(str(prior.get("content") or ""), prior.get("reply_context"))
            key = tuple(str(restored.get(field) or "") for field in ("role", "content", "tool_call_id", "tool_name"))
            preserved = {key: prior[key] for key in ("content", "public_content", "candidate_result_ref") if isinstance(prior.get(key), str)}
            if isinstance(prior.get("reply_context"), Mapping):
                preserved["reply_context"] = {"text": str(prior["reply_context"].get("text") or "")[:2_000]}
            public_history.setdefault(key, []).append(preserved)
        for index, message in enumerate(messages):
            item = message.to_dict()
            key = tuple(str(item.get(field) or "") for field in ("role", "content", "tool_call_id", "tool_name"))
            prior = public_history.get(key)
            if index < current_user_index and prior:
                item.update(prior.pop(0))
            tool_calls = item.get("tool_calls")
            if isinstance(tool_calls, list):
                item["tool_calls"] = [
                    {**dict(call), "arguments": {}}
                    for call in tool_calls
                    if isinstance(call, Mapping)
                ]
            stored.append(item)
        if 0 <= current_user_index < len(stored):
            stored[current_user_index] = {
                **stored[current_user_index],
                "content": original_message,
            }
            if reply_context and str(reply_context.get("text") or "").strip():
                stored[current_user_index]["reply_context"] = {"text": str(reply_context["text"]).strip()[:2_000]}
        return stored

    @staticmethod
    def _restore_messages(state: SessionState) -> list[ModelMessage]:
        messages: list[ModelMessage] = []
        for item in state.conversation[-60:]:
            if not isinstance(item, Mapping):
                continue
            try:
                restored = dict(item)
                if item.get("role") == "user":
                    restored["content"] = AgentSession._with_reply_context(str(item.get("content") or ""), item.get("reply_context"))
                message = ModelMessage.from_dict(restored)
            except Exception as exc:  # noqa: BLE001 - isolate malformed persisted rows
                logger.warning("忽略无效 Agent 会话消息 type=%s", type(exc).__name__)
                continue
            if message.role in {"user", "assistant", "tool"}:
                messages.append(message)
        return messages

    @staticmethod
    def _tool_error_message(
        call: ModelToolCall, error: ToolPipelineError
    ) -> ModelMessage:
        content = json.dumps(
            {
                "ok": False,
                "status": "error",
                "code": error.code,
                "error": str(error),
                "instruction": "请根据错误修正参数、换用本轮其他工具，或向用户明确说明限制。",
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return ModelMessage(
            role="tool",
            content=content,
            tool_call_id=call.call_id,
            tool_name=call.name,
        )
