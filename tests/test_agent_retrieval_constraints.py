"""召回阶段只切分意图/排除条件，不授予或拒绝业务执行权限。"""
from __future__ import annotations

import pytest

from app.agent.domain_catalog import build_tool_specs
from app.agent.kernel.capabilities import CapabilityRetriever, ToolCatalog, _query_terms
from app.agent.kernel.ports.existing_actions import catalog_from_tool_specs
from tests.test_agent_kernel_core import read_tool


@pytest.fixture(scope='module')
def catalog():
    return catalog_from_tool_specs(build_tool_specs())


@pytest.mark.parametrize('titles', [('9号秘事', '黑镜'), ('银河边界', '深海回声')])
@pytest.mark.parametrize('constraints', [
    '不找资源、不刷新、不播放、不修改。',
    '不要找资源，不要刷新，不要播放，不要修改。',
    '别找资源；别刷新；别播放；别修改。',
])
def test_independent_library_queries_survive_negative_action_terms(catalog, titles, constraints):
    message = f'查一下 Jellyfin 里是否有《{titles[0]}》，再统计一下《{titles[1]}》实际收录了多少集。只查询现有媒体库，{constraints}'
    selection = CapabilityRetriever().retrieve(message, catalog)
    assert {'library.search', 'library.count_series_episodes'} <= set(selection.names), selection.names
    assert len(selection.names) <= 10
    assert not {'library.trigger_patrol_now', 'media_proxy.restart_instance', 'indexer.search_resources'} & set(selection.names)


@pytest.mark.parametrize('title', ['《不要抬头》', '「禁止播放」', '“不要删除”', '"别用播放器"', '`不要修改`'])
def test_quoted_title_is_not_a_negative_action(title):
    message = f'搜索 {title}，不要下载。'
    positive, terms, excluded = _query_terms(message)
    assert title.casefold() in positive
    assert '搜索' in terms
    assert '下载' in excluded and '下载' not in terms
    assert '抬头' not in excluded and '删除' not in excluded and '播放' not in excluded


@pytest.mark.parametrize('message', ['不用再跟踪这个任务了', '不要再跟踪了', '不要下载'])
def test_negative_only_request_retains_its_cancellation_intent(message):
    positive, terms, excluded = _query_terms(message)
    assert positive == message and terms and not excluded


def test_negative_clause_does_not_remove_same_term_from_positive_goal():
    positive, terms, excluded = _query_terms('搜索媒体库，不要搜索资源站。')
    assert '搜索媒体库' in positive
    assert terms['搜索'] and '搜索' not in excluded
    assert excluded['资源站']


def test_related_metadata_cannot_reinsert_a_negated_neighbor():
    from dataclasses import replace
    source = replace(read_tool('cloud.list', description='检查目录', examples=('检查目录',)),
                     metadata={'related_tools': ('cloud.erase',)})
    negative = read_tool('cloud.erase', description='删除目录', examples=('删除文件',))
    neutral = read_tool('cloud.inspect', description='读取对象')
    selection = CapabilityRetriever(minimum=1, maximum=2).retrieve('检查目录，不要删除文件', ToolCatalog([source, negative, neutral]))
    assert selection.names == ('cloud.list',)


def test_negation_is_ranking_only_and_does_not_remove_tool_discoverability():
    tool = read_tool('cloud.erase', description='删除对象', examples=('删除对象',))
    catalog = ToolCatalog([tool])
    CapabilityRetriever(minimum=1, maximum=1).retrieve('检查目录，不要删除对象', catalog)
    assert catalog.get(tool.name) is tool
    assert tool in catalog.visible({})


def test_read_tool_disclaimer_is_not_misread_as_a_download_capability():
    search = read_tool('source.search', description='搜索资源站；不会自动下载', examples=('搜索资源站',))
    download = read_tool('source.download', description='下载资源', examples=('下载资源',))
    catalog = ToolCatalog([search, download])
    selection = CapabilityRetriever(minimum=1, maximum=1).retrieve('搜索资源站，不要下载', catalog)
    assert selection.names == ('source.search',)


@pytest.mark.parametrize('message', ['不是让你删除，是让你检查目录', '并非叫你删除，检查目录就行', '没叫你删除，只检查目录'])
def test_correction_clause_keeps_positive_goal(message):
    positive, terms, excluded = _query_terms(message)
    assert '检查目录' in positive and '检查' in terms
    assert '删除' in excluded and '删除' not in terms


def test_visibility_requirements_are_applied_before_constraint_ranking():
    from dataclasses import replace
    hidden = replace(read_tool('library.hidden', description='检查媒体库'),
                     availability=lambda context: False)
    requires_ref = replace(read_tool('library.bound', description='检查媒体库'),
                           context_requirements=frozenset({'library_ref'}))
    ordinary = read_tool('library.search', description='检查媒体库')
    selection = CapabilityRetriever(minimum=1, maximum=3).retrieve(
        '检查媒体库，不要下载', ToolCatalog([hidden, requires_ref, ordinary]))
    assert selection.names == ('library.search',)


def test_resource_search_survives_a_no_download_constraint(catalog):
    selection = CapabilityRetriever().retrieve('先搜索资源，但不要提交下载', catalog)
    assert 'indexer.search_resources' in selection.names


def test_repeating_a_negative_clause_does_not_cancel_a_positive_goal():
    _, positive, excluded = _query_terms('搜索媒体库，不要搜索资源站，不要搜索站外，不要搜索下载目录。')
    assert positive['搜索'] and '搜索' not in excluded
    assert excluded['资源站'] and excluded['站外'] and excluded['下载']


def test_current_input_is_not_reintroduced_as_unfiltered_history(catalog):
    message = '查一下 Jellyfin 里是否有《9号秘事》，再统计一下《黑镜》实际收录了多少集。只查询现有媒体库，不找资源、不刷新、不播放、不修改。'
    retriever = CapabilityRetriever()
    plain = retriever.retrieve(message, catalog)
    with_current = retriever.retrieve(message, catalog, context={'recent_user_messages': (message,)})
    assert with_current.names == plain.names
    assert with_current.scores == plain.scores


def test_native_session_first_window_includes_both_read_goals(catalog):
    """原生Session当前user落盘后的召回上下文；脚本模型，不冒充真实LLM。"""
    import asyncio
    from app.agent.kernel.session import AgentSession
    from app.agent.kernel.pipeline import ToolPipeline
    from app.agent.kernel.state import AgentInput, InMemorySessionStateStore
    from app.agent.kernel.model import ModelEvent, ModelEventType
    from tests.test_agent_kernel_core import ScriptedModel

    async def run():
        model = ScriptedModel([[ModelEvent(ModelEventType.TEXT_DELTA, text='窗口测试'),
                                ModelEvent(ModelEventType.FINISH, finish_reason='stop')]])
        store = InMemorySessionStateStore()
        session = AgentSession(model=model, catalog=catalog, retriever=CapabilityRetriever(),
                               state_store=store, pipeline=ToolPipeline(catalog=catalog, state_store=store))
        message = '查一下 Jellyfin 里是否有《9号秘事》，再统计一下《黑镜》实际收录了多少集。只查询现有媒体库，不找资源、不刷新、不播放、不修改。'
        _ = [event async for event in session.run(AgentInput(owner='test-owner', session_id='test-window', message=message))]
        names = {tool['name'] for tool in model.requests[0].tools}
        assert {'library__search', 'library__count_series_episodes'} <= names
        assert model.requests[0].messages[-1].content == message
    asyncio.run(run())
