"""历史 NSFW 日志不随来源配置变化退回 TMDB；隔离数据库与云端。"""
from contextlib import nullcontext
from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from app import database as db
from app.clients.guangya import GuangYaFile
from app.modules.organize import OrganizeRules
from app.modules.organize_correction import OrganizeCorrectionService
from app.modules.scraper import Candidate, MatchResult, TMDBScraper
from tests.support import isolated_test_database


@pytest.fixture
def log_case():
    with isolated_test_database():
        client = MagicMock()
        client.file_info.return_value = None
        service = OrganizeCorrectionService(client=client, scraper=TMDBScraper())
        rules = OrganizeRules(
            target_dir_id='archive', region_split=False, year_split=False,
            link_strm=False, clean_empty=False, notify_enabled=False,
            nsfw_enabled=True, nsfw_source_ids='["adult-source"]',
            nsfw_category_name='NSFW',
        )
        with patch('app.modules.organize_correction.OrganizeRules.from_config', return_value=rules):
            yield service, rules
        service.close()
        service.scraper.close()


def make_log(provider='clean_title', source=''):
    name = 'SODAH-003.mp4'
    log_id = db.add_organize_log(
        'guangya', 'input', 'NSFW/SODAH-003/' + name, 'video', 'success', '',
        source_dir_id=source, provider=provider, external_id='SODAH-003',
        media_type='movie', original_name=name, current_name=name,
        original_parent_id='input', current_parent_id='existing', legacy_incomplete=False,
    )
    db.add_organize_log_items(log_id, [{
        'file_id': 'video', 'role': 'video', 'original_parent_id': 'input',
        'original_name': name, 'current_parent_id': 'existing', 'current_name': name,
        'size': 100, 'etag': 'etag', 'status': 'success',
    }])
    return log_id


@pytest.mark.parametrize('provider', ['clean_title', 'metatube'])
@pytest.mark.parametrize('source', ['', 'removed-source', 'adult-source'])
def test_saved_nsfw_provider_selects_nsfw_detail(log_case, provider, source):
    service, _ = log_case
    detail = service.detail(make_log(provider, source))
    assert detail['recognition']['nsfw_only'] is True
    assert detail['recognition']['provider'] == 'metatube'
    assert detail['recognition']['query_placeholder'] == '输入番号或包含番号的文件名'


@pytest.mark.parametrize('provider', ['clean_title', 'metatube'])
def test_disabled_source_config_does_not_change_saved_identity(log_case, provider):
    service, rules = log_case
    disabled = replace(rules, nsfw_enabled=False, nsfw_source_ids='')
    with patch('app.modules.organize_correction.OrganizeRules.from_config', return_value=disabled):
        detail = service.detail(make_log(provider, 'adult-source'))
    assert detail['recognition']['nsfw_only'] is True
    assert disabled.nsfw_enabled is False  # 不修改全局自动扫描规则。


@pytest.mark.parametrize('provider', ['clean_title', 'metatube'])
def test_search_uses_metatube_and_clean_title_fallback(log_case, provider):
    service, _ = log_case
    log_id = make_log(provider)
    recognizer = MagicMock()
    recognizer.candidates.return_value = [Candidate(
        tmdb_id='', title='SODAH-003', year='', score=1.0, media_type='movie',
        provider='metatube', external_id='javbus:sodah003',
    )]
    with (
        patch.object(service.organizer, '_nsfw_recognizer_lease', return_value=nullcontext(recognizer)),
        patch.object(service.scraper, 'search_candidates') as tmdb,
    ):
        assert service.search_candidates(log_id, 'SODAH-003')[0]['provider'] == 'metatube'
        recognizer.candidates.return_value = []
        assert service.search_candidates(log_id, 'SODAH-003')[0]['provider'] == 'clean_title'
        tmdb.assert_not_called()


def test_missing_metatube_config_reports_error_without_tmdb(log_case):
    service, _ = log_case
    log_id = make_log()
    with (
        patch.object(service.scraper, 'search_candidates') as tmdb,
        pytest.raises(ValueError, match='未配置可用的 MetaTube'),
    ):
        service.search_candidates(log_id, 'SODAH-003')
    tmdb.assert_not_called()


@pytest.mark.parametrize('provider', ['clean_title', 'metatube'])
def test_preview_keeps_nsfw_rules_and_rejects_tmdb_candidate(log_case, provider):
    service, _ = log_case
    log_id = make_log(provider)
    recognizer = MagicMock()
    recognizer.resolve.return_value = (
        MatchResult(title='SODAH-003', media_type='movie', provider='metatube',
                    external_id='SODAH-003', status='matched'),
        {'number': 'SODAH-003'},
    )
    with patch.object(service.organizer, '_nsfw_recognizer_lease', return_value=nullcontext(recognizer)):
        preview = service.preview_reorganize(log_id, '', 'movie', provider=provider, external_id='SODAH-003')
    assert preview['target_path'] == 'NSFW/SODAH-003'
    assert preview['rules_snapshot']['nsfw_exclusive'] is True
    with (
        patch.object(service.scraper, 'match_from_tmdb') as tmdb,
        pytest.raises(ValueError, match='请选择有效的成人内容候选'),
    ):
        service.preview_reorganize(log_id, '123', 'movie', provider='tmdb', external_id='123')
    tmdb.assert_not_called()


@pytest.mark.parametrize('provider,source,expected', [
    ('tmdb', '', False), ('', '', False), ('', 'adult-source', True),
])
def test_unidentified_and_ordinary_logs_keep_source_rules(log_case, provider, source, expected):
    service, _ = log_case
    log_id = make_log(provider, source)
    assert service.detail(log_id)['recognition']['nsfw_only'] is expected
    if not expected:
        with patch.object(service.scraper, 'search_candidates', return_value=[]) as tmdb:
            assert service.search_candidates(log_id, '普通电影') == []
        tmdb.assert_called_once()


def test_execution_uses_same_effective_rules_and_persists_nsfw(log_case):
    service, _ = log_case
    log_id = make_log()
    remote = {
        'video': GuangYaFile('video', 'SODAH-003.mp4', False, 100, 'etag', 'existing'),
        'category': GuangYaFile('category', 'NSFW', True, parent_id='archive'),
        'media': GuangYaFile('media', 'SODAH-003', True, parent_id='category'),
    }
    service.client.file_info.side_effect = remote.get
    service.client.list_dir.side_effect = lambda parent: [item for item in remote.values() if item.parent_id == parent]
    service.client.move.side_effect = lambda ids, parent: setattr(remote[ids[0]], 'parent_id', parent)
    service.client.rename.side_effect = lambda file_id, name: setattr(remote[file_id], 'name', name)
    preview = service.preview_reorganize(log_id, '', 'movie', provider='clean_title', external_id='SODAH-003')
    with patch.object(service, '_notify_reorganize_result', return_value=[]):
        result = service._execute_reorganize(log_id, 'nsfw-log-reorganize', preview['version'], preview)
    assert result['success'] is True
    assert remote['video'].parent_id == 'media'
    get_organize_log_result = db.get_organize_log(log_id)
    assert get_organize_log_result is not None
    saved = dict(get_organize_log_result)
    assert (saved['provider'], saved['external_id']) == ('clean_title', 'SODAH-003')
    assert service.detail(log_id)['recognition']['nsfw_only'] is True


def test_execution_still_rejects_changed_destination_rules(log_case):
    service, rules = log_case
    log_id = make_log()
    preview = service.preview_reorganize(log_id, '', 'movie', provider='clean_title', external_id='SODAH-003')
    with (
        patch('app.modules.organize_correction.OrganizeRules.from_config', return_value=replace(rules, target_dir_id='different')),
        patch.object(service, '_verify_items'),
        pytest.raises(ValueError, match='整理规则已变化'),
    ):
        service._execute_reorganize(log_id, 'changed-rules', preview['version'], preview)
    service.client.move.assert_not_called()
    service.client.create_dir.assert_not_called()
