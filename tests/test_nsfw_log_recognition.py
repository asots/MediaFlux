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
from app.modules.nsfw import NsfwRecognizer, clear_nsfw_cache
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


def make_log(provider='clean_title', source='', *, name='SODAH-003.mp4', status='success'):
    log_id = db.add_organize_log(
        'guangya', 'input', 'NSFW/SODAH-003/' + name if status == 'success' else name, 'video', status, '',
        source_dir_id=source, provider=provider, external_id='SODAH-003' if provider else '',
        media_type='movie', original_name=name, current_name=name,
        original_parent_id='input', current_parent_id='existing', legacy_incomplete=False,
    )
    db.add_organize_log_items(log_id, [{
        'file_id': 'video', 'role': 'video', 'original_parent_id': 'input',
        'original_name': name, 'current_parent_id': 'existing', 'current_name': name,
        'size': 100, 'etag': 'etag', 'status': status,
    }])
    return log_id


@pytest.mark.parametrize('filename', ['SSNI596C.mp4', 'SSNI-596C.mp4', 'SSNI396C.mp4'])
def test_compact_skipped_log_search_preview_and_reorganize_share_identity(log_case, filename):
    service, _ = log_case
    number = 'SSNI-396' if '396' in filename else 'SSNI-596'
    log_id = make_log('', 'adult-source', name=filename, status='skipped')
    metadata = {'provider': 'JavBus', 'id': number, 'number': number,
                'title': '归档测试', 'release_date': '2019-01-01'}
    remote = {
        'video': GuangYaFile('video', filename, False, 100, 'etag', 'existing'),
        'category': GuangYaFile('category', 'NSFW', True, parent_id='archive'),
    }
    service.client.file_info.side_effect = remote.get
    service.client.list_dir.side_effect = lambda parent: [item for item in remote.values() if item.parent_id == parent]
    service.client.move.side_effect = lambda ids, parent: setattr(remote[ids[0]], 'parent_id', parent)
    service.client.rename.side_effect = lambda fid, name: setattr(remote[fid], 'name', name)
    recognizer = NsfwRecognizer('http://synthetic.invalid/compact-log')
    clear_nsfw_cache()
    try:
        with patch.object(recognizer.client, '_get', side_effect=lambda path, **kw: [metadata] if path.endswith('/search') else metadata), \
                patch.object(service.organizer, '_nsfw_recognizer_lease', return_value=nullcontext(recognizer)), \
                patch.object(service.scraper, 'search_candidates') as tmdb:
            candidates = service.search_candidates(log_id, filename)
            assert len(candidates) == 1
            assert service.search_candidates(log_id, number + 'C.mp4') == candidates
            candidate = candidates[0]
            assert candidate['provider'] == 'metatube'
            automatic = recognizer.match(filename)
            assert automatic is not None
            assert automatic.external_id == candidate['external_id']
            preview = service.preview_reorganize(log_id, '', 'movie', provider=candidate['provider'], external_id=candidate['external_id'])
            assert preview['match']['external_id'] == 'JavBus:' + number
            assert preview['target_path'].startswith('NSFW/' + number)
            service.client.move.assert_not_called()
            service.client.rename.assert_not_called()
            remote['media'] = GuangYaFile('media', preview['media_dir'], True, parent_id='category')
            with patch.object(service, '_notify_reorganize_result', return_value=[]):
                result = service.reorganize(log_id, 'compact-log-reorganize', preview['version'], '', 'movie',
                    provider=candidate['provider'], external_id=candidate['external_id'])
            assert result['success'] is True
            assert remote['video'].parent_id == 'media'
            assert remote['video'].name == preview['file_name']
            saved = db.get_organize_log(log_id)
            assert saved is not None
            assert saved['status'] == 'success'
            assert saved['provider'] == 'metatube'
            assert saved['external_id'] == 'JavBus:' + number
            tmdb.assert_not_called()
    finally:
        recognizer.close()
        clear_nsfw_cache()


@pytest.mark.parametrize('filename,error', [('SSNI596C.mp4', '番号与原文件不一致'),
                                           ('unknown.mp4', '未提取到可校验番号')])
def test_manual_candidate_cannot_override_missing_or_different_source_identity(log_case, filename, error):
    service, _ = log_case
    log_id = make_log('', 'adult-source', name=filename, status='skipped')
    recognizer = MagicMock()
    recognizer.resolve.return_value = (MatchResult(provider='metatube'), {'number': 'SSNI-999'})
    with patch.object(service.organizer, '_nsfw_recognizer_lease', return_value=nullcontext(recognizer)), \
            pytest.raises(ValueError, match=error):
        service.preview_reorganize(log_id, '', 'movie', provider='metatube', external_id='JavBus:SSNI-999')
    service.client.move.assert_not_called()
    service.client.rename.assert_not_called()


@pytest.mark.parametrize('filename', ['SSNI596C.mp4', 'SSNI-596C.mp4'])
def test_compact_identity_is_shared_by_directory_and_telegram_validation(filename):
    from app.modules.directory_media import DirectoryInspection, MediaSnapshot
    from app.modules.directory_scrape import DirectoryScrapeService
    from app.modules.directory_scrape_errors import DirectoryScrapeRequestError
    from app.modules.organize_confirmations import _validate_metatube_confirmation_identity

    rules = OrganizeRules(nsfw_enabled=True, nsfw_exclusive=True)
    inspection = DirectoryInspection(
        directory_id='source', directory_name='incoming', media_type='movie', suggested_query='',
        videos=(MediaSnapshot('video', 'source', filename, 100, 'etag', 'video', ''),),
        companions=(), counts={}, mixed=False, fingerprint='synthetic',
    )
    payload = {'directory': 'incoming', 'files': [{'name': filename}]}
    DirectoryScrapeService._validate_metatube_source_identity(inspection, {'number': 'SSNI-596'}, rules)
    _validate_metatube_confirmation_identity(payload, {'number': 'SSNI-596'}, rules)
    with pytest.raises(DirectoryScrapeRequestError, match='番号与当前目录不一致'):
        DirectoryScrapeService._validate_metatube_source_identity(inspection, {'number': 'SSNI-999'}, rules)
    with pytest.raises(ValueError, match='番号与待确认文件不一致'):
        _validate_metatube_confirmation_identity(payload, {'number': 'SSNI-999'}, rules)


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
