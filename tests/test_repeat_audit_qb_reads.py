"""qB 列表型读取拒绝无效响应，不能把读取失败报告为空任务/空文件。"""
import unittest
from unittest.mock import Mock, patch

from app.agent.provider_models import ProviderGatewayError
from app.agent.providers.qbittorrent import QBittorrentProviderTransport
from app.clients.qbittorrent import QBittorrentClient


class QBReadContractTests(unittest.TestCase):
    def setUp(self):
        self.client = QBittorrentClient('http://synthetic.invalid', api_key='synthetic')
        self.addCleanup(self.client.close)

    def test_empty_list_is_valid_for_all_readers(self):
        with patch.object(self.client, '_get', return_value=Mock(json=lambda: [])):
            self.assertEqual(self.client.list_torrents(), [])
            self.assertEqual(self.client.get_torrent_files('a' * 40), [])
            self.assertFalse(self.client.get_completion('a' * 40))

    def test_invalid_envelope_never_becomes_an_empty_success(self):
        readers = [self.client.list_torrents,
                   lambda: self.client.get_torrent_files('a' * 40),
                   lambda: self.client.get_completion('a' * 40)]
        for payload in (None, {}, {'error': 'unavailable'}, '', 0, [None], ['invalid']):
            for index, reader in enumerate(readers):
                with self.subTest(payload=payload, reader=index), patch.object(
                    self.client, '_get', return_value=Mock(json=lambda: payload)
                ):
                    with self.assertRaises((ValueError, TypeError, KeyError, AttributeError)):
                        reader()

    def test_valid_files_and_torrent_completion_preserve_existing_projection(self):
        with patch.object(self.client, '_get', return_value=Mock(json=lambda: [
            {'index': 7, 'name': 'Show/E01.mkv', 'size': 1024, 'progress': .5},
            {'name': 'Show/E02.mkv', 'size': 2048, 'progress': 1},
        ])):
            files = self.client.get_torrent_files('a' * 40)
        self.assertEqual([(item.index, item.name, item.size, item.progress) for item in files],
                         [(7, 'Show/E01.mkv', 1024, .5), (1, 'Show/E02.mkv', 2048, 1)])
        with patch.object(self.client, '_get', return_value=Mock(json=lambda: [
            {'hash': 'a' * 40, 'name': 'Show', 'state': 'stoppedUP', 'progress': 1},
        ])):
            self.assertTrue(self.client.get_completion('a' * 40))
            self.assertEqual(self.client.list_torrents()[0].name, 'Show')

    def test_provider_reports_unavailable_instead_of_zero_files_and_closes_client(self):
        transport = QBittorrentProviderTransport()
        with patch.object(transport, '_client', return_value=self.client), patch.object(
            self.client, '_get', return_value=Mock(json=lambda: {'error': 'not a file list'})
        ), patch.object(self.client, 'close', wraps=self.client.close) as close:
            with self.assertRaises(ProviderGatewayError) as error:
                transport.execute_read('configured:qbittorrent', 'qb.torrents.files', {'torrent_ref': 'a' * 40})
            self.assertEqual(error.exception.code, 'provider_unavailable')
            close.assert_called_once()
