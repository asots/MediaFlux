"""历史识别知识的默认/损坏别名不能破坏整个词库或制造单字别名。"""

from tests.support import assert_present
import json

from app import database as db
from app.modules import recognition_knowledge as knowledge
from app.modules.scraper import TMDBScraper
from tests.support import IsolatedDatabaseTestCase, isolated_test_database


class KnowledgeHistoryAuditTests(IsolatedDatabaseTestCase):
    def setUp(self):
        self.enterContext(isolated_test_database())
        knowledge.reset_runtime_state_for_tests()
        self.addCleanup(knowledge.reset_runtime_state_for_tests)
        self.entry = knowledge.create_entry({
            'knowledge_type': 'release_group', 'canonical_value': 'ArchiveTeam', 'aliases': [],
        })

    def _historical_aliases(self, value):
        raw = json.dumps(value, ensure_ascii=False)
        with db.get_conn() as conn:
            conn.execute('UPDATE recognition_knowledge SET aliases_json=? WHERE id=?', (raw, self.entry['id']))
        knowledge.invalidate_active_cache()
        return raw

    def test_schema_default_empty_aliases_keeps_canonical_recognition_and_relearning_guard(self):
        self._historical_aliases([])
        assert (knowledge.lookup('ArchiveTeam')) is not None
        assert (knowledge.lookup_any('ArchiveTeam')) is not None
        self.assertIsNone(TMDBScraper._unknown_release_group_candidate('[ArchiveTeam] Correct Anime S01E01.mkv'))

    def test_invalid_alias_shapes_do_not_block_other_entries_or_modify_historical_bytes(self):
        for value in (None, 42, True, {'龘': 'not-an-alias-list'}, '龘麤'):
            with self.subTest(value=value):
                raw = self._historical_aliases(value)
                assert (knowledge.lookup('Loli-House')) is not None
                assert (knowledge.lookup('ArchiveTeam')) is not None
                self.assertIsNone(knowledge.lookup('龘'))
                self.assertTrue(any(row['id'] == self.entry['id'] for row in knowledge.list_entries()['items']))
                with db.get_conn() as conn:
                    self.assertEqual(conn.execute('SELECT aliases_json FROM recognition_knowledge WHERE id=?', (self.entry['id'],)).fetchone()[0], raw)

    def test_restarting_index_and_explicit_repair_preserve_identity_and_disabled_state(self):
        self._historical_aliases(None)
        knowledge.reset_runtime_state_for_tests()
        db.init_db()
        lookup_result = knowledge.lookup('ArchiveTeam')
        assert lookup_result is not None
        self.assertEqual(lookup_result['id'], self.entry['id'])
        repaired = knowledge.update_entry(self.entry['id'], {'aliases': ['ArchiveAlias'], 'disabled': True})
        self.assertEqual(repaired['id'], self.entry['id'])
        self.assertTrue(repaired['disabled'])
        self.assertIsNone(knowledge.lookup('ArchiveAlias'))
        lookup_any_result = knowledge.lookup_any('ArchiveAlias')
        assert lookup_any_result is not None
        self.assertEqual(lookup_any_result['id'], self.entry['id'])
        knowledge.reset_runtime_state_for_tests()
        lookup_any_result_2 = knowledge.lookup_any('ArchiveAlias')
        assert lookup_any_result_2 is not None
        self.assertTrue(lookup_any_result_2['disabled'])

    def test_valid_aliases_keep_order_and_canonical_is_not_duplicated(self):
        self._historical_aliases(['ArchiveTeam', 'ArchiveAlias', '别名'])
        row = knowledge.get_entry(self.entry['id'])
        assert row is not None
        self.assertEqual(row['aliases'], ['ArchiveTeam', 'ArchiveAlias', '别名'])
        self.assertEqual(assert_present(knowledge.lookup('别名'))['id'], self.entry['id'])
