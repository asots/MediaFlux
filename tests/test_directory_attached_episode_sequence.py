"""紧贴中文片名的目录集号推断：真实解析、人工确认与执行快照回归。"""
import unicodedata
from dataclasses import replace
from unittest.mock import patch

from app.modules.directory_media import (
    DirectoryMediaInspector,
    MediaSnapshot,
    _VideoIdentity,
)
from app.modules.directory_scrape import DirectoryScrapeService, DirectoryScrapeStore
from app.modules.directory_scrape_errors import (
    DirectoryScrapeConflictError,
    DirectoryScrapeRequestError,
)
from app.modules.organize import OrganizeRules
from app.modules.scraper import TMDBScraper
from tests.support import IsolatedDatabaseTestCase
from tests.test_guangya_directory_scrape import (
    _dir,
    _DirectoryScrapeTMDB,
    _file,
    _MutableTreeClient,
)

ROOT_NAME = "05 龙珠改：魔人布欧篇"


def _names(numbers=range(1, 62), *, title="龙珠改布欧篇", edition="龙论版", width=2):
    return [f"{title}{number:0{width}d}（{edition}）.mp4" for number in numbers]


def _identities(names):
    return [
        _VideoIdentity(
            file=MediaSnapshot(
                file_id=f"v-{i}", parent_id="source", name=name, size=1024,
                etag=f"etag-{i}", role="video", relative_dir="",
            ),
            special=False, special_container=False, media_type="movie",
            title=name.rsplit(".", 1)[0], year="", tmdb_id="",
        )
        for i, name in enumerate(names, 1)
    ]


class DirectoryAttachedEpisodeSequenceTests(IsolatedDatabaseTestCase):
    def setUp(self):
        self.parser = TMDBScraper("offline-fixture")
        self.addCleanup(self.parser.close)
        self.scraper = _DirectoryScrapeTMDB()
        # 只替换离线 TMDB 元数据；文件名与父目录均走正式解析器。
        self.scraper.parse_media = self.parser.parse_media
        self.rules = OrganizeRules(
            target_dir_id="archive", small_file_mb=0, region_split=False,
            year_split=False, clean_empty=False, link_strm=False, notify_enabled=False,
        )

    def _cloud(self, names=None, *, rules=None):
        videos = [_file(f"v-{i}", name, "source")
                  for i, name in enumerate(_names() if names is None else names, 1)]
        client = _MutableTreeClient(
            {"source": list(reversed(videos)), "archive": []},
            {"source": _dir("source", ROOT_NAME), "archive": _dir("archive", "媒体库"),
             **{item.file_id: item for item in videos}},
        )
        service = DirectoryScrapeService(
            client=client, scraper=self.scraper, store=DirectoryScrapeStore(),
            rules_loader=lambda: rules or self.rules,
        )
        self.addCleanup(service.close)
        return service, client

    def test_real_61_episode_inspect_requires_manual_without_recognition(self):
        service, _client = self._cloud()
        inspected = service.inspect("owner", "source")
        inspection = service.store.get_inspection("owner", inspected["inspection_id"]).inspection
        self.assertEqual(inspected["directory"]["name"], ROOT_NAME)
        self.assertEqual((inspected["media_type"], inspected["counts"]["video"]), ("tv", 61))
        self.assertTrue(inspected["requires_manual_match"])
        self.assertIn("人工", inspected["manual_match_reason"])
        self.assertEqual([item.episode for item in inspection.videos], list(range(1, 62)))
        self.assertEqual(inspection.pending_videos, ())
        with patch.object(service, "_recognize") as recognize, \
                patch.object(self.scraper, "match") as match:
            result = service.auto_match("owner", inspected["inspection_id"])
        recognize.assert_not_called()
        match.assert_not_called()
        self.assertEqual((result["status"], result["media_type"]), ("requires_manual", "tv"))
        self.assertEqual(result["candidates"], [])

    def test_preview_fixed_match_and_execution_keep_all_61_positions(self):
        service, client = self._cloud()
        inspected = service.inspect("owner", "source")
        preview = service.preview(
            "owner", inspected["inspection_id"], "123", "tv",
            season=2, numbering_mode="standard",
        )
        self.assertEqual(len(preview["plans"]), 61)
        self.assertEqual(len(client.tree["source"]), 61)  # 预览不得移动源文件。
        for plan in preview["plans"]:
            episode = int(plan["file_id"].split("-")[-1])
            self.assertEqual((plan["season"], plan["episode"]), (2, episode))
            mapping = plan["episode_mapping"]
            self.assertEqual((plan["source_episode"], mapping["source_episode"]), (episode, episode))
            self.assertEqual((mapping["target_season"], mapping["target_episode"]), (2, episode))
            self.assertEqual(plan["new_name"], f"钢铁侠：装甲冒险.2009.S02E{episode:02d}.mp4")
            self.assertTrue(plan["target_path"].endswith("/Season 2"))
        result = service.execute_preview("owner", preview["preview_id"])
        self.assertEqual(result["stats"]["moved"], 61)
        self.assertEqual(result["stats"].get("failed", 0), 0)
        for plan in preview["plans"]:
            actual = client.infos[plan["file_id"]]
            self.assertEqual(actual.name, plan["new_name"])
            self.assertEqual(client.infos[actual.parent_id].name, "Season 2")
        self.assertEqual(client.tree["source"], [])

    def test_source_changes_invalidate_preview_and_execution(self):
        for stage in ("preview", "execute"):
            with self.subTest(stage=stage):
                service, client = self._cloud()
                inspected = service.inspect("owner", "source")
                if stage == "execute":
                    preview = service.preview("owner", inspected["inspection_id"], "123", "tv")
                # 删除末集仍有有效的 1..60 连续序列，必须因快照变化而拒绝。
                client.tree["source"] = [v for v in client.tree["source"] if v.file_id != "v-61"]
                with patch.object(self.scraper, "match_from_tmdb") as resolve, \
                        self.assertRaisesRegex(DirectoryScrapeConflictError, "目录内容已变化"):
                    if stage == "preview":
                        service.preview("owner", inspected["inspection_id"], "123", "tv")
                    else:
                        service.execute_preview("owner", preview["preview_id"])
                resolve.assert_not_called()
                self.assertEqual(client.tree["archive"], [])

    def test_real_inspect_rejects_another_complete_work_and_numbered_movies(self):
        cases = {
            "second_work": _names() + _names(range(1, 5), title="灌篮高手"),
            "few_movies": _names(range(1, 4)),
            "year": [name.replace(".mp4", ".2024.mp4") for name in _names(range(1, 5))],
            "tmdb": [name.replace(".mp4", "{tmdb-123}.mp4") for name in _names(range(1, 5))],
        }
        for label, names in cases.items():
            with self.subTest(label=label):
                service, _client = self._cloud(names)
                with self.assertRaisesRegex(DirectoryScrapeRequestError, "多个不同媒体"):
                    service.inspect("owner", "source")

    def test_single_file_and_parser_do_not_inherit_directory_inference(self):
        service, _client = self._cloud()
        service.inspect("owner", "source")
        inspected = service.inspect_file("owner", "v-1")
        inspection = service.store.get_inspection("owner", inspected["inspection_id"]).inspection
        self.assertEqual(inspected["media_type"], "movie")
        self.assertFalse(inspected["requires_manual_match"])
        self.assertEqual(len(inspection.videos), 1)
        self.assertIsNone(inspection.videos[0].episode)
        parsed = self.parser.parse_media(_names([1])[0], ROOT_NAME)
        self.assertEqual((parsed.source_season, parsed.source_episode), (None, None))

    def test_nsfw_inspect_never_calls_attached_sequence_inference(self):
        # for_source 以启用开关和来源白名单收敛 NSFW，不能只设 exclusive。
        service, _client = self._cloud(
            rules=replace(self.rules, nsfw_enabled=True, nsfw_source_ids='["source"]'))
        with patch.object(DirectoryMediaInspector, "_infer_attached_episode_sequence",
                          wraps=DirectoryMediaInspector._infer_attached_episode_sequence) as infer, \
                self.assertRaisesRegex(DirectoryScrapeRequestError, "多个不同媒体"):
            service.inspect("owner", "source")
        infer.assert_not_called()

    def test_nested_files_require_selecting_the_subdirectory_as_root(self):
        service, client = self._cloud(_names(range(1, 5)))
        child = _dir("part", "第二季", "source")
        client.infos[child.file_id] = child
        client.tree[child.file_id] = []
        client.tree["source"].append(child)
        client.move([f"v-{i}" for i in range(1, 5)], child.file_id)
        with self.assertRaises(DirectoryScrapeRequestError):
            service.inspect("owner", "source")
        inspected = service.inspect("owner", child.file_id)
        inspection = service.store.get_inspection("owner", inspected["inspection_id"]).inspection
        self.assertEqual(inspected["media_type"], "tv")
        self.assertTrue(inspected["requires_manual_match"])
        self.assertEqual([item.episode for item in inspection.videos], [1, 2, 3, 4])
        self.assertTrue(all(item.relative_dir == "" for item in inspection.videos))

    def test_four_files_nfkc_and_three_digit_sequences_are_supported(self):
        for width in (2, 3):
            with self.subTest(width=width):
                names = _names(range(1, 5), width=width)
                if width == 3:
                    names = [name.translate(str.maketrans("0123456789", "０１２３４５６７８９"))
                             for name in names]
                identities = [replace(item, title=unicodedata.normalize("NFKC", item.title))
                              for item in _identities(names)]
                inferred = DirectoryMediaInspector._infer_attached_episode_sequence(identities)
                self.assertEqual([item.file.episode for item in inferred], [1, 2, 3, 4])
                self.assertTrue(all(item.title == "龙珠改布欧篇" and item.media_type == "tv"
                                    for item in inferred))

    def test_insufficient_or_conflicting_evidence_is_unchanged(self):
        base = _identities(_names(range(1, 5)))
        cases = {
            "gap": _identities(_names([1, 2, 4, 5])),
            "duplicate": _identities(_names([1, 2, 2, 3])),
            "no_padding": _identities(_names(range(10, 14))),
            "width": _identities(_names([1, 2]) + _names([3, 4], width=3)),
            "edition": _identities(_names([1, 2]) + _names([3, 4], edition="剧场版")),
            "exact_title": _identities(_names([1, 2]) + _names([3, 4], title="龙珠改布欧编")),
            "directory": [replace(item, file=replace(item.file, relative_dir=f"part-{i // 2}"))
                          for i, item in enumerate(base)],
            "nested_directory": [replace(item, file=replace(item.file, relative_dir="子目录"))
                                 for item in base],
            "rewritten_title": [replace(item, title="人工指定标题") for item in base],
            "year": [replace(item, year="2024") for item in base],
            "tmdb": [replace(item, tmdb_id="123") for item in base],
            "season": [replace(item, file=replace(item.file, season=2)) for item in base],
            "episode": [replace(item, file=replace(item.file, episode=9)) for item in base],
            "not_version": _identities(_names(range(1, 5), edition="国语")),
        }
        for label, identities in cases.items():
            with self.subTest(label=label):
                self.assertEqual(DirectoryMediaInspector._infer_attached_episode_sequence(identities),
                                 identities)
        with self.assertRaises(DirectoryScrapeRequestError):
            DirectoryMediaInspector._select_primary_media_group(base)  # 默认不得开启推断。
