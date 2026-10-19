from __future__ import annotations

import copy
from typing import cast
from unittest.mock import Mock

from app.clients.guangya import GuangYaClient, GuangYaFile
from app.modules.organize import Organizer, OrganizeRules
from app.modules.scraper import MatchResult, TMDBScraper
from tests.support import IsolatedDatabaseTestCase, release_parse_result


class _TreeClient:
    def __init__(self, file: GuangYaFile) -> None:
        self.tree = {"source": [file], "archive": []}
        self.info = {
            "source": GuangYaFile("source", "待整理", True, parent_id="0"),
            "archive": GuangYaFile("archive", "整理", True, parent_id="0"),
        }

    def list_dir(self, file_id: str):
        return list(self.tree.get(file_id, []))

    def file_info(self, file_id: str):
        return self.info.get(file_id)


def _scraper(match: MatchResult, episode_count: int) -> TMDBScraper:
    scraper = TMDBScraper()
    scraper.match = Mock(side_effect=lambda *args, **kwargs: copy.deepcopy(match))
    scraper.get_detail = Mock(return_value={
        "id": 82684, "genres": [], "origin_country": [],
        "seasons": [{"season_number": 1, "episode_count": episode_count}],
    })
    return scraper


class OrganizePlanContractTests(IsolatedDatabaseTestCase):
    def preview(self, file: GuangYaFile, scraper: TMDBScraper):
        return Organizer(client=cast(GuangYaClient, _TreeClient(file)), scraper=scraper).organize(
            "source", OrganizeRules(
                target_dir_id="archive", small_file_mb=0, region_split=False,
                year_split=False, clean_empty=False, link_strm=False, notify_enabled=False,
                media_info_enabled=False, media_probe_enabled=False,
            ), dry_run=True, automatic=True,
        )

    def test_public_preview_preserves_source_separately_from_cached_target(self):
        filename = "Example.Show.S03E49.mkv"
        file = GuangYaFile("cached", filename, False, 1024, "etag", "source")
        match = MatchResult(
            tmdb_id="82684", external_id="82684", provider="tmdb",
            title="Example Show", year="2018", media_type="tv", confidence=1.0,
            status="matched", matched_by="search", metadata={"episode_mapping": {
                "source_season": 3, "source_episode": 49,
                "target_season": 1, "target_episode": 49,
                "mode": "absolute", "reason": "absolute_numbering_rolled_over_tmdb_seasons",
                "confidence": 1.0, "changed": True,
            }},
        )
        scraper = _scraper(match, 92)
        scraper.parse_media = Mock(side_effect=lambda filename, parent_path="", _match=None: release_parse_result(
            {"type": "tv", "season": 1, "episode": 49},
            filename=filename, parent_path=parent_path, source_season=3, source_episode=49,
        ))
        outcomes = []
        for _ in range(2):
            plans, stats = self.preview(file, scraper)
            self.assertEqual(len(plans), 1)
            plan = plans[0]
            self.assertEqual(plan.action, "move")
            self.assertEqual((plan.season, plan.episode), (1, 49))
            self.assertEqual((plan.source_season, plan.source_episode), (3, 49))
            self.assertEqual(stats["need_confirm"], 0)
            outcomes.append((plan.action, plan.target_path, plan.new_name, plan.source_season, plan.source_episode))
        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(file.name, filename)
        self.assertEqual(file.parent_id, "source")

    def test_automatic_proof_cannot_bypass_final_position_rejection(self):
        file = GuangYaFile("proof", "Example.Show.S01E99.mkv", False, 1024, "etag", "source")
        proof = {
            "version": 2, "kind": "tmdb_tv_episode_identity",
            "provider": "tmdb", "external_id": "82684", "media_type": "tv",
            "confidence": 0.85, "recognition_threshold": 0.9,
            "automatic_match_preset": "balanced", "global_threshold": 0.9,
            "strong_title_score": 0.85, "candidate_count": 1, "candidate_gap": 1.0,
            "decision_constraints": [], "selected_constraints": [],
            "expected_year": "", "candidate_year": "2018",
            "source_title_key": "exampleshow", "matched_title_key": "exampleshow",
            "source_position": {"season": 1, "episode": 1},
            "target_position": {"season": 1, "episode": 1},
            "position_validation": {
                "required": True, "passed": True, "season": 1,
                "episode": 1, "episode_count": 12, "reason": "episode_verified",
            },
        }
        scraper = _scraper(MatchResult(
            tmdb_id="82684", external_id="82684", provider="tmdb",
            title="Example Show", year="2018", media_type="tv", confidence=0.85,
            threshold=0.9, status="low_confidence", need_confirm=True,
            error="匹配置信度 85% 低于严格模式阈值 90%",
            matched_by="search", metadata={"verified_automatic_identity_proof": proof},
        ), 12)

        plans, stats = self.preview(file, scraper)

        self.assertEqual(len(plans), 1)
        plan = plans[0]
        self.assertEqual(plan.action, "skip")
        assert plan.match is not None
        self.assertTrue(plan.match.need_confirm)
        self.assertEqual(plan.match.metadata["final_position_validation"]["reason"], "episode_out_of_range")
        self.assertTrue(plan.match.metadata["tmdb_detail_force_refreshed"])
        self.assertFalse(plan.match.metadata.get("verified_automatic_identity_proof_accepted"))
        self.assertEqual(stats["need_confirm"], 1)
