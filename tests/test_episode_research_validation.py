"""复杂季集研究的机器证明门禁；全部使用隔离合成元数据。"""
from __future__ import annotations

import copy
import unittest

from app.modules.episode_research import EpisodeEvidenceReader, EpisodeResearchError, normalize_case


def case_payload():
    return {"identity": "Fixture Show", "reason": "文件集号超出 TMDB 记录范围",
            "directory": "/private/Fixture Show",
            "files": [{"file_id": f"private-{n}", "name": f"Fixture.Show.S01E{n:02d}.NF.WEB-DL.mkv",
                       "season": 1, "episode": n, "size": 1024} for n in range(11, 15)],
            "candidates": [{"tmdb_id": "100", "title": "Fixture Show", "year": "2018", "media_type": "tv"}]}


GROUP_ID = "0123456789abcdef01234567"
OTHER_GROUP_ID = "1123456789abcdef01234567"


class EvidenceClient:
    def __init__(self):
        self.requests = []
        self.closed = False
        self.data = {
            "/tv/100": {"id": 100, "name": "Fixture Show", "first_air_date": "2018-04-25",
                        "seasons": [{"season_number": 0, "episode_count": 4}, {"season_number": 1, "episode_count": 10}]},
            "/tv/100/alternative_titles": {"id": 100, "results": []},
            "/tv/100/episode_groups": {"id": 100, "results": [{"id": GROUP_ID, "name": "Streaming Order", "group_count": 1, "episode_count": 14}]},
            f"/tv/episode_group/{GROUP_ID}": {"id": GROUP_ID, "name": "Streaming Order", "groups": [{
                "order": 1, "name": "Season 1", "episodes": [
                    {"id": 1000+n, "order": n-1, "season_number": 1 if n <= 10 else 0,
                     "episode_number": n if n <= 10 else n-10, "name": f"Episode {n}"} for n in range(1, 15)
                ]}]},
            "/tv/100/season/1": {"season_number": 1, "episodes": [
                {"id": 1000+n, "season_number": 1, "episode_number": n, "name": f"Regular {n}"} for n in range(1, 11)]},
            "/tv/100/season/0": {"season_number": 0, "episodes": [
                {"id": 1010+n, "season_number": 0, "episode_number": n, "name": f"Special {n}"} for n in range(1, 5)]},
        }

    def get(self, path, **kwargs):
        self.requests.append(path)
        return copy.deepcopy(self.data[path])

    def close(self):
        self.closed = True


class EpisodeResearchValidationTests(unittest.TestCase):
    def setUp(self):
        self.client = EvidenceClient()
        self.case = normalize_case(case_payload())
        self.reader = EpisodeEvidenceReader(self.case, client=self.client)
        self.addCleanup(self.reader.close)

    def prove(self, group=GROUP_ID):
        self.reader.inspect_candidate(0)
        self.reader.list_groups(0)
        self.reader.read_group(0, group)
        return self.reader.validate(0, group)

    def test_group_order_compiles_all_original_files_to_current_stable_episode_ids(self):
        proposal = self.prove()
        self.assertEqual(proposal["status"], "verified")
        self.assertEqual(proposal["case_key"], self.case["case_key"])
        self.assertEqual([(x["target_season"], x["target_episode"], x["episode_id"]) for x in proposal["mappings"]],
                         [(0, n, 1010+n) for n in range(1, 5)])
        self.assertEqual([x["source_episode"] for x in proposal["mappings"]], [11, 12, 13, 14])
        self.assertTrue(all("api_key" not in x["url"] for x in proposal["evidence"]))
        self.assertEqual(len(self.client.requests), 5)

    def test_case_projection_does_not_include_internal_ids_or_absolute_paths(self):
        self.assertNotIn("private-", repr(self.case))
        self.assertNotIn("/private/", repr(self.case))
        self.assertEqual(self.case["files"][0]["name"], case_payload()["files"][0]["name"])

    def test_cached_selected_group_must_be_revalidated_against_current_episode_ids(self):
        self.client.data["/tv/100/season/0"]["episodes"][0]["id"] = 9999
        with self.assertRaises(EpisodeResearchError):
            self.prove()

    def test_group_not_in_candidate_index_is_never_requested(self):
        self.reader.inspect_candidate(0)
        self.reader.list_groups(0)
        with self.assertRaises(EpisodeResearchError):
            self.reader.read_group(0, OTHER_GROUP_ID)
        self.assertNotIn(f"/tv/episode_group/{OTHER_GROUP_ID}", self.client.requests)

    def test_conflicting_valid_group_orders_cannot_be_resolved_by_model_preference(self):
        original = copy.deepcopy(self.client.data[f"/tv/episode_group/{GROUP_ID}"])
        original["id"] = OTHER_GROUP_ID
        original["groups"][0]["episodes"][-4]["episode_number"] = 4
        original["groups"][0]["episodes"][-1]["episode_number"] = 1
        self.client.data[f"/tv/episode_group/{OTHER_GROUP_ID}"] = original
        self.client.data["/tv/100/episode_groups"]["results"].append(
            {"id": OTHER_GROUP_ID, "name": "Alternate Cut", "group_count": 1, "episode_count": 14})
        with self.assertRaises(EpisodeResearchError) as raised:
            self.prove()
        self.assertEqual(raised.exception.code, "ambiguous_episode_groups")

    def test_gapped_or_duplicate_source_files_fail_closed(self):
        for change in ("gap", "duplicate", "secret", "missing", "count"):
            with self.subTest(change=change):
                payload = case_payload()
                if change == "gap": payload["files"].pop(1)
                if change == "duplicate": payload["files"].append(copy.deepcopy(payload["files"][0]))
                if change == "secret": payload["files"][0]["name"] = "password=secret123.S01E11.mkv"
                if change == "missing": payload["files"][0]["name"] = "Unknown.mkv"; payload["files"][0].pop("episode")
                if change == "count": payload["files"] *= 30
                with self.assertRaises(EpisodeResearchError):
                    case = normalize_case(payload)
                    reader = EpisodeEvidenceReader(case, client=self.client)
                    try:
                        reader.inspect_candidate(0); reader.list_groups(0); reader.read_group(0, GROUP_ID)
                        reader.validate(0, GROUP_ID)
                    finally: reader.close()

    def test_numeric_coercion_and_wrong_tmdb_binding_are_rejected(self):
        for kind in ("work", "episode", "group_order", "target_number", "target_season"):
            with self.subTest(kind=kind):
                client = EvidenceClient()
                if kind == "work": client.data["/tv/100"]["id"] = 200
                elif kind == "episode": client.data[f"/tv/episode_group/{GROUP_ID}"]["groups"][0]["episodes"][-4]["id"] = 1011.5
                elif kind == "group_order": client.data[f"/tv/episode_group/{GROUP_ID}"]["groups"][0]["order"] = True
                elif kind == "target_number": client.data["/tv/100/season/0"]["episodes"][0]["episode_number"] = 1.5
                else: client.data["/tv/100/season/0"]["season_number"] = 1
                reader = EpisodeEvidenceReader(self.case, client=client)
                with self.assertRaises(EpisodeResearchError):
                    reader.inspect_candidate(0); reader.list_groups(0); reader.read_group(0, GROUP_ID); reader.validate(0, GROUP_ID)
                reader.close()

    def test_frozen_candidate_does_not_prove_unrelated_source_title(self):
        payload = case_payload()
        for item in payload["files"]:
            item["name"] = item["name"].replace("Fixture.Show", "Unrelated.Show")
        reader = EpisodeEvidenceReader(normalize_case(payload), client=self.client)
        with self.assertRaises(EpisodeResearchError) as raised:
            reader.inspect_candidate(0)
        self.assertEqual(raised.exception.code, "source_identity_unproven")
        reader.close()

    def test_official_alternative_title_binds_foreign_language_release(self):
        self.client.data["/tv/100"]["name"] = "测试作品"
        self.client.data["/tv/100/alternative_titles"]["results"] = [{"title": "Fixture Show", "iso_3166_1": "US"}]
        proposal = self.prove()
        self.assertEqual(proposal["status"], "verified")
        self.assertIn("/tv/100/alternative_titles", self.client.requests)

    def test_request_budget_and_closed_reader_do_not_continue_requests(self):
        reader = EpisodeEvidenceReader(self.case, client=self.client, max_requests=1)
        reader.inspect_candidate(0)
        with self.assertRaises(EpisodeResearchError): reader.list_groups(0)
        self.assertEqual(len(self.client.requests), 1)
        reader.close()
        with self.assertRaises(EpisodeResearchError): reader.inspect_candidate(0)
