"""发布格式教学的字段、回放、确认和统一解析链契约。"""
from __future__ import annotations

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import re
import time
import threading
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import database as db
from app.modules.recognition import formats
from app.modules.scraper import TMDBScraper, _parse_release_core
from tests.support import IsolatedDatabaseTestCase

TEMPLATE = "[Example-Team][{title}][track{episode}r{version}][{resolution}].mkv"
PARENT = "/Anime/Teaching"


def filename(episode: int, title: str = "星海航行", version: int = 2) -> str:
    return f"[Example-Team][{title}][track{episode:03d}r{version}][1080p].mkv"


def teaching(*, scope: str = "directory") -> dict:
    titles = ("星海航行", "银河物语") if scope == "release" else ("星海航行", "星海航行")
    return {
        "draft": {"name": "轨道式发布编号", "template": TEMPLATE, "scope": scope,
                  "parent_path": "" if scope == "release" else PARENT},
        "examples": [{"filename": filename(number, title), "title": title, "episode": number}
                     for number, title in zip((13, 14), titles)],
        "filenames": [filename(15), "unrelated.mkv"],
    }


def save_teaching(value: dict | None = None) -> tuple[dict, dict]:
    value = value or teaching()
    preview = formats.preview(value)
    if not preview["can_save"]:
        raise AssertionError(preview)
    payload = {**value, "confirmed": True, "preview_token": preview["preview_token"]}
    item, created = formats.save(payload)
    if not created:
        raise AssertionError("fixture must create a fresh rule")
    return item, payload


class ReleaseFormatTests(IsolatedDatabaseTestCase):
    def setUp(self) -> None:
        with db.get_conn() as conn:
            conn.execute("DELETE FROM recognition_format_rules")
            conn.execute("DELETE FROM recognition_preprocess_rules WHERE builtin_key=''")
        formats.invalidate_cache()
        from app.modules.recognition_preprocess_rules import invalidate_active_cache
        invalidate_active_cache()

    def test_preview_is_read_only_and_projects_raw_fields_without_remapping(self):
        result = formats.preview(teaching())
        self.assertTrue(result["can_save"])
        self.assertTrue(all(row["passed"] for row in result["examples"]))
        self.assertEqual(result["rows"][0]["after"], {"title": "星海航行", "episode": 15, "season": None})
        self.assertEqual(result["rows"][1]["status"], "unmatched")
        self.assertEqual(result["summary"]["matched"], 1)
        self.assertEqual(formats.list_rules(), [])
        self.assertIsNone(_parse_release_core(filename(15), PARENT).context.episode)

    def test_confirmed_save_is_reused_by_context_and_release_projection(self):
        save_teaching()
        parser = TMDBScraper("offline-fixture")
        self.addCleanup(parser.close)
        context = _parse_release_core(filename(15), PARENT).context
        parsed = parser.parse_media(filename(15), PARENT)
        self.assertEqual((context.normalized_title, context.episode), ("星海航行", 15))
        self.assertEqual((parsed.title, parsed.source_episode, parsed.effective_episode), ("星海航行", 15, 15))
        self.assertEqual(context.cleaned_components["format_version"], ["2"])
        self.assertEqual(context.cleaned_components["format_resolution"], ["1080p"])
        self.assertTrue(any(e.source == "release_format" for e in parsed.evidence))
        self.assertEqual(parser.parse_source_position(filename(15), PARENT), (None, 15))

    def test_directory_scope_does_not_leak_to_other_or_child_directories(self):
        save_teaching()
        for parent in ("/Anime/Other", PARENT + "/Disc 1", ""):
            with self.subTest(parent=parent):
                self.assertIsNone(_parse_release_core(filename(15), parent).context.episode)
        self.assertEqual(_parse_release_core(filename(15), PARENT + "/").context.episode, 15)
        self.assertEqual(_parse_release_core(filename(15), PARENT.replace("/", "\\")).context.episode, 15)

    def test_release_scope_requires_different_titles_and_fixed_prefix(self):
        value = teaching(scope="release")
        result = formats.preview(value)
        self.assertTrue(result["can_save"])
        save_teaching(value)
        self.assertEqual(_parse_release_core(filename(88, "第三部作品"), "/Elsewhere").context.episode, 88)
        value = teaching()
        value["draft"]["scope"] = "release"
        with self.assertRaisesRegex(ValueError, "不同作品"):
            formats.preview(value)
        value = teaching(scope="release")
        value["draft"]["template"] = "{title} - {episode}.mkv"
        with self.assertRaisesRegex(ValueError, "固定发布前缀"):
            formats.preview(value)

    def test_repeated_save_returns_same_id_without_reenabling_disabled_rule(self):
        item, payload = save_teaching()
        duplicate, created = formats.save(payload)
        self.assertFalse(created)
        self.assertEqual(duplicate["id"], item["id"])
        disabled = formats.change(item["id"], {"revision": item["revision"], "disabled": True})["item"]
        duplicate, created = formats.save(payload)
        self.assertFalse(created)
        self.assertTrue(duplicate["disabled"])
        self.assertEqual(duplicate["revision"], disabled["revision"])
        self.assertEqual(len(formats.list_rules()), 1)

    def test_late_retry_cannot_resurrect_a_deleted_rule(self):
        item, payload = save_teaching()
        formats.change(item["id"], {"revision": item["revision"]}, delete=True)
        with self.assertRaises(formats.FormatConflict):
            formats.save(payload)
        self.assertEqual(formats.list_rules(), [])

    def test_concurrent_confirmation_creates_only_one_rule(self):
        value = teaching()
        result = formats.preview(value)
        payload = {**value, "confirmed": True, "preview_token": result["preview_token"]}
        barrier = threading.Barrier(2)
        evaluate = formats._evaluate

        def evaluate_together(*args):
            result = evaluate(*args)
            barrier.wait(timeout=10)
            return result

        with patch.object(formats, "_evaluate", side_effect=evaluate_together), ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: formats.save(payload), range(2)))
        self.assertEqual(sorted(created for _, created in results), [False, True])
        self.assertEqual(len({item["id"] for item, _ in results}), 1)
        self.assertEqual(len(formats.list_rules()), 1)

    def test_toggle_delete_and_stale_versions_are_immediately_visible(self):
        item, _ = save_teaching()
        self.assertEqual(_parse_release_core(filename(15), PARENT).context.episode, 15)
        item = formats.change(item["id"], {"revision": item["revision"], "disabled": True})["item"]
        self.assertIsNone(_parse_release_core(filename(15), PARENT).context.episode)
        with self.assertRaises(formats.FormatConflict):
            formats.change(item["id"], {"revision": item["revision"] - 1, "disabled": False})
        item = formats.change(item["id"], {"revision": item["revision"], "disabled": False})["item"]
        self.assertEqual(_parse_release_core(filename(15), PARENT).context.episode, 15)
        self.assertEqual(formats.change(item["id"], {"revision": item["revision"]}, delete=True), {"deleted": True})
        self.assertIsNone(_parse_release_core(filename(15), PARENT).context.episode)

    def test_draft_batch_and_labels_cannot_change_after_preview(self):
        value = teaching()
        result = formats.preview(value)
        for part in ("name", "template", "directory", "examples", "filenames"):
            updated = deepcopy(value)
            if part == "name": updated["draft"]["name"] = "changed"
            elif part == "template": updated["draft"]["template"] += "x.mkv"
            elif part == "directory": updated["draft"]["parent_path"] = "/Other"
            elif part == "examples": updated["examples"][0]["episode"] = 99
            else: updated["filenames"].append(filename(55))
            with self.subTest(part=part), self.assertRaises(formats.FormatConflict):
                formats.save({**updated, "preview_token": result["preview_token"], "confirmed": True})
        self.assertEqual(formats.list_rules(), [])

    def test_expired_or_forged_confirmation_is_rejected(self):
        value = teaching()
        result = formats.preview(value)
        for token in ("bad-token", result["preview_token"] + "bad"):
            with self.subTest(token=token), self.assertRaises(formats.FormatConflict):
                formats.save({**value, "preview_token": token, "confirmed": True})
        with patch("itsdangerous.timed.time.time", return_value=time.time() + 901), self.assertRaises(formats.FormatConflict):
            formats.save({**value, "preview_token": result["preview_token"], "confirmed": True})
        with self.assertRaisesRegex(ValueError, "确认"):
            formats.save({**value, "preview_token": result["preview_token"]})

    def test_restore_cache_invalidation_also_expires_preview(self):
        value = teaching()
        result = formats.preview(value)
        formats.invalidate_cache()
        with self.assertRaises(formats.FormatConflict):
            formats.save({**value, "confirmed": True, "preview_token": result["preview_token"]})
        self.assertEqual(formats.list_rules(), [])

    def test_process_restart_expires_preview_even_when_database_is_unchanged(self):
        value = teaching()
        result = formats.preview(value)
        with patch.object(formats, "_PREVIEW_EPOCH", "another-process"), self.assertRaises(formats.FormatConflict):
            formats.save({**value, "confirmed": True, "preview_token": result["preview_token"]})
        self.assertEqual(formats.list_rules(), [])

    def test_registry_change_requires_new_preview(self):
        first = teaching()
        result = formats.preview(first)
        other = teaching()
        other["draft"]["parent_path"] = "/Other"
        save_teaching(other)
        with self.assertRaisesRegex(formats.FormatConflict, "变化"):
            formats.save({**first, "preview_token": result["preview_token"], "confirmed": True})
        self.assertEqual(len(formats.list_rules()), 1)

    def test_bad_labels_and_already_correct_samples_cannot_be_promoted(self):
        value = teaching()
        value["examples"][0]["episode"] = 999
        result = formats.preview(value)
        self.assertFalse(result["can_save"])
        self.assertEqual(result["preview_token"], "")
        value["draft"]["template"] = "{title}.S{season}E{episode}.mkv"
        value["examples"] = [{"filename": f"Example.S01E0{i}.mkv", "title": "Example", "episode": i, "season": 1} for i in (1, 2)]
        result = formats.preview(value)
        self.assertFalse(result["can_save"])
        self.assertTrue(any("没有改善" in w for w in result["warnings"]))

    def test_templates_are_literal_bounded_and_require_unambiguous_fields(self):
        for template in ("{title}{episode}.mkv", "{title}-{unknown}-{episode}.mkv", "{title}-{episode}-{episode}.mkv",
                         "[Group]{title}.mkv", "../{title}-{episode}.mkv", "{title}-{episode}.srt"):
            with self.subTest(template=template), self.assertRaises(ValueError):
                formats.compile_template(template)
        regex = formats.compile_template("[A+B](.{title}.)-{episode}.mkv").regex
        self.assertIsNotNone(regex.fullmatch("[A+B](.Hello.)-12.mkv"))
        self.assertIsNone(regex.fullmatch("AAAB-Hello-12.mkv"))

    def test_invalid_counts_boolean_positions_and_season_labels_are_rejected(self):
        changes = (
            lambda x: x.update(examples=x["examples"][:1]),
            lambda x: x.update(filenames=[filename(20)] * 101),
            lambda x: x["examples"][0].update(episode=True),
            lambda x: x["examples"][0].update(episode=0),
            lambda x: x["examples"][1].update(episode=13),
            lambda x: x["draft"].update(parent_path=""),
            lambda x: x["draft"].update(scope=[]),
            lambda x: x["draft"].update(template="[Group]{title}-S{season}E{episode}.mkv"),
        )
        for index, change in enumerate(changes):
            value = teaching(); change(value)
            with self.subTest(index=index), self.assertRaises(ValueError):
                formats.preview(value)

    def test_batch_limit_and_unmatched_files_are_preserved_without_writing(self):
        value = teaching()
        value["filenames"] = [filename(i) for i in range(15, 114)] + ["not-this-format.mkv"]
        result = formats.preview(value)
        self.assertEqual(len(result["rows"]), 100)
        self.assertEqual(result["summary"]["matched"], 99)
        self.assertEqual(result["rows"][-1]["before"], result["rows"][-1]["after"])
        self.assertEqual(formats.list_rules(), [])

    def test_special_content_and_explicit_positions_are_not_overridden(self):
        value = teaching()
        value["filenames"] = [filename(1, "NCOP"), filename(2, "SP02"), filename(3, "Show S01E09"), filename(4, "访谈")]
        result = formats.preview(value)
        self.assertEqual([row["status"] for row in result["rows"]], ["blocked", "blocked", "conflict", "blocked"])
        self.assertTrue(all(row["before"] == row["after"] for row in result["rows"]))
        self.assertFalse(result["can_save"])
        value["draft"]["parent_path"] = "/Anime/Extras"
        result = formats.preview(value)
        self.assertFalse(result["can_save"])

    def test_all_deterministic_positions_win_over_conflicting_taught_fields(self):
        template = "[Example-Team][track{episode}r{version}]{title}.mkv"
        rule = {**teaching()["draft"], "template": template, "disabled": False}
        titles = ("Show [03]", "Show - <03>", "Show - 03", "Show S02 [03(15)]",
                  "Show S02E03", "Show 第3集", "Show 2x03")
        for title in titles:
            with self.subTest(title=title):
                name = f"[Example-Team][track015r2]{title}.mkv"
                before = _parse_release_core(name, PARENT, _format_rules=[]).context
                after = _parse_release_core(name, PARENT, _format_rules=[rule]).context
                self.assertEqual(before.episode, 3)
                self.assertEqual(after.episode, before.episode)
                self.assertIn("release_format_conflicts", after.cleaned_components)
                self.assertNotIn("release_formats", after.cleaned_components)

    def test_numeric_conflict_requires_confirmation_before_online_matching(self):
        save_teaching()
        parser = TMDBScraper()
        self.addCleanup(parser.close)
        with patch.object(parser, "match_from_tmdb", side_effect=AssertionError("冲突不得自动放行")):
            result = parser.match(filename(15, "Show [03]"), PARENT)
        self.assertTrue(result.need_confirm)
        self.assertEqual(result.matched_by, "release_format_conflict")

    def test_extra_word_inside_a_normal_title_does_not_block_teaching(self):
        value = teaching()
        value["filenames"] = [filename(15, "Interview with the Stars")]
        result = formats.preview(value)
        self.assertEqual(result["rows"][0]["status"], "matched")
        self.assertEqual(result["rows"][0]["after"]["title"], "Interview with the Stars")

    def test_existing_preprocess_offset_remains_separate_from_source_number(self):
        from app.modules.recognition_preprocess_rules import create_rule
        save_teaching()
        create_rule({"name": "作品内旧偏移", "matcher_type": "text", "pattern": "[Example-Team]",
                     "scope": "filename", "action": "episode_offset", "numeric_value": -12})
        parser = TMDBScraper("offline-fixture")
        self.addCleanup(parser.close)
        parsed = parser.parse_media(filename(13), PARENT)
        self.assertEqual((parsed.source_episode, parsed.effective_episode), (13, 1))

    def test_replay_blocks_new_template_that_conflicts_with_saved_examples(self):
        save_teaching()
        value = teaching()
        value["draft"]["template"] = TEMPLATE.replace("track{episode}r{version}", "track0{version}r{episode}")
        value["examples"] = [{"filename": filename(i, version=e), "title": "星海航行", "episode": e}
                             for i, e in ((3, 13), (4, 14))]
        result = formats.preview(value)
        self.assertFalse(result["can_save"])
        self.assertGreater(result["summary"]["regressions"], 0)
        self.assertEqual(len(formats.list_rules()), 1)

    def test_imported_conflicting_formats_require_confirmation_before_matching(self):
        item, _ = save_teaching()
        with db.get_conn() as conn:
            conn.execute("INSERT INTO recognition_format_rules(signature,name,template,scope,parent_path,examples_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
                         ("external-import", "冲突格式", TEMPLATE.replace("track{episode}r{version}", "track0{version}r{episode}"),
                          "directory", PARENT, json.dumps(item["examples"]), db.now(), db.now()))
        formats.invalidate_cache()
        parser = TMDBScraper("offline-fixture")
        self.addCleanup(parser.close)
        with patch.object(parser, "match_from_tmdb", side_effect=AssertionError("格式冲突不得进入匹配放行")):
            result = parser.match(filename(15), PARENT)
        self.assertEqual(result.matched_by, "release_format_conflict")
        self.assertTrue(result.need_confirm)

    def test_format_snapshot_is_loaded_once_not_once_per_filename(self):
        save_teaching()
        formats.invalidate_cache()
        original = db.get_conn
        statements = []
        @contextmanager
        def traced():
            with original() as conn:
                conn.set_trace_callback(statements.append)
                yield conn
        with patch.object(db, "get_conn", traced):
            for episode in range(15, 40):
                self.assertEqual(_parse_release_core(filename(episode), PARENT).context.episode, episode)
        reads = [sql for sql in statements if sql.startswith("SELECT * FROM recognition_format_rules")]
        self.assertEqual(len(reads), 1)


class ReleaseFormatApiTests(IsolatedDatabaseTestCase):
    def setUp(self) -> None:
        with db.get_conn() as conn:
            conn.execute("DELETE FROM recognition_format_rules")
        formats.invalidate_cache()

    @staticmethod
    def csrf(response) -> str:
        match = re.search(r'name="csrf_token" (?:content|value)="([^"]+)"', response.text)
        if match is None:
            raise AssertionError("CSRF token missing")
        return match.group(1)

    def client(self):
        from app.main import create_app
        client = TestClient(create_app(), raise_server_exceptions=False)
        self.addCleanup(client.close)
        response = client.get("/login")
        response = client.post("/login", data={"csrf_token": self.csrf(response), "username": "admin", "password": "123456"}, follow_redirects=False)
        self.assertEqual(response.status_code, 302, response.text)
        headers = {"X-CSRF-Token": self.csrf(client.get("/organize-rules"))}
        return client, headers

    def test_login_and_csrf_are_required(self):
        from app.main import create_app
        client = TestClient(create_app(), raise_server_exceptions=False)
        self.addCleanup(client.close)
        self.assertEqual(client.get("/api/tools/release-formats").status_code, 401)
        client, _ = self.client()
        self.assertEqual(client.post("/api/tools/release-formats/preview", json=teaching()).status_code, 403)

    def test_preview_confirm_repeat_toggle_and_delete_http_flow(self):
        client, headers = self.client()
        value = teaching()
        preview = client.post("/api/tools/release-formats/preview", json=value, headers=headers)
        self.assertEqual(preview.status_code, 200, preview.text)
        payload = {**value, "confirmed": True, "preview_token": preview.json()["preview_token"]}
        created = client.post("/api/tools/release-formats", json=payload, headers=headers)
        self.assertEqual(created.status_code, 201, created.text)
        self.assertEqual(client.post("/api/tools/release-formats", json=payload, headers=headers).status_code, 200)
        item = created.json()["item"]
        updated = client.put(f"/api/tools/release-formats/{item['id']}", json={"disabled": True, "revision": item["revision"]}, headers=headers)
        self.assertEqual(updated.status_code, 200, updated.text)
        stale = client.put(f"/api/tools/release-formats/{item['id']}", json={"disabled": False, "revision": item["revision"]}, headers=headers)
        self.assertEqual(stale.status_code, 409)
        item = updated.json()["item"]
        deleted = client.request("DELETE", f"/api/tools/release-formats/{item['id']}", json={"revision": item["revision"]}, headers=headers)
        self.assertEqual(deleted.status_code, 200, deleted.text)
        self.assertEqual(client.get("/api/tools/release-formats").json()["items"], [])

    def test_changed_payload_and_invalid_template_return_contract_errors(self):
        client, headers = self.client()
        value = teaching()
        result = client.post("/api/tools/release-formats/preview", json=value, headers=headers).json()
        value["filenames"] = [filename(99)]
        response = client.post("/api/tools/release-formats", json={**value, "confirmed": True, "preview_token": result["preview_token"]}, headers=headers)
        self.assertEqual(response.status_code, 409, response.text)
        value["draft"]["template"] = "{title}-{invalid}-{episode}.mkv"
        response = client.post("/api/tools/release-formats/preview", json=value, headers=headers)
        self.assertEqual(response.status_code, 400)
        self.assertIsInstance(response.json()["error"], str)
