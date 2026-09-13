"""发布格式教学 Agent 动作的真实隔离数据库契约。"""
from __future__ import annotations

import json
import time
from copy import deepcopy
from unittest.mock import patch

from app import database as db
from app.agent.errors import AgentToolError
from app.agent.release_format_actions import (
    prepare_release_format,
    preview_release_format,
    save_release_format_confirmed,
    teaching_arguments,
)
from app.modules.recognition import formats
from app.modules.scraper import _parse_release_core
from tests.support import IsolatedDatabaseTestCase

TEMPLATE = "[Example-Team][{title}][track{episode}r{version}][{resolution}].mkv"
PARENT = "/Anime/Teaching"


def filename(episode: int, title: str = "星海航行") -> str:
    return f"[Example-Team][{title}][track{episode:03d}r2][1080p].mkv"


def teaching(parent: str = PARENT) -> dict:
    return {
        "draft": {
            "name": "轨道式发布编号",
            "template": TEMPLATE,
            "scope": "directory",
            "parent_path": parent,
        },
        "examples": [
            {"filename": filename(13), "title": "星海航行", "episode": 13},
            {"filename": filename(14), "title": "星海航行", "episode": 14},
        ],
        "filenames": [filename(15), "unrelated.mkv"],
    }


class ReleaseFormatAgentActionTests(IsolatedDatabaseTestCase):
    def setUp(self) -> None:
        with db.get_conn() as conn:
            conn.execute("DELETE FROM recognition_format_rules")
        formats.invalidate_cache()

    @staticmethod
    def rule_count() -> int:
        with db.get_conn() as conn:
            return int(
                conn.execute("SELECT COUNT(*) FROM recognition_format_rules").fetchone()[0]
            )

    def test_read_is_safe_and_does_not_write_or_create_confirmation(self) -> None:
        result = preview_release_format(teaching())
        self.assertTrue(result.ok)
        self.assertEqual(result.status, "preview")
        self.assertEqual(result.data["review_required"], 1)
        self.assertEqual(result.data["total"], 2)
        self.assertEqual(len(result.data["resources"]), 2)
        self.assertEqual(self.rule_count(), 0)
        self.assertIsNone(_parse_release_core(filename(15), PARENT).context.episode)
        public = json.dumps(result.to_dict(), ensure_ascii=False)
        self.assertNotIn("preview_token", public)
        self.assertNotIn(TEMPLATE, public)
        self.assertNotIn(PARENT, public)
        for resource in result.data["resources"]:
            self.assertIn("before", resource)
            self.assertIn("after", resource)
            self.assertIn("filename", resource)
        effects = "".join(result.data["effects"])
        self.assertIn("特别篇/受保护 0 个", effects)
        self.assertIn("格式冲突 0 个", effects)
        self.assertIn("原有识别流程", effects)
        self.assertIn("不移动文件", effects)
        self.assertIn("不绑定 TMDB", effects)
        self.assertIn("不偏移季集编号", effects)
        self.assertNotIn("preview_token", json.dumps(result.model_data, ensure_ascii=False))

    def test_sample_only_preview_counts_samples_and_keeps_raw_model_summary(self) -> None:
        value = teaching()
        value["filenames"] = []
        result = preview_release_format(value)
        self.assertEqual(result.status, "preview")
        self.assertEqual(result.data["total"], 2)
        self.assertEqual(result.data["count"], 2)
        self.assertEqual(result.data["sample_count"], 2)
        self.assertEqual(result.data["batch_total"], 0)
        self.assertEqual(result.data["review_required"], 0)
        self.assertEqual(result.data["summary"]["total"], 2)
        self.assertEqual(result.model_data["summary"]["total"], 0)
        self.assertIn("已核对 2 个标注样本", result.summary)

    def test_confirmation_keeps_dotted_names_and_long_title_episode_visible(self) -> None:
        for title in ("星海航行", "星海航行" * 40):
            with self.subTest(long_title=len(title) > 100):
                value = teaching()
                value["draft"]["template"] = "Example.Team.{title}.track{episode}r{version}.{resolution}.mkv"
                value["examples"] = [
                    {"filename": f"Example.Team.{title}.track{episode:03d}r2.1080p.mkv",
                     "title": title, "episode": episode} for episode in (13, 14)
                ]
                value["filenames"] = []
                result, _ = prepare_release_format(value)
                text = result.data["resources"][0]["title"]
                self.assertIn("Example.Team.", text)
                self.assertIn("第13集", text)
                self.assertIn(title, text)

    def test_prepare_reuses_preview_token_and_stays_read_only(self) -> None:
        captured: list[dict] = []
        original = formats.preview

        def preview(value: dict) -> dict:
            result = original(value)
            captured.append(result)
            return result

        with patch.object(formats, "preview", side_effect=preview):
            result, token = prepare_release_format(teaching())
        self.assertTrue(result.ok)
        self.assertEqual(result.status, "confirmation_required")
        self.assertEqual(token, captured[0]["preview_token"])
        self.assertEqual(self.rule_count(), 0)

    def test_confirm_writes_one_rule_and_core_uses_it(self) -> None:
        value = teaching()
        preview, token = prepare_release_format(value)
        saved = save_release_format_confirmed(value, token)
        self.assertTrue(saved.ok)
        self.assertTrue(saved.data["created"])
        self.assertTrue(saved.data["enabled"])
        self.assertEqual(self.rule_count(), 1)
        self.assertEqual(_parse_release_core(filename(15), PARENT).context.episode, 15)
        self.assertNotIn("preview_token", json.dumps(preview.to_model_dict(), ensure_ascii=False))

    def test_repeated_confirmation_returns_same_id_without_second_insert(self) -> None:
        value = teaching()
        _, token = prepare_release_format(value)
        first = save_release_format_confirmed(value, token)
        second = save_release_format_confirmed(value, token)
        self.assertTrue(first.data["created"])
        self.assertTrue(second.data["duplicate"])
        self.assertFalse(second.data["created"])
        self.assertTrue(second.data["enabled"])
        with db.get_conn() as conn:
            rows = conn.execute("SELECT id FROM recognition_format_rules").fetchall()
        self.assertEqual(len(rows), 1)
        self.assertEqual(first.model_data["draft"], second.model_data["draft"])

    def test_disabled_duplicate_is_not_reported_as_enabled_or_reenabled(self) -> None:
        value = teaching()
        _, token = prepare_release_format(value)
        first = save_release_format_confirmed(value, token)
        with db.get_conn() as conn:
            row = conn.execute(
                "SELECT id,revision FROM recognition_format_rules"
            ).fetchone()
        formats.change(row[0], {"revision": row[1], "disabled": True})
        repeated = save_release_format_confirmed(value, token)
        self.assertTrue(repeated.data["duplicate"])
        self.assertFalse(repeated.data["enabled"])
        self.assertIn("已停用", repeated.summary)
        self.assertNotIn("已启用", repeated.summary)
        self.assertIsNone(_parse_release_core(filename(15), PARENT).context.episode)
        self.assertEqual(first.data["enabled"], True)

    def test_cancel_without_confirmation_has_no_write(self) -> None:
        _, _token = prepare_release_format(teaching())
        self.assertEqual(self.rule_count(), 0)
        self.assertIsNone(_parse_release_core(filename(15), PARENT).context.episode)

    def test_prepare_rejects_unreviewable_preview_without_writing(self) -> None:
        value = teaching()
        value["examples"][0]["episode"] = 999
        preview = preview_release_format(value)
        self.assertTrue(preview.ok)
        self.assertTrue(preview.data["review_required"])
        with self.assertRaises(AgentToolError) as raised:
            prepare_release_format(value)
        self.assertEqual(raised.exception.code, "precondition_failed")
        self.assertEqual(self.rule_count(), 0)

    def test_unknown_fields_and_insufficient_samples_are_rejected(self) -> None:
        value = teaching()
        for extra in ({"confirmed": True}, {"preview_token": "private"}, {"unknown": 1}):
            invalid = {**deepcopy(value), **extra}
            with self.subTest(extra=extra), self.assertRaises(AgentToolError):
                teaching_arguments(invalid)
        insufficient = deepcopy(value)
        insufficient["examples"] = insufficient["examples"][:1]
        with self.assertRaises(AgentToolError):
            teaching_arguments(insufficient)

    def test_tampered_expired_changed_and_deleted_confirmations_are_public_stale_errors(self) -> None:
        value = teaching()
        _, token = prepare_release_format(value)
        changed = deepcopy(value)
        changed["filenames"].append(filename(16))
        with self.assertRaises(AgentToolError) as tampered:
            save_release_format_confirmed(changed, token)
        self.assertEqual(tampered.exception.code, "confirmation_stale")
        with patch("itsdangerous.timed.time.time", return_value=time.time() + 901), self.assertRaises(AgentToolError) as expired:
            save_release_format_confirmed(value, token)
        self.assertEqual(expired.exception.code, "confirmation_stale")

        other = teaching("/Anime/Other")
        _, other_token = prepare_release_format(other)
        save_release_format_confirmed(other, other_token)
        with self.assertRaises(AgentToolError) as changed_registry:
            save_release_format_confirmed(value, token)
        self.assertEqual(changed_registry.exception.code, "confirmation_stale")

        candidate = teaching("/Anime/Candidate")
        _, candidate_token = prepare_release_format(candidate)
        with db.get_conn() as conn:
            row = conn.execute(
                "SELECT id,revision FROM recognition_format_rules WHERE parent_path=?",
                ("/Anime/Other",),
            ).fetchone()
        formats.change(row[0], {"revision": row[1]}, delete=True)
        with self.assertRaises(AgentToolError) as deleted:
            save_release_format_confirmed(candidate, candidate_token)
        self.assertEqual(deleted.exception.code, "confirmation_stale")
        self.assertEqual(self.rule_count(), 0)
