"""GCID 导出、校验和预览必须共享能落为文件树的路径合同。"""

from __future__ import annotations

from unittest import TestCase
from unittest.mock import Mock, patch

from app.clients.guangya import GuangYaFile
from app.modules import gcid_import
from app.modules.gcid_manifest import (
    ManifestValidationError,
    export_manifest,
    normalize_manifest_v2,
)
from tests.test_gcid_import import _manifest


class GCIDManifestTreeAuditTests(TestCase):
    def test_file_directory_prefix_collisions_are_rejected_in_any_order(self):
        for parent, child in (
            ("Film", "Film/a.mkv"),
            ("Film", "film/sub/a.mkv"),
            ("Film/Part", "Film/Part/a.mkv"),
        ):
            rows = [
                {"path": parent, "size": 1, "gcid": "a"},
                {"path": child, "size": 2, "gcid": "b"},
            ]
            for items in (rows, list(reversed(rows))):
                with (
                    self.subTest(paths=[r["path"] for r in items]),
                    self.assertRaisesRegex(ManifestValidationError, "文件.*目录.*冲突"),
                ):
                    normalize_manifest_v2(
                        {
                            **_manifest(
                                sorted(rows, key=lambda r: r["path"].casefold())
                            ),
                            "files": items,
                        }
                    )

    def test_invalid_tree_is_rejected_before_preview_is_retained(self):
        payload = _manifest(
            [
                {"path": "Film", "size": 1, "gcid": "a"},
                {"path": "Film/a.mkv", "size": 2, "gcid": "b"},
            ]
        )
        with patch.object(
            gcid_import._preview_store,
            "create",
            wraps=gcid_import._preview_store.create,
        ) as create:
            with self.assertRaisesRegex(ManifestValidationError, "文件.*目录.*冲突"):
                gcid_import.create_preview(
                    payload, target_dir_id="target", owner_id="audit"
                )
        create.assert_not_called()

    def test_export_cannot_emit_unimportable_sanitized_path_collisions(self):
        for names in (("A/B.mkv", "A\\B.mkv"), ("A.mkv", "a.mkv")):
            client = Mock()
            client.list_dir.return_value = [
                GuangYaFile(str(i), name, False, i + 1, f"gcid-{i}", "root")
                for i, name in enumerate(names)
            ]
            with (
                self.subTest(names=names),
                self.assertRaisesRegex(ManifestValidationError, "重复路径"),
            ):
                export_manifest(client, "root")
            client.file_info.assert_not_called()

    def test_valid_prefix_siblings_empty_and_normal_export_roundtrip(self):
        client = Mock()
        client.list_dir.side_effect = lambda directory: {
            "root": [
                GuangYaFile("a", "Film", False, 1, "a", "root"),
                GuangYaFile("folder", "Film-more", True, 0, "", "root"),
            ],
            "folder": [GuangYaFile("b", "a.mkv", False, 2, "b", "folder")],
        }[directory]
        payload = export_manifest(client, "root", "历史导出")
        self.assertEqual(normalize_manifest_v2(payload).to_dict(), payload)
        preview = gcid_import.create_preview(
            payload, target_dir_id="target", owner_id="audit"
        )
        self.assertEqual((preview["file_count"], preview["total_size"]), (2, 3))
        self.assertEqual(len(preview["tree"]), 2)
        client.list_dir.side_effect = None
        client.list_dir.return_value = []
        empty = export_manifest(client, "root")
        self.assertEqual(normalize_manifest_v2(empty).files, ())
