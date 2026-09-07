"""媒体分页不能因短页/空页而把尚未读取的上游总量当成完整清单。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from app.clients.emby import EmbyClient
from app.clients.jellyfin import JellyfinClient
from app.modules.media_subscriptions import MediaSubscriptionService


class MediaInventoryCompletenessTests(unittest.TestCase):
    @staticmethod
    def readers(client, *, cap=5):
        return {
            "series": lambda: client.list_library_series(max_series=cap, page_size=2),
            "identity": lambda: client.list_media_identity_inventory(
                "tv", max_items=cap, page_size=2
            ),
            "recommendations": lambda: client.list_recommendation_candidates(
                "viewer", max_items=cap, page_size=2
            ),
            "episodes": lambda: client.list_series_episode_inventory(
                "series", max_episodes=cap, page_size=2
            ),
        }

    def client(self, cls):
        client = cls("http://synthetic.invalid", "synthetic")
        client._cached_user_id = "viewer"
        self.addCleanup(client.close)
        return client

    @staticmethod
    def item(n):
        return {
            "Id": str(n),
            "Name": f"Show {n}",
            "Type": "Series",
            "ProviderIds": {"Tmdb": str(n)},
            "ParentIndexNumber": 1,
            "IndexNumber": n,
        }

    def test_short_empty_or_filtered_page_with_remaining_total_is_incomplete(self):
        for cls in (EmbyClient, JellyfinClient):
            client = self.client(cls)
            for name, reader in self.readers(client).items():
                for items in ([], [self.item(1)], [self.item(1), None]):
                    with (
                        self.subTest(server=cls.__name__, operation=name, items=items),
                        patch.object(
                            client,
                            "_request",
                            return_value={"Items": items, "TotalRecordCount": 3},
                        ) as request,
                    ):
                        result = reader()
                        self.assertTrue(result.truncated)
                        self.assertEqual(result.total, 3)
                        self.assertEqual(request.call_count, 1)

    def test_later_empty_page_keeps_unread_total_visible(self):
        client = self.client(JellyfinClient)
        for name, reader in self.readers(client).items():
            pages = [
                {"Items": [self.item(1), self.item(2)], "TotalRecordCount": 3},
                {"Items": [], "TotalRecordCount": 3},
            ]
            with (
                self.subTest(operation=name),
                patch.object(client, "_request", side_effect=pages) as request,
            ):
                result = reader()
                self.assertTrue(result.truncated)
                self.assertEqual(
                    [
                        call.kwargs["params"]["StartIndex"]
                        for call in request.call_args_list
                    ],
                    [0, 2],
                )

    def test_valid_empty_complete_and_capped_results_keep_their_meaning(self):
        for cls in (EmbyClient, JellyfinClient):
            client = self.client(cls)
            for cap, payload, truncated in (
                (5, {"Items": [], "TotalRecordCount": 0}, False),
                (
                    5,
                    {"Items": [self.item(1), self.item(2)], "TotalRecordCount": 2},
                    False,
                ),
                (1, {"Items": [self.item(1)], "TotalRecordCount": 2}, True),
            ):
                for name, reader in self.readers(client, cap=cap).items():
                    with (
                        self.subTest(server=cls.__name__, operation=name, cap=cap),
                        patch.object(client, "_request", return_value=payload),
                    ):
                        self.assertEqual(reader().truncated, truncated)

    def test_incomplete_episode_inventory_blocks_subscription_resource_search(self):
        client = self.client(JellyfinClient)
        from app.clients.base import SeriesCandidate
        from app.services import _series_inventory_source_result

        with patch.object(
            client,
            "_request",
            return_value={"Items": [self.item(1)], "TotalRecordCount": 3},
        ):
            source = _series_inventory_source_result(
                server_type="jellyfin",
                server_name="Synthetic",
                client=client,
                selected_items=[SeriesCandidate("series", "Synthetic", "2026", "1")],
                candidates=[],
                status="ready",
                mapping_status="mapped",
                max_episodes=2000,
                include_specials=False,
            )
        self.assertEqual(source["episodes"], [(1, 1)])
        self.assertEqual(source["local_total"], 3)
        ready, reason = MediaSubscriptionService._inventory_complete([source])
        self.assertFalse(ready)
        self.assertIn("未执行资源搜索", reason)

    def test_repeated_item_ids_abort_inventory_before_becoming_complete(self):
        for cls in (EmbyClient, JellyfinClient):
            client = self.client(cls)
            for name, reader in self.readers(client, cap=100).items():
                payload = {
                    "Items": [self.item(1), self.item(2)],
                    "TotalRecordCount": 100,
                }
                with (
                    self.subTest(server=cls.__name__, operation=name),
                    patch.object(client, "_request", return_value=payload) as request,
                ):
                    with self.assertRaisesRegex(ValueError, "重复"):
                        reader()
                    self.assertEqual(request.call_count, 2)

    def test_distinct_versions_of_the_same_episode_are_not_duplicate_items(self):
        client = self.client(JellyfinClient)
        first, second = self.item(1), self.item(2)
        second["IndexNumber"] = 1
        with patch.object(
            client,
            "_request",
            return_value={"Items": [first, second], "TotalRecordCount": 2},
        ):
            inventory = client.list_series_episode_inventory("series", page_size=2)
        self.assertEqual(inventory.episodes, [(1, 1)])
        self.assertEqual(inventory.total, 2)
        self.assertFalse(inventory.truncated)
