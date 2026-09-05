"""C03：不重新接入另一条本地播放路径。"""
import unittest

from app.modules import media_proxy


class LocalResponseRetirementTests(unittest.TestCase):
    def test_C03_unused_local_file_response_is_retired(self):
        self.assertFalse(hasattr(media_proxy, "local_file_response"))
        self.assertFalse(hasattr(media_proxy, "_file_chunks"))
