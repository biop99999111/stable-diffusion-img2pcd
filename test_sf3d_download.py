"""No-network regression tests for interrupted HF streams."""
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from requests.exceptions import ChunkedEncodingError, HTTPError
from sf3d_workflow import download_snapshot


class DownloadTests(unittest.TestCase):
    def test_retry_keeps_revision_and_cache(self):
        download = Mock(side_effect=[ChunkedEncodingError("broken"), "/cache/snapshot"])
        with patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}), patch("sf3d_workflow.time.sleep"):
            self.assertEqual(download_snapshot("fixed-sha"), "/cache/snapshot")
        self.assertEqual(download.call_count, 2)
        self.assertEqual(download.call_args_list[0], download.call_args_list[1])
        self.assertNotIn("force_download", download.call_args.kwargs)

    def test_retries_are_bounded(self):
        download = Mock(side_effect=ChunkedEncodingError("broken"))
        with patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}), patch("sf3d_workflow.time.sleep"):
            with self.assertRaises(ChunkedEncodingError):
                download_snapshot("sha", attempts=3)
        self.assertEqual(download.call_count, 3)

    def test_http_error_is_not_retried(self):
        download = Mock(side_effect=HTTPError("403"))
        with patch.dict(sys.modules, {"huggingface_hub": SimpleNamespace(snapshot_download=download)}):
            with self.assertRaises(HTTPError):
                download_snapshot("sha")
        download.assert_called_once()


if __name__ == "__main__":
    unittest.main()
