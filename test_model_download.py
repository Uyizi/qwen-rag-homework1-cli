import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

try:
    import model_download
except ImportError:
    model_download = None


class ModelDownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = b"local-model-data" * 1024
        cls.ranges = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                cls.ranges.append(self.headers.get("Range"))
                offset = int(self.headers.get("Range", "bytes=0-").split("=")[1].split("-")[0])
                if self.path == "/ignore-range":
                    offset = 0
                self.send_response(206 if offset else 200)
                if offset:
                    self.send_header("Content-Range", f"bytes {offset}-{len(cls.payload)-1}/{len(cls.payload)}")
                self.send_header("Content-Length", str(len(cls.payload) - offset))
                self.end_headers()
                self.wfile.write(cls.payload[offset:])

            def log_message(self, *args):
                pass

        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/model"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.assertIsNotNone(model_download, "尚未实现模型自动下载")
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.target = Path(self.folder.name) / "model.safetensors"
        self.ranges.clear()

    def download(self, url=None, size=None):
        model_download.download_file(url or self.url, self.target,
                                     len(self.payload) if size is None else size,
                                     progress=lambda text: None, retries=1)

    def test_download_then_skip_complete_file(self):
        self.download()
        self.assertEqual(self.target.read_bytes(), self.payload)
        self.download()
        self.assertEqual(len(self.ranges), 1)

    def test_resume_partial_download(self):
        self.target.with_name(self.target.name + ".part").write_bytes(self.payload[:100])
        self.download()
        self.assertEqual(self.ranges, ["bytes=100-"])
        self.assertEqual(self.target.read_bytes(), self.payload)

    def test_server_ignoring_range_restarts_without_duplicate_bytes(self):
        self.target.with_name(self.target.name + ".part").write_bytes(self.payload[:100])
        self.download(self.url.replace("/model", "/ignore-range"))
        self.assertEqual(self.target.read_bytes(), self.payload)

    def test_truncated_download_never_becomes_model_file(self):
        with self.assertRaises((OSError, RuntimeError)):
            self.download(size=len(self.payload) + 1)
        self.assertFalse(self.target.exists())

    def test_manifest_rejects_escape_and_untrusted_urls(self):
        for path, url in (("../config.json", self.url), ("model.bin", "http://example.com/model")):
            manifest = {"models": [{"key": "generation_model", "folder": "qwen",
                        "files": [{"path": path, "url": url, "size": 10}]}]}
            with self.assertRaises(ValueError):
                model_download.validate_manifest(manifest)

    def test_model_install_sets_paths_and_keeps_custom_model(self):
        root = Path(self.folder.name)
        manifest = {"models": [{"key": "generation_model", "folder": "qwen",
                    "files": [{"path": "model.safetensors", "url": self.url, "size": len(self.payload)}]}]}
        config = {"generation_model": "", "embedding_model": "/custom/e5"}
        result = model_download.ensure_models(root, config, manifest, progress=lambda text: None)
        self.assertEqual(result["generation_model"].casefold(),
                         model_download.normalized_path(root / "models" / "qwen").casefold())
        self.assertEqual(result["embedding_model"], "/custom/e5")
        self.assertEqual(config["generation_model"], "")
        self.assertEqual((root / "models/qwen/model.safetensors").read_bytes(), self.payload)

    def test_manifest_prefers_public_asset_api_and_rejects_injected_api_url(self):
        api_url = "https://api.github.com/repos/Uyizi/qwen-rag-homework1-cli/releases/assets/123"
        entry = {"path": "config.json", "size": 10, "url": self.url, "api_url": api_url}
        manifest = {"models": [{"key": "generation_model", "folder": "qwen", "files": [entry]}]}
        with patch.object(model_download, "download_file") as download:
            model_download.ensure_models(Path(self.folder.name), {}, manifest, progress=lambda text: None)
        self.assertEqual(download.call_args.args[0], api_url)
        entry["api_url"] = "https://example.com/private"
        with self.assertRaises(ValueError):
            model_download.validate_manifest(manifest)


if __name__ == "__main__":
    unittest.main()
