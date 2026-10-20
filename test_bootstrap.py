import unittest
import json
import os
import re
import tempfile
from pathlib import Path
from unittest.mock import patch
import bootstrap
import setup_runtime
from bootstrap import normalize_path, validate_config, decode_output
from setup_runtime import direct_environment


class BootstrapTests(unittest.TestCase):
    def test_wsl_output_accepts_windows_and_utf8_encodings(self):
        text = "WSL 本地环境提示"
        for encoding in ("utf-8", "gb18030", "utf-16", "utf-16-le"):
            self.assertEqual(decode_output(text.encode(encoding)), text)

    def test_windows_model_path_is_converted_for_wsl(self):
        self.assertEqual(normalize_path(r"D:\Local Models\Qwen"), "/mnt/d/Local Models/Qwen")
        self.assertEqual(normalize_path("/home/user/venv/bin/python"), "/home/user/venv/bin/python")

    def test_missing_model_location_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_config({"wsl_python": "/home/user/python", "generation_model": "", "embedding_model": "/models/e5"})

    def test_existing_configuration_keeps_runtime_parameters(self):
        config = {"wsl_python": "/home/user/python", "generation_model": r"D:\Models\Qwen", "embedding_model": "/models/e5", "top_k": 3}
        normalized = validate_config(config)
        self.assertEqual(normalized["generation_model"], "/mnt/d/Models/Qwen")
        self.assertEqual(normalized["top_k"], 3)
        self.assertEqual(config["generation_model"], r"D:\Models\Qwen")

    def test_first_start_prepares_models_without_path_questions(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config.example.json").write_text(json.dumps({"wsl_python": "", "generation_model": "", "embedding_model": ""}))
            ready = {"wsl_python": "/home/test/venv/bin/python", "generation_model": "/mnt/d/models/qwen", "embedding_model": "/mnt/d/models/e5"}
            with patch.object(bootstrap, "ROOT", root), patch.object(bootstrap, "prepare_backend", return_value=ready["wsl_python"]) as backend, patch.object(bootstrap, "ensure_models", return_value=ready) as models, patch("builtins.input", side_effect=AssertionError("不应要求手填路径")), patch.dict(os.environ, {}, clear=True):
                self.assertEqual(bootstrap.prepare_config(), ready)
            backend.assert_called_once()
            models.assert_called_once()
            self.assertTrue((root / "config.json").is_file())

    def test_check_only_does_not_download_or_create_environment(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "config.example.json").write_text(json.dumps({"wsl_python": "", "generation_model": "", "embedding_model": ""}))
            with patch.object(bootstrap, "ROOT", root), patch.object(bootstrap, "prepare_backend") as backend, patch.object(bootstrap, "ensure_models") as models, patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(ValueError):
                    bootstrap.prepare_config(check_only=True)
            backend.assert_not_called()
            models.assert_not_called()

    def test_install_ignores_proxy_without_changing_process_environment(self):
        with patch.dict(os.environ, {"HTTPS_PROXY": "http://127.0.0.1:9999", "http_proxy": "http://127.0.0.1:9999"}):
            env = direct_environment()
            self.assertNotIn("HTTPS_PROXY", env)
            self.assertNotIn("http_proxy", env)
            self.assertEqual(env["NO_PROXY"], "*")
            self.assertIn("HTTPS_PROXY", os.environ)

    def test_frontend_installs_from_public_index_and_bundled_wheels(self):
        with patch.object(bootstrap.importlib.util, "find_spec", return_value=None), patch.object(bootstrap.subprocess, "run") as run:
            bootstrap.check_frontend()
        command = run.call_args.args[0]
        self.assertIn("--index-url", command)
        self.assertIn("--find-links", command)
        self.assertNotIn("--no-index", command)
        self.assertEqual(run.call_args.kwargs["env"]["NO_PROXY"], "*")

    def test_launcher_selects_supported_python_when_newer_version_is_installed(self):
        choose = getattr(setup_runtime, "choose_windows_python", None)
        self.assertIsNotNone(choose, "启动器尚未选择兼容的 Python")
        def version(command, **kwargs):
            result = [3, 14] if command[0] == "python314.exe" else [3, 13]
            return type("Result", (), {"returncode": 0, "stdout": json.dumps(result)})()
        with patch.object(setup_runtime.subprocess, "run", side_effect=version):
            self.assertEqual(choose(["python314.exe", "python313.exe"]), "python313.exe")

    def test_launcher_reports_missing_supported_python_before_creating_environment(self):
        choose = getattr(setup_runtime, "choose_windows_python", None)
        self.assertIsNotNone(choose, "启动器尚未检查 Python 兼容版本")
        result = type("Result", (), {"returncode": 0, "stdout": "[3,14]"})()
        with patch.object(setup_runtime.subprocess, "run", return_value=result):
            with self.assertRaises(RuntimeError):
                choose(["python314.exe"])

    def test_driver_check_supports_current_and_legacy_cuda_headers(self):
        for header in ("CUDA Version: 13.0", "CUDA UMD Version: 13.3"):
            match = re.search(setup_runtime.CUDA_VERSION_PATTERN, header)
            self.assertIsNotNone(match)
            self.assertEqual(match[1], "13")


if __name__ == "__main__":
    unittest.main()
