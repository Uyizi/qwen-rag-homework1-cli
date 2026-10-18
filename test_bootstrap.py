import unittest
from bootstrap import normalize_path, validate_config, decode_output


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


if __name__ == "__main__":
    unittest.main()
