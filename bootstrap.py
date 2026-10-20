import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PATH_KEYS = ("wsl_python", "generation_model", "embedding_model")


def normalize_path(value):
    value = value.strip().replace("\\", "/")
    if re.match(r"^[A-Za-z]:/", value):
        return "/mnt/" + value[0].lower() + value[2:]
    return value


def validate_config(config):
    result = dict(config)
    for key in PATH_KEYS:
        if not isinstance(result.get(key), str) or not result[key].strip():
            raise ValueError("缺少配置 " + key)
        result[key] = normalize_path(result[key])
        if not result[key].startswith("/"):
            raise ValueError("请使用绝对路径 " + key)
    return result


def prepare_config(configure=False):
    path = ROOT / "config.json"
    config = json.loads((path if path.exists() else ROOT / "config.example.json").read_text(encoding="utf-8"))
    variables = {"wsl_python": "QWEN_WSL_PYTHON", "generation_model": "QWEN_MODEL_PATH", "embedding_model": "E5_MODEL_PATH"}
    for key, variable in variables.items():
        if os.environ.get(variable):
            config[key] = os.environ[variable]
    if configure or any(not config.get(key) for key in PATH_KEYS):
        if not sys.stdin.isatty():
            raise ValueError("先运行 bootstrap.py --configure 或设置三个模型路径环境变量")
        labels = {"wsl_python": "WSL Python 路径", "generation_model": "本地 Qwen 模型目录", "embedding_model": "本地 E5 模型目录"}
        for key in PATH_KEYS:
            current = config.get(key, "")
            config[key] = input(f"{labels[key]} [{current}]: ").strip() or current
    config = validate_config(config)
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return config


def check_frontend(wheel_dir=None):
    missing = [name for name in ("numpy", "fitz", "rich", "prompt_toolkit") if importlib.util.find_spec(name) is None]
    if missing and wheel_dir is None:
        wheel_dir = ROOT / "vendor" / "wheels"
    if missing and wheel_dir:
        wheels = Path(wheel_dir).resolve()
        if not wheels.is_dir():
            raise ValueError("离线 wheel 目录不存在")
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--find-links", str(wheels), "-r", str(ROOT / "requirements.txt")], check=True)
    elif missing:
        raise ValueError("缺少前端依赖 " + ", ".join(missing) + " 请按 README 离线安装或指定 --wheel-dir")


def decode_output(data):
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16", errors="replace")
    if data[1:8:2].count(b"\x00") >= 2:
        return data.decode("utf-16-le", errors="replace")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("gb18030", errors="replace")


def check_backend(config, verbose=True):
    script = """import json,sys
from pathlib import Path
import torch,transformers
from importlib.metadata import version
from transformers import Qwen3_5ForConditionalGeneration
for location in sys.argv[1:]:
    folder=Path(location)
    if not (folder/'config.json').is_file():
        raise SystemExit('Model config not found: '+location)
    if not any(folder.glob('*.safetensors')) and not any(folder.glob('*.bin')):
        raise SystemExit('Model weights not found: '+location)
if not torch.cuda.is_available():
    raise SystemExit('CUDA GPU is required')
if not torch.cuda.is_bf16_supported():
    raise SystemExit('BF16 support is required')
print(json.dumps({'python':sys.version.split()[0],'torch':torch.__version__,'transformers':transformers.__version__,'accelerate':version('accelerate'),'safetensors':version('safetensors'),'gpu':torch.cuda.get_device_name()},ensure_ascii=False))
"""
    result = subprocess.run(["wsl", "--exec", config["wsl_python"], "-c", script, config["generation_model"], config["embedding_model"]], capture_output=True)
    stdout, stderr = decode_output(result.stdout), decode_output(result.stderr)
    if result.returncode:
        raise RuntimeError(stderr.strip() or stdout.strip() or "WSL 环境检查失败")
    if stderr.strip():
        print(stderr.strip(), file=sys.stderr)
    if verbose:
        print(stdout.strip())


def main():
    parser = argparse.ArgumentParser(description="配置并启动本地 Qwen 聊天与知识库")
    parser.add_argument("--configure", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--wheel-dir")
    args, cli_args = parser.parse_known_args()
    os.environ.update(PYTHONUTF8="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    try:
        config = prepare_config(args.configure)
        check_frontend(args.wheel_dir)
        check_backend(config, verbose=args.check)
        if args.check:
            print("环境检查通过 未加载模型权重")
            return 0
        return subprocess.run([sys.executable, str(ROOT / "cli.py"), *cli_args], cwd=ROOT).returncode
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
