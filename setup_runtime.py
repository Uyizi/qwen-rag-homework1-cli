import os
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

CUDA_VERSION_PATTERN = r"CUDA(?:\s+UMD)?\s+Version:\s*(\d+)"


def direct_environment():
    result = {key: value for key, value in os.environ.items() if not key.lower().endswith("_proxy")}
    result.update(NO_PROXY="*", no_proxy="*", PIP_CONFIG_FILE=os.devnull)
    return result


def choose_windows_python(candidates):
    supported = []
    for candidate in dict.fromkeys(candidates):
        try:
            result = subprocess.run([candidate, "-I", "-c", "import json,sys; print(json.dumps(list(sys.version_info[:2])))"],
                                    capture_output=True, text=True, timeout=10)
            version = tuple(json.loads(result.stdout)) if result.returncode == 0 else ()
            if (3, 10) <= version <= (3, 13):
                supported.append((version, candidate))
        except (OSError, ValueError, subprocess.SubprocessError):
            continue
    if not supported:
        raise RuntimeError("请安装 Windows Python 3.10至3.13 推荐3.13 然后重新启动")
    return max(supported, key=lambda item: item[0])[1]


def create_frontend(root):
    candidates = [sys.executable]
    launcher = shutil.which("py")
    if launcher:
        result = subprocess.run([launcher, "-0p"], capture_output=True, text=True, timeout=15)
        for line in result.stdout.splitlines():
            match = re.search(r"\s([A-Za-z]:\\.+\.exe)\s*$", line)
            if match:
                candidates.append(match[1])
    python = choose_windows_python(candidates)
    subprocess.run([python, "-m", "venv", str(Path(root).resolve() / ".venv")], check=True)


def prepare_backend(requirements):
    path = str(Path(requirements).resolve()).replace("\\", "/")
    if len(path) > 2 and path[1] == ":":
        path = "/mnt/" + path[0].lower() + path[2:]
    script = '''import os,re,shutil,subprocess,sys
from pathlib import Path
if sys.version_info < (3,10):
    raise SystemExit('WSL requires Python 3.10 or newer')
nvidia=shutil.which('nvidia-smi') or '/usr/lib/wsl/lib/nvidia-smi'
gpu=subprocess.run([nvidia],capture_output=True,text=True)
if gpu.returncode:
    raise SystemExit('WSL cannot access the NVIDIA GPU; install a compatible NVIDIA driver')
cuda=re.search(sys.argv[2],gpu.stdout)
if not cuda or int(cuda[1])<13:
    raise SystemExit('This runtime requires an NVIDIA driver supporting CUDA 13 or newer')
root=Path.home()/'.local/share/qwen-rag-homework1'
python=root/'venv/bin/python'
marker=root/'installed-requirements.txt'
requirements=Path(sys.argv[1]).read_text()
if not python.is_file() or not marker.is_file() or marker.read_text()!=requirements:
    print('Preparing WSL model environment',file=sys.stderr,flush=True)
    root.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,'-m','venv',str(root/'venv')],check=True,stdout=sys.stderr)
    env={k:v for k,v in os.environ.items() if not k.lower().endswith('_proxy')}
    env.update(NO_PROXY='*',no_proxy='*',PIP_CONFIG_FILE='/dev/null')
    subprocess.run([str(python),'-m','pip','install','--index-url','https://pypi.org/simple',
        '--extra-index-url','https://download.pytorch.org/whl/cu130','-r',sys.argv[1]],
        check=True,env=env,stdout=sys.stderr)
    marker.write_text(requirements)
subprocess.run([str(python),'-c',
    "import torch; assert torch.cuda.is_available(), 'CUDA GPU unavailable'; assert torch.cuda.is_bf16_supported(), 'BF16 GPU required'"],
    check=True,stdout=sys.stderr)
print(python)
'''
    result = subprocess.run(["wsl", "--exec", "python3", "-c", script, path, CUDA_VERSION_PATTERN],
                            stdout=subprocess.PIPE, env=direct_environment())
    if result.returncode:
        raise RuntimeError("WSL 环境准备失败 请确认 Linux 已安装 Python 3.10以上和 python3-venv 且 NVIDIA 驱动支持 CUDA 13")
    python = result.stdout.decode("utf-8-sig").strip()
    if not python.startswith("/") or "\n" in python:
        raise RuntimeError("无法确定 WSL Python 路径")
    return python


if __name__ == "__main__":
    try:
        if len(sys.argv) != 3 or sys.argv[1] != "--frontend":
            raise RuntimeError("用法 setup_runtime.py --frontend 项目目录")
        create_frontend(sys.argv[2])
    except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
