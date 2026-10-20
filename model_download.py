import http.client
import json
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath


def direct_opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def validate_manifest(manifest):
    seen = set()
    for model in manifest.get("models", []):
        if model["key"] not in {"generation_model", "embedding_model"} or model["key"] in seen:
            raise ValueError("模型清单存在未知或重复配置项")
        seen.add(model["key"])
        for name in [model["folder"], *(entry["path"] for entry in model["files"])]:
            if not name or name in {".", ".."} or PurePosixPath(name).name != name or any(c in name for c in "\\:"):
                raise ValueError("模型清单包含非法文件路径")
        for entry in model["files"]:
            for address in [entry["url"], *([entry["api_url"]] if "api_url" in entry else [])]:
                url = urllib.parse.urlsplit(address)
                trusted = url.scheme == "https" and (
                    url.netloc == "github.com" and url.path.startswith("/Uyizi/qwen-rag-homework1-cli/releases/download/")
                    or url.netloc == "api.github.com" and re.fullmatch(r"/repos/Uyizi/qwen-rag-homework1-cli/releases/assets/\d+", url.path))
                local_test = url.scheme == "http" and url.hostname in {"127.0.0.1", "localhost"}
                if not (trusted or local_test) or not isinstance(entry["size"], int) or entry["size"] <= 0:
                    raise ValueError("模型清单包含不可信下载地址或文件大小")
    if not seen:
        raise ValueError("模型清单为空")
    return manifest


def show_progress(message):
    print(message, file=sys.stderr, flush=True)


def download_file(url, destination, size, progress=show_progress, retries=3):
    destination = Path(destination)
    if destination.is_file() and destination.stat().st_size == size:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    opener = direct_opener()
    for attempt in range(retries):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == size:
            partial.replace(destination)
            return
        if offset > size:
            with partial.open("wb"):
                pass
            offset = 0
        if shutil.disk_usage(destination.parent).free < size - offset + 100 * 1024**2:
            raise OSError("磁盘空间不足 请为模型和依赖预留至少25GB")
        headers = {"User-Agent": "qwen-rag-launcher", "Accept-Encoding": "identity", "Accept": "application/octet-stream"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with opener.open(request, timeout=45) as response:
                status = response.status
                if status == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if not match or int(match[1]) != offset or int(match[3]) != size:
                        raise RuntimeError("断点续传响应与模型清单不符")
                elif status == 200:
                    offset = 0
                else:
                    raise RuntimeError(f"模型下载返回 HTTP {status}")
                last_update = 0
                with partial.open("ab" if offset else "wb") as file:
                    while block := response.read(4 * 1024**2):
                        file.write(block)
                        offset += len(block)
                        if offset > size:
                            raise RuntimeError("下载内容超过模型清单中的文件大小")
                        now = time.monotonic()
                        if now - last_update >= 5:
                            progress(f"下载 {destination.name} {offset / size:.0%}")
                            last_update = now
            if offset != size:
                raise OSError(f"文件下载未完成 {destination.name} {offset}/{size}")
            partial.replace(destination)
            progress(f"已下载 {destination.name}")
            return
        except (OSError, http.client.HTTPException) as exc:
            if attempt + 1 == retries:
                raise OSError(f"模型下载失败 已保留断点 请重新启动后继续 {destination.name}: {exc}") from exc
            progress(f"连接中断 正在续传 {destination.name}")
            time.sleep(1)


def normalized_path(value):
    value = str(value).replace("\\", "/")
    if re.match(r"^[A-Za-z]:/", value):
        return "/mnt/" + value[0].lower() + value[2:]
    return value


def ensure_models(root, config, manifest=None, progress=show_progress):
    root = Path(root).resolve()
    if manifest is None:
        manifest = json.loads((root / "model-manifest.json").read_text(encoding="utf-8"))
    validate_manifest(manifest)
    result = dict(config)
    announced = False
    for model in manifest["models"]:
        target = root / "models" / model["folder"]
        configured = result.get(model["key"])
        if configured and normalized_path(configured) != normalized_path(target):
            continue
        for entry in model["files"]:
            destination = target / entry["path"]
            if not destination.is_file() or destination.stat().st_size != entry["size"]:
                if not announced:
                    progress("首次使用需要下载模型 完整模型约9.6GB 中断后可重新启动续传")
                    announced = True
                download_file(entry.get("api_url", entry["url"]), destination, entry["size"], progress)
        result[model["key"]] = normalized_path(target)
    return result
