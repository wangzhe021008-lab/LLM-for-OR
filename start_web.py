"""Open the local workspace once; repeated shortcut clicks reuse its server."""
from __future__ import annotations

import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.request import urlopen
import webbrowser

from web_app import APP_KIND, ROOT, project_id


def healthy(port):
    try:
        with urlopen(f"http://127.0.0.1:{port}/api/health", timeout=0.5) as response:
            info = json.load(response)
            return info.get("kind") == APP_KIND and info.get("project_id") == project_id(ROOT)
    except (OSError, ValueError):
        return False


def available(port):
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def launch(open_browser=True):
    diagnostics = ROOT / "outputs/diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    for port in range(8765, 8785):
        if healthy(port):
            url = f"http://127.0.0.1:{port}"
            if open_browser:
                webbrowser.open(url)
            return url
    port = next((p for p in range(8765, 8785) if available(p)), None)
    if port is None:
        raise RuntimeError("本地端口 8765～8784 均被占用，请稍后重试。")
    python = ROOT / ".venv/Scripts/python.exe"
    if not python.is_file():
        raise RuntimeError("缺少 location/.venv Python 环境，请先按 README 安装。")
    log_path = diagnostics / "web_server.log"
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen([str(python), "-X", "utf8", str(ROOT / "web_app.py"), "--port", str(port)],
                                   cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                   creationflags=flags, shell=False, close_fds=True)
    for _ in range(100):
        if healthy(port):
            url = f"http://127.0.0.1:{port}"
            (diagnostics / "web_server.json").write_text(json.dumps({"pid": process.pid, "port": port, "url": url}), encoding="utf-8")
            if open_browser:
                webbrowser.open(url)
            return url
        if process.poll() is not None:
            raise RuntimeError(f"网页启动失败，请查看 {log_path}")
        time.sleep(0.2)
    raise RuntimeError(f"网页启动超时，请查看 {log_path}")


if __name__ == "__main__":
    try:
        url = launch()
        if sys.stdout:
            print(url)
    except Exception as exc:
        if sys.stderr:
            print(str(exc), file=sys.stderr)
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, str(exc), "ORLLM 网页启动失败", 0x10)
        raise SystemExit(1)
