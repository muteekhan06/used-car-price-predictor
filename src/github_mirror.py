from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def github_mirror_enabled() -> bool:
    return bool(os.getenv("GITHUB_LOG_TOKEN") and os.getenv("GITHUB_LOG_REPO"))


def _headers() -> dict[str, str]:
    token = os.getenv("GITHUB_LOG_TOKEN", "")
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "used-car-price-predictor",
    }


def _contents_url(repo: str, path: str, branch: str) -> str:
    return f"https://api.github.com/repos/{repo}/contents/{path}?ref={branch}"


def _request_json(url: str, method: str = "GET", payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(url, method=method, data=data, headers=_headers())
    request.add_header("Content-Type", "application/json")
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def mirror_prediction(payload: dict) -> dict:
    if not github_mirror_enabled():
        return {"enabled": False, "mirrored": False, "detail": "GitHub mirror is disabled."}

    repo = os.getenv("GITHUB_LOG_REPO", "").strip()
    branch = os.getenv("GITHUB_LOG_BRANCH", "main").strip() or "main"
    base_dir = os.getenv("GITHUB_LOG_DIR", "prediction-logs").strip() or "prediction-logs"
    now = datetime.now(timezone.utc)
    target_path = str(Path(base_dir) / f"{now:%Y}" / f"{now:%m}" / f"{now:%Y-%m-%d}.jsonl").replace("\\", "/")
    url = _contents_url(repo, target_path, branch)

    existing_text = ""
    sha = None
    try:
        current = _request_json(url)
        sha = current.get("sha")
        encoded = current.get("content", "").replace("\n", "")
        existing_text = base64.b64decode(encoded).decode("utf-8") if encoded else ""
    except HTTPError as exc:
        if exc.code != 404:
            raise

    line = json.dumps(payload, ensure_ascii=True) + "\n"
    content = base64.b64encode((existing_text + line).encode("utf-8")).decode("utf-8")
    body = {
        "message": f"log prediction {payload.get('prediction_id', 'unknown')}",
        "content": content,
        "branch": branch,
    }
    if sha:
        body["sha"] = sha

    response = _request_json(f"https://api.github.com/repos/{repo}/contents/{target_path}", method="PUT", payload=body)
    return {
        "enabled": True,
        "mirrored": True,
        "path": target_path,
        "commit_sha": response.get("commit", {}).get("sha"),
    }
