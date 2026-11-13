"""List provider models without displaying request headers or credentials."""
import json
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

root = Path(__file__).resolve().parents[1]
credentials = dict(line.split("=", 1) for line in (root / ".env").read_text(encoding="utf-8").splitlines()
                   if line.strip() and not line.startswith("#"))
providers = [
    ("jev", "https://api.typesafe.ai/v1/models", "TYPESAFE_API_KEY"),
    ("deepseek", "https://api.deepseek.com/models", "DEEPSEEK_API_KEY"),
]
results = {}
for provider, url, key_name in providers:
    request = Request(url, headers={"Authorization": "Bearer " + credentials[key_name], "Accept": "application/json"})
    try:
        with urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        results[provider] = {"status": "ok", "models": payload}
    except HTTPError as exc:
        results[provider] = {"status": "error", "http_status": exc.code}
    except (URLError, TimeoutError, ValueError):
        results[provider] = {"status": "error", "reason": "connection_or_response_error"}
output = root / "outputs/diagnostics/api_models.json"
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
for provider, result in results.items():
    if result["status"] == "ok":
        models = result["models"]
        entries = models.get("data", models.get("models", [])) if isinstance(models, dict) else models
        print(provider, "available_models:", [entry.get("id", entry.get("name")) for entry in entries if isinstance(entry, dict)])
    else:
        print(provider, result)
