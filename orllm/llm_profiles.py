"""Named LLM configurations; each submitted task owns a frozen copy."""
from __future__ import annotations

import copy
import json
from pathlib import Path


PROFILES = {
    "deepseek": ("DeepSeek API", "llm", "deepseek-flash"),
    "qwen_local": ("本地 Qwen3.5 9B", "qwen", "Qwen3.5-9B-Q8_0"),
}
# Only configuration fields are copied to run artifacts, never literal credentials.
CONFIG_FIELDS = {
    "base_url", "model", "api_key_env", "allow_no_api_key", "response_format",
    "json_schema_supported", "timeout_seconds", "max_tokens", "chat_max_tokens", "workflow_contract",
    "token_parameter", "backend", "context_size", "context_margin_tokens",
    "chat_template_kwargs", "mapping_enable_thinking", "temperature", "top_p", "top_k", "min_p",
    "presence_penalty", "repeat_penalty", "seed", "provider_id", "provider_label",
}


def load_profile(root, source="deepseek"):
    if not isinstance(source, str) or source not in PROFILES:
        raise ValueError("模型来源无效，请选择 DeepSeek API 或本地 Qwen3.5 9B。")
    label, stem, _ = PROFILES[source]
    folder = Path(root) / "config"
    path = folder / f"{stem}.local.json"
    if not path.exists():
        path = folder / f"{stem}.example.json"
    if path.exists():
        config = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(config, dict) or "api_key" in config:
            raise ValueError("模型配置必须是 JSON 对象，密钥请放在环境变量中。")
    elif source == "deepseek":
        config = {}  # Existing workspaces can show the UI before setup is complete.
    else:
        raise ValueError("缺少本地 Qwen 配置，请检查 config/qwen.local.json。")
    config = copy.deepcopy({key: value for key, value in config.items() if key in CONFIG_FIELDS})
    config.update(provider_id=source, provider_label=label)
    return config


def profile_options(root):
    options = []
    for source, (label, _, default_model) in PROFILES.items():
        try:
            config = load_profile(root, source)
            options.append({"id": source, "label": label, "model": config.get("model", default_model),
                            "available": bool(config.get("base_url") and config.get("model")),
                            "thinking": config.get("chat_template_kwargs", {}).get("enable_thinking"),
                            "mapping_thinking": config.get("mapping_enable_thinking", config.get("chat_template_kwargs", {}).get("enable_thinking")),
                            "context_size": config.get("context_size")})
        except (ValueError, OSError):
            options.append({"id": source, "label": label, "model": default_model, "available": False})
    return options


def model_identity(config):
    """Public experiment metadata: no API keys, environment values or prompts."""
    fields = CONFIG_FIELDS - {"api_key_env", "allow_no_api_key"}
    return copy.deepcopy({key: value for key, value in config.items() if key in fields})
