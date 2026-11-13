"""Request options, llama.cpp context checks and redacted experiment traces."""
from __future__ import annotations

from hashlib import sha256
import json
import math
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, ProxyHandler

from .llm_profiles import model_identity


def local_backend(config):
    return config.get("backend") == "llama.cpp"


def open_completion(request, timeout, config, default_open):
    if local_backend(config):
        return build_opener(ProxyHandler({})).open(request, timeout=timeout)
    return default_open(request, timeout=timeout)


def request_options(body, config, trace=None):
    for key in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repeat_penalty", "seed"):
        if key not in config:
            continue
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f"{key} 必须为有限数值")
        if key in {"top_k", "seed"} and (not isinstance(value, int) or value < 0):
            raise ValueError(f"{key} 必须为非负整数")
        if key in {"top_p", "min_p"} and not 0 <= value <= 1:
            raise ValueError(f"{key} 必须在 0 到 1 之间")
        if key in {"temperature", "repeat_penalty"} and value < 0:
            raise ValueError(f"{key} 不能为负数")
        body[key] = value
    if "chat_template_kwargs" in config:
        kwargs = config["chat_template_kwargs"]
        if (not isinstance(kwargs, dict) or set(kwargs) != {"enable_thinking"}
                or not isinstance(kwargs["enable_thinking"], bool)):
            raise ValueError("chat_template_kwargs 只能包含布尔值 enable_thinking")
        body["chat_template_kwargs"] = dict(kwargs)
    if trace is not None:
        trace.update(model_identity(config), status="request_prepared")
    if local_backend(config):
        context = check_local_context(config, body)
        if trace is not None:
            trace["context"] = context
        # Refuse context shifting: a long request must not silently lose its prefix.
        body["n_keep"] = -1
    if trace is not None:
        trace["request_sha256"] = sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return body


def local_json(base, path, payload=None):
    parsed = urlsplit(base)
    if parsed.scheme != "http" or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError("llama.cpp 配置必须使用本机 HTTP 地址")
    url = f"{parsed.scheme}://{parsed.netloc}{path}"
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with build_opener(ProxyHandler({})).open(request, timeout=15) as response:
            return json.load(response)
    except HTTPError as exc:
        raise RuntimeError(f"本地模型检查接口 HTTP {exc.code}，请检查 llama.cpp 服务状态。") from None
    except (URLError, TimeoutError, OSError):
        raise RuntimeError("无法连接本地 Qwen，请先运行 local_ai 文件夹中的“启动模型.bat”。") from None


def check_local_context(config, body):
    configured = config.get("context_size", 16384)
    margin = config.get("context_margin_tokens", 128)
    if type(configured) is not int or configured <= 0 or type(margin) is not int or margin < 0:
        raise ValueError("context_size 和 context_margin_tokens 配置无效")
    props = local_json(config["base_url"], "/props")
    actual = props.get("default_generation_settings", {}).get("n_ctx")
    if type(actual) is not int or actual <= 0:
        raise RuntimeError("本地服务未返回有效的上下文大小，已停止发送任务。")
    models = local_json(config["base_url"], "/v1/models").get("data", [])
    if config["model"] not in [item.get("id") for item in models]:
        raise RuntimeError("本地服务加载的模型与 Qwen 配置不一致，已停止发送任务。")
    rendered = local_json(config["base_url"], "/apply-template", {
        "messages": body["messages"], "chat_template_kwargs": body.get("chat_template_kwargs", {}),
    }).get("prompt")
    if not isinstance(rendered, str):
        raise RuntimeError("无法检查本地模型的提示词长度，已停止发送任务。")
    tokens = local_json(config["base_url"], "/tokenize", {
        "content": rendered, "add_special": False, "parse_special": True,
    }).get("tokens")
    if not isinstance(tokens, list):
        raise RuntimeError("本地 tokenizer 未返回有效 token 计数。")
    limit = min(configured, actual)
    output = body.get("max_tokens", body.get("max_completion_tokens"))
    prompt = len(tokens)
    if prompt + output + margin > limit:
        raise RuntimeError(f"本地上下文不足：输入 {prompt} + 输出预算 {output} + 预留 {margin} > {limit} token。"
                           "请缩短问题或新建对话，或在配置中调整输出预算/上下文；本次未执行推理。")
    return {"prompt_tokens": prompt, "output_budget": output, "margin_tokens": margin,
            "configured_context": configured, "server_context": actual, "effective_context": limit,
            "remaining_tokens": limit - prompt - output - margin}


def start_trace(trace):
    if trace is not None:
        trace["_started"] = time.monotonic()


def record_response(trace, payload):
    if trace is None:
        return
    trace["elapsed_seconds"] = time.monotonic() - trace.pop("_started", time.monotonic())
    trace["status"] = "response_received"
    trace["response_model"] = payload.get("model")
    choice = (payload.get("choices") or [{}])[0]
    trace["finish_reason"] = choice.get("finish_reason")
    message = choice.get("message")
    trace["reasoning_present"] = isinstance(message, dict) and bool(message.get("reasoning_content"))
    usage = payload.get("usage")
    if isinstance(usage, dict):
        trace["usage"] = {key: value for key, value in usage.items()
                          if key in {"prompt_tokens", "completion_tokens", "total_tokens"}
                          and type(value) is int and value >= 0}
        details = usage.get("completion_tokens_details", {})
        if isinstance(details, dict) and type(details.get("reasoning_tokens")) is int:
            trace["usage"]["reasoning_tokens"] = details["reasoning_tokens"]


def complete_trace(trace):
    if trace is not None:
        trace["status"] = "validated"


def incomplete_message(choice, stage):
    return (f"{stage}输出不完整：finish_reason={choice.get('finish_reason')!r}。"
            "可能达到输出或上下文上限；请调整 token 预算或关闭思考后重试，未启动求解。")
