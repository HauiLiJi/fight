import json
import os
import socket
import urllib.error
import urllib.request
from pathlib import Path

from .chinese_summary import build_chinese_state_summary


DEFAULT_TIMEOUT_S = 0.8
DEFAULT_MODEL = "qwen35b"
DEFAULT_MIN_CALL_INTERVAL_SIM_S = 15.0
DEFAULT_TRIGGER_MODE = "wave"
TRIGGER_MODES = {"wave", "default"}
DEFAULT_PLAN_SELECTION = "survival"
PLAN_SELECTIONS = {"survival", "attack"}
DEFAULT_PLAN_SELECTION_MODE = "user"
PLAN_SELECTION_MODES = {"user", "llm"}
CONFIG_PATH = Path(__file__).with_name("llm_config.json")


def _load_config():
    """Load local defaults without making a bad config stop the fallback agent."""
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
            config = json.load(config_file)
        return config if isinstance(config, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _setting(config, env_name, config_name, default=""):
    return os.getenv(env_name, "").strip() or config.get(config_name, default)


def _number_setting(config, name, default, minimum=None, maximum=None):
    value = config.get(name, default)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if minimum is not None and number < minimum:
        return default
    if maximum is not None and number > maximum:
        return default
    return number


def _bool_setting(config, name, default):
    value = config.get(name, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return default


def llm_enabled():
    config = _load_config()
    if not _bool_setting(config, "enable_llm", True):
        return False
    return bool(_setting(config, "RED_QWEN_BASE_URL", "base_url"))


def get_min_call_interval_sim_s():
    config = _load_config()
    return _number_setting(
        config,
        "llm_min_call_interval_sim_s",
        DEFAULT_MIN_CALL_INTERVAL_SIM_S,
        minimum=0.1,
    )


def get_llm_trigger_mode():
    config = _load_config()
    mode = config.get("llm_trigger_mode", DEFAULT_TRIGGER_MODE)
    return mode if isinstance(mode, str) and mode in TRIGGER_MODES else DEFAULT_TRIGGER_MODE


def get_llm_plan_selection():
    config = _load_config()
    selection = config.get("llm_plan_selection", DEFAULT_PLAN_SELECTION)
    return (
        selection
        if isinstance(selection, str) and selection in PLAN_SELECTIONS
        else DEFAULT_PLAN_SELECTION
    )


def get_llm_plan_selection_mode():
    config = _load_config()
    mode = config.get("llm_plan_selection_mode", DEFAULT_PLAN_SELECTION_MODE)
    return (
        mode
        if isinstance(mode, str) and mode in PLAN_SELECTION_MODES
        else DEFAULT_PLAN_SELECTION_MODE
    )


def build_chat_completion_request(prompt_text, state_summary):
    config = _load_config()
    return {
        "model": _setting(config, "RED_QWEN_MODEL", "model", DEFAULT_MODEL),
        "messages": [
            {"role": "system", "content": prompt_text},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "态势摘要": build_chinese_state_summary(state_summary),
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            },
        ],
        "temperature": _number_setting(config, "temperature", 0.2, 0.0, 1.0),
        "max_tokens": int(_number_setting(config, "max_tokens", 180, minimum=1)),
        "response_format": {"type": "json_object"},
    }


def request_tactic(prompt_text, state_summary, timeout_s=DEFAULT_TIMEOUT_S, request_payload=None):
    config = _load_config()
    base_url = _setting(config, "RED_QWEN_BASE_URL", "base_url")
    if not base_url:
        return {"ok": False, "error": "missing_base_url"}

    url = base_url.rstrip("/") + "/chat/completions"
    payload = request_payload or build_chat_completion_request(prompt_text, state_summary)
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
    }
    api_key = _setting(config, "RED_QWEN_API_KEY", "api_key")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = urllib.request.Request(
        url=url,
        data=body,
        headers=headers,
        method="POST",
    )
    try:
        configured_timeout = _number_setting(config, "timeout_s", timeout_s, minimum=0.1)
        with urllib.request.urlopen(request, timeout=configured_timeout) as response:
            raw_body = response.read()
            status_code = getattr(response, "status", None)
            if status_code is None:
                status_code = response.getcode()
    except urllib.error.HTTPError as error:
        return {
            "ok": False,
            "error": "http_error",
            "status_code": getattr(error, "code", None),
            "message": str(error),
            "raw_response": _read_error_response(error),
        }
    except urllib.error.URLError as error:
        return {
            "ok": False,
            "error": "network_error",
            "message": str(error.reason),
        }
    except socket.timeout:
        return {
            "ok": False,
            "error": "timeout",
        }
    except OSError as error:
        return {
            "ok": False,
            "error": "os_error",
            "message": str(error),
        }

    if status_code < 200 or status_code >= 300:
        return {
            "ok": False,
            "error": "bad_status",
            "status_code": status_code,
        }

    try:
        response_json = json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return {
            "ok": False,
            "error": "invalid_response_json",
            "message": str(error),
        }

    message_content = _extract_message_content(response_json)
    reasoning_content = _extract_reasoning_content(response_json)
    if not message_content:
        return {
            "ok": False,
            "error": "empty_message_content",
            "status_code": status_code,
            "raw_response": response_json,
            "assistant_content": None,
            "reasoning_content": reasoning_content,
        }
    try:
        tactic_json = json.loads(message_content)
    except json.JSONDecodeError as error:
        return {
            "ok": False,
            "error": "invalid_tactic_json",
            "message": str(error),
            "status_code": status_code,
            "raw_response": response_json,
            "raw_content": message_content,
            "assistant_content": message_content,
            "reasoning_content": reasoning_content,
        }
    return {
        "ok": True,
        "status_code": status_code,
        "raw_response": response_json,
        "raw_content": message_content,
        "assistant_content": message_content,
        "reasoning_content": reasoning_content,
        "tactic": tactic_json,
    }


def _extract_message_content(response_json):
    return _extract_message_field(response_json, "content") or ""


def _extract_reasoning_content(response_json):
    return _extract_message_field(response_json, "reasoning_content")


def _extract_message_field(response_json, field_name):
    choices = response_json.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    first_choice = choices[0] or {}
    message = first_choice.get("message") or {}
    content = message.get(field_name)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        fragments = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                fragments.append(item.get("text", ""))
        return "".join(fragments)
    return None


def _read_error_response(error):
    try:
        raw_body = error.read()
        if not raw_body:
            return None
        try:
            return json.loads(raw_body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return raw_body.decode("utf-8", errors="replace")
    except OSError:
        return None
