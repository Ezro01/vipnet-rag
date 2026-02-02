"""LLM через Yandex rest-assistant API (Yandex GPT / Qwen)."""

import json

from config import (
    RAG_LLM_MAX_NEW_TOKENS,
    YANDEX_CLOUD_FOLDER,
    YANDEX_CLOUD_API_KEY,
    YANDEX_CLOUD_MODEL,
    YANDEX_CLOUD_BASE_URL,
)


def _prompt_to_messages(prompt: str) -> list:
    """Разбивает промпт на system + user (instructions + input)."""
    sep = "Контекст из документации:"
    if sep not in prompt:
        return [{"role": "user", "content": prompt}]
    idx = prompt.index(sep)
    system = prompt[:idx].strip()
    user_part = prompt[idx:].rstrip()
    if user_part.endswith("\n\nОтвет:"):
        user_part = user_part[:- len("\n\nОтвет:")].strip()
    return [{"role": "system", "content": system}, {"role": "user", "content": user_part}]


def _parse_yandex_response(data: dict) -> str:
    """Извлекает текст из ответа API; при ошибке или пустом ответе — RuntimeError."""
    err = data.get("error") or data.get("message")
    if err:
        msg = err if isinstance(err, str) else (err.get("message") or str(err))
        raise RuntimeError(f"Yandex API: {msg}")
    text = (data.get("output_text") or "").strip()
    if text:
        return text
    for item in (data.get("output") or []):
        for block in (item.get("content") or []):
            if block.get("type") in ("output_text", "text") or block.get("text"):
                t = (block.get("text") or "").strip()
                if t:
                    return t
    result = data.get("result") or {}
    for alt in (result.get("alternatives") or []):
        t = (alt.get("message") or {}).get("text") or ""
        if t.strip():
            return t.strip()
    raise RuntimeError("Yandex API вернул пустой текст. Фрагмент: " + json.dumps(data, ensure_ascii=False)[:800])


def _yandex_generate(prompt: str, max_new_tokens: int) -> str:
    """Ответ через Yandex rest-assistant API."""
    import httpx

    messages = _prompt_to_messages(prompt)
    if len(messages) >= 2 and messages[0].get("role") == "system":
        instructions = messages[0].get("content") or messages[0].get("text") or ""
        user_input = messages[1].get("content") or messages[1].get("text") or prompt
    else:
        instructions = ""
        user_input = (messages[0].get("content") or messages[0].get("text") or prompt) if messages else prompt

    base_url = (YANDEX_CLOUD_BASE_URL or "https://rest-assistant.api.cloud.yandex.net/v1").rstrip("/")
    model_uri = f"gpt://{YANDEX_CLOUD_FOLDER}/{YANDEX_CLOUD_MODEL}"
    payload = {"model": model_uri, "temperature": 0.3, "instructions": instructions, "input": user_input, "max_output_tokens": int(max_new_tokens)}

    try:
        import openai
        with httpx.Client(timeout=120.0) as http_client:
            kwargs = {"api_key": YANDEX_CLOUD_API_KEY, "base_url": base_url, "http_client": http_client}
            try:
                client = openai.OpenAI(**kwargs, project=YANDEX_CLOUD_FOLDER)
            except TypeError:
                client = openai.OpenAI(**kwargs)
            if hasattr(client, "responses") and hasattr(client.responses, "create"):
                response = client.responses.create(**payload)
                return (getattr(response, "output_text", None) or "").strip()
    except Exception:
        pass

    with httpx.Client(timeout=120.0) as client:
        resp = client.post(
            f"{base_url}/responses",
            headers={"Authorization": f"Api-Key {YANDEX_CLOUD_API_KEY}", "Content-Type": "application/json", "x-folder-id": YANDEX_CLOUD_FOLDER},
            json=payload,
        )
    if resp.status_code != 200:
        raise RuntimeError(f"Yandex API error {resp.status_code}: {resp.text[:500]}")
    return _parse_yandex_response(resp.json())


def load_llm():
    return None, None


def llm_generate(prompt: str, max_new_tokens: int = None) -> str:
    max_new_tokens = max_new_tokens or RAG_LLM_MAX_NEW_TOKENS
    if not YANDEX_CLOUD_API_KEY or not YANDEX_CLOUD_FOLDER:
        raise RuntimeError("Задайте YANDEX_CLOUD_API_KEY и YANDEX_CLOUD_FOLDER в .env")
    return _yandex_generate(prompt, max_new_tokens)


def llm_generate_stream(prompt: str, max_new_tokens: int = None):
    max_new_tokens = max_new_tokens or RAG_LLM_MAX_NEW_TOKENS
    if not YANDEX_CLOUD_API_KEY or not YANDEX_CLOUD_FOLDER:
        raise RuntimeError("Задайте YANDEX_CLOUD_API_KEY и YANDEX_CLOUD_FOLDER в .env")
    text = _yandex_generate(prompt, max_new_tokens)
    for word in text.split():
        yield word + " "
