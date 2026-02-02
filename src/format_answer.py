"""Форматирование ответа LLM: артефакты PDF, код-блоки, заголовки, списки."""

import re


def format_pretty_answer(text: str) -> str:
    """
    Приводит сырой текст ответа LLM к читаемому виду:
    убирает артефакты PDF, оформляет код-блоки, заголовки ###/####, нумерованные и маркированные списки.
    """
    s = (text or "").strip()
    if not s:
        return ""
    s = re.sub(r"<image:[^>]*>", "", s)
    s = re.sub(r"``", "", s)
    for lang in ("bash", "plaintext"):
        s = re.sub(rf"`{lang}\s*\n\s*(hostname#[^\n]+)\s*\n\s*`", r"\n```bash\n\1\n```\n\n", s)
        s = re.sub(rf"`{lang}\s*\n\s*(hostname#[^\n]+)\s*\n\s+(?=[A-ZА-Я])", r"\n```bash\n\1\n```\n\n", s)
        s = re.sub(rf"`{lang} (hostname#[^`]*?)\s*`", r"\n```bash\n\1\n```\n\n", s)
        s = re.sub(rf"`{lang} (hostname#[^`]*?)\s{{2,}}(?=[A-ZА-Я])", r"\n```bash\n\1\n```\n\n", s)
    s = re.sub(r"```(\w+)\s+([^\n`]+)", r"```\1\n\2", s)
    s = re.sub(r"([^\n])```", r"\1\n```", s)
    s = re.sub(r"```(?!\w)\s+", r"```\n\n", s)
    s = re.sub(r"([^\n])(###\s+)", r"\1\n\n\2", s)
    s = re.sub(r"([^\n])(####\s+)", r"\1\n\n\2", s)
    s = re.sub(r"(###\s+[^\n#]+?)\s*#\s*(\n|$)", r"\1\2", s)
    s = re.sub(r"\n\s*#\s*\n", "\n\n", s)
    s = re.sub(r"\s+#\s*(\n|$)", r"\1", s)
    s = re.sub(r"(###\s+[^\n]+?)(\d+\.\s+\*\*)", r"\1\n\n\2", s)
    s = re.sub(r"([.!?])\s+(\d+\.\s+\*\*)", r"\1\n\n\2", s)
    s = re.sub(r"(\*\*)\s+(\d+\.\s+\*\*)", r"\1\n\n\2", s)
    s = re.sub(r"`\s+(\d+\.\s+\*\*)", r"`\n\n\1", s)
    s = re.sub(r"([^\n])\s+(- \*\*)", r"\1\n\n\2", s)
    s = re.sub(r"([:.\)*])\s+(1\.\s+)", r"\1\n\n\2", s)
    s = re.sub(r"^тройка DNS-сервера\b", "### Настройка DNS-сервера", s, flags=re.MULTILINE)
    lines = [ln.rstrip() for ln in s.splitlines()]
    return "\n".join(lines).strip()
