"""Интерактивный чат в терминале: RAG + LLM, вывод ответа по словам."""

import re

from rag import init_rag, rag_pipeline
from llm import load_llm, llm_generate_stream


def _print_pretty_answer(text: str) -> None:
    """Форматирование ответа LLM для терминала: артефакты PDF, код-блоки, заголовки."""
    s = (text or "").strip()
    if not s:
        return
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
    print("\n".join(lines).strip())


def run_chat():
    print("Загрузка RAG...")
    init_rag()
    load_llm()
    print("Готово. Введите вопрос (пустая строка — выход).\n")
    while True:
        try:
            q = input("Вопрос: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        context, top_chunks, prompt = rag_pipeline(q, top_k_retrieve=50, top_k_rerank=5, mode="chat")
        print("\nОтвет:\n")
        answer_parts = []
        try:
            for chunk in llm_generate_stream(prompt):
                answer_parts.append(chunk)
        except RuntimeError as e:
            print(f"(Ошибка: {e})")
        answer_text = "".join(answer_parts).strip()
        if not answer_text:
            print("(Модель не вернула текст. Возможны ограничения API, длина промпта или ошибка разбора ответа.)")
        else:
            _print_pretty_answer(answer_text)
        if answer_text and "источники" not in answer_text.lower():
            print("\nИсточники:")
            seen = set()
            for c in top_chunks:
                doc = c.get("doc_name", c.get("doc", ""))
                ch = c.get("chapter_title", "")
                sid = c.get("section_id", "")
                key = (doc, ch, c.get("section", ""), sid)
                if key in seen:
                    continue
                seen.add(key)
                print(f"  - {' | '.join(p for p in (doc, ch, sid) if p)}")
            print()
