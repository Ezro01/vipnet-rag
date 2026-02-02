"""Интерактивный чат в терминале: RAG + LLM, вывод ответа по словам."""

from format_answer import format_pretty_answer
from rag import init_rag, rag_pipeline
from llm import load_llm, llm_generate_stream


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
            print(format_pretty_answer(answer_text))
        if answer_text and "источники" not in answer_text.lower():
            print("\nИсточники:")
            seen = set()
            for c in top_chunks:
                doc = c.get("doc_name", c.get("doc", ""))
                ch = c.get("chapter_title", "")
                sid = c.get("section_id", "")
                key = (doc, ch, sid)
                if key in seen:
                    continue
                seen.add(key)
                print(f"  - {' | '.join(p for p in (doc, ch, sid) if p)}")
            print()
