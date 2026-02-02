#!/usr/bin/env python3
"""
Генерация синтетического QA-бенчмарка: чанки → LLM → вопрос + ответ + relevant_chunk_ids.

Запуск (из корня проекта):
  python -m rag_benchmark.generate_queries [--samples 300] [--max-chars 2500]
"""

import argparse
import json
import os
import random
import sys

# корень проекта = родитель rag_benchmark
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))
except ImportError:
    pass

from tqdm import tqdm

from config import DATASET_JSON

BENCHMARK_DIR = os.path.join(os.path.dirname(__file__), "datasets")
OUTPUT_BENCHMARK = os.path.join(BENCHMARK_DIR, "benchmark.json")

# Literal: один прямой вопрос по фрагменту (рекомендация ревьюера)
PROMPT_QUESTION_ONLY = """Ты — эксперт по технической документации ViPNet Coordinator HW.
По следующему фрагменту документации сформулируй ровно один сложный технический вопрос (на который можно ответить фразой из фрагмента).
Фрагмент:
{chunk}
Ответь строго одной строкой, начиная со слова «Вопрос:» и без лишнего текста.
Вопрос:"""

# Paraphrase + Scenario: 2 перефразирования + 1 сценарный вопрос (репрезентативный бенчмарк по ревьюеру)
PROMPT_PARAPHRASE_SCENARIO = """По техническому вопросу к документации ViPNet Coordinator HW сгенерируй 3 варианта:
1) Перефразирование (те же смысл, другие слова).
2) Ещё одно перефразирование (сленг/сокращения допустимы).
3) Сценарный вопрос (практическая задача пользователя).
Исходный вопрос: {question}
Ответь строго тремя строками, по одному варианту на строку, без нумерации и пояснений."""


def extract_reference_answer(chunk_text: str) -> str:
    """Эталонный ответ = реально существующая фраза из чанка (самое длинное предложение или первое содержательное)."""
    import re
    text = (chunk_text or "").strip()
    if not text:
        return ""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
    if not sentences:
        return text[:500].strip()
    return max(sentences, key=len).strip()


def _parse_three_lines(raw: str) -> list:
    """Из ответа LLM извлекаем до 3 строк (Literal / Paraphrase / Scenario)."""
    lines = [ln.strip() for ln in (raw or "").strip().split("\n") if ln.strip()]
    out = []
    for ln in lines[:3]:
        if ln.lower().startswith("вопрос:"):
            ln = ln[7:].strip()
        if ln and len(ln) > 10:
            out.append(ln)
    return out


def generate(samples: int = 100, max_chunk_chars: int = 2500, seed: int = 42, add_paraphrases: bool = True):
    """Генерация бенчмарка: Literal + (опционально) Paraphrase и Scenario по ревьюеру."""
    from llm import llm_generate

    with open(DATASET_JSON, encoding="utf-8") as f:
        chunks = json.load(f)

    random.seed(seed)
    n = min(samples, len(chunks))
    chosen = random.sample(chunks, n)

    os.makedirs(BENCHMARK_DIR, exist_ok=True)
    dataset = []

    for c in tqdm(chosen, desc="Generate QA"):
        text = (c.get("text") or "")[:max_chunk_chars]
        if len(text.strip()) < 100:
            continue
        prompt = PROMPT_QUESTION_ONLY.format(chunk=text)
        try:
            qa = llm_generate(prompt, max_new_tokens=1024)
        except Exception as e:
            tqdm.write(f"LLM error: {e}")
            continue
        q_literal = (qa or "").strip()
        if q_literal.startswith("Вопрос:"):
            q_literal = q_literal[7:].strip()
        if not q_literal:
            continue
        a = extract_reference_answer(text)
        if not a:
            continue
        # Literal
        dataset.append({
            "question": q_literal,
            "answer": a,
            "relevant_chunk_ids": [c["chunk_id"]],
            "query_type": "literal",
        })
        # Paraphrase + Scenario (2–3 варианта от LLM)
        if add_paraphrases:
            try:
                prompt2 = PROMPT_PARAPHRASE_SCENARIO.format(question=q_literal)
                out = llm_generate(prompt2, max_new_tokens=512)
                for q_alt in _parse_three_lines(out):
                    if q_alt != q_literal:
                        dataset.append({
                            "question": q_alt,
                            "answer": a,
                            "relevant_chunk_ids": [c["chunk_id"]],
                            "query_type": "paraphrase_or_scenario",
                        })
            except Exception:
                pass

    with open(OUTPUT_BENCHMARK, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(dataset)} QA pairs → {OUTPUT_BENCHMARK}")
    return len(dataset)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic QA benchmark from chunks + LLM")
    parser.add_argument("--samples", type=int, default=300, help="Number of chunks to sample")
    parser.add_argument("--max-chars", type=int, default=2500, help="Max chars per chunk for prompt")
    parser.add_argument("--no-paraphrases", action="store_true", help="Only literal questions (no paraphrase/scenario)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate(samples=args.samples, max_chunk_chars=args.max_chars, seed=args.seed, add_paraphrases=not args.no_paraphrases)


if __name__ == "__main__":
    main()
