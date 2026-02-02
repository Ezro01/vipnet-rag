# RAG Benchmark

Оценка качества RAG: **тестовые запросы всегда генерируются** из чанков (LLM), ручных эталонов нет.

**Production pipeline:** retrieval (Embedding + FAISS + Hybrid RRF) + reranker (BAAI/bge-reranker-v2-m3) по умолчанию. Бенчмарк: Literal + Paraphrase + Scenario; e2e: **SemanticSimilarity** (основная метрика), EM, F1, ROUGE-L.

## Методология (рекомендация ревьюера)

- **Literal QA** — прямой вопрос по фрагменту; retrieval Recall@20 обычно 0.90–0.98.
- **Semantic IR** (Paraphrase + Scenario) — перефразирование и сценарные вопросы; Recall@20 ожидаемо ниже (0.55–0.85). Снижение retrieval при переходе к semantic — **ожидаемая цена за реализм** бенчмарка.
- **SemanticSimilarity** — основная метрика качества ответа (0.8+ = production-grade RAG). EM и ROUGE-L малоинформативны для генеративных ответов.
- В отчёте retrieval показывается **раздельно по типам** (`--eval-mode all`): Literal vs Paraphrase/Scenario.
- Для финального отчёта: подчеркнуть, что **деградация retrieval при semantic** — ожидаемая цена за реализм; **SemanticSimilarity** — основная метрика качества RAG.

## Запуск (из корня проекта)

```bash
# Полный цикл: генерация QA → retrieval → e2e
python main.py eval
# или
python -m rag_benchmark.run_all
```

С параметрами:

```bash
python -m rag_benchmark.run_all --samples 30 --e2e-limit 5 --eval-mode all --output report.json
```

- `--samples 30` — сколько чанков взять для генерации QA (на каждый добавляются paraphrase/scenario при включённой опции).
- `--e2e-limit 20` — лимит примеров для e2e (по умолчанию 20).
- `--eval-mode all` — retrieval по всем запросам + разбивка по типам (literal / paraphrase_or_scenario). `literal` — только literal QA; `semantic` — только paraphrase+scenario.
- `--no-paraphrases` (в generate_queries) — только literal-вопросы.
- `--skip-e2e` — не запускать e2e (только retrieval).
- `--output report.json` — путь к файлу отчёта.

## Что происходит

1. **Генерация** — из чанков LLM генерирует Literal-вопросы и (опционально) Paraphrase/Scenario; сохраняется `benchmark.json` с полями `question`, `answer`, `relevant_chunk_ids`, `query_type`.
2. **Retrieval** — поиск по каждому вопросу; метрики Recall@k, MRR@k, nDCG@k (k=1,3,5,10,20).
3. **End-to-end** — RAG-ответ сравнивается с эталоном: **SemanticSimilarity** (основная), ExactMatch, F1_token, ROUGE-L.

## Структура

```
rag_benchmark/
├── generate_queries.py   # чанки → LLM → question, answer, relevant_chunk_ids
├── evaluate_retrieval.py # Recall@k, MRR@k, nDCG@k
├── evaluate_end2end.py   # SemanticSimilarity, EM, F1, ROUGE-L
├── metrics.py
├── run_all.py            # генерация + все оценки
└── datasets/
    └── benchmark.json   # создаётся при запуске
```

Ручных эталонов (gold.json и т.п.) нет — бенчмарк каждый раз строится заново.
