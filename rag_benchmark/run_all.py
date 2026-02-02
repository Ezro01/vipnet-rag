#!/usr/bin/env python3
"""
Запуск оценки RAG: генерация QA (LLM) → retrieval → e2e. Reranker из метрик исключён.

Эталонных файлов нет — бенчмарк создаётся автоматически из чанков.

Запуск (из корня проекта):
  python -m rag_benchmark.run_all [--samples 30] [--e2e-limit 5] [--output report.json]
"""

import argparse
import json
import os
import sys

# Снизить риск segfault на macOS при загрузке FAISS/NumPy (retrieval и e2e)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_DIR = os.path.join(_PROJECT_ROOT, "src")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

BENCHMARK_DIR = os.path.join(os.path.dirname(__file__), "datasets")
BENCHMARK_JSON = os.path.join(BENCHMARK_DIR, "benchmark.json")


def main():
    parser = argparse.ArgumentParser(description="RAG Benchmark: generate QA → retrieval → e2e")
    parser.add_argument("--samples", type=int, default=30, help="Number of QA pairs to generate from chunks")
    parser.add_argument("--e2e-limit", type=int, default=20, help="Limit samples for e2e (slow)")
    parser.add_argument("--skip-e2e", action="store_true", help="Skip end-to-end evaluation")
    parser.add_argument("--eval-mode", default="all", choices=("literal", "semantic", "all"),
                        help="Retrieval: literal=только literal QA; semantic=paraphrase+scenario; all=все + разбивка по типам")
    parser.add_argument("--output", default=None, help="Save full report JSON here")
    args = parser.parse_args()

    os.makedirs(BENCHMARK_DIR, exist_ok=True)

    # 1. Всегда генерируем тестовые запросы (вопрос + ответ + relevant_chunk_ids)
    from rag_benchmark.generate_queries import generate
    n = generate(samples=args.samples)
    if n == 0:
        print("No QA pairs generated. Check Yandex API and dataset.", file=sys.stderr)
        sys.exit(1)
    print(f"Generated {n} QA pairs → {BENCHMARK_JSON}")

    report = {"benchmark": BENCHMARK_JSON, "n_queries": n}

    # 2. Оценка retrieval (используем кэш HF, без сети). Режим: literal / semantic / all (разбивка по типам)
    from rag_benchmark.evaluate_retrieval import evaluate as eval_retrieval
    retrieval_report = eval_retrieval(BENCHMARK_JSON, [1, 3, 5, 10, 20], eval_mode=args.eval_mode)
    report["retrieval"] = retrieval_report["retrieval"]
    report["retrieval_report"] = retrieval_report
    print("\n=== Retrieval ===\n", json.dumps(report["retrieval"], indent=2))
    if retrieval_report.get("retrieval_by_type"):
        print("\n--- По типам запросов (Literal vs Semantic) ---")
        for qtype, info in retrieval_report["retrieval_by_type"].items():
            print(f"\n{qtype}: n={info['n_queries']}")
            print(json.dumps(info["metrics"], indent=2))

    # 3. End-to-end (опционально)
    if not args.skip_e2e:
        from rag_benchmark.evaluate_end2end import evaluate as eval_e2e
        report["end2end"] = eval_e2e(BENCHMARK_JSON, limit=args.e2e_limit)
        print("\n=== End-to-End ===\n", json.dumps(report["end2end"].get("metrics", {}), indent=2))
    else:
        report["end2end"] = {"skipped": True}

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nFull report saved: {args.output}")


if __name__ == "__main__":
    main()
