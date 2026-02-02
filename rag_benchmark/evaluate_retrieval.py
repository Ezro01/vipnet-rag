#!/usr/bin/env python3
"""
Оценка качества retrieval: Recall@k, MRR@k, nDCG@k.
Бенчмарк — только сгенерированный (question, answer, relevant_chunk_ids).
"""

import argparse
import json
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))
except ImportError:
    pass

# Загрузка моделей только из кэша (без запросов к Hugging Face)
os.environ["HF_HUB_OFFLINE"] = "1"

from tqdm import tqdm

from rag_benchmark.metrics import recall_at_k, mrr_at_k, ndcg_at_k

BENCHMARK_DIR = os.path.join(os.path.dirname(__file__), "datasets")
DEFAULT_BENCHMARK = os.path.join(BENCHMARK_DIR, "benchmark.json")
TOP_K_DEFAULT = [1, 3, 5, 10, 20]


def load_benchmark(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def _compute_stats(top_k_list: list, n_processed: int, stats: dict) -> dict:
    """Нормализация счётчиков в средние по k."""
    n = n_processed if n_processed > 0 else 1
    out = {}
    for k in top_k_list:
        out[k] = {m: round(stats[k][m] / n, 4) for m in stats[k]}
    return out


def evaluate(benchmark_path: str, top_k_list: list, eval_mode: str = "all") -> dict:
    """
    Оценка retrieval по бенчмарку.
    eval_mode: "literal" — только query_type=literal; "semantic" — только paraphrase_or_scenario; "all" — все + разбивка по типам.
    """
    from rag import init_rag, retrieve_top_k

    init_rag()
    data = load_benchmark(benchmark_path)
    # Фильтр по режиму (рекомендация ревьюера: раздельно literal / semantic)
    if eval_mode == "literal":
        data = [s for s in data if (s.get("query_type") or "").strip().lower() == "literal"]
    elif eval_mode == "semantic":
        data = [s for s in data if (s.get("query_type") or "").strip().lower() in ("paraphrase_or_scenario", "paraphrase", "scenario")]
    # else: all — без фильтра

    stats = {k: {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0} for k in top_k_list}
    max_k = max(top_k_list)
    n_processed = 0
    # По типам для отчёта (рекомендация ревьюера: Literal ~0.95, Semantic ~0.75)
    by_type = {}  # query_type -> {k: {recall, mrr, ndcg}, n}
    type_counts = {}

    for sample in tqdm(data, desc="Retrieval eval"):
        q = (sample.get("question") or sample.get("query") or "").strip()
        gold_ids = set(str(g) for g in (sample.get("relevant_chunk_ids") or []))
        if not gold_ids:
            continue

        retrieved = retrieve_top_k(q, top_k=max_k)
        ranked_ids = [r["chunk_id"] for r in retrieved]
        qtype = (sample.get("query_type") or "unknown").strip().lower()
        if qtype not in ("literal", "paraphrase_or_scenario", "paraphrase", "scenario"):
            qtype = "other"

        for k in top_k_list:
            stats[k]["recall"] += recall_at_k(ranked_ids, gold_ids, k)
            stats[k]["mrr"] += mrr_at_k(ranked_ids, gold_ids, k)
            stats[k]["ndcg"] += ndcg_at_k(ranked_ids, gold_ids, k)
        n_processed += 1

        if eval_mode == "all" and qtype in ("literal", "paraphrase_or_scenario", "paraphrase", "scenario"):
            if qtype not in by_type:
                by_type[qtype] = {k: {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0} for k in top_k_list}
                type_counts[qtype] = 0
            for k in top_k_list:
                by_type[qtype][k]["recall"] += recall_at_k(ranked_ids, gold_ids, k)
                by_type[qtype][k]["mrr"] += mrr_at_k(ranked_ids, gold_ids, k)
                by_type[qtype][k]["ndcg"] += ndcg_at_k(ranked_ids, gold_ids, k)
            type_counts[qtype] += 1

    retrieval = _compute_stats(top_k_list, n_processed, stats)
    result = {"n_queries": n_processed, "eval_mode": eval_mode, "retrieval": retrieval}

    if eval_mode == "all" and by_type:
        result["retrieval_by_type"] = {}
        for qtype, cnt in type_counts.items():
            result["retrieval_by_type"][qtype] = {
                "n_queries": cnt,
                "metrics": _compute_stats(top_k_list, cnt, by_type[qtype]),
            }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK, help="Path to benchmark.json (from generate_queries)")
    parser.add_argument("--top-k", default="1,3,5,10,20", help="Comma-separated k values")
    parser.add_argument("--eval-mode", default="all", choices=("literal", "semantic", "all"),
                        help="literal=только literal QA; semantic=paraphrase+scenario; all=все + разбивка по типам")
    parser.add_argument("--output", default=None, help="Save report JSON here")
    args = parser.parse_args()

    benchmark_path = args.benchmark
    if not os.path.isfile(benchmark_path):
        print(f"Benchmark not found: {benchmark_path}. Run: python -m rag_benchmark.run_all", file=sys.stderr)
        sys.exit(1)

    top_k_list = [int(x.strip()) for x in args.top_k.split(",") if x.strip()]
    if not top_k_list:
        top_k_list = TOP_K_DEFAULT

    report = evaluate(benchmark_path, top_k_list, eval_mode=args.eval_mode)
    print("\n=== Retrieval metrics ===\n")
    print(json.dumps(report["retrieval"], indent=2))
    print(f"\nn_queries: {report['n_queries']} (eval_mode={report.get('eval_mode', 'all')})")
    if report.get("retrieval_by_type"):
        print("\n--- По типам запросов (Literal vs Semantic) ---")
        for qtype, info in report["retrieval_by_type"].items():
            print(f"\n{qtype}: n={info['n_queries']}")
            print(json.dumps(info["metrics"], indent=2))

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nReport saved: {args.output}")


if __name__ == "__main__":
    main()
