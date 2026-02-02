#!/usr/bin/env python3
"""
End-to-End RAG: вопрос → RAG + LLM → метрики SemanticSimilarity (основная) и F1.

Запуск из корня проекта:
  python -m rag_benchmark.evaluate_end2end [--benchmark path] [--limit N]
"""

import argparse
import json
import os
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# Снижает риск segfault на macOS при загрузке FAISS/NumPy/PyTorch
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))
except ImportError:
    pass

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")

from tqdm import tqdm

from rag_benchmark.metrics import f1_token_overlap, normalize_answer, semantic_similarity

BENCHMARK_DIR = os.path.join(os.path.dirname(__file__), "datasets")
DEFAULT_BENCHMARK = os.path.join(BENCHMARK_DIR, "benchmark.json")


def load_benchmark(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else [data]


def evaluate(benchmark_path: str, limit: int = None, timeout_sec: float = 180) -> dict:
    from rag import init_rag, rag_pipeline, build_prompt_for_llm
    from llm import llm_generate
    from config import RAG_LLM_MAX_NEW_TOKENS, TOP_K_RERANK, MAX_CONTEXT_CHARS

    init_rag()
    data = load_benchmark(benchmark_path)
    if limit:
        data = data[:limit]

    f1_sum = 0.0
    sem_sum = 0.0
    n = 0
    total = len(data)

    for i, sample in enumerate(tqdm(data, desc="E2E eval")):
        q = (sample.get("question") or sample.get("query") or "").strip()
        ref = (sample.get("answer") or "").strip()
        if not q or not ref:
            continue
        pred = ""
        try:
            context, top_chunks, _ = rag_pipeline(
                q,
                top_k_retrieve=30,
                top_k_rerank=TOP_K_RERANK,
                max_context_chars=MAX_CONTEXT_CHARS,
                mode="eval",
            )
            prompt = build_prompt_for_llm(q, context, mode="eval")
            with ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(llm_generate, prompt, RAG_LLM_MAX_NEW_TOKENS)
                pred = (future.result(timeout=timeout_sec) or "").strip()
        except FuturesTimeoutError:
            tqdm.write(f"Sample {i + 1}/{total}: timeout ({timeout_sec}s), skipping")
        except Exception as e:
            tqdm.write(f"Sample {i + 1}/{total}: {e}")
        pred_norm = normalize_answer(pred)
        ref_norm = normalize_answer(ref)
        f1_sum += f1_token_overlap(pred_norm, ref_norm, normalize=False)
        try:
            sem_sum += semantic_similarity(pred, ref, normalize=True)
        except Exception:
            pass
        n += 1

    if n == 0:
        n = 1
    return {
        "n_queries": n,
        "metrics": {
            "SemanticSimilarity": round(sem_sum / n, 4),
            "F1_token": round(f1_sum / n, 4),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK, help="Path to benchmark.json (with 'answer' field)")
    parser.add_argument("--limit", type=int, default=None, help="Max number of samples (for quick test)")
    parser.add_argument("--timeout", type=float, default=180, help="Max seconds per LLM call (default 180)")
    parser.add_argument("--output", default=None, help="Save report JSON here")
    args = parser.parse_args()

    benchmark_path = args.benchmark
    if not os.path.isfile(benchmark_path):
        print("Benchmark not found. Run: python -m rag_benchmark.run_all", file=sys.stderr)
        sys.exit(1)

    report = evaluate(benchmark_path, limit=args.limit, timeout_sec=args.timeout)
    print("\n=== End-to-End RAG metrics ===\n")
    print(json.dumps(report["metrics"], indent=2))
    print(f"\nn_queries: {report['n_queries']}")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Report saved: {args.output}")


if __name__ == "__main__":
    main()
