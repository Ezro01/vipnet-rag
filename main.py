#!/usr/bin/env python3
"""
Точка входа: проверка данных → при необходимости препроцессинг (PDF + векторизация) → RAG + LLM.

  python main.py         — интерактивный чат (RAG + LLM)
  python main.py app     — FastAPI сервер
  python main.py eval    — оценка RAG (генерация QA → retrieval → [rerank для сравнения] → e2e)

Если нет dataset.json, embeddings.npy или embeddings_meta.json — сначала запускается
препроцессинг (PDF → чанки → эмбеддинги, FAISS, BM25), затем чат или app.
Оценка (eval): python -m rag_benchmark.run_all — без ручных эталонов.
"""

import os
import sys
import warnings

_project_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.join(_project_dir, "src")
if _project_dir not in sys.path:
    sys.path.insert(0, _project_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

warnings.filterwarnings("ignore")

os.environ.setdefault("OMP_NUM_THREADS", "10")
os.environ.setdefault("MKL_NUM_THREADS", "10")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "10")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def need_preprocessing():
    from config import DATASET_JSON, OUTPUT_EMBEDDINGS, OUTPUT_META, FAISS_INDEX, BM25_CORPUS_JSON
    required = (DATASET_JSON, OUTPUT_EMBEDDINGS, OUTPUT_META, FAISS_INDEX, BM25_CORPUS_JSON)
    return not all(os.path.isfile(p) for p in required)


def run_preprocessing():
    from config import (
        OUTPUT_JSON,
        DATASET_JSON,
        OUTPUT_EMBEDDINGS,
        OUTPUT_META,
        FAISS_INDEX,
        BM25_CORPUS_JSON,
        get_compute_device,
    )
    from pdf_to_json import main as run_pdf
    from vectorize_chunks import main as run_vectorize
    from vectorize_chunks import build_faiss_bm25_only

    print(f"Устройство: {get_compute_device()}")
    out_dir = os.path.dirname(OUTPUT_JSON) or "data"
    os.makedirs(out_dir, exist_ok=True)

    has_dataset = os.path.isfile(DATASET_JSON)
    has_embeddings = os.path.isfile(OUTPUT_EMBEDDINGS)
    has_meta = os.path.isfile(OUTPUT_META)
    has_faiss = os.path.isfile(FAISS_INDEX)
    has_bm25 = os.path.isfile(BM25_CORPUS_JSON)

    if not has_dataset:
        print("Нет dataset.json — запуск PDF → чанки...")
        run_pdf()
        has_dataset = True

    if not has_embeddings or not has_meta:
        print("Нет эмбеддингов или meta — запуск векторизации...")
        run_vectorize()
    elif not has_faiss or not has_bm25:
        print("Нет faiss.index или bm25_corpus.json — сборка из готовых эмбеддингов...")
        build_faiss_bm25_only()


def main():
    arg = (sys.argv[1] if len(sys.argv) > 1 else "").strip().lower()
    if arg == "app":
        # Сервер: препроцессинг только через API (/preprocess/parse, /preprocess/index)
        from app import run_app
        run_app()
        return
    if need_preprocessing():
        print("Не найдены данные для RAG — запуск препроцессинга (PDF → векторизация)...")
        run_preprocessing()
    else:
        print("Данные для RAG найдены — запуск без препроцессинга.")
        os.environ["HF_HUB_OFFLINE"] = "1"

    if arg == "eval":
        from rag_benchmark.run_all import main as run_benchmark
        # убираем "eval" из argv, чтобы argparse в run_all не получал лишний аргумент
        argv_save = sys.argv
        sys.argv = [argv_save[0]]
        try:
            run_benchmark()
        finally:
            sys.argv = argv_save
    else:
        from chat import run_chat
        run_chat()


if __name__ == "__main__":
    import multiprocessing as mp
    try:
        mp.set_start_method("fork", force=True)
    except RuntimeError:
        pass
    try:
        import torch
        from config import OMP_NUM_THREADS
        torch.set_num_threads(OMP_NUM_THREADS)
    except Exception:
        pass
    main()
