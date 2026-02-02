"""
Векторизация чанков: E5-base (CPU на macOS), FAISS HNSW, BM25 corpus.
Оптимизировано под MacBook: CPU embeddings (без MPS), batch=64, FAISS HNSW (AVX2 + OMP).
"""

import os
import json
import platform
import subprocess
import sys
import tempfile

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("OMP_NUM_THREADS", "10")
os.environ.setdefault("MKL_NUM_THREADS", "10")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "10")
if platform.system() == "Darwin":
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from config import (
    DATASET_JSON,
    OUTPUT_EMBEDDINGS,
    OUTPUT_META,
    FAISS_INDEX,
    BM25_CORPUS_JSON,
    VECTORIZE_DEVICE,
    VECTORIZE_MODEL,
    get_embed_device,
    EMBED_MODEL,
    FAISS_HNSW_M,
    FAISS_EF_CONSTRUCTION,
    FAISS_EF_SEARCH,
    OMP_NUM_THREADS,
    IS_DARWIN,
)

import numpy as np
from tqdm import tqdm

MODEL_LARGE = "intfloat/multilingual-e5-large"
MODEL_BASE = "intfloat/multilingual-e5-base"
MODEL_SMALL = "intfloat/multilingual-e5-small"
MAX_LENGTH = 512
# macOS/CPU: batch 64 для e5-base; CUDA можно больше
BATCH_SIZE = 64 if (IS_DARWIN or get_embed_device() == "cpu") else 32


def _get_model_name():
    """Выбор модели эмбеддингов.

    VECTORIZE_MODEL=small/large переопределяет всё.
    Если явно задан EMBED_MODEL в .env — используем его даже на macOS (для экспериментов с e5-large / bge-m3).
    Иначе по умолчанию: e5-base на macOS (скорость+качество), large на CUDA.
    """
    if VECTORIZE_MODEL == "small":
        return MODEL_SMALL
    if VECTORIZE_MODEL == "large":
        return MODEL_LARGE
    if os.environ.get("EMBED_MODEL"):
        return EMBED_MODEL
    # По умолчанию: e5-base на macOS, large на CUDA/CPU-серверах
    if IS_DARWIN:
        return MODEL_BASE
    return EMBED_MODEL if EMBED_MODEL else MODEL_LARGE


def _get_device():
    if VECTORIZE_DEVICE is not None:
        return VECTORIZE_DEVICE
    return get_embed_device()


def _run_with_sentence_transformers(prefixed: list, n: int, device_str: str, model_name: str):
    from sentence_transformers import SentenceTransformer
    print(f"   Загрузка модели {model_name}...")
    model = SentenceTransformer(model_name, device=device_str)
    dim = model.get_sentence_embedding_dimension()
    print("   Векторизация батчами...")
    embeddings = model.encode(
        prefixed,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings, dtype=np.float32), dim


def _run_with_transformers(prefixed: list, n: int, device, model_name: str):
    import torch
    from transformers import AutoTokenizer, AutoModel

    def mean_pool(last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    print(f"   Загрузка {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    dim = model.config.hidden_size
    embeddings = np.zeros((n, dim), dtype=np.float32)
    for start in tqdm(range(0, n, BATCH_SIZE), desc="Batches"):
        batch_texts = prefixed[start : start + BATCH_SIZE]
        batch_dict = tokenizer(
            batch_texts,
            max_length=MAX_LENGTH,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
        with torch.no_grad():
            out = model(**batch_dict)
            pooled = mean_pool(out.last_hidden_state, batch_dict["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings[start : start + len(batch_texts)] = pooled.cpu().numpy()
    return embeddings, dim


def _build_faiss_and_bm25(embeddings: np.ndarray, chunks: list):
    dim = embeddings.shape[1]
    # На macOS при OMP_NUM_THREADS=1 снижаем риск segfault
    omp = 1 if os.environ.get("OMP_NUM_THREADS") == "1" else OMP_NUM_THREADS
    try:
        import faiss
        faiss.omp_set_num_threads(omp)
        # HNSW: ~x5 быстрее поиск при сопоставимом качестве
        index = faiss.IndexHNSWFlat(dim, FAISS_HNSW_M, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = FAISS_EF_CONSTRUCTION
        index.hnsw.efSearch = FAISS_EF_SEARCH
        index.add(embeddings.astype(np.float32))
        faiss.write_index(index, FAISS_INDEX)
        print(f"   FAISS HNSW index → {FAISS_INDEX}")
    except Exception as e:
        print(f"   FAISS не сохранён: {e}")
        try:
            import faiss
            faiss.omp_set_num_threads(omp)
            index = faiss.IndexFlatIP(dim)
            index.add(embeddings.astype(np.float32))
            faiss.write_index(index, FAISS_INDEX)
            print(f"   FAISS FlatIP → {FAISS_INDEX}")
        except Exception as e2:
            print(f"   FAISS Flat fallback не сохранён: {e2}")

    try:
        from rank_bm25 import BM25Okapi
        tokenized = [c["text"].lower().split() for c in chunks]
        with open(BM25_CORPUS_JSON, "w", encoding="utf-8") as f:
            json.dump(tokenized, f, ensure_ascii=False)
        print(f"   BM25 corpus → {BM25_CORPUS_JSON}")
    except ImportError:
        print("   rank_bm25 не установлен — BM25 corpus не сохранён")
    except Exception as e:
        print(f"   BM25 corpus не сохранён: {e}")


def build_faiss_bm25_only():
    """Создаёт только faiss.index и bm25_corpus.json из существующих embeddings.npy и dataset.json."""
    # Снижает риск segfault на macOS при сборке FAISS (OMP/MKL)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    embeddings = np.load(OUTPUT_EMBEDDINGS).astype(np.float32)
    with open(DATASET_JSON, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    assert len(chunks) == embeddings.shape[0], "Число чанков и строк эмбеддингов не совпадает"
    print("Сборка FAISS и BM25 из готовых эмбеддингов...")
    _build_faiss_and_bm25(embeddings, chunks)
    print("✅ Готово: FAISS index и BM25 corpus")


def _run_encoding_worker(payload_path: str):
    """Worker: загрузка, кодирование, сохранение (для subprocess при необходимости)."""
    with open(payload_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    with open(payload["dataset_path"], "r", encoding="utf-8") as f:
        chunks = json.load(f)
    texts = [c["text"] for c in chunks]
    n = len(texts)
    prefixed = ["passage: " + t for t in texts]
    use_st = payload.get("use_sentence_transformers", False)
    device_str = payload["device_str"]
    model_name = payload["model_name"]

    if use_st:
        embeddings, dim = _run_with_sentence_transformers(prefixed, n, device_str, model_name)
    else:
        import torch
        embeddings, dim = _run_with_transformers(prefixed, n, torch.device(device_str), model_name)

    np.save(payload["output_emb_path"], embeddings)
    meta = {
        "chunk_ids": [c["chunk_id"] for c in chunks],
        "model": model_name,
        "dim": int(dim),
        "n_chunks": n,
    }
    with open(payload["output_meta_path"], "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    _build_faiss_and_bm25(embeddings, chunks)


def main():
    out_dir = os.path.dirname(DATASET_JSON) or os.path.dirname(OUTPUT_EMBEDDINGS)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(DATASET_JSON, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    n = len(chunks)
    texts = [c["text"] for c in chunks]
    prefixed = ["passage: " + t for t in texts]
    model_name = _get_model_name()
    print(f"Векторизация: модель {model_name}, чанков: {n}")

    use_st = False
    try:
        from sentence_transformers import SentenceTransformer
        use_st = True
    except ImportError:
        pass

    device_str = _get_device()
    print(f"   Устройство: {device_str} (embeddings на CPU на macOS — без segfault)")

    # На macOS: один процесс, CPU, batch 64 — стабильно и быстро
    if use_st:
        embeddings, dim = _run_with_sentence_transformers(prefixed, n, device_str, model_name)
    else:
        import torch
        embeddings, dim = _run_with_transformers(
            prefixed, n, torch.device(device_str), model_name
        )
    np.save(OUTPUT_EMBEDDINGS, embeddings)
    meta = {
        "chunk_ids": [c["chunk_id"] for c in chunks],
        "model": model_name,
        "dim": int(dim),
        "n_chunks": n,
    }
    with open(OUTPUT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    # На macOS FAISS с несколькими OMP-потоками часто зависает — перед сборкой индекса ставим 1
    if IS_DARWIN:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
    _build_faiss_and_bm25(embeddings, chunks)

    print("✅ Готово:", OUTPUT_EMBEDDINGS, OUTPUT_META)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        _run_encoding_worker(sys.argv[2])
    else:
        main()
