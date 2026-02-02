"""Конфигурация из .env: пути, модели RAG/LLM, FAISS, API."""

import os
import platform

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")


def _env(key: str, default: str = "") -> str:
    return (os.environ.get(key) or default).strip()


# Пути
INPUT_DIR = _env("INPUT_DIR", "data/ViPNet Coordinator HW 5.3.2_docs")
DATA_DIR = _env("RAG_DATA_DIR", "data")
OUTPUT_JSON = _env("OUTPUT_JSON", os.path.join("data", "dataset.json"))
DATASET_JSON = _env("DATASET_JSON") or os.path.join(DATA_DIR, "dataset.json")
OUTPUT_EMBEDDINGS = _env("OUTPUT_EMBEDDINGS") or os.path.join(DATA_DIR, "embeddings.npy")
OUTPUT_META = _env("OUTPUT_META") or os.path.join(DATA_DIR, "embeddings_meta.json")
FAISS_INDEX = _env("FAISS_INDEX") or os.path.join(DATA_DIR, "faiss.index")
BM25_CORPUS_JSON = _env("BM25_CORPUS_JSON") or os.path.join(DATA_DIR, "bm25_corpus.json")

# RAG (алиасы для rag.py)
RAG_DATASET_JSON, RAG_EMBEDDINGS_NPY, RAG_EMBEDDINGS_META = DATASET_JSON, OUTPUT_EMBEDDINGS, OUTPUT_META

# Модели и устройства
RAG_RERANKER_MODEL = _env("RAG_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
RAG_USE_RERANKER = os.environ.get("RAG_USE_RERANKER", "1").strip().lower() in ("1", "true", "yes")
# Для Docker без сети используйте e5-base (он предзагружен в образ). e5-large нужен интернет или кэш HF
EMBED_MODEL = _env("EMBED_MODEL", "intfloat/multilingual-e5-large")
VECTORIZE_DEVICE = _env("VECTORIZE_DEVICE") or None
VECTORIZE_MODEL = _env("VECTORIZE_MODEL", "").lower()

# Yandex GPT
YANDEX_CLOUD_FOLDER = _env("YANDEX_CLOUD_FOLDER", "").strip()
YANDEX_CLOUD_API_KEY = _env("YANDEX_CLOUD_API_KEY", "").strip()
YANDEX_CLOUD_MODEL = _env("YANDEX_CLOUD_MODEL", "qwen2.5-7b-instruct/latest").strip()
YANDEX_CLOUD_BASE_URL = _env("YANDEX_CLOUD_BASE_URL", "https://rest-assistant.api.cloud.yandex.net/v1").strip()
RAG_LLM_MAX_NEW_TOKENS = int(_env("RAG_LLM_MAX_NEW_TOKENS", "2048"))

# FAISS, реранкер, потоки
FAISS_HNSW_M = int(_env("FAISS_HNSW_M", "32"))
FAISS_EF_CONSTRUCTION = int(_env("FAISS_EF_CONSTRUCTION", "200"))
FAISS_EF_SEARCH = int(_env("FAISS_EF_SEARCH", "64"))
RERANKER_BATCH_SIZE = int(_env("RERANKER_BATCH_SIZE", "16"))
OMP_NUM_THREADS = int(_env("OMP_NUM_THREADS", "10"))
PDF_N_WORKERS = int(_env("PDF_N_WORKERS", "0"))

# API
API_HOST = _env("API_HOST", "0.0.0.0")
API_PORT = int(_env("API_PORT", "8000"))
RAG_SYSTEM_PROMPT = _env("RAG_SYSTEM_PROMPT", "").strip()

# RAG: top-k, контекст, RRF, expansion, кэш, compression
TOP_K_VECTOR = int(os.environ.get("TOP_K_VECTOR", "50"))
TOP_K_BM25 = int(os.environ.get("TOP_K_BM25", "50"))
TOP_K_RERANK = int(os.environ.get("TOP_K_RERANK", "6"))
MAX_CONTEXT_CHARS = int(os.environ.get("MAX_CONTEXT_CHARS", "7000"))
RAG_RRF_K = int(os.environ.get("RAG_RRF_K", "60"))
RAG_QUERY_EXPANSION = os.environ.get("RAG_QUERY_EXPANSION", "1").strip().lower() in ("1", "true", "yes")
RAG_MULTI_QUERY_LLM = os.environ.get("RAG_MULTI_QUERY_LLM", "0").strip().lower() in ("1", "true", "yes")
RAG_EMBED_CACHE_SIZE = int(os.environ.get("RAG_EMBED_CACHE_SIZE", "512"))
RAG_RERANK_CACHE_SIZE = int(os.environ.get("RAG_RERANK_CACHE_SIZE", "512"))
RAG_COMPRESS_CONTEXT = os.environ.get("RAG_COMPRESS_CONTEXT", "0").strip().lower() in ("1", "true", "yes")

IS_DARWIN = platform.system() == "Darwin"

# Chunking (pdf_to_json)
CHUNK_TOKENS_BY_TYPE = {
    "cli": 120, "network_cli": 120, "table": 150, "instruction": 220,
    "theory": 380, "algorithm": 220, "config": 150, "network": 260,
    "diagram": 150, "toc": 100,
}
CHUNK_TOKENS_DEFAULT = 380


def get_compute_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if IS_DARWIN:
            return "cpu"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def get_embed_device() -> str:
    if VECTORIZE_DEVICE is not None:
        return VECTORIZE_DEVICE
    if IS_DARWIN:
        return "cpu"
    return get_compute_device()
