"""RAG: hybrid retrieval (FAISS + BM25, RRF) → [rerank] → optional compression → prompt. Query expansion, LRU cache."""

import json
import re
from functools import lru_cache

import numpy as np

from config import (
    RAG_DATASET_JSON as DATASET_JSON,
    RAG_EMBEDDINGS_NPY as EMBEDDINGS_NPY,
    RAG_EMBEDDINGS_META as EMBEDDINGS_META,
    FAISS_INDEX,
    BM25_CORPUS_JSON,
    RAG_USE_RERANKER,
    RAG_RERANKER_MODEL as RERANKER_MODEL,
    EMBED_MODEL,
    TOP_K_VECTOR,
    TOP_K_BM25,
    TOP_K_RERANK,
    RERANKER_BATCH_SIZE,
    FAISS_EF_SEARCH,
    OMP_NUM_THREADS,
    IS_DARWIN,
    RAG_SYSTEM_PROMPT as CONFIG_RAG_SYSTEM_PROMPT,
    MAX_CONTEXT_CHARS,
    RAG_RRF_K,
    RAG_QUERY_EXPANSION,
    RAG_MULTI_QUERY_LLM,
    RAG_EMBED_CACHE_SIZE,
    RAG_RERANK_CACHE_SIZE,
    RAG_COMPRESS_CONTEXT,
    get_compute_device,
    get_embed_device,
)

_chunks = None
_embeddings = None
_meta = None
_faiss_index = None
_bm25 = None
_bm25_corpus = None
_query_encoder = None
_reranker = None
_use_hybrid = None


def load_rag_data():
    global _chunks, _embeddings, _meta, _faiss_index, _bm25, _bm25_corpus, _use_hybrid
    if _chunks is not None:
        return _chunks, _embeddings, _meta
    with open(DATASET_JSON, "r", encoding="utf-8") as f:
        _chunks = json.load(f)
    _embeddings = np.load(EMBEDDINGS_NPY).astype(np.float32)
    with open(EMBEDDINGS_META, "r", encoding="utf-8") as f:
        _meta = json.load(f)
    assert len(_chunks) == len(_embeddings) == len(_meta["chunk_ids"])
    try:
        import faiss
        # На macOS FAISS с несколькими OMP-потоками часто зависает при загрузке/поиске
        faiss.omp_set_num_threads(1 if IS_DARWIN else OMP_NUM_THREADS)
        print("Загрузка индекса FAISS...", flush=True)
        _faiss_index = faiss.read_index(FAISS_INDEX)
        print("FAISS загружен.", flush=True)
        if hasattr(_faiss_index, "hnsw") and _faiss_index.hnsw is not None:
            _faiss_index.hnsw.efSearch = FAISS_EF_SEARCH
    except Exception:
        _faiss_index = None
    try:
        from rank_bm25 import BM25Okapi
        with open(BM25_CORPUS_JSON, "r", encoding="utf-8") as f:
            _bm25_corpus = json.load(f)
        _bm25 = BM25Okapi(_bm25_corpus)
        _use_hybrid = True
    except Exception:
        _bm25 = None
        _bm25_corpus = None
        _use_hybrid = False
    return _chunks, _embeddings, _meta


def _expand_query_llm(query: str) -> list:
    q = query.strip()
    if not q:
        return [q]
    try:
        from llm import llm_generate
        prompt = (
            "По техническому вопросу к документации ViPNet Coordinator HW сгенерируй 2 варианта перефразирования "
            "(тот же смысл, другие слова; можно сленг или практический сценарий). "
            "Ответь строго двумя строками, по одному варианту на строку, без нумерации.\nВопрос: " + q
        )
        raw = llm_generate(prompt, max_new_tokens=256)
        lines = [ln.strip() for ln in (raw or "").strip().split("\n") if ln.strip() and len(ln.strip()) > 10]
        variants = [ln.lstrip("012.-) ").strip() for ln in lines[:2] if ln]
        if variants:
            return [q] + variants
    except Exception:
        pass
    return [q]


def expand_query(query: str) -> list:
    q = (query or "").strip()
    if not q:
        return [q]
    if RAG_MULTI_QUERY_LLM:
        return _expand_query_llm(query)
    if not RAG_QUERY_EXPANSION:
        return [q]
    return [q, f"инструкция {q}", f"настройка {q}", f"пример {q}"]


def _get_query_encoder():
    global _query_encoder
    load_rag_data()
    if _query_encoder is not None:
        return _query_encoder
    from sentence_transformers import SentenceTransformer
    model_name = _meta.get("model", EMBED_MODEL)
    _query_encoder = SentenceTransformer(model_name, device=get_embed_device())
    return _query_encoder


def _encode_query_impl(query: str) -> np.ndarray:
    model = _get_query_encoder()
    prefixed = "query: " + query.strip()
    vec = model.encode([prefixed], normalize_embeddings=True)
    return np.asarray(vec[0], dtype=np.float32)


@lru_cache(maxsize=RAG_EMBED_CACHE_SIZE if RAG_EMBED_CACHE_SIZE > 0 else 128)
def _encode_query_cached(query: str) -> tuple:
    return tuple(_encode_query_impl(query).tolist())


def encode_query(query: str, use_cache: bool = True) -> np.ndarray:
    if use_cache and RAG_EMBED_CACHE_SIZE > 0:
        return np.array(_encode_query_cached(query), dtype=np.float32)
    return _encode_query_impl(query)


def _retrieve_single_query(query: str, top_k: int) -> tuple:
    chunks, embeddings, meta = load_rag_data()
    q = encode_query(query).reshape(1, -1).astype(np.float32)
    n = len(chunks)
    vec_ids = []
    if _faiss_index is not None:
        D, I = _faiss_index.search(q, top_k)
        for i in I[0]:
            if 0 <= i < n:
                vec_ids.append(int(i))
    else:
        scores = (embeddings @ q.T).ravel()
        idx = np.argsort(scores)[::-1][:top_k]
        vec_ids = [int(i) for i in idx]
    bm25_ids = []
    if _use_hybrid and _bm25 is not None:
        q_tokens = query.lower().split()
        bm25_scores = _bm25.get_scores(q_tokens)
        bm25_top = np.argsort(bm25_scores)[-TOP_K_BM25:][::-1]
        bm25_ids = [int(i) for i in bm25_top if 0 <= i < n]
    return vec_ids, bm25_ids


def _rrf_fusion(ranked_lists: list, k: int = RAG_RRF_K) -> list:
    scores = {}
    for ranked_ids in ranked_lists:
        for rank, doc_id in enumerate(ranked_ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def retrieve_top_k(query: str, top_k: int = TOP_K_VECTOR):
    queries = expand_query(query)
    all_vec_ids, all_bm25_ids = [], []
    for q in queries:
        vec_ids, bm25_ids = _retrieve_single_query(q, top_k)
        all_vec_ids.append(vec_ids)
        all_bm25_ids.append(bm25_ids)
    vec_fused = _rrf_fusion(all_vec_ids, k=RAG_RRF_K)
    bm25_fused = _rrf_fusion(all_bm25_ids, k=RAG_RRF_K) if _use_hybrid and _bm25 and any(all_bm25_ids) else []
    if bm25_fused:
        merged = _rrf_fusion([[i for i, _ in vec_fused], [i for i, _ in bm25_fused]], k=RAG_RRF_K)
    else:
        merged = vec_fused
    chunks, _, _ = load_rag_data()
    results = []
    for doc_id, score in merged[:top_k]:
        c = chunks[doc_id].copy()
        c["retrieve_score"] = c["score"] = score
        results.append(c)
    return results


def _get_reranker():
    global _reranker
    if _reranker is not None:
        return _reranker
    from sentence_transformers import CrossEncoder
    _reranker = CrossEncoder(RERANKER_MODEL, max_length=512, device=get_compute_device())
    return _reranker


def _rerank_impl(query: str, candidates: list, top_k: int) -> list:
    if not candidates or (len(candidates) <= top_k and not (candidates and candidates[0].get("text"))):
        return candidates[:top_k]
    model = _get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = np.asarray(model.predict(pairs, batch_size=RERANKER_BATCH_SIZE))
    idx = np.argsort(scores)[::-1][:top_k]
    return [dict(candidates[i], score=float(scores[i]), rerank_score=float(scores[i])) for i in idx]


@lru_cache(maxsize=RAG_RERANK_CACHE_SIZE if RAG_RERANK_CACHE_SIZE > 0 else 128)
def _rerank_cached(query: str, chunk_ids_tuple: tuple, top_k: int) -> tuple:
    load_rag_data()
    chunk_map = {c["chunk_id"]: c for c in _chunks}
    candidates = [chunk_map[cid] for cid in chunk_ids_tuple if cid in chunk_map]
    result = _rerank_impl(query, candidates, top_k)
    return tuple((r["chunk_id"], r.get("score", 0.0)) for r in result)


def rerank(query: str, candidates: list, top_k: int = TOP_K_RERANK, use_cache: bool = True) -> list:
    if not candidates:
        return []
    chunk_ids_tuple = tuple(sorted(c["chunk_id"] for c in candidates))
    if use_cache and RAG_RERANK_CACHE_SIZE > 0 and len(chunk_ids_tuple) <= 500:
        cached = _rerank_cached(query, chunk_ids_tuple, top_k)
        load_rag_data()
        chunk_map = {c["chunk_id"]: c for c in _chunks}
        return [dict(chunk_map[rid], score=sc, rerank_score=sc) for rid, sc in cached if rid in chunk_map]
    return _rerank_impl(query, candidates, top_k)


def init_rag():
    load_rag_data()
    _get_query_encoder()
    if RAG_USE_RERANKER:
        _get_reranker()


def reset_rag():
    """Сброс кэша RAG в памяти — следующий load_rag_data() загрузит данные заново."""
    global _chunks, _embeddings, _meta, _faiss_index, _bm25, _bm25_corpus, _query_encoder, _reranker
    _chunks = None
    _embeddings = None
    _meta = None
    _faiss_index = None
    _bm25 = None
    _bm25_corpus = None
    _query_encoder = None
    _reranker = None


def compress_context(chunks: list, query: str, max_chars_per_chunk: int = 1500) -> list:
    if not RAG_COMPRESS_CONTEXT or not chunks or not (query or "").strip():
        return chunks
    q_tokens = set(re.findall(r"\w+", (query or "").lower()))
    if not q_tokens:
        return chunks
    result = []
    for c in chunks:
        text = c.get("text", "")
        sentences = re.split(r"(?<=[.!?])\s+", text)
        kept, total = [], 0
        for s in sentences:
            s_lower = s.lower()
            if any(t in s_lower for t in q_tokens) or total < 200:
                kept.append(s)
                total += len(s)
                if total >= max_chars_per_chunk:
                    break
        if not kept:
            kept = [text[:max_chars_per_chunk]] if text else []
        r = dict(c, text=" ".join(kept).strip() or text[:max_chars_per_chunk])
        result.append(r)
    return result


def _context_block_header(r: dict) -> str:
    doc = r.get("doc_name", r.get("doc", ""))
    section = r.get("section", "")
    subsection = r.get("subsection", "")
    section_id = r.get("section_id", "")
    chapter = r.get("chapter_title", "")
    parts = [doc]
    if chapter:
        parts.append(chapter)
    parts.append(section)
    if subsection and subsection != section:
        parts.append(subsection)
    parts.append(section_id)
    return "[" + " | ".join(p for p in parts if p) + "]"


def format_context(search_results: list, max_chars: int = 8000, separator: str = "\n\n---\n\n") -> str:
    parts, total = [], 0
    for r in search_results:
        block = f"{_context_block_header(r)}\n{r.get('text', '')}"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return separator.join(parts)


CHAT_PROMPT = (
    "Ты — эксперт по документации ViPNet Coordinator HW (многофункциональный шлюз безопасности). "
    "Используй предоставленный контекст как основной источник. "
    "Если информации в контексте недостаточно — логично обобщи, не противореча контексту. "
    "Если ответа в контексте нет, напиши: «В предоставленных материалах этой информации нет». "
    "Отвечай в формате markdown. Обязательно используй отдельные строки и читаемую структуру: "
    "основные разделы оформляй заголовками уровня `###`, вложенные — `####`; шаги действий пиши "
    "нумерованными списками (1., 2., 3.), параметры и варианты — маркированными списками (- элемент). "
    "Код, CLI‑команды и конфигурации оформляй в моноширинных блоках с тройными обратными кавычками. "
)

STRICT_QA_PROMPT = (
    "Ты — система точного извлечения информации.\n"
    "Используй ТОЛЬКО данный контекст.\n"
    "Ответь ТОЛЬКО фразой из контекста.\n"
    "Ничего не переформулируй.\n"
    "Ничего не добавляй.\n"
    "Если точного ответа нет — напиши: NOT FOUND."
)


def build_prompt_for_llm(query: str, context: str, system_prompt: str = None, mode: str = "chat") -> str:
    if mode == "eval":
        return f"{STRICT_QA_PROMPT}\n\nКонтекст:\n{context}\n\nВопрос: {query}\n\nОтвет:"
    system = system_prompt or (CONFIG_RAG_SYSTEM_PROMPT if CONFIG_RAG_SYSTEM_PROMPT else None) or CHAT_PROMPT
    return f"{system}\n\nКонтекст из документации:\n{context}\n\nВопрос: {query}\n\nОтвет:"


def rag_pipeline(query: str, top_k_retrieve: int = TOP_K_VECTOR, top_k_rerank: int = TOP_K_RERANK, max_context_chars: int = None, mode: str = "chat"):
    if max_context_chars is None:
        max_context_chars = MAX_CONTEXT_CHARS
    candidates = retrieve_top_k(query, top_k=top_k_retrieve)
    top_chunks = rerank(query, candidates, top_k=top_k_rerank) if RAG_USE_RERANKER else candidates[:top_k_rerank]
    top_chunks = compress_context(top_chunks, query)
    context = format_context(top_chunks, max_chars=max_context_chars)
    prompt = build_prompt_for_llm(query, context, mode=mode)
    return context, top_chunks, prompt


def main():
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Как настроить VPN-туннель?"
    print(f"RAG: hybrid retrieve → {'rerank → ' if RAG_USE_RERANKER else ''}LLM\n")
    print(f"Запрос: {q}\n")
    context, top_chunks, prompt = rag_pipeline(q)
    print("Топ чанков:")
    for i, r in enumerate(top_chunks, 1):
        print(f"  {i}. [{r.get('doc_name', '')} | {r.get('section', '')}]")
        print((r["text"][:350] + "...") if len(r["text"]) > 350 else r["text"])
    print("\nПромпт (первые 1200 символов):")
    print(prompt[:1200] + "..." if len(prompt) > 1200 else prompt)


if __name__ == "__main__":
    main()
