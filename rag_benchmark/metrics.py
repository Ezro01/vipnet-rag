"""
Метрики качества RAG: retrieval, reranking, end-to-end.
Answer normalization для стабильного сравнения (EM, F1, ROUGE-L).
"""

import re
import numpy as np


def normalize_answer(s: str) -> str:
    """Нормализация ответа перед сравнением (как у ревьюера): lowercase, убрать лишние символы и пробелы."""
    if not s or not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s./:-]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def normalize(s: str) -> str:
    """Алиас для normalize_answer (совместимость с планом ревьюера)."""
    return normalize_answer(s)


def recall_at_k(ranked_ids: list, gold_ids: set, k: int) -> float:
    """Recall@k: 1 если хотя бы один релевантный в top-k, иначе 0."""
    if not gold_ids or k <= 0:
        return 0.0
    top = ranked_ids[:k]
    return 1.0 if any(i in gold_ids for i in top) else 0.0


def mrr_at_k(ranked_ids: list, gold_ids: set, k: int) -> float:
    """MRR@k: 1/rank первого релевантного в top-k, иначе 0."""
    if not gold_ids or k <= 0:
        return 0.0
    for idx, i in enumerate(ranked_ids[:k]):
        if i in gold_ids:
            return 1.0 / (idx + 1)
    return 0.0


def ndcg_at_k(ranked_ids: list, gold_ids: set, k: int) -> float:
    """nDCG@k с бинарной релевантностью (0/1)."""
    if not gold_ids or k <= 0:
        return 0.0
    dcg = 0.0
    for i, idx in enumerate(ranked_ids[:k]):
        if idx in gold_ids:
            dcg += 1.0 / np.log2(i + 2)
    n = min(len(gold_ids), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(n))
    if idcg <= 0:
        return 0.0
    return dcg / idcg


def precision_at_k(ranked_ids: list, gold_ids: set, k: int) -> float:
    """Precision@k: доля релевантных среди top-k."""
    if k <= 0:
        return 0.0
    top = ranked_ids[:k]
    hits = sum(1 for i in top if i in gold_ids)
    return hits / len(top)


def map_at_k(ranked_ids: list, gold_ids: set, k: int) -> float:
    """Average Precision@k (для одного запроса)."""
    if not gold_ids or k <= 0:
        return 0.0
    prec_sum = 0.0
    hits = 0
    for i, idx in enumerate(ranked_ids[:k]):
        if idx in gold_ids:
            hits += 1
            prec_sum += hits / (i + 1)
    return prec_sum / len(gold_ids) if gold_ids else 0.0


# --- End-to-End: сравнение ответа с эталоном ---

def _rouge_l_score(pred: str, ref: str) -> float:
    """ROUGE-L F1 (упрощённо по longest common subsequence)."""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        s = scorer.score(ref, pred)
        return s["rougeL"].fmeasure
    except Exception:
        return 0.0


def _bleu_score(pred: str, ref: str) -> float:
    """BLEU (sentence_bleu, 4-gram)."""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        ref_tokens = ref.split()
        pred_tokens = pred.split()
        if not ref_tokens:
            return 0.0
        sm = SmoothingFunction()
        return float(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=sm.method1))
    except Exception:
        return 0.0


def exact_match(pred: str, ref: str, normalize: bool = True) -> float:
    """Exact Match: 1.0 если нормализованные строки совпадают."""
    if normalize:
        pred = normalize_answer(pred)
        ref = normalize_answer(ref)
    return 1.0 if pred == ref else 0.0


def f1_token_overlap(pred: str, ref: str, normalize: bool = True) -> float:
    """F1 по токенам (пересечение слов / precision & recall)."""
    if normalize:
        pred = normalize_answer(pred)
        ref = normalize_answer(ref)
    pred_tok = set(pred.split())
    ref_tok = set(ref.split())
    if not ref_tok:
        return 1.0 if not pred_tok else 0.0
    if not pred_tok:
        return 0.0
    common = len(pred_tok & ref_tok)
    prec = common / len(pred_tok)
    rec = common / len(ref_tok)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def rouge_l(pred: str, ref: str, normalize: bool = True) -> float:
    """ROUGE-L F1 (требует pip install rouge-score)."""
    if normalize:
        pred = normalize_answer(pred)
        ref = normalize_answer(ref)
    return _rouge_l_score(pred, ref)


def bleu(pred: str, ref: str) -> float:
    """BLEU (не для QA; оставлен для совместимости)."""
    return _bleu_score(pred, ref)


# --- Semantic similarity (рекомендация ревьюера: смысл, а не строки) ---
_similarity_encoder = None


def _get_similarity_encoder():
    global _similarity_encoder
    if _similarity_encoder is not None:
        return _similarity_encoder
    try:
        import os
        import sys
        _root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from config import EMBED_MODEL, get_embed_device
        from sentence_transformers import SentenceTransformer
        _similarity_encoder = SentenceTransformer(EMBED_MODEL, device=get_embed_device())
        return _similarity_encoder
    except Exception:
        return None


def semantic_similarity(pred: str, ref: str, normalize: bool = True) -> float:
    """Косинусная близость эмбеддингов (0..1). Для RAG QA: один факт — разные формулировки."""
    if not pred or not ref:
        return 0.0
    if normalize:
        pred = normalize_answer(pred)
        ref = normalize_answer(ref)
    if not pred or not ref:
        return 0.0
    enc = _get_similarity_encoder()
    if enc is None:
        return 0.0
    prefixed = ["passage: " + pred, "passage: " + ref]
    vecs = enc.encode(prefixed, normalize_embeddings=True)
    sim = float(np.dot(vecs[0], vecs[1]))
    return max(0.0, min(1.0, sim))
