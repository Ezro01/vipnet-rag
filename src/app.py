"""FastAPI: предобработка (parse, index) и /ask — RAG + LLM по документации."""

import os

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel


def _sources_from_chunks(top_chunks: list) -> list:
    """Список источников из top_chunks для ответа API: doc, chapter, section_id (без section)."""
    return [
        {
            "doc": c.get("doc_name", c.get("doc", "")),
            "chapter": c.get("chapter_title", ""),
            "section_id": c.get("section_id", ""),
        }
        for c in top_chunks
    ]


def _preprocessing_required():
    from config import DATASET_JSON, OUTPUT_EMBEDDINGS, OUTPUT_META, FAISS_INDEX, BM25_CORPUS_JSON
    return {
        "dataset": os.path.isfile(DATASET_JSON),
        "embeddings": os.path.isfile(OUTPUT_EMBEDDINGS),
        "meta": os.path.isfile(OUTPUT_META),
        "faiss": os.path.isfile(FAISS_INDEX),
        "bm25": os.path.isfile(BM25_CORPUS_JSON),
    }


def _on_startup():
    status = _preprocessing_required()
    if not all(status.values()):
        return
    try:
        from rag import init_rag
        init_rag()
    except Exception as e:
        print("RAG init warning:", e)


app = FastAPI(
    title="ViPNet Coordinator RAG API",
    description="Предобработка (парсинг PDF, индексы) и вопрос-ответ по документации (RAG + LLM)",
    on_startup=[_on_startup],
)


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    sources: list[dict]


class ParseRequest(BaseModel):
    input_dir: str | None = None


class ParseResponse(BaseModel):
    ok: bool
    message: str
    chunks_count: int | None = None


class IndexResponse(BaseModel):
    ok: bool
    message: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/preprocess/status")
def preprocess_status():
    """Какие файлы предобработки уже есть (dataset, embeddings, faiss, bm25)."""
    status = _preprocessing_required()
    ready = all(status.values())
    return {"ready": ready, "files": status}


@app.post("/preprocess/parse", response_model=ParseResponse)
def preprocess_parse(req: ParseRequest | None = Body(None)):
    """
    Парсинг PDF из папки в чанки (dataset.json).
    Передайте папку с документацией в body: {"input_dir": "data/docs"}.
    Если input_dir не передан — используется INPUT_DIR из конфига.
    """
    import json
    from config import DATASET_JSON, DATA_DIR
    from pdf_to_json import main as run_pdf

    input_dir = (req.input_dir if req and req.input_dir else None) or os.environ.get("INPUT_DIR") or os.path.join(DATA_DIR, "docs")
    input_dir = (input_dir or "").strip()
    if not input_dir:
        input_dir = os.path.join(DATA_DIR, "docs")
    if not os.path.isabs(input_dir):
        input_dir = os.path.abspath(input_dir)
    if not os.path.isdir(input_dir):
        raise HTTPException(status_code=400, detail=f"Папка не найдена: {input_dir}")

    out_path = DATASET_JSON
    out_dir = os.path.dirname(out_path) or DATA_DIR
    os.makedirs(out_dir, exist_ok=True)
    try:
        run_pdf(input_dir=input_dir, output_json=out_path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    with open(out_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return ParseResponse(ok=True, message=f"Сохранено в {out_path}", chunks_count=len(chunks))


@app.post("/preprocess/index", response_model=IndexResponse)
def preprocess_index():
    """
    Построение индексов из dataset.json: эмбеддинги, FAISS, BM25.
    Вызывать после /preprocess/parse. После успеха RAG подхватит данные при следующем /ask
    или после вызова POST /preprocess/reload.
    """
    from config import DATASET_JSON, OUTPUT_EMBEDDINGS, OUTPUT_META, FAISS_INDEX, BM25_CORPUS_JSON
    from vectorize_chunks import main as run_vectorize
    from vectorize_chunks import build_faiss_bm25_only

    if not os.path.isfile(DATASET_JSON):
        raise HTTPException(status_code=400, detail="Сначала выполните POST /preprocess/parse (нужен dataset.json)")

    if os.path.isfile(OUTPUT_EMBEDDINGS) and os.path.isfile(OUTPUT_META):
        if not os.path.isfile(FAISS_INDEX) or not os.path.isfile(BM25_CORPUS_JSON):
            try:
                build_faiss_bm25_only()
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        return IndexResponse(ok=True, message="FAISS и BM25 собраны из готовых эмбеддингов")
    try:
        run_vectorize()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return IndexResponse(ok=True, message="Эмбеддинги, FAISS и BM25 построены")


@app.post("/preprocess/reload")
def preprocess_reload():
    """Сбросить кэш RAG в памяти — следующий /ask загрузит данные заново."""
    try:
        from rag import reset_rag
        reset_rag()
        return {"ok": True, "message": "Кэш RAG сброшен"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    question = (req.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="question is empty")

    status = _preprocessing_required()
    if not all(status.values()):
        raise HTTPException(
            status_code=503,
            detail="Данные не готовы. Выполните POST /preprocess/parse и POST /preprocess/index",
        )

    try:
        from rag import rag_pipeline, init_rag
        from llm import llm_generate
        init_rag()
    except ImportError as e:
        raise HTTPException(status_code=503, detail=f"RAG/LLM not available: {e}")

    try:
        from format_answer import format_pretty_answer
        _, top_chunks, prompt = rag_pipeline(question, top_k_retrieve=30, top_k_rerank=5, mode="chat")
        answer = llm_generate(prompt)
        answer = format_pretty_answer(answer or "")
        return AskResponse(answer=answer, sources=_sources_from_chunks(top_chunks))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def run_app():
    from config import API_HOST, API_PORT
    import uvicorn
    uvicorn.run(app, host=API_HOST, port=API_PORT)
