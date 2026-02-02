# RAG API (FastAPI) по документации ViPNet Coordinator HW
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Предзагрузка моделей Hugging Face в образ — без этого в офлайне/без сети будет ошибка
# Используйте EMBED_MODEL=intfloat/multilingual-e5-base в .env (по умолчанию)
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('intfloat/multilingual-e5-large'); \
from sentence_transformers import CrossEncoder; \
CrossEncoder('BAAI/bge-reranker-v2-m3'); \
"

COPY . .

ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=4
# Для офлайн: задайте HF_HUB_OFFLINE=1 и используйте EMBED_MODEL=intfloat/multilingual-e5-large (он уже в образе)
EXPOSE 8000

CMD ["python", "main.py", "app"]
