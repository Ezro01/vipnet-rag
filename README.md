# ViPNet Coordinator RAG — вопрос-ответ по документации

RAG-система (Retrieval-Augmented Generation) для работы с документацией ViPNet Coordinator HW 5: парсинг PDF в чанки, гибридный поиск (эмбеддинги + BM25), реранкинг и генерация ответов через Yandex GPT / Qwen.

---

## Содержание

- [Возможности](#возможности)
- [Требования](#требования)
- [Установка и настройка](#установка-и-настройка)
- [Запуск](#запуск)
- [API (FastAPI)](#api-fastapi)
- [Docker](#docker)
- [Структура проекта](#структура-проекта)
- [Конфигурация](#конфигурация)
- [Оценка качества (RAG Benchmark)](#оценка-качества-rag-benchmark)
- [Частые вопросы](#частые-вопросы)

---

## Возможности

- **Парсинг PDF** — извлечение текста по блокам, сегментация по оглавлению, семантическое разбиение на чанки (параграфы, CLI-блоки, таблицы и т.д.).
- **Гибридный поиск** — FAISS (эмбеддинги E5) + BM25, слияние рангов через RRF; опционально расширение запроса (перефразирования через LLM).
- **Реранкинг** — CrossEncoder (BAAI/bge-reranker-v2-m3) для отбора топ-кандидатов.
- **Генерация ответов** — Yandex Cloud REST API (Yandex GPT / Qwen); контекст из найденных чанков + системный промпт.
- **Форматирование ответов** — автоматическое приведение ответа LLM к читаемому виду: заголовки, списки, код-блоки, удаление артефактов PDF.
- **Три режима использования:**
  - **Интерактивный чат в терминале** — `python main.py`.
  - **HTTP API (FastAPI)** — `python main.py app`; отдельные эндпоинты для парсинга, построения индексов и вопросов.
  - **Оценка качества** — `python main.py eval` (генерация QA, retrieval, end-to-end метрики).
- **Docker** — образ с предзагруженными моделями эмбеддингов и реранкера; данные в volume, предобработка через API.

---

## Требования

- **Python 3.11** (рекомендуется).
- Доступ в интернет для первого скачивания моделей Hugging Face (либо использование предзагруженного Docker-образа).
- **Yandex Cloud**: API-ключ и Folder ID для LLM (Yandex GPT / Qwen). Модели эмбеддингов и реранкера — из Hugging Face (Sentence Transformers).

Поддерживаются: CPU, CUDA, macOS (MPS не используется для эмбеддингов — только CPU для стабильности).

---

## Установка и настройка

### 1. Клонирование и виртуальное окружение

```bash
cd /path/to/project
python3.11 -m venv venv
source venv/bin/activate   # Linux/macOS
# или: venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 2. Переменные окружения

Скопируйте шаблон и заполните обязательные поля:

```bash
cp .env.example .env
```

В `.env` обязательно задайте:

- **YANDEX_CLOUD_FOLDER** — идентификатор каталога в Yandex Cloud.
- **YANDEX_CLOUD_API_KEY** — API-ключ для доступа к Yandex GPT.

Остальные параметры (пути, модели, порт API) можно оставить по умолчанию или задать при необходимости (см. [Конфигурация](#конфигурация)).

### 3. Подготовка данных (локально, без API)

Если вы запускаете **чат** (`python main.py`) или **eval** (`python main.py eval`) и ещё не делали предобработку:

- Положите PDF документации в папку, указанную в **INPUT_DIR** (по умолчанию `data/ViPNet Coordinator HW 5.3.2_docs`).
- При первом запуске `main.py` автоматически выполнится: парсинг PDF → dataset.json → векторизация → FAISS + BM25. При следующих запусках готовые файлы будут подхватываться из `data/`.

Если запускаете только **API** (`python main.py app`), предобработка не стартует автоматически — её нужно выполнить через эндпоинты (см. [API](#api-fastapi)).

---

## Запуск

### Интерактивный чат в терминале

```bash
python main.py
```

После загрузки RAG введите вопрос; пустая строка — выход. Ответ выводится по словам (стриминг от LLM), затем список источников (doc, chapter, section_id).

### FastAPI-сервер

```bash
python main.py app
```

По умолчанию сервер слушает `http://0.0.0.0:8000`. Документация: `http://localhost:8000/docs` (Swagger UI).

### Оценка качества RAG

```bash
python main.py eval
# или
python -m rag_benchmark.run_all
```

Подробнее см. [RAG Benchmark](#оценка-качества-rag-benchmark).

---

## API (FastAPI)

### Обзор эндпоинтов

| Метод | Путь | Описание |
|--------|------|----------|
| GET | `/health` | Проверка доступности сервиса. |
| GET | `/preprocess/status` | Наличие файлов предобработки: dataset, embeddings, faiss, bm25; поле `ready` — все ли готовы для `/ask`. |
| POST | `/preprocess/parse` | Парсинг PDF из папки в чанки (dataset.json). Тело: `{"input_dir": "data/docs"}` или пустое (используется INPUT_DIR из конфига). |
| POST | `/preprocess/index` | Построение индексов из dataset.json: эмбеддинги, FAISS, BM25. Вызывать после `/preprocess/parse`. |
| POST | `/preprocess/reload` | Сброс кэша RAG в памяти; следующий `/ask` загрузит данные заново. |
| POST | `/ask` | Вопрос к RAG: контекст из документации + ответ LLM. Требует готовых данных (после parse + index). |

### Типичный сценарий

1. **Проверить статус:**  
   `GET /preprocess/status` → смотреть `ready` и `files`.

2. **Загрузить PDF в папку** (например `data/docs`) и распарсить:  
   `POST /preprocess/parse` с телом `{"input_dir": "data/docs"}` (или без тела, если в конфиге уже указана нужная папка).

3. **Построить индексы:**  
   `POST /preprocess/index`.

4. **При необходимости обновить кэш:**  
   `POST /preprocess/reload` (если данные пересобрали при уже запущенном сервере).

5. **Задавать вопросы:**  
   `POST /ask` с телом `{"question": "Как настроить DNS-сервер?"}`.

### Формат ответа `/ask`

```json
{
  "answer": "Текст ответа с форматированием (заголовки, списки, код).",
  "sources": [
    {
      "doc": "02 ViPNet Coordinator HW 5. Настройка в CLI.pdf",
      "chapter": "Настройка агрегированных сетевых интерфейсов",
      "section_id": "22"
    }
  ]
}
```

Ответ приводится к читаемому виду (заголовки, списки, код-блоки); в `sources` — только `doc`, `chapter`, `section_id`.

---

## Docker

### Сборка и запуск

```bash
cp .env.example .env
# Заполните YANDEX_CLOUD_FOLDER и YANDEX_CLOUD_API_KEY в .env

docker compose up --build
```

Сервис будет доступен на порту **8000** (или на порту из `API_PORT` в `.env`).

### Данные и предобработка в Docker

- Каталог **./data** монтируется в контейнер как **/app/data**. Все артефакты (dataset.json, embeddings.npy, faiss.index, bm25_corpus.json) сохраняются на хосте — при перезапуске контейнера предобработка заново не запускается, если файлы уже есть.
- Предобработка в режиме API выполняется только по запросам: **POST /preprocess/parse** и **POST /preprocess/index**.
- Чтобы использовать свою папку с PDF:
  - Смонтируйте её, например: `./docs:/app/data/docs` (раскомментируйте соответствующий volume в `docker-compose.yml`).
  - Вызовите `POST /preprocess/parse` с телом `{"input_dir": "data/docs"}`.
  - Затем `POST /preprocess/index`.

### Модели в образе

В образ при сборке предзагружаются модели Hugging Face: **intfloat/multilingual-e5-base** (или e5-large, см. Dockerfile) и **BAAI/bge-reranker-v2-m3**. Для работы без интернета в `.env` укажите ту же модель эмбеддингов, что загружена в образ (по умолчанию — e5-base в комментариях; в текущем Dockerfile может быть e5-large — сверьте с `ENV`/`RUN` в Dockerfile).

---

## Структура проекта

```
.
├── main.py              # Точка входа: чат / app / eval
├── config.py            # Конфигурация из .env
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── src/
│   ├── app.py           # FastAPI: предобработка + /ask
│   ├── chat.py          # Интерактивный чат в терминале
│   ├── rag.py           # RAG: загрузка данных, retrieval, rerank, промпт
│   ├── llm.py           # Yandex GPT / Qwen (REST API)
│   ├── pdf_to_json.py   # PDF → чанки (dataset.json)
│   ├── vectorize_chunks.py  # Эмбеддинги, FAISS, BM25
│   └── format_answer.py # Форматирование ответа LLM для вывода
├── rag_benchmark/       # Оценка качества RAG
│   ├── run_all.py       # Генерация QA + retrieval + e2e
│   ├── generate_queries.py
│   ├── evaluate_retrieval.py
│   ├── evaluate_end2end.py
│   └── metrics.py
└── data/                # Данные (монтируется в Docker)
    ├── dataset.json     # Чанки после парсинга PDF
    ├── embeddings.npy
    ├── embeddings_meta.json
    ├── faiss.index
    ├── bm25_corpus.json
    └── ViPNet Coordinator HW 5.3.2_docs/  # PDF по умолчанию
```

---

## Конфигурация

Основные переменные (файл `.env` или окружение):

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| **RAG_DATA_DIR** | Каталог данных (dataset, эмбеддинги, FAISS, BM25) | `data` |
| **INPUT_DIR** | Папка с PDF для парсинга | `data/ViPNet Coordinator HW 5.3.2_docs` |
| **DATASET_JSON** | Путь к dataset.json | `data/dataset.json` |
| **OUTPUT_EMBEDDINGS** | Файл эмбеддингов | `data/embeddings.npy` |
| **FAISS_INDEX** | Индекс FAISS | `data/faiss.index` |
| **BM25_CORPUS_JSON** | Корпус для BM25 | `data/bm25_corpus.json` |
| **EMBED_MODEL** | Модель эмбеддингов (Hugging Face) | `intfloat/multilingual-e5-base` (рекомендуется для офлайна в Docker) |
| **RAG_RERANKER_MODEL** | Модель реранкера | `BAAI/bge-reranker-v2-m3` |
| **YANDEX_CLOUD_FOLDER** | Каталог Yandex Cloud | — |
| **YANDEX_CLOUD_API_KEY** | API-ключ Yandex Cloud | — |
| **YANDEX_CLOUD_MODEL** | Модель LLM (Yandex GPT / Qwen) | `qwen2.5-7b-instruct/latest` |
| **API_HOST** | Хост FastAPI | `0.0.0.0` |
| **API_PORT** | Порт FastAPI | `8000` |

Дополнительные параметры (топ-k, RRF, кэш, чанкинг) задаются в `config.py` и при необходимости через переменные окружения (см. комментарии в коде и в `.env.example`).

---

## Оценка качества (RAG Benchmark)

Бенчмарк строит тестовые вопросы из чанков (Literal, Paraphrase, Scenario), оценивает retrieval (Recall@k, MRR@k, nDCG@k) и end-to-end ответы (SemanticSimilarity, F1_token и др.). Ручных эталонов нет — всё генерируется при запуске.

### Запуск

```bash
python main.py eval
# или с параметрами:
python -m rag_benchmark.run_all --samples 30 --e2e-limit 20 --eval-mode all --output report.json
```

- **--samples** — число чанков для генерации QA.
- **--e2e-limit** — лимит примеров для e2e.
- **--eval-mode** — `all` (разбивка по типам запросов), `literal` или `semantic`.
- **--skip-e2e** — только retrieval, без e2e.
- **--output** — путь к файлу отчёта.

Подробнее см. `rag_benchmark/README.md`.

---

## Частые вопросы

**Ошибка загрузки модели с Hugging Face (intfloat/multilingual-e5-large и т.п.)**  
- Убедитесь, что есть доступ в интернет при первом запуске, либо используйте модель, предзагруженную в Docker-образ (например, e5-base). В `.env` для офлайна укажите ту же модель, что в образе.

**Второй запрос к RAG «висит»**  
- Раньше это могло быть из-за незакрытого HTTP-клиента к Yandex API. В текущей версии клиент закрывается после каждого запроса; при повторении проблемы проверьте сеть и лимиты API.

**После построения индексов через API старые данные в памяти**  
- Вызовите **POST /preprocess/reload**, затем снова **POST /ask** — RAG перечитает данные с диска.

**macOS: зависание при загрузке или сборке FAISS**  
- В коде для macOS при работе с FAISS выставляется один поток (OMP_NUM_THREADS=1), чтобы избежать зависаний. Если проблема остаётся — проверьте версию faiss-cpu и наличие файлов индекса.

**Источники в ответе**  
- В API в `sources` возвращаются только поля `doc`, `chapter`, `section_id`. В терминальном чате вывод такой же: doc | chapter | section_id, без section.

---

## Лицензия и контакты

Проект предназначен для работы с документацией ViPNet Coordinator HW. Использование Yandex Cloud API и моделей Hugging Face регламентируется их условиями использования.
