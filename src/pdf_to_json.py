"""
PDF → JSON для RAG: извлечение по блокам, TOC-сегментация, семантический chunking.
"""

import os
import re
import json
import fitz
from tqdm import tqdm

from config import (
    INPUT_DIR,
    OUTPUT_JSON,
    PDF_N_WORKERS,
    CHUNK_TOKENS_BY_TYPE,
    CHUNK_TOKENS_DEFAULT,
)

PAGE_SEP = "\n\n"
MAX_CHUNK_TOKENS = 220
MAX_CHUNK_CHARS = 220 * 4
MIN_CHUNK_CHARS = 40  # фильтр мусора: чанки короче 40 символов отбрасываем

DOC_TYPE_MAP = {
    "подготовка к работе": "preparation",
    "настройка в cli": "cli_config",
    "настройка в web": "web_config",
    "справочник": "reference",
    "история версий": "version_history",
    "перечень совместимых трансиверов": "transceivers",
}

TOC_ENTRY_RE = re.compile(
    r"(.+?)\s*\.{2,}\s*(\d{1,3})\s*(?=\s|\n|$)",
    re.MULTILINE,
)


def parse_doc_metadata(pdf_path: str) -> dict:
    basename = os.path.basename(pdf_path)
    name_without_ext = basename[:-4] if basename.lower().endswith(".pdf") else basename
    m = re.match(r"^(\d+)\s+(.+)", name_without_ext)
    doc_index = m.group(1) if m else "00"
    rest = (m.group(2) or "").strip()
    doc_type = "other"
    for key, value in DOC_TYPE_MAP.items():
        if key in rest.lower():
            doc_type = value
            break
    return {
        "doc_index": doc_index,
        "doc_type": doc_type,
        "doc_title_ru": rest,
        "product": "ViPNet Coordinator HW 5",
        "doc_name": basename,
    }


def extract_pages(path: str) -> list:
    """Извлечение текста по блокам (сохраняет таблицы и структуру)."""
    doc = fitz.open(path)
    pages = []
    for page in doc:
        try:
            blocks = page.get_text("blocks")
            if blocks:
                text = "\n\n".join(
                    b[4].strip() for b in blocks if len(b) > 4 and (b[4] or "").strip()
                )
            else:
                text = page.get_text()
        except Exception:
            text = page.get_text()
        if len(text.strip()) > 50:
            pages.append(text)
    doc.close()
    return pages


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"Стр\.\s*\d+(?:\s*из\s*\d+)?", "", text, flags=re.IGNORECASE)
    return text.strip()


def extract_toc_items(full_text: str) -> list:
    toc_start = full_text.find("Содержание")
    if toc_start == -1:
        toc_start = full_text.find("Оглавление")
    if toc_start == -1:
        return []
    search_zone = full_text[toc_start : toc_start + 6000]
    items, seen = [], set()
    skip = {"содержание", "оглавление"}
    for m in TOC_ENTRY_RE.finditer(search_zone):
        title = re.sub(r"\s+", " ", m.group(1).strip())
        if len(title) < 2 or title in seen or title.lower() in skip or re.match(r"^\d+$", title):
            continue
        seen.add(title)
        items.append(title)
    return items


def find_section_start(text: str, title: str, start_from: int) -> int:
    if start_from >= len(text):
        return -1
    pattern = r"(?:^|\n)\s*" + re.escape(title) + r"(?:\s|\n|$)"
    match = re.search(pattern, text[start_from:], re.IGNORECASE)
    if match:
        return start_from + match.start()
    pos = text.find(title, start_from)
    return pos if pos != -1 else -1


def split_by_toc_sections(full_text: str, toc_items: list) -> list:
    if not toc_items:
        return [("0", "Основной текст", full_text)]
    toc_start = full_text.find("Содержание") or full_text.find("Оглавление") or 0
    toc_end = toc_start + min(6000, len(full_text) - toc_start)
    positions = []
    for title in toc_items:
        start = toc_end if not positions else positions[-1][0] + 10
        pos = find_section_start(full_text, title, start)
        if pos != -1:
            positions.append((pos, title))
    if not positions:
        return [("0", "Основной текст", full_text)]
    positions.sort(key=lambda x: x[0])
    blocks = []
    for i, (start, title) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(full_text)
        content = full_text[start:end].strip()
        if len(content) >= 15:
            blocks.append((str(i + 1), title, content))
    return blocks


def classify_chunk(text: str) -> str:
    t = text.lower()
    if re.search(r"(^|\n)\s*[>#]\s*\w+", text):
        return "cli"
    if re.search(r"ping|traceroute|ip\s+addr|ifconfig|nmcli", t):
        return "network_cli"
    if re.search(r"\bшаг\s*\d+|выполните|следует|нажмите", t):
        return "algorithm"
    if re.search(r"настрой|установ|сконфигур", t):
        return "instruction"
    if re.search(r"ip\s*адрес|порт|маршрут|шлюз|dns|vpn|nat", t):
        return "network"
    if re.search(r"пример конфигурации|config|параметр|значение", t):
        return "config"
    if re.search(r"\bcli\b|команда|введите|shell|console", t):
        return "cli"
    if "|" in text or "таблица" in t:
        return "table"
    if "рис." in t or "схем" in t:
        return "diagram"
    if re.search(r"содержание|оглавление|введение\s*\.{2,}", t) or (
        len(t) < 800 and re.search(r"\.{3,}\s*\d+\s*$", text)
    ):
        return "toc"
    return "theory"


_pdf_tokenizer = None


def get_pdf_tokenizer():
    global _pdf_tokenizer
    if _pdf_tokenizer is not None:
        return _pdf_tokenizer
    try:
        from transformers import AutoTokenizer
        _pdf_tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
    except Exception:
        pass
    return _pdf_tokenizer


def count_tokens(text: str) -> int:
    global _pdf_tokenizer
    if _pdf_tokenizer is None:
        _pdf_tokenizer = get_pdf_tokenizer()
    if _pdf_tokenizer is not None:
        ids = _pdf_tokenizer.encode(
            text, add_special_tokens=False, truncation=True, max_length=MAX_CHUNK_TOKENS
        )
        return len(ids)
    return min(len(text) // 4, MAX_CHUNK_TOKENS)


def _split_by_char_limit(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list:
    if len(text) <= max_chars:
        return [text] if text.strip() else []
    parts, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            for sep in ".!?\n ":
                last = text.rfind(sep, start, end + 1)
                if last > start:
                    end = last + (1 if sep != " " else 0)
                    break
        parts.append(text[start:end].strip())
        start = end
    return [p for p in parts if p]


def _token_count_approx(text: str) -> int:
    tok = get_pdf_tokenizer()
    if tok is not None:
        return len(tok.encode(text, add_special_tokens=False, truncation=True, max_length=4096))
    return len(text) // 4


def _structural_cli_blocks(text: str) -> list:
    """1 команда + пояснение = 1 chunk. Разрез по $ # >."""
    blocks = re.split(r"(?:^|\n)(?:\$|#|>)\s*", text)
    out = [b.strip() for b in blocks if len(b.strip()) > 30]
    return out if len(out) > 1 else []


def _structural_steps(text: str) -> list:
    """1 шаг = 1 chunk. Разрез по «Шаг N» или «N)»."""
    steps = re.split(r"\n\s*(?:Шаг\s*\d+\.?|[0-9]+\))\s*", text, flags=re.IGNORECASE)
    out = [s.strip() for s in steps if len(s.strip()) > 50]
    return out if len(out) > 2 else []


def _split_table_rows(text: str) -> list:
    """Таблица: 1 строка = 1 chunk."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return []
    out = []
    buf = []
    for ln in lines:
        if len(ln) >= MIN_CHUNK_CHARS:
            if buf:
                out.append("\n".join(buf))
                buf = []
            out.append(ln)
        else:
            buf.append(ln)
    if buf:
        merged = "\n".join(buf)
        if len(merged) >= MIN_CHUNK_CHARS:
            out.append(merged)
    return out


def _split_config_params(text: str) -> list:
    """Конфиг: 1 параметр = 1 chunk (по строкам или блокам)."""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip() and len(ln.strip()) >= 20]
    return lines if lines else [text] if len(text.strip()) >= MIN_CHUNK_CHARS else []


def _split_paragraphs_to_chunks(text: str, max_tokens: int) -> list:
    """
    Семантический split по абзацам: заголовок + описание + список.
    Один чанк = одна законченная мысль раздела, а не произвольный набор предложений.
    """
    max_chars = min(MAX_CHUNK_CHARS, max_tokens * 4)
    # Абзацы: разделяем по пустым строкам или двойным переносам
    paras = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if not paras:
        return []
    chunks = []
    buf = []
    buf_text = ""

    for p in paras:
        candidate = (buf_text + "\n\n" + p) if buf_text else p
        n_tok = _token_count_approx(candidate)
        if n_tok <= max_tokens and len(candidate) <= max_chars:
            buf.append(p)
            buf_text = candidate
            continue

        # Текущий буфер уже достаточно большой — сохраняем его как чанк
        if buf_text.strip() and len(buf_text.strip()) >= MIN_CHUNK_CHARS:
            chunks.append(buf_text.strip())

        # Если абзац сам по себе слишком большой — режем по символам
        if _token_count_approx(p) > max_tokens or len(p) > max_chars:
            chunks.extend(_split_by_char_limit(p, max_chars))
            buf, buf_text = [], ""
        else:
            buf = [p]
            buf_text = p

    if buf_text.strip() and len(buf_text.strip()) >= MIN_CHUNK_CHARS:
        chunks.append(buf_text.strip())
    return chunks


def smart_chunk(text: str, max_tokens: int = MAX_CHUNK_TOKENS) -> list:
    """Структурный split: CLI блоки → шаги → абзацы. Один факт = один chunk."""
    text = (text or "").strip()
    if not text:
        return []
    max_chars = min(MAX_CHUNK_CHARS, max_tokens * 4)

    # 1) CLI-блоки: по $ # >
    cli_blocks = _structural_cli_blocks(text)
    if len(cli_blocks) > 1:
        return [b for b in cli_blocks if len(b) >= MIN_CHUNK_CHARS]

    # 2) Пошаговые алгоритмы
    steps = _structural_steps(text)
    if len(steps) > 2:
        return [s for s in steps if len(s) >= MIN_CHUNK_CHARS]

    # 3) По умолчанию: по абзацам с лимитом токенов
    return _split_paragraphs_to_chunks(text, max_tokens)


def split_long_chunk(text: str, max_tokens: int = MAX_CHUNK_TOKENS) -> list:
    """Длинный текст: по строкам/абзацам с лимитом токенов."""
    max_chars = min(MAX_CHUNK_CHARS, max_tokens * 4)
    if len(text) <= max_chars:
        return [text] if text.strip() and len(text.strip()) >= MIN_CHUNK_CHARS else []
    lines, chunks, buf, buf_len = text.split("\n"), [], [], 0
    for line in lines:
        line_len = len(line) + 1
        if buf_len + line_len > max_chars and buf:
            chunk = "\n".join(buf)
            chunks.extend(
                _split_by_char_limit(chunk, max_chars)
                if len(chunk) > max_chars
                else [chunk]
            )
            buf, buf_len = [], 0
        if len(line) > max_chars:
            if buf:
                chunks.append("\n".join(buf))
                buf, buf_len = [], 0
            chunks.extend(_split_by_char_limit(line, max_chars))
        else:
            buf.append(line)
            buf_len += line_len
    if buf:
        chunk = "\n".join(buf)
        chunks.extend(
            _split_by_char_limit(chunk, max_chars) if len(chunk) > max_chars else [chunk]
        )
    return [c for c in chunks if len(c.strip()) >= MIN_CHUNK_CHARS]


def _get_chapter_title_map(blocks: list) -> dict:
    """Строит иерархию: chapter_id → chapter_title (первый раздел в главе = название главы)."""
    chapter_title = {}
    for section_id, title, _ in blocks:
        chapter_id = section_id.split(".")[0] if "." in section_id else section_id
        if chapter_id not in chapter_title:
            chapter_title[chapter_id] = title
    return chapter_title


def process_document(pdf_path: str) -> list:
    """PDF → layout extraction → structural segmentation → semantic split → micro-chunks (100–220 tok)."""
    meta = parse_doc_metadata(pdf_path)
    pages = extract_pages(pdf_path)
    if not pages:
        return []
    full_text = PAGE_SEP.join(clean_text(p) for p in pages)
    toc_items = extract_toc_items(full_text)
    blocks = split_by_toc_sections(full_text, toc_items)
    chapter_title_map = _get_chapter_title_map(blocks)
    records, chunk_counter = [], 0
    for section_id, title, content in blocks:
        chapter_id = section_id.split(".")[0] if "." in section_id else section_id
        subsection = section_id if "." in section_id and section_id != chapter_id else ""
        chapter_title = chapter_title_map.get(chapter_id, title)
        ctype = classify_chunk(content)
        max_tokens = CHUNK_TOKENS_BY_TYPE.get(ctype, CHUNK_TOKENS_DEFAULT)
        section_path = f"{section_id} {title}".strip()

        # Типо-зависимая сегментация: 1 факт = 1 chunk
        if ctype == "table":
            sub_chunks = _split_table_rows(content)
            if not sub_chunks:
                sub_chunks = split_long_chunk(content, max_tokens)
        elif ctype == "config":
            sub_chunks = _split_config_params(content)
            if not sub_chunks:
                sub_chunks = split_long_chunk(content, max_tokens)
        elif ctype in ("cli", "network_cli"):
            cli_parts = _structural_cli_blocks(content)
            sub_chunks = [b for b in cli_parts if len(b) >= MIN_CHUNK_CHARS] if cli_parts else []
            if not sub_chunks:
                sub_chunks = smart_chunk(content, max_tokens)
        elif ctype in ("instruction", "algorithm"):
            step_parts = _structural_steps(content)
            sub_chunks = [s for s in step_parts if len(s) >= MIN_CHUNK_CHARS] if step_parts else []
            if not sub_chunks:
                sub_chunks = smart_chunk(content, max_tokens)
        else:
            sub_chunks = smart_chunk(content, max_tokens)
            if not sub_chunks:
                sub_chunks = split_long_chunk(content, max_tokens)

        for chunk in sub_chunks:
            chunk = chunk.strip()
            if len(chunk) < MIN_CHUNK_CHARS:
                continue
            chunk_counter += 1
            safe_section = section_id.replace(".", "_")
            chunk_id = f"{meta['doc_index']}_{safe_section}_{chunk_counter:04d}"
            records.append({
                "chunk_id": chunk_id,
                "doc_name": meta["doc_name"],
                "doc_index": meta["doc_index"],
                "doc_type": meta["doc_type"],
                "doc_title_ru": meta["doc_title_ru"],
                "product": meta["product"],
                "chapter_id": chapter_id,
                "chapter_title": chapter_title,
                "section": title,
                "subsection": subsection or title,
                "section_id": section_id,
                "section_path": section_path,
                "type": ctype,
                "text": chunk,
                "tokens": count_tokens(chunk),
            })
    return records


def main(input_dir=None, output_json=None):
    """Парсинг PDF из папки в JSON-чанки. По умолчанию — INPUT_DIR и OUTPUT_JSON из config."""
    in_dir = (input_dir or INPUT_DIR).strip()
    out_path = (output_json or OUTPUT_JSON).strip()
    if not os.path.isabs(in_dir):
        in_dir = os.path.abspath(in_dir)
    if not os.path.isabs(out_path):
        out_path = os.path.abspath(out_path)
    out_dir = os.path.dirname(out_path) or "data"
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.isdir(in_dir):
        raise FileNotFoundError(f"Папка не найдена: {in_dir}")
    pdfs = sorted(f for f in os.listdir(in_dir) if f.lower().endswith(".pdf"))
    paths = [os.path.join(in_dir, pdf) for pdf in pdfs if os.path.isfile(os.path.join(in_dir, pdf))]
    print(f"PDF: найдено {len(paths)} файлов в {in_dir}")
    all_chunks = []
    if PDF_N_WORKERS and PDF_N_WORKERS > 1:
        from multiprocessing import Pool
        with Pool(PDF_N_WORKERS) as pool:
            for records in tqdm(
                pool.imap(process_document, paths),
                total=len(paths),
                desc="PDF",
            ):
                all_chunks.extend(records)
    else:
        for path in tqdm(paths, desc="PDF"):
            all_chunks.extend(process_document(path))
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"PDF: сохранено {len(all_chunks)} чанков → {out_path}")


if __name__ == "__main__":
    main()
