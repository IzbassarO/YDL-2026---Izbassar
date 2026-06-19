"""
build_index.py — превращает corpus.jsonl в FAISS-индекс.

Шаги:
  1. Читаем документы.
  2. Режем на чанки (CHUNK_SIZE / CHUNK_OVERLAP).
  3. Получаем эмбеддинги через text-1024 (батчами).
  4. Нормализуем векторы и кладём в FAISS (IndexFlatIP = косинусная близость).
  5. Сохраняем index.faiss + meta.json (тексты чанков и их источники).

Запуск:  python build_index.py
"""
import json
import os
import time

import numpy as np
import requests
import faiss

import config


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start += size - overlap
    return [c for c in chunks if c]


def embed_batch(texts: list[str]) -> np.ndarray:
    """Эмбеддинги для списка текстов. Поддерживает батч или поэлементно."""
    vectors = []
    for t in texts:
        payload = {"model": config.EMB_MODEL, "input": t}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.EMB_KEY}",
        }
        for attempt in range(3):
            try:
                r = requests.post(
                    config.EMB_URL, json=payload, headers=headers,
                    timeout=config.REQUEST_TIMEOUT,
                )
                r.raise_for_status()
                data = r.json()
                vec = data["data"][0]["embedding"]
                vectors.append(vec)
                break
            except Exception as e:
                if attempt == 2:
                    raise RuntimeError(f"Эмбеддинг не получен: {e}")
                time.sleep(1.5 * (attempt + 1))
    arr = np.array(vectors, dtype="float32")
    return arr


def main():
    if not os.path.exists(config.CORPUS_PATH):
        raise SystemExit("Нет corpus.jsonl — сначала запусти: python scrape.py")

    docs = []
    with open(config.CORPUS_PATH, encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))

    # 1. Чанкинг
    meta = []  # параллельно векторам: {text, source, title}
    for d in docs:
        for ch in chunk_text(d["text"], config.CHUNK_SIZE, config.CHUNK_OVERLAP):
            meta.append({
                "text": ch,
                "source": d["source"],
                "title": d.get("title", ""),
            })
    print(f"Документов: {len(docs)} → чанков: {len(meta)}")

    # 2. Эмбеддинги
    print("Получаю эмбеддинги через text-1024…")
    texts = [m["text"] for m in meta]
    vecs = embed_batch(texts)
    print(f"Векторы: {vecs.shape}")

    if vecs.shape[1] != config.EMB_DIM:
        print(f"  ! Внимание: размерность {vecs.shape[1]} != EMB_DIM {config.EMB_DIM}. "
              f"Поправь EMB_DIM в config.py.")

    # 3. Нормализация → косинус через inner product
    faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    # 4. Сохранение
    faiss.write_index(index, config.INDEX_PATH)
    with open(config.META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)

    print(f"\nГотово:")
    print(f"  {config.INDEX_PATH}  ({index.ntotal} векторов)")
    print(f"  {config.META_PATH}")
    print("Следующий шаг: streamlit run app.py")


if __name__ == "__main__":
    main()
