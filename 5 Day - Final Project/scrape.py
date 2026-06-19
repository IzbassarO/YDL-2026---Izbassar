"""
scrape.py — сбор данных с сайта фонда Есенова.

Стратегия:
  1. Обходим домен yessenovfoundation.org начиная со стартовых страниц,
     идём только по внутренним ссылкам, ограничены MAX_PAGES.
  2. Из каждой страницы вытаскиваем заголовок + читаемый текст.
  3. Дополнительно добавляем "процедурные карточки" вручную (PROCEDURE_CARDS) —
     это явные ответы на процедурные вопросы со ссылками
     ("где результаты", "как подать заявку"). RAG найдёт их так же,
     как и текст сайта, и отдаст пользователю со ссылкой.

Результат: data/corpus.jsonl — по одному документу на строку:
  {"id": ..., "source": url, "title": ..., "text": ...}

Запуск:  python scrape.py
"""
import json
import time
import re
from collections import deque
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

import config


HEADERS = {
    "User-Agent": "YDL2026-student-project/1.0 (educational RAG bot)"
}


def clean_text(html: str) -> tuple[str, str]:
    """Возвращает (title, plain_text), выкидывая мусор."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "svg"]):
        tag.decompose()
    title = (soup.title.string.strip() if soup.title and soup.title.string else "")
    # Собираем видимый текст блоками
    parts = []
    for el in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "th", "blockquote"]):
        t = el.get_text(" ", strip=True)
        if t and len(t) > 1:
            parts.append(t)
    text = "\n".join(parts)
    # Сжимаем лишние пробелы/переводы строк
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return title, text


def same_domain(url: str, base_netloc: str) -> bool:
    try:
        return urlparse(url).netloc in ("", base_netloc)
    except Exception:
        return False


def crawl() -> list[dict]:
    base_netloc = urlparse(config.BASE_SITE).netloc
    seen = set()
    queue = deque()
    for p in config.SEED_PATHS:
        queue.append(urljoin(config.BASE_SITE, p))

    docs = []
    while queue and len(docs) < config.MAX_PAGES:
        url = queue.popleft()
        url = url.split("#")[0].rstrip("/")
        if not url or url in seen:
            continue
        seen.add(url)
        try:
            r = requests.get(url, headers=HEADERS, timeout=config.REQUEST_TIMEOUT)
            if r.status_code != 200 or "text/html" not in r.headers.get("Content-Type", ""):
                continue
        except Exception as e:
            print(f"  ! пропуск {url}: {e}")
            continue

        title, text = clean_text(r.text)
        if len(text) > 200:  # отбрасываем пустышки
            docs.append({"source": url, "title": title, "text": text})
            print(f"  + [{len(docs)}] {url}  ({len(text)} симв.)")

        # собираем внутренние ссылки
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            nxt = urljoin(url, a["href"]).split("#")[0].rstrip("/")
            if same_domain(nxt, base_netloc) and nxt not in seen:
                # игнорируем файлы
                if not re.search(r"\.(pdf|jpg|jpeg|png|zip|docx?|xlsx?|mp4)$", nxt, re.I):
                    queue.append(nxt)

        time.sleep(config.CRAWL_DELAY)

    return docs


# --- Процедурные карточки: явные ответы со ссылками ---
# Отредактируй ссылки/тексты под реальные данные фонда.
# Именно через них реализован сценарий "где результаты / как получить доступ".
PROCEDURE_CARDS = [
    {
        "source": "manual://procedure/results",
        "title": "Где публикуются результаты по заявке",
        "text": (
            "Результаты рассмотрения заявок на гранты и стипендии фонда Есенова "
            "публикуются на официальном сайте фонда и дополнительно сообщаются "
            "заявителю по электронной почте, указанной в заявке. "
            "Проверить статус и итоги можно на странице: "
            "https://yessenovfoundation.org . "
            "Если результатов ещё нет — значит рассмотрение продолжается; "
            "точные сроки публикации указываются в условиях конкретной программы."
        ),
    },
    {
        "source": "manual://procedure/apply",
        "title": "Как подать заявку",
        "text": (
            "Заявка подаётся через официальный сайт фонда Есенова в разделе "
            "соответствующей программы. Перейдите по ссылке "
            "https://yessenovfoundation.org , выберите программу, "
            "ознакомьтесь с требованиями и заполните форму. "
            "После отправки заявки вы получите подтверждение на email."
        ),
    },
]


def main():
    print("Скрейпинг сайта фонда…")
    docs = crawl()
    print(f"Собрано страниц: {len(docs)}")

    docs.extend(PROCEDURE_CARDS)
    print(f"Добавлено процедурных карточек: {len(PROCEDURE_CARDS)}")

    import os
    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(config.CORPUS_PATH, "w", encoding="utf-8") as f:
        for i, d in enumerate(docs):
            d["id"] = i
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"\nГотово → {config.CORPUS_PATH} ({len(docs)} документов)")
    print("Следующий шаг: python build_index.py")


if __name__ == "__main__":
    main()
