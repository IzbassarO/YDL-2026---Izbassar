"""
Центральная конфигурация проекта.

Секреты (ключи API) НЕ хранятся в этом файле — они читаются из .env или из
переменных окружения. См. .env.example. Файл .env в .gitignore — в репозиторий
не попадёт.
"""
import os


def _load_dotenv():
    """Лёгкий загрузчик .env без внешних зависимостей.
    Не перезаписывает уже выставленные переменные окружения."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key, val = key.strip(), val.strip().strip('"').strip("'")
            os.environ.setdefault(key, val)


_load_dotenv()


def _require(name: str) -> str:
    """Возвращает секрет из окружения или падает с понятной ошибкой."""
    val = os.getenv(name)
    if not val:
        raise RuntimeError(
            f"Не задан секрет {name}. Скопируй .env.example в .env и впиши ключи "
            f"(cp .env.example .env), либо выставь переменную окружения {name}."
        )
    return val


# --- LLM (chat) ---
CHAT_URL = os.getenv("CHAT_URL", "https://llm.alem.ai/v1/chat/completions")
CHAT_KEY = _require("CHAT_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemma4")

# --- Embeddings ---
EMB_URL = os.getenv("EMB_URL", "https://llm.alem.ai/v1/embeddings")
EMB_KEY = _require("EMB_KEY")
EMB_MODEL = os.getenv("EMB_MODEL", "text-1024")
EMB_DIM = 1024  # text-1024

# --- Email (MailerSend) ---
MAILERSEND_KEY = _require("MAILERSEND_KEY")
FROM_EMAIL = "info@app.commit.kz"
FROM_NAME = "Yessenov Data Lab"
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "izok2004@gmail.com")  # шлём только себе

# --- Пути ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CORPUS_PATH = os.path.join(DATA_DIR, "corpus.jsonl")
INDEX_PATH = os.path.join(DATA_DIR, "index.faiss")
META_PATH = os.path.join(DATA_DIR, "meta.json")

# --- Скрейпинг ---
BASE_SITE = "https://yessenovfoundation.org"
# Стартовые страницы; скрейпер пойдёт по внутренним ссылкам в пределах домена
SEED_PATHS = ["/", "/ru", "/en"]
MAX_PAGES = 40          # потолок страниц, чтобы не уйти в бесконечность
CRAWL_DELAY = 0.5       # вежливая пауза между запросами, сек

# --- Чанкинг ---
CHUNK_SIZE = 700        # символов в чанке
CHUNK_OVERLAP = 120     # перекрытие, чтобы не рвать смысл

# --- RAG ---
TOP_K = 4               # сколько чанков подаём в контекст
# Порог косинусной близости. Если лучший чанк ниже — бот говорит "не знаю".
# Калибруется после первого запуска (см. README).
RELEVANCE_THRESHOLD = 0.35

# --- Лимиты чатбота ---
MAX_MESSAGES_PER_SESSION = 25   # сообщений пользователя за сессию
MIN_SECONDS_BETWEEN = 3.0       # анти-спам пауза между запросами
REQUEST_TIMEOUT = 60            # таймаут HTTP, сек
