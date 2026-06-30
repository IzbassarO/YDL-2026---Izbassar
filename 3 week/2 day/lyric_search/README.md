# 🔍 Lyric Semantic Search — поиск песен по смыслу, словам и опечаткам

**YDL 2026 · Applied NLP · Day 2 Lab Project**

~57k англоязычных песен → строки → **три** поисковых индекса. Вводишь запрос своими словами,
сравниваешь, как ищут разные методы. Без тяжёлых моделей (BERT/Gemma) — только техники Day 2.

## Три режима (переключатель в артефакте)
| Режим | Что сравнивает | Техника | Силён в |
|-------|----------------|---------|---------|
| **Meaning** | смысл строки | **idf-взвешенные** GloVe-эмбеддинги + cosine | синонимы, настроение |
| **Keywords** | совпадение слов | tf-idf + cosine | точные слова, имена |
| **Fuzzy** | 3-символьные куски (`lon·ond·ndo·don`) | char-trigram + cosine / Dice | опечатки |

### Улучшения логики (не только дизайн)
- **idf-взвешенное усреднение (SIF-трюк):** каждое слово в «векторе смысла» весит по своему idf —
  частые `the/and/I` почти не считаются, смысловые слова доминируют. Убирает stopword-мусор в выдаче.
- **Разнообразие выдачи:** дубли строк убираются, не более 2 строк из одной песни в топе.
- Веса idf согласованы между Python (`search.py`) и браузером (`search.html`).

**Killer-демо:** запрос с опечатками `heartbrokn and lonley` — tf-idf не находит ничего,
GloVe ломается, а **trigram** спокойно находит «Heartbroken when you left my world».

## Файлы
| Файл | Что делает |
|------|-----------|
| `preprocess.py` | песни → чистые строки (`data/lines.parquet`) |
| `build_index.py` | tf-idf индекс + усреднённые GloVe-эмбеддинги строк |
| `build_trigram.py` | char-trigram tf-idf индекс (fuzzy) |
| `search.py` | `search_tfidf / search_glove / search_trigram / compare` |
| `export_web.py` | self-contained HTML с переключателем 3 режимов |
| `lyric_search_demo.ipynb` | **главный артефакт**: история end-to-end с живыми выводами |

## Артефакты
- `artifacts/search.html` — интерактивный поисковик (3 режима, один файл, работает в браузере)
- `data/song2vec*`/индексы — `lines.parquet`, `tfidf.pkl`, `glove_lines.npy`, `trigram.pkl`

## Воспроизвести
```bash
# окружение: conda base (Python 3.13) + gensim, sklearn, pandas, plotly
python preprocess.py     # -> data/lines.parquet
python build_index.py    # -> tf-idf + GloVe индексы (качает GloVe ~128MB один раз)
python build_trigram.py  # -> char-trigram индекс
python export_web.py     # -> artifacts/search.html
python search.py         # быстрый тест в терминале (compare на примерах)
```

## Идея проекта
> Один корпус — три взгляда: по **смыслу** (эмбеддинги), по **словам** (tf-idf) и по **буквам/опечаткам**
> (триграммы). Каждый метод выигрывает в своём регионе запросов — поэтому переключатель, а не один поиск.

## Задел под iOS (на будущее)
`search.html` уже самодостаточный (данные + поиск внутри файла). Ту же логику
(усреднить эмбеддинги → косинус; или Dice по триграммам) легко повторить в приложении на устройстве.
