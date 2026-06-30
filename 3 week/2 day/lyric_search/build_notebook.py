"""Assemble and execute the lyric semantic-search story notebook (English)."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

nb = new_notebook()
c = []

c.append(new_markdown_cell(
"""# Lyric Semantic Search — searching songs by meaning, not just words

**YDL 2026 · Applied NLP · Day 2 Lab Project**

We take ~57k English songs, split every song's lyrics into single lines, and build **three** search
indexes over those lines:

| Method | What it compares | Technique | Strong at |
|--------|------------------|-----------|-----------|
| **tf-idf** | matching **words** | tf-idf + cosine | exact words |
| **GloVe** | the **meaning** of a line | averaged embeddings + cosine | synonyms, mood |
| **trigram** | **3-character chunks** (lon·ond·ndo·don) | char-3gram + cosine | typos, fuzzy |

> No heavy models (no BERT/Gemma) — only Day-2 tools: tf-idf, averaged word embeddings,
> cosine similarity, and character n-grams.

**The core idea:** the query *"feeling heartbroken and alone"* matched by keywords only finds lines that
literally contain "feeling"; matched by meaning it finds genuinely sad lines like *"Feeling sad and all
alone"* — even with no shared words.

**Note on the corpus:** we search the **lyric text itself** (one line per document). The artist and song
title are only labels showing where each line came from — we do **not** search by title or author."""))

c.append(new_markdown_cell("## 0. Load the indexes (tf-idf + GloVe + trigram)"))
c.append(new_code_cell(
"""import search   # search_tfidf / search_glove / search_trigram / compare
search._load()
print(f"lines in corpus:   {len(search._lines):,}")
print(f"tf-idf vocab:      {len(search._tfidf.vocabulary_):,} words")
print(f"GloVe matrix:      {search._gmat.shape[0]:,} lines x {search._gmat.shape[1]}d")
print(f"trigram vocab:     {len(search._tri.vocabulary_):,} character 3-grams")"""))

c.append(new_markdown_cell(
"""## 1. Three modes on the same query

The queries are written in **our own words** — no line contains them verbatim.
`compare()` runs all three methods side by side."""))
c.append(new_code_cell('search.compare("feeling heartbroken and alone")'))
c.append(new_code_cell('search.compare("a cold and lonely winter night")'))

c.append(new_markdown_cell(
"""### What about typos?

This is where trigrams shine. The misspelled query **"heartbrokn and lonley"**:
tf-idf finds nothing (no exact words), GloVe breaks (the misspelled words aren't in its vocabulary),
but **trigram still finds** the right lines through shared 3-character chunks."""))
c.append(new_code_cell('search.compare("heartbrokn and lonley")   # intentional typos'))

c.append(new_markdown_cell(
"""**Summary — each method wins in its own regime, so they complement each other:**
- **tf-idf** — exact words (names, places), but blind to synonyms and typos;
- **GloVe** — meaning and mood (sad ≈ heartbroken ≈ lonely), but unaware of typos;
- **trigram** — robust to typos and fuzzy spelling, but understands no meaning.

That is why the artifact ships a **mode switcher**, not one single "correct" search."""))

c.append(new_markdown_cell(
"""## 2. Interactive search (3-mode switcher)

A standalone search page (`artifacts/search.html`) with **Meaning / Keywords / Fuzzy** buttons —
switch the mode and compare. Everything runs in the browser (JS): GloVe averaging + cosine,
tf-idf over words, and Dice over character trigrams. One file, no server, no heavy models.

> Open `artifacts/search.html` in a browser, or use it right here:"""))
c.append(new_code_cell(
'''from IPython.display import IFrame
IFrame("artifacts/search.html", width="100%", height=640)'''))

nb["cells"] = c
nb.metadata["kernelspec"] = {"name": "anaconda3", "display_name": "Python (anaconda3)", "language": "python"}

from nbclient import NotebookClient
NotebookClient(nb, timeout=600, kernel_name="anaconda3",
               resources={"metadata": {"path": "."}}).execute()
with open("lyric_search_demo.ipynb", "w") as f:
    nbf.write(nb, f)
print("saved -> lyric_search_demo.ipynb (executed, English)")
