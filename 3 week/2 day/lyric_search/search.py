"""Semantic lyric search: query -> nearest lyric lines, two ways.

tf-idf  : keyword search  (cosine over bag-of-words)
GloVe   : meaning search  (cosine over averaged word embeddings)
"""
import re
import json
import pickle
import numpy as np
import pandas as pd
import gensim.downloader as api
from sklearn.metrics.pairwise import cosine_similarity

_lines = _tfidf = _tmat = _gmat = _wv = None
_tri = _trimat = None
_widf = _wdefault = None
TOK = re.compile(r"[a-z']+")


def _load():
    global _lines, _tfidf, _tmat, _gmat, _wv, _tri, _trimat, _widf, _wdefault
    if _lines is not None:
        return
    _lines = pd.read_parquet("data/lines.parquet")
    blob = pickle.load(open("data/tfidf.pkl", "rb"))
    _tfidf, _tmat = blob["vectorizer"], blob["matrix"]
    _gmat = np.load("data/glove_lines.npy")
    _wv = api.load("glove-wiki-gigaword-100")
    tri = pickle.load(open("data/trigram.pkl", "rb"))
    _tri, _trimat = tri["vectorizer"], tri["matrix"]
    w = json.load(open("data/word_idf.json"))
    _widf, _wdefault = w["idf"], w["default"]


def _embed(text):
    """idf-weighted average of GloVe vectors (frequent fillers count less)."""
    num = np.zeros(_gmat.shape[1], dtype=np.float32)
    den = 0.0
    for w in TOK.findall(text.lower()):
        if w in _wv:
            wt = _widf.get(w, _wdefault)
            num += wt * _wv[w]
            den += wt
    if den == 0:
        return num
    v = num / den
    return v / max(np.linalg.norm(v), 1e-9)


def _top(scores, n, per_song=2):
    """Top-n indices, but drop duplicate lines and cap each song to `per_song`."""
    out, seen, per = [], set(), {}
    for i in np.argsort(-scores):
        if scores[i] <= 0:
            break
        r = _lines.iloc[i]
        key = r["line"].strip().lower()
        if key in seen or per.get(r["song"], 0) >= per_song:
            continue
        seen.add(key)
        per[r["song"]] = per.get(r["song"], 0) + 1
        out.append(i)
        if len(out) >= n:
            break
    return out


def _format(idx, scores):
    out = []
    for i in idx:
        r = _lines.iloc[i]
        out.append((float(scores[i]), r["line"], r["artist"], r["song"]))
    return out


def search_tfidf(query, n=5):
    _load()
    q = _tfidf.transform([query])
    scores = cosine_similarity(q, _tmat).ravel()
    return _format(_top(scores, n), scores)


def search_glove(query, n=5):
    _load()
    scores = _gmat @ _embed(query)          # cosine (both sides normalised)
    return _format(_top(scores, n), scores)


def search_trigram(query, n=5):
    """Fuzzy / typo-tolerant: cosine over character-trigram tf-idf."""
    _load()
    q = _tri.transform([query.lower()])
    scores = cosine_similarity(q, _trimat).ravel()
    return _format(_top(scores, n), scores)


def compare(query, n=5):
    print(f'\nQuery: "{query}"')
    methods = [("tf-idf   (keywords)", search_tfidf),
               ("GloVe    (meaning) ", search_glove),
               ("trigram  (fuzzy)   ", search_trigram)]
    for name, fn in methods:
        print(f"\n  ── {name} ──")
        hits = fn(query, n)
        if not hits or hits[0][0] == 0:
            print("     (no keyword overlap — nothing found)")
            continue
        for score, line, artist, song in hits:
            print(f'     {score:.2f}  "{line}"  — {artist}, {song}')


if __name__ == "__main__":
    for q in ["feeling heartbroken and alone",
              "i want to dance all night long",
              "missing my old hometown"]:
        compare(q)
