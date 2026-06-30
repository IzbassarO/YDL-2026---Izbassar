"""Build the two search indexes over lyric lines: tf-idf and averaged GloVe.

tf-idf  = bag of words: matches on the *words* you typed.
GloVe   = average of word vectors: matches on *meaning*, even with no shared words.
Both are techniques straight from the Day-2 lab (no heavy models, no BERT).
"""
import pickle
import numpy as np
import pandas as pd
import gensim.downloader as api
from sklearn.feature_extraction.text import TfidfVectorizer

lines = pd.read_parquet("data/lines.parquet")
texts = lines["line"].tolist()
print(f"indexing {len(texts):,} lines")

# --- tf-idf index ---
tfidf = TfidfVectorizer(stop_words="english", min_df=2)
tfidf_matrix = tfidf.fit_transform(texts)          # sparse, already L2-normalised
print(f"tf-idf vocab: {len(tfidf.vocabulary_):,}")

# --- averaged GloVe index ---
print("loading GloVe...")
wv = api.load("glove-wiki-gigaword-100")
TOK = __import__("re").compile(r"[a-z']+")

def embed(text):
    """Average the GloVe vectors of the words in `text` (a meaning vector)."""
    vecs = [wv[w] for w in TOK.findall(text.lower()) if w in wv]
    if not vecs:
        return np.zeros(wv.vector_size, dtype=np.float32)
    return np.mean(vecs, axis=0)

glove_matrix = np.vstack([embed(t) for t in texts]).astype(np.float32)
# L2-normalise so a dot product == cosine similarity
norms = np.linalg.norm(glove_matrix, axis=1, keepdims=True)
glove_matrix /= np.clip(norms, 1e-9, None)
print(f"GloVe line matrix: {glove_matrix.shape}")

# --- save ---
with open("data/tfidf.pkl", "wb") as f:
    pickle.dump({"vectorizer": tfidf, "matrix": tfidf_matrix}, f)
np.save("data/glove_lines.npy", glove_matrix)
print("saved -> data/tfidf.pkl, data/glove_lines.npy")
