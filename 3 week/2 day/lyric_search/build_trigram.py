"""Third index: character-trigram tf-idf over lyric lines.

A line becomes a bag of 3-character chunks ("london" -> lon, ond, ndo, don).
Two strings that share many trigrams are similar *as text* — so this tolerates
misspellings ("londconsider" ~ "london") and matches sound-alike words. It's
the fuzzy/typo-robust counterpart to keyword and meaning search.
"""
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

lines = pd.read_parquet("data/lines.parquet")
# char_wb keeps n-grams inside word boundaries (pads with spaces)
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 3), min_df=2)
matrix = vec.fit_transform(lines["line"].str.lower())
print(f"trigram vocab: {len(vec.vocabulary_):,}  matrix: {matrix.shape}")

with open("data/trigram.pkl", "wb") as f:
    pickle.dump({"vectorizer": vec, "matrix": matrix}, f)
print("saved -> data/trigram.pkl")
