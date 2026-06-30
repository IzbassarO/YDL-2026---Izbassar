"""Lyrics CSV -> a clean list of searchable lyric lines with metadata.

Each "document" we search is a single lyric line (e.g. "my whole world fell
apart"), tagged with its artist + song so results are quotable.
"""
import re
import pandas as pd


def build_lines(csv_path, n_songs=8000, min_words=4, max_words=18, seed=42):
    """Return a DataFrame[line, artist, song] of deduped lyric lines."""
    df = pd.read_csv(csv_path).dropna(subset=["text", "artist", "song"])
    # Sample songs to keep the index snappy and the artifact small
    if n_songs and n_songs < len(df):
        df = df.sample(n=n_songs, random_state=seed)

    rows = []
    seen = set()
    for artist, song, text in zip(df["artist"], df["song"], df["text"]):
        for raw in text.splitlines():
            line = re.sub(r"\s+", " ", raw).strip()
            if not line:
                continue
            n = len(line.split())
            if n < min_words or n > max_words:
                continue
            key = line.lower()
            if key in seen:                # drop repeated chorus lines globally
                continue
            seen.add(key)
            rows.append((line, artist, song))

    out = pd.DataFrame(rows, columns=["line", "artist", "song"])
    return out


if __name__ == "__main__":
    lines = build_lines("data/spotify_millsongdata.csv")
    lines.to_parquet("data/lines.parquet")
    print(f"lyric lines: {len(lines):,}")
    print(lines.sample(5, random_state=1).to_string(index=False))
