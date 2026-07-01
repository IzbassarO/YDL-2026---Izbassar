"""Organize the DAWN images into train/val folders by road-weather class.

DAWN (Kaggle YOLO export) is a flat images/ folder; the weather condition is in
each filename prefix. We fold the variants into four road-condition classes:
    fog  <- foggy, haze, mist
    rain <- rain_storm
    snow <- snow_storm
    sand <- sand_storm, sand_storm_g, dusttornado
This is the "road condition" label the CNN learns (mirrors Aqyl Jol's AstanaSnow/etc.).
"""
import os
import random
import shutil

SRC = "data/images"
OUT = "data/split"
VAL_FRAC = 0.15
random.seed(42)


def label_of(name):
    n = name.lower()
    if n.startswith(("foggy", "haze", "mist")):
        return "fog"
    if n.startswith("rain"):
        return "rain"
    if n.startswith("snow"):
        return "snow"
    if n.startswith(("sand", "dust")):
        return "sand"
    return None


def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    buckets = {}
    for f in os.listdir(SRC):
        if not f.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        lab = label_of(f)
        if lab:
            buckets.setdefault(lab, []).append(f)

    counts = {}
    for lab, files in buckets.items():
        random.shuffle(files)
        n_val = max(1, int(len(files) * VAL_FRAC))
        for split, subset in [("val", files[:n_val]), ("train", files[n_val:])]:
            d = os.path.join(OUT, split, lab)
            os.makedirs(d, exist_ok=True)
            for f in subset:
                shutil.copy(os.path.join(SRC, f), os.path.join(d, f))
        counts[lab] = (len(files) - n_val, n_val)

    print("class    train  val")
    for lab in sorted(counts):
        tr, va = counts[lab]
        print(f"{lab:6}   {tr:5}  {va:4}")
    print(f"total images: {sum(sum(c) for c in counts.values())}")


if __name__ == "__main__":
    main()
