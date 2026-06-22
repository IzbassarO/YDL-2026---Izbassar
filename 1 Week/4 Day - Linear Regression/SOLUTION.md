# Titanic — Linear Regression Solution (YDL 2026, Day 4)

**Constraint:** the ML model must be `sklearn.linear_model.LinearRegression` — no classifier.
**Result:** baseline submission scored **0.77033** on Kaggle; after adding a group-survival
feature and trimming the feature set, honest cross-validation rose from ~0.83 to **~0.85**.

- Notebook: `notebooks/titanic_linear_regression.ipynb`
- Submission: `submissions/submission_linear_regression.csv` (418 rows, `PassengerId,Survived`)

---

## How it works, step by step

### 1. Load data
Read `train.csv` (891 rows) and `test.csv` (418 rows); keep the test `PassengerId` for the
submission.

### 2. EDA
Confirmed missing values (`Age`, `Cabin`, a little `Embarked`/`Fare`) and that survival depends
strongly on **Sex** (74% female vs 19% male), **Pclass** (63% → 47% → 24%), and **Embarked**.

### 3. Feature engineering
A single shared function processes train and test identically. Train + test are concatenated
**only** to compute fills and group statistics — this uses no target values, so there is no leak.

| Feature | What / why |
|---|---|
| `Title` | Parsed from `Name`; synonyms merged (`Mlle/Ms→Miss`, `Mme→Mrs`) and rare titles → `Rare`. Encodes age/sex/social status compactly. |
| `FamilySize` | `SibSp + Parch + 1`. |
| `IsChild` | `Age < 16` — children were prioritised for lifeboats. |
| `HasCabin` | Whether a cabin was recorded (proxy for wealth / known passengers). |
| `FareLog` | `log1p(Fare)` to tame the heavy right skew. |
| `Sex`, `Pclass` | Encoded numerically / one-hot. |
| **`FamSurv`** | **Family / group survival — the key upgrade (see §5).** |

### 4. Missing-value handling (no single global mean)
- **Age** → median by `Title` + `Pclass`, then by `Title`, then global median.
- **Fare** → median by `Pclass`.
- **Embarked** → most frequent value.

### 5. Family / group survival feature — the main improvement
People travelling together largely shared the same fate. For each passenger we look at the
**known outcomes of their relatives and ticket-mates**:

1. Group by **surname + fare** (real families). If any *other* member survived → `1`; if all
   *other* known members died → `0`.
2. For anyone still unknown, fall back to **same ticket** groups.
3. Default `0.5` when nothing is known.

It stays honest (not "leaked answers"):
- The passenger themselves is **always excluded** before looking — never uses its own label.
- Test passengers have `Survived = NaN`, so their value is driven **only by train relatives**.
- **171 / 418** test passengers (96 → 1, 75 → 0) get a confident signal; the rest stay neutral.

### 6. Encoding & scaling
Categoricals (`Title`, `Pclass`) one-hot encoded; train/test columns aligned so they match.
Features standardised with `StandardScaler` (fit on training data only) — preprocessing, not a model.

### 7. Threshold tuning (fold-safe cross-validation)
Linear regression outputs a continuous score, so we need a cut-off. Accuracy is averaged over
**5-fold × 6-repeat = 30 folds** for thresholds from 0.20 to 0.80. Critically, `FamSurv` is
**recomputed inside each fold** from that fold's training labels only — otherwise validation
labels would leak through relatives and inflate the estimate.
→ Best threshold ≈ **0.53**, honest CV accuracy ≈ **0.849**.

### 8. Final model
Retrain `LinearRegression` on **all** training rows, predict the test set, and apply the tuned
threshold to get 0/1 predictions.

### 9. Submission
Write `submissions/submission_linear_regression.csv` — exactly `PassengerId,Survived`, 418 rows.

---

## Why this beats the 0.770 baseline
- The baseline had a large CV→leaderboard gap (CV 0.83, LB 0.77) = **overfitting**. Trimming
  to a lean feature set (dropping noisy `Deck` dummies, `FarePerPerson`, etc.) shrinks that gap.
- The **group-survival feature** injects real, honest signal about who lived/died together —
  the single biggest lever on this dataset — lifting honest CV to ~0.85.
- Everything is still a plain `LinearRegression` with a tuned threshold: explainable and
  reproducible by running the notebook top to bottom.

---

## How to run
```bash
# from the project root (Anaconda has pandas / numpy / scikit-learn)
/opt/anaconda3/bin/jupyter nbconvert --to notebook --execute --inplace \
  notebooks/titanic_linear_regression.ipynb
```
Or open the notebook and **Run All**. The submission is written to `submissions/`.
