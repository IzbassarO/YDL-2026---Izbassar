# YDL 2026 Day 4 — Titanic Kaggle with Linear Regression Only

## Goal

Build a Kaggle Titanic competition solution using **only `sklearn.linear_model.LinearRegression`**.

The task is intentionally constrained: the model must be linear regression, even though Titanic survival is a binary classification problem. The main challenge is to improve the result through **feature engineering, missing-value handling, encoding, and threshold tuning**.

Competition: https://www.kaggle.com/competitions/titanic

---

## Absolute Rules

Claude Code must follow these rules strictly:

1. Use **only**:

```python
from sklearn.linear_model import LinearRegression
```

2. Do **not** use any other ML model:
   - No LogisticRegression
   - No RandomForest
   - No DecisionTree
   - No XGBoost / CatBoost / LightGBM
   - No SVM
   - No kNN
   - No neural networks
   - No ensemble models

3. The model output will be a continuous number. Convert it to binary predictions:
   - `1` = survived
   - `0` = did not survive

4. Tune the threshold manually or through validation data.

5. The final Kaggle submission must contain exactly two columns:

```text
PassengerId,Survived
```

6. The submission must contain exactly **418 prediction rows plus the header**.

---

## Data Files

Expected Kaggle files:

```text
train.csv
test.csv
gender_submission.csv
```

Expected structure:

```text
project/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
├── notebooks/
│   └── titanic_linear_regression.ipynb
├── src/
│   └── titanic_linear_regression.py
├── submissions/
│   └── submission_linear_regression.csv
└── CLAUDE.md
```

If folders do not exist, create them.

---

## Dataset Description

Target column in `train.csv`:

```text
Survived
```

Main raw features:

| Column | Meaning |
|---|---|
| `PassengerId` | Passenger identifier |
| `Pclass` | Ticket class: 1, 2, 3 |
| `Name` | Passenger name |
| `Sex` | Passenger sex |
| `Age` | Age in years |
| `SibSp` | Number of siblings/spouses aboard |
| `Parch` | Number of parents/children aboard |
| `Ticket` | Ticket number |
| `Fare` | Passenger fare |
| `Cabin` | Cabin number |
| `Embarked` | Port of embarkation: C, Q, S |

---

## Required Implementation Plan

### 1. Load data

Load:

```python
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
```

Keep `PassengerId` from `test.csv` for the final submission.

---

### 2. Basic EDA

Print or inspect:

```python
train.head()
train.info()
train.isna().sum()
train["Survived"].value_counts(normalize=True)
```

Also check survival rate by:

```python
Sex
Pclass
Embarked
Age groups
FamilySize
Title
```

---

### 3. Feature engineering

Create a shared preprocessing function that works for both train and test.

#### Family features

```python
FamilySize = SibSp + Parch + 1
IsAlone = 1 if FamilySize == 1 else 0
```

Optional:

```python
SmallFamily = 1 if FamilySize between 2 and 4 else 0
LargeFamily = 1 if FamilySize >= 5 else 0
```

#### Title feature from name

Extract title from `Name`.

Examples:

```text
Braund, Mr. Owen Harris -> Mr
Cumings, Mrs. John Bradley -> Mrs
Heikkinen, Miss. Laina -> Miss
```

Suggested mapping:

```python
rare_titles = [
    "Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major",
    "Rev", "Sir", "Jonkheer", "Dona"
]
```

Map rare titles into `"Rare"`.

Normalize:

```python
Mlle -> Miss
Ms -> Miss
Mme -> Mrs
```

Useful final title categories:

```text
Mr, Mrs, Miss, Master, Rare
```

#### Cabin feature

Use cabin availability:

```python
HasCabin = 1 if Cabin is not missing else 0
```

Optional: extract deck letter from cabin:

```python
Deck = first letter of Cabin, or "Unknown"
```

#### Ticket feature

Optional but useful:

```python
TicketPrefix
TicketGroupSize
```

Simpler option:

```python
TicketGroupSize = number of passengers with the same Ticket
```

Compute this using combined train + test data, because ticket groups may appear across both files. This is allowed because it does not use the target values from the test set.

---

### 4. Missing values

Do not fill all missing ages with one global mean. Use smarter grouped filling.

Recommended:

```python
Age = Age filled by median grouped by Title and Pclass
```

Fallback:

```python
Age = Age filled by median grouped by Title
Age = Age filled by global median
```

For `Fare`:

```python
Fare = Fare filled by median grouped by Pclass
```

For `Embarked`:

```python
Embarked = Embarked filled with most frequent value
```

For `Cabin`:

```python
Cabin = Cabin filled with "Unknown"
```

---

### 5. Encoding

Use numeric encoding suitable for linear regression.

Recommended approach:

```python
pd.get_dummies(..., drop_first=False)
```

Categorical columns to encode:

```text
Sex
Embarked
Title
Deck
Pclass
```

Important: after encoding train and test separately, align columns:

```python
X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)
```

---

### 6. Scaling

Linear regression can work without scaling, but scaling may help when feature ranges differ.

Use:

```python
from sklearn.preprocessing import StandardScaler
```

This is allowed because it is preprocessing, not a model.

Fit scaler only on training data:

```python
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)
```

---

### 7. Validation split

Create validation data from `train.csv`.

Use:

```python
from sklearn.model_selection import train_test_split
```

Recommended:

```python
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

Train only on `X_tr`, evaluate on `X_val`.

---

### 8. Train Linear Regression

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_tr_scaled, y_tr)

val_raw = model.predict(X_val_scaled)
```

---

### 9. Threshold tuning

Because linear regression outputs numbers, test multiple thresholds.

Example:

```python
thresholds = np.arange(0.25, 0.76, 0.01)

best_threshold = 0.5
best_acc = 0

for t in thresholds:
    preds = (val_raw >= t).astype(int)
    acc = accuracy_score(y_val, preds)
    if acc > best_acc:
        best_acc = acc
        best_threshold = t

print(best_threshold, best_acc)
```

Use the best threshold for test predictions.

---

### 10. Retrain on full training data

After choosing the feature set and threshold:

```python
model = LinearRegression()
model.fit(X_train_scaled, y)

test_raw = model.predict(X_test_scaled)
test_preds = (test_raw >= best_threshold).astype(int)
```

---

### 11. Create Kaggle submission

```python
submission = pd.DataFrame({
    "PassengerId": test["PassengerId"],
    "Survived": test_preds
})

submission.to_csv("submissions/submission_linear_regression.csv", index=False)
```

Check:

```python
submission.shape
submission.head()
submission["Survived"].value_counts()
```

The shape must be:

```text
(418, 2)
```

---

## Suggested Baseline Code

Claude Code should create a runnable script at:

```text
src/titanic_linear_regression.py
```

The script should:

1. Load data from `data/`
2. Preprocess train and test consistently
3. Train `LinearRegression`
4. Tune threshold on validation set
5. Retrain on all training data
6. Save submission to `submissions/submission_linear_regression.csv`
7. Print validation accuracy and selected threshold

---

## Suggested Notebook

Claude Code should also create a notebook at:

```text
notebooks/titanic_linear_regression.ipynb
```

Notebook sections:

1. Title and task description
2. Import libraries
3. Load data
4. EDA
5. Feature engineering
6. Missing value handling
7. Encoding
8. Train/validation split
9. LinearRegression training
10. Threshold tuning
11. Final training
12. Submission generation
13. Short conclusion for demo

---

## Demo Notes for 15:30

Prepare a short explanation:

```text
I used only LinearRegression as required.
The main features were Sex, Pclass, Age, Fare, Embarked, FamilySize, IsAlone, Title, and HasCabin.
Missing Age was filled using grouped medians by Title and Pclass, not one global mean.
Since LinearRegression outputs continuous values, I tested thresholds from 0.25 to 0.75 and selected the threshold with the best validation accuracy.
Then I retrained the model on the full training set and generated the Kaggle submission file.
```

---

## Quality Checklist

Before finishing, verify:

- [ ] Only `LinearRegression` is used as the ML model.
- [ ] No classification model is imported.
- [ ] Missing `Age` is handled thoughtfully.
- [ ] Categories are encoded numerically.
- [ ] Train and test columns are aligned.
- [ ] Threshold is tuned using validation data.
- [ ] Final CSV has only `PassengerId` and `Survived`.
- [ ] Final CSV has 418 rows.
- [ ] File saved to `submissions/submission_linear_regression.csv`.
- [ ] Code runs from the project root without manual changes.

---

## Important Warning

Do not try to get a perfect leaderboard score by using leaked Titanic answers from the internet. The goal is a fair and understandable solution using only the given train/test data and `LinearRegression`.

The final result should be honest, explainable, and ready to present.
