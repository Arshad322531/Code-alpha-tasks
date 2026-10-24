import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# ---------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------
DATA_PATH = "credit_data.csv"   # apni CSV file ka naam yahan likho
TARGET_COL = "creditworthy"     # target column (1 = good, 0 = bad)


# ---------------------------------------------------------------------
# STEP 1: LOAD DATA
# ---------------------------------------------------------------------
def make_demo_data(n=5000):
    """Demo financial data banata hai (sirf tab use hota hai jab CSV na ho)."""
    rng = np.random.default_rng(RANDOM_STATE)
    df = pd.DataFrame({
        "age": rng.integers(21, 65, n),
        "income": rng.normal(60000, 20000, n).clip(15000, None),
        "total_debt": rng.normal(20000, 12000, n).clip(0, None),
        "loan_amount": rng.normal(15000, 8000, n).clip(1000, None),
        "credit_history_years": rng.integers(0, 30, n),
        "num_late_payments": rng.poisson(1.5, n),
        "num_open_accounts": rng.integers(1, 15, n),
        "employment_years": rng.integers(0, 35, n),
    })
    # Risk score: zyada debt aur late payments = bura credit
    score = (
        -1.5 * (df["total_debt"] / df["income"])
        - 0.6 * df["num_late_payments"]
        - 0.8 * (df["loan_amount"] / df["income"])
        + 0.05 * df["credit_history_years"]
        + 0.03 * df["employment_years"]
        + 1.5
        + rng.normal(0, 0.8, n)
    )
    df[TARGET_COL] = (score > 0).astype(int)
    return df


if os.path.exists(DATA_PATH):
    print(f"Loading dataset from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
else:
    print("CSV not found -> demo data use ho raha hai.")
    df = make_demo_data()

print("\nDataset shape:", df.shape)
print(df.head())
print("\nTarget distribution:\n", df[TARGET_COL].value_counts())

# Missing values handle karna (numeric columns ko median se fill)
df = df.fillna(df.median(numeric_only=True))


# ---------------------------------------------------------------------
# STEP 2: FEATURE ENGINEERING
# ---------------------------------------------------------------------
# Naye features jo financial behaviour ko behtar dikhate hain.
# (Ye tab bante hain jab original columns data mein maujood hon.)
def add_features(data):
    data = data.copy()
    cols = data.columns
    if {"total_debt", "income"} <= set(cols):
        # Debt-to-Income ratio: income ke muqable mein kitna qarz hai
        data["debt_to_income"] = data["total_debt"] / (data["income"] + 1)
    if {"loan_amount", "income"} <= set(cols):
        # Loan-to-Income ratio: loan income ke muqable mein kitna bara hai
        data["loan_to_income"] = data["loan_amount"] / (data["income"] + 1)
    if {"num_late_payments", "credit_history_years"} <= set(cols):
        # Late payments per year: payment history ka behaviour
        data["late_per_year"] = data["num_late_payments"] / (data["credit_history_years"] + 1)
    return data


df = add_features(df)

# Agar koi text (categorical) columns hon to unhe numbers mein badal do
df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]


# ---------------------------------------------------------------------
# STEP 3: TRAIN / TEST SPLIT + SCALING
# ---------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# Scaling Logistic Regression ke liye zaroori hai (features ek scale par aayein)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)


# ---------------------------------------------------------------------
# STEP 4: TRAIN MODELS
# ---------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced",
                                            random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                            random_state=RANDOM_STATE, n_jobs=-1),
}

# Logistic Regression scaled data par, baqi trees original data par chalte hain
use_scaled = {"Logistic Regression": True, "Decision Tree": False, "Random Forest": False}

results = []
probs = {}
preds = {}

for name, model in models.items():
    Xtr = X_train_sc if use_scaled[name] else X_train
    Xte = X_test_sc if use_scaled[name] else X_test
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    prob = model.predict_proba(Xte)[:, 1]
    preds[name], probs[name] = pred, prob

    # ---------------------------------------------------------------
    # STEP 5: EVALUATION METRICS
    # Precision : jin ko "good" kaha, un mein se kitne sach mein good the
    # Recall    : sab good logon mein se kitnon ko pakda
    # F1-Score  : Precision aur Recall ka balance
    # ROC-AUC   : model good/bad ko kitni achi tarah alag karta hai
    # ---------------------------------------------------------------
    results.append({
        "Model": name,
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1-Score": f1_score(y_test, pred),
        "ROC-AUC": roc_auc_score(y_test, prob),
    })

results_df = pd.DataFrame(results).set_index("Model").round(4)
print("\n================ MODEL COMPARISON ================")
print(results_df)

best_name = results_df["ROC-AUC"].idxmax()
print(f"\nBest model (ROC-AUC ke hisaab se): {best_name}")
print("\nClassification report:\n",
      classification_report(y_test, preds[best_name],
                            target_names=["Bad credit (0)", "Good credit (1)"]))


# ---------------------------------------------------------------------
# STEP 6: GRAPHS
# ---------------------------------------------------------------------
# 6a) ROC curves (sab models ek graph mein)
plt.figure(figsize=(7, 5))
for name in models:
    fpr, tpr, _ = roc_curve(y_test, probs[name])
    plt.plot(fpr, tpr, label=f"{name} (AUC = {results_df.loc[name, 'ROC-AUC']:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Credit Scoring")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.close()

# 6b) Confusion matrix (best model)
ConfusionMatrixDisplay.from_predictions(
    y_test, preds[best_name],
    display_labels=["Bad", "Good"], cmap="Blues"
)
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.close()

# 6c) Feature importance (Random Forest) - kaun sa feature sab se zyada asar karta hai
rf = models["Random Forest"]
importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
plt.figure(figsize=(7, 5))
sns.barplot(x=importance.values, y=importance.index, color="steelblue")
plt.title("Feature Importance (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()

print("\nGraphs saved: roc_curve.png, confusion_matrix.png, feature_importance.png")


# ---------------------------------------------------------------------
# STEP 7: SAVE BEST MODEL + EXAMPLE PREDICTION
# ---------------------------------------------------------------------
joblib.dump({"model": models[best_name], "scaler": scaler,
             "columns": list(X.columns), "scaled": use_scaled[best_name]},
            "credit_model.pkl")
print("Model saved: credit_model.pkl")

# Ek naye customer ki prediction (test set ka pehla customer)
sample = X_test.iloc[[0]]
sample_in = scaler.transform(sample) if use_scaled[best_name] else sample
p_good = models[best_name].predict_proba(sample_in)[0, 1]
print(f"\nExample customer: probability of GOOD credit = {p_good:.2%} "
      f"-> {'APPROVE' if p_good >= 0.5 else 'REJECT'}")