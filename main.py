import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ─────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────
df = pd.read_csv(r"C:\Users\admin\Downloads\Healthcare.csv")

print(f"Loaded: {df.shape}")

# ─────────────────────────────
# 2. CLEANING
# ─────────────────────────────
df.drop(columns=["Patient_ID"], inplace=True)
df.drop_duplicates(inplace=True)

df["Symptoms"] = df["Symptoms"].str.lower()
df["Gender"] = df["Gender"].str.strip().str.lower()

df["Age"] = df["Age"].fillna(df["Age"].median())

df["Gender"] = df["Gender"].map({"male": 0, "female": 1})

df.dropna(subset=["Symptoms", "Disease", "Gender"], inplace=True)

# ─────────────────────────────
# 3. SYMPTOM PROCESSING (FAST)
# ─────────────────────────────
row_list = df["Symptoms"].apply(
    lambda x: [s.strip() for s in x.split(",")]
).tolist()

mlb = MultiLabelBinarizer()
X_symptoms = mlb.fit_transform(row_list)

# ─────────────────────────────
# 4. NUMERIC FEATURES
# ─────────────────────────────
scaler = StandardScaler()

num_features = scaler.fit_transform(
    df[["Age"]]
)

# ─────────────────────────────
# 5. FINAL FEATURE MATRIX
# ─────────────────────────────
X = np.hstack([
    X_symptoms,
    num_features,
    df[["Gender"]].values
])

# ─────────────────────────────
# 6. LABEL ENCODING
# ─────────────────────────────
le = LabelEncoder()
y = le.fit_transform(df["Disease"])

# ─────────────────────────────
# 7. TRAIN TEST SPLIT
# ─────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ─────────────────────────────
# 8. MODEL (FAST + STRONG BASELINE)
# ─────────────────────────────
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# ─────────────────────────────
# 9. EVALUATION
# ─────────────────────────────
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

print("\nAccuracy:", round(acc * 100, 2), "%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ─────────────────────────────
# 10. SAVE MODEL
# ─────────────────────────────
joblib.dump({
    "model": model,
    "mlb": mlb,
    "le": le,
    "scaler": scaler
}, "disease_model.pkl")

print("\nModel saved successfully!")