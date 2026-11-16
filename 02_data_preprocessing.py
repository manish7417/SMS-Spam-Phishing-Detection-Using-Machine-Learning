
# ==========================================================
# SMS SPAM & PHISHING DETECTION
# STEP 2: COMPLETE DATA PREPROCESSING PIPELINE
# UCI + MENDELEY DATASET
# ==========================================================

import pandas as pd
import numpy as np
import re
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.sparse import save_npz

# ----------------------------------------------------------
# 1. PATH CONFIGURATION
# ----------------------------------------------------------

UCI_PATH = "dataset/SMSSpamCollection"

MENDELEY_PATH = "dataset/additional/Dataset_5971.csv"

OUTPUT_DIR = "outputs"
MODEL_DIR = "model"
COMBINED_DIR = "dataset/combined"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(COMBINED_DIR, exist_ok=True)

# ----------------------------------------------------------
# 2. LOAD UCI DATASET
# ----------------------------------------------------------

uci = pd.read_csv(
    UCI_PATH,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

uci["source"] = "UCI"

# ----------------------------------------------------------
# 3. LOAD MENDELEY DATASET
# ----------------------------------------------------------

mendeley = pd.read_csv(
    MENDELEY_PATH,
    encoding="utf-8-sig"
)

mendeley.columns = mendeley.columns.str.strip().str.upper()

mendeley = mendeley.rename(
    columns={
        "LABEL": "label",
        "TEXT": "message",
        "URL": "url",
        "EMAIL": "email",
        "PHONE": "phone"
    }
)

mendeley["source"] = "Mendeley"

# ----------------------------------------------------------
# 4. SELECT COMMON COLUMNS
# ----------------------------------------------------------

uci = uci[["label", "message", "source"]].copy()

additional_columns = [
    "label", "message", "source",
    "url", "email", "phone"
]

mendeley = mendeley[additional_columns].copy()

# ----------------------------------------------------------
# 5. COMBINE BOTH DATASETS
# ----------------------------------------------------------

df = pd.concat(
    [uci, mendeley],
    ignore_index=True,
    sort=False
)

print("\n========== RAW DATA ==========")
print("UCI Records:", len(uci))
print("Mendeley Records:", len(mendeley))
print("Total Raw Records:", len(df))

# ----------------------------------------------------------
# 6. NORMALIZE LABELS
# ----------------------------------------------------------

df["label"] = (
    df["label"]
    .astype(str)
    .str.lower()
    .str.strip()
)

df["original_label"] = df["label"]

label_mapping = {
    "ham": "ham",
    "spam": "spam",
    "smishing": "spam"
}

df["label"] = df["label"].map(label_mapping)

# Remove invalid labels and empty messages
df = df.dropna(subset=["label", "message"]).copy()

df["message"] = df["message"].astype(str).str.strip()

df = df[df["message"] != ""].copy()

# ----------------------------------------------------------
# 7. NORMALIZE TEXT FOR DUPLICATE DETECTION
# ----------------------------------------------------------

def normalize_message(text):

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


df["text_key"] = df["message"].apply(
    normalize_message
)

# ----------------------------------------------------------
# 8. DETECT CONFLICTING LABELS
# ----------------------------------------------------------

duplicate_groups = df.groupby("text_key")["label"].nunique()

conflict_keys = duplicate_groups[
    duplicate_groups > 1
].index

conflicts = df[
    df["text_key"].isin(conflict_keys)
].copy()

conflicts.to_csv(
    f"{OUTPUT_DIR}/label_conflicts.csv",
    index=False
)

print("\nConflicting Messages:", len(conflicts))

# Exclude conflicting messages
df = df[
    ~df["text_key"].isin(conflict_keys)
].copy()

# ----------------------------------------------------------
# 9. MERGE DUPLICATES AND PRESERVE SOURCES
# ----------------------------------------------------------

def merge_values(series):

    values = [
        str(value).strip()
        for value in series
        if pd.notna(value)
        and str(value).strip()
        and str(value).lower() != "nan"
    ]

    return ", ".join(sorted(set(values))) if values else np.nan


combined = (
    df.groupby("text_key", sort=False)
    .agg(
        message=("message", "first"),
        label=("label", "first"),
        source=("source", merge_values),
        original_label=("original_label", merge_values),
        url=("url", merge_values),
        email=("email", merge_values),
        phone=("phone", merge_values)
    )
    .reset_index(drop=True)
)

# Save combined dataset
combined.to_csv(
    f"{COMBINED_DIR}/combined_sms.csv",
    index=False,
    encoding="utf-8"
)

print("\n========== MERGING RESULTS ==========")

print("Final Unique Records:", len(combined))

print("\nClass Distribution:")
print(combined["label"].value_counts())

print("\nSource Distribution:")
print(combined["source"].value_counts())

# ----------------------------------------------------------
# 10. TEXT PREPROCESSING
# ----------------------------------------------------------

def clean_text(text):

    text = text.lower()

    # Replace URLs
    text = re.sub(
        r"http\S+|www\S+",
        " urltoken ",
        text
    )

    # Replace email addresses
    text = re.sub(
        r"\S+@\S+",
        " emailtoken ",
        text
    )

    # Replace phone numbers
    text = re.sub(
        r"\+?\d[\d\s\-]{6,}\d",
        " phonetoken ",
        text
    )

    # Replace remaining numbers
    text = re.sub(
        r"\d+",
        " numtoken ",
        text
    )

    # Remove special characters
    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


combined["clean_message"] = combined["message"].apply(
    clean_text
)

combined = combined[
    combined["clean_message"].str.len() > 0
].copy()

# Save cleaned dataset
combined.to_csv(
    f"{COMBINED_DIR}/processed_combined_sms.csv",
    index=False
)

# ----------------------------------------------------------
# 11. SEPARATE FEATURES AND TARGET
# ----------------------------------------------------------

X = combined["clean_message"]
y = combined["label"]

# ----------------------------------------------------------
# 12. TRAIN-TEST SPLIT
# ----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ----------------------------------------------------------
# 13. TF-IDF FEATURE EXTRACTION
# ----------------------------------------------------------

tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=10000,
    sublinear_tf=True
)

# Fit only on training data
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform testing data
X_test_tfidf = tfidf.transform(X_test)

# ----------------------------------------------------------
# 14. SAVE PROCESSED DATA
# ----------------------------------------------------------

save_npz(
    f"{OUTPUT_DIR}/X_train_combined.npz",
    X_train_tfidf
)

save_npz(
    f"{OUTPUT_DIR}/X_test_combined.npz",
    X_test_tfidf
)

pd.DataFrame({
    "label": y_train
}).to_csv(
    f"{OUTPUT_DIR}/y_train_combined.csv",
    index=False
)

pd.DataFrame({
    "label": y_test
}).to_csv(
    f"{OUTPUT_DIR}/y_test_combined.csv",
    index=False
)

# Save test messages for later error analysis
pd.DataFrame({
    "message": X_test.values,
    "label": y_test.values
}).to_csv(
    f"{OUTPUT_DIR}/test_messages_combined.csv",
    index=False
)

# Save vectorizer
joblib.dump(
    tfidf,
    f"{MODEL_DIR}/tfidf_combined.pkl"
)

# ----------------------------------------------------------
# 15. FINAL PREPROCESSING REPORT
# ----------------------------------------------------------

print("\n========== PREPROCESSING RESULTS ==========")

print("Final Dataset:", len(combined))

print("Training Samples:", X_train_tfidf.shape[0])
print("Testing Samples:", X_test_tfidf.shape[0])

print("Vocabulary Size:", len(tfidf.vocabulary_))

print("Training Matrix:", X_train_tfidf.shape)
print("Testing Matrix:", X_test_tfidf.shape)

print("\nTraining Class Distribution:")
print(y_train.value_counts())

print("\nTesting Class Distribution:")
print(y_test.value_counts())

print("\n========== SAVED FILES ==========")

print("1. combined_sms.csv")
print("2. processed_combined_sms.csv")
print("3. X_train_combined.npz")
print("4. X_test_combined.npz")
print("5. y_train_combined.csv")
print("6. y_test_combined.csv")
print("7. tfidf_combined.pkl")

print("\nCOMPLETE DATA PREPROCESSING SUCCESSFUL!")
