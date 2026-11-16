
# =====================================================
# SMS SPAM & PHISHING DETECTION
# STEP 4: DATASET MERGING
# UCI + MENDELEY DATASET
# =====================================================

import pandas as pd
import os
import re

# -----------------------------------------------------
# 1. DATASET PATHS
# -----------------------------------------------------

UCI_PATH_OPTIONS = [
    "dataset/original/SMSSpamCollection",
    "dataset/SMSSpamCollection"
]

ADDITIONAL_PATH = "dataset/additional/Dataset_5971.csv"

UCI_PATH = next(
    (p for p in UCI_PATH_OPTIONS if os.path.exists(p)),
    None
)

if UCI_PATH is None:
    raise FileNotFoundError("UCI dataset not found!")

if not os.path.exists(ADDITIONAL_PATH):
    raise FileNotFoundError("Additional dataset not found!")

# -----------------------------------------------------
# 2. LOAD UCI DATASET
# -----------------------------------------------------

uci = pd.read_csv(
    UCI_PATH,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

uci["source"] = "UCI"

# -----------------------------------------------------
# 3. LOAD ADDITIONAL DATASET
# -----------------------------------------------------

additional = pd.read_csv(
    ADDITIONAL_PATH,
    encoding="utf-8-sig"
)

additional = additional.rename(
    columns={
        "LABEL": "label",
        "TEXT": "message",
        "URL": "url",
        "EMAIL": "email",
        "PHONE": "phone"
    }
)

additional["source"] = "Mendeley"

# -----------------------------------------------------
# 4. SELECT REQUIRED COLUMNS
# -----------------------------------------------------

uci = uci[["label", "message", "source"]]

additional = additional[
    ["label", "message", "source", "url", "email", "phone"]
]

# -----------------------------------------------------
# 5. COMBINE DATASETS
# -----------------------------------------------------

df = pd.concat(
    [uci, additional],
    ignore_index=True,
    sort=False
)

print("\n========== RAW COMBINED DATA ==========")
print("Total Records:", len(df))
print("Columns:", df.columns.tolist())

# -----------------------------------------------------
# 6. NORMALIZE LABELS
# -----------------------------------------------------

df["label"] = df["label"].astype(str).str.strip().str.lower()

label_mapping = {
    "ham": "ham",
    "spam": "spam",
    "smishing": "spam"
}

df["original_label"] = df["label"]

df["label"] = df["label"].map(label_mapping)

# Remove unknown labels
df = df.dropna(subset=["label", "message"])

# -----------------------------------------------------
# 7. CLEAN MESSAGE TEXT
# -----------------------------------------------------

df["message"] = df["message"].astype(str).str.strip()

df = df[df["message"] != ""]

# Normalize text for duplicate identification
def normalize_message(text):

    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["text_key"] = df["message"].apply(
    normalize_message
)

# -----------------------------------------------------
# 8. CHECK DUPLICATES AND LABEL CONFLICTS
# -----------------------------------------------------

duplicate_rows = df[
    df.duplicated(subset=["text_key"], keep=False)
]

conflict_keys = (
    duplicate_rows.groupby("text_key")["label"]
    .nunique()
)

conflict_keys = conflict_keys[
    conflict_keys > 1
].index

conflicts = df[
    df["text_key"].isin(conflict_keys)
].copy()

# Save conflicting records separately
os.makedirs("outputs", exist_ok=True)

conflicts.to_csv(
    "outputs/label_conflicts.csv",
    index=False
)

# Exclude conflicting messages from training dataset
df = df[
    ~df["text_key"].isin(conflict_keys)
].copy()

# -----------------------------------------------------
# 9. REMOVE DUPLICATES
# -----------------------------------------------------

def merge_sources(series):
    return ", ".join(sorted(set(series)))

combined = (
    df.groupby("text_key", sort=False)
    .agg(
        message=("message", "first"),
        label=("label", "first"),
        source=("source", merge_sources),
        original_label=("original_label", merge_sources),
        url=("url", "first"),
        email=("email", "first"),
        phone=("phone", "first")
    )
    .reset_index(drop=True)
)

# -----------------------------------------------------
# 10. SAVE FINAL DATASET
# -----------------------------------------------------

os.makedirs("dataset/combined", exist_ok=True)

combined.to_csv(
    "dataset/combined/combined_sms.csv",
    index=False,
    encoding="utf-8"
)

# -----------------------------------------------------
# 11. DISPLAY FINAL RESULTS
# -----------------------------------------------------

print("\n========== DATASET MERGING RESULTS ==========")

print("Raw Combined Records:", len(uci) + len(additional))

print("Conflicting Records:", len(conflicts))

print("Final Unique Records:", len(combined))

print("\n========== FINAL CLASS DISTRIBUTION ==========")

print(combined["label"].value_counts())

print("\n========== SOURCE DISTRIBUTION ==========")

print(combined["source"].value_counts())

print("\n========== FINAL DATASET INFORMATION ==========")

print(combined.info())

print("\nFinal Dataset Saved Successfully!")
print("Location: dataset/combined/combined_sms.csv")

print("\nDATASET MERGING COMPLETED!")
