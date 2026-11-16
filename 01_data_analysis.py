
# =====================================================
# SMS SPAM & PHISHING DETECTION
# STEP 1: EXPLORATORY DATA ANALYSIS (EDA)
# UCI + MENDELEY + COMBINED DATASET
# =====================================================

import pandas as pd
import matplotlib.pyplot as plt
import os

# -----------------------------------------------------
# 1. PATH CONFIGURATION
# -----------------------------------------------------

UCI_PATH = "dataset/SMSSpamCollection"
MENDELEY_PATH = "dataset/additional/Dataset_5971.csv"
COMBINED_PATH = "dataset/combined/combined_sms.csv"

os.makedirs("outputs", exist_ok=True)

# -----------------------------------------------------
# 2. LOAD ALL DATASETS
# -----------------------------------------------------

uci = pd.read_csv(
    UCI_PATH,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8"
)

mendeley = pd.read_csv(MENDELEY_PATH)

combined = pd.read_csv(COMBINED_PATH)

print("\n========== DATASET OVERVIEW ==========")

print("UCI Dataset:", uci.shape)
print("Mendeley Dataset:", mendeley.shape)
print("Combined Dataset:", combined.shape)

# -----------------------------------------------------
# 3. NORMALIZE LABELS
# -----------------------------------------------------

uci["label"] = uci["label"].str.lower().str.strip()

mendeley["LABEL"] = (
    mendeley["LABEL"].str.lower().str.strip()
)

combined["label"] = (
    combined["label"].str.lower().str.strip()
)

# Convert smishing into spam for binary analysis

mendeley["LABEL"] = mendeley["LABEL"].replace({
    "smishing": "spam"
})

# -----------------------------------------------------
# 4. BASIC DATASET INFORMATION
# -----------------------------------------------------

def dataset_report(name, df, label_column, message_column):

    print(f"\n========== {name} ==========")

    print("Total Records:", len(df))

    print("\nClass Distribution:")
    print(df[label_column].value_counts())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Records:")
    print(df.duplicated(
        subset=[message_column]
    ).sum())

    print("\nMessage Length Statistics:")

    lengths = df[message_column].fillna("").astype(str).str.len()

    print(lengths.describe())


dataset_report(
    "UCI DATASET",
    uci,
    "label",
    "message"
)

dataset_report(
    "MENDELEY DATASET",
    mendeley,
    "LABEL",
    "TEXT"
)

dataset_report(
    "COMBINED DATASET",
    combined,
    "label",
    "message"
)

# -----------------------------------------------------
# 5. FINAL CLASS DISTRIBUTION
# -----------------------------------------------------

class_counts = combined["label"].value_counts()

print("\n========== FINAL CLASS DISTRIBUTION ==========")

print(class_counts)

class_counts.to_csv(
    "outputs/class_distribution.csv"
)

# -----------------------------------------------------
# 6. SOURCE DISTRIBUTION
# -----------------------------------------------------

source_counts = combined["source"].value_counts()

print("\n========== SOURCE DISTRIBUTION ==========")

print(source_counts)

source_counts.to_csv(
    "outputs/source_distribution.csv"
)

# -----------------------------------------------------
# 7. MESSAGE LENGTH ANALYSIS
# -----------------------------------------------------

combined["message_length"] = (
    combined["message"].fillna("").astype(str).str.len()
)

print("\n========== MESSAGE LENGTH ==========")

print(
    combined.groupby("label")["message_length"].describe()
)

# -----------------------------------------------------
# 8. URL, EMAIL AND PHONE ANALYSIS
# -----------------------------------------------------

print("\n========== METADATA ANALYSIS ==========")

for column in ["url", "email", "phone"]:

    if column in combined.columns:

        values = (
            combined[column]
            .fillna("Not available")
            .astype(str)
            .str.lower()
        )

        print(f"\n{column.upper()} Distribution:")

        print(values.value_counts())

# -----------------------------------------------------
# 9. CLASS DISTRIBUTION GRAPH
# -----------------------------------------------------

plt.figure(figsize=(8, 5))

class_counts.plot(
    kind="bar",
    rot=0
)

plt.title("Combined SMS Class Distribution")
plt.xlabel("Message Category")
plt.ylabel("Number of Messages")
plt.tight_layout()

plt.savefig(
    "outputs/class_distribution.png",
    dpi=300
)

plt.close()

# -----------------------------------------------------
# 10. SOURCE DISTRIBUTION GRAPH
# -----------------------------------------------------

plt.figure(figsize=(9, 5))

source_counts.plot(
    kind="bar",
    rot=0
)

plt.title("Dataset Source Distribution")
plt.xlabel("Dataset Source")
plt.ylabel("Number of Messages")
plt.tight_layout()

plt.savefig(
    "outputs/source_distribution.png",
    dpi=300
)

plt.close()

# -----------------------------------------------------
# 11. MESSAGE LENGTH GRAPH
# -----------------------------------------------------

plt.figure(figsize=(9, 5))

for label in combined["label"].unique():

    data = combined.loc[
        combined["label"] == label,
        "message_length"
    ]

    plt.hist(
        data,
        bins=40,
        alpha=0.5,
        label=label
    )

plt.title("Message Length Distribution")
plt.xlabel("Message Length (Characters)")
plt.ylabel("Frequency")
plt.legend()

plt.tight_layout()

plt.savefig(
    "outputs/message_length.png",
    dpi=300
)

plt.close()

# -----------------------------------------------------
# 12. SAVE ANALYSIS SUMMARY
# -----------------------------------------------------

summary = pd.DataFrame({
    "Dataset": [
        "UCI",
        "Mendeley",
        "Combined"
    ],
    "Total Records": [
        len(uci),
        len(mendeley),
        len(combined)
    ]
})

summary.to_csv(
    "outputs/dataset_summary.csv",
    index=False
)


# PHISHING PATTERN INVESTIGATION

import pandas as pd

df = pd.read_csv(
    "dataset/combined/combined_sms.csv"
)

keywords = [
    "card blocked",
    "click the link",
    "unlock your card",
    "account suspended",
    "verify your account",
    "credit card",
    "kyc expired",
    "reactivate"
]

for keyword in keywords:

    print("\n==========", keyword, "==========")

    results = df[
        df["message"].str.contains(
            keyword,
            case=False,
            regex=False,
            na=False
        )
    ]

    if len(results) == 0:
        print("No matching messages found.")
    else:
        print(
            results[["label", "message"]]
            .head(10)
            .to_string(index=False)
        )

        print("\nLabel Distribution:")
        print(results["label"].value_counts())

# -----------------------------------------------------
# 13. FINAL STATUS
# -----------------------------------------------------

print("\n========================================")
print("EXPLORATORY DATA ANALYSIS COMPLETED!")
print("========================================")

print("\nGenerated Files:")

print("1. class_distribution.png")
print("2. source_distribution.png")
print("3. message_length.png")
print("4. class_distribution.csv")
print("5. source_distribution.csv")
print("6. dataset_summary.csv")

print("\nAll analysis files saved in outputs/")

# DETAILED PHISHING KEYWORD INVESTIGATION

import pandas as pd

df = pd.read_csv("dataset/combined/combined_sms.csv")

keywords = [
    "card",
    "blocked",
    "unlock",
    "click",
    "link",
    "account",
    "verify",
    "bank",
    "urgent",
    "suspended",
    "password",
    "security",
    "kyc",
    "update",
    "confirm"
]

for keyword in keywords:
    results = df[
        df["message"].str.contains(
            keyword, case=False, regex=False, na=False
        )
    ]

    print(f"\n{'='*45}")
    print(f"KEYWORD: {keyword}")
    print(f"TOTAL: {len(results)}")
    print("LABEL DISTRIBUTION:")
    print(results["label"].value_counts().to_string())

    print("\nSPAM EXAMPLES:")
    spam_examples = results[results["label"] == "spam"]["message"].head(3)
    for msg in spam_examples:
        print("-", msg)

    print("\nHAM EXAMPLES:")
    ham_examples = results[results["label"] == "ham"]["message"].head(2)
    for msg in ham_examples:
        print("-", msg)
