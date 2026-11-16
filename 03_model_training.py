
# ==========================================================
# SMS SPAM & PHISHING DETECTION
# STEP 3: FOUR ML MODELS TRAINING & COMPARISON
# ==========================================================

import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.sparse import load_npz

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

# ----------------------------------------------------------
# 1. LOAD PREPROCESSED DATA
# ----------------------------------------------------------

print("\n========== LOADING DATA ==========")

X_train = load_npz("outputs/X_train_combined.npz")
X_test = load_npz("outputs/X_test_combined.npz")

y_train = pd.read_csv(
    "outputs/y_train_combined.csv"
)["label"]

y_test = pd.read_csv(
    "outputs/y_test_combined.csv"
)["label"]

print("Training Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

print("\nTraining Labels:")
print(y_train.value_counts())

print("\nTesting Labels:")
print(y_test.value_counts())

# ----------------------------------------------------------
# 2. DEFINE FOUR ML MODELS
# ----------------------------------------------------------

models = {

    "Naive Bayes": MultinomialNB(),

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Linear SVM": LinearSVC(
        class_weight="balanced",
        random_state=42,
        max_iter=5000
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

results = []
trained_models = {}
predictions_store = {}
scores_store = {}

# ----------------------------------------------------------
# 3. TRAIN AND EVALUATE MODELS
# ----------------------------------------------------------

for name, model in models.items():

    print("\n" + "=" * 55)
    print("TRAINING MODEL:", name)
    print("=" * 55)

    # Train
    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Get continuous scores for ROC-AUC
    if hasattr(model, "predict_proba"):

        spam_index = list(model.classes_).index("spam")

        scores = model.predict_proba(X_test)[:, spam_index]

    else:

        scores = model.decision_function(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        pos_label="spam",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        pos_label="spam",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        pos_label="spam",
        zero_division=0
    )

    auc = roc_auc_score(
        (y_test == "spam").astype(int),
        scores
    )

    # Store results
    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": auc
    })

    trained_models[name] = model
    predictions_store[name] = predictions
    scores_store[name] = scores

    # Classification report
    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("Accuracy:", round(accuracy, 4))
    print("Spam Precision:", round(precision, 4))
    print("Spam Recall:", round(recall, 4))
    print("Spam F1 Score:", round(f1, 4))
    print("ROC AUC:", round(auc, 4))

# ----------------------------------------------------------
# 4. MODEL COMPARISON
# ----------------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
).reset_index(drop=True)

print("\n========== FINAL MODEL COMPARISON ==========")

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)

results_df.to_csv(
    "outputs/model_results_combined.csv",
    index=False
)

# ----------------------------------------------------------
# 5. SELECT BEST MODEL
# ----------------------------------------------------------

best_model_name = results_df.iloc[0]["Model"]

best_model = trained_models[best_model_name]

best_predictions = predictions_store[best_model_name]

best_scores = scores_store[best_model_name]

print("\n" + "=" * 55)
print("BEST MODEL:", best_model_name)
print("Best Spam F1 Score:",
      round(results_df.iloc[0]["F1 Score"], 4))
print("=" * 55)

# ----------------------------------------------------------
# 6. LOAD TF-IDF VECTORIZER
# ----------------------------------------------------------

vectorizer = joblib.load(
    "model/tfidf_combined.pkl"
)

# ----------------------------------------------------------
# 7. SAVE BEST MODEL
# ----------------------------------------------------------

joblib.dump(
    {
        "vectorizer": vectorizer,
        "classifier": best_model,
        "model_name": best_model_name
    },
    "model/spam_detection_model_combined.pkl"
)

print("\nBest model saved successfully!")

# ----------------------------------------------------------
# 8. CONFUSION MATRIX
# ----------------------------------------------------------

cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=["ham", "spam"]
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["HAM", "SPAM"],
    yticklabels=["HAM", "SPAM"]
)

plt.title("Confusion Matrix - " + best_model_name)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix_combined.png",
    dpi=300
)

plt.close()

# ----------------------------------------------------------
# 9. MODEL COMPARISON GRAPH
# ----------------------------------------------------------

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC AUC"
]

results_df.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(12, 6),
    ylim=(0, 1)
)

plt.title("Four ML Models - Performance Comparison")
plt.ylabel("Performance Score")
plt.xlabel("Machine Learning Algorithms")
plt.xticks(rotation=15)
plt.legend(loc="lower right")

plt.tight_layout()

plt.savefig(
    "outputs/model_comparison_combined.png",
    dpi=300
)

plt.close()

# ----------------------------------------------------------
# 10. ROC CURVE COMPARISON
# ----------------------------------------------------------

plt.figure(figsize=(8, 6))

for name, scores in scores_store.items():

    fpr, tpr, thresholds = roc_curve(
        (y_test == "spam").astype(int),
        scores
    )

    auc = roc_auc_score(
        (y_test == "spam").astype(int),
        scores
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC={auc:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Model Comparison")
plt.legend()

plt.tight_layout()

plt.savefig(
    "outputs/roc_curve_combined.png",
    dpi=300
)

plt.close()

# ----------------------------------------------------------
# 11. SAVE ALL TEST PREDICTIONS
# ----------------------------------------------------------

prediction_df = pd.DataFrame({
    "Actual": y_test.values
})

for name in models:

    column_name = name.replace(" ", "_")

    prediction_df[
        column_name + "_Prediction"
    ] = predictions_store[name]

prediction_df.to_csv(
    "outputs/test_predictions_combined.csv",
    index=False
)

# ----------------------------------------------------------
# 12. SAVE INDIVIDUAL CLASSIFICATION REPORTS
# ----------------------------------------------------------

for name in models:

    report = classification_report(
        y_test,
        predictions_store[name],
        zero_division=0
    )

    filename = name.lower().replace(" ", "_")

    with open(
        f"outputs/{filename}_report.txt",
        "w",
        encoding="utf-8"
    ) as file:

        file.write(report)

# ----------------------------------------------------------
# 13. FINAL STATUS
# ----------------------------------------------------------

print("\n" + "=" * 55)
print("ALL FOUR MODELS TRAINED SUCCESSFULLY!")
print("=" * 55)

print("\nBEST MODEL:", best_model_name)

print("\nGENERATED FILES:")
print("1. model_results_combined.csv")
print("2. confusion_matrix_combined.png")
print("3. model_comparison_combined.png")
print("4. roc_curve_combined.png")
print("5. test_predictions_combined.csv")
print("6. spam_detection_model_combined.pkl")

print("\nMODEL TRAINING COMPLETED!")
