
# ==========================================================
# SMS SPAM & PHISHING DETECTION SYSTEM
# STEP 4: STREAMLIT WEB APPLICATION
# UPDATED VERSION WITH MODEL DEBUGGING
# ==========================================================

import streamlit as st
import joblib
import re
import os
import numpy as np

# ----------------------------------------------------------
# 1. PAGE CONFIGURATION
# ----------------------------------------------------------

st.set_page_config(
    page_title="SMS Spam & Phishing Detector",
    page_icon="📩",
    layout="centered"
)

# ----------------------------------------------------------
# 2. TEXT PREPROCESSING
# IMPORTANT: Keep consistent with training preprocessing
# ----------------------------------------------------------

def clean_text(text):

    text = str(text).lower()

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

    # Replace numbers
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


# ----------------------------------------------------------
# 3. LOAD TRAINED MODEL
# ----------------------------------------------------------

MODEL_PATH = "model/spam_detection_model_combined.pkl"


@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):
        return None

    return joblib.load(MODEL_PATH)


model_data = load_model()


# ----------------------------------------------------------
# 4. APPLICATION HEADER
# ----------------------------------------------------------

st.title("📩 SMS Spam & Phishing Detector")

st.write(
    """
    An AI/ML based SMS classification system
    using Natural Language Processing (NLP),
    TF-IDF and Machine Learning.
    """
)

st.divider()


# ----------------------------------------------------------
# 5. CHECK MODEL AVAILABILITY
# ----------------------------------------------------------

if model_data is None:

    st.error(
        "Trained model not found! "
        "Please run 03_model_training.py first."
    )

    st.stop()


vectorizer = model_data["vectorizer"]
classifier = model_data["classifier"]

model_name = model_data.get(
    "model_name",
    type(classifier).__name__
)


# Show active model
st.success(f"Active Machine Learning Model: {model_name}")

# Verify model classes
if not hasattr(classifier, "classes_"):

    st.error("Invalid trained classifier.")

    st.stop()


# ----------------------------------------------------------
# 6. USER INPUT
# ----------------------------------------------------------

st.subheader("Enter Your SMS Message")

message = st.text_area(
    "SMS Message",
    placeholder=(
        "Example: Dear customer, your bank card "
        "has been blocked. Click the link to unlock it."
    ),
    height=150
)


# ----------------------------------------------------------
# 7. SMS PREDICTION
# ----------------------------------------------------------

if st.button("🔍 Analyze SMS", type="primary"):

    if not message.strip():

        st.warning("Please enter an SMS message.")

    else:

        # Step 1: Clean message
        cleaned_message = clean_text(message)

        if not cleaned_message:

            st.warning(
                "Message contains no usable text."
            )

        else:

            # Step 2: TF-IDF transformation
            transformed = vectorizer.transform(
                [cleaned_message]
            )

            # Step 3: Check recognized features
            recognized_count = transformed.nnz

            st.divider()

            st.subheader("Prediction Result")

            if recognized_count == 0:

                st.error(
                    "Unable to classify this message reliably."
                )

                st.warning(
                    "No words from this message were recognized "
                    "by the trained TF-IDF vocabulary. "
                    "Please test with another message."
                )

            else:

                # Step 4: Model prediction
                prediction = classifier.predict(
                    transformed
                )[0]

                # Step 5: Calculate decision score
                decision_score = None

                if hasattr(classifier, "decision_function"):

                    decision_score = float(
                        np.asarray(
                            classifier.decision_function(
                                transformed
                            )
                        ).ravel()[0]
                    )

                # Step 6: Display prediction
                if prediction == "spam":

                    st.error(
                        "🚨 SPAM / PHISHING MESSAGE DETECTED"
                    )

                    st.write(
                        "The machine learning model has "
                        "classified this message as suspicious."
                    )

                elif prediction == "ham":

                    st.success(
                        "✅ GENUINE MESSAGE (HAM)"
                    )

                    st.write(
                        "The machine learning model has "
                        "classified this message as genuine."
                    )

                else:

                    st.warning(
                        f"Unknown prediction label: {prediction}"
                    )

                # --------------------------------------------------
                # 8. MODEL ANALYSIS
                # --------------------------------------------------

                st.divider()

                st.subheader("📊 Message Analysis")

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "Recognized Features",
                        recognized_count
                    )

                with col2:

                    st.metric(
                        "Total Vocabulary",
                        len(vectorizer.vocabulary_)
                    )

                if decision_score is not None:

                    st.metric(
                        "SVM Decision Score",
                        f"{decision_score:.4f}"
                    )

                    st.caption(
                        "Decision score is a classification margin, "
                        "not a probability or confidence percentage. "
                        "Its sign is interpreted according to "
                        "the classifier's class ordering."
                    )

                # --------------------------------------------------
                # 9. DEBUGGING INFORMATION
                # --------------------------------------------------

                with st.expander(
                    "🔧 View Model Debugging Information"
                ):

                    st.write("**Active Model:**", model_name)

                    st.write(
                        "**Original Message:**",
                        message
                    )

                    st.write(
                        "**Cleaned Message:**",
                        cleaned_message
                    )

                    st.write(
                        "**Model Classes:**",
                        list(classifier.classes_)
                    )

                    st.write(
                        "**Recognized Features:**",
                        recognized_count
                    )

                    if decision_score is not None:

                        st.write(
                            "**Decision Score:**",
                            decision_score
                        )

                        st.write(
                            "**Class Direction:**",
                            str(classifier.classes_[1])
                            if decision_score > 0
                            else str(classifier.classes_[0])
                        )

                    # Show recognized vocabulary terms
                    feature_names = (
                        vectorizer.get_feature_names_out()
                    )

                    feature_indices = transformed.indices
                    feature_values = transformed.data

                    ranked = np.argsort(
                        feature_values
                    )[::-1][:15]

                    recognized_words = [
                        (
                            feature_names[feature_indices[i]],
                            round(float(feature_values[i]), 4)
                        )
                        for i in ranked
                    ]

                    st.write("**Top Recognized TF-IDF Features:**")

                    if recognized_words:

                        st.dataframe(
                            {
                                "Word / Feature": [
                                    item[0]
                                    for item in recognized_words
                                ],
                                "TF-IDF Weight": [
                                    item[1]
                                    for item in recognized_words
                                ]
                            },
                            use_container_width=True,
                            hide_index=True
                        )

                # --------------------------------------------------
                # 10. IMPORTANT USER NOTICE
                # --------------------------------------------------

                st.info(
                    "AI predictions are not guaranteed to be correct. "
                    "Never share OTP, PIN, CVV or banking passwords "
                    "through suspicious links or messages."
                )


# ----------------------------------------------------------
# 11. FOOTER
# ----------------------------------------------------------

st.divider()

st.caption(
    "Developed using Python, Pandas, NLP, TF-IDF "
    "and Machine Learning."
)

st.caption(
    "SMS Spam & Phishing Detection | Academic AIML Project"
)
