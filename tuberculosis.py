import streamlit as st
import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="TB Classification App", layout="centered")
st.title("🫁 Tuberculosis Classification System")
st.write(
    "Deployable ML app for tuberculosis detection. "
    "Select a model and input data for prediction."
)
st.divider()

# -----------------------------
# Select model
# -----------------------------
model_option = st.selectbox(
    "Select classification model:",
    ["Decision Tree", "Random Forest", "Logistic Regression", "Naive Bayes"]
)

# -----------------------------
# Upload dataset
# -----------------------------
uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------
    # Detect numeric features & target
    # -----------------------------
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < df.shape[1]:
        st.info("Non-numeric columns (IDs, names) removed automatically for training.")

    # Assume last numeric column is target
    X = numeric_df.iloc[:, :-1]
    y = numeric_df.iloc[:, -1]

    # Convert continuous target to binary (0/1)
    if y.nunique() > 2 or y.dtype not in [int, object]:
        st.warning("Target values appear continuous. Converting to binary classification...")
        y = (y > y.median()).astype(int)

    st.info(f"Features used: {list(X.columns)}")
    st.info(f"Target column detected: {y.name}")

    # -----------------------------
    # Train-test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -----------------------------
    # Scaling
    # -----------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -----------------------------
    # Initialize model
    # -----------------------------
    if model_option == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_option == "Random Forest":
        model = RandomForestClassifier(random_state=42)
    elif model_option == "Logistic Regression":
        model = LogisticRegression(max_iter=1000)
    elif model_option == "Naive Bayes":
        model = GaussianNB()

    # -----------------------------
    # Train model
    # -----------------------------
    model.fit(X_train_scaled, y_train)

    # -----------------------------
    # Evaluate
    # -----------------------------
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    st.subheader("Model Evaluation Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{accuracy:.2f}")
        st.metric("Precision", f"{precision:.2f}")
    with col2:
        st.metric("Recall", f"{recall:.2f}")
        st.metric("F1-score", f"{f1:.2f}")

    st.info("F1-score is emphasized for imbalanced datasets.")

    st.divider()
    st.subheader("Prediction")

    input_method = st.radio("Select input method:", ("Manual Input", "Upload CSV"))

    # -----------------------------
    # Manual Input
    # -----------------------------
    if input_method == "Manual Input":
        st.write("Enter numeric feature values:")
        user_input = []
        for col in X.columns:
            value = st.number_input(f"{col}", value=0.0)
            user_input.append(value)

        if st.button("Predict"):
            input_array = np.array(user_input).reshape(1, -1)
            input_scaled = scaler.transform(input_array)
            prediction = model.predict(input_scaled)[0]

            if prediction == 1:
                st.error("🧬 Prediction Result: Tuberculosis")
            else:
                st.success("✅ Prediction Result: Normal")

    # -----------------------------
    # CSV Upload
    # -----------------------------
    else:
        uploaded_pred_file = st.file_uploader(
            "Upload CSV for prediction", type=["csv"], key="pred"
        )
        if uploaded_pred_file is not None:
            input_df = pd.read_csv(uploaded_pred_file)
            numeric_input = input_df.select_dtypes(include=[np.number])
            scaled_input = scaler.transform(numeric_input)
            predictions = model.predict(scaled_input)
            input_df["Prediction"] = ["Tuberculosis" if p == 1 else "Normal" for p in predictions]
            st.success("Prediction completed!")
            st.dataframe(input_df)

st.caption("SECB3203 Mini Project | ML-based Disease Classification using Streamlit")

