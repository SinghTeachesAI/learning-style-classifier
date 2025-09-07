import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Load Data
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("student_learning_data.csv")

# =========================
# Train Model
# =========================
@st.cache_resource
def train_model(df):
    X = df.drop(['student_id', 'learning_style'], axis=1)
    y = df['learning_style']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y_encoded)

    return clf, le

# =========================
# Evaluate Model
# =========================
@st.cache_resource
def evaluate_model(df, clf, le):
    X = df.drop(['student_id', 'learning_style'], axis=1)
    y = le.transform(df['learning_style'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    return report, y_test, y_pred

# =========================
# Streamlit UI
# =========================
st.title("🎓 Learning Style Classifier")
st.write("This app predicts a student's learning style based on quiz interaction data.")

# Load dataset & train model
df = load_data()
clf, le = train_model(df)
report, y_test, y_pred = evaluate_model(df, clf, le)

# =========================
# Random Student Simulator
# =========================
if st.button("🎲 Generate Random Student"):
    st.session_state["video"] = np.random.randint(0, 100)
    st.session_state["reading"] = np.random.randint(0, 100)
    st.session_state["practice"] = np.random.randint(0, 100)
    st.session_state["quizzes"] = np.random.randint(1, 20)
    st.session_state["score"] = np.random.randint(50, 100)

video = st.slider("🎥 Time on Video (min)", 0, 100, st.session_state.get("video", 30))
reading = st.slider("📖 Time on Reading (min)", 0, 100, st.session_state.get("reading", 30))
practice = st.slider("📝 Time on Practice (min)", 0, 100, st.session_state.get("practice", 30))
quizzes = st.slider("🧾 Number of Quizzes Attempted", 0, 20, st.session_state.get("quizzes", 10))
score = st.slider("📊 Average Quiz Score (%)", 0, 100, st.session_state.get("score", 75))

# =========================
# Single Prediction
# =========================
if st.button("🔮 Predict Learning Style"):
    input_data = pd.DataFrame([{
        'time_on_video': video,
        'time_on_reading': reading,
        'time_on_practice': practice,
        'num_quizzes_attempted': quizzes,
        'avg_quiz_score': score
    }])
    prediction = clf.predict(input_data)[0]
    label = le.inverse_transform([prediction])[0]

    st.success(f"🧠 Predicted Learning Style: **{label}**")

    # Personalized recommendation
    if label == "Visual":
        st.info("📺 Recommendation: Focus on video lectures, visual diagrams, and animations.")
    elif label == "Textual":
        st.info("📖 Recommendation: Read more textbooks, lecture notes, and articles.")
    else:
        st.info("📝 Recommendation: Solve practice problems, quizzes, and hands-on tasks.")

# =========================
# Batch Prediction from CSV
# =========================
st.subheader("📂 Batch Prediction from CSV")
uploaded_file = st.file_uploader("Upload CSV with student data", type="csv")

if uploaded_file:
    new_data = pd.read_csv(uploaded_file)
    st.write("Preview:", new_data.head())

    predictions = clf.predict(new_data)
    predicted_labels = le.inverse_transform(predictions)
    new_data["Predicted Learning Style"] = predicted_labels

    st.write("✅ Predictions Completed")
    st.dataframe(new_data)

    csv = new_data.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Predictions as CSV", csv, "predictions.csv", "text/csv")

# =========================
# Model Evaluation
# =========================
with st.expander("📊 Model Evaluation"):
    st.json(report)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# =========================
# Feature Importance
# =========================
with st.expander("🔍 Feature Importance"):
    importances = clf.feature_importances_
    feature_names = df.drop(['student_id', 'learning_style'], axis=1).columns
    feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by="Importance", ascending=False)
    st.bar_chart(feat_df.set_index("Feature"))

# =========================
# Data Preview
# =========================
with st.expander("📊 Show Learning Style Distribution"):
    st.dataframe(df.head())
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='learning_style', ax=ax)
    ax.set_title("Distribution of Learning Styles")
    st.pyplot(fig)
