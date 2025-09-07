import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time

# =========================
# Page Config & Theme
# =========================
st.set_page_config(
    page_title="🎓 Learning Style Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.stButton>button {
    background-color: #4CAF50;
    color: white;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Load Data
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("student_learning_data.csv")

df = load_data()

# =========================
# Train or Load Model
# =========================
@st.cache_resource
def train_model(df):
    X = df.drop(['student_id', 'learning_style'], axis=1)
    y = df['learning_style']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y_encoded)

    joblib.dump(clf, "model.pkl")
    joblib.dump(le, "label_encoder.pkl")
    return clf, le

try:
    clf = joblib.load("model.pkl")
    le = joblib.load("label_encoder.pkl")
except FileNotFoundError:
    clf, le = train_model(df)

# =========================
# Evaluate Model
# =========================
def evaluate_model(df, clf, le):
    X = df.drop(['student_id', 'learning_style'], axis=1)
    y = le.transform(df['learning_style'])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    return report, y_test, y_pred

report, y_test, y_pred = evaluate_model(df, clf, le)

# =========================
# Sidebar Inputs
# =========================
st.sidebar.header("🎛️ Student Input Simulator")

# Session State Defaults
for key, default in zip(["video","reading","practice","quizzes","score"], [30,30,30,10,75]):
    if key not in st.session_state:
        st.session_state[key] = default

if st.sidebar.button("🎲 Generate Random Student"):
    st.session_state["video"] = np.random.randint(0, 100)
    st.session_state["reading"] = np.random.randint(0, 100)
    st.session_state["practice"] = np.random.randint(0, 100)
    st.session_state["quizzes"] = np.random.randint(1, 20)
    st.session_state["score"] = np.random.randint(50, 100)

video = st.sidebar.slider("🎥 Time on Video (min)", 0, 100, st.session_state["video"])
reading = st.sidebar.slider("📖 Time on Reading (min)", 0, 100, st.session_state["reading"])
practice = st.sidebar.slider("📝 Time on Practice (min)", 0, 100, st.session_state["practice"])
quizzes = st.sidebar.slider("🧾 Number of Quizzes Attempted", 0, 20, st.session_state["quizzes"])
score = st.sidebar.slider("📊 Average Quiz Score (%)", 0, 100, st.session_state["score"])

# =========================
# Main Layout
# =========================
st.title("🎓 Learning Style Classifier")
st.write("Predict a student's learning style and get personalized recommendations.")

col1, col2 = st.columns([2, 3])

# -------------------------
# Single Prediction
# -------------------------
with col1:
    st.subheader("🔮 Predict Individual Student Style")
    if st.button("Predict Learning Style"):
        input_data = pd.DataFrame([{
            'time_on_video': video,
            'time_on_reading': reading,
            'time_on_practice': practice,
            'num_quizzes_attempted': quizzes,
            'avg_quiz_score': score
        }])
        prediction = clf.predict(input_data)[0]
        label = le.inverse_transform([prediction])[0]
        probs = clf.predict_proba(input_data)[0]

        st.success(f"🧠 Predicted Learning Style: **{label}**")
        st.bar_chart(pd.DataFrame([probs], columns=le.classes_))

        if label == "Visual":
            st.info("📺 Recommendation: Focus on video lectures, visual diagrams, and animations.")
        elif label == "Textual":
            st.info("📖 Recommendation: Read more textbooks, lecture notes, and articles.")
        else:
            st.info("📝 Recommendation: Solve practice problems, quizzes, and hands-on tasks.")

# -------------------------
# Model Evaluation
# -------------------------
with col2:
    st.subheader("📊 Model Evaluation")
    st.json(report)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# -------------------------
# Batch Prediction
# -------------------------
st.subheader("📂 Batch Prediction from CSV")
uploaded_file = st.file_uploader("Upload CSV with student data", type="csv")

if uploaded_file:
    new_data = pd.read_csv(uploaded_file)
    st.write("Preview:", new_data.head())
    required_features = ['time_on_video','time_on_reading','time_on_practice','num_quizzes_attempted','avg_quiz_score']
    
    if set(required_features).issubset(new_data.columns):
        with st.spinner("Processing batch predictions..."):
            time.sleep(1)
            predictions = clf.predict(new_data[required_features])
            new_data["Predicted Learning Style"] = le.inverse_transform(predictions)
            st.success("✅ Predictions Completed")
            st.dataframe(new_data)
            csv = new_data.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Predictions as CSV", csv, "predictions.csv", "text/csv")
    else:
        st.error(f"Missing columns! Required: {required_features}")

# -------------------------
# Feature Importance
# -------------------------
with st.expander("🔍 Feature Importance"):
    feat_df = pd.DataFrame({
        'Feature': df.drop(['student_id','learning_style'], axis=1).columns,
        'Importance': clf.feature_importances_
    }).sort_values(by="Importance", ascending=True)
    st.bar_chart(feat_df.set_index("Feature"))

# -------------------------
# Learning Style Distribution
# -------------------------
with st.expander("📊 Learning Style Distribution"):
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='learning_style', ax=ax)
    ax.set_title("Distribution of Learning Styles")
    st.pyplot(fig)

# -------------------------
# Interactive Feature Sensitivity
# -------------------------
st.subheader("⚙️ Explore Feature Impact")

feature_to_adjust = st.selectbox(
    "Select a feature to vary",
    ['time_on_video','time_on_reading','time_on_practice','num_quizzes_attempted','avg_quiz_score']
)

min_val = int(df[feature_to_adjust].min())
max_val = int(df[feature_to_adjust].max())

adjust_values = st.slider(
    f"Adjust {feature_to_adjust}",
    min_val, max_val, (min_val, max_val)
)

simulate_df = pd.DataFrame([{feature_to_adjust: val,
                             **{f: st.session_state.get(f, df[f].mean()) for f in df.drop(['student_id','learning_style',feature_to_adjust], axis=1).columns}}
                            for val in range(adjust_values[0], adjust_values[1]+1, max(1,(adjust_values[1]-adjust_values[0])//20))])

prob_matrix = clf.predict_proba(simulate_df)
prob_df = pd.DataFrame(prob_matrix, columns=le.classes_)
prob_df[feature_to_adjust] = simulate_df[feature_to_adjust]

st.line_chart(prob_df.set_index(feature_to_adjust))

# -------------------------
# Multi-Student "What-If" Simulator
# -------------------------
st.subheader("🧪 Compare Multiple Students")
num_students = st.number_input("Number of hypothetical students", 1, 5, 2)

multi_data = []
for i in range(num_students):
    st.markdown(f"**Student {i+1}**")
    student_dict = {}
    for f in ['time_on_video','time_on_reading','time_on_practice','num_quizzes_attempted','avg_quiz_score']:
        student_dict[f] = st.slider(f"{f} (Student {i+1})", int(df[f].min()), int(df[f].max()), int(df[f].mean()), key=f"{f}_{i}")
    multi_data.append(student_dict)

if st.button("Compare Students"):
    multi_df = pd.DataFrame(multi_data)
    preds = clf.predict(multi_df)
    multi_df['Predicted Learning Style'] = le.inverse_transform(preds)
    st.dataframe(multi_df)

# -------------------------
# Feedback
# -------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("💬 Feedback")
feedback = st.sidebar.text_area("Share your feedback or suggestions:")
if st.sidebar.button("Submit Feedback"):
    st.sidebar.success("Thank you for your feedback!")
