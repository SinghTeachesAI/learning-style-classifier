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
    st.session_state_
