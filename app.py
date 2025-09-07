import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def load_data():
    return pd.read_csv("student_learning_data.csv")

@st.cache_resource
def train_model(df):
    X = df.drop(['student_id', 'learning_style'], axis=1)
    y = df['learning_style']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y_encoded)
    return clf, le

st.title("🎓 Learning Style Classifier")
st.write("Enter student quiz interaction data to predict their learning style.")

df = load_data()
clf, le = train_model(df)

video = st.slider("Time on Video (min)", 0, 100, 30)
reading = st.slider("Time on Reading (min)", 0, 100, 30)
practice = st.slider("Time on Practice (min)", 0, 100, 30)
quizzes = st.slider("Number of Quizzes Attempted", 0, 20, 10)
score = st.slider("Average Quiz Score (%)", 0, 100, 75)

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
    st.success(f"🧠 Predicted Learning Style: **{label}**")

with st.expander("📊 Show Learning Style Distribution"):
    st.dataframe(df.head())
    chart = sns.countplot(data=df, x='learning_style')
    st.pyplot(plt.gcf())
    plt.clf()
