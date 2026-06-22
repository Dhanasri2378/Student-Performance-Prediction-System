import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load Dataset
df = pd.read_csv("student_data.csv")

# Features and Target
X = df[["studytime", "failures", "absences", "G1", "G2"]]
y = df["G3"]

# Train Model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# UI
st.title("🎓 Student Performance Prediction System")

st.write("Enter Student Details")

studytime = st.number_input("Study Time", min_value=1, max_value=10, value=2)
failures = st.number_input("Failures", min_value=0, max_value=10, value=0)
absences = st.number_input("Absences", min_value=0, max_value=100, value=4)
G1 = st.number_input("G1 Marks", min_value=0, max_value=20, value=15)
G2 = st.number_input("G2 Marks", min_value=0, max_value=20, value=14)

if st.button("Predict Final Grade"):

    new_student = pd.DataFrame({
        "studytime":[studytime],
        "failures":[failures],
        "absences":[absences],
        "G1":[G1],
        "G2":[G2]
    })

    prediction = model.predict(new_student)

    st.success(
        f"Predicted Final Grade (G3): {prediction[0]:.2f}"
    )