import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("student_data.csv")

X = df[["studytime","failures","absences","G1","G2"]]
y = df["G3"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = LinearRegression()

model.fit(X_train,y_train)

print("Model Training Completed")
new_student = pd.DataFrame({
    "studytime":[2],
    "failures":[0],
    "absences":[4],
    "G1":[15],
    "G2":[14]
})
prediction = model.predict(new_student)

print("Predicted Final Grade:", prediction[0])
from sklearn.metrics import r2_score

y_pred = model.predict(X_test)

score = r2_score(y_test,y_pred)

print("Accuracy Score:", score)