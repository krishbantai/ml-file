import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Read CSV File
data = pd.read_csv("Student.csv")

# Input and Output
X = data[["Study_Hours", "Attendance", "Assignments", "Previous_Marks"]]
y = data["Pass"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Model
model = LogisticRegression()

# Train Model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy =", accuracy_score(y_test, y_pred))

# Predict New Student
new_student = pd.DataFrame(
    [[6, 80, 6, 65]],
    columns=["Study_Hours", "Attendance", "Assignments", "Previous_Marks"]
)

prediction = model.predict(new_student)
probability = model.predict_proba(new_student)

print("Prediction =", prediction)
print("Probability =", probability)

# Graph
hours = np.linspace(1, 9, 200)

graph_data = pd.DataFrame({
    "Study_Hours": hours,
    "Attendance": np.repeat(data["Attendance"].mean(), 200),
    "Assignments": np.repeat(data["Assignments"].mean(), 200),
    "Previous_Marks": np.repeat(data["Previous_Marks"].mean(), 200)
})

prob = model.predict_proba(graph_data)[:, 1]

plt.figure(figsize=(8, 5))

plt.scatter(
    data["Study_Hours"], data["Pass"],
    color="blue", s=80, label="Actual Data"
)

plt.plot(
    hours, prob,
    color="red", linewidth=3, label="Logistic Curve"
)

plt.xlabel("Study Hours")
plt.ylabel("Probability of Pass")
plt.title("Logistic Regression on Non-Separable Data")
plt.grid(True)
plt.legend()

plt.show()
