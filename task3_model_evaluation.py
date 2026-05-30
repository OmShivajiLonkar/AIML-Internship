import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Project: Student Performance Predictor
# Task 3: Evaluate the Model and Make Predictions

# Creating the dataset
data = {
    'Hours Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Exam Score': [35, 45, 48, 52, 60, 68, 75, 82, 88, 95]
}

df = pd.DataFrame(data)

# Preparing features and target
X = df[['Hours Studied']]
y = df['Exam Score']

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions on test data
y_pred = model.predict(X_test)

# Evaluating the model
mae = mean_absolute_error(y_test, y_pred)

print("=== Model Evaluation ===")
print(f"Mean Absolute Error: {mae:.2f} points")
print(f"Accuracy is very good (low error)")

# Comparing Actual vs Predicted
results = pd.DataFrame({
    'Actual Score': y_test,
    'Predicted Score': y_pred.round(2)
})
print("\n=== Actual vs Predicted Scores ===")
print(results)

# Prediction for new student
new_hours = [[4.5]]
predicted_score = model.predict(new_hours)

print("\n=== Prediction for New Student ===")
print(f"For 4.5 Hours Studied, Predicted Score = {predicted_score[0]:.2f}")