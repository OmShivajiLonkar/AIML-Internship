import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Project: Student Performance Predictor
# Task 2: Train a Linear Regression Model

# Creating the dataset
data = {
    'Hours Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Exam Score': [35, 45, 48, 52, 60, 68, 75, 82, 88, 95]
}

df = pd.DataFrame(data)

# Preparing X (features) and y (target)
X = df[['Hours Studied']]
y = df['Exam Score']

# Splitting into train and test (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the model
model = LinearRegression()
model.fit(X_train, y_train)

# Printing model parameters
print("=== Linear Regression Model Parameters ===")
print(f"Slope (Coefficient): {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")

# Final Equation
print(f"\nBest Fit Equation: Score = {model.coef_[0]:.2f} * Hours + {model.intercept_:.2f}")