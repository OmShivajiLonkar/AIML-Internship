import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

# Project 2: Flower Classifier
# Task 6: Improve Model with Feature Scaling

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train k-NN on scaled data
knn_scaled = KNeighborsClassifier(n_neighbors=3)
knn_scaled.fit(X_train_scaled, y_train)

# Evaluate
y_pred = knn_scaled.predict(X_test_scaled)
accuracy = knn_scaled.score(X_test_scaled, y_test)

print("=== Task 6: k-NN with Feature Scaling ===")
print(f"Accuracy after Scaling: {accuracy * 100:.2f}%")

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))