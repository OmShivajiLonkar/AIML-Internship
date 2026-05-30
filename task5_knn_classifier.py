import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Project 2: Flower Classifier
# Task 5: Train a k-Nearest Neighbors (k-NN) Classifier

# Load the Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Check accuracy
accuracy = knn.score(X_test, y_test)

print("=== Task 5: k-NN Classifier ===")
print(f"Test Accuracy: {accuracy * 100:.2f}%")