import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Project 2: Iris Flower Classifier
# Task 4: Prepare the Data (Load & Split)

# Load the famous Iris dataset
iris = load_iris()

# Features (X) and Target (y)
X = iris.data      # Sepal length, sepal width, petal length, petal width
y = iris.target    # 0 = setosa, 1 = versicolor, 2 = virginica

# Convert to DataFrame for better understanding
df = pd.DataFrame(X, columns=iris.feature_names)
df['Species'] = y
df['Species'] = df['Species'].map({0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'})

# Display basic information
print("=== Iris Dataset Overview ===")
print(f"Total Samples: {X.shape[0]}")
print(f"Features: {iris.feature_names}")
print(f"Target Classes: {iris.target_names}")

print("\n=== First 5 Rows ===")
print(df.head())

print("\n=== Class Distribution ===")
print(df['Species'].value_counts())

# Splitting data into Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("\n=== Data Split Summary ===")
print(f"Training Samples: {X_train.shape[0]}")
print(f"Testing Samples: {X_test.shape[0]}")
print(f"Training Features Shape: {X_train.shape}")