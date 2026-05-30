import pandas as pd

# Project: Student Performance Predictor
# Task 1: Load and Explore the Data

# Creating the dataset
data = {
    'Hours Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Exam Score': [35, 45, 48, 52, 60, 68, 75, 82, 88, 95]
}

df = pd.DataFrame(data)

# Exploring the data
print("=== First 5 Rows of the Dataset ===")
print(df.head())
print("\n")

print("=== Statistical Summary ===")
print(df.describe())
print("\n")

print("=== Checking for Missing Values ===")
print(df.isnull().sum())
print("\n")

print("=== Correlation Between Hours and Score ===")
print(df.corr())