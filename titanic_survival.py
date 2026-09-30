# Titanic Survival Prediction using Logistic Regression

import pandas as pd
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load Titanic dataset
df = sns.load_dataset("titanic")

# Select useful columns
df = df[["survived", "pclass", "sex", "age", "fare"]]

# Remove rows with missing values
df = df.dropna()

# Convert gender into numbers
df["sex"] = df["sex"].map({"male": 0, "female": 1})

# Separate input features and target
X = df[["pclass", "sex", "age", "fare"]]
y = df["survived"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Model training completed!")

# Predict survival
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy * 100, "%")

# Take passenger details from user
pclass = int(input("\nEnter passenger class (1, 2, or 3): "))
sex = input("Enter gender (male/female): ").lower()
age = float(input("Enter age: "))
fare = float(input("Enter fare: "))

# Convert gender into number
sex_value = 0 if sex == "male" else 1

# Create DataFrame for new passenger
new_passenger = pd.DataFrame({
    "pclass": [pclass],
    "sex": [sex_value],
    "age": [age],
    "fare": [fare]
})

# Predict survival
prediction = model.predict(new_passenger)

# Display result
if prediction[0] == 1:
    print("Prediction: Passenger may have survived.")
else:
    print("Prediction: Passenger may not have survived.")