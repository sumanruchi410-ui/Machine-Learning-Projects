# World Happiness Prediction using Linear Regression

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Create a simple World Happiness dataset
data = {
    "GDP": [1.2, 1.0, 0.8, 1.5, 0.7, 1.3, 1.1, 0.9, 1.4, 0.6,
            1.6, 1.0, 0.8, 1.3, 1.2],

    "Family": [1.3, 1.2, 1.0, 1.5, 0.9, 1.4, 1.2, 1.0, 1.4, 0.8,
               1.6, 1.1, 1.0, 1.3, 1.2],

    "Health": [0.9, 0.8, 0.7, 1.0, 0.6, 0.9, 0.8, 0.7, 0.9, 0.6,
               1.0, 0.8, 0.7, 0.9, 0.8],

    "Freedom": [0.7, 0.6, 0.5, 0.8, 0.4, 0.7, 0.6, 0.5, 0.8, 0.4,
                0.9, 0.6, 0.5, 0.7, 0.6],

    "Trust": [0.3, 0.2, 0.2, 0.4, 0.1, 0.3, 0.2, 0.2, 0.4, 0.1,
              0.5, 0.2, 0.2, 0.3, 0.2],

    "Generosity": [0.2, 0.1, 0.2, 0.3, 0.1, 0.2, 0.1, 0.2, 0.3, 0.1,
                   0.3, 0.2, 0.1, 0.2, 0.2],

    "Happiness": [7.2, 6.8, 6.1, 7.8, 5.8, 7.3, 6.7, 6.2, 7.5, 5.5,
                  8.0, 6.5, 6.0, 7.1, 6.6]
}

# Create DataFrame
df = pd.DataFrame(data)

print("World Happiness Dataset:")
print(df)

# Separate input features and target
X = df[["GDP", "Family", "Health", "Freedom", "Trust", "Generosity"]]
y = df["Happiness"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel training completed!")

# Predict test data
y_pred = model.predict(X_test)

# Calculate R2 score
score = r2_score(y_test, y_pred)

print("R2 Score:", score)

# Take user input
print("\nEnter country details:")

gdp = float(input("GDP: "))
family = float(input("Family: "))
health = float(input("Health: "))
freedom = float(input("Freedom: "))
trust = float(input("Trust: "))
generosity = float(input("Generosity: "))

# Create new country
new_country = pd.DataFrame({
    "GDP": [gdp],
    "Family": [family],
    "Health": [health],
    "Freedom": [freedom],
    "Trust": [trust],
    "Generosity": [generosity]
})

# Predict happiness
prediction = model.predict(new_country)

print("\nPredicted Happiness Score:", prediction[0])

# Display graph
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Happiness")
plt.ylabel("Predicted Happiness")
plt.title("Actual vs Predicted Happiness")
plt.show()