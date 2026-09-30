# House Price Prediction using Linear Regression

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Create a larger house dataset
data = {
    "Area": [500, 600, 750, 800, 1000, 1100, 1200, 1250, 1400, 1500,
             1600, 1750, 1800, 2000, 2200],

    "Bedrooms": [1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                 3, 4, 4, 4, 5],

    "Price": [20, 24, 30, 32, 40, 44, 48, 50, 56, 60,
              64, 70, 72, 80, 90]
}

# Create DataFrame
df = pd.DataFrame(data)

print("House Dataset:")
print(df)

# Separate input features and target
X = df[["Area", "Bedrooms"]]
y = df["Price"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# Create and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel training completed!")

# Test the model
y_pred = model.predict(X_test)

# Calculate R2 score
score = r2_score(y_test, y_pred)

print("R2 Score:", score)

# Take house details from user
area = float(input("\nEnter house area in sq. ft: "))
bedrooms = int(input("Enter number of bedrooms: "))

# Create DataFrame for the new house
new_house = pd.DataFrame({
    "Area": [area],
    "Bedrooms": [bedrooms]
})

# Predict house price
prediction = model.predict(new_house)

print("Predicted house price:", prediction[0], "lakhs")

# Plot actual vs predicted prices
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()