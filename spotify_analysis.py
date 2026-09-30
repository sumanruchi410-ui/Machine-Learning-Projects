# Spotify Song Analysis and Popularity Prediction

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Create a sample Spotify dataset
data = {
    "Danceability": [0.80, 0.65, 0.90, 0.55, 0.75, 0.60, 0.85, 0.70, 0.50, 0.95,
                     0.78, 0.68, 0.88, 0.58, 0.72],

    "Energy": [0.85, 0.70, 0.95, 0.60, 0.80, 0.65, 0.90, 0.75, 0.55, 0.98,
               0.82, 0.72, 0.92, 0.62, 0.78],

    "Valence": [0.75, 0.60, 0.85, 0.50, 0.70, 0.55, 0.80, 0.65, 0.45, 0.90,
                0.78, 0.62, 0.88, 0.52, 0.68],

    "Tempo": [120, 110, 130, 100, 125, 105, 128, 115, 95, 135,
              122, 112, 132, 102, 118],

    "Duration": [210, 190, 220, 180, 205, 195, 215, 200, 175, 225,
                 212, 192, 218, 185, 202],

    "Popularity": [85, 70, 92, 55, 80, 62, 88, 73, 48, 95,
                   87, 68, 91, 58, 76]
}

# Create DataFrame
df = pd.DataFrame(data)

# Display the dataset
print("Spotify Dataset:")
print(df)

# Display basic information
print("\nDataset shape:")
print(df.shape)

print("\nDataset information:")
print(df.info())

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Separate input features and target
X = df[[
    "Danceability",
    "Energy",
    "Valence",
    "Tempo",
    "Duration"
]]

y = df["Popularity"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

# Create and train the model
model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel training completed!")

# Predict popularity
y_pred = model.predict(X_test)

# Calculate R2 score
score = r2_score(y_test, y_pred)

print("R2 Score:", score)

# Take song details from user
print("\nEnter song details:")

danceability = float(input("Danceability (0-1): "))
energy = float(input("Energy (0-1): "))
valence = float(input("Valence (0-1): "))
tempo = float(input("Tempo: "))
duration = float(input("Duration in seconds: "))

# Create new song DataFrame
new_song = pd.DataFrame({
    "Danceability": [danceability],
    "Energy": [energy],
    "Valence": [valence],
    "Tempo": [tempo],
    "Duration": [duration]
})

# Predict song popularity
prediction = model.predict(new_song)

print("\nPredicted Song Popularity:", prediction[0])

# Create visualization
plt.scatter(y_test, y_pred)

plt.xlabel("Actual Popularity")
plt.ylabel("Predicted Popularity")
plt.title("Actual vs Predicted Spotify Popularity")

plt.show()