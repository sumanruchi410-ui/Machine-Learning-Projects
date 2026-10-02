import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


# Load the training data
train_data = pd.read_csv(
    "train_FD001.txt",
    sep=r"\s+",
    header=None
)

# Give names to the columns
columns = [
    "engine",
    "cycle",
    "setting1",
    "setting2",
    "setting3"
]

for i in range(1, 22):
    columns.append("sensor" + str(i))

train_data.columns = columns

print("Training data loaded successfully!")
print("\nFirst 5 rows:")
print(train_data.head())

print("\nDataset shape:")
print(train_data.shape)


# Calculate Remaining Useful Life (RUL)
last_cycle = train_data.groupby("engine")["cycle"].max()

train_data["RUL"] = train_data.apply(
    lambda row: last_cycle[row["engine"]] - row["cycle"],
    axis=1
)

print("\nRUL calculated successfully!")

print("\nSample RUL values:")
print(train_data[["engine", "cycle", "RUL"]].head(10))


# Select the columns used for prediction
features = [
    "cycle",
    "setting1",
    "setting2",
    "setting3"
]

for i in range(1, 22):
    features.append("sensor" + str(i))


X = train_data[features]
y = train_data["RUL"]


# Split the data
X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining rows:", len(X_train))
print("Validation rows:", len(X_valid))


# Create the Random Forest model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


# Train the model
print("\nTraining the model...")

model.fit(X_train, y_train)

print("Model training completed!")


# Make predictions
predictions = model.predict(X_valid)


# Check validation performance
mae = mean_absolute_error(
    y_valid,
    predictions
)

rmse = mean_squared_error(
    y_valid,
    predictions
) ** 0.5

print("\nValidation Results")
print("--------------------")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))


# Load test data
test_data = pd.read_csv(
    "test_FD001.txt",
    sep=r"\s+",
    header=None
)

test_data.columns = columns

print("\nTest data loaded successfully!")
print("Test data shape:", test_data.shape)


# Load the actual RUL values
actual_rul = pd.read_csv(
    "RUL_FD001.txt",
    sep=r"\s+",
    header=None
)

actual_rul = actual_rul[0]

print("\nActual RUL data loaded successfully!")


# Take the last available cycle for each test engine
last_test_data = test_data.groupby(
    "engine"
).tail(1)

last_test_data = last_test_data.sort_values(
    "engine"
)


# Prepare test data
X_test = last_test_data[features]


# Predict RUL for test engines
test_predictions = model.predict(X_test)


# Compare actual and predicted values
results = pd.DataFrame({
    "Engine": last_test_data["engine"].values,
    "Actual RUL": actual_rul.values,
    "Predicted RUL": test_predictions
})

results["Error"] = (
    results["Predicted RUL"] -
    results["Actual RUL"]
)


print("\nRUL Predictions:")
print(results.head(10))


# Calculate test performance
test_mae = mean_absolute_error(
    results["Actual RUL"],
    results["Predicted RUL"]
)

test_rmse = mean_squared_error(
    results["Actual RUL"],
    results["Predicted RUL"]
) ** 0.5


print("\nTest Results")
print("--------------------")
print("MAE:", round(test_mae, 2))
print("RMSE:", round(test_rmse, 2))


# Plot actual vs predicted RUL
plt.figure(figsize=(10, 5))

plt.plot(
    results["Engine"],
    results["Actual RUL"],
    label="Actual RUL"
)

plt.plot(
    results["Engine"],
    results["Predicted RUL"],
    label="Predicted RUL"
)

plt.xlabel("Engine Number")
plt.ylabel("Remaining Useful Life")

plt.title("Actual vs Predicted RUL")

plt.legend()

plt.tight_layout()

plt.savefig("rul_prediction.png")

plt.show()


# Find important features
importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("\nMost important features:")
print(
    importance.head(10).to_string(index=False)
)


# Plot feature importance
top_features = importance.head(10)

plt.figure(figsize=(9, 5))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")

plt.title("Important Features for RUL Prediction")

plt.tight_layout()

plt.savefig("feature_importance.png")

plt.show()


print("\nProject completed successfully!")