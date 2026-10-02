import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# Load the dataset
data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Dataset loaded successfully!")
print(data.head())
print("\nDataset shape:", data.shape)


# Clean the data
data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"],
    errors="coerce"
)

data["Churn"] = data["Churn"].map({
    "Yes": 1,
    "No": 0
})

# Customer ID is not useful for prediction
data.drop("customerID", axis=1, inplace=True)


# Separate features and target
X = data.drop("Churn", axis=1)
y = data["Churn"]


# Find numerical and categorical columns
numeric_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_columns = X.select_dtypes(
    include=["object"]
).columns


# Prepare numerical data
numeric_process = Pipeline([
    ("fill", SimpleImputer(strategy="median"))
])


# Prepare categorical data
categorical_process = Pipeline([
    ("fill", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore"))
])


# Combine both preprocessing steps
preprocess = ColumnTransformer([
    ("numbers", numeric_process, numeric_columns),
    ("categories", categorical_process, categorical_columns)
])


# Create the Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)


# Connect preprocessing and model
final_model = Pipeline([
    ("preprocessing", preprocess),
    ("model", model)
])


# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# Train the model
print("\nTraining the model...")

final_model.fit(X_train, y_train)

print("Model training completed!")


# Make predictions
predictions = final_model.predict(X_test)


# Check accuracy
accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", round(accuracy * 100, 2), "%")


# Show detailed results
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=["No Churn", "Churn"]
    )
)


# Confusion matrix
matrix = confusion_matrix(y_test, predictions)

print("Confusion Matrix:")
print(matrix)

plt.figure(figsize=(6, 5))
plt.imshow(matrix)

plt.title("Customer Churn - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.colorbar()

plt.xticks([0, 1], ["No Churn", "Churn"])
plt.yticks([0, 1], ["No Churn", "Churn"])

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            matrix[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()


# Find important features
trained_model = final_model.named_steps["model"]
preprocessor = final_model.named_steps["preprocessing"]

feature_names = preprocessor.get_feature_names_out()
importance = trained_model.feature_importances_

features = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

features = features.sort_values(
    "Importance",
    ascending=False
)

print("\nMost important features:")
print(features.head(10).to_string(index=False))


# Plot important features
top_features = features.head(10)

plt.figure(figsize=(9, 5))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Features Affecting Customer Churn")

plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()


# SHAP explanation
print("\nCreating SHAP explanation...")

X_test_transformed = preprocessor.transform(X_test)

explainer = shap.TreeExplainer(trained_model)

shap_values = explainer.shap_values(X_test_transformed)

plt.figure()

if isinstance(shap_values, list):
    shap.summary_plot(
        shap_values[1],
        X_test_transformed,
        feature_names=feature_names,
        show=False
    )
else:
    shap.summary_plot(
        shap_values,
        X_test_transformed,
        feature_names=feature_names,
        show=False
    )

plt.title("SHAP - Customer Churn Explanation")
plt.tight_layout()

plt.savefig(
    "shap_summary.png",
    bbox_inches="tight"
)

plt.show()


# Save the trained model
joblib.dump(
    final_model,
    "churn_model.joblib"
)

print("\nModel saved!")
print("Customer churn prediction project completed.")