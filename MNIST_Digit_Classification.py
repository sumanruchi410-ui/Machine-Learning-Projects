# Load and explore handwritten digit dataset

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

# Load the dataset
digits = load_digits()

# Store image data and target values
X = digits.data
y = digits.target

# Display basic information
print("Number of images:", len(X))
print("Image data shape:", X.shape)
print("Target shape:", y.shape)

# Display the first handwritten digit clearly
plt.figure(figsize=(5, 5))
plt.imshow(digits.images[0], cmap="gray", interpolation="nearest")
plt.title("Digit: " + str(y[0]))
plt.axis("off")
plt.show()

# Split data into training and testing sets

from sklearn.model_selection import train_test_split

# Split 80% for training and 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data:", X_train.shape)
print("Testing data:", X_test.shape)

# Train the KNN machine learning model

from sklearn.neighbors import KNeighborsClassifier

# Create the model
model = KNeighborsClassifier(n_neighbors=3)

# Train the model using training data
model.fit(X_train, y_train)

print("Model training completed!")


# Check model accuracy

from sklearn.metrics import accuracy_score

# Predict the digits from test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")

# Predict one digit from the test dataset

index = 10

prediction = model.predict([X_test[index]])

print("Actual digit:", y_test[index])
print("Predicted digit:", prediction[0])

# Test the model on 10 different digits

for i in range(10):
    prediction = model.predict([X_test[i]])

    print("Actual:", y_test[i], "| Predicted:", prediction[0])

