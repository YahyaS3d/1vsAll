import warnings
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning

# Suppress convergence warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Read x data from CSV file
x_df = pd.read_csv('ex2_x_data.csv', header=None)

# Read y data from CSV file
y_df = pd.read_csv('ex2_y_data.csv', header=None)

# Split data into features and labels
x = x_df.to_numpy()
y = y_df.to_numpy().ravel()

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Train a logistic regression model for each class
num_classes = len(np.unique(y))
models = []
for i in range(num_classes):
    y_train_class = (y_train == i).astype(int)
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train_class)
    models.append(model)

# Make predictions on the test set
y_pred = np.zeros((len(y_test), num_classes))
for i, model in enumerate(models):
    y_pred[:, i] = model.predict_proba(x_test)[:, 1]

# Convert probabilities to class labels
y_pred_class = np.argmax(y_pred, axis=1)

# Set a larger value for the max_iter parameter
clf = LogisticRegression(max_iter=1000)

# Fit the model
clf.fit(x_train, y_train)

# Make predictions
y_pred = clf.predict(x_test)

# Print the confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
print("Confusion matrix:\n", conf_mat)

# Calculate accuracy rate
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy rate
print("Accuracy rate:", accuracy)
