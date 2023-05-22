import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Suppress convergence warnings
warnings.filterwarnings("ignore")

# Read x data from CSV file
x_df = pd.read_csv('ex2_x_data.csv', header=None)

# Read y data from CSV file
y_df = pd.read_csv('ex2_y_data.csv', header=None)

# Split data into features and labels
x = x_df.to_numpy()
y = y_df.to_numpy().ravel()

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# Train a logistic regression model for each class (one-vs-all)
classes = np.unique(y)
models = []
for class_label in classes:
    y_train_class = np.where(y_train == class_label, 1, 0)
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train, y_train_class)
    models.append(model)

# Make predictions on the test set
y_pred = np.zeros((len(y_test), len(classes)))
for i, model in enumerate(models):
    y_pred[:, i] = model.predict_proba(x_test)[:, 1]

# Convert probabilities to class labels
y_pred_class = np.argmax(y_pred, axis=1)

# Calculate confusion matrix
conf_mat = confusion_matrix(y_test, y_pred_class)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Confusion Matrix")
plt.show()

# Calculate and plot class-wise precision, recall, and F1 scores
num_classes = len(classes)
precisions = np.zeros(num_classes)
recalls = np.zeros(num_classes)
f1_scores = np.zeros(num_classes)
for i in range(num_classes):
    true_positives = conf_mat[i, i]
    false_positives = np.sum(conf_mat[:, i]) - true_positives
    false_negatives = np.sum(conf_mat[i, :]) - true_positives

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)

    precisions[i] = precision
    recalls[i] = recall
    f1_scores[i] = f1_score

    print("Class {}: Precision = {:.4f}, Recall = {:.4f}, F1 Score = {:.4f}".format(i, precision, recall, f1_score))

plt.figure(figsize=(10, 6))
plt.bar(classes, precisions, label="Precision")
plt.bar(classes, recalls, label="Recall")
plt.bar(classes, f1_scores, label="F1 Score")
plt.xlabel("Class")
plt.ylabel("Score")
plt.title("Class-wise Precision, Recall, and F1 Score")
plt.legend()
plt.show()
