import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


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
    model = LogisticRegression()
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
from sklearn.metrics import confusion_matrix
# Compute confusion matrix and print results
conf_mat = confusion_matrix(y_test, y_pred) #kxk = matrix up to 10 x 10
print("Confusion matrix:\n", conf_mat)

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# import numpy as np
# from sklearn.metrics import confusion_matrix
#
# # Read x data from file
# x_df = pd.read_csv('ex2_x_data.csv', header=None)
#
# # Read y data from file
# y_df = pd.read_csv('ex2_y_data.csv', header=None)
#
# # Convert dataframes to numpy arrays
# x = x_df.values
# y = y_df.values.flatten() # Make y a 1D array
# # Split data into training and testing sets
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
#
# # Create binary classifiers
# classifiers = []
# k = 10
# for i in range(k):
#     # Create binary target vector for class i
#     y_binary = np.where(y_train == i, 1, 0)
#
#     # Train binary classifier for class i
#     clf = LogisticRegression(random_state=0).fit(x_train, y_binary)
#
#     # Add binary classifier to list of classifiers
#     classifiers.append(clf)
#
# # Predict class labels for test set
# y_pred = np.zeros_like(y_test)
# for i, clf in enumerate(classifiers):
#     # Predict probabilities for class i
#     y_prob = clf.predict_proba(x_test)[:, 1]
#
#     # Update predicted class labels for class i
#     # y_pred[y_prob > threshold] = i
#     y_pred[y_prob > 0.5] = i
#
# # Calculate confusion matrix
# conf_mat = confusion_matrix(y_test, y_pred)
#
# # Print confusion matrix
# print(conf_mat)
