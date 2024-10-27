import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import accuracy_score

# Load data
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Check for class imbalance
unique, counts = np.unique(labels, return_counts=True)
print("Class distribution:", dict(zip(unique, counts)))

# Split data into training, validation, and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Initialize the model with more regularization
model = RandomForestClassifier(max_depth=8, min_samples_split=20, min_samples_leaf=10, n_estimators=100)

# Train the model on the training set
model.fit(x_train, y_train)

# Evaluate on the test set
y_predict = model.predict(x_test)
test_accuracy = accuracy_score(y_predict, y_test)
print('The test accuracy of the model is {:.2f}%'.format(test_accuracy * 100))

# Perform cross-validation on the training set to assess model performance
cv_scores = cross_val_score(model, x_train, y_train, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score: {:.2f}%".format(np.mean(cv_scores) * 100))

# Check feature importances
feature_importances = model.feature_importances_
print("Feature importances:", feature_importances)

# Plot learning curves
train_sizes, train_scores, test_scores = learning_curve(model, x_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 10))

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, label="Training score")
plt.plot(train_sizes, test_scores_mean, label="Validation score")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Learning Curves")
plt.show()

f = open('model.p', 'wb')
pickle.dump({'model': model}, f)
f.close()