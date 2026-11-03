# Practical 5: Support Vector Machines (SVM)

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Step 2: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 4: Define SVM parameters
parameters = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

# Step 5: Optimize parameters
grid = GridSearchCV(SVC(), parameters, cv=5)
grid.fit(X_train, y_train)

# Step 6: Select best SVM model
model = grid.best_estimator_

# Step 7: Predict test data
y_pred = model.predict(X_test)

# Step 8: Evaluate model
print("Best Parameters:", grid.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
