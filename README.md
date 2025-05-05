Stacking classifier


# Importing necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score

# Loading the iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# Defining base learners
base_learners = [
    ('knn', KNeighborsClassifier(n_neighbors=3)),
    ('dt', DecisionTreeClassifier(random_state=1))
]

# Defining meta learner
meta_learner = LogisticRegression(random_state=1)

# Creating the stacking classifier
stc = StackingClassifier(estimators=base_learners, final_estimator=meta_learner)

# Training the model
stc.fit(X_train, y_train)

# Making predictions
y_pred = stc.predict(X_test)

# Evaluating the model
print("Accuracy:", accuracy_score(y_test, y_pred))
