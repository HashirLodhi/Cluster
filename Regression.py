from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

import matplotlib.pyplot as plt

# Load the Iris dataset
iris = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
X = iris.drop('species', axis=1).values
y = iris['species'].values

# Encode species labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train Logistic Regression
logreg = LogisticRegression(max_iter=200)
logreg.fit(X, y_encoded)

# Plot decision boundaries (using first two features for visualization)

x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel(), 
                         np.full(xx.ravel().shape, X[:,2].mean()), 
                         np.full(xx.ravel().shape, X[:,3].mean())])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
scatter = plt.scatter(X[:, 0], X[:, 1], c=y_encoded, s=50, cmap='viridis', edgecolor='k')
plt.xlabel('sepal_length')
plt.ylabel('sepal_width')
plt.title("Logistic Regression on Iris Dataset")
plt.legend(handles=scatter.legend_elements()[0], labels=list(le.classes_))
plt.show()

# Interactive prediction on custom data
print("Enter values for a custom Iris flower (sepal_length, sepal_width, petal_length, petal_width):")
try:
    custom = input("Comma-separated (e.g., 5.1,3.5,1.4,0.2): ")
    custom_data = [float(x.strip()) for x in custom.split(',')]
    if len(custom_data) != 4:
        print("Please enter exactly 4 values.")
    else:
        pred = logreg.predict([custom_data])[0]
        print(f"The custom data point is predicted as: {le.inverse_transform([pred])[0]}")
except Exception as e:
    print("Invalid input:", e)