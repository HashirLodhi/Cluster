from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the Iris dataset
iris = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
X = iris.drop('species', axis=1).values
y = iris['species'].values

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# Map clusters to species names
cluster_to_species = {}
for i in range(3):
    labels, counts = np.unique(y[y_kmeans == i], return_counts=True)
    cluster_to_species[i] = labels[np.argmax(counts)]

# Plot the clusters (using first two features for visualization)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.xlabel('sepal_length')
plt.ylabel('sepal_width')
plt.title("K-Means Clustering on Iris Dataset")
plt.show()

# Interactive prediction on custom data
print("Enter values for a custom Iris flower (sepal_length, sepal_width, petal_length, petal_width):")
try:
    custom = input("Comma-separated (e.g., 5.1,3.5,1.4,0.2): ")
    custom_data = [float(x.strip()) for x in custom.split(',')]
    if len(custom_data) != 4:
        print("Please enter exactly 4 values.")
    else:
        cluster = kmeans.predict([custom_data])[0]
        species_name = cluster_to_species[cluster]
        print(f"The custom data point belongs to cluster: {cluster} (predicted species: {species_name})")
except Exception as e:
    print("Invalid input:", e)