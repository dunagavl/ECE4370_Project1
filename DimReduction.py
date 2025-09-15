import json
import numpy as np
from matplotlib import pyplot as plt
from PCA import *

# Load the data and analyze
with (open(r'C:\Users\Victor\PycharmProjects\ECE_4370\Project1\DimensionalityReduction.json', 'rt')) as f:
    D = json.load(f)
    X = np.array(D, dtype=float)
    f.close()
pca_data = pca(X)
# Project the vectors back onto the original vector space
new_basis = pca_data.project(X)

# Setup the figure
fig, ax = plt.subplots()
# Extract the vectors of largest variance
plt.scatter(new_basis[:, 0], new_basis[:, 1])
ax.set_aspect(1)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA projection onto first two components")
plt.show()