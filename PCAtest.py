import numpy as np
import pandas as pd
from PCA_implementation import PCA

df = pd.DataFrame({
    'X1': [1, 5, 3, 0, 4],
    'X2': [7, 3, 5, 8, 4]
})

print("Dataset:")
print(df)

X = df.values

pca = PCA(n_components=2)
pca.fit(X)

print("\nMean:")
print(pca.mean_)
print("\nComponents:")
print(pca.components_)
print("\nEigenvalues:")
print(pca.eigenvalues_)
print("\nTransformed Data:")
print(pca.fit_transform(X))