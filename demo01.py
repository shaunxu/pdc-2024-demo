# from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import umap
import pandas as pd

df = pd.read_csv("data/pc-manual.csv", skipinitialspace=True, usecols=["chunk_vector", "chunk_content"])
vectors = []
for s in df["chunk_vector"]:
    vector = []
    for e in s.split(","):
        vector.append(float(e))
    vectors.append(vector)

reducer = umap.UMAP(random_state=0, transform_seed=0)
umap_vectors = reducer.fit_transform(vectors)
print(umap_vectors)

plt.figure()
plt.scatter(umap_vectors[:, 0], umap_vectors[:, 1], marker=".")
plt.show()
