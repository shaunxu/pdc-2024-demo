import matplotlib.pyplot as plt
from _pc_manual import get_umap

[umap_passage_vectors, umap_transform] = get_umap()

plt.figure()
plt.scatter(umap_passage_vectors[:, 0], umap_passage_vectors[:, 1], marker=".", c="gray", s=10)
plt.show()
