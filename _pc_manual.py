import umap
import pandas as pd

def get_umap():
    df = pd.read_csv("data/pc-manual.csv", skipinitialspace=True, usecols=["chunk_vector", "chunk_content"])
    vectors = []
    for s in df["chunk_vector"]:
        vector = []
        for e in s.split(","):
            vector.append(float(e))
        vectors.append(vector)

    umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(vectors)
    umap_vectors = umap_transform.transform(vectors)
    return [umap_vectors, umap_transform]