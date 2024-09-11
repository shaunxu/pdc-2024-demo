from _embedding import Embedding
from _pc_manual import get_umap
import matplotlib.pyplot as plt

def scatter_query_vector(query: str, color: str):
    query_vectors = embedding.embed([query])
    umap_query_vectors = umap_transform.transform(query_vectors)
    plt.scatter(umap_query_vectors[:, 0], umap_query_vectors[:, 1], marker="x", c=color, linewidths=3, s=80)

[umap_passage_vectors, umap_transform] = get_umap()
embedding = Embedding(model_path="/Users/shaunxu/huggingface/bge-m3")

plt.figure()
plt.scatter(umap_passage_vectors[:, 0], umap_passage_vectors[:, 1], marker=".", c="gray", s=10)

scatter_query_vector("如何创建工作项", color="red")
scatter_query_vector("如何创建用户故事", color="red")

scatter_query_vector("知识管理中能否创建思维导图", color="green")
scatter_query_vector("May I create a mindmap in a page", color="green")

scatter_query_vector("页面里面能否引用工作项", color="blue")

plt.show()
