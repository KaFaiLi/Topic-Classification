"""
Phase 4: Cluster Representation
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def select_medoid(embeddings: list[list[float]], texts: list[str], labels: list[int]) -> dict[int, str]:
    """
    For each cluster label (excluding -1), select the medoid text.
    Returns mapping: cluster_label -> exemplar text.
    """
    medoids = {}
    embeddings_arr = np.array(embeddings)
    for lbl in set(labels):
        if lbl == -1:
            continue
        indices = [i for i, l in enumerate(labels) if l == lbl]
        sub_emb = embeddings_arr[indices]
        sim = cosine_similarity(sub_emb)
        sums = sim.sum(axis=1)
        medoid_idx = indices[sums.argmax()]
        medoids[lbl] = texts[medoid_idx]
    return medoids

def summarize_cluster(cluster_text: str, summarizer) -> str:
    """
    Summarize cluster_text using a transformers summarization pipeline.
    """
    result = summarizer(cluster_text, max_length=60, min_length=10, do_sample=False)
    return result[0]['summary_text'].strip()
