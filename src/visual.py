"""
Visualisation module: PCA plots for retrieved patterns.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple

HAS_PLOTTING = False
PCA = None
try:
    from sklearn.decomposition import PCA
    HAS_PLOTTING = True
except ImportError:
    pass

def compute_pca(points: np.ndarray, n_components: int = 2) -> np.ndarray:
    if not HAS_PLOTTING:
        raise RuntimeError('Plotting/PCA not available.')
    pca = PCA(n_components=n_components)
    return pca.fit_transform(points)

def retrieve_indices(fraud_vectors: np.ndarray, query: np.ndarray, top_k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    q = query / (np.linalg.norm(query) + 1e-8)
    F = fraud_vectors.copy()
    F_norm = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-8)
    similarities = F_norm.dot(q)
    top_idx = np.argsort(similarities)[-top_k:][::-1]
    return top_idx, similarities[top_idx]

def generate_sample_plot(i: int, X: np.ndarray, fraud_vectors: np.ndarray, fraud_texts: List[str], plot_dir: str = 'plots') -> Optional[str]:
    if not HAS_PLOTTING or len(fraud_vectors) <= 1:
        return None
    try:
        points_2d = compute_pca(fraud_vectors)
        top_idx, top_similarities = retrieve_indices(fraud_vectors, X[i], top_k=5)
        fig, ax = plt.subplots(figsize=(6,4))
        ax.scatter(points_2d[:,0], points_2d[:,1], c='lightgray', s=10, alpha=0.6)
        colours = ['red','orange','blue','green','purple']
        for rank, idx in enumerate(top_idx):
            ax.scatter(points_2d[idx,0], points_2d[idx,1], c=colours[rank%len(colours)], s=80, edgecolor='k')
            text = fraud_texts[idx] if idx < len(fraud_texts) else f'Pattern {idx}'
            short = text if len(text) < 60 else text[:57] + '...'
            ax.annotate(f"#{rank+1}: {short}\n(sim={top_similarities[rank]:.2f})", 
                    (points_2d[idx,0], points_2d[idx,1]), xytext=(5,5), textcoords='offset points', fontsize=8)
        ax.set_title(f'Sample {i} — Retrieved fraud patterns (top 5)')
        ax.set_xlabel('PCA 1')
        ax.set_ylabel('PCA 2')
        os.makedirs(plot_dir, exist_ok=True)
        path = os.path.join(plot_dir, f'sample_{i}_retrieved.png')
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path
    except Exception as e:
        print(f'Plot generation failed for sample {i}: {e}')
        return None