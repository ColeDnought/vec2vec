import torch
import numpy as np
from scipy.linalg import svd
from sklearn.decomposition import PCA
from ripser import ripser
from persim import wasserstein


def center_and_normalize(X):
    X = X - X.mean(dim=0, keepdim=True)
    X = X / (X.norm(dim=1, keepdim=True) + 1e-10)
    return X


def compute_cca(X, Y, k=None):
    # Assume X and Y are (n_samples, n_features)
    X = center_and_normalize(X.clone())
    Y = center_and_normalize(Y.clone())
    
    # Compute covariance matrices
    Cxx = X.T @ X / (X.shape[0] - 1)
    Cyy = Y.T @ Y / (Y.shape[0] - 1)
    Cxy = X.T @ Y / (X.shape[0] - 1)

    # Compute whitening transforms (convert to numpy for SVD, then back to torch)
    Ux, Sx, _ = svd(Cxx.numpy())
    Uy, Sy, _ = svd(Cyy.numpy())
    
    # Convert back to torch tensors
    Ux = torch.from_numpy(Ux).float()
    Uy = torch.from_numpy(Uy).float()
    Sx = torch.from_numpy(Sx).float()
    Sy = torch.from_numpy(Sy).float()
    
    Wx = Ux @ torch.diag(1.0 / torch.sqrt(Sx + 1e-10))
    Wy = Uy @ torch.diag(1.0 / torch.sqrt(Sy + 1e-10))

    # Compute canonical correlations
    T = Wx.T @ Cxy @ Wy
    U, S, Vt = svd(T.numpy())
    return S if k is None else S[:k]


def compute_svcca(X, Y, variance_threshold=0.99):
    def svd_reduce(X):
        U, S, Vt = svd(X.numpy(), full_matrices=False)
        explained = np.cumsum(S**2) / np.sum(S**2)
        rank = np.searchsorted(explained, variance_threshold) + 1
        return torch.tensor(U[:, :rank]) @ torch.diag(torch.tensor(S[:rank]))

    X_red = svd_reduce(X)
    Y_red = svd_reduce(Y)
    return compute_cca(X_red, Y_red)


def compute_pwcca(X, Y):
    # Compute canonical correlation directions and scores
    X = center_and_normalize(X.clone())
    Y = center_and_normalize(Y.clone())
    C = compute_cca(X, Y)

    # Project X onto canonical directions to get weights
    proj_weights = torch.sum(X * X, dim=1)
    proj_weights = proj_weights / proj_weights.sum()
    # C is numpy array from compute_cca, convert to tensor
    C_tensor = torch.from_numpy(C).float()
    weighted_corr = sum(w * c for w, c in zip(proj_weights, C_tensor))
    return weighted_corr.item()


def compute_persistence_diagram(X: torch.Tensor, 
                                max_dim: int = 1, 
                                n_samples: int = 5000, 
                                use_pca: bool = False, 
                                n_components: int = 50
                                ) -> list[np.ndarray]:
    """
    Compute persistence diagrams for a point cloud using Vietoris-Rips filtration.
    
    Args:
        X: Point cloud tensor of shape (n_points, n_features)
        max_dim: Maximum homology dimension to compute (0 = connected components, 1 = loops)
        n_samples: Number of points to subsample (for computational efficiency)
        use_pca: Whether to apply PCA before computing persistence (avoids wide matrix warning)
        n_components: Number of PCA components to reduce to (only used if use_pca=True)
    
    Returns:
        List of persistence diagrams, one per dimension
    """ 
    # Convert to numpy and subsample if needed
    X_np = X.numpy() if isinstance(X, torch.Tensor) else X
    if len(X_np) > n_samples:
        indices = np.random.choice(len(X_np), n_samples, replace=False)
        X_np = X_np[indices]
    
    # Reduce dimensionality if needed to avoid "more columns than rows" warning
    # and to speed up distance computations
    if use_pca and X_np.shape[1] > n_components:
        pca = PCA(n_components=n_components)
        X_np = pca.fit_transform(X_np)
    
    # Compute persistence diagrams
    result = ripser(X_np, maxdim=max_dim)
    return result['dgms']


def compute_wasserstein_distance(X: torch.Tensor, Y: torch.Tensor, 
                                  dim: int = 1, n_samples: int = 500) -> float:
    """
    Compute the 2-Wasserstein distance between persistence diagrams of two point clouds.
    
    Args:
        X: First point cloud tensor
        Y: Second point cloud tensor  
        dim: Homology dimension to compare (0 = connected components, 1 = loops/cycles)
        n_samples: Number of points to subsample for persistence computation
    
    Returns:
        2-Wasserstein distance between the persistence diagrams
    """    
    # Compute persistence diagrams for both point clouds
    dgms_X = compute_persistence_diagram(X, max_dim=dim, n_samples=n_samples)
    dgms_Y = compute_persistence_diagram(Y, max_dim=dim, n_samples=n_samples)
    
    # Get the diagrams for the specified dimension
    dgm_X = dgms_X[dim]
    dgm_Y = dgms_Y[dim]
    
    # Remove infinite persistence points (they correspond to essential features)
    dgm_X = dgm_X[np.isfinite(dgm_X).all(axis=1)]
    dgm_Y = dgm_Y[np.isfinite(dgm_Y).all(axis=1)]
    
    # Compute 2-Wasserstein distance
    return wasserstein(dgm_X, dgm_Y, matching=False)