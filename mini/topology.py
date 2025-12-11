import torch
import numpy as np
from scipy.linalg import svd
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA
from ripser import ripser
from persim import wasserstein


def compute_cca(X, Y, k=None):
    
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


def compute_pwcca(X, Y, n_components=None):
    """
    Compute Projection Weighted CCA.
    
    Weights canonical correlations by how much variance each canonical
    direction captures in the original X space.
    
    Args:
        X: First set of embeddings (n_samples, n_features)
        Y: Second set of embeddings (n_samples, n_features)  
        n_components: Number of CCA components to use
    
    Returns:
        Weighted mean of canonical correlations
    """
    if isinstance(X, torch.Tensor):
        X_np = X.clone().numpy()
    else:
        X_np = X.copy()
    if isinstance(Y, torch.Tensor):
        Y_np = Y.clone().numpy()
    else:
        Y_np = Y.copy()
    
    # Center the data
    X_centered = X_np - X_np.mean(axis=0, keepdims=True)
    Y_centered = Y_np - Y_np.mean(axis=0, keepdims=True)
    
    # Limit components
    max_components = min(X_np.shape[0] - 1, X_np.shape[1], Y_np.shape[1])
    if n_components is None:
        n_components = min(50, max_components)
    else:
        n_components = min(n_components, max_components)
    
    # Fit CCA
    cca = CCA(n_components=n_components)
    cca.fit(X_centered, Y_centered)
    X_c, Y_c = cca.transform(X_centered, Y_centered)
    
    # Get CCA directions for X (x_weights_ are the projection directions)
    # Shape: (n_features, n_components)
    W_x = cca.x_weights_
    
    # Compute how much of X's variance each canonical direction captures
    # Project X onto each canonical direction and compute variance
    # X_c[:, i] = X_centered @ W_x[:, i], so variance is sum of squared projections
    component_variances = np.var(X_c, axis=0)  # Variance per component
    
    # Compute canonical correlations
    corrs = []
    for i in range(n_components):
        corr = np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]
        if np.isnan(corr):
            corr = 0.0
        corrs.append(corr)
    corrs = np.array(corrs)
    
    # Weight correlations by variance captured
    weights = component_variances / (component_variances.sum() + 1e-10)
    weighted_corr = np.sum(weights * corrs)
    
    return float(weighted_corr)


def compute_cca_metrics(X, Y, k=None):
    """
    Compute CCA and PWCCA in a single pass.
    Returns a dictionary with 'mean_cca' and 'pwcca'.
    """
    if isinstance(X, torch.Tensor):
        X = X.numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.numpy()
    
    # Center the data
    X_c = X - X.mean(axis=0)
    Y_c = Y - Y.mean(axis=0)
    
    # Compute covariance matrices
    Cxx = X_c.T @ X_c / (X.shape[0] - 1)
    Cyy = Y_c.T @ Y_c / (Y.shape[0] - 1)
    Cxy = X_c.T @ Y_c / (X.shape[0] - 1)

    # Compute whitening transforms
    Ux, Sx, _ = svd(Cxx)
    Uy, Sy, _ = svd(Cyy)
    
    Wx = Ux @ np.diag(1.0 / np.sqrt(Sx + 1e-10))
    Wy = Uy @ np.diag(1.0 / np.sqrt(Sy + 1e-10))

    # Compute canonical correlations
    T = Wx.T @ Cxy @ Wy
    U, S, Vt = svd(T)
    
    corrs = S if k is None else S[:k]
    
    # Compute PWCCA weights
    # Canonical weights for X
    A = Wx @ U
    # Canonical variates for X
    H = X_c @ A
    
    # Compute correlations between H and X
    # Normalize columns of H and X
    H_norm = H / (np.linalg.norm(H, axis=0, keepdims=True) + 1e-10)
    X_norm = X_c / (np.linalg.norm(X_c, axis=0, keepdims=True) + 1e-10)
    
    # Correlation matrix (n_components, n_features)
    correlations = H_norm.T @ X_norm
    
    # Sum of absolute correlations
    alpha = np.sum(np.abs(correlations), axis=1)
    
    # Weighted average
    pwcca = np.sum(alpha * S) / np.sum(alpha)
    
    return {
        "mean_cca": np.mean(corrs),
        "pwcca": pwcca
    }


def compute_persistence_diagram(X: torch.Tensor, 
                                max_dim: int = 1, 
                                n_samples: int = 1000, 
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
    if isinstance(X, torch.Tensor):
        X_np = X.detach().cpu().float().numpy()
    else:
        X_np = np.array(X, dtype=np.float32, copy=False)

    if np.isnan(X_np).any() or np.isinf(X_np).any():
        raise ValueError("Input to compute_persistence_diagram contains NaNs or Infs")

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
                                  dim: int = 1, n_samples: int = 500,
                                  seed: int | None = 42,
                                  n_bootstrap: int = 1) -> tuple[float, float]:
    """
    Compute the 2-Wasserstein distance between persistence diagrams of two point clouds.
    
    Args:
        X: First point cloud tensor
        Y: Second point cloud tensor  
        dim: Homology dimension to compare (0 = connected components, 1 = loops/cycles)
        n_samples: Number of points to subsample for persistence computation
        seed: Random seed for reproducible subsampling (None for random each time)
        n_bootstrap: Number of bootstrap iterations to average over. Higher values
                     reduce variance without increasing per-iteration memory/time.
        return_std: If True, also return the standard deviation across bootstrap samples.
    
    Returns:
        2-Wasserstein distance between the persistence diagrams (mean over bootstrap samples).
        If return_std=True, returns (mean, std) tuple.
    """
    # Pre-convert to numpy to avoid repeated overhead in bootstrap loop
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().float().numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.detach().cpu().float().numpy()

    distances = []
    
    for i in range(n_bootstrap):
        # Use different seed for each bootstrap iteration
        iter_seed = None if seed is None else seed + i
        
        if iter_seed is not None:
            np.random.seed(iter_seed)
        
        # Compute persistence diagrams for both point clouds
        dgms_X = compute_persistence_diagram(X, max_dim=dim, n_samples=n_samples)
        
        if iter_seed is not None:
            np.random.seed(iter_seed + 1000)  # Different subsample for Y
        
        dgms_Y = compute_persistence_diagram(Y, max_dim=dim, n_samples=n_samples)
        # Get the diagrams for the specified dimension
        dgm_X = dgms_X[dim]
        dgm_Y = dgms_Y[dim]
        
        # Remove infinite persistence points (they correspond to essential features)
        dgm_X = dgm_X[np.isfinite(dgm_X).all(axis=1)]
        dgm_Y = dgm_Y[np.isfinite(dgm_Y).all(axis=1)]

        # Compute 2-Wasserstein distance
        dist = wasserstein(dgm_X, dgm_Y, matching=False)
        distances.append(dist)
    
    mean_dist = float(np.mean(distances))
    std_dist = float(np.std(distances))
    return mean_dist, std_dist