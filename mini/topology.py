import torch
import numpy as np
from scipy.linalg import svd
from scipy.stats import pearsonr


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
