import numpy as np
from tqdm.auto import trange
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import quadratic_assignment
from scipy.stats import skew
from sklearn.cluster import KMeans

import torch
import torch.nn.functional as F

from mini.utils import train_test_split
from utils.eval_utils import compute_topk_accuracy_and_avg_rank
from mini.topology import compute_wasserstein_distance

import wandb

from warnings import filterwarnings
from scipy.optimize import OptimizeWarning
filterwarnings("ignore", category=OptimizeWarning)


def cos_sim_matrix(X, Y, batch_size: int = 8_000):
    """
    Compute cosine similarity matrix between X and Y.
    Uses batching to avoid OOM errors for large matrices.
    """
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
    if isinstance(Y, np.ndarray):
        Y = torch.from_numpy(Y)
    X_norm = X / X.norm(dim=-1, keepdim=True)
    Y_norm = Y / Y.norm(dim=-1, keepdim=True)
    
    if batch_size is None or len(X) <= batch_size:
        return X_norm @ Y_norm.T
    
    # Batched computation
    result = torch.zeros(len(X), len(Y), dtype=X.dtype, device=X.device)
    for i in range(0, len(X), batch_size):
        batch_end = min(i + batch_size, len(X))
        result[i:batch_end] = X_norm[i:batch_end] @ Y_norm.T
    return result

def tensor(x):
    return torch.tensor(x).float()

def N(X, dim=-1, **kwargs):
    return F.normalize(X, dim=dim, **kwargs)

def sim(X, Y):
    X, Y = tensor(X), tensor(Y)
    # center the tensors
    # TODO: centering probably not necessary, and perhaps not implemented correctly, might need transpose somewhere and stuff
    H = torch.eye(len(X), device=X.device) - (1/len(X)) * torch.ones((len(X), len(X)), device=X.device)
    return H @ X @ Y.T @ H

def train_orthogonal_linear(X, Y):
    solution, _ = orthogonal_procrustes(X, Y)
    return tensor(solution)

def eval_score(X_eval, Y_eval, W, backward=False):
    if backward:
        return torch.round(torch.cosine_similarity(X_eval, Y_eval @ W.T, dim=-1).mean(), decimals=2)
    else:
        return torch.round(torch.cosine_similarity(X_eval @ W, Y_eval, dim=-1).mean(), decimals=2)

def aligned_centroids(X_train, Y_train, n_runs=300, n_clusters=50, method='2opt', subsample=None, seed=42):
    options = {'P0': 'randomized', 'maximize': True, 'rng': np.random.default_rng(seed)}
    if subsample is not None:
        X_train, Y_train = X_train[torch.randperm(len(X_train))[:subsample]], Y_train[torch.randperm(len(Y_train))[:subsample]]

    clusterer1 = KMeans(n_clusters=n_clusters)
    clusterer1.fit(X_train)
    clusterer2 = KMeans(n_clusters=n_clusters)
    clusterer2.fit(Y_train)
    centers1, centers2 = clusterer1.cluster_centers_, clusterer2.cluster_centers_
    kernel1 =  sim(centers1, centers1).float()
    kernel2 = sim(centers2, centers2).float()

    quad = None
    # need to re-run the QAP a few times because it's not very good at finding the global optimum (even 2opt)
    for i in range(n_runs):
        new_quad = quadratic_assignment(kernel1, kernel2, method=method, options=options)
        if quad is None or quad.fun < new_quad.fun:
            quad = new_quad
    centers2 = centers2[quad.col_ind] # pyright: ignore[reportOptionalMemberAccess]
    return tensor(centers1), tensor(centers2)


def log_metrics(step_name, X_transf, Y_eval, ground_truth):    
    # Compute accuracy and rank
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    X_transf_dev = X_transf.to(device)
    Y_eval_dev = Y_eval.to(device)
    
    sims = X_transf_dev @ Y_eval_dev.T
    sorted_indices = torch.argsort(sims, dim=1, descending=True)

    acc, avg_rank = compute_topk_accuracy_and_avg_rank(sorted_indices, ground_truth, k=1)
    
    # Compute Wasserstein distance
    wasser_mean, wasser_std = compute_wasserstein_distance(X_transf, Y_eval, n_samples=1000, n_bootstrap=5)
    
    # Cosine similarity
    cos_sim = torch.cosine_similarity(X_transf_dev, Y_eval_dev, dim=-1).mean().item()

    # Hubness
    knn_indices = sorted_indices[:, 0]
    knn_indices_flat = knn_indices.reshape(-1)
    occurrence_counts = torch.bincount(knn_indices_flat, minlength=len(Y_eval)).float()
    occurrence_counts_np = occurrence_counts.cpu().numpy()
    hubness_skew = skew(occurrence_counts_np)
    share_orphans = (occurrence_counts == 0).float().mean().item()

    wandb.log({
        "step_name": step_name,
        "acc_top1": acc,
        "avg_rank": avg_rank,
        "cosine_similarity": cos_sim,
        "wasserstein_mean": wasser_mean,
        "wasserstein_std": wasser_std,
        "hubness_skew": hubness_skew,
        "orphans": share_orphans,
    })
    print(f"[{step_name}] Acc@1: {acc:.4f}, Rank: {avg_rank:.2f}, CosSim: {cos_sim:.4f}, WD: {wasser_mean:.4f}, HubSkew: {hubness_skew:.4f}, Orphans: {share_orphans:.4f}")


def main(ds, model1, model2, num_train, num_test, run_name, ds2=None, source_1_ratio = 0.5, k=50, num_procrustes_iters=100, num_cluster_iters=2, num_clusters=500, seed=42):
    run = wandb.init(
        project="unsupervised_disc",
        name=run_name,
        config={
            "ds": ds,
            "model1": model1,
            "model2": model2,
            "num_train": num_train,
            "num_test": num_test,
            "k": k,
            "n_procrustes_iters": num_procrustes_iters,
            "num_cluster_iters": num_cluster_iters,
            "seed": seed,
    })

    ds1_model1, ds1_model2 = np.load(f'embeddings/{ds}/{model1}.npy'), np.load(f'embeddings/{ds}/{model2}.npy')
    ds1_model1, ds1_model2 = tensor(ds1_model1), tensor(ds1_model2)

    if ds2 is not None:
        ds2_model1, ds2_model2 = np.load(f'embeddings/{ds2}/{model1}.npy'), np.load(f'embeddings/{ds2}/{model2}.npy')
        ds2_model1, ds2_model2 = tensor(ds2_model1), tensor(ds2_model2)
        ds1_model1 = torch.cat([ds1_model1, ds2_model1], dim=0)
        ds1_model2 = torch.cat([ds1_model2, ds2_model2], dim=0)
    
        X_train, Y_train, X_eval, Y_eval = train_test_split(
            ds1_model1, ds1_model2,
            ds2_model1, ds2_model2,
            num_train_samples=num_train,
            num_test_samples=num_test,
            source1_ratio=source_1_ratio
        )
    else:
        X_train, Y_train, X_eval, Y_eval = train_test_split(
            ds1_model1, ds1_model2,
            num_train_samples=num_train,
            num_test_samples=num_test
        )

    ground_truth = torch.eye(len(X_eval))

    log_metrics("no_mapping", X_eval, Y_eval, ground_truth)

    ## Match anchors
    all_centers1, all_centers2 = [], []
    for _ in trange(30, desc="Matching Anchors"):
        centers1, centers2 = aligned_centroids(X_train, Y_train, subsample=10_000, n_clusters=20, 
                                               n_runs=30, method='2opt', seed=seed)
        all_centers1.append(centers1)
        all_centers2.append(centers2)


    all_centers1 = torch.cat(all_centers1, dim=0)
    all_centers2 = torch.cat(all_centers2, dim=0)

    sim1 = cos_sim_matrix(X_train, all_centers1)
    sim2 = cos_sim_matrix(Y_train, all_centers2)
    sim_similarity = cos_sim_matrix(sim1, sim2, 1024)

    top_similar = sim_similarity.topk(dim=-1, k=k).indices
    coefs =  torch.ones(k) / k # N(1 / (1 + torch.arange(k))**.5, p=1) #
    Y_matched = Y_train[top_similar].transpose(-1, -2) @ coefs

    ## Train linear mapping
    W = train_orthogonal_linear(X_train, Y_matched)
    log_metrics("initial_mapping", X_eval @ W, Y_eval, ground_truth)

    ## Refinement: Iterative Procrustes
    steps = trange(num_procrustes_iters, desc="Iterative Procrustes")
    for _ in steps:
        sample_points = X_train[torch.randperm(len(X_train))[:1000]] # TODO: Prune for hubness?
        sample_similarities = cos_sim_matrix(sample_points @ W, Y_train)

        neighbors = sample_similarities.topk(dim=-1, k=k).indices
        sample_matched = Y_train[neighbors].mean(dim=1)

        W_new = train_orthogonal_linear(sample_points, sample_matched)
        W = 0.5 * W + 0.5 * W_new
        score = eval_score(X_eval, Y_eval, W)
        steps.set_postfix({'Eval score': score.item()})

    log_metrics("refinement_procrustes", X_eval @ W, Y_eval, ground_truth)

    ## Refinement: Cluster-based Alignment Correction
    for _ in trange(num_cluster_iters, desc="Cluster-based Refinement"):

        kmeans1 = KMeans(n_clusters=num_clusters).fit(X_train)
        centers1 = tensor(kmeans1.cluster_centers_)
        kmeans2 = KMeans(n_clusters=num_clusters, init=centers1 @ W).fit(Y_train) # pyright: ignore[reportArgumentType]
        centers2 = tensor(kmeans2.cluster_centers_)

        W_new = train_orthogonal_linear(centers1, centers2)
        W = 0.5 * W + 0.5 * W_new

    log_metrics("refinement_cluster", X_eval @ W, Y_eval, ground_truth)
    run.finish()

    return W, X_eval, Y_eval

if __name__ == "__main__":
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    run_name = "distro_split_80-20"
    main(ds='nq', ds2='fineweb', model1='gte', model2='e5', num_train=128_000, num_test=16_000, run_name=run_name,
            num_procrustes_iters=200, num_cluster_iters=5, num_clusters=1024, seed=seed)