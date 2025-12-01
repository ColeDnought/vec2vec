import os
import torch
import numpy as np
from csv import DictWriter
from tqdm.auto import trange
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import quadratic_assignment
from sklearn.cluster import KMeans
import torch.nn.functional as F
from warnings import filterwarnings
from mini.topology import compute_cca, compute_svcca, compute_pwcca

filterwarnings("ignore", category=FutureWarning)

def cos_sim_matrix(X, Y):
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
    if isinstance(Y, np.ndarray):
        Y = torch.from_numpy(Y)
    X_norm = X / X.norm(dim=-1, keepdim=True)
    Y_norm = Y / Y.norm(dim=-1, keepdim=True)
    return X_norm @ Y_norm.T

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

def rank(X):
    return torch.argsort(torch.argsort(X, dim=-1), dim=-1)

def eval_score(X_eval, Y_eval, W, backward=False):
    if backward:
        return torch.round(torch.cosine_similarity(X_eval, Y_eval @ W.T, dim=-1).mean(), decimals=2).item()
    else:
        return torch.round(torch.cosine_similarity(X_eval @ W, Y_eval, dim=-1).mean(), decimals=2).item()

def report(X_eval, Y_eval, W, top_n) -> dict:
    """
    Reports performance of transformed X.

    Returns:
        dict with accuracy, average rank, cosine similarities, and topological metrics
    """
    X_transformed = X_eval @ W
    
    ranks = rank(cos_sim_matrix(X_transformed, Y_eval)).diagonal()
    acc = (len(X_eval) - 1 == ranks).float().mean().item() # type: ignore
    avg_rank = len(X_eval) - ranks.float().mean().item()
    cossim = top_cosine_similarity(X_transformed, Y_eval, k=top_n)
    
    # Matched cosine similarity: similarity between corresponding pairs
    matched_cossim = torch.cosine_similarity(X_transformed, Y_eval, dim=-1).mean().item()
    
    # Topological analysis
    cca_corrs = compute_cca(X_transformed, Y_eval)
    mean_cca = float(np.mean(cca_corrs))
    
    svcca_corrs = compute_svcca(X_transformed, Y_eval)
    mean_svcca = float(np.mean(svcca_corrs))
    
    pwcca = compute_pwcca(X_transformed, Y_eval)

    print("Top-1 Accuracy:", acc)
    print("Average Rank:", avg_rank)
    print(f"Average cos sim @ top {top_n}:", cossim)
    print("Matched cos sim:", matched_cossim)
    print("Mean CCA:", mean_cca)
    print("Mean SVCCA:", mean_svcca)
    print("PWCCA:", pwcca)

    return {
        'accuracy': acc,
        'avg_rank': avg_rank,
        'cosine_similarity': cossim,
        'matched_cosine_similarity': matched_cossim,
        'mean_cca': mean_cca,
        'mean_svcca': mean_svcca,
        'pwcca': pwcca,
    }

def top_cosine_similarity(X: torch.Tensor, Y: torch.Tensor, k=1) -> float:
    sims = cos_sim_matrix(X, Y)
    topk = sims.topk(k=k, dim=-1).values
    return torch.round(topk.mean(), decimals=2).item()

def train_test_split(
    embeddings_x1: torch.Tensor, embeddings_y1: torch.Tensor,  # Data source 1 (e.g., nq)
    embeddings_x2: torch.Tensor, embeddings_y2: torch.Tensor,  # Data source 2 (e.g., trec)
    num_train_samples: int,
    num_test_samples: int,
    source1_ratio: float = 0.5,  # Ratio of data source 1 in training set only
):
    """
    Splits embeddings from two data sources into training and testing sets.
    - Training sets: Independently shuffled, NO correspondence between X_train and Y_train
                     Mix of both data sources controlled by source1_ratio
    - Test sets: 1:1 matched (same indices used for X and Y), always 50/50 split
    
    Args:
        embeddings_x1, embeddings_y1: First data source (e.g., nq stella/e5)
        embeddings_x2, embeddings_y2: Second data source (e.g., trec stella/e5)
        num_train_samples: Total number of samples for each training set
        num_test_samples: Total number of samples for evaluation (1:1 matched, always 50/50)
        source1_ratio: Proportion of training data from source 1 (0.0 to 1.0)
    """
    assert 0.0 <= source1_ratio <= 1.0, "source1_ratio must be between 0 and 1"
    
    # Calculate samples from each source for training
    num_train_from_source1 = int(num_train_samples * source1_ratio)
    num_train_from_source2 = num_train_samples - num_train_from_source1
    
    # Test set is always 50/50
    num_test_from_source1 = num_test_samples // 2
    num_test_from_source2 = num_test_samples - num_test_from_source1
    
    # Reserve test samples from both sources (1:1 matched - same indices for X and Y)
    assert num_test_from_source1 <= len(embeddings_x1), "Not enough samples in source 1 for test set"
    assert num_test_from_source2 <= len(embeddings_x2), "Not enough samples in source 2 for test set"
    
    X_test_s1 = embeddings_x1[:num_test_from_source1]
    Y_test_s1 = embeddings_y1[:num_test_from_source1]
    X_test_s2 = embeddings_x2[:num_test_from_source2]
    Y_test_s2 = embeddings_y2[:num_test_from_source2]
    
    # Remaining data available for training
    remaining_x1 = embeddings_x1[num_test_from_source1:]
    remaining_y1 = embeddings_y1[num_test_from_source1:]
    remaining_x2 = embeddings_x2[num_test_from_source2:]
    remaining_y2 = embeddings_y2[num_test_from_source2:]
    
    assert num_train_from_source1 <= len(remaining_x1), f"Not enough samples in source 1 for training (need {num_train_from_source1}, have {len(remaining_x1)})"
    assert num_train_from_source2 <= len(remaining_x2), f"Not enough samples in source 2 for training (need {num_train_from_source2}, have {len(remaining_x2)})"
    
    # Sample from source 1 (independently for X and Y)
    indices_x1 = torch.randperm(len(remaining_x1))[:num_train_from_source1]
    indices_y1 = torch.randperm(len(remaining_y1))[:num_train_from_source1]
    X_train_s1 = remaining_x1[indices_x1]
    Y_train_s1 = remaining_y1[indices_y1]
    
    # Sample from source 2 (independently for X and Y)
    indices_x2 = torch.randperm(len(remaining_x2))[:num_train_from_source2]
    indices_y2 = torch.randperm(len(remaining_y2))[:num_train_from_source2]
    X_train_s2 = remaining_x2[indices_x2]
    Y_train_s2 = remaining_y2[indices_y2]
    
    # Combine and shuffle training data
    X_train = torch.cat([X_train_s1, X_train_s2])
    Y_train = torch.cat([Y_train_s1, Y_train_s2])
    shuffle_x = torch.randperm(len(X_train))
    shuffle_y = torch.randperm(len(Y_train))
    X_train = X_train[shuffle_x]
    Y_train = Y_train[shuffle_y]
    
    # Combine test data (maintain 1:1 matching by using same shuffle for X and Y)
    X_test = torch.cat([X_test_s1, X_test_s2])
    Y_test = torch.cat([Y_test_s1, Y_test_s2])
    shuffle_test = torch.randperm(len(X_test))
    X_test = X_test[shuffle_test]
    Y_test = Y_test[shuffle_test]  # Same shuffle to maintain 1:1 correspondence

    return X_train, Y_train, X_test, Y_test

def aligned_centroids(X_train, Y_train, n_runs=300, n_clusters=50, method='2opt', subsample=None, verbose=True):
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
    for i in trange(n_runs, leave=False, disable=not verbose):
        new_quad = quadratic_assignment(kernel1, kernel2, method=method, options={'maximize': True})
        if quad is None or quad.fun < new_quad.fun:
            quad = new_quad
    centers2 = centers2[quad.col_ind] # type: ignore
    return tensor(centers1), tensor(centers2)

def anchor(X_train, Y_train, k: int = 50, anchor_steps: int = 30, subsample: int = 10_000, n_clusters: int = 20, n_runs: int = 30):
    all_centers1, all_centers2 = [], []
    for _ in trange(anchor_steps, desc='Finding anchor clusters'):
        centers1, centers2 = aligned_centroids(X_train, Y_train, subsample=subsample, n_clusters=n_clusters, n_runs=n_runs, method='2opt')
        all_centers1.append(centers1)
        all_centers2.append(centers2)

    all_centers1 = torch.cat(all_centers1, dim=0)
    all_centers2 = torch.cat(all_centers2, dim=0)

    sim1 = cos_sim_matrix(X_train, all_centers1)
    sim2 = cos_sim_matrix(Y_train, all_centers2)
    sim_similarity = cos_sim_matrix(sim1, sim2)
    
    # Free intermediate similarity matrices
    del sim1, sim2

    top_similar = sim_similarity.topk(dim=-1, k=k).indices
    del sim_similarity

    # Memory-efficient averaging: compute weighted sum incrementally instead of 
    # creating huge (N, k, dim) tensor with Y_train[top_similar]
    coefs = torch.ones(k) / k
    Y_matched = torch.zeros(len(X_train), Y_train.shape[1], dtype=Y_train.dtype)
    for i in range(k):
        Y_matched += coefs[i] * Y_train[top_similar[:, i]]

    return Y_matched

def refine_1(X_train, Y_train, n_iters=100, k=50, n_samples=1000):
    Y_matched = anchor(X_train, Y_train)
    W = train_orthogonal_linear(X_train, Y_matched)
    
    for _ in trange(n_iters, desc='Refining'):
        sample_points = X_train[torch.randperm(len(X_train))[:n_samples]]
        sample_similarities = cos_sim_matrix(sample_points @ W, Y_train)
        neighbors = sample_similarities.topk(dim=-1, k=k).indices
        sample_matched = Y_train[neighbors].mean(dim=1)
        
        W_new = train_orthogonal_linear(sample_points, sample_matched)
        W = 0.5 * W + 0.5 * W_new ## Why this way?
    return W

def refine_2(X_train, Y_train, W: torch.Tensor, n_iters=2, n_clusters=500):
    for _ in trange(n_iters):
        kmeans1 = KMeans(n_clusters=n_clusters).fit(X_train)
        centers1 = tensor(kmeans1.cluster_centers_)

        kmeans2 = KMeans(n_clusters=n_clusters, init=centers1 @ W).fit(Y_train) # type: ignore
        centers2 = tensor(kmeans2.cluster_centers_)

        # print('Self consistency of KMeans', torch.cosine_similarity(centers1 @ W, centers2, dim=-1).mean())

        W_new = train_orthogonal_linear(centers1, centers2)
        W = 0.5 * W + 0.5 * W_new
        # print('Eval score:', eval_score(X_eval, Y_eval, W))
    return W

@torch.no_grad()
def train(X_train, Y_train) -> torch.Tensor:
    W_raw = refine_1(X_train, Y_train, n_iters=100, k=50, n_samples=1000)
    W = refine_2(X_train, Y_train, W_raw, n_iters=2, n_clusters=500)
    return W

def run_experiment(
        ds1: str = 'nq', 
        ds2: str = 'trec-covid-corpus', 
        model1: str = 'stella', 
        model2: str = 'e5', 
        num_train: int = 28_000, 
        num_test: int = 5_000, 
        ratio: float = 0.5
    ) -> torch.Tensor:
    nq_stella, nq_e5 = torch.load(f'embeddings/{ds1}/{model1}.pt'), torch.load(f'embeddings/{ds1}/{model2}.pt')
    trec_stella, trec_e5 = torch.load(f'embeddings/{ds2}/{model1}.pt'), torch.load(f'embeddings/{ds2}/{model2}.pt')

    X_train, Y_train, X_eval, Y_eval = train_test_split(
        nq_stella, nq_e5,
        trec_stella, trec_e5,
        num_train_samples=num_train,
        num_test_samples=num_test,
        source1_ratio=ratio,
    )

    W = train(N(X_train), N(Y_train))

    metrics = report(N(X_eval), N(Y_eval), W, top_n=1)

    # Write results to CSV
    csv_path = 'distro_mixin.csv'
    fieldnames = ['ds1', 'ds2', 'model1', 'model2', 'num_train', 'num_test', 'ratio', 
                  'accuracy', 'avg_rank', 'cosine_similarity', 'matched_cosine_similarity',
                  'mean_cca', 'mean_svcca', 'pwcca']
    
    # Check if file exists to determine if we need to write header
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='') as f:
        writer = DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'ds1': ds1,
            'ds2': ds2,
            'model1': model1,
            'model2': model2,
            'num_train': num_train,
            'num_test': num_test,
            'ratio': ratio,
            **metrics
        })
        
    return W

if __name__ == '__main__':
    ## Load toml config here if desired
    # import sys
    # import toml

    # toml_file = sys.argv[1] if len(sys.argv) > 1 else 'mini/linear_config.toml'
    # config = toml.load(toml_file)

    # run_experiment(**config)
    for ratio in np.arange(0, 1.1, 0.1):
        run_experiment(ratio=float(ratio))
