import torch

def _split_single_dataset(
    embeddings_x: torch.Tensor, 
    embeddings_y: torch.Tensor,
    num_train_samples: int,
    num_test_samples: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split a single dataset into train/test sets.
    - Training sets: Independently shuffled, NO correspondence between X_train and Y_train
    - Test sets: 1:1 matched (same indices for X and Y)
    """
    total_size = len(embeddings_x)
    assert num_train_samples + num_test_samples <= total_size, \
        f"Not enough samples (need {num_train_samples + num_test_samples}, have {total_size})"
    
    # Reserve test samples (1:1 matched - same indices for X and Y)
    X_test = embeddings_x[:num_test_samples]
    Y_test = embeddings_y[:num_test_samples]
    
    # Remaining data for training
    remaining_x = embeddings_x[num_test_samples:]
    remaining_y = embeddings_y[num_test_samples:]
    
    # Sample independently for X and Y (no correspondence)
    indices_x = torch.randperm(len(remaining_x))[:num_train_samples]
    indices_y = torch.randperm(len(remaining_y))[:num_train_samples]
    X_train = remaining_x[indices_x]
    Y_train = remaining_y[indices_y]
    
    return X_train, Y_train, X_test, Y_test


def _split_two_datasets(
    embeddings_x1: torch.Tensor, embeddings_y1: torch.Tensor,
    embeddings_x2: torch.Tensor, embeddings_y2: torch.Tensor,
    num_train_samples: int,
    num_test_samples: int,
    source1_ratio: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split two datasets into train/test sets with mixing.
    - Training sets: Mix of both sources controlled by source1_ratio, independently shuffled
    - Test sets: 1:1 matched, always 50/50 split between sources
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


def train_test_split(
    embeddings_x1: torch.Tensor, embeddings_y1: torch.Tensor,
    embeddings_x2: torch.Tensor | None = None, embeddings_y2: torch.Tensor | None = None,
    num_train_samples: int = 10000,
    num_test_samples: int = 1000,
    normalize: bool = True,
    source1_ratio: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Splits embeddings from one or two data sources into training and testing sets.
    
    Args:
        embeddings_x1, embeddings_y1: First data source (e.g., nq stella/e5)
        embeddings_x2, embeddings_y2: Second data source (optional, e.g., trec stella/e5)
        num_train_samples: Total number of samples for each training set
        num_test_samples: Total number of samples for evaluation (1:1 matched)
        source1_ratio: Proportion of training data from source 1 (0.0 to 1.0), ignored if single dataset
    
    Returns:
        X_train, Y_train, X_test, Y_test
    """
    if embeddings_x2 is None or embeddings_y2 is None:
        splits = _split_single_dataset(embeddings_x1, embeddings_y1, num_train_samples, num_test_samples)

    else:
        splits = _split_two_datasets(
            embeddings_x1, embeddings_y1, 
            embeddings_x2, embeddings_y2,
            num_train_samples, num_test_samples, source1_ratio
        )
    if normalize:
        splits = tuple(torch.nn.functional.normalize(e, dim=1) for e in splits)
    return splits # type: ignore