import torch
import torch.nn.functional as F


def rmse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the root mean squared error across items in a batch.

    Args:
        x: Tensor of shape (batch_size, ...).
        y: Tensor of the same shape as `x`.

    Returns:
        A scalar tensor containing the mean RMSE across all batch items.
    """
    return ((x - y) ** 2).sum(dim=1).sqrt().mean()

def rec_loss_fn(ins, recons, logger, prefix=""):
    """
    Reconstruction loss computed using cosine similarity.

    For each embedding key in `ins`, this computes 1 - cosine_similarity(original, reconstructed)
    averaged across the batch and then averaged across all keys. This is used to ensure the
    reconstructed embedding matches the original embedding in vector-space.

    The function also logs an RMSE and cosine-based reconstruction value for each key to the
    provided `logger`.

    Args:
        ins: A dict mapping embedding flags to original embeddings (Tensor of shape (B, D)).
        recons: A dict mapping flags to their reconstructions with same shapes as `ins`.
        logger: A logger instance with a `logkv(key, value)` method; used for logging metrics.
        prefix: Optional string used as a prefix for log keys.

    Returns:
        A scalar tensor representing the mean reconstruction loss across all provided embeddings.
    """
    assert ins.keys() == recons.keys()
    loss = None
    for flag, emb in ins.items():
        recons_loss = 1 - F.cosine_similarity(emb, recons[flag], dim=1).mean()
        logger.logkv(f"{prefix}{flag}_recons_rmse", rmse(emb, recons[flag]))
        logger.logkv(f"{prefix}{flag}_recons_cos", recons_loss)
        if loss is None:
            loss = recons_loss
        else:
            loss += recons_loss
    return loss / len(ins)


# def rec_margin_loss_fn(ins, recons, logger, prefix="", margin: float = 0.1):
#     """Penalizes embeddings from being more than `margin` similarity away from at least
#     one embedding."""
#     assert ins.keys() == recons.keys()
#     loss = None
#     for flag, emb in ins.items():
#         A = emb / emb.norm(dim=1, p=2, keepdim=True)
#         B = recons[flag] / recons[flag].norm(dim=1, p=2, keepdim=True)
#         # B = B.mean(dim=1, keepdim=True)

#         cos_distances = 1 - F.cosine_similarity(A, B, dim=1)
#         recons_loss_cos = cos_distances.mean()
#         margin_loss = (cos_distances - margin).clamp(min=0).mean()
#         logger.logkv(f"{prefix}{flag}_recons_rmse", rmse(emb, recons[flag]))
#         logger.logkv(f"{prefix}{flag}_recons_cos", recons_loss_cos)
#         if loss is None:
#             loss = margin_loss
#         else:
#             loss += margin_loss
#     return loss / len(ins)

def uni_loss_fn(emb, trans, src_emb, tgt_emb, logger):
    """
    One-to-one unidirectional matching loss.

    Computes a cosine similarity-based loss between the source embeddings and their
    translations. Primarily used in scenarios where a single source-target translation
    is measured rather than multiple keyed translations.

    Args:
        emb: Tensor of source embeddings (B, D).
        trans: Tensor of corresponding translated embeddings (B, D).
        src_emb: Source embedding flag name (used for logging).
        tgt_emb: Target embedding flag name (used for logging).
        logger: A logger instance for metric logging.

    Returns:
        A scalar tensor containing the mean cosine-based unidirectional loss.
    """
    uni_loss = 1 - F.cosine_similarity(emb, trans, dim=1).mean()
    logger.logkv(f"{src_emb}_{tgt_emb}_uni_rmse", rmse(emb, trans))
    logger.logkv(f"{src_emb}_{tgt_emb}_uni_cos", uni_loss)
    return uni_loss


def trans_loss_fn(ins, translations, logger, prefix=""):
    """
    Multi-target translation loss.

    For each target embedding in `ins`, this compares the true embedding to each of the
    corresponding translated embeddings in `translations` using 1 - cosine similarity, logs
    RMSE and cosine values, and returns the mean loss across all keys and mappings.

    Args:
        ins: A dict mapping target flags to ground-truth embeddings (B, D).
        translations: A nested dict mapping target_flag -> {source_flag: translated_tensor}.
        logger: Logger instance used to record RMSE and cosine metrics.
        prefix: Prefix for logged metric keys.

    Returns:
        A scalar tensor containing the mean translation loss across all mappings.
    """
    assert ins.keys() == translations.keys()
    loss = None
    for target_flag, emb in ins.items():
        for flag, trans in translations[target_flag].items():
            trans_loss = 1 - F.cosine_similarity(emb, trans, dim=1).mean()
            logger.logkv(f"{prefix}{flag}_{target_flag}_trans_rmse", rmse(emb, trans))
            logger.logkv(f"{prefix}{flag}_{target_flag}_trans_cos", trans_loss)
            
            if loss is None:
                loss = trans_loss
            else:
                loss += trans_loss

    return (loss / len(ins))


def contrastive_loss_fn(ins, translations, logger) -> torch.Tensor:
    """
    TODO: Think about this + test
    Contrastive loss based on cross-entropy over cosine similarity scores.

    This loss normalizes target and translated embeddings and computes a similarity matrix
    (A @ B^T). It then treats the problem as a classification where each example must be matched
    to the correct counterpart using cross entropy over similarity scores (scaled by 50).
    Intended to encourage translated embeddings to have the highest similarity with their
    target counterpart in the batch.

    Note: This uses a temperature/scaling factor of 50; also it is currently not used directly
    in the main training loop but is available for experimental objectives.

    Args:
        ins: Dict mapping out_name -> original embeddings.
        translations: Dict mapping out_name -> {in_name -> translated embeddings}.
        logger: Optional logger used to record loss values.

    Returns:
        A scalar tensor representing the mean contrastive loss over all mappings.
    """
    # TODO: Think about this + test.
    loss = None
    EPS = 1e-10
    count = 0
    for out_name in ins.keys():
        for in_name in translations[out_name].keys():
            B = ins[out_name].detach()
            B = B / (B.norm(dim=1, keepdim=True) + EPS)
            in_sims = B @ B.T
            A = translations[out_name][in_name]
            A = A / (A.norm(dim=1, keepdim=True) + EPS)
            out_sims_reflected = A @ B.T
            contrastive_loss = torch.nn.functional.cross_entropy(
                out_sims_reflected * 50,
                torch.arange(in_sims.shape[0], device=in_sims.device)
            )
            if logger is not None:
                logger.logkv(f"{in_name}_{out_name}_contrastive", contrastive_loss)

            if loss is None:
                loss = contrastive_loss
            else:
                loss += contrastive_loss
            count += 1
    return loss / count

def vsp_loss_fn(ins, translations, logger) -> torch.Tensor:
    """
    Vector-Similarity-Preserving (VSP) loss.

    This loss encourages the pairwise similarity structure of translated embeddings to
    match that of the source embeddings. For each mapping, it computes the pairwise
    similarity matrices of the source and translated embeddings (normalized), then computes
    the mean absolute difference between those matrices. A reflected version between A and B
    is also computed to measure cross-similarity preservation. The two terms are summed.

    Args:
        ins: Dict mapping out_name -> source embeddings.
        translations: Dict mapping out_name -> {in_name -> translated embeddings}.
        logger: Optional logger used to record VSP metrics.

    Returns:
        A scalar tensor representing the average VSP loss across all mappings.
    """
    loss = None
    EPS = 1e-10
    count = 0
    for out_name in ins.keys():
        for in_name in translations[out_name].keys():
            B = ins[out_name].detach()
            B = B / (B.norm(dim=1, keepdim=True) + EPS)
            in_sims = B @ B.T
            A = translations[out_name][in_name]
            A = A / (A.norm(dim=1, keepdim=True) + EPS)
            out_sims = A @ A.T
            out_sims_reflected = A @ B.T
            vsp_loss = (in_sims - out_sims).abs().mean()
            vsp_loss_reflected = (in_sims - out_sims_reflected).abs().mean()
            if logger is not None:
                logger.logkv(f"{in_name}_{out_name}_vsp", vsp_loss)
                logger.logkv(f"{in_name}_{out_name}_vsp_reflected", vsp_loss_reflected)

            if loss is None:
                loss = vsp_loss + vsp_loss_reflected
            else:
                loss += vsp_loss + vsp_loss_reflected
            count += 1
    return loss / count


def get_grad_norm(model: torch.nn.Module) -> torch.Tensor:
    """
    Compute the global L2 norm of gradients for a model.

    Sums the squared 2-norms of all parameter gradients, then takes the square root
    to return the L2 norm of the full gradient vector. Useful for logging gradient
    magnitudes and diagnosing training instabilities.

    Args:
        model: A PyTorch module whose .parameters() are examined.

    Returns:
        The L2 norm of the model's gradients as a scalar tensor.
    """
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)  # Calculate the 2-norm of the gradients
            total_norm += param_norm.detach() ** 2
    total_norm = total_norm ** (1. / 2)  # Take the square root to get the total norm
    return total_norm