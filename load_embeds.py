import os
import random

import numpy as np
from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader

from utils.collate import MultiencoderTokenizedDataset, TokenizedCollator
from utils.model_utils import load_encoder
from utils.utils import get_num_proc
from utils.streaming_utils import load_streaming_embeddings, process_batch

from tqdm.auto import tqdm

accelerator = Accelerator()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def process_all(dataloader, encoder_dict, encoder_name, normalize_embeddings):
    """Process all batches from dataloader and return concatenated embeddings."""
    all_embeddings = []
    
    for batch in tqdm(dataloader, desc=f"Embedding with {encoder_name.split('/')[-1]}"):
        # Process batch to get embeddings
        emb_dict = process_batch(batch, encoder_dict, normalize_embeddings, device)
        embeddings = emb_dict[encoder_name].cpu()
        all_embeddings.append(embeddings)
    
    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)
    torch.cuda.empty_cache()
    
    return all_embeddings


def load_embeddings(
    dataset,
    x_encoder,
    y_encoder,
    num_points=8000,
    bs=32,
    max_seq_length=512,
    seed=42,
    train_dataset_seed=42,
    val_dataset_seed=42,
    sampling_seed=42,
    val_size=1000,
    normalize_embeddings=True,
    mixed_precision=None,
):
    os.environ["TOKENIZERS_PARALLELISM"] = "0"
    
    # Set seeds
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load encoders
    sup_enc = load_encoder(x_encoder, mixed_precision=mixed_precision)
    unsup_enc = load_encoder(y_encoder, mixed_precision=mixed_precision)
    
    # Create encoder dictionaries for MultiencoderTokenizedDataset
    sup_encs = {x_encoder: sup_enc}
    unsup_encs = {y_encoder: unsup_enc}
    
    num_workers = min(get_num_proc(), 8)
    
    # Load dataset
    dset = load_streaming_embeddings(dataset)

    # Split into train/val
    dset_dict = dset.train_test_split(test_size=val_size, seed=val_dataset_seed)
    dset = dset_dict["train"]
    
    # Shuffle and select subsets
    dset = dset.shuffle(seed=train_dataset_seed)
    supset = dset.select(range(num_points))
    unsupset = dset.select(range(num_points, num_points * 2))
    
    # Set format to "python" to avoid NumPy 2.0 compatibility issues
    supset.set_format("python")
    unsupset.set_format("python")
    
    # Wrap datasets
    supset = MultiencoderTokenizedDataset(
        dataset=supset,
        encoders=sup_encs,
        n_embs_per_batch=1,  # Changed from n_embs_per_batch - only 1 encoder provided
        batch_size=bs,
        max_length=max_seq_length,
        seed=sampling_seed,
    )
    unsupset = MultiencoderTokenizedDataset(
        dataset=unsupset,
        encoders=unsup_encs,
        n_embs_per_batch=1,
        batch_size=bs,
        max_length=max_seq_length,
        seed=sampling_seed,
    )
    
    # Create dataloaders
    sup_dataloader = DataLoader(
        supset,
        batch_size=bs,
        num_workers=num_workers // 2,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=None,
        collate_fn=TokenizedCollator(),
        drop_last=True,
    )
    unsup_dataloader = DataLoader(
        unsupset,
        batch_size=bs,
        num_workers=num_workers // 2,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=None,
        collate_fn=TokenizedCollator(),
        drop_last=True,
    )
    
    sup_embeddings = process_all(sup_dataloader, sup_encs, x_encoder, normalize_embeddings)
    unsup_embeddings = process_all(unsup_dataloader, unsup_encs, y_encoder, normalize_embeddings)
    
    return sup_embeddings, unsup_embeddings


if __name__ == "__main__":
    # Call with keyword arguments
    sup_embeddings, unsup_embeddings = load_embeddings(
        dataset='nq',
        num_points=8000,
        bs=32,
        x_encoder='infgrad/stella-base-en-v2',
        y_encoder='intfloat/e5-base-v2',
    )