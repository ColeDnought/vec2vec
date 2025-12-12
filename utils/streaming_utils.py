import os

import datasets
import torch

from datasets import Features, Value, load_dataset
from utils.beir_dl import HFDataLoader


os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"



def _prepare_retrieval_dataset(d: datasets.Dataset) -> datasets.Dataset:
    q_ds = d.remove_columns([c for c in d.column_names if c not in {'query', 'dataset'}]).rename_column('query', 'text')
    d_ds = d.remove_columns([c for c in d.column_names if c not in {'document', 'dataset'}]).rename_column('document', 'text')
    return datasets.concatenate_datasets([q_ds, d_ds])


def _load_retrieval_dataset() -> datasets.Dataset:
    d1 = datasets.load_dataset(
        "jxm/nomic_embed_supervised",
        num_proc=32,
        keep_in_memory=False,
    )["train"]
    d1 = _prepare_retrieval_dataset(d1)
    d2 = datasets.load_dataset(
        "jxm/nomic_embed_unsupervised",
        num_proc=32,
        keep_in_memory=False,
    )["train"]
    d2 = _prepare_retrieval_dataset(d2)
    return datasets.concatenate_datasets([d1, d2])


def load_streaming_embeddings(
        dataset_name: str,
        split_flag: str = "train",
        streaming: bool = False,
    ) -> datasets.Dataset:
    num_proc = None if streaming else 8
    if dataset_name == 'nq':
        dset = load_dataset("jxm/nq_corpus_dpr", split=split_flag, streaming=streaming)
    elif dataset_name == 'fineweb':
        dset = load_dataset("HuggingFaceFW/fineweb", streaming=streaming, num_proc=num_proc, keep_in_memory=False)["train"]
    elif dataset_name == 'fineweb-medium':
        dset = load_dataset("HuggingFaceFW/fineweb", "sample-350BT", streaming=streaming, num_proc=num_proc, keep_in_memory=False)["train"]
    elif dataset_name == "fineweb-tiny":
        dset = load_dataset("HuggingFaceFW/fineweb", "sample-10BT", streaming=streaming, num_proc=num_proc, keep_in_memory=False)["train"]
    elif dataset_name == "nq-corpus":
        dset = load_dataset("BeIR/nq", "corpus", streaming=streaming, num_proc=num_proc)["corpus"]
    elif dataset_name == "arguana-corpus":
        dset = load_dataset("BeIR/arguana", "corpus", streaming=streaming, num_proc=num_proc)["corpus"]
    elif dataset_name == "arguana-queries":
        dset = load_dataset("BeIR/arguana", "queries", streaming=streaming, num_proc=num_proc)["queries"]
    elif dataset_name == "fiqa-corpus":
        dset = load_dataset("BeIR/fiqa", "corpus", streaming=streaming, num_proc=num_proc)["corpus"]
    elif dataset_name == "fiqa-queries":
        dset = load_dataset("BeIR/fiqa", "queries", streaming=streaming, num_proc=num_proc)["queries"]
    elif dataset_name == "quora-corpus":
        dset = load_dataset("BeIR/quora", "corpus", streaming=streaming, num_proc=num_proc)["corpus"]
    elif dataset_name == "quora-queries":
        dset = load_dataset("BeIR/quora", "queries", streaming=streaming, num_proc=num_proc)["queries"]
    elif dataset_name == "trec-covid-corpus":
        dset = load_dataset("BeIR/trec-covid", "corpus", streaming=streaming, num_proc=num_proc)["corpus"]
    elif dataset_name == "trec-covid-queries":
        dset = load_dataset("BeIR/trec-covid", "queries", streaming=streaming, num_proc=num_proc)["queries"]
    elif dataset_name == "fever-corpus":
        dset = load_dataset("BeIR/fever", "corpus", streaming=streaming, num_proc=num_proc)["corpus"]
    elif dataset_name == "fever-queries":
        dset = load_dataset("BeIR/fever", "queries", streaming=streaming, num_proc=num_proc)["queries"]
    elif dataset_name == "scifact-corpus":
        dset = load_dataset("BeIR/scifact", "corpus", streaming=streaming, num_proc=num_proc)["corpus"]
    elif dataset_name == "scifact-queries":
        dset = load_dataset("BeIR/scifact", "queries", streaming=streaming, num_proc=num_proc)["queries"]
    elif dataset_name == "msmarco-corpus":
        dset = load_dataset("BeIR/msmarco", "corpus", streaming=streaming, num_proc=num_proc)["corpus"]
    elif dataset_name == "msmarco-queries":
        dset = load_dataset("BeIR/msmarco", "queries", streaming=streaming, num_proc=num_proc)["queries"]
    elif dataset_name == "trec-news":
        dset = load_dataset("BeIR/trec-news-generated-queries", streaming=streaming, num_proc=num_proc, split="train").remove_columns(['_id', 'title', 'query'])
    elif dataset_name == "retrieval":
        dset = _load_retrieval_dataset()
    else:
        raise NotImplementedError()

    # Don't apply torch format to streaming datasets - they'll be materialized later
    if not streaming:
        dset = dset.with_format("torch")
    return dset


def vec2text_mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    B, S, D = hidden_states.shape
    unmasked_outputs = hidden_states * attention_mask[..., None]
    pooled_outputs = unmasked_outputs.sum(dim=1) / attention_mask.sum(dim=1)[:, None]
    assert pooled_outputs.shape == (B, D)
    return pooled_outputs


def get_embeddings(text_list,
                   encoder,
                   tokenizer,
                   max_length,
                   device):

    inputs = tokenizer(text_list,
                       return_tensors="pt",
                       max_length=max_length,
                       truncation=True,
                       padding="max_length").to(device)

    with torch.no_grad():
        model_output = encoder(**inputs)
        hidden_state = model_output.last_hidden_state
        embeddings = vec2text_mean_pool(hidden_state, inputs['attention_mask'])

    return embeddings

def embed(x, encoder, tokenizer, max_length=32, device='cpu'):
    embeddings = get_embeddings(x['text'], encoder, tokenizer, max_length, device)
    return {
        'text': x['text'],
        'text_embeddings': embeddings
    }

def forward_embedding_sentence_transformers(enc, features, normalize_embeddings: bool = True, mixed_precision: str = None):
    output_value  = "sentence_embedding"
    if mixed_precision is not None:
        if mixed_precision == 'bf16':
            enc_type = torch.bfloat16
        elif mixed_precision == 'fp16':
            enc_type = torch.float16
        else:
            raise ValueError(f"Unknown mixed precision flag {mixed_precision}")
    else:
        enc_type = torch.float32
    with torch.no_grad(), torch.autocast("cuda", dtype=enc_type):
        out_features = enc.forward(features)
    embeddings = out_features[output_value]
    embeddings = embeddings.detach()
    if normalize_embeddings:
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    return embeddings.to(torch.float32)


def process_batch(batch, encoders, normalize_embeddings, device='cpu'):
    ins = {}    
    batch_embs = [k.replace("_input_ids", "") for k in batch.keys() if k.endswith("_input_ids")]
    for emb in batch_embs:
        encoders[emb].to(device)
        emb_inputs = { k.replace(f"{emb}_", ""): v.to(device) for k, v in batch.items() if k.startswith(f"{emb}_") }
        ins[emb] = forward_embedding_sentence_transformers(
            encoders[emb], emb_inputs,
            normalize_embeddings=normalize_embeddings
        )
    return ins


class NanoBeirHFDataLoaderOverride(HFDataLoader):
    def _load_qrels(self, split):
        qrels_ds = load_dataset(
            self.hf_repo,
            "qrels"
        )["train"]
        qrels_ds = qrels_ds.add_column("score", [1] * len(qrels_ds))
        features = Features(
            {
                "query-id": Value("string"),
                "corpus-id": Value("string"),
                "score": Value("float"),
            }
        )
        qrels_ds = qrels_ds.cast(features)
        self.qrels = qrels_ds


def load_beir_style_dataset(dataset: str):
    if 'nano' in dataset.lower():
        corpus, queries, qrels = NanoBeirHFDataLoaderOverride(
            hf_repo=f"zeta-alpha-ai/{dataset}",
            streaming=False,
            keep_in_memory=True
        ).load()
    else:
        corpus, queries, qrels = HFDataLoader(
            hf_repo=f"BeIR/{dataset.lower()}",
            streaming=False,
            keep_in_memory=False
        ).load(split="test")
    return corpus, queries, qrels


def distribution_split(
    dataset1: datasets.Dataset | datasets.IterableDataset,
    dataset2: datasets.Dataset | datasets.IterableDataset | None = None,
    num_sup_samples: int = 10000,
    num_unsup_samples: int = 10000,
    num_val_samples: int = 1000,
    source1_ratio_sup: float = 1.0,
    source1_ratio_unsup: float = 0.0,
    train_seed: int = 42,
    val_seed: int = 42,
) -> tuple[datasets.Dataset, datasets.Dataset, datasets.Dataset]:
    """
    Splits one or two datasets into supervised, unsupervised, and validation sets
    with configurable mixing ratios. Works with both Dataset and IterableDataset.
    For IterableDataset inputs, the function materializes the data into regular Datasets.
    
    This enables "distribution split" training where:
    - Supervised set can be sourced from a mix of dataset1 and dataset2
    - Unsupervised set can be sourced from a different mix of dataset1 and dataset2
    - Validation set is always 50/50 from both sources (if two datasets provided)
    
    Args:
        dataset1: First dataset (e.g., domain A) - can be Dataset or IterableDataset
        dataset2: Second dataset (optional, e.g., domain B) - can be Dataset or IterableDataset
        num_sup_samples: Number of samples for supervised training
        num_unsup_samples: Number of samples for unsupervised training
        num_val_samples: Number of samples for validation (1:1 matched if two datasets)
        source1_ratio_sup: Proportion of supervised data from dataset1 (0.0 to 1.0)
        source1_ratio_unsup: Proportion of unsupervised data from dataset1 (0.0 to 1.0)
        train_seed: Random seed for training set shuffling
        val_seed: Random seed for validation set shuffling
    
    Returns:
        supset, unsupset, valset: The three Dataset splits (always regular Dataset, even for streaming input)
    """
    is_streaming = isinstance(dataset1, datasets.IterableDataset)
    
    if dataset2 is None:
        if is_streaming:
            return _distribution_split_single_streaming(
                dataset1, num_sup_samples, num_unsup_samples, num_val_samples, train_seed, val_seed
            )
        else:
            return _distribution_split_single(
                dataset1, num_sup_samples, num_unsup_samples, num_val_samples, train_seed, val_seed
            )
    else:
        if is_streaming:
            return _distribution_split_two_streaming(
                dataset1, dataset2,
                num_sup_samples, num_unsup_samples, num_val_samples,
                source1_ratio_sup, source1_ratio_unsup,
                train_seed, val_seed
            )
        else:
            return _distribution_split_two(
                dataset1, dataset2,
                num_sup_samples, num_unsup_samples, num_val_samples,
                source1_ratio_sup, source1_ratio_unsup,
                train_seed, val_seed
            )


def _distribution_split_single(
    dataset: datasets.Dataset,
    num_sup_samples: int,
    num_unsup_samples: int,
    num_val_samples: int,
    train_seed: int,
    val_seed: int,
) -> tuple[datasets.Dataset, datasets.Dataset, datasets.Dataset]:
    """
    Split a single dataset into sup/unsup/val sets.
    - Validation set is reserved first
    - Sup and unsup are independently sampled from remaining data (no correspondence)
    """
    total_needed = num_sup_samples + num_unsup_samples + num_val_samples
    assert total_needed <= len(dataset), \
        f"Not enough samples (need {total_needed}, have {len(dataset)})"
    
    # Reserve validation samples first
    dataset_shuffled = dataset.shuffle(seed=val_seed)
    valset = dataset_shuffled.select(range(num_val_samples))
    remaining = dataset_shuffled.select(range(num_val_samples, len(dataset_shuffled)))
    
    # Independently sample for sup and unsup (shuffle with different effective seeds)
    remaining_for_sup = remaining.shuffle(seed=train_seed)
    remaining_for_unsup = remaining.shuffle(seed=train_seed + 1)  # Different seed for independence
    
    supset = remaining_for_sup.select(range(num_sup_samples))
    unsupset = remaining_for_unsup.select(range(num_unsup_samples))
    
    return supset, unsupset, valset


def _distribution_split_two(
    dataset1: datasets.Dataset,
    dataset2: datasets.Dataset,
    num_sup_samples: int,
    num_unsup_samples: int,
    num_val_samples: int,
    source1_ratio_sup: float,
    source1_ratio_unsup: float,
    train_seed: int,
    val_seed: int,
) -> tuple[datasets.Dataset, datasets.Dataset, datasets.Dataset]:
    """
    Split two datasets into sup/unsup/val sets with mixing.
    - Sup set: mix controlled by source1_ratio_sup
    - Unsup set: mix controlled by source1_ratio_unsup  
    - Val set: always 50/50 split between sources
    """
    assert 0.0 <= source1_ratio_sup <= 1.0, "source1_ratio_sup must be between 0 and 1"
    assert 0.0 <= source1_ratio_unsup <= 1.0, "source1_ratio_unsup must be between 0 and 1"
    
    # Calculate samples from each source for supervised
    num_sup_source1 = int(num_sup_samples * source1_ratio_sup)
    num_sup_source2 = num_sup_samples - num_sup_source1
    
    # Calculate samples from each source for unsupervised
    num_unsup_source1 = int(num_unsup_samples * source1_ratio_unsup)
    num_unsup_source2 = num_unsup_samples - num_unsup_source1
    
    # Validation set is always 50/50
    num_val_source1 = num_val_samples // 2
    num_val_source2 = num_val_samples - num_val_source1
    
    # Validate we have enough samples
    total_from_source1 = num_sup_source1 + num_unsup_source1 + num_val_source1
    total_from_source2 = num_sup_source2 + num_unsup_source2 + num_val_source2
    assert total_from_source1 <= len(dataset1), \
        f"Not enough samples in dataset1 (need {total_from_source1}, have {len(dataset1)})"
    assert total_from_source2 <= len(dataset2), \
        f"Not enough samples in dataset2 (need {total_from_source2}, have {len(dataset2)})"
    
    # Shuffle datasets
    ds1_shuffled = dataset1.shuffle(seed=val_seed)
    ds2_shuffled = dataset2.shuffle(seed=val_seed)
    
    # Reserve validation samples first
    val_s1 = ds1_shuffled.select(range(num_val_source1))
    val_s2 = ds2_shuffled.select(range(num_val_source2))
    
    remaining1 = ds1_shuffled.select(range(num_val_source1, len(ds1_shuffled)))
    remaining2 = ds2_shuffled.select(range(num_val_source2, len(ds2_shuffled)))
    
    # Sample for supervised (independently shuffled)
    remaining1_for_sup = remaining1.shuffle(seed=train_seed)
    remaining2_for_sup = remaining2.shuffle(seed=train_seed)
    
    sup_s1 = remaining1_for_sup.select(range(num_sup_source1)) if num_sup_source1 > 0 else None
    sup_s2 = remaining2_for_sup.select(range(num_sup_source2)) if num_sup_source2 > 0 else None
    
    # Sample for unsupervised (different shuffle for independence)
    remaining1_for_unsup = remaining1.shuffle(seed=train_seed + 1)
    remaining2_for_unsup = remaining2.shuffle(seed=train_seed + 1)
    
    unsup_s1 = remaining1_for_unsup.select(range(num_unsup_source1)) if num_unsup_source1 > 0 else None
    unsup_s2 = remaining2_for_unsup.select(range(num_unsup_source2)) if num_unsup_source2 > 0 else None
    
    # Combine datasets
    sup_parts = [d for d in [sup_s1, sup_s2] if d is not None]
    unsup_parts = [d for d in [unsup_s1, unsup_s2] if d is not None]
    val_parts = [d for d in [val_s1, val_s2] if d is not None]
    
    supset = datasets.concatenate_datasets(sup_parts).shuffle(seed=train_seed + 2)
    unsupset = datasets.concatenate_datasets(unsup_parts).shuffle(seed=train_seed + 3)
    valset = datasets.concatenate_datasets(val_parts).shuffle(seed=val_seed + 1)
    
    return supset, unsupset, valset


def _distribution_split_single_streaming(
    dataset: datasets.IterableDataset,
    num_sup_samples: int,
    num_unsup_samples: int,
    num_val_samples: int,
    train_seed: int,
    val_seed: int,
) -> tuple[datasets.Dataset, datasets.Dataset, datasets.Dataset]:
    """
    Split a single streaming dataset into sup/unsup/val sets.
    Materializes the streaming data into regular Datasets.
    - Validation set is reserved first
    - Sup and unsup are independently sampled from remaining data (no correspondence)
    """
    # Shuffle and collect samples
    dataset_shuffled = dataset.shuffle(seed=val_seed, buffer_size=10000)
    
    # Collect all needed samples in one pass
    total_needed = num_val_samples + num_sup_samples + num_unsup_samples
    samples = []
    for i, sample in enumerate(dataset_shuffled):
        if i >= total_needed:
            break
        samples.append(sample)
    
    # Split the collected samples
    valset = datasets.Dataset.from_list(samples[:num_val_samples])
    remaining = samples[num_val_samples:]
    
    # Shuffle remaining for sup and unsup independently using numpy
    import numpy as np
    rng_sup = np.random.default_rng(train_seed)
    rng_unsup = np.random.default_rng(train_seed + 1)
    
    sup_indices = rng_sup.permutation(len(remaining))[:num_sup_samples]
    unsup_indices = rng_unsup.permutation(len(remaining))[:num_unsup_samples]
    
    supset = datasets.Dataset.from_list([remaining[i] for i in sup_indices])
    unsupset = datasets.Dataset.from_list([remaining[i] for i in unsup_indices])
    
    return supset, unsupset, valset


def _distribution_split_two_streaming(
    dataset1: datasets.IterableDataset,
    dataset2: datasets.IterableDataset,
    num_sup_samples: int,
    num_unsup_samples: int,
    num_val_samples: int,
    source1_ratio_sup: float,
    source1_ratio_unsup: float,
    train_seed: int,
    val_seed: int,
) -> tuple[datasets.Dataset, datasets.Dataset, datasets.Dataset]:
    """
    Split two streaming datasets into sup/unsup/val sets with mixing.
    Materializes the streaming data into regular Datasets.
    - Sup set: mix controlled by source1_ratio_sup
    - Unsup set: mix controlled by source1_ratio_unsup  
    - Val set: always 50/50 split between sources
    """
    import numpy as np
    
    assert 0.0 <= source1_ratio_sup <= 1.0, "source1_ratio_sup must be between 0 and 1"
    assert 0.0 <= source1_ratio_unsup <= 1.0, "source1_ratio_unsup must be between 0 and 1"
    
    # Calculate samples from each source for supervised
    num_sup_source1 = int(num_sup_samples * source1_ratio_sup)
    num_sup_source2 = num_sup_samples - num_sup_source1
    
    # Calculate samples from each source for unsupervised
    num_unsup_source1 = int(num_unsup_samples * source1_ratio_unsup)
    num_unsup_source2 = num_unsup_samples - num_unsup_source1
    
    # Validation set is always 50/50
    num_val_source1 = num_val_samples // 2
    num_val_source2 = num_val_samples - num_val_source1
    
    # Calculate total needed from each source
    total_from_source1 = num_val_source1 + max(num_sup_source1, num_unsup_source1)
    total_from_source2 = num_val_source2 + max(num_sup_source2, num_unsup_source2)
    
    # Collect samples from dataset1
    print(f"Collecting {total_from_source1} samples from dataset1...")
    ds1_shuffled = dataset1.shuffle(seed=val_seed, buffer_size=10000)
    samples1 = []
    for i, sample in enumerate(ds1_shuffled):
        if i >= total_from_source1:
            break
        samples1.append(sample)
    
    # Collect samples from dataset2
    print(f"Collecting {total_from_source2} samples from dataset2...")
    ds2_shuffled = dataset2.shuffle(seed=val_seed, buffer_size=10000)
    samples2 = []
    for i, sample in enumerate(ds2_shuffled):
        if i >= total_from_source2:
            break
        samples2.append(sample)
    
    # Split into val and remaining
    val_samples1 = samples1[:num_val_source1]
    val_samples2 = samples2[:num_val_source2]
    remaining1 = samples1[num_val_source1:]
    remaining2 = samples2[num_val_source2:]
    
    # Sample for supervised
    rng_sup = np.random.default_rng(train_seed)
    sup_samples = []
    if num_sup_source1 > 0:
        indices = rng_sup.permutation(len(remaining1))[:num_sup_source1]
        sup_samples.extend([remaining1[i] for i in indices])
    if num_sup_source2 > 0:
        indices = rng_sup.permutation(len(remaining2))[:num_sup_source2]
        sup_samples.extend([remaining2[i] for i in indices])
    
    # Sample for unsupervised
    rng_unsup = np.random.default_rng(train_seed + 1)
    unsup_samples = []
    if num_unsup_source1 > 0:
        indices = rng_unsup.permutation(len(remaining1))[:num_unsup_source1]
        unsup_samples.extend([remaining1[i] for i in indices])
    if num_unsup_source2 > 0:
        indices = rng_unsup.permutation(len(remaining2))[:num_unsup_source2]
        unsup_samples.extend([remaining2[i] for i in indices])
    
    # Shuffle combined samples
    rng_final = np.random.default_rng(train_seed + 2)
    rng_final.shuffle(sup_samples)
    rng_final = np.random.default_rng(train_seed + 3)
    rng_final.shuffle(unsup_samples)
    
    val_samples = val_samples1 + val_samples2
    rng_val = np.random.default_rng(val_seed + 1)
    rng_val.shuffle(val_samples)
    
    # Convert to Datasets
    supset = datasets.Dataset.from_list(sup_samples)
    unsupset = datasets.Dataset.from_list(unsup_samples)
    valset = datasets.Dataset.from_list(val_samples)
    
    return supset, unsupset, valset