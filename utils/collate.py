from typing import Any, Dict, List

import collections
import math
import os
import random

import datasets
import numpy as np
import torch

from utils.tokenization import get_tokenizer_max_length


class TokenizedCollator:
    # TODO: Fix to use separate tokenizers

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        os.environ['TOKENIZERS_PARALLELISM'] = '0'

        out_ex = collections.defaultdict(list)
        for ex in features:
            for col in ex:
                out_ex[col].append(ex[col])

        extra_keys = []
        for k, v in out_ex.items():
            if isinstance(v, list) and isinstance(v[0], int):
                out_ex[k] = torch.tensor(v)
            if isinstance(v, list) and isinstance(v[0], list):
                if (len(v[0]) > 0) and isinstance(v[0][0], int):
                    out_ex[k] = torch.tensor(v)
                else:
                    # skip empty lists
                    continue
            else:
                try:
                    out_ex[k] = torch.stack(v)
                except TypeError:
                    # can't stack string, etc. -- just leave em
                    extra_keys.append(k)

        return dict(out_ex)


class MultiencoderTokenizedDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset: datasets.Dataset, 
            encoders: dict[str, Any],
            n_embs_per_batch: int, 
            batch_size: int,
            max_length: int, 
            seed: int = 42,
        ):
        self.tokenizers = {
            enc_name: enc.tokenizer for enc_name, enc in encoders.items()
        }
        self.tokenizer_names = list(sorted(self.tokenizers.keys()))
        self.n_embs_per_batch = n_embs_per_batch
        self.batch_size = batch_size
        self.max_length = max_length
        
        self.n_batches = math.ceil(len(dataset) / self.batch_size)
        self.batch_token_name_idxs = []
        for _ in range(self.n_batches):
            self.batch_token_name_idxs.append(self._get_batch_token_name_idxs())

        self._len = len(dataset)
        self.dataset = dataset
        new_indices = np.arange(len(dataset))
        # shuffle indices
        np.random.Generator(np.random.PCG64(seed)).shuffle(new_indices)
        self.new_indices = list(new_indices)
        
    def _get_batch_token_name_idxs(self) -> tuple[int, int]:
        return random.sample(range(len(self.tokenizers)), k=self.n_embs_per_batch)
    
    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        tok_name_idxs = self.batch_token_name_idxs[idx // self.batch_size]
        tok_names = [self.tokenizer_names[i] for i in tok_name_idxs]
        tokenizer_max_lengths = [
            get_tokenizer_max_length(self.tokenizers[tok_name])
            for tok_name in tok_names
        ]
        max_length = min(*tokenizer_max_lengths, self.max_length)
        _max_num_chars = max_length * 5

        ex_text = self.dataset[int(self.new_indices[idx])]["text"]
        ex_text = ex_text[:_max_num_chars]
        output = {}
        for tok_name in tok_names:
            tt = self.tokenizers[tok_name](
                ex_text,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            if tok_name == 'clip':
                tt["image_text_info"] = torch.tensor(1)

            output.update({f"{tok_name}_{key}": value for key, value in tt.items()})
        
        if "token_name_idxs" in output: output.pop("token_name_idxs")
        return { k: v.flatten() for k,v in output.items()}


class MultiEncoderClassificationDataset(MultiencoderTokenizedDataset):
    def __init__(
            self,
            dataset: datasets.Dataset, 
            encoders: dict[str, Any],
            n_embs_per_batch: int, 
            batch_size: int,
            max_length: int, 
            seed: int = 42,
            return_texts: bool = False,
        ):
        super().__init__(dataset, encoders, n_embs_per_batch, batch_size, max_length, seed)
        self.labels = None
        self.label_texts = None
        self.encoders = encoders
        self.return_texts = return_texts

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        out = super().__getitem__(idx)
        out["label"] = self.dataset[int(self.new_indices[idx])]["label"]
        if self.return_texts:
            out["text"] = self.dataset[int(self.new_indices[idx])]["text"]
        return out

class CycleDataset(torch.utils.data.Dataset):
    """Dataset for cycle consistency with image->text translation"""
    def __init__(
            self,
            dataset: datasets.Dataset,
            text_encoders: dict[str, Any],  # e.g., {'gte': model}
            image_encoders: dict[str, Any],  # e.g., {'clip': model}
            n_embs_per_batch: int,
            batch_size: int,
            max_length: int,
            seed: int = 42,
        ):
        self.text_tokenizers = {
            enc_name: enc.tokenizer for enc_name, enc in text_encoders.items()
        }
        self.image_processors = {
            enc_name: enc.processor if hasattr(enc, 'processor') else None 
            for enc_name, enc in image_encoders.items()
        }
        
        self.text_tokenizer_names = list(sorted(self.text_tokenizers.keys()))
        self.image_encoder_names = list(sorted(self.image_processors.keys()))
        self.n_embs_per_batch = n_embs_per_batch
        self.batch_size = batch_size
        self.max_length = max_length
        
        # For CycleDataset, we always use all encoders (no sampling needed)
        self._len = len(dataset)
        self.dataset = dataset
        new_indices = np.arange(len(dataset))
        np.random.Generator(np.random.PCG64(seed)).shuffle(new_indices)
        self.new_indices = list(new_indices)
    
    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Always use all text tokenizers (typically just one)
        tok_names = self.text_tokenizer_names
        
        ex = self.dataset[int(self.new_indices[idx])]
        ex_text = ex["reconstruction"]  # text reconstruction
        ex_image = ex["generation"]  # PIL image
        
        output = {}
        
        # Process text with text encoders (e.g., GTE)
        for tok_name in tok_names:
            tokenizer_max_length = get_tokenizer_max_length(self.text_tokenizers[tok_name])
            max_length = min(tokenizer_max_length, self.max_length)
            _max_num_chars = max_length * 5
            ex_text_truncated = ex_text[:_max_num_chars]
            
            tt = self.text_tokenizers[tok_name](
                ex_text_truncated,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )
            output.update({f"{tok_name}_{key}": value.flatten() for key, value in tt.items()})
        
        # Process image with image encoders (e.g., CLIP)
        for img_name in self.image_encoder_names:
            if self.image_processors[img_name] is not None:
                # CLIP processor
                img_inputs = self.image_processors[img_name](
                    images=ex_image,
                    return_tensors="pt"
                )
                # Don't flatten pixel_values - keep 4D shape [1, C, H, W]
                # Just squeeze the batch dimension to get [C, H, W]
                for key, value in img_inputs.items():
                    if key == 'pixel_values':
                        output[f"{img_name}_{key}"] = value.squeeze(0)  # [C, H, W]
                    else:
                        output[f"{img_name}_{key}"] = value.flatten()
                # Add image_text_info marker: 0 for image, 1 for text
                # Since this is an image input, mark as 0
                output[f"{img_name}_image_text_info"] = torch.tensor(0)
            else:
                # Fallback: use default transforms if no processor
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize(224),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                       std=[0.26862954, 0.26130258, 0.27577711])
                ])
                img_tensor = transform(ex_image)
                output[f"{img_name}_pixel_values"] = img_tensor
        
        return output