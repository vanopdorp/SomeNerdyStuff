import os
import numpy as np
import torch

def build_token_blocks(block_size, split, cache_dir=None):
    from datasets import load_dataset
    import tiktoken

    ds = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
    )

    ds = ds[split]

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    def tokenize(example):
        ids = enc.encode_ordinary(example["text"])
        if ids:
            ids.append(eot)
        return {"ids": ids}

    ds = ds.map(tokenize, remove_columns=ds.column_names, num_proc=os.cpu_count() or 1)
    chunks = [np.asarray(x, dtype=np.uint16) for x in ds["ids"] if len(x) > 0]
    all_ids = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.uint16)

    step = block_size + 1
    n_blocks = len(all_ids) // step
    all_ids = all_ids[: n_blocks * step]
    data = all_ids.reshape(n_blocks, step).astype(np.int64)
    return torch.from_numpy(data)


class BlockDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        block = self.data[idx]
        return block[:-1].clone(), block[1:].clone()
