"""
Save the tinystories dataset as tokenized shards.
Create DataLoader that loads the saved shards.
"""

import os
import multiprocessing as mp
import numpy as np
from transformers import GPT2TokenizerFast
from datasets import load_dataset 
from tqdm import tqdm 
import time
import torch
import random
from torch.utils.data import IterableDataset, DataLoader

class ShardDataset(IterableDataset):
    """
    Iterable dataset from tokenized shards.
    Each shard is npy array of 1e6 token ids.
    Each example is
        x = tokens[i: i + block_size]
        y = tokens[i + 1: i + block_size + 1]
    """
    def __init__(
        self, 
        shard_dir,
        split = "train", 
        block_size = 1024, 
        shuffle = True, 
        seed = 1337,
        infinite = False
    ):
        super().__init__()

        self.shard_dir = shard_dir
        self.split = split
        self.block_size = block_size
        self.shuffle = shuffle
        self.seed = seed
        self.infinite = infinite
        shard_file_name = sorted([file for file in os.listdir(self.shard_dir) if ".npy" in file])
        
        if self.split == "train":
            # files except for _000000.npy
            self.shard_paths = [os.path.join(self.shard_dir,s) for s in shard_file_name if "_000000.npy" not in s]
        else: 
            # file _000000.npy is validation set
            self.shard_paths = [os.path.join(self.shard_dir,s) for s in shard_file_name if "_000000.npy" in s]
        if len(self.shard_paths) == 0:
            raise FileNotFoundError(f"No *.npy files found in {self.shard_dir}")
        _tokens = np.load(self.shard_paths[0],mmap_mode='r')
        seq_per_shard = _tokens.shape[0] // self.block_size # number of sequences per shard
        self._total_length = seq_per_shard * len(self.shard_paths)
        # print(_tokens.shape[0], shard_length, self._total_length)
    
    def __len__(self):
        return self._total_length
         

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        rng = random.Random(self.seed + worker_id)
        shard_paths = self.shard_paths[worker_id::num_workers]
        if len(shard_paths) == 0:
            raise RuntimeError(
                f"Worker {worker_id} got no shards. "
                f"num_workers={num_workers}, num_shards={len(self.shard_paths)}"
            )
        while True:
            if self.shuffle:
                rng.shuffle(shard_paths)
            # for p in shard_paths:
            #     print(p)
            for shard_path in shard_paths:
                tokens = np.load(shard_path,mmap_mode='r').astype(np.int32)
                if len(tokens) <= self.block_size + 1:
                    continue
                max_start = len(tokens) - (self.block_size + 1)
                start_indices = list(range(0, max_start,self.block_size))
                if self.shuffle: 
                    rng.shuffle(start_indices)
                for start in start_indices:
                    chunk = tokens[start: start + self.block_size + 1]
                    x = torch.tensor(chunk[:-1], dtype=torch.long)
                    y = torch.tensor(chunk[1:], dtype=torch.long)
                    yield x, y
            if not self.infinite: # not infinite dataloader, for epoch control
                break

    """
    Iterable dataset from tokenized shards.
    Each shard is npy array of 1e6 token ids.
    Each example is
        x = tokens[i: i + block_size]
        y = tokens[i + 1: i + block_size + 1]
    """
    def __init__(
        self, 
        shard_dir,
        split = "train", 
        block_size = 1024, 
        shuffle = True, 
        seed = 1337
    ):
        super().__init__()

        self.shard_dir = shard_dir
        self.split = split
        self.block_size = block_size
        self.shuffle = shuffle
        self.seed = seed
        shard_file_name = sorted([file for file in os.listdir(self.shard_dir) if ".npy" in file])
        
        if self.split == "train":
            # files except for _000000.npy
            self.shard_paths = [os.path.join(self.shard_dir,s) for s in shard_file_name if "_000000.npy" not in s]
        else: 
            # file _000000.npy is validation set
            self.shard_paths = [os.path.join(self.shard_dir,s) for s in shard_file_name if "_000000.npy" in s]
        if len(self.shard_paths) == 0:
            raise FileNotFoundError(f"No *.npy files found in {self.shard_dir}")
        _tokens = np.load(self.shard_paths[0],mmap_mode='r')
        seq_per_shard = _tokens.shape[0] // self.block_size # number of sequences per shard
        self._total_length = seq_per_shard * len(self.shard_paths) # total number of sequences in all shards
    
    def __len__(self):
        return self._total_length

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        rng = random.Random(self.seed + worker_id)
        shard_paths = self.shard_paths[worker_id::num_workers]
        if len(shard_paths) == 0:
            raise RuntimeError(
                f"Worker {worker_id} got no shards. "
                f"num_workers={num_workers}, num_shards={len(self.shard_paths)}"
            )
        while True:
            if self.shuffle:
                rng.shuffle(shard_paths)
            # for p in shard_paths:
            #     print(p)
            for shard_path in shard_paths:
                tokens = np.load(shard_path,mmap_mode='r').astype(np.int32)
                if len(tokens) <= self.block_size + 1:
                    continue
                max_start = len(tokens) - (self.block_size + 1)
                start_indices = list(range(0, max_start,self.block_size))
                if self.shuffle: 
                    rng.shuffle(start_indices)
                for start in start_indices:
                    chunk = tokens[start: start + self.block_size + 1]
                    x = torch.tensor(chunk[:-1], dtype=torch.long)
                    y = torch.tensor(chunk[1:], dtype=torch.long)
                    yield x, y

def create_dataloader(
    shard_dir,
    split = "train", 
    batch_size = 32, 
    block_size = 1024,
    shuffle = True, 
    seed = 1337
):
    shard_dataset = ShardDataset(
        shard_dir = shard_dir, 
        split = split, 
        block_size = block_size , 
        shuffle = shuffle, 
        seed = seed
    )
    shard_dataloader = DataLoader(
        shard_dataset,
        batch_size = batch_size
    )
    return shard_dataloader

def test_dataloader():
    local_data_dir = "tinystories"
    DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..",local_data_dir)
    train_dataloader = create_dataloader(
        shard_dir=DATA_CACHE_DIR, 
        split="train", 
        batch_size=4,
        block_size=8
    )
    val_dataloader = create_dataloader(
        shard_dir=DATA_CACHE_DIR, 
        split="val", 
        shuffle=False, 
        batch_size=4,
        block_size=8
    )
    print("testing the dataloader -- training set")
    x, y = next(iter(train_dataloader))
    print(x.shape)
    print(y.shape)
    print(x.dtype, y.dtype)
    print(x)
    print(y)
    print("testing the dataloader -- validation set")
    x, y = next(iter(val_dataloader))
    print(x.shape)
    print(y.shape)
    print(x.dtype, y.dtype)
    print(x)
    print(y)

def save_shard(tokens_np, out_path):
    np.save(out_path, tokens_np)
    print(f"Saved {out_path} | shape={tokens_np.shape} | dtype={tokens_np.dtype}")

def load_tokens_from_shard(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

def main():
    # Local data directory
    local_data_dir = "tinystories"
    DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..",local_data_dir)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # Download dataset
    DATASET_NAME = "karpathy/tinystories-gpt4-clean"
    raw_dataset = load_dataset(DATASET_NAME,split="train")

    # Download dataset and save as shards
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    eos_id = tokenizer.eos_token_id

    shard_size = int(1e6)
    max_shards = 10

    example_seen = 0
    shard_index = 0
    shard_token_count = 0
    shard_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    progress_bar = tqdm(total=max_shards * shard_size, desc="Tokenizing")
    for example in raw_dataset:
        example_seen += 1
        text = example['text']
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        token_ids.append(eos_id)
        if shard_token_count + len(token_ids) <= shard_size:
            # still has room left
            shard_tokens_np[shard_token_count:shard_token_count+len(token_ids)] = token_ids
            shard_token_count += len(token_ids)
            progress_bar.update(len(token_ids))
        else:
            # save remainder and start new shard
            remainder = shard_size - shard_token_count
            shard_tokens_np[shard_token_count:] = token_ids[:remainder]
            progress_bar.update(remainder)
            out_path = os.path.join(DATA_CACHE_DIR, f"tinystories_{shard_index:06d}")
            save_shard(shard_tokens_np,out_path)
            # shard_token_count + remainder = shard_size, reset to 0
            shard_token_count = 0 

            if shard_index >= max_shards-1:
                # max number of shards reached, stop
                progress_bar.close()
                print(f"Finished with {max_shards} shards.")
                break
            
            shard_index += 1
            overflow = len(token_ids) - remainder
            shard_tokens_np[:overflow] = token_ids[remainder:]
            shard_token_count = overflow
            progress_bar.update(overflow)

    if shard_token_count > 0:
        print("Save the last partial shard")
        out_path = os.path.join(DATA_CACHE_DIR, f"tinystories_{shard_index:06d}")
        save_shard(shard_tokens_np[:shard_token_count],out_path)

    shard_file_name = sorted([file for file in os.listdir(DATA_CACHE_DIR) if ".npy" in file])
    saved_shards = [os.path.join(DATA_CACHE_DIR,s) for s in shard_file_name]
    print(f"\nProcessed {example_seen} examples.")
    for s in saved_shards:
        print(s)

if __name__ == "__main__":
    main()
    # test_dataloader()
    