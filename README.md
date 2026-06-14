# BabyGPT

A learning project building a GPT-2 style language model from scratch in PyTorch. Each notebook adds one concept on top of the previous, going from a bare transformer to a trained, fine-tuned model that uploads to HuggingFace Hub.

## Structure

```
0_pretrain/     incremental pretraining notebooks (01–12)
1_finetune/     supervised fine-tuning with HuggingFace Trainer
2_finetune_raw/ raw fine-tuning loop
model/          BabyGPT model definition (HF-compatible)
fineweb.py      FinewebEDU dataset preparation and sharding
hellaswag.py    HellaSwag benchmark evaluation
speedrun.sbatch SLURM batch script for HPC cluster runs
```

## Pretraining progression (`0_pretrain/`)

| Notebook | What's added |
|---|---|
| 01 | Load pretrained GPT-2 weights from HuggingFace; run generations |
| 02 | Train from scratch on Shakespeare on a single CUDA GPU |
| 03 | Proper DataLoader + GPT-2 style weight initialization |
| 04 | Mixed precision (bf16), `torch.compile`, Flash Attention |
| 05 | Gradient clipping + cosine LR scheduler with warmup |
| 06 | Weight decay + fused AdamW optimizer |
| 07 | Gradient accumulation to simulate large batch sizes |
| 08 | Distributed Data Parallel (DDP) across multiple GPUs |
| 09 | Switch dataset to FinewebEDU 10B tokens |
| 10 | Validation loss tracking |
| 11 | HellaSwag accuracy benchmark during training |
| 12 | Checkpointing — save and resume training |

## Fine-tuning (`1_finetune/`, `2_finetune_raw/`)

- **`1_finetune/`** — SFT on the TinyStories dataset using the HuggingFace `Trainer` API
- **`2_finetune_raw/`** — same fine-tuning implemented from scratch with a custom shard dataloader and training loop

## Model (`model/BabyGPT.py`)

`BabyGPTForCausalLM` is a GPT-2 sized transformer (124M params: 12 layers, 12 heads, 768 embedding dim) registered as a HuggingFace `PreTrainedModel`. It can be loaded, saved, and pushed to the HuggingFace Hub with standard Transformers APIs.


