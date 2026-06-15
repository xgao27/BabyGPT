# BabyGPT

A learning project building a GPT-2 style language model from scratch in PyTorch. Each notebook adds one concept on top of the previous, going from a bare transformer to a trained, fine-tuned model that uploads to HuggingFace Hub. All the progressive steps are then combined into standalone training scripts for pretraining and finetuning. The pretraining stage follows the [NanoGPT tutorial](https://github.com/karpathy/build-nanogpt) by Andrej Karpathy.

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

## Demo — story relay

Four LoRA fine-tuned checkpoints (v1–v4, trained on TinyStories with increasing LoRA rank) take turns continuing a story 16 tokens at a time (`1_finetune/eval_babygpt_tinystories.ipynb`).

**Prompt:** *A big fat squirrel flies*

> **v1:** into the sky. One by one, we see a rainbow coming through the sky
>
> **v2:** . It is so colorful it reminds us that a strong storm is about to fall
>
> **v3:** . We zoom and zoom and our eyes go dark for a while. Then,
>
> **v4:** something wonderful happens. The sun's light shines and a small golden sun pops into
>
> **v1:** the sky. It is so bright it gives us a great feeling of love and
>
> **v2:** happiness. At first, it was a gray, cloudless cloud. It
>
> **v3:** was so nice and bright that everyone wanted to catch it. But then, something
>
> **v4:** unexpected happened. The big fat squirrel saw a shiny rock in the sky. He
>
> *(... continues for 315 tokens total)*

**Full story:**

> A big fat squirrel flies into the sky. One by one, we see a rainbow coming through the sky. It is so colorful it reminds us that a strong storm is about to fall. We zoom and zoom and our eyes go dark for a while. Then, something wonderful happens. The sun's light shines and a small golden sun pops into the sky. It is so bright it gives us a great feeling of love and happiness.
>
> At first, it was a gray, cloudless cloud. It was so nice and bright that everyone wanted to catch it. But then, something unexpected happened. The big fat squirrel saw a shiny rock in the sky. He noticed it was a magical spark. He got to touch the spark and then he got the magical spark. He laughed and got the spark. His enthusiasm was so high that he dared the clouds by running away. The sun was not very strong now and so he turned the wind to save the day.
>
> When the sun went down, the rain was gone and the sun started to light up the little world. It was magical because it was all powered by the magic. When the sun went back, the sky lit up and the rainbow moved again. In the end, both friends were so happy. They zipped to their houses to hug the sun for a huge hug. They found their way home together in shining colors. The rain was gone too.
>
> With their great spark and their beautiful rainbow, the big fat squirrel and the little world of suns and rain, they had a big happiness.


