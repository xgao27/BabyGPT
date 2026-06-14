import os
import math
from tqdm import tqdm
from huggingface_hub import login
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, set_seed, BitsAndBytesConfig
from datasets import load_dataset, Dataset, DatasetDict
import wandb
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import time
from transformers import TrainerCallback

device = "cuda" if torch.cuda.is_available() else "cpu"

# log in to wandb and hugging face
wb_token = "" # removed in public repo
hf_token = "" # removed in public repo
wandb.login(key=wb_token)
login(hf_token, add_to_git_credential=True)

# repo information
BASE_MODEL = "littleBallOfFur/baby-gpt-base"
PROJECT_NAME = "baby-gpt-sft-tinystories"
RUN_NAME =  f"{datetime.now():%Y-%m-%d_%H.%M.%S}"
PROJECT_RUN_NAME = f"{PROJECT_NAME}-{RUN_NAME}" # for wandb run
HUB_SUFFIX = "-v5" # save to hub under this
HUB_MODEL_NAME = f"littleBallOfFur/{PROJECT_NAME}{HUB_SUFFIX}"
# HUB_MODEL_NAME = f"littleBallOfFur/{PROJECT_NAME}"
DATASET_NAME = "karpathy/tinystories-gpt4-clean"

# "littleBallOfFur/baby-gpt-sft-tinystories-dnf": did not finish, step 28500, 85%, wandb 2026-04-24_20.04.12
# "littleBallOfFur/baby-gpt-sft-tinystories-v1": r = 16, attn only, wandb 2026-04-24_22.15.40
# "littleBallOfFur/baby-gpt-sft-tinystories-v2": r = 32, attn only, wandb 2026-04-25_04.15.36
# "littleBallOfFur/baby-gpt-sft-tinystories-v3": r = 32, attn + mlp, wandb 2026-04-25_07.28.57
# "littleBallOfFur/baby-gpt-sft-tinystories-v4": r = 64, attn + mlp, wandb 2026-04-25_10.02.20


# load BabyGPT base model 
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL,trust_remote_code=True,)
# base_model.to(device)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL,trust_remote_code=True,)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
base_model.config.pad_token_id = tokenizer.pad_token_id

# get the (clean) tinystories dataset
raw_dataset = load_dataset(DATASET_NAME)
# split into 80% train, 10% validation, 10% test: (2186107, 273263, 273264)
split_1 = raw_dataset['train'].train_test_split(test_size=0.1, seed=42)
train_dataset = split_1["train"]
split_2 = split_1["test"].train_test_split(test_size=0.5, seed=42)
test_dataset = split_2["test"]
val_dataset = split_2["train"]

# filter for shorter stories (most of them are under 256)
MAX_TOKENS = 512
def keep_short(batch):
    texts = batch["text"]
    lengths = [
        len(ids)
        for ids in tokenizer(texts, add_special_tokens=True)["input_ids"]
    ]
    return [n <= MAX_TOKENS for n in lengths]

train_dataset = train_dataset.shuffle(seed=42).filter(keep_short, batched=True)
# use a smaller validation set
val_dataset = val_dataset.shuffle(seed=42).select(range(5000)).filter(keep_short, batched = True).select(range(2048))


EPOCHS = 1 # 2186107/64 = 34k steps, 68 saves
BATCH_SIZE = 64
MAX_SEQUENCE_LENGTH = 512
GRADIENT_ACCUMULATION_STEPS = 1

LORA_R = 32 * 2
TARGET_MODULES = ["attn.c_attn","attn.c_proj","mlp.c_fc","mlp.c_proj"]
LORA_DROPOUT = 0.1
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.01
LR_SCHEDULER_TYPE = 'cosine'
WEIGHT_DECAY = 0.01

use_cuda = torch.cuda.is_available()
OPTIMIZER = "paged_adamw_32bit" if use_cuda else "adamw_torch"

use_bf16 = False
if use_cuda:
    capability = torch.cuda.get_device_capability()
    use_bf16 = capability[0] >= 8

LOG_STEPS = 10 
SAVE_STEPS = 500 
LOG_TO_WANDB = True
EVAL_STEPS = 500



class TimingCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start = time.time()
        self.epoch_start = None
        self.step_start = None
        self.step_times = []

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start = time.time()
        print(f"starting epoch {state.epoch}")

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        if self.step_start is None:
            return
        dt = time.time() - self.step_start
        self.step_times.append(dt)

        avg_step = sum(self.step_times) / len(self.step_times)
        elapsed = time.time() - self.train_start

        if state.max_steps and state.max_steps > 0:
            remaining_steps = state.max_steps - state.global_step
            eta_seconds = remaining_steps * avg_step
            eta_str = time.strftime("%H:%M:%S", time.gmtime(max(0, eta_seconds)))
        else:
            eta_str = "unknown"

        if state.global_step % 10 == 0:
            print(
                f"step {state.global_step}: "
                f"{dt:.2f}s | avg_step {avg_step:.2f}s | elapsed {elapsed/60:.1f}m | eta {eta_str}"
            )

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.epoch_start is None:
            return
        epoch_time = time.time() - self.epoch_start
        print(f"epoch {state.epoch} finished in {epoch_time/60:.2f} min")

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.train_start
        avg_step = sum(self.step_times) / len(self.step_times) if self.step_times else 0
        print(f"training finished in {total_time/60:.2f} min")
        print(f"average step time: {avg_step:.2f}s")

lora_parameters = LoraConfig(
    lora_alpha=LORA_R*2,
    lora_dropout=LORA_DROPOUT,
    r=LORA_R,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=TARGET_MODULES,
)

# Training parameters
train_parameters = SFTConfig(
    output_dir=PROJECT_RUN_NAME,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    optim=OPTIMIZER,
    save_steps=SAVE_STEPS,
    save_total_limit=10,
    logging_steps=LOG_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.001,
    fp16=use_cuda and not use_bf16,
    bf16=use_bf16,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=WARMUP_RATIO,
    # packing=True,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    report_to="wandb" if LOG_TO_WANDB else 'none',
    run_name=RUN_NAME,
    max_length=MAX_SEQUENCE_LENGTH,
    save_strategy="steps",
    hub_strategy="every_save",
    push_to_hub=True,
    hub_model_id=HUB_MODEL_NAME,
    hub_private_repo=True,
    eval_strategy="steps",
    eval_steps=EVAL_STEPS,
    dataset_text_field="text",
    gradient_checkpointing=False,
)

fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    peft_config=lora_parameters,
    args=train_parameters,
    processing_class=tokenizer,
    # callbacks=[TimingCallback()],
)

if LOG_TO_WANDB:
    os.environ["WANDB_PROJECT"] = PROJECT_NAME
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_WATCH"] = "false"

    wandb.init(project=PROJECT_NAME, name=RUN_NAME)

fine_tuning.train()

if LOG_TO_WANDB:
  wandb.finish()