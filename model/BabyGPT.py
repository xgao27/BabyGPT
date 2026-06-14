import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F

from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput
from transformers.generation import GenerationMixin

class BabyGPTConfig(PretrainedConfig):
    model_type = "babygpt"

    def __init__(
        self,
        block_size: int = 1024,
        vocab_size: int = 50304,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        bos_token_id: int = 50256,
        eos_token_id: int = 50256,
        pad_token_id: int | None = None,
        **kwargs,
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            **kwargs,
        )

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd

        self.max_position_embeddings = block_size
        self.hidden_size = n_embd
        self.num_hidden_layers = n_layer
        self.num_attention_heads = n_head
        self.tie_word_embeddings = True


class CausalSelfAttention(nn.Module):
    def __init__(self, config: BabyGPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x, attention_mask=None):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if attention_mask is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            causal_mask = torch.ones((T, T), dtype=torch.bool, device=x.device).tril()
            causal_mask = causal_mask.view(1, 1, T, T)

            key_padding_mask = attention_mask.to(torch.bool).view(B, 1, 1, T)
            combined_mask = causal_mask & key_padding_mask

            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=combined_mask, is_causal=False
            )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config: BabyGPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config: BabyGPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, attention_mask=None):
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class BabyGPTForCausalLM(PreTrainedModel,GenerationMixin):
    config_class = BabyGPTConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = False

    def __init__(self, config: BabyGPTConfig):
        super().__init__(config)

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.post_init()
        self.tie_weights()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5

            torch.nn.init.normal_(module.weight, mean=0.0, std=std)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_input_embeddings(self):
        return self.transformer.wte

    def set_input_embeddings(self, new_embeddings):
        self.transformer.wte = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_lm_head):
        self.lm_head = new_lm_head

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        position_ids=None,
        return_dict=None,
        **kwargs,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is None:
            raise ValueError("You must pass input_ids.")

        B, T = input_ids.size()

        if T > self.config.block_size:
            raise ValueError(
                f"Cannot forward sequence of length {T}; "
                f"block_size is only {self.config.block_size}."
            )

        if position_ids is None:
            position_ids = torch.arange(0, T, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0)

        tok_emb = self.transformer.wte(input_ids)
        pos_emb = self.transformer.wpe(position_ids)
        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x, attention_mask=attention_mask)

        hidden_states = self.transformer.ln_f(x)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=logits,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        attention_mask=None,
        **kwargs,
    ):
        if input_ids.size(1) > self.config.block_size:
            input_ids = input_ids[:, -self.config.block_size :]

            if attention_mask is not None:
                attention_mask = attention_mask[:, -self.config.block_size :]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        decay_params = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for _, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            fused=use_fused,
        )

        return optimizer