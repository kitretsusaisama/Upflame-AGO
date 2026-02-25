import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import UpFlameAGORotaryEmbedding, apply_rotary_pos_emb

class InfiniAttention(nn.Module):
    """
    Infini-Attention mechanism as described in Google's paper.
    Combines local masked attention with a compressive memory matrix.

    A = softmax(Q K^T) V + sigma(Q) M_{prev}
    M_{new} = M_{prev} + sigma(K)^T V
    """
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.segment_len = config.segment_len

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Gating for mixing local and memory attention
        self.beta = nn.Parameter(torch.zeros(1))

        self.rotary_emb = UpFlameAGORotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta
        )

    def forward(self, hidden_states, attention_mask, position_ids, past_key_value, output_attentions, use_cache):
        bsz, seq_len, _ = hidden_states.shape

        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = query.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # RoPE
        cos, sin = self.rotary_emb(value, seq_len=seq_len)
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # 1. Local Attention (Standard SDPA)
        local_context = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, is_causal=True)

        # 2. Compressive Memory Attention (Infini)
        # Assuming past_key_value contains the memory matrix M from previous segment
        memory_state = None
        if past_key_value is not None and len(past_key_value) > 2:
            memory_state = past_key_value[2] # M

        if memory_state is None:
            memory_state = torch.zeros(bsz, self.num_heads, self.head_dim, self.head_dim, device=query.device, dtype=query.dtype)

        # Retrieve from memory: sigma(Q) M
        sigma_q = F.elu(query) + 1.0
        memory_output = torch.matmul(sigma_q, memory_state)

        # Update memory: M_new = M + sigma(K)^T V
        sigma_k = F.elu(key) + 1.0
        # Check shapes: K is [b, h, s, d], V is [b, h, s, d]
        # We want to sum over s to get [b, h, d, d]
        update = torch.matmul(sigma_k.transpose(-2, -1), value)
        new_memory_state = memory_state + update

        # Combine
        beta = torch.sigmoid(self.beta)
        output = beta * local_context + (1 - beta) * memory_output

        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        output = self.o_proj(output)

        present_key_value = (key, value, new_memory_state) if use_cache else None

        return output, None, present_key_value
