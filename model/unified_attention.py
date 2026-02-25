import torch
import torch.nn as nn
from typing import Optional, Tuple
from .transformer import UpFlameAGORMSNorm
from .infini_attention import InfiniAttention
from .moe_layer import MoELayer
from .memory_layer import MemoryCompression, VectorMemoryAttention

class UnifiedAttentionBlock(nn.Module):
    """
    Unified Attention Block combining:
    1. Local Attention
    2. Segment Attention (via InfiniAttention's recurrence or separate)
    3. Infini-Attention (Infinite Context)
    4. Vector Memory Attention (External)
    5. World State Attention
    """
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # 1. & 3. Local + Infini Attention
        # InfiniAttention handles both local sliding window and long-term compressed state
        self.attention = InfiniAttention(config, layer_idx)

        # 4. Vector Memory Attention
        self.vector_attn = VectorMemoryAttention(config)

        # 5. World State Attention (Simplified as Cross Attention to World State)
        if config.use_world_state:
            self.world_state_attn = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads // 4, batch_first=True)
        else:
            self.world_state_attn = None

        # Feed Forward (MoE)
        self.moe = MoELayer(config)

        self.input_layernorm = UpFlameAGORMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = UpFlameAGORMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_moe_layernorm = UpFlameAGORMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Memory Compression Layer
        self.memory_compression = MemoryCompression(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        persistent_state: Optional[torch.Tensor] = None,
        world_state: Optional[torch.Tensor] = None,
        vector_memory: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Unified Attention (Infini + Local)
        # Note: InfiniAttention manages its own KV cache and memory state
        attn_output, attn_weights, present_key_value = self.attention(
            hidden_states,
            attention_mask,
            position_ids,
            past_key_value,
            output_attentions,
            use_cache
        )

        # Vector Memory Injection
        if vector_memory is not None:
            vec_out = self.vector_attn(hidden_states, vector_memory)
            attn_output = attn_output + vec_out

        # World State Injection
        if self.world_state_attn is not None and world_state is not None:
            # Cross attend to world state
            # world_state: [bs, state_len, dim]
            ws_out, _ = self.world_state_attn(hidden_states, world_state, world_state)
            attn_output = attn_output + ws_out

        hidden_states = residual + attn_output

        # MoE Feed Forward
        residual = hidden_states
        hidden_states = self.pre_moe_layernorm(hidden_states)
        hidden_states, router_logits = self.moe(hidden_states)
        hidden_states = residual + hidden_states

        # Memory Compression & Update (Side effect or return)
        # In a real recurrent model, we would update the persistent state here.
        # For this block, we assume InfiniAttention updated the fast weight memory.
        # Global compression might happen at layer output or separate head.

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        if use_cache:
            outputs += (present_key_value,)

        return outputs
