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
        self.use_infini_attention = config.use_infini_attention
        if self.use_infini_attention:
            self.attention = InfiniAttention(config, layer_idx)
        else:
            from .attention import UpFlameAGOAttention
            self.attention = UpFlameAGOAttention(config, layer_idx)

        # 4. Vector Memory Attention
        if config.use_vector_memory:
            self.vector_attn = VectorMemoryAttention(config)
        else:
            self.vector_attn = None

        # 5. World State Attention (Simplified as Cross Attention to World State)
        if config.use_world_state:
            self.world_state_attn = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads // 4, batch_first=True)
        else:
            self.world_state_attn = None

        # Feed Forward (MoE vs Standard)
        self.use_moe = config.use_moe
        if self.use_moe:
            self.feed_forward = MoELayer(config)
        else:
            from .moe_layer import ExpertLayer
            self.feed_forward = ExpertLayer(config) # Acts as standard MLP when MoE is disabled

        self.input_layernorm = UpFlameAGORMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = UpFlameAGORMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_moe_layernorm = UpFlameAGORMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Memory Compression Layer
        if config.use_vector_memory or config.use_infini_attention:
            self.memory_compression = MemoryCompression(config)
        else:
            self.memory_compression = None

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

        # Unified Attention (Infini + Local vs Standard)
        if self.use_infini_attention:
            attn_output, attn_weights, present_key_value = self.attention(
                hidden_states,
                attention_mask,
                position_ids,
                past_key_value,
                output_attentions,
                use_cache
            )
        else:
            attn_output, attn_weights, present_key_value = self.attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache
            )

        # Vector Memory Injection
        if self.vector_attn is not None and vector_memory is not None:
            vec_out = self.vector_attn(hidden_states, vector_memory)
            attn_output = attn_output + vec_out

        # World State Injection
        if self.world_state_attn is not None and world_state is not None:
            # Cross attend to world state
            # world_state: [bs, state_len, dim]
            ws_out, _ = self.world_state_attn(hidden_states, world_state, world_state)
            attn_output = attn_output + ws_out

        hidden_states = residual + attn_output

        # Feed Forward
        residual = hidden_states
        hidden_states = self.pre_moe_layernorm(hidden_states)

        if self.use_moe:
            hidden_states, router_logits = self.feed_forward(hidden_states)
        else:
            hidden_states = self.feed_forward(hidden_states)

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
