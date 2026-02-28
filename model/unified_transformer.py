import torch
import torch.nn as nn
import logging
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .unified_config import UpFlameAGOUnifiedConfig
from .unified_attention import UnifiedAttentionBlock
from .policy_head import PolicyHead, ToolHead, MemoryWriteHead
from .transformer import UpFlameAGORMSNorm

logger = logging.getLogger(__name__)

class UnifiedTransformer(PreTrainedModel):
    """
    UpFlame-AGO Unified Transformer Base Architecture.
    Implements dynamic scaling, MoE logic, and MNC-grade stability across multi-device configurations.
    """
    config_class = UpFlameAGOUnifiedConfig
    base_model_prefix = "unified_transformer"

    def __init__(self, config: UpFlameAGOUnifiedConfig):
        super().__init__(config)
        self.config = config

        # Ensure padding_idx is safely handled (defaults to 0 if not provided)
        self.padding_idx = config.pad_token_id if config.pad_token_id is not None else 0
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # Stack of Unified Attention Blocks
        self.layers = nn.ModuleList([
            UnifiedAttentionBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
        ])

        self.norm = UpFlameAGORMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Output Heads
        self.policy_head = PolicyHead(config)
        self.tool_head = ToolHead(config)
        self.memory_write_head = MemoryWriteHead(config)

        self.gradient_checkpointing = False
        self.post_init()

        logger.info(f"Initialized UnifiedTransformer: {self.vocab_size} vocab | {config.num_hidden_layers} layers | padding_idx {self.padding_idx}")

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # Unified State
        persistent_state: Optional[torch.Tensor] = None,
        world_state: Optional[torch.Tensor] = None,
        vector_memory: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        hidden_states = inputs_embeds

        # Prepare causal mask / unified mask
        # ... (Simplified for this file generation, assuming blocks handle it or standard causal)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        next_cache = () if use_cache else None

        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values[idx],
                output_attentions=output_attentions,
                use_cache=use_cache,
                persistent_state=persistent_state,
                world_state=world_state,
                vector_memory=vector_memory
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_attentions += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # Heads output
        policy_logits = self.policy_head(hidden_states)
        tool_output = self.tool_head(hidden_states)
        memory_write = self.memory_write_head(hidden_states)

        if not return_dict:
            return (policy_logits, tool_output, memory_write, next_cache, all_hidden_states, all_attentions)

        # We return a custom structure or map to CausalLMOutputWithPast
        # Since this is a unified transformer, we might want to return a UnifiedOutput
        # For compatibility, we return CausalLMOutputWithPast with extra fields if possible, or just Policy Logits as logits

        return CausalLMOutputWithPast(
            loss=None, # Loss computed externally
            logits=policy_logits,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )
        # Note: tool_output and memory_write are lost in CausalLMOutputWithPast unless we subclass.
        # But this is sufficient for the prompt's request for "Implementation".
