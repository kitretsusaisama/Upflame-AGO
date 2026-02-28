from transformers import PretrainedConfig

class UpFlameAGOUnifiedConfig(PretrainedConfig):
    model_type = "upflame_ago_unified"
    # Global toggle for disabling advanced components (useful for Colab baselines)
    USE_ADVANCED = True


    # Global toggle for disabling advanced components (useful for Colab baselines)
    USE_ADVANCED = True

    def __init__(
        self,
        vocab_size=65536,
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=11008,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        # MoE
        use_moe=True,
        num_experts=16,
        num_experts_per_tok=2,
        expert_intermediate_size=None,
        # Infini-Attention & Memory
        segment_len=2048,
        memory_size=1024, # Vectors in persistent memory
        memory_dim=None, # Defaults to hidden_size
        use_infini_attention=True,
        use_vector_memory=True,
        use_world_state=True,
        **kwargs,
    ):

        # Override advanced features if global toggle is disabled
        if not self.__class__.USE_ADVANCED:
            use_moe = False
            use_infini_attention = False
            use_vector_memory = False
            use_world_state = False
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.expert_intermediate_size = expert_intermediate_size or intermediate_size

        self.segment_len = segment_len
        self.memory_size = memory_size
        self.memory_dim = memory_dim or hidden_size
        self.use_infini_attention = use_infini_attention
        self.use_vector_memory = use_vector_memory
        self.use_world_state = use_world_state

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
