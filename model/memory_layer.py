import torch
import torch.nn as nn

class MemoryCompression(nn.Module):
    """
    Compresses hidden states into sparse memory vectors.
    """
    def __init__(self, config):
        super().__init__()
        self.compressor = nn.Linear(config.hidden_size, config.hidden_size // 4)
        self.decompressor = nn.Linear(config.hidden_size // 4, config.hidden_size)

    def forward(self, hidden_states):
        # Simple projection for starter
        compressed = self.compressor(hidden_states)
        return compressed

class VectorMemoryAttention(nn.Module):
    """
    Retrieves information from external vector memory and injects it into the stream.
    """
    def __init__(self, config):
        super().__init__()
        self.query_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.key_proj = nn.Linear(config.hidden_size, config.hidden_size) # Memory keys
        self.value_proj = nn.Linear(config.hidden_size, config.hidden_size) # Memory values
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.scale = config.hidden_size ** -0.5

    def forward(self, hidden_states, memory_vectors):
        # hidden_states: [bs, seq, dim]
        # memory_vectors: [bs, mem_len, dim] (Retrieved top-k)

        query = self.query_proj(hidden_states)
        key = self.key_proj(memory_vectors)
        value = self.value_proj(memory_vectors)

        # Simple Cross Attention
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        context = torch.matmul(attn, value)

        return self.out_proj(context)
