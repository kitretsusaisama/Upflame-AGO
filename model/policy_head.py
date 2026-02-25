import torch
import torch.nn as nn

class PolicyHead(nn.Module):
    """
    Standard Language Modeling Head (Next Token Prediction) / RL Policy
    """
    def __init__(self, config):
        super().__init__()
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, hidden_states):
        return self.head(hidden_states)

class ToolHead(nn.Module):
    """
    Dedicated head for structured tool interaction.
    Could output JSON-like structure or special tool tokens.
    For now, it projects to a tool-action space.
    """
    def __init__(self, config):
        super().__init__()
        # Assuming a fixed set of tool actions or a sub-vocabulary
        self.tool_dim = 1024 # Placeholder dimension
        self.head = nn.Linear(config.hidden_size, self.tool_dim)

    def forward(self, hidden_states):
        return self.head(hidden_states)

class MemoryWriteHead(nn.Module):
    """
    Head for deciding what to write to persistent memory.
    Outputs a vector to be stored.
    """
    def __init__(self, config):
        super().__init__()
        self.head = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, hidden_states):
        return torch.tanh(self.head(hidden_states))
