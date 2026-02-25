import torch
import torch.nn as nn
import torch.nn.functional as F

class ExpertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.expert_intermediate_size, bias=False)
        self.w2 = nn.Linear(config.expert_intermediate_size, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.expert_intermediate_size, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.experts = nn.ModuleList([ExpertLayer(config) for _ in range(self.num_experts)])
        self.router = nn.Linear(config.hidden_size, self.num_experts, bias=False)

    def forward(self, hidden_states):
        bsz, seq_len, dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, dim)

        router_logits = self.router(hidden_states)
        routing_weights = F.softmax(router_logits, dim=1)

        weights, selected_experts = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True)
        weights = weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros_like(hidden_states)

        # Simple loop routing (can be optimized with scattering/gathering for production)
        for i in range(self.num_experts):
            expert = self.experts[i]
            # Find tokens that selected this expert
            # selected_experts is [tokens, k]
            idx_in_k = (selected_experts == i).nonzero(as_tuple=True)
            # idx_in_k[0] is token indices, idx_in_k[1] is which rank (0 or 1)

            token_indices = idx_in_k[0]
            if len(token_indices) == 0:
                continue

            selected_tokens = hidden_states[token_indices]
            expert_out = expert(selected_tokens)

            # Weighted sum
            # We need the weight for this expert for these tokens
            # weights is [tokens, k]
            # We need weights[token_indices, idx_in_k[1]]
            expert_weights = weights[token_indices, idx_in_k[1]].unsqueeze(1)

            final_hidden_states.index_add_(0, token_indices, expert_out * expert_weights)

        final_hidden_states = final_hidden_states.view(bsz, seq_len, dim)
        return final_hidden_states, router_logits
