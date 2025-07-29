import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.fc2(out)
        return out

class MoELayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts):
        super(MoELayer, self).__init__()
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim) for _ in range(num_experts)])
        self.gate = nn.Linear(input_dim, num_experts)
        self.num_experts = num_experts

    def forward(self, x):
        # x shape: [batch_size, sequence_length, input_dim]
        batch_size, sequence_length, input_dim = x.size()

        # Compute gating scores for each expert
        gate_logits = self.gate(x)  # shape: [batch_size, sequence_length, num_experts]
        gate_probs = F.softmax(gate_logits, dim=-1)  # shape: [batch_size, sequence_length, num_experts]

        # Route tokens to experts based on gate probabilities
        expert_outputs = []
        for expert_idx in range(self.num_experts):
            # Mask for current expert
            expert_mask = (torch.argmax(gate_probs, dim=-1) == expert_idx).unsqueeze(-1)
            # Extract tokens routed to this expert
            expert_input = x * expert_mask

            # Forward pass through the expert
            expert_output = self.experts[expert_idx](expert_input)
            expert_outputs.append(expert_output)

        # Combine expert outputs
        combined_output = sum(expert_outputs)
        expert_indices = torch.argmax(gate_probs, dim=-1)

        return combined_output, gate_probs, expert_indices

def load_balancing_loss(gate_probs, expert_indices, num_experts, alpha=0.01):
    batch_size, sequence_length, _ = gate_probs.size()

    # Calculate f_i: fraction of tokens routed to expert i
    token_counts = torch.zeros(num_experts, device=gate_probs.device)
    for expert_idx in range(num_experts):
        expert_mask = (expert_indices == expert_idx)
        token_counts[expert_idx] = expert_mask.sum() / (batch_size * sequence_length)

    # Calculate P_i: average routing probability for expert i
    P_i = gate_probs.mean(dim=(0, 1))  # Average over batch and sequence dimensions

    # Load balancing loss
    loss = num_experts * torch.dot(token_counts, P_i)

    return alpha * loss