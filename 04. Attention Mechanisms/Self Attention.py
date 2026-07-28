import torch
import torch.nn as nn

class SelfAttention(nn.Module):

    def __init__(self, hidden_size):

        super().__init__()

        self.query = nn.Linear(
            hidden_size,
            hidden_size
        )

        self.key = nn.Linear(
            hidden_size,
            hidden_size
        )

        self.value = nn.Linear(
            hidden_size,
            hidden_size
        )

    def forward(self, x, mask=None):

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = torch.bmm(
            Q,
            K.transpose(1, 2)
        ) / (K.size(-1) ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(
                ~mask,
                float("-inf")
            )

        attention = torch.softmax(
            scores,
            dim = -1
        )

        output = torch.bmm(
            attention,
            V
        )

        return output, attention