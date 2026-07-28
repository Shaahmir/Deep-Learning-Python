import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):

    def __init__(self, hidden_size, num_heads):

        super().__init__()

        assert hidden_size % num_heads == 0

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

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

        self.out = nn.Linear(
            hidden_size,
            hidden_size
        )

    def forward(self, x, mask = None):

        batch_size, seq_len, _ = x.size()

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        Q = Q.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        K = K.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        V = V.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        scores = torch.matmul(
            Q,
            K.transpose(-2, -1)
        ) / (self.head_dim ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(
                ~mask,
                float("-inf")
            )

        attention = torch.softmax(
            scores,
            dim = -1
        )

        output = torch.matmul(
            attention,
            V
        )

        output = output.transpose(
            1,
            2
        ).contiguous().view(
            batch_size,
            seq_len,
            self.hidden_size
        )

        output = self.out(output)

        return output, attention