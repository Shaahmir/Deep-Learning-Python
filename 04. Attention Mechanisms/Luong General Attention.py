import torch
import torch.nn as nn

class LuongAttention(nn.Module):

    def __init__(self, encoder_hidden_size, decoder_hidden_size):

        super().__init__()

        self.W = nn.Linear(
            decoder_hidden_size,
            encoder_hidden_size * 2,
            bias = False
        )

    def forward(self, hidden, encoder_outputs, mask = None):

        hidden = self.W(hidden).unsqueeze(2)

        scores = torch.bmm(
            encoder_outputs,
            hidden
        ).squeeze(2)

        if mask is not None:
            scores = scores.masked_fill(
                ~mask,
                float("-inf")
            )

        attention = torch.softmax(
            scores,
            dim = 1
        )

        return attention