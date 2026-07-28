import torch
import torch.nn as nn

class LuongAttention(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, hidden, encoder_outputs, mask = None):

        hidden = hidden.unsqueeze(2)

        scores = torch.bmm(
            encoder_outputs,
            hidden
        ).squeeze(-1)

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