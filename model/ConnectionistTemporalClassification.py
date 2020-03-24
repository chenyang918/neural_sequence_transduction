import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import Conv1d, ReLU


class CTCNetwork(nn.Module):
    def __init__(self):
        super(CTCNetwork, self).__init__()

        self.lstm = nn.LSTM(100,
                            128,
                            num_layers=1,
                            bidirectional=True,
                            batch_first=True)