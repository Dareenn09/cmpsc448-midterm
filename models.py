"""CNN and RNN (LSTM) text classifiers."""
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class TextCNN(nn.Module):
    """CNN: one 1D convolution of a single width + max-over-time pooling."""

    def __init__(self, vocab_size, num_classes, emb_dim=128, num_filters=128,
                 kernel_size=3, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.conv = nn.Conv1d(emb_dim, num_filters, kernel_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x, lengths):
        e = self.embedding(x).transpose(1, 2)            # (B, emb, T)
        h = torch.relu(self.conv(e)).max(dim=2).values    # (B, filters)
        return self.fc(self.dropout(h))


class LSTMClassifier(nn.Module):
    """Unidirectional LSTM; the final hidden state summarizes the whole text."""

    def __init__(self, vocab_size, num_classes, emb_dim=128, hidden=128,
                 num_layers=1, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=num_layers,
                            batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x, lengths):
        e = self.dropout(self.embedding(x))
        # Packing makes the LSTM skip padding tokens
        packed = pack_padded_sequence(e, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h = h_n[-1]                                       # last layer's final hidden state
        return self.fc(self.dropout(h))
