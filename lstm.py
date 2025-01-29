import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx,
                 n_layers=2, dropout=0.3):
        """
        Initialize LSTM model.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Number of hidden units
            output_dim: Number of output classes
            pad_idx: Padding token index
            n_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(LSTMClassifier, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.embedding_dropout = nn.Dropout(0.2)

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(embedding_dim)

        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout * 0.5)

        # Batch normalization
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)

        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        # Layer normalization
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, text, text_lengths):
        """
        Forward pass of the model.

        Args:
            text: Input text tensor
            text_lengths: Length of each sequence

        Returns:
            Output tensor
        """
        # Embedding
        embedded = self.embedding(text)
        embedded = self.layer_norm1(embedded)
        embedded = self.embedding_dropout(embedded)

        # Pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # LSTM
        packed_output, (hidden, _) = self.lstm(packed_embedded)
        output = self.dropout1(hidden[-1])

        # First fully connected layer
        output = self.batch_norm1(output)
        output = self.fc1(output)
        output = F.relu(output)

        # Layer normalization and dropout
        output = self.layer_norm2(output)
        output = self.dropout2(output)

        # Final batch norm and output layer
        output = self.batch_norm2(output)
        output = self.fc2(output)

        return output
