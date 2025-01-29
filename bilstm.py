import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """Attention mechanism for BiLSTM."""

    def __init__(self, hidden_dim):
        """
        Initialize attention mechanism.

        Args:
            hidden_dim: Size of hidden dimension
        """
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim * 2, 1)

    def forward(self, encoder_outputs):
        """
        Forward pass of attention mechanism.

        Args:
            encoder_outputs: Outputs from encoder

        Returns:
            attention_weighted_encoding: Weighted sum of encoder outputs
            attention_weights: Attention weights
        """
        attention_weights = F.softmax(
            self.attention(encoder_outputs),
            dim=1
        )

        attention_weighted_encoding = torch.sum(
            attention_weights * encoder_outputs,
            dim=1
        )

        return attention_weighted_encoding, attention_weights

class BiLSTMClassifier(nn.Module):
    """BiLSTM classifier with attention."""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, pad_idx,
                 n_layers=2, dropout=0.3):
        """
        Initialize BiLSTM model.

        Args:
            vocab_size: Size of vocabulary
            embedding_dim: Dimension of embeddings
            hidden_dim: Number of hidden units
            output_dim: Number of output classes
            pad_idx: Padding token index
            n_layers: Number of BiLSTM layers
            dropout: Dropout rate
        """
        super(BiLSTMClassifier, self).__init__()

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.embedding_dropout = nn.Dropout(0.3)

        # Layer normalization
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # BiLSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
            batch_first=True
        )

        # Attention mechanism
        self.attention = Attention(hidden_dim)

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)

        # Skip connection
        self.skip_connection = nn.Linear(embedding_dim, hidden_dim * 2)

        # Output layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

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
        embedded = self.layer_norm(embedded)
        embedded = self.embedding_dropout(embedded)

        # Pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # BiLSTM
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Attention
        attended_output, attention_weights = self.attention(output)

        # Skip connection
        skip_connection = self.skip_connection(torch.mean(embedded, dim=1))
        output = attended_output + skip_connection

        # Regularization
        output = self.batch_norm(output)
        output = self.dropout1(output)

        # Final layers
        output = F.relu(self.fc1(output))
        output = self.dropout2(output)
        output = self.fc2(output)

        return output
