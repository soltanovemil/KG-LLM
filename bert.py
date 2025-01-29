import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

class BERTClassifier(nn.Module):
    def __init__(self, num_labels, dropout=0.3):
        """
        Initialize BERT classifier.

        Args:
            num_labels: Number of output classes
            dropout: Dropout rate
        """
        super(BERTClassifier, self).__init__()

        # BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Freeze initial layers (optional)
        self.frozen_layers = 8
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
        for i in range(self.frozen_layers):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False

        # Get hidden size from BERT config
        hidden_size = self.bert.config.hidden_size

        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size)

        # Dense layers
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        """
        Forward pass of the model.

        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask

        Returns:
            Output tensor
        """
        # BERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output

        # First dropout and batch norm
        pooled_output = self.dropout1(pooled_output)
        pooled_output = self.batch_norm(pooled_output)

        # Dense layer with ReLU
        pooled_output = F.relu(self.dense(pooled_output))

        # Final dropout and classification
        pooled_output = self.dropout2(pooled_output)
        logits = self.classifier(pooled_output)

        return logits
