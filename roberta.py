import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaModel

class RoBERTaClassifier(nn.Module):
    def __init__(self, num_labels, dropout=0.2):
        super(RoBERTaClassifier, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        hidden_size = self.roberta.config.hidden_size
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)  
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        output = self.layer_norm(pooled_output)
        output = self.dropout1(output)
        logits = self.classifier(output)
        return logits
