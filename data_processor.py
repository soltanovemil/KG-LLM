import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Union
from sklearn.model_selection import train_test_split
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from transformers import AutoTokenizer
from config.model_config import DATA_PARAMS

class DataProcessor:
    """Class for processing and preparing data for models."""

    def __init__(self):
        """Initialize DataProcessor."""
        self.tokenizer = get_tokenizer('basic_english')
        self.vocab = None
        self.label2id = None
        self.id2label = None

    def create_label_mappings(self, labels: List[str]) -> None:
        """
        Create label to ID mappings.

        Args:
            labels: List of string labels
        """
        unique_labels = sorted(list(set(labels)))
        self.label2id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

    def build_vocab(self, texts: List[str], min_freq: int = 2) -> None:
        """
        Build vocabulary from texts.

        Args:
            texts: List of input texts
            min_freq: Minimum frequency for vocabulary words
        """
        def yield_tokens(data_iter):
            for text in data_iter:
                yield self.tokenizer(text.lower())

        self.vocab = build_vocab_from_iterator(
            yield_tokens(texts),
            min_freq=min_freq,
            specials=['<unk>', '<pad>']
        )
        self.vocab.set_default_index(self.vocab['<unk>'])

    def prepare_sequence_data(
        self,
        texts: List[str],
        labels: List[str],
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Dict:
        """
        Prepare data for sequence models (LSTM/BiLSTM).

        Args:
            texts: Input texts
            labels: Input labels
            test_size: Proportion of test set
            val_size: Proportion of validation set

        Returns:
            Dictionary containing train, validation, and test splits
        """
        # Create label mappings if not exists
        if self.label2id is None:
            self.create_label_mappings(labels)

        # Convert labels to IDs
        numerical_labels = [self.label2id[label] for label in labels]

        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            texts, numerical_labels,
            test_size=(test_size + val_size),
            random_state=DATA_PARAMS['random_state']
        )

        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=test_size/(test_size + val_size),
            random_state=DATA_PARAMS['random_state']
        )

        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'label2id': self.label2id,
            'id2label': self.id2label
        }

    def create_sequence_dataloaders(
        self,
        split_data: Dict,
        batch_size: int = 16
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create DataLoaders for sequence models.

        Args:
            split_data: Dictionary containing data splits
            batch_size: Batch size for DataLoader

        Returns:
            Train, validation, and test DataLoaders
        """
        # Build vocabulary if not exists
        if self.vocab is None:
            self.build_vocab(split_data['X_train'])

        def process_split(texts, labels):
            # Tokenize and numericalize
            sequences = [
                [self.vocab[token.lower()] for token in self.tokenizer(text)]
                for text in texts
            ]

            # Get lengths and pad sequences
            lengths = torch.tensor([len(seq) for seq in sequences])
            max_len = max(lengths)

            padded = torch.full(
                (len(sequences), max_len),
                self.vocab['<pad>'],
                dtype=torch.long
            )

            for idx, seq in enumerate(sequences):
                padded[idx, :len(seq)] = torch.tensor(seq)

            return TensorDataset(
                padded,
                lengths,
                torch.tensor(labels)
            )

        # Process each split
        train_data = process_split(split_data['X_train'], split_data['y_train'])
        val_data = process_split(split_data['X_val'], split_data['y_val'])
        test_data = process_split(split_data['X_test'], split_data['y_test'])

        # Create DataLoaders
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size)
        test_loader = DataLoader(test_data, batch_size=batch_size)

        return train_loader, val_loader, test_loader

    def create_transformer_dataloaders(
        self,
        split_data: Dict,
        model_name: str,
        batch_size: int = 16
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create DataLoaders for transformer models.

        Args:
            split_data: Dictionary containing data splits
            model_name: Name of the transformer model
            batch_size: Batch size for DataLoader

        Returns:
            Train, validation, and test DataLoaders
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def process_split(texts, labels):
            encodings = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=DATA_PARAMS['max_length'],
                return_tensors='pt'
            )

            return TensorDataset(
                encodings['input_ids'],
                encodings['attention_mask'],
                torch.tensor(labels)
            )

        # Process each split
        train_data = process_split(split_data['X_train'], split_data['y_train'])
        val_data = process_split(split_data['X_val'], split_data['y_val'])
        test_data = process_split(split_data['X_test'], split_data['y_test'])

        # Create DataLoaders
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size)
        test_loader = DataLoader(test_data, batch_size=batch_size)

        return train_loader, val_loader, test_loader
