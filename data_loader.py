import json
from typing import List, Dict
import pandas as pd
from collections import Counter
from sklearn.utils import shuffle

def load_dataset(file_path: str) -> List[Dict]:
    """
    Load a JSON dataset file.

    Args:
        file_path: Path to the JSON file

    Returns:
        List of dictionaries containing the dataset
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def combine_datasets(
    date_of_birth_data: List[Dict],
    education_data: List[Dict],
    place_of_death_data: List[Dict],
    place_of_birth_data: List[Dict],
    institution_data: List[Dict]
) -> List[Dict]:
    """
    Combine multiple datasets into one.

    Args:
        Various dataset lists

    Returns:
        Combined dataset
    """
    return (date_of_birth_data + education_data + place_of_death_data +
            place_of_birth_data + institution_data)

def process_relations(data: List[Dict], pred_mapping: Dict[str, str]) -> List[Dict]:
    """
    Process and map relations to their simplified forms.

    Args:
        data: Raw dataset
        pred_mapping: Dictionary mapping full relation names to simplified ones

    Returns:
        Processed dataset with simplified relations
    """
    processed_data = []
    for entry in data:
        if entry['pred'] in pred_mapping:
            processed_data.append({
                'relation': pred_mapping[entry['pred']],
                'subject': entry['sub'],
                'object': entry['obj'],
                'evidences': entry['evidences']
            })
    return processed_data

def balance_dataset(data: List[Dict], shuffle_data: bool = True) -> List[Dict]:
    """
    Balance the dataset by undersampling to match the smallest class.

    Args:
        data: Input dataset
        shuffle_data: Whether to shuffle the final dataset

    Returns:
        Balanced dataset
    """
    # Count relations
    relations = [entry['relation'] for entry in data]
    relation_counts = Counter(relations)

    # Get target size (size of smallest class)
    target_size = min(relation_counts.values())

    # Balance dataset
    balanced_data = []
    for relation in relation_counts:
        class_data = [entry for entry in data if entry['relation'] == relation]
        if len(class_data) > target_size:
            class_data = shuffle(class_data)[:target_size]
        balanced_data.extend(class_data)

    # Shuffle if requested
    if shuffle_data:
        balanced_data = shuffle(balanced_data)

    return balanced_data

def extract_features_labels(data: List[Dict]) -> tuple:
    """
    Extract sentences and labels from the dataset.

    Args:
        data: Processed dataset

    Returns:
        Tuple of (sentences, labels)
    """
    sentences = [entry['evidences'][0]['snippet'] for entry in data]
    labels = [entry['relation'] for entry in data]
    return sentences, labels

class DataLoader:
    """Main data loading class."""

    def __init__(self, file_paths: Dict[str, str], pred_mapping: Dict[str, str]):
        """
        Initialize DataLoader.

        Args:
            file_paths: Dictionary of dataset paths
            pred_mapping: Dictionary mapping relations
        """
        self.file_paths = file_paths
        self.pred_mapping = pred_mapping
        self.raw_data = None
        self.processed_data = None
        self.balanced_data = None

    def load_all_data(self) -> None:
        """Load all datasets and combine them."""
        datasets = {
            name: load_dataset(path)
            for name, path in self.file_paths.items()
        }
        self.raw_data = combine_datasets(*datasets.values())

    def process_data(self) -> None:
        """Process the loaded data."""
        if self.raw_data is None:
            raise ValueError("Data not loaded. Call load_all_data() first.")
        self.processed_data = process_relations(self.raw_data, self.pred_mapping)

    def balance_data(self) -> None:
        """Balance the processed data."""
        if self.processed_data is None:
            raise ValueError("Data not processed. Call process_data() first.")
        self.balanced_data = balance_dataset(self.processed_data)

    def get_features_labels(self) -> tuple:
        """Get features and labels from balanced data."""
        if self.balanced_data is None:
            raise ValueError("Data not balanced. Call balance_data() first.")
        return extract_features_labels(self.balanced_data)

    def get_data_stats(self) -> Dict:
        """Get statistics about the dataset."""
        if self.balanced_data is None:
            raise ValueError("Data not balanced. Call balance_data() first.")

        relations = [entry['relation'] for entry in self.balanced_data]
        relation_counts = Counter(relations)

        return {
            'total_samples': len(self.balanced_data),
            'relation_distribution': dict(relation_counts),
            'unique_relations': len(relation_counts)
        }
