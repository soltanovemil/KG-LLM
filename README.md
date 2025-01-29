# KG-LLM: A Happy Marriage

This project explores the integration of Large Language Models (LLMs) with Knowledge Graph (KG) construction and reasoning tasks. By leveraging models such as LSTM, BiLSTM, BERT, and RoBERTa, we aim to streamline the KG creation process and advance knowledge representation.

## Project Overview

The project analyzes performance metrics, training history, and visualizations to highlight the capabilities and limitations of each approach in tasks like relation classification and knowledge graph visualization.

### Models Implemented
- LSTM (74% accuracy)
- BiLSTM (81% accuracy)
- BERT (86% accuracy)
- RoBERTa (85% accuracy)

### Dataset
The dataset focuses on five types of relationships:
- date_of_birth
- place_of_birth
- place_of_death
- institution
- degree

Dataset source: [Relation Extraction Corpus](https://github.com/google-research-datasets/relation-extraction-corpus)

## Project Structure
```
KG-LLM/
├── config/
│   └── model_config.py
├── data/
│   ├── data_loader.py
│   └── data_processor.py
├── models/
│   ├── lstm.py
│   ├── bilstm.py
│   ├── bert.py
│   └── roberta.py
├── knowledge_graph/
│   ├── graph_builder.py
│   └── visualizer.py
├── training/
│   ├── train.py
│   └── evaluate.py
└── utils/
    └── helpers.py
```

## Requirements
- Python 3.8+
- PyTorch 2.2.0
- torchtext 0.17.0
- transformers 4.37.2
- networkx
- matplotlib
- seaborn
- pandas
- numpy

## Installation
```bash
pip install torch==2.2.0 torchtext==0.17.0 transformers==4.37.2
```

## Results
The project achieved the following results:
- Highest accuracy with BERT (86%)
- Strong performance from RoBERTa (85%)
- Balanced performance from BiLSTM (81%)
- Efficient baseline from LSTM (74%)

## Documentation
- Report: Detailed project report
- Presentation: Project presentation slides

## Author
Emil Soltanov

## Acknowledgments
- Hugging Face Transformers
- PyTorch
- Google Research Datasets
