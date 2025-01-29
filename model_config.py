LSTM_CONFIG = {
  'embedding_dim': [256],
  'hidden_dim': [256], 
  'n_layers': [2],
  'dropout': [0.5],
  'learning_rates': [1e-4],
  'batch_size': [64],
  'epochs': 20
}

BILSTM_CONFIG = {
  'embedding_dim': [256],
  'hidden_dim': [512],
  'n_layers': [2],
  'dropout': [0.5], 
  'learning_rates': [5e-5],
  'batch_size': [64],
  'epochs': 15
}

BERT_CONFIG = {
  'dropout': [0.4],
  'learning_rates': [1e-5], 
  'batch_size': [32],
  'epochs': 5,
  'warmup_steps': 200,
  'gradient_accum_steps': 2
}

ROBERTA_CONFIG = {
  'dropout': [0.5],
  'learning_rates': [3e-6],
  'batch_size': [16],
  'epochs': 5, 
  'warmup_steps': 400,
  'gradient_accum_steps': 4
}

TRAIN_PARAMS = {
  'early_stopping_patience': 3,
  'gradient_clip_val': 0.1,
  'weight_decay': 1e-3,
  'monitor': 'val_loss',
  'class_weights': True
}

DATA_PARAMS = {
  'max_length': 128,
  'test_size': 0.2,
  'validation_size': 0.1,
  'random_state': 42
}
