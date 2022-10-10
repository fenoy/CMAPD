params = {
    'setup': {
        'seed': 10
    },
    'environment': {
        'map': './env/grid.map',
        'min_collective_size': 1,
        'max_collective_size': 5,
        'capacity': 3,
        'penalty': 1000
    },
    'model': {
        'input_size': [21, 35],
        'd_model': 128,
        'nhead': 8,
        'dim_feedforward': 512,
        'num_layers': 3
    },
    'training': {
        'n_agents': 1,
        'n_tasks': [10, 20],
        'batch_size': 256,
        'train_size': 40960,#204800,
        'eval_size': 5120,#25600,
        'learning_rate': 0.0001,
        'n_epochs': 100
    }
}
