import configparser as config

c = config.ConfigParser()

c['DEFAULT'] = {
    'annotation_file': '',
    'imgs_folder': '',
    'path_to_save': ''
}
c['HYPERS'] = {
    'embed_dim': 512,
    'ffd': 512,
    'num_heads': 2,
    'ratio': 0.3
}
c['DATASET_PARAMS'] = {
    'training_batch_size': 64,
    'validation_batch_size': 5,
    'num_samples':  6000,
    'initial_learning_rate': 1e-3,
    'training_length': 0.95,
    'epochs': 10
}

with open('config.ini', 'w') as f:
    c.write(f)
