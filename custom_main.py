import tensorflow as tf
from utils_dir.load_dataset_util import load_dataset
from utils_dir.preprocess_dataset_util import preprocess_data

def main(conf_load_dict: dict, conf_preprocess_dict: dict):

    BATCH_SIZE = 64
    BUFFER_SIZE = 100
    embedding_dim = 256

    print(f"----> Dataset Load.")
    data = load_dataset(conf_load_dict)

    print(f"----> Preprocess Data.")
    x_train, y_train, x_val, y_val, x_test, y_test = \
        preprocess_data(data, conf_preprocess_dict)

    dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

    # Shuffle and batch
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    pass

if __name__ == "__main__":

    conf_load_dict: dict = {
        'sequence_type': 'dna',
        'path': './bins_translated',
        'columns_names': [
            'Sequences','Count','Unnamed: 0','Label','Translated_sequences','Protein_length'
        ],
        'train_bins': [1,2,3],
        'val_bins': [4],
        'test_bins': [5],
    }

    conf_preprocess_dict: dict = {
        'padding': 'post',
        'maxlen': 14000,
        'onehot_flag': False,
    }

    dict_images: dict = {
        'loss': {
            'title': 'Training With Validation Loss',
            'fig_name': 'train_val_loss',
            'fig_format': 'png',
            'savefig_flag': True
        },
        'acc': {
            'title': 'Training With Validation Accuracy',
            'fig_name': 'train_val_acc',
            'fig_format': 'png',
            'savefig_flag': True
        },
        'roc_curve': {
            'title': 'Roc Curve',
            'fig_name': 'roc_curve',
            'fig_format': 'png',
            'savefig_flag': True
        },
        'confusion_matrix': {
            'title': 'Confusion Matrix',
            'fig_name': 'confusion_matrix',
            'fig_format': 'png',
            'savefig_flag': True
        }
    }
    main(conf_load_dict, conf_preprocess_dict)
    pass
