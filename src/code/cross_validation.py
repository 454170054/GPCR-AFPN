import pandas as pd
from metric import Metrics, calculate_mean_metric, save_result
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras
import tensorflow as tf
import feature_extraction
import network
import glob
import os


def cross_validation_results(x, y, count, seed):
    valid_m = Metrics('valid_' + str(count))
    n = 5
    skf = StratifiedKFold(shuffle=True, n_splits=n, random_state=seed)
    i = 0
    for train_index, valid_index in skf.split(x, y):
        print("-" * 25 + f'{i}th flod' + "-" * 25)
        train_seq, y_train = x[train_index].values.flatten().tolist(), y[train_index]
        valid_seq, y_valid = x[valid_index].values.flatten().tolist(), y[valid_index]
        ## fast_text
        dir_name = "cross_validation_results"
        fastText_model = feature_extraction.train_model(train_seq, dir_name)
        x_train = feature_extraction.get_features_from_datasets(train_seq, 'train', fastText_model, dir_name)
        x_valid = feature_extraction.get_features_from_datasets(valid_seq, 'valid', fastText_model, dir_name)

        afpn = network.create_model(x_train.shape)
        early_stopping = keras.callbacks.EarlyStopping(monitor='val_final_output3_acc', patience=50)
        file_path = r'../../resources/cross_validation_results/best.h5'
        best_saving_1 = tf.keras.callbacks.ModelCheckpoint(file_path, monitor='val_final_output3_acc', mode='auto',
                                                           verbose=0, save_best_only=True, save_weights_only=True)
        afpn.fit(x_train, y_train, epochs=500, validation_data=(x_valid, y_valid),
                            callbacks=[early_stopping, best_saving_1], verbose=0, batch_size=128, shuffle=True)
        afpn.load_weights(file_path)
        valid_probs = afpn.predict(x_valid, verbose=0)[2]
        valid_m.add([y_valid.values.flatten().tolist(), valid_probs.flatten().tolist()])
        tf.keras.backend.clear_session()
        del afpn
        del x_train
        del x_valid
        for file in glob.glob('../../resources/cross_validation_results/*.model'):
            os.remove(file)
        for file in glob.glob('../../resources/cross_validation_results/*.txt'):
            os.remove(file)
        i += 1
    valid_metrix = valid_m.calculate_metrics()
    return valid_metrix


def load_data():
    '''
    load datasets
    :return: datasets
    '''
    train_set = pd.read_csv('../../resources/datasets/train_dataset.csv')
    test_set = pd.read_csv('../../resources/datasets/test_dataset.csv')
    return train_set['seq'], train_set['label'], test_set['seq'], test_set['label']


if __name__ == '__main__':
    train_seq, train_labels, test_seq, test_labels = load_data()
    valid_metrics = []
    seeds = [111, 222, 333, 444, 555]
    dir_name = "cross_validation_results"
    for i in range(5):
        print("*" * 30 + f'{i}th 5-fold cross validation' + "*" * 30)
        valid_metric = cross_validation_results(train_seq, train_labels, i, seeds[i])
        valid_metrics.append(valid_metric)
    calculate_mean_metric(valid_metrics, 'valid')
    save_result('valid', valid_metrics, dir_name)

