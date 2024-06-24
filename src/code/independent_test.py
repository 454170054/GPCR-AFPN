import pandas as pd
from metric import Metrics, save_result
from tensorflow import keras
import tensorflow as tf
import feature_extraction
import network


def independent_test(x, y, x_test, y_test):
    test_m = Metrics('test')
    dir_name = "independent_results"
    fastText_model = feature_extraction.train_model(x.values.flatten(), dir_name)
    x_train = feature_extraction.get_features_from_datasets(x, 'train', fastText_model, dir_name)
    x_test = feature_extraction.get_features_from_datasets(x_test, 'test', fastText_model, dir_name)
    model = network.create_model(x_train.shape)
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_final_output3_acc', patience=50)
    file_path = r'../../resources/independent_results/best.h5'
    best_saving_1 = tf.keras.callbacks.ModelCheckpoint(file_path, monitor='val_final_output3_acc',mode='auto',
                                                        verbose=0, save_best_only=True, save_weights_only=True)
    model.fit(x_train, y, epochs=500, callbacks=[early_stopping, best_saving_1], verbose=0, batch_size=128, shuffle=True, validation_split=0.05)
    model.load_weights(file_path)
    test_probs = model.predict(x_test)[2]
    test_m.add([y_test.values.flatten().tolist(), test_probs.flatten().tolist()])
    tf.keras.backend.clear_session()
    return test_m


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
    test_m = independent_test(train_seq, train_labels, test_seq, test_labels)
    metrics = test_m.calculate_metrics()
    a = []
    dir_name = "independent_results"
    a.append(metrics)
    save_result('test', a, dir_name)