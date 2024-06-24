import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Lambda, LayerNormalization, Concatenate, BatchNormalization
from tensorflow.keras.models import Model


def multiply(args):
    return tf.multiply(args[0], args[1])


def reduce_sum(x):
    return keras.backend.sum(x, axis=-1, keepdims=True)


def multi_head_attention_block(input_data, n_heads):
    scores = []
    for i in range(n_heads):
        attn = Dense(input_data.shape[1], activation=tf.nn.softmax)(input_data)
        output = Lambda(multiply)([input_data, attn])
        scores.append(output)
    x = Concatenate(axis=1)(scores)
    x = Dense(x.shape[1] // n_heads, activation=tf.nn.gelu)(x)
    x = Dropout(0.5)(x)
    return LayerNormalization()(keras.layers.add([input_data, x]))


def features_pyramid_block(input_data):
    n_layer_o = input_data.shape[1]
    n_layer_1 = n_layer_o // 2
    n_layer_2 = n_layer_1 // 2
    output0 = Dense(n_layer_o, activation=tf.nn.gelu)(input_data)
    output0 = LayerNormalization()(output0)
    output0 = Dropout(0.5)(output0)
    output1 = Dense(n_layer_1, activation=tf.nn.gelu)(output0)
    output1 = LayerNormalization()(output1)
    output2 = Dense(n_layer_2, activation=tf.nn.gelu)(output1)
    output2 = LayerNormalization()(output2)
    output2 = Dropout(0.5)(output2)

    x_feature_pyramid_1 = Dense(n_layer_1, activation=tf.nn.gelu)(output2)
    x_feature_pyramid_1 = LayerNormalization()(x_feature_pyramid_1)
    feature_map_1 = keras.layers.add([x_feature_pyramid_1, output1])
    x_feature_pyramid_0 = Dense(n_layer_o, activation=tf.nn.gelu)(feature_map_1)
    x_feature_pyramid_0 = LayerNormalization()(x_feature_pyramid_0)
    x_feature_pyramid_0 = Dropout(0.5)(x_feature_pyramid_0)
    feature_map_0 = keras.layers.add([x_feature_pyramid_0, output0])
    return feature_map_1, feature_map_0


def attention_block(input_1, input_2):
    final_output1 = Dense(1, activation=tf.nn.sigmoid, name="final_output1")(input_1)
    final_output2 = Dense(1, activation=tf.nn.sigmoid, name="final_output2")(input_2)
    final_p = Concatenate(axis=1)([final_output1, final_output2])
    final_p_attn = Dense(2, activation=tf.nn.softmax)(final_p)
    final_output3 = Lambda(multiply)([final_p_attn, final_p])
    final_output3 = Lambda(reduce_sum, name='final_output3')(final_output3)
    return final_output1, final_output2, final_output3


def create_model(shape):
    input_data = keras.layers.Input(shape=(shape[1],))
    x = input_data
    intput_1, input_2 = features_pyramid_block(x)
    final_output1, final_output2, final_output3 = attention_block(intput_1, input_2)
    model = Model(inputs=[input_data], outputs=[final_output1, final_output2, final_output3])
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss={
                      'final_output1': 'binary_crossentropy',
                      'final_output2': 'binary_crossentropy',
                      'final_output3': 'binary_crossentropy'
                  },
                  metrics={
                      'final_output1': 'acc',
                      'final_output2': 'acc',
                      'final_output3': 'acc'
                  },
                  experimental_run_tf_function=False)
    return model
