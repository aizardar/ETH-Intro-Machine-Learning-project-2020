import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import backend as K
import tensorflow.keras.layers as layers
import kerastuner


def threelayers_task1(input_shape, loss, output_layer, task):
    model = keras.Sequential()

    model.add(keras.layers.BatchNormalization(axis=-1))
    # Define first fully connected layer
    model.add(keras.layers.Dense(400,
                                 input_shape=input_shape,
                                 activation=tf.nn.relu,
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=keras.regularizers.l2(l=1e-3)))

    # Add dropout for overfitting avoidance and batch normalization layer
    model.add(keras.layers.Dropout(rate=0.3))
    model.add(keras.layers.BatchNormalization(scale=False))

    # Add second fully connected layer
    model.add(keras.layers.Dense(250,
                                 activation=tf.nn.relu,
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=keras.regularizers.l2(l=1e-3)))

    # Add dropout for overfitting avoidance and batch normalization layer
    model.add(keras.layers.Dropout(rate=0.3))
    model.add(keras.layers.BatchNormalization(scale=False))
    model.add(keras.layers.Flatten())
    # Add output layer
    model.add(keras.layers.Dense(11, activation=output_layer))
    model.compile(optimizer='adagrad',
                  loss=loss,
                  metrics=[dice_coef, 'mse', keras.metrics.AUC()])

    return model


def toy_ResNet(input_shape, loss):
    inputs = keras.Input(shape=input_shape, name="img")
    x = layers.Conv2D(32, 3, activation="relu")(inputs)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    block_1_output = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_1_output)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    block_2_output = layers.add([x, block_1_output])

    # x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_2_output)
    # x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    # block_3_output = layers.add([x, block_2_output])

    x = layers.Conv2D(64, 3, activation="relu")(block_2_output)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(11)(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adagrad',
                  loss=loss,
                  metrics=[dice_coef, 'mse', keras.metrics.AUC()])
    return model


def threelayers(input_shape, loss, output_layer, task):
    model = keras.Sequential()

    model.add(keras.layers.BatchNormalization(axis=-1))
    # Define first fully connected layer
    model.add(keras.layers.Dense(400,
                                 input_shape=input_shape,
                                 activation=tf.nn.relu,
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=keras.regularizers.l2(l=1e-3)))

    # Add dropout for overfitting avoidance and batch normalization layer
    model.add(keras.layers.Dropout(rate=0.3))
    model.add(keras.layers.BatchNormalization(scale=False))

    # Add second fully connected layer
    model.add(keras.layers.Dense(250,
                                 activation=tf.nn.relu,
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=keras.regularizers.l2(l=1e-3)))

    # Add dropout for overfitting avoidance and batch normalization layer
    model.add(keras.layers.Dropout(rate=0.3))
    model.add(keras.layers.BatchNormalization(scale=False))
    model.add(keras.layers.Flatten())
    if task == 1 or task == 12:
        # Add output layer
        if task == 12:
            model.add(keras.layers.Dense(11, activation=output_layer))
        else:
            model.add(keras.layers.Dense(10, activation=output_layer))
        model.compile(optimizer='adagrad',
                      loss=loss,
                      metrics=[dice_coef, 'mse', keras.metrics.AUC()])
    if task == 2:
        model.add(keras.layers.Dense(1, activation=output_layer))
        model.compile(optimizer='adagrad', loss=loss,
                      metrics=['sparse_categorical_accuracy', dice_coef, tf.keras.metrics.AUC()])
    if task == 3:
        model.add(keras.layers.Dense(4, activation='linear'))
        model.compile(optimizer='adagrad',
                      loss=loss,
                      metrics=['mse'])

    return model


def lstm(input_shape, loss, params, task):
    lstm_layer = tf.keras.layers.RNN(
        tf.keras.layers.LSTMCell(64),
        input_shape=(None, input_shape[0]))
    model = tf.keras.models.Sequential([
        lstm_layer,
        tf.keras.layers.BatchNormalization()]
    )
    if task == 1 or task == 12:
        if task == 12:
            model.add(Dense(11, activation=params['output_layer']))
        else:
            model.add(Dense(10, activation=params['output_layer']))
        model.compile(optimizer='adagrad',
                      loss=loss,
                      metrics=[dice_coef, 'mse', keras.metrics.AUC()])
    if task == 2:
        model.add(keras.layers.Dense(1, activation=params['output_layer']))
        model.compile(optimizer='adagrad', loss=loss,
                      metrics=['sparse_categorical_accuracy', dice_coef, tf.keras.metrics.AUC()])
    if task == 3:
        model.add(keras.layers.Dense(4))
        model.compile(optimizer='adagrad',
                      loss=loss,
                      metrics=['mse'])
    return model


class recurrent_net(kerastuner.HyperModel):
    def __init__(self, input_shape, loss, output_layer, task):
        super(recurrent_net, self).__init__()
        self.input_shape = input_shape
        self.loss = loss
        self.output_layer = output_layer
        self.task = task

    def build(self, hp):
        input_shape, loss, output_layer, task = [self.input_shape, self.loss, self.output_layer, self.task]
        inputs = tf.keras.Input(shape=input_shape)
        x = inputs
        for i in range(hp.Int('conv_blocks', 1, 5, default=1)):
            filters = hp.Int('filters_' + str(i), 32, 256, step=32)
            for _ in range(2):
                x = tf.keras.layers.Convolution2D(
                    filters, kernel_size=(3, 3), padding='same')(x)
                x = tf.keras.layers.BatchNormalization()(x)
                x = tf.keras.layers.ReLU()(x)
            if hp.Choice('pooling_' + str(i), ['avg', 'max']) == 'max':
                x = tf.keras.layers.MaxPool2D()(x)
            else:
                x = tf.keras.layers.AvgPool2D()(x)
        x = tf.keras.layers.GlobalAvgPool2D()(x)
        x = tf.keras.layers.Dense(
            hp.Int('hidden_size', 30, 100, step=10, default=50),
            activation='relu')(x)
        x = tf.keras.layers.Dropout(
            hp.Float('dropout', 0, 0.5, step=0.1, default=0.5))(x)

        if task == 1 or task == 12:
            if task == 12:
                outputs = keras.layers.Dense(11, activation=output_layer)(x)
            else:
                outputs = keras.layers.Dense(10, activation=output_layer)(x)
            model = tf.keras.Model(inputs, outputs)
            model.compile(optimizer='adagrad',
                          loss=loss,
                          metrics=[dice_coef, 'mse', keras.metrics.AUC()])
        if task == 2:
            outputs = keras.layers.Dense(1, activation=output_layer)(x)
            model = tf.keras.Model(inputs, outputs)
            model.compile(optimizer='adagrad', loss=loss,
                          metrics=['sparse_categorical_accuracy', dice_coef, tf.keras.metrics.AUC()])
        if task == 3:
            outputs = keras.layers.Dense(4, activation='linear')(x)
            model = tf.keras.Model(inputs, outputs)
            model.compile(optimizer='adagrad',
                          loss=loss,
                          metrics=['mse'])

        return model


class dense_model(kerastuner.HyperModel):
    def __init__(self, input_shape, loss, output_layer, task):
        super(dense_model, self).__init__()
        self.input_shape = input_shape
        self.loss = loss
        self.output_layer = output_layer
        self.task = task

    def build(self, hp):
        input_shape, loss, output_layer, task = [self.input_shape, self.loss, self.output_layer, self.task]
        model = Sequential()
        for i in range(hp.Int('Batchnorm', 0, 1, default=1)):
            model.add(
                keras.layers.BatchNormalization(axis=-1, scale=hp.Choice('bn_scale_' + str(0), ['True', 'False'])))
        model.add(keras.layers.Dense(hp.Int('units_' + str(0), min_value=32,
                                            max_value=512,
                                            step=32, default=400),
                                     activation=tf.nn.relu,
                                     kernel_initializer='he_normal',
                                     kernel_regularizer=keras.regularizers.l2(l=1e-3)))
        for i in range(hp.Int('num_layers', 1, 4, default=1)):
            model.add(keras.layers.Dense(units=hp.Int('units_' + str(i),
                                                      min_value=32,
                                                      max_value=512,
                                                      step=32, default=250),
                                         activation='relu'))
            model.add(keras.layers.Dropout(rate=hp.Choice('dropout_rate_' + str(i), [0.0, 0.3, 0.5])))
        model.add(keras.layers.BatchNormalization(axis=-1, scale=hp.Choice('bn_scale_' + str(1), ['True', 'False'])))
        model.add(keras.layers.Flatten())
        if task == 1 or task == 12:
            if task == 12:
                model.add(Dense(11, activation=output_layer))
            else:
                model.add(Dense(10, activation=output_layer))
            model.compile(optimizer='adagrad',
                          loss=loss,
                          metrics=[dice_coef, 'mse', keras.metrics.AUC()])
        if task == 2:
            model.add(keras.layers.Dense(1, activation=output_layer))
            model.compile(optimizer='adagrad', loss=loss,
                          metrics=['sparse_categorical_accuracy', dice_coef, tf.keras.metrics.AUC()])
        if task == 3:
            model.add(keras.layers.Dense(4, activation='linear'))
            model.compile(optimizer='adagrad',
                          loss=loss,
                          metrics=['mse'])
        return model


def svm(input_shape, loss, output_layer):
    model = Sequential()
    model.add(keras.layers.BatchNormalization(axis=-1))
    model.add(Dense(64, activation='relu', input_shape=input_shape))
    model.add(keras.layers.Flatten())
    model.add(Dense(10, kernel_regularizer=l2(0.01)))
    model.add(Activation(output_layer))
    model.compile(loss=loss,
                  optimizer='adadelta',
                  metrics=['mse'])
    return model


class temp_model(kerastuner.HyperModel):
    def __init__(self, input_shape, loss, output_layer, task):
        super(temp_model, self).__init__()
        self.input_shape = input_shape
        self.loss = loss
        self.output_layer = output_layer
        self.task = task

    def build(self, hp):
        input_shape, loss, output_layer, task = [self.input_shape, self.loss, self.output_layer, self.task]
        model = Sequential()
        for i in range(hp.Int('num_dense_layers', 2, 4)):
            model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                                min_value=32,
                                                max_value=512,
                                                step=32),
                                   activation='relu'))
        model.add(keras.layers.Flatten())
        model.add(Dense(11, activation='softmax'))
        model.compile(optimizer='adagrad',
                      loss=loss,
                      metrics=[dice_coef, 'mse', keras.metrics.AUC()])

        return model


class simple_conv_model(kerastuner.HyperModel):

    def __init__(self, input_shape, loss, output_layer, task):
        super(simple_conv_model, self).__init__()
        self.input_shape = input_shape
        self.loss = loss
        self.output_layer = output_layer
        self.task = task

    def build(self, hp):
        input_shape, loss, output_layer, task = [self.input_shape, self.loss, self.output_layer, self.task]
        model = Sequential()
        for i in range(hp.Int('num_conv_layers', 1, 4)):
            filters = hp.Int('filters_' + str(i), 32, 256, step=32)
            model.add(keras.layers.Conv2D(filters, kernel_size=(
                hp.Int('x_kernel_size_{}'.format(i), 1, 3), hp.Int('y_kernel_size_{}'.format(i), 1, 3)),
                                          activation='relu'))

        model.add(keras.layers.Flatten())
        for i in range(hp.Int('num_dense_layers', 1, 4)):
            model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                                min_value=32,
                                                max_value=512,
                                                step=32),
                                   activation='relu'))
        if task == 1 or task == 12:
            if task == 12:
                model.add(Dense(11, activation=output_layer))
            else:
                model.add(Dense(10, activation=output_layer))
            model.compile(optimizer='adagrad',
                          loss=loss,
                          metrics=[dice_coef, 'mse', keras.metrics.AUC()])
        if task == 2:
            model.add(keras.layers.Dense(1, activation=output_layer))
            model.compile(optimizer='adagrad', loss=loss,
                          metrics=['sparse_categorical_accuracy', dice_coef, tf.keras.metrics.AUC()])
        if task == 3:
            model.add(keras.layers.Dense(4, activation='linear'))
            model.compile(optimizer='adagrad',
                          loss=loss,
                          metrics=['mse'])
        return model


def simple_model(input_shape, loss):
    model = tf.keras.Sequential([
        keras.layers.BatchNormalization(axis=-1),
        tf.keras.layers.Dense(32, input_shape=input_shape, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        # tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss=loss,
                  metrics=[dice_coef, 'mse'])
    return model


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


if __name__ == '__main__':  # not working for some reason: pydotplus.graphviz.InvocationException: GraphViz's executables not found

    model = simple_model()

    from tensorflow.keras.utils import plot_model

    plot_model(model, to_file='model.png')
