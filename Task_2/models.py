import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import backend as K

def threelayers(input_shape, loss, output_layer):
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
    model.add(keras.layers.Dense(10, activation=output_layer))
    # Define optimizer
    model.compile(optimizer='adagrad',
                  loss=loss,
                  metrics=[dice_coef, 'mse', keras.metrics.AUC()])
    return model

def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.BatchNormalization(axis=-1, scale= hp.Choice('bn_scale_' + str(0), ['True', 'False'])))
    model.add(keras.layers.Dense(hp.Int('units_' + str(0),min_value=32,
                                            max_value=512,
                                            step=32),
                                 input_shape=(12, 36),
                                 activation=tf.nn.relu,
                                 kernel_initializer='he_normal',
                                 kernel_regularizer=keras.regularizers.l2(l=1e-3)))
    for i in range(hp.Int('num_layers', 2, 20)):
        model.add(keras.layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value=32,
                                            max_value=512,
                                            step=32),
                               activation='relu'))
        model.add(keras.layers.Dropout(rate=hp.Choice('dropout_rate_' + str(i), [0.0, 0.3, 0.5])))
    model.add(keras.layers.BatchNormalization(axis=-1, scale= hp.Choice('bn_scale_' + str(1), ['True', 'False'])))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(10, activation='sigmoid'))
    model.compile(
        optimizer=keras.optimizers.Adadelta(
            hp.Choice('learning_rate', [1.0])),
        loss='binary_crossentropy',
        metrics=[dice_coef, 'mse', keras.metrics.AUC()])
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
                  metrics=[dice_coef, 'mse'])
    return model


def simple_model(input_shape, loss):
  model = tf.keras.Sequential([
    keras.layers.BatchNormalization(axis=-1),
    tf.keras.layers.Dense(32, input_shape = input_shape, activation='relu'),
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
    return 1-dice_coef(y_true, y_pred)




if __name__ == '__main__':      #not working for some reason: pydotplus.graphviz.InvocationException: GraphViz's executables not found

    model = simple_model()

    from tensorflow.keras.utils import plot_model

    plot_model(model, to_file='model.png')
