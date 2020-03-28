import tensorflow.keras as keras
import tensorflow as tf

def threelayers(input_shape):

    model = keras.Sequential()

    # Define first fully connected layer
    model.add(keras.layers.Dense(400,
                                 input_shape=(120,),
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

    # Add output layer
    model.add(keras.layers.Dense(5, activation=None))

    # Define optimizer
    optimizer = tf.keras.optimizers.SGD(momentum=0.7, nesterov=True)
    return model

def simple_model(input_shape, loss):
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, input_shape = input_shape, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    # tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='sigmoid')
  ])

  model.compile(optimizer='adam',
                loss=loss,
                metrics=['accuracy'])
  return model

if __name__ == '__main__':      #not working for some reason: pydotplus.graphviz.InvocationException: GraphViz's executables not found

    model = simple_model()

    from tensorflow.keras.utils import plot_model

    plot_model(model, to_file='model.png')