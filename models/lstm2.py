import keras
from keras import layers
from keras import models


def LSTM_2(mods, weights=None, input_shape=None, classes=11):
    img_input = layers.Input(shape=input_shape, name='Input')
    x = img_input
    x = layers.LSTM(128, name='lstm_1', return_sequences=True)(x)
    x = layers.LSTM(128, name='lstm_2', return_sequences=False)(x)
    x = layers.Dense(classes, kernel_initializer='he_normal', name="dense1")(x)
    x = layers.Activation('softmax', name='fc_soft')(x)
    model = models.Model(img_input, x, name='LSTM-2')

    # Load weights.
    if weights is not None:
        model.load_weights(weights)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model
