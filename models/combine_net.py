import keras
from keras import layers
from keras import models
import os, yaml
from rmldataset2016 import load_data_snrs

class combine_net():
    def __init__(self, class_names, weights = None, input_shape=None, classes=11):
        self.class_names = class_names
        self.net(weights, input_shape, classes)

    def net(self, weights=None, input_shape=None, classes=11):
        img_input = layers.Input(shape=input_shape, name='Input')
        x = img_input

        L = 6
        img_input = layers.Input(shape=input_shape, name='Input')
        x = img_input
        for i in range(L):
            x = self.residual_stack(input_tensor=x, filters=32, stage=i + 1, strides=2)

        x = layers.CuDNNLSTM(50)(x)
        dr = 0.5

        x = layers.Dense(128, activation='selu', kernel_initializer='he_normal', name="dense1")(x)
        x = layers.Dropout(dr)(x)

        x = layers.Dense(128, activation='selu', kernel_initializer='he_normal', name="dense2")(x)
        x = layers.Dropout(dr)(x)

        x = layers.Dense(classes, kernel_initializer='he_normal', name="dense3")(x)
        x = layers.Activation('softmax')(x)

        self.model = models.Model(img_input, x, name='resnetLSTM-like')

        # Load weights.
        if weights is not None:
            self.model.load_weights(weights)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()

    def residual_unit(self, input_tensor, kernel_size, filters, stage, block):
        """The identity block is the block that has no conv layer at shortcut.

        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of
                middle conv layer at main path
            filters: list of integers, the filters of 2 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names

        # Returns
            Output tensor for the block.
        """
        filters1, filters2 = filters
        bn_axis = 2

        conv_name_base = 'res_stack' + str(stage) + '_block' + block
        bn_name_base = 'bn_stack' + str(stage) + '_block' + block

        x = layers.Conv1D(filters=filters1, kernel_size=kernel_size,
                          padding='same',
                          kernel_initializer='he_normal',
                          name=conv_name_base + '_u1')(input_tensor)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '_u1')(x)
        x = layers.Activation('relu')(x)

        x = layers.Conv1D(filters=filters2, kernel_size=1,
                          padding='same',
                          kernel_initializer='he_normal',
                          name=conv_name_base + '_u2')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '_u2')(x)

        x = layers.add([x, input_tensor])
        x = layers.Activation('relu')(x)
        return x

    def residual_stack(self, input_tensor,
                       filters,
                       stage,
                       strides=2):
        """A block that has a conv layer at shortcut.

        # Arguments
            input_tensor: input tensor
            kernel_size: default 3, the kernel size of
                middle conv layer at main path
            filters: list of integers, the filters of 3 conv layer at main path
            stage: integer, current stage label, used for generating layer names
            block: 'a','b'..., current block label, used for generating layer names
            strides: Strides for the first conv layer in the block.

        # Returns
            Output tensor for the block.

        Note that from stage 3,
        the first conv layer at main path is with strides=(2, 2)
        And the shortcut should have strides=(2, 2) as well
        """
        filters1 = filters
        bn_axis = 2

        conv_name_base = 'res_stack' + str(stage) + '_'
        bn_name_base = 'bn_stack' + str(stage) + '_'
        mp_name_base = 'mp_stack' + str(stage)

        x = layers.Conv1D(filters1, 1, strides=strides,
                          kernel_initializer='he_normal',
                          name=conv_name_base + 'a')(input_tensor)
        x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + 'a')(x)

        taps = 8
        x = self.residual_unit(input_tensor=x, kernel_size=taps, filters=[32, 32], stage=stage, block='b')
        x = self.residual_unit(input_tensor=x, kernel_size=taps, filters=[32, 32], stage=stage, block='c')

        x = layers.MaxPooling1D(pool_size=3, strides=1, padding='same', name=mp_name_base)(x)

        return x
    def train(self, X_train, Y_train, X_test, Y_test, filepath, batch_size = 256, epochs = 300):
        self.history = self.model.fit(X_train,
                            Y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            verbose=2,
                            validation_data=(X_test, Y_test),
                            callbacks=[
                                keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=0,
                                                                save_best_only=True, mode='auto')
                            ])
    def predict(self, signals, filepath = None, batch_size = 256):
        if filepath is not None:
            self.model.load_weights(filepath)
        probabilities = self.model.predict(signals, batch_size=batch_size)
        decoded_predictions = list()
        for im in probabilities:
            decoded_prediction = list()
            for i in range(im.shape[0]):
                decoded_prediction.append(('probability', self.class_names[i], im[i]))
            decoded_prediction.sort(key=lambda x: x[2], reverse=True)
            decoded_predictions.append(decoded_prediction)
        return decoded_predictions


if __name__ == "__main__":
    config_file = '/media/lmk/0101d15f-da66-48f5-a1f6-85b33d22f422/mhb/one-pixel-attack-signal-keras/config.yaml'
    CONFIG = yaml.safe_load(open(config_file))
    os.environ["CUDA_VISIBLE_DEVICES"] = CONFIG["gpu_id"]
    snr_define = 0
    filepath = "/media/lmk/0101d15f-da66-48f5-a1f6-85b33d22f422/mhb/one-pixel-attack-signal-keras/models/combine_net.whs.h5"
    (mods, snrs, lbl), (X_train, Y_train), (X_test, Y_test), (train_idx, test_idx) = load_data_snrs(
        '/media/lmk/0101d15f-da66-48f5-a1f6-85b33d22f422/mhb/one-pixel-attack-signal-keras/dataset/RML2016.10a_dict.pkl', [i for i in range(snr_define, 20, 2)], train_rate=0.9)
    CN = combine_net(mods, weights=None, input_shape = (128, 2), classes=11)
    CN.train(X_train, Y_train, X_test, Y_test, filepath, epochs=300)