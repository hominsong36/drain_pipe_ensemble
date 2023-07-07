import tensorflow.keras.backend as K
from tensorflow.keras.layers import *


class ResUnit(Layer):
    def __init__(self, filter_in, filter_out, kernel_size, dropout_rate=0.2):
        super().__init__()

        self.relu0 = Activation('relu')
        self.relu1 = Activation('relu')

        self.batchnorm0 = BatchNormalization()
        self.batchnorm1 = BatchNormalization()

        self.conv0 = Conv2D(filter_out, kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv1 = Conv2D(filter_out, kernel_size, activation='relu', padding='same', kernel_initializer='he_normal')

        self.dropout0 = Dropout(rate=dropout_rate)
        self.dropout1 = Dropout(rate=dropout_rate)
        self.add = Add()

        if filter_in == filter_out:
            self.mediator = Lambda(lambda x: x)
        else:
            self.mediator = Conv2D(filter_out, (1, 1), activation='relu', padding='same', use_bias=False)

    def __call__(self, unit_input, training=True):
        batchnorm0 = self.batchnorm0(unit_input, training=training)
        relu0 = self.relu0(batchnorm0)
        conv0 = self.conv0(relu0)
        dropout0 = self.dropout0(conv0)

        batchnorm1 = self.batchnorm1(dropout0, training=training)
        relu1 = self.relu1(batchnorm1)
        conv1 = self.conv1(relu1)
        dropout1 = self.dropout1(conv1)

        mediator = self.mediator(unit_input)

        adding = self.add([mediator, dropout1])

        return adding


class ResLayer(Layer):
    def __init__(self, filter_in, filter_out, filters, kernel_size, dropout_rate=0.2):
        super().__init__()

        self.units = []

        filters1 = [filter_out] * filters
        filters2 = [filter_out] * filters
        filters1[0] = filter_in

        for filter1, filter2 in zip(filters1, filters2):
            self.units.append(ResUnit(filter1, filter2, kernel_size, dropout_rate))

    def __call__(self, layer_input, training=True):
        x = layer_input
        for unit in self.units:
            x = unit(x, training=training)

        return x


class DenseUnit(Layer):
    def __init__(self, filter_out, kernel_size, dropout_rate=0.2):
        super().__init__()

        self.relu = Activation('relu')

        self.batchnorm = BatchNormalization()
        self.conv = Conv2D(filter_out, kernel_size, padding='same', activation='relu', kernel_initializer='he_normal')
        self.dropout = Dropout(rate=dropout_rate)
        self.concat = Concatenate()

    def __call__(self, unit_input, training=True):
        batchnorm = self.batchnorm(unit_input, training=training)
        relu = self.relu(batchnorm)
        conv = self.conv(relu)
        dropout = self.dropout(conv)
        concat = self.concat([unit_input, dropout])

        return concat


class DenseLayer(Layer):
    def __init__(self, num_unit, filter_out, kernel_size, dropout_rate=0.2):
        super().__init__()
        self.units = []

        for _ in range(num_unit):
            self.units.append(DenseUnit(filter_out, kernel_size, dropout_rate))

    def __call__(self, layer_input, training=True):
        x = layer_input
        for unit in self.units:
            x = unit(x, training=training)

        return x


class TransitionLayer(Layer):
    def __init__(self, kernel_size, pooling=False, compression_factor=0.5):
        super().__init__()

        self.compression_factor = compression_factor
        self.kernel_size = kernel_size

        if pooling:
            self.pool = AveragePooling2D((2, 2))
        else:
            self.pool = None

    def __call__(self, layer_input):
        reduced_filters = int(K.int_shape(layer_input)[-1] * self.compression_factor)
        mediator = Conv2D(reduced_filters, self.kernel_size, activation='relu', padding='same',
                          kernel_initializer='he_normal')(
            layer_input)
        if self.pool is not None:
            mediator = self.pool(mediator)

        return mediator
