from keras.utils.vis_utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *

from models.Layers.layers import *
from models.Utils.utils import *

def dice(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3)) + 1e-6
    denominator = tf.reduce_sum(y_true, axis=(1, 2, 3)) + tf.reduce_sum(y_pred, axis=(1, 2, 3)) + 1e-6

    return numerator / denominator

def DenseUNet(dropout_rate=0.2, training=True, pretrained_weights=None, input_size=(256, 256, 3),
              mediator_filter_begin=8, num_layer=2):
    mediator_filter = mediator_filter_begin
    inputs = Input(input_size)

    conv1 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = DenseLayer(num_layer, mediator_filter // num_layer, (3, 3), dropout_rate)(conv1, training)
    trans1 = TransitionLayer((3, 3))(conv1)
    pool1 = MaxPooling2D((2, 2))(trans1)

    mediator_filter *= 2

    conv2 = Conv2D(mediator_filter, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(pool1)
    conv2 = DenseLayer(num_layer, mediator_filter // num_layer, (3, 3), dropout_rate)(conv2, training)
    trans2 = TransitionLayer((3, 3))(conv2)
    pool2 = MaxPooling2D((2, 2))(trans2)

    mediator_filter *= 2

    conv3 = Conv2D(mediator_filter, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(pool2)
    conv3 = DenseLayer(num_layer, mediator_filter // num_layer, (3, 3), dropout_rate)(conv3, training)
    trans3 = TransitionLayer((3, 3))(conv3)
    pool3 = MaxPooling2D((2, 2))(trans3)

    mediator_filter *= 2

    conv4 = Conv2D(mediator_filter, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(pool3)
    conv4 = DenseLayer(num_layer, mediator_filter // num_layer, (3, 3), dropout_rate)(conv4, training)
    trans4 = TransitionLayer((3, 3))(conv4)
    pool4 = MaxPooling2D((2, 2))(trans4)

    mediator_filter *= 2

    conv5 = Conv2D(mediator_filter, (1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(pool4)
    conv5 = DenseLayer(num_layer, mediator_filter // num_layer, (3, 3), dropout_rate)(conv5, training)
    trans5 = TransitionLayer((3, 3))(conv5)

    mediator_filter //= 2

    up6 = Conv2D(mediator_filter, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(trans5))
    merge6 = Add()([trans4, up6])
    conv6 = DenseLayer(num_layer, mediator_filter // num_layer, (3, 3), dropout_rate)(merge6, training)
    trans6 = TransitionLayer((3, 3))(conv6)
    conv6 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(
        trans6)

    mediator_filter //= 2

    up7 = Conv2D(mediator_filter, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = Add()([trans3, up7])
    conv7 = DenseLayer(num_layer, mediator_filter // num_layer, (3, 3), dropout_rate)(merge7, training)
    trans7 = TransitionLayer((3, 3))(conv7)
    conv7 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(
        trans7)

    mediator_filter //= 2

    up8 = Conv2D(mediator_filter, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = Add()([trans2, up8])
    conv8 = DenseLayer(num_layer, mediator_filter // num_layer, (3, 3), dropout_rate)(merge8, training)
    trans8 = TransitionLayer((3, 3))(conv8)
    conv8 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(
        trans8)

    mediator_filter //= 2

    up9 = Conv2D(mediator_filter, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = Add()([trans1, up9])
    conv9 = DenseLayer(num_layer, mediator_filter // num_layer, (3, 3), dropout_rate)(merge9, training)
    trans9 = TransitionLayer((3, 3))(conv9)
    conv9 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(
        trans9)

    conv10 = Conv2D(4, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, (1, 1), activation='linear', padding='same', kernel_initializer='he_normal')(conv10)
    conv10 = Activation('sigmoid')(conv10)

    model = Model(inputs, conv10)

    model.compile(optimizer=Adam(lr=lr_schedule(0)), loss=dice_loss, metrics=[dice])

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


if __name__ == "__main__":
    denseunet = DenseUNet(dropout_rate=0.2, training=True)
    denseunet.summary()
    plot_model(model=denseunet, to_file='../images/denseunet.png')
