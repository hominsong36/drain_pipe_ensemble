from keras.utils.vis_utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *

from models.Layers.layers import *
from models.Utils.utils import *


def DenseWNet(dropout_rate=0.2, training=True, pretrained_weights=None, input_size=(256, 256, 3),
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

    conv9 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(
        conv9)
    conv9 = DenseLayer(num_layer, mediator_filter // num_layer, (3, 3), dropout_rate)(conv9, training)
    trans9 = TransitionLayer((3, 3))(conv9)
    pool9 = MaxPooling2D((2, 2))(trans9)

    mediator_filter *= 2

    conv10 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(
        pool9)
    conv10 = DenseLayer(num_layer, mediator_filter // num_layer, (3, 3), dropout_rate)(conv10, training)
    trans10 = TransitionLayer((3, 3))(conv10)
    merge10 = Add()([conv8, trans10])
    pool10 = MaxPooling2D((2, 2))(merge10)

    mediator_filter *= 2

    conv11 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(
        pool10)
    conv11 = DenseLayer(num_layer, mediator_filter // num_layer, (3, 3), dropout_rate)(conv11, training)
    trans11 = TransitionLayer((3, 3))(conv11)
    merge11 = Add()([conv7, trans11])
    pool11 = MaxPooling2D((2, 2))(merge11)

    mediator_filter *= 2

    conv12 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(
        pool11)
    conv12 = DenseLayer(num_layer, mediator_filter // num_layer, (3, 3), dropout_rate)(conv12, training)
    trans12 = TransitionLayer((3, 3))(conv12)
    merge12 = Add()([conv6, trans12])
    pool12 = MaxPooling2D((2, 2))(merge12)

    mediator_filter *= 2

    conv13 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(
        pool12)
    conv13 = DenseLayer(num_layer, mediator_filter // num_layer, (3, 3), dropout_rate)(conv13, training)
    trans13 = TransitionLayer((3, 3))(conv13)
    merge13 = Add()([trans5, trans13])

    mediator_filter //= 2

    up14 = Conv2D(mediator_filter, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(merge13))
    merge14 = Add()([trans12, up14, trans4])
    conv14 = DenseLayer(num_layer, mediator_filter // num_layer, (3, 3), dropout_rate)(merge14, training)
    trans14 = TransitionLayer((3, 3))(conv14)
    conv14 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(
        trans14)

    mediator_filter //= 2

    up15 = Conv2D(mediator_filter, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv14))
    merge15 = Add()([trans11, up15, trans3])
    conv15 = DenseLayer(num_layer, mediator_filter // num_layer, (3, 3), dropout_rate)(merge15, training)
    trans15 = TransitionLayer((3, 3))(conv15)
    conv15 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(
        trans15)

    mediator_filter //= 2

    up16 = Conv2D(mediator_filter, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv15))
    merge16 = Add()([trans10, up16, trans2])
    conv16 = DenseLayer(num_layer, mediator_filter // num_layer, (3, 3), dropout_rate)(merge16, training)
    trans16 = TransitionLayer((3, 3))(conv16)
    conv16 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(
        trans16)

    mediator_filter //= 2

    up17 = Conv2D(mediator_filter, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv16))
    merge17 = Add()([trans9, up17, trans1])
    conv17 = DenseLayer(num_layer, mediator_filter // num_layer, (3, 3), dropout_rate)(merge17, training)
    trans17 = TransitionLayer((3, 3))(conv17)
    conv17 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(
        trans17)

    conv18 = Conv2D(4, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv17)
    conv18 = Conv2D(1, (1, 1), activation='linear', kernel_initializer='he_normal')(conv18)
    conv18 = Activation('sigmoid')(conv18)

    model = Model(inputs, conv18)

    model.compile(optimizer=Adam(lr=lr_schedule(0), clipnorm=0.8), loss=dice_loss, metrics=[dice])

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


if __name__ == "__main__":
    densewnet = DenseWNet()
    densewnet.summary()
    plot_model(model=densewnet, to_file='../images/densewnet.png')
