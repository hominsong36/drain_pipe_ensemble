from keras.utils.vis_utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *

from models.Layers.layers import *
from models.Utils.utils import *


def ResWNet(dropout_rate=0.2, training=True, pretrained_weights=None, input_size=(256, 256, 3),
            mediator_filter_begin=8):
    mediator_filter = mediator_filter_begin
    inputs = Input(input_size)

    conv1 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = ResLayer(mediator_filter, mediator_filter, 2, (3, 3), dropout_rate)(conv1, training=training)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    mediator_filter *= 2

    conv2 = Conv2D(mediator_filter, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(pool1)
    conv2 = ResLayer(mediator_filter, mediator_filter, 2, (3, 3), dropout_rate)(conv2, training=training)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    mediator_filter *= 2

    conv3 = Conv2D(mediator_filter, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(pool2)
    conv3 = ResLayer(mediator_filter, mediator_filter, 2, (3, 3), dropout_rate)(conv3, training=training)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    mediator_filter *= 2

    conv4 = Conv2D(mediator_filter, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(pool3)
    conv4 = ResLayer(mediator_filter, mediator_filter, 2, (3, 3), dropout_rate)(conv4, training=training)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    mediator_filter *= 2

    conv5 = Conv2D(mediator_filter, (3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(pool4)
    conv5 = ResLayer(mediator_filter, mediator_filter, 2, (3, 3), dropout_rate)(conv5, training=training)

    mediator_filter //= 2

    up6 = Conv2D(mediator_filter, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv5))
    merge6 = Add()([conv4, up6])
    conv6 = ResLayer(mediator_filter, mediator_filter, 2, (3, 3), dropout_rate)(merge6, training=training)
    conv6 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    mediator_filter //= 2

    up7 = Conv2D(mediator_filter, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = Add()([conv3, up7])
    conv7 = ResLayer(mediator_filter, mediator_filter, 2, (3, 3), dropout_rate)(merge7, training=training)
    conv7 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    mediator_filter //= 2

    up8 = Conv2D(mediator_filter, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = Add()([conv2, up8])
    conv8 = ResLayer(mediator_filter, mediator_filter, 2, (3, 3), dropout_rate)(merge8, training=training)
    conv8 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    mediator_filter //= 2

    up9 = Conv2D(mediator_filter, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = Add()([conv1, up9])
    conv9 = ResLayer(mediator_filter, mediator_filter, 2, (3, 3), dropout_rate)(merge9, training=training)
    conv9 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv9 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(
        conv9)
    conv9 = ResLayer(mediator_filter, mediator_filter, 2, (3, 3), dropout_rate)(conv9, training=training)
    pool9 = MaxPooling2D(pool_size=(2, 2))(conv9)

    mediator_filter *= 2

    conv10 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(
        pool9)
    conv10 = ResLayer(mediator_filter, mediator_filter, 2, (3, 3), dropout_rate)(conv10, training=training)
    merge10 = Add()([conv8, conv10])
    pool10 = MaxPooling2D(pool_size=(2, 2))(merge10)

    mediator_filter *= 2

    conv11 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(
        pool10)
    conv11 = ResLayer(mediator_filter, mediator_filter, 2, (3, 3), dropout_rate)(conv11, training=training)
    merge11 = Add()([conv7, conv11])
    pool11 = MaxPooling2D(pool_size=(2, 2))(merge11)

    mediator_filter *= 2

    conv12 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(
        pool11)
    conv12 = ResLayer(mediator_filter, mediator_filter, 2, (3, 3), dropout_rate)(conv12, training=training)
    merge12 = Add()([conv6, conv12])
    pool12 = MaxPooling2D(pool_size=(2, 2))(merge12)

    mediator_filter *= 2

    conv13 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(
        pool12)
    conv13 = ResLayer(mediator_filter, mediator_filter, 2, (3, 3), dropout_rate)(conv13, training=training)
    merge13 = Add()([conv5, conv13])

    mediator_filter //= 2

    up14 = Conv2D(mediator_filter, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(merge13))
    merge14 = Add()([conv12, up14, conv4])
    conv14 = ResLayer(mediator_filter, mediator_filter, 2, (3, 3), dropout_rate)(merge14, training=training)
    conv14 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(
        conv14)

    mediator_filter //= 2

    up15 = Conv2D(mediator_filter, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv14))
    merge15 = Add()([conv11, up15, conv3])
    conv15 = ResLayer(mediator_filter, mediator_filter, 2, (3, 3), dropout_rate)(merge15, training=training)
    conv15 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(
        conv15)

    mediator_filter //= 2

    up16 = Conv2D(mediator_filter, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv15))
    merge16 = Add()([conv10, up16, conv2])
    conv16 = ResLayer(mediator_filter, mediator_filter, 2, (3, 3), dropout_rate)(merge16, training=training)
    conv16 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(
        conv16)

    mediator_filter //= 2

    up17 = Conv2D(mediator_filter, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv16))
    merge17 = Add()([conv9, up17, conv1])
    conv17 = ResLayer(mediator_filter, mediator_filter, 2, (3, 3), dropout_rate)(merge17, training=training)
    conv17 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(
        conv17)

    conv18 = Conv2D(4, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv17)
    conv18 = Conv2D(1, (1, 1), activation='linear', kernel_initializer='he_normal')(conv18)
    conv18 = Activation('sigmoid')(conv18)

    model = Model(inputs, conv18)

    model.compile(optimizer=Adam(lr=lr_schedule(0)), loss=dice_loss, metrics=[dice])

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


if __name__ == "__main__":
    reswnet = ResWNet()
    reswnet.summary()
    plot_model(model=reswnet, to_file='../images/reswnet.png')
