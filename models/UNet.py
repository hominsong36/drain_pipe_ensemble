from keras.utils.vis_utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *

from models.Layers.layers import *
from models.Utils.utils import *


def UNet(dropout_rate=0.2, pretrained_weights=None, input_size=(256, 256, 3),
         mediator_filter_begin=8):
    mediator_filter = mediator_filter_begin
    inputs = Input(input_size)
    conv1 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    mediator_filter *= 2

    conv2 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    mediator_filter *= 2

    conv3 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    mediator_filter *= 2

    conv4 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(dropout_rate)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    mediator_filter *= 2

    conv5 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(dropout_rate)(conv5)

    mediator_filter //= 2

    up6 = Conv2D(mediator_filter, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = Add()([conv4, up6])
    conv6 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    mediator_filter //= 2

    up7 = Conv2D(mediator_filter, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = Add()([conv3, up7])
    conv7 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    mediator_filter //= 2

    up8 = Conv2D(mediator_filter, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = Add()([conv2, up8])
    conv8 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    mediator_filter //= 2

    up9 = Conv2D(mediator_filter, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = Add()([conv1, up9])
    conv9 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv10 = Conv2D(4, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, (1, 1), activation='linear', kernel_initializer='he_normal')(conv10)
    conv10 = Activation('sigmoid')(conv10)

    model = Model(inputs, conv10)

    optimizer = Adam(lr=lr_schedule(0))
    model.compile(optimizer=optimizer, loss=dice_loss, metrics=[dice])

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


if __name__ == "__main__":
    unet = UNet()
    unet.summary()
    plot_model(model=unet, to_file='../images/unet.png')
