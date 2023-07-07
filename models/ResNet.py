from keras.utils.vis_utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import *

from models.Layers.layers import *
from models.Utils.utils import *


def ResNet(dropout_rate=0.2, training=True, pretrained_weights=None, input_size=(256, 256, 3),
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
    conv6 = ResLayer(mediator_filter, mediator_filter, 2, (3, 3), dropout_rate)(up6, training=training)
    conv6 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    mediator_filter //= 2

    up7 = Conv2D(mediator_filter, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    conv7 = ResLayer(mediator_filter, mediator_filter, 2, (3, 3), dropout_rate)(up7, training=training)
    conv7 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    mediator_filter //= 2

    up8 = Conv2D(mediator_filter, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    conv8 = ResLayer(mediator_filter, mediator_filter, 2, (3, 3), dropout_rate)(up8, training=training)
    conv8 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    mediator_filter //= 2

    up9 = Conv2D(mediator_filter, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    conv9 = ResLayer(mediator_filter, mediator_filter, 2, (3, 3), dropout_rate)(up9, training=training)
    conv9 = Conv2D(mediator_filter, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)

    conv10 = Conv2D(4, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, (1, 1), activation='linear', kernel_initializer='he_normal')(conv10)
    conv10 = Activation('sigmoid')(conv10)

    model = Model(inputs, conv10)

    model.compile(optimizer=Adam(lr=lr_schedule(0)), loss=dice_loss, metrics=[dice])

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model


if __name__ == "__main__":
    resnet = ResNet()
    resnet.summary()
    plot_model(model=resnet, to_file='../images/resnet.png')
