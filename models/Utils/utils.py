import tensorflow as tf

tf.to_int = lambda x: tf.cast(x, tf.int32)


def lr_schedule(epoch):
    lr = 1e-4
    if epoch > 800:
        lr *= 1e-2
    elif epoch > 600:
        lr *= 5e-2
    elif epoch > 400:
        lr *= 1e-1
    elif epoch > 200:
        lr *= 5e-1
    return lr


def dice_loss(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3)) + 1e-6
    denominator = tf.reduce_sum(y_true, axis=(1, 2, 3)) + tf.reduce_sum(y_pred, axis=(1, 2, 3)) + 1e-6

    return 1 - numerator / denominator

def dice(y_true, y_pred):
    numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=(1, 2, 3)) + 1e-6
    denominator = tf.reduce_sum(y_true, axis=(1, 2, 3)) + tf.reduce_sum(y_pred, axis=(1, 2, 3)) + 1e-6

    return numerator / denominator


#def mean_iou(y_true, y_pred):
#    
#    metric = tf.keras.metrics.MeanIoU(num_classes=2)
#    metric.update_state(y_true, y_pred)
#    
#    return metric.result()
