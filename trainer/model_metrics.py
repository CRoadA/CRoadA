import tensorflow as tf


def _dice_coef(y_true, y_pred, smooth=1e-6):
    """Calculate the Dice coefficient for binary segmentation."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)

    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    denominator = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)

    return (2.0 * intersection + smooth) / (denominator + smooth)


def _dice_loss(y_true, y_pred):
    """Calculate the Dice loss for binary segmentation."""
    return 1.0 - _dice_coef(y_true, y_pred)
