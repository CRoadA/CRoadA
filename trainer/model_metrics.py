import tensorflow as tf


@tf.keras.utils.register_keras_serializable()
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


@tf.keras.utils.register_keras_serializable()
def _dice_loss(y_true, y_pred):
    """Calculate the Dice loss for binary segmentation."""
    return 1.0 - _dice_coef(y_true, y_pred)


@tf.keras.utils.register_keras_serializable()
class FocalDiceLoss(tf.keras.losses.Loss):
    """Binary focal loss + weighted dice loss (serializable; no lambdas)."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.75,
        dice_weight: float = 0.5,
        connectivity_weight: float = 0.2,
        from_logits: bool = False,
        name: str = "focal_dice_loss",
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.dice_weight = dice_weight
        self.connectivity_weight = connectivity_weight
        self.from_logits = from_logits
        self._focal = tf.keras.losses.BinaryFocalCrossentropy(
            gamma=gamma,
            alpha=alpha,
            from_logits=from_logits,
        )

    def call(self, y_true, y_pred):
        focal_part = self._focal(y_true, y_pred)
        dice_part = self.dice_weight * _dice_loss(y_true, y_pred)
        conn_part = self.connectivity_weight * self.connectivity_loss(y_true, y_pred)

        return focal_part + dice_part + conn_part

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {
                "gamma": self.gamma,
                "alpha": self.alpha,
                "dice_weight": self.dice_weight,
                "connectivity_weight": self.connectivity_weight,
                "from_logits": self.from_logits,
            }
        )
        return cfg

    def connectivity_loss(self, y_true, y_pred):
        dy_true = y_true[:, 1:, :, :] - y_true[:, :-1, :, :]
        dx_true = y_true[:, :, 1:, :] - y_true[:, :, :-1, :]

        dy_pred = y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :]
        dx_pred = y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :]

        return tf.reduce_mean(tf.abs(dy_true - dy_pred)) + tf.reduce_mean(tf.abs(dx_true - dx_pred))
