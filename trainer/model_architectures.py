import tensorflow as tf

# test model
def base_clipping_model(clipping_size=512, clipping_surplus=64, third_dimension=4, **kwargs):
    inputs = tf.keras.layers.Input(shape=(clipping_size, clipping_size, third_dimension))
    x = tf.keras.layers.Conv2D(16, 5, activation="relu", padding="same", strides=1)(inputs)
    x = tf.keras.layers.Conv2D(8, 5, activation="relu", padding="same", strides=1)(x)
    x = tf.keras.layers.Conv2D(4, 5, activation="relu", padding="same", strides=1)(x)

    # How much to crop to get rid of surplus
    crop = clipping_surplus // 2
    x = tf.keras.layers.Cropping2D(cropping=((crop, crop), (crop, crop)))(x)

    # One output layer with two channels
    x = tf.keras.layers.Conv2D(2, 1, activation=None, name="output")(x)

    # Split outputs into two separate heads
    out_is_street = tf.keras.layers.Lambda(lambda t: tf.keras.activations.sigmoid(t[..., 0:1]), name="is_street", dtype="float32")(x)
    out_altitude = tf.keras.layers.Lambda(lambda t: t[..., 1:2], name="altitude", dtype="float32")(x)
    outputs = [out_is_street, out_altitude]

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

# UNet architecture for clipping model proposed by ChatGPT
def unet(clipping_size=512, clipping_surplus=64, third_dimension=4, **kwargs):
    inputs = tf.keras.layers.Input(shape=(clipping_size, clipping_size, third_dimension))

    # Encoder
    c1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = tf.keras.layers.MaxPooling2D()(c1)

    c2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D()(c2)

    c3 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(p2)
    c3 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D()(c3)

    # Bottleneck
    b = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(p3)
    b = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(b)
    # Decoder
    u3 = tf.keras.layers.UpSampling2D()(b)
    u3 = tf.keras.layers.concatenate([u3, c3])
    c4 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(u3)
    c4 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(c4)

    u2 = tf.keras.layers.UpSampling2D()(c4)
    u2 = tf.keras.layers.concatenate([u2, c2])
    c5 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(u2)
    c5 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(c5)

    u1 = tf.keras.layers.UpSampling2D()(c5)
    u1 = tf.keras.layers.concatenate([u1, c1])
    c6 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(u1)
    c6 = tf.keras.layers.Conv2D(32, 3, activation='relu', padding='same')(c6)

    # Crop to remove surplus
    crop = clipping_surplus // 2
    c6 = tf.keras.layers.Cropping2D(cropping=((crop, crop), (crop, crop)))(c6)

    # Output heads
    street_out = tf.keras.layers.Conv2D(1, 1, activation='sigmoid', name='is_street')(c6)
    altitude_out = tf.keras.layers.Conv2D(1, 1, activation='linear', name='altitude')(c6)

    model = tf.keras.Model(inputs, [street_out, altitude_out])
    return model

def alex_inspired(clipping_size=512, clipping_surplus=64, third_dimension=4, **kwargs):
    # AlexNet-inspired architecture for clipping model

    # Input layer
    inputs = tf.keras.layers.Input(shape=(clipping_size, clipping_size, third_dimension))

    # Encoder
    c1 = tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4, activation='relu', padding="same")(inputs)
    s2 = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="valid")(c1)
    c3 = tf.keras.layers.Conv2D(filters=256, kernel_size=5, strides=1, activation='relu', padding="same")(s2)
    s4 = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="valid")(c3)
    c5 = tf.keras.layers.Conv2D(filters=384, kernel_size=3, strides=1, activation='relu', padding="same")(s4)
    c6 = tf.keras.layers.Conv2D(filters=384, kernel_size=3, strides=1, activation='relu', padding="same")(c5)
    c7 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, activation='relu', padding="same")(c6)
    s8 = tf.keras.layers.MaxPooling2D(pool_size=3, strides=2, padding="valid")(c7)

    # Bottleneck
    b9 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, activation='relu', padding="same")(s8)
    b10 = tf.keras.layers.Conv2D(filters=512, kernel_size=3, strides=1, activation='relu', padding="same")(b9)

    # Decoder
    u11 = tf.keras.layers.UpSampling2D()(b10)
    u11 = tf.keras.layers.concatenate([u11, c7])
    c12 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, activation='relu', padding="same")(u11)
    c13 = tf.keras.layers.Conv2D(filters=384, kernel_size=3, strides=1, activation='relu', padding="same")(c12)
    c14 = tf.keras.layers.Conv2D(filters=384, kernel_size=3, strides=1, activation='relu', padding="same")(c13)
    u15 = tf.keras.layers.UpSampling2D()(c14)
    u15 = tf.keras.layers.concatenate([u15, c3])
    c16 = tf.keras.layers.Conv2D(filters=256, kernel_size=5, strides=1, activation='relu', padding="same")(u15)
    u17 = tf.keras.layers.UpSampling2D()(c16)
    u17 = tf.keras.layers.concatenate([u17, c1])
    c18 = tf.keras.layers.Conv2D(filters=96, kernel_size=11, strides=4, activation='relu', padding="same")(u17)

    # Crop to remove surplus
    crop = clipping_surplus // 2
    c18 = tf.keras.layers.Cropping2D(cropping=((crop, crop), (crop, crop)))(c18)

    # Output heads
    out_is_street = tf.keras.layers.Conv2D(1, 1, activation='sigmoid', name='is_street')(c18)
    out_altitude = tf.keras.layers.Conv2D(1, 1, activation='linear', name='altitude')(c18)
    outputs = [out_is_street, out_altitude]

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model