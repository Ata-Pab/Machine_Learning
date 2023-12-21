import tensorflow as tf
import losses
from attention_modules import Conv2DLayerBN, Conv2DLayerRes, CBAM, AttentionLayerUNet

def build_ConvAEModelV1(input_shape, latent_dim, layer_sizes=[32, 64, 128, 128, 256, 256, 256], **kwargs):
    num_strides = len(layer_sizes)
    num_dense = int(input_shape[0]/(2**num_strides))
    _kernel_size=5
    _strides=2
    _act_end="lrelu"
    _padding="same"
    _use_bias=True
    _lrelu_alpha=0.3

    # Argument setting
    for key, value in kwargs.items():
        if key == "kernel_size":
            _kernel_size = value
        if key == "strides":
            _strides = value
        if key == "act_end":
            _act_end = value
        if key == "padding":
            _padding = value
        if key == "use_bias":
            _use_bias = value
        if key == "lrelu_alpha":
            _lrelu_alpha = value
    
    # Encoder Part
    input = tf.keras.Input(shape=input_shape)

    # Conv blocks
    x = Conv2DLayerBN(filters=layer_sizes[0], kernel_size=_kernel_size, strides=_strides, 
                      padding=_padding, use_bias=_use_bias, act_end=_act_end, lrelu_alpha=_lrelu_alpha,
                      name='conv2d_block_0')(input)
    for ix, num_filter in enumerate(layer_sizes[1:]):
      _name = 'conv2d_block_' + str((ix+1))
      x = Conv2DLayerBN(filters=num_filter, kernel_size=_kernel_size, strides=_strides, 
                        padding=_padding, use_bias=_use_bias, act_end=_act_end, lrelu_alpha=_lrelu_alpha,
                        name='conv2d_block_0')(x)

    # Flatten layer
    x = tf.keras.layers.Flatten(name='flatten_layer')(x)
    x = tf.keras.layers.Dense(units=(num_dense*num_dense*256))(x)

    #x = BatchNormalization(name='bacth_norm_1')(x)
    x = tf.keras.layers.LeakyReLU(alpha=_lrelu_alpha)(x)
    # Latent vector
    x = tf.keras.layers.Dense(units=latent_dim, name='latent_layer')(x)
    x = tf.keras.layers.LeakyReLU(alpha=_lrelu_alpha)(x)

    # Decoder Part
    x = tf.keras.layers.Dense(units=(num_dense*num_dense*256))(x)
    #x = BatchNormalization(name='batch_norm_2')(x)
    x = tf.keras.layers.LeakyReLU(alpha=_lrelu_alpha)(x)

    x = tf.keras.layers.Reshape((num_dense,num_dense,256), name='reshape_latent')(x)

    # Transposed Conv2D block
    for ix, num_filter in enumerate(layer_sizes[-2::-1]):  # Reverse layer_sizes list and remove first element of the reversed array
        _name = 'conv2d_transpose_block_' + str((ix+1))
        x = tf.keras.layers.Conv2DTranspose(filters=num_filter, kernel_size=_kernel_size, 
                                            strides=_strides, padding=_padding, use_bias=_use_bias, name=_name)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        # Set activation func to ReLu for Conv2dTrasnpose blocks
        x = tf.keras.layers.ReLU()(x)

    output = tf.keras.layers.Conv2DTranspose(filters=input_shape[2], kernel_size=_kernel_size,
                                             strides=_strides, padding=_padding, use_bias=_use_bias,
    use_bias=True, activation='sigmoid')(x)

    return tf.keras.Model(input, output, name="conv_ae_model")