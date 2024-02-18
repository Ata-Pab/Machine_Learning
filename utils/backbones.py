import tensorflow as tf
from utils.attention_modules import Conv2DLayerBN, Conv2DLayerRes, CBAM, AttentionLayerUNet, Conv2DLayerResCBAM
import warnings

def build_ConvAEModelV1(input_shape, latent_dim, layer_sizes=[32, 64, 128, 128, 256, 256, 256], **kwargs):
    num_strides = len(layer_sizes)

    if (input_shape[0] < 224) or (input_shape[1] < 224):
        raise ValueError("Input width or height cannot be lower than 224")
    
    num_dense = int(input_shape[0]/(2**num_strides))
    use_dense_layers = True

    if (num_dense*num_dense*layer_sizes[-1]) > 4096:
        use_dense_layers = False
        warnings.warn("The number of neurons in the Dense layer should not exceed 2048 to fit into RAM. Resize image (downsize) or increase number of layers to be applied! Dense layers will not be used in encoder part. It makes accuracy down.", ResourceWarning)
        #raise ValueError("The number of neurons in the Dense layer should not exceed 2048 to fit into RAM. Resize image (downsize) or increase number of layers to be applied!")
    
    if num_dense <= 0: num_dense = 1
    
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
                        name=_name)(x)

    # Flatten layer
    x = tf.keras.layers.Flatten(name='flatten_layer')(x)
    if use_dense_layers:
        x = tf.keras.layers.Dense(units=(num_dense*num_dense*layer_sizes[-1]))(x)
        #x = BatchNormalization(name='bacth_norm_1')(x)
        x = tf.keras.layers.LeakyReLU(alpha=_lrelu_alpha)(x)

    # Latent vector
    x = tf.keras.layers.Dense(units=latent_dim, name='latent_layer')(x)
    x = tf.keras.layers.LeakyReLU(alpha=_lrelu_alpha)(x)

    # Decoder Part
    x = tf.keras.layers.Dense(units=(num_dense*num_dense*layer_sizes[-1]))(x)
    #x = BatchNormalization(name='batch_norm_2')(x)
    x = tf.keras.layers.LeakyReLU(alpha=_lrelu_alpha)(x)

    x = tf.keras.layers.Reshape((num_dense,num_dense,layer_sizes[-1]), name='reshape_latent')(x)

    # Transposed Conv2D blocks
    for ix, num_filter in enumerate(layer_sizes[-2::-1]):  # Reverse layer_sizes list and remove first element of the reversed array
        _name = 'conv2d_transpose_block_' + str((ix+1))
        x = tf.keras.layers.Conv2DTranspose(filters=num_filter, kernel_size=_kernel_size, 
                                            strides=_strides, padding=_padding, use_bias=_use_bias, name=_name)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        # Set activation func to ReLu for Conv2dTrasnpose blocks
        x = tf.keras.layers.ReLU()(x)

    output = tf.keras.layers.Conv2DTranspose(filters=input_shape[2], kernel_size=_kernel_size,
                                             strides=_strides, padding=_padding, use_bias=_use_bias,
                                              activation='sigmoid')(x)

    return tf.keras.Model(input, output, name="conv_ae_model")


def build_CBAMConvAEModelV1(input_shape, latent_dim, layer_sizes=[32, 64, 128, 128, 256, 256, 256], 
                                 reduction_ratio=16, attention_for_decoder=True, **kwargs):
    '''
    CBAM AutoEncoder: Convolutional Block Attention Module AutoEncoder
    '''
    num_strides = len(layer_sizes)

    if (input_shape[0] < 224) or (input_shape[1] < 224):
        raise ValueError("Input width or height cannot be lower than 224")
    
    num_dense = int(input_shape[0]/(2**num_strides))
    use_dense_layers = True

    if (num_dense*num_dense*layer_sizes[-1]) > 4096:
        use_dense_layers = False
        warnings.warn("The number of neurons in the Dense layer should not exceed 2048 to fit into RAM. Resize image (downsize) or increase number of layers to be applied! Dense layers will not be used in encoder part. It makes accuracy down.", ResourceWarning)
        #raise ValueError("The number of neurons in the Dense layer should not exceed 2048 to fit into RAM. Resize image (downsize) or increase number of layers to be applied!")
    
    if num_dense <= 0: num_dense = 1

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
    x = CBAM(gate_channels=layer_sizes[0], reduction_ratio=reduction_ratio)(x)
    for ix, num_filter in enumerate(layer_sizes[1:]):
        _name = 'conv2d_block_' + str((ix+1))
        x = Conv2DLayerBN(filters=num_filter, kernel_size=_kernel_size, strides=_strides, 
                          padding=_padding, use_bias=_use_bias, act_end=_act_end, lrelu_alpha=_lrelu_alpha,
                          name=_name)(x)
        x = CBAM(gate_channels=num_filter, reduction_ratio=reduction_ratio)(x)

    # Flatten layer
    x = tf.keras.layers.Flatten(name='flatten_layer')(x)
    if use_dense_layers:
        x = tf.keras.layers.Dense(units=(num_dense*num_dense*layer_sizes[-1]))(x)
        #x = BatchNormalization(name='bacth_norm_1')(x)
        x = tf.keras.layers.LeakyReLU(alpha=_lrelu_alpha)(x)

    # Latent vector
    x = tf.keras.layers.Dense(units=latent_dim, name='latent_layer')(x)
    x = tf.keras.layers.LeakyReLU(alpha=_lrelu_alpha)(x)

    # Decoder Part
    x = tf.keras.layers.Dense(units=(num_dense*num_dense*layer_sizes[-1]))(x)
    #x = BatchNormalization(name='batch_norm_2')(x)
    x = tf.keras.layers.LeakyReLU(alpha=_lrelu_alpha)(x)

    x = tf.keras.layers.Reshape((num_dense,num_dense,layer_sizes[-1]), name='reshape_latent')(x)

    # Transposed Conv2D blocks
    for ix, num_filter in enumerate(layer_sizes[-2::-1]):  # Reverse layer_sizes list and remove first element of the reversed array
        _name = 'conv2d_transpose_block_' + str((ix+1))
        x = tf.keras.layers.Conv2DTranspose(filters=num_filter, kernel_size=_kernel_size, 
                                            strides=_strides, padding=_padding, use_bias=_use_bias, name=_name)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        # Set activation func to ReLu for Conv2dTrasnpose blocks
        x = tf.keras.layers.ReLU()(x)
        if attention_for_decoder:
            x = CBAM(gate_channels=num_filter, reduction_ratio=reduction_ratio)(x)

    output = tf.keras.layers.Conv2DTranspose(filters=input_shape[2], kernel_size=_kernel_size,
                                             strides=_strides, padding=_padding, use_bias=_use_bias,
                                              activation='sigmoid')(x)

    return tf.keras.Model(input, output, name="conv_ae_model")

def build_ResCBAMConvAEModelV1(input_shape, latent_dim, layer_sizes=[32, 64, 128, 128, 256, 256, 256], 
                                 reduction_ratio=16, attention_for_decoder=True, **kwargs):
    '''
    ResCBAM AutoEncoder: Residual Convolutional Block Attention Module AutoEncoder
    '''
    num_strides = len(layer_sizes)

    if (input_shape[0] < 224) or (input_shape[1] < 224):
        raise ValueError("Input width or height cannot be lower than 224")
    
    num_dense = int(input_shape[0]/(2**num_strides))
    use_dense_layers = True

    if (num_dense*num_dense*layer_sizes[-1]) > 4096:
        use_dense_layers = False
        warnings.warn("The number of neurons in the Dense layer should not exceed 2048 to fit into RAM. Resize image (downsize) or increase number of layers to be applied! Dense layers will not be used in encoder part. It makes accuracy down.", ResourceWarning)
        #raise ValueError("The number of neurons in the Dense layer should not exceed 2048 to fit into RAM. Resize image (downsize) or increase number of layers to be applied!")
    
    if num_dense <= 0: num_dense = 1

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
    x = Conv2DLayerResCBAM(filters=layer_sizes[0], kernel_size=_kernel_size, strides=_strides, 
                      padding=_padding, use_bias=_use_bias, act_end=_act_end, lrelu_alpha=_lrelu_alpha,
                      reduction_ratio=reduction_ratio, name='conv2d_block_0')(input)
    for ix, num_filter in enumerate(layer_sizes[1:]):
        _name = 'conv2d_block_' + str((ix+1))
        x = Conv2DLayerResCBAM(filters=num_filter, kernel_size=_kernel_size, strides=_strides, 
                          padding=_padding, use_bias=_use_bias, act_end=_act_end, lrelu_alpha=_lrelu_alpha,
                          reduction_ratio=reduction_ratio, name=_name)(x)

    # Flatten layer
    x = tf.keras.layers.Flatten(name='flatten_layer')(x)
    if use_dense_layers:
        x = tf.keras.layers.Dense(units=(num_dense*num_dense*layer_sizes[-1]))(x)
        #x = BatchNormalization(name='bacth_norm_1')(x)
        x = tf.keras.layers.LeakyReLU(alpha=_lrelu_alpha)(x)

    # Latent vector
    x = tf.keras.layers.Dense(units=latent_dim, name='latent_layer')(x)
    x = tf.keras.layers.LeakyReLU(alpha=_lrelu_alpha)(x)

    # Decoder Part
    x = tf.keras.layers.Dense(units=(num_dense*num_dense*layer_sizes[-1]))(x)
    #x = BatchNormalization(name='batch_norm_2')(x)
    x = tf.keras.layers.LeakyReLU(alpha=_lrelu_alpha)(x)

    x = tf.keras.layers.Reshape((num_dense,num_dense,layer_sizes[-1]), name='reshape_latent')(x)

    # Transposed Conv2D blocks
    for ix, num_filter in enumerate(layer_sizes[-2::-1]):  # Reverse layer_sizes list and remove first element of the reversed array
        _name = 'conv2d_transpose_block_' + str((ix+1))
        x = tf.keras.layers.Conv2DTranspose(filters=num_filter, kernel_size=_kernel_size, 
                                            strides=_strides, padding=_padding, use_bias=_use_bias, name=_name)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        # Set activation func to ReLu for Conv2dTrasnpose blocks
        x = tf.keras.layers.ReLU()(x)
        if attention_for_decoder:
            x = CBAM(gate_channels=num_filter, reduction_ratio=reduction_ratio)(x)

    output = tf.keras.layers.Conv2DTranspose(filters=input_shape[2], kernel_size=_kernel_size,
                                             strides=_strides, padding=_padding, use_bias=_use_bias,
                                              activation='sigmoid')(x)

    return tf.keras.Model(input, output, name="conv_ae_model")