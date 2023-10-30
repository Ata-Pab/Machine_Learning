import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras import backend as K

def conv_block(x, filter_size, size, dropout, batch_norm=False):
    conv = layers.Conv2D(size, (filter_size, filter_size), padding="same")(x)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
        # (batch_size, height, width, channels): axis=3 -> channels
    conv = layers.Activation("relu")(conv)

    conv = layers.Conv2D(size, (filter_size, filter_size), padding="same")(conv)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation("relu")(conv)

    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    return conv

def repeat_elem(tensor, rep):
    # lambda function to repeat the elements of a tensor along an axis by a
    # factor of rep. If tensor has shape (None, 256, 256, 3), lambda will return
    # a tensor of shape (None, 256,256,6), if specified axis=3 and rep=2.

    return layers.Lambda(lambda x, repnum: tf.repeat(x, repnum, axis=3),   # K.repeat_elements(x, repnum, axis=3),
                          arguments={'repnum': rep})(tensor)

def res_conv_block(x, filter_size, size, dropout, batch_norm=False):
    '''
    There are two options for Residual convolutional layer.
    Either put activation function before the addition with shortcut
    or after the addition (which would be as proposed in the original resNet).

    1. conv - BN - Activation - conv - BN - Activation
                                          - shortcut - BN - shortcut + BN

    2. conv - BN - Activation - conv - BN
                                     - shortcut - BN - shortcut + BN - Activation

    ref: https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf  (fig.4)
    '''

    # Conv2D - BN - Activation
    conv = layers.Conv2D(size, (filter_size, filter_size), padding='same')(x)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    conv = layers.Activation('relu')(conv)

    # Conv2D - BN - Dropout
    conv = layers.Conv2D(size, (filter_size, filter_size), padding='same')(conv)
    if batch_norm is True:
        conv = layers.BatchNormalization(axis=3)(conv)
    #conv = layers.Activation('relu')(conv)    #Activation before addition with shortcut
    if dropout > 0:
        conv = layers.Dropout(dropout)(conv)

    # Conv2D - BN  (for shortcut)
    shortcut = layers.Conv2D(size, kernel_size=(1, 1), padding='same')(x)
    if batch_norm is True:
        shortcut = layers.BatchNormalization(axis=3)(shortcut)

    res_path = layers.add([shortcut, conv])
    res_path = layers.Activation('relu')(res_path)    # Activation after addition with shortcut (Original residual block)
    return res_path

def gating_signal(input, out_size, batch_norm=False):
    """
    resize the down layer feature map into the same dimension as the up layer feature map
    using 1x1 conv
    :return: the gating feature map with the same dimension of the up layer feature map
    """
    # Conv2D - BN - Activation
    x = layers.Conv2D(out_size, (1, 1), padding='same')(input)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    return x

# Sample attention block
def attention_block(x, gating, inter_shape):
  shape_x = K.int_shape(x)  # x
  shape_g = K.int_shape(gating)  # g

  # Getting x to the same shape as gating signal (128 to 64)
  theta_x = layers.Conv2D(inter_shape, (2,2), strides=(2,2), padding='same')(x)
  shape_theta_x = K.int_shape(theta_x)

  # Getting the gating signal to the same number of filters as the inter_shape
  phi_g = layers.Conv2D(inter_shape, (1,1), padding='same')(gating)
  upsample_g = layers.Conv2DTranspose(inter_shape, (3, 3),
                      strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),
                      padding='same')(phi_g)  # 16

  concat_xg = layers.add([upsample_g, theta_x])
  act_xg = layers.Activation('relu')(concat_xg)

  psi = layers.Conv2D(1, (1,1), padding='same')(act_xg)
  sigmoid_xg = layers.Activation('sigmoid')(psi)
  shape_sigmoid = K.int_shape(sigmoid_xg)

  upsample_psi = layers.UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)

  y = layers.multiply([upsample_psi, x])

  result = layers.Conv2D(shape_x[3], (1,1), padding='same')(y)
  result_bn = layers.BatchNormalization()(result)

  return result_bn

def UNet(input_shape, filter_size=3, filter_num=64, up_sample_size=2, num_classes=1, dropout_rate=0.0, batch_norm=True, verbose=0):
    '''
    filter_num: number of filters for the conv layer
    filter_size: size of the convolutional filter (3,3)
    up_sample_size: size of upsampling filters

    :return U-Net Model
    '''
    inputs = layers.Input(input_shape, dtype=tf.float32)

    # Downsampling layers
    # DownRes 1, Conv - Pool
    conv_128 = conv_block(inputs, filter_size, filter_num, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)

    # DownRes 2, Conv - Pool
    conv_64 = conv_block(pool_64, filter_size, 2*filter_num, dropout_rate, batch_norm)
    pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)

    # DownRes 3, Conv - Pool
    conv_32 = conv_block(pool_32, filter_size, 4*filter_num, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)

    # DownRes 4, Conv - Pool
    conv_16 = conv_block(pool_16, filter_size, 8*filter_num, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)

    # DownRes 5, Conv
    conv_8 = conv_block(pool_8, filter_size, 16*filter_num, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, UpSamp - Concat - Conv
    up_16 = layers.UpSampling2D(size=(up_sample_size, up_sample_size), data_format="channels_last")(conv_8)
    up_16 = layers.concatenate([up_16, conv_16], axis=3)  # Concat with conv_16
    up_conv_16 = conv_block(up_16, filter_size, 8*filter_num, dropout_rate, batch_norm)

    # UpRes 7, UpSamp - Concat - Conv
    up_32 = layers.UpSampling2D(size=(up_sample_size, up_sample_size), data_format="channels_last")(up_conv_16)
    up_32 = layers.concatenate([up_32, conv_32], axis=3)  # Concat with conv_32
    up_conv_32 = conv_block(up_32, filter_size, 4*filter_num, dropout_rate, batch_norm)

    # UpRes 8, UpSamp - Concat - Conv
    up_64 = layers.UpSampling2D(size=(up_sample_size, up_sample_size), data_format="channels_last")(up_conv_32)
    up_64 = layers.concatenate([up_64, conv_64], axis=3)  # Concat with conv_64
    up_conv_64 = conv_block(up_64, filter_size, 2*filter_num, dropout_rate, batch_norm)

    # UpRes 9, UpSamp - Concat - Conv
    up_128 = layers.UpSampling2D(size=(up_sample_size, up_sample_size), data_format="channels_last")(up_conv_64)
    up_128 = layers.concatenate([up_128, conv_128], axis=3) # Concat with conv_128
    up_conv_128 = conv_block(up_128, filter_size, filter_num, dropout_rate, batch_norm)

    # 1*1 convolutional layers
    # Conv - BN - Activation
    conv_final = layers.Conv2D(num_classes, kernel_size=(1,1))(up_conv_128)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('sigmoid')(conv_final)  #Change to softmax for multichannel

    # Model
    model = models.Model(inputs, conv_final, name="UNet")
    if verbose > 0:
        model.summary()
    return model

def Attention_UNet(input_shape, filter_size=3, filter_num=64, up_sample_size=2, num_classes=1, dropout_rate=0.0, batch_norm=True, verbose=0):
    '''
    filter_num: number of filters for the conv layer
    filter_size: size of the convolutional filter (3,3)
    up_sample_size: size of upsampling filters

    :return Attention Model
    '''
    inputs = layers.Input(input_shape, dtype=tf.float32)

    # Downsampling layers - Same as U-Net architecture
    # DownRes 1, Conv - Pool
    conv_128 = conv_block(inputs, filter_size, filter_num, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)

    # DownRes 2, Conv - Pool
    conv_64 = conv_block(pool_64, filter_size, 2*filter_num, dropout_rate, batch_norm)
    pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)

    # DownRes 3, Conv - Pool
    conv_32 = conv_block(pool_32, filter_size, 4*filter_num, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)

    # DownRes 4, Conv - Pool
    conv_16 = conv_block(pool_16, filter_size, 8*filter_num, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)

    # DownRes 5, Conv
    conv_8 = conv_block(pool_8, filter_size, 16*filter_num, dropout_rate, batch_norm)

    # Upsampling layers
    # UpRes 6, Attention_Gated_Concat - UpSamp - Concat - Double_Residual_Conv
    gating_16 = gating_signal(conv_8, 8*filter_num, batch_norm)
    att_16 = attention_block(conv_16, gating_16, 8*filter_num)
    up_16 = layers.UpSampling2D(size=(up_sample_size, up_sample_size), data_format="channels_last")(conv_8)
    up_16 = layers.concatenate([up_16, att_16], axis=3)
    up_conv_16 = conv_block(up_16, filter_size, 8*filter_num, dropout_rate, batch_norm)

    # UpRes 7, Attention_Gated_Concat - UpSamp - Concat - Double_Residual_Conv
    gating_32 = gating_signal(up_conv_16, 4*filter_num, batch_norm)
    att_32 = attention_block(conv_32, gating_32, 4*filter_num)
    up_32 = layers.UpSampling2D(size=(up_sample_size, up_sample_size), data_format="channels_last")(up_conv_16)
    up_32 = layers.concatenate([up_32, att_32], axis=3)
    up_conv_32 = conv_block(up_32, filter_size, 4*filter_num, dropout_rate, batch_norm)

    # UpRes 8, Attention_Gated_Concat - UpSamp - Concat - Double_Residual_Conv
    gating_64 = gating_signal(up_conv_32, 2*filter_num, batch_norm)
    att_64 = attention_block(conv_64, gating_64, 2*filter_num)
    up_64 = layers.UpSampling2D(size=(up_sample_size, up_sample_size), data_format="channels_last")(up_conv_32)
    up_64 = layers.concatenate([up_64, att_64], axis=3)
    up_conv_64 = conv_block(up_64, filter_size, 2*filter_num, dropout_rate, batch_norm)

    # UpRes 9, Attention_Gated_Concat - UpSamp - Concat - Double_Residual_Conv
    gating_128 = gating_signal(up_conv_64, filter_num, batch_norm)
    att_128 = attention_block(conv_128, gating_128, filter_num)
    up_128 = layers.UpSampling2D(size=(up_sample_size, up_sample_size), data_format="channels_last")(up_conv_64)
    up_128 = layers.concatenate([up_128, att_128], axis=3)
    up_conv_128 = conv_block(up_128, filter_size, filter_num, dropout_rate, batch_norm)

    # 1*1 convolutional layers
    # Conv - BN - Activation
    conv_final = layers.Conv2D(num_classes, kernel_size=(1,1))(up_conv_128)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('sigmoid')(conv_final)  #Change to softmax for multichannel

    # Model integration
    model = models.Model(inputs, conv_final, name="Attention_UNet")
    if verbose > 0:
        model.summary()
    return model

def Attention_ResUNet(input_shape, filter_size=3, filter_num=64, up_sample_size=2, num_classes=1, dropout_rate=0.0, batch_norm=True, verbose=0):
    '''
    filter_num: number of filters for the conv layer
    filter_size: size of the convolutional filter (3,3)
    up_sample_size: size of upsampling filters

    :return Attention ResNet Model
    '''
    inputs = layers.Input(input_shape, dtype=tf.float32)

    # Downsampling layers
    # DownRes 1, Double_Residual_Conv - Pool
    conv_128 = res_conv_block(inputs, filter_size, filter_num, dropout_rate, batch_norm)
    pool_64 = layers.MaxPooling2D(pool_size=(2,2))(conv_128)

    # DownRes 2, Double_Residual_Conv - Pool
    conv_64 = res_conv_block(pool_64, filter_size, 2*filter_num, dropout_rate, batch_norm)
    pool_32 = layers.MaxPooling2D(pool_size=(2,2))(conv_64)

    # DownRes 3, Double_Residual_Conv - Pool
    conv_32 = res_conv_block(pool_32, filter_size, 4*filter_num, dropout_rate, batch_norm)
    pool_16 = layers.MaxPooling2D(pool_size=(2,2))(conv_32)

    # DownRes 4, Double_Residual_Conv - Pool
    conv_16 = res_conv_block(pool_16, filter_size, 8*filter_num, dropout_rate, batch_norm)
    pool_8 = layers.MaxPooling2D(pool_size=(2,2))(conv_16)

    # DownRes 5, Conv
    conv_8 = res_conv_block(pool_8, filter_size, 16*filter_num, dropout_rate, batch_norm)

    # Upsampling layers - Exactly same with the Attention U-Net Upsampling
    # UpRes 6, Attention_Gated_Concat - UpSamp - Concat - Double_Residual_Conv
    gating_16 = gating_signal(conv_8, 8*filter_num, batch_norm)
    att_16 = attention_block(conv_16, gating_16, 8*filter_num)
    up_16 = layers.UpSampling2D(size=(up_sample_size, up_sample_size), data_format="channels_last")(conv_8)
    up_16 = layers.concatenate([up_16, att_16], axis=3)
    up_conv_16 = conv_block(up_16, filter_size, 8*filter_num, dropout_rate, batch_norm)

    # UpRes 7, Attention_Gated_Concat - UpSamp - Concat - Double_Residual_Conv
    gating_32 = gating_signal(up_conv_16, 4*filter_num, batch_norm)
    att_32 = attention_block(conv_32, gating_32, 4*filter_num)
    up_32 = layers.UpSampling2D(size=(up_sample_size, up_sample_size), data_format="channels_last")(up_conv_16)
    up_32 = layers.concatenate([up_32, att_32], axis=3)
    up_conv_32 = conv_block(up_32, filter_size, 4*filter_num, dropout_rate, batch_norm)

    # UpRes 8, Attention_Gated_Concat - UpSamp - Concat - Double_Residual_Conv
    gating_64 = gating_signal(up_conv_32, 2*filter_num, batch_norm)
    att_64 = attention_block(conv_64, gating_64, 2*filter_num)
    up_64 = layers.UpSampling2D(size=(up_sample_size, up_sample_size), data_format="channels_last")(up_conv_32)
    up_64 = layers.concatenate([up_64, att_64], axis=3)
    up_conv_64 = conv_block(up_64, filter_size, 2*filter_num, dropout_rate, batch_norm)

    # UpRes 9, Attention_Gated_Concat - UpSamp - Concat - Double_Residual_Conv
    gating_128 = gating_signal(up_conv_64, filter_num, batch_norm)
    att_128 = attention_block(conv_128, gating_128, filter_num)
    up_128 = layers.UpSampling2D(size=(up_sample_size, up_sample_size), data_format="channels_last")(up_conv_64)
    up_128 = layers.concatenate([up_128, att_128], axis=3)
    up_conv_128 = conv_block(up_128, filter_size, filter_num, dropout_rate, batch_norm)

    # 1*1 convolutional layers
    # Conv - BN - Activation
    conv_final = layers.Conv2D(num_classes, kernel_size=(1,1))(up_conv_128)
    conv_final = layers.BatchNormalization(axis=3)(conv_final)
    conv_final = layers.Activation('sigmoid')(conv_final)  #Change to softmax for multichannel

    # Model integration
    model = models.Model(inputs, conv_final, name="AttentionResUNet")
    if verbose > 0:
        model.summary()
    return model