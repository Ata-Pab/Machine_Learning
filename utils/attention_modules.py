# -*- coding: utf-8 -*-
"""attention_modules.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zzjxWzmlw6YhLf_vb2F_8yB3Sdnx9Vjp

Reference:

https://blog.paperspace.com/attention-mechanisms-in-computer-vision-cbam/
https://arxiv.org/abs/1807.06521

The code from the Reference book was written with Pytorch. I wrote the code with TensorFlow v2.
"""

import tensorflow as tf

class Conv2DLayerBN(tf.keras.layers.Conv2D):
    '''
    Conv2DLayerBN
    2D Convolutional layer with batch normalization, activation end and dropout features
    act_end: Activation function for end of the conv + BN -> conv + BN + Act
    batch_norm: Batch normalization
    lrelu_alpha: Leaky ReLU activation function alpha
    dropout_end: Dropout rate (0.0 to 1.0) at the end of conv. block -> conv + BN + Act + Dropout
    '''
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', 
                 activation=None, act_end=None, batch_norm=True, lrelu_alpha=0.3, 
                 dropout_end=0.0, *args, **kwargs):
        # Call the constructor of the base class (tf.keras.layers.Conv2D)
        super(Conv2DLayerBN, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=activation,
            *args,
            **kwargs
        )
        act_funcs = ["relu", "lrelu", "sigmoid"]
        assert((act_end == None) or (act_end in act_funcs))
        assert((dropout_end >= 0.0) and (dropout_end <= 1.0))

        self.batch_norm = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.01) if batch_norm else None
        self.dropout_end = dropout_end

        if act_end == "relu":
            #act_func = tf.nn.relu()
            self.activation = tf.keras.layers.Activation(act_end)
        elif act_end == "lrelu":
            #act_func = tf.nn.leaky_relu(alpha=lrelu_alpha)
            #self.activation = tf.keras.layers.Activation(act_func)
            self.activation = tf.keras.layers.LeakyReLU(alpha=lrelu_alpha)
        elif act_end == "sigmoid":
            #act_func = tf.keras.activations.sigmoid()
            self.activation = tf.keras.layers.Activation(act_end)
        else:
            self.activation = None

        # ..Add additional customization or modifications...

        # __________________________________________________

    def build(self, input_shape):
        # Add any additional setup or customization for the layer's weights here
        # This method is called the first time the layer is used, based on the input_shape.

        # Make sure to call the build method of the base class
        super(Conv2DLayerBN, self).build(input_shape)

    def call(self, inputs):
        # You can access the weights using self.weights and perform computations using TensorFlow operations.
        # Example:
        # output = tf.nn.conv2d(inputs, self.kernel, strides=self.strides, padding=self.padding)
        # if self.activation is not None:
        #     output = self.activation(output)
        # return output

        # ...Add forward pass activations here...

        # ______________________________________
        x = super(Conv2DLayerBN, self).call(inputs)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout_end > 0.0:
            x = tf.keras.layers.Dropout(self.dropout_end)(x)
        return x

    def get_config(self):
        # ...Add additional configuration parameters for serialization...

        # _______________________________________________________________
        # This method is used to save the configuration of the layer when the model is saved.
        config = super(Conv2DLayerBN, self).get_config()
        # Add your custom parameters to config dictionary
        return config

class Conv2DLayerRes(tf.keras.layers.Layer):
    '''
    Conv2DLayerRes
    2D Residual Convolutional layer with batch normalization, activation end 
    and dropout features.
    There are two options for Residual convolutional layer.
    Either put activation function before the addition with shortcut
    or after the addition (which would be as proposed in the original resNet).

    1. conv + BN + Act + conv + BN + Act + shortcut + BN + shortcut + BN
    2. conv + BN + Act + conv + BN + shortcut + BN + shortcut + BN + Act

    ref: https://arxiv.org/ftp/arxiv/papers/1802/1802.06955.pdf

    act_end: Activation function for end of the conv + BN -> conv + BN + Act
    batch_norm: Batch normalization
    lrelu_alpha: Leaky ReLU activation function alpha
    dropout_end: Dropout rate (0.0 to 1.0) at the end of conv. block -> conv + BN + Act + Dropout
    '''
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', 
                 activation=None, act_end=None, batch_norm=True, lrelu_alpha=0.3, 
                 dropout_end=0.0, *args, **kwargs):
        # Call the constructor of the base class (Conv2DLayerBN)
        super(Conv2DLayerRes, self).__init__()

        act_funcs = ["relu", "lrelu", "sigmoid"]
        assert((act_end == None) or (act_end in act_funcs))
        assert((dropout_end >= 0.0) and (dropout_end <= 1.0))

        # 1. Conv2D: conv + BN + Activation
        self.firstConv2DLayer = Conv2DLayerBN(filters=filters, kernel_size=kernel_size,
                                                strides=strides, padding=padding,
                                                activation=activation, act_end=act_end, 
                                                batch_norm=batch_norm, lrelu_alpha=lrelu_alpha,
                                                dropout_end=0.0)
        
        # 2. Conv2D: conv + BN + Dropout
        self.secondConv2DLayer = Conv2DLayerBN(filters=filters, kernel_size=kernel_size,
                                                #strides=strides, padding=padding,
                                                strides=1, padding="same",
                                                activation=activation, act_end=None, 
                                                batch_norm=batch_norm, dropout_end=dropout_end)
        
        # 3. ShortCutConv2D: conv + BN (kernel_size= (1,1))
        self.shortcutConv2DLayer = Conv2DLayerBN(filters=filters, kernel_size=1,
                                                strides=strides, padding=padding,
                                                activation=None, act_end=None, 
                                                batch_norm=batch_norm, dropout_end=dropout_end)

    def call(self, inputs):
        # ...Add forward pass activations here...

        # ______________________________________
        conv_res = self.firstConv2DLayer(inputs)
        conv_res = self.secondConv2DLayer(conv_res)
        shortcut = self.shortcutConv2DLayer(inputs)
        residual = tf.keras.layers.Add()([shortcut, conv_res])
        return tf.keras.layers.Activation('relu')(residual)

class Conv2DLayerResCBAM(tf.keras.layers.Layer):
    '''
    Conv2DLayerResCBAM (ResBlock + CBAM)
    2D Residual Convolutional layer with batch normalization, activation end 
    and dropout features.
    There are two options for Residual convolutional layer.
    Either put activation function before the addition with shortcut
    or after the addition (which would be as proposed in the original resNet).

    1. conv + BN + Act_end + Dropout
    2. Convolutional Block Attention Module (CBAM) - Channel Attention + Spatial Attention

    conv + BN + Act_end + Dropout + CBAM + Add

    ref: https://arxiv.org/pdf/1807.06521.pdf

    act_end: Activation function for end of the conv + BN -> conv + BN + Act
    batch_norm: Batch normalization
    lrelu_alpha: Leaky ReLU activation function alpha
    dropout_end: Dropout rate (0.0 to 1.0) at the end of conv. block -> conv + BN + Act + Dropout
    '''
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', 
                 activation=None, act_end=None, batch_norm=True, lrelu_alpha=0.3, 
                 dropout_end=0.0, reduction_ratio=16, *args, **kwargs):
        # Call the constructor of the base class (Conv2DLayerBN)
        super(Conv2DLayerResCBAM, self).__init__()

        act_funcs = ["relu", "lrelu", "sigmoid"]
        assert((act_end == None) or (act_end in act_funcs))
        assert((dropout_end >= 0.0) and (dropout_end <= 1.0))

        # 1. Conv2D: conv + BN + Activation
        self.firstConv2DLayer = Conv2DLayerBN(filters=filters, kernel_size=kernel_size,
                                                strides=strides, padding=padding,
                                                activation=activation, act_end=act_end, 
                                                batch_norm=batch_norm, lrelu_alpha=lrelu_alpha,
                                                dropout_end=0.0)
        
        # 2. Convolutional Block Attention Module
        self.attentionLayer = CBAM(gate_channels=filters, reduction_ratio=reduction_ratio)

        # 3. ShortCutConv2D: conv + BN (kernel_size= (1,1))
        self.shortcutConv2DLayer = Conv2DLayerBN(filters=filters, kernel_size=1,
                                                strides=strides, padding=padding,
                                                activation=None, act_end=None, 
                                                batch_norm=batch_norm, dropout_end=dropout_end)
        
    def call(self, inputs):
        # ...Add forward pass activations here...

        # ______________________________________
        conv_res = self.firstConv2DLayer(inputs)
        conv_res = self.attentionLayer(conv_res)
        shortcut = self.shortcutConv2DLayer(inputs)
        residual = tf.keras.layers.Add()([shortcut, conv_res])
        return tf.keras.layers.Activation('relu')(residual)

class ChannelPool(tf.keras.layers.Layer):
    '''
    ChannelPool
    Calculate Max and Avg. pooling for given tensor
    Concatenate them to produce a tensor for being input of MLP
    In a 4D image tensor, the dimensions typically represent the 
    (batch size, height, width, channels). The axis=3 corresponds 
    to the channels or depth dimension (Channel-wise operation).
    '''
    def call(self, inputs):
        # axis=3 refers to the channels dimension. "keepdims=True" 
        # argument ensures that the resulting tensor retains the 
        # dimensions along axis 3, even if the size is reduced to 1. 
        # The output will have the shape of the original tensor with 
        # the channels dimension reduced to size 1 along that axis.
        max_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=3, keepdims=True))(inputs)
        mean_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))(inputs)
        
        #max_pool = tf.expand_dims(tf.reduce_max(inputs, axis=1), axis=1)
        #mean_pool = tf.expand_dims(tf.reduce_mean(inputs, axis=1), axis=1)
        #return tf.concat([max_pool, mean_pool], axis=1)

        # tf.keras.layers.Concatenate: takes a list of tensors as input and concatenates
        # them along the specified axis to produce a single tensor.
        return tf.keras.layers.Concatenate(axis=3)([mean_pool, max_pool])


class SpatialGate(tf.keras.layers.Layer):
    '''
    SpatialGate - SAM (Spatial Attention Module)
    Takes input tensors that are the output of a convolutional layer.
    Applies Max. and Avg. pooling operations to compress tensors
    Then applies a Conv. operation that takes a tensor with input shape (2 x H x W)
    and gives a tensor with shape (1 x H x W)
    Last, Sigmoid function is employed to give probabilistic results for a given
    tensor

    Reference: https://arxiv.org/abs/1807.06521
    '''
    def __init__(self, kernel_size=7, stride=1):
        super(SpatialGate, self).__init__()

        self.padding='valid' if ((kernel_size-1) // 2) == 0 else 'same'

        self.channelPooling = ChannelPool()
        self.spatialConv2D = Conv2DLayerBN(filters=1, kernel_size=kernel_size, strides=stride,
                         padding=self.padding, act_end='sigmoid', batch_norm=True)
    
    def call(self, inputs):
        x_pool = self.channelPooling(inputs)
        attention = self.spatialConv2D(x_pool)
        return tf.keras.layers.Multiply()([inputs, attention])

#class Flatten(tf.keras.layers.Layer):
#    def call(self, x):
#        return tf.reshape(x, (tf.shape(x)[0], -1))

class ChannelGate(tf.keras.layers.Layer):
    '''
    ChannelGate - CAM (Channel Attention Module)
    Takes gate channels as input tensors
    Applies pooling process to the given tensors seperately and puts them into MLP
    block
    Applies element-wise summation for results of the MLP layers
    Last, Sigmoid function is employed to give probabilistic results for a given
    tensors
    Pool types:
    'avg': Avg. pooling
    'max': Max. pooling
    'lp': Power Avg. pooling
    'lse': Logarithmic Summed Exponential (LSE) pooling

    Reference: https://arxiv.org/abs/1807.06521
    '''
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        # Check valid pool types
        valid_pool_types = ['avg', 'max', 'lp', 'lse']
        for pool_type in pool_types:
            assert(pool_type in valid_pool_types)
        self.pool_types = pool_types

        self.gate_channels = gate_channels
        self.reduction_ratio = reduction_ratio

        # MLP (Multi Layer Perceptron)
        self.mlp = tf.keras.Sequential([
            #Flatten(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(self.gate_channels // self.reduction_ratio),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(self.gate_channels)
        ])

    def logsumexp_2d(self, tensor, axis):
        '''
        It is often applied to avoid numerical instability when 
        dealing with exponentials of large numbers. 
        tensor: Input tf.tensor
        axis: axis identifier for global max and sum operations
        '''
        tensor_flatten = tf.reshape(tensor, (tf.shape(tensor)[0], tf.shape(tensor)[1], -1))
        s = tf.reduce_max(tensor_flatten, axis=axis, keepdims=True)
        outputs = s + tf.math.log(tf.reduce_sum(tf.math.exp(tensor_flatten - s), axis=axis, keepdims=True))
        return outputs

    def call(self, x):
        #channel_att_sum = None
        pooling_list = []
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                #avg_pool = tf.reduce_mean(x, axis=[2, 3], keepdims=True)
                # 4D tensor representing images, the axes [2, 3] correspond 
                # to the spatial dimensions (height and width)
                avg_pool = tf.keras.layers.GlobalAveragePooling2D()(x)
                #channel_att_raw = self.mlp(avg_pool)
                avg_pool = self.mlp(avg_pool)
                pooling_list.append(avg_pool)
            elif pool_type == 'max':
                max_pool = tf.keras.layers.GlobalMaxPooling2D()(x)
                #max_pool = tf.reduce_max(x, axis=[2, 3], keepdims=True)
                max_pool = tf.keras.layers.Reshape((1,1, self.gate_channels))(max_pool)
                #channel_att_raw = self.mlp(max_pool)
                max_pool = self.mlp(max_pool)
                pooling_list.append(max_pool)
            elif pool_type == 'lp':
                lp_pool = tf.norm(x, ord=2, axis=[2, 3], keepdims=True)
                #channel_att_raw = self.mlp(lp_pool)
                lp_pool = self.mlp(lp_pool)
                pooling_list.append(lp_pool)
            elif pool_type == 'lse':
                lse_pool = self.logsumexp_2d(x, axis=2)
                #channel_att_raw = self.mlp(lse_pool)
                lse_pool = self.mlp(lse_pool)
                pooling_list.append(lse_pool)

            #if channel_att_sum is None:
            #    channel_att_sum = channel_att_raw
            #else:
            #    channel_att_sum = channel_att_sum + channel_att_raw
        attention = tf.keras.layers.Add()(pooling_list)

        #scale = tf.math.sigmoid(channel_att_sum)
        attention = tf.keras.layers.Activation('sigmoid')(attention)
        #return x * scale
        return tf.keras.layers.Multiply()([x, attention])

class CBAM(tf.keras.layers.Layer):
    '''
    CBAM (Convolutional Block Attention Module)
    CAM + SAM blocks
    Applies Channel Attention Module for given tensors
    sam_block: sets whether SAM (Spatial Attention Block) will be used
    reduction_ratio: The higher the reduction ratio, the fewer the number of
    neurons in the bottleneck

    Reference: https://arxiv.org/abs/1807.06521
    '''
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], sam_block=True):
        super(CBAM, self).__init__()
        # Check valid pool types
        valid_pool_types = ['avg', 'max', 'lp', 'lse']
        for pool_type in pool_types:
            assert(pool_type in valid_pool_types)

        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.sam_block = sam_block

        if sam_block:
            self.SpatialGate = SpatialGate()

    def call(self, inputs):
        x_out = self.ChannelGate(inputs)
        if self.sam_block:
            x_out = self.SpatialGate(x_out)
        return x_out
    
class AttentionLayerUNet(tf.keras.layers.Layer):
    '''
    AttentionLayerUNet
    Provides Attention layer used for Attention U-Net
    conv_input: previous convolutional layer's output
    gate_input: output of the second conv. layer before the last
    filters= number of kernels to be applied to conv outputs
    '''
    def __init__(self, filters):
        super(AttentionLayerUNet, self).__init__()
        self.filers = filters

        # Getting x to the same shape as gating signal (128 to 64)
        self.thetaConv2D = tf.keras.layers.Conv2D(filters=filters, kernel_size=(2,2), 
                                              strides=(2,2), padding='same')
        
        # Getting the gating signal to the same number of filters as the inter_shape
        self.phiGateConv2D = tf.keras.layers.Conv2D(filters=filters, kernel_size=(1,1), 
                                       strides=(1,1), padding='same')
        
        self.psiConv2D = tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), padding='same')
    
    def repeat_layer(self, tensor, rep):
        # lambda function to repeat the elements of a tensor along an axis by a
        # factor of rep. If tensor has shape (None, 256, 256, 3), lambda will return
        # a tensor of shape (None, 256,256,6), if specified axis=3 and rep=2.
        # K.repeat_layerents(x, repnum, axis=3),
        return tf.keras.layers.Lambda(lambda x, repnum: tf.repeat(x, repnum, axis=3),
                              arguments={'repnum': rep})(tensor)
    
    def call(self, conv_input, gate_input):
        shape_conv_input = tf.keras.backend.int_shape(conv_input)
        shape_gate_input = tf.keras.backend.int_shape(gate_input)

        theta_x = self.thetaConv2D(conv_input)
        phi_g = self.phiGateConv2D(gate_input)

        shape_theta_x = tf.keras.backend.int_shape(theta_x)
        stride = (shape_theta_x[1] // shape_gate_input[1], shape_theta_x[2] // shape_gate_input[2])

        upsample_g = tf.keras.layers.Conv2DTranspose(self.filers, (3, 3),
                                 strides=stride, padding='same')(phi_g)

        concat_xg = tf.keras.layers.add([upsample_g, theta_x])
        act_xg = tf.keras.layers.Activation('relu')(concat_xg)
        
        psi = self.psiConv2D(act_xg)
        sigmoid_xg = tf.keras.layers.Activation('sigmoid')(psi)

        shape_sigmoid = tf.keras.backend.int_shape(sigmoid_xg)
        upsampling_size = (shape_conv_input[1] // shape_sigmoid[1], shape_conv_input[2] // shape_sigmoid[2])
        
        upsample_psi = tf.keras.layers.UpSampling2D(size=upsampling_size)(sigmoid_xg)  # 32

        upsample_psi = self.repeat_layer(upsample_psi, shape_conv_input[3])

        y = tf.keras.layers.Multiply()([upsample_psi, conv_input])

        return Conv2DLayerBN(filters=shape_conv_input[3], kernel_size=1, padding='same', batch_norm=True)(y)

class Conv2DTransposeBN(tf.keras.layers.Conv2DTranspose):
    '''
    Conv2DTransposeBN
    2D Transposed Convolutional layer with batch normalization, activation end and dropout features
    act_end: Activation function for end of the conv + BN -> conv + BN + Act
    batch_norm: Batch normalization
    lrelu_alpha: Leaky ReLU activation function alpha
    dropout_end: Dropout rate (0.0 to 1.0) at the end of conv. block -> conv + BN + Act + Dropout
    '''
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', 
                 activation=None, act_end=None, batch_norm=True, lrelu_alpha=0.3, 
                 dropout_end=0.0, *args, **kwargs):
        # Call the constructor of the base class (tf.keras.layers.Conv2D)
        super(Conv2DTransposeBN, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=activation,
            *args,
            **kwargs
        )
        act_funcs = ["relu", "lrelu", "sigmoid"]
        assert((act_end == None) or (act_end in act_funcs))
        assert((dropout_end >= 0.0) and (dropout_end <= 1.0))

        self.batch_norm = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.01) if batch_norm else None
        self.dropout_end = dropout_end

        if act_end == "relu":
            #act_func = tf.nn.relu()
            self.activation = tf.keras.layers.Activation(act_end)
        elif act_end == "lrelu":
            #act_func = tf.nn.leaky_relu(alpha=lrelu_alpha)
            #self.activation = tf.keras.layers.Activation(act_func)
            self.activation = tf.keras.layers.LeakyReLU(alpha=lrelu_alpha)
        elif act_end == "sigmoid":
            #act_func = tf.keras.activations.sigmoid()
            self.activation = tf.keras.layers.Activation(act_end)
        else:
            self.activation = None

        # ..Add additional customization or modifications...

        # __________________________________________________

    def build(self, input_shape):
        # Add any additional setup or customization for the layer's weights here
        # This method is called the first time the layer is used, based on the input_shape.

        # Make sure to call the build method of the base class
        super(Conv2DTransposeBN, self).build(input_shape)

    def call(self, inputs):
        # You can access the weights using self.weights and perform computations using TensorFlow operations.
        # Example:
        # output = tf.nn.conv2d(inputs, self.kernel, strides=self.strides, padding=self.padding)
        # if self.activation is not None:
        #     output = self.activation(output)
        # return output

        # ...Add forward pass activations here...

        # ______________________________________
        x = super(Conv2DTransposeBN, self).call(inputs)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.dropout_end > 0.0:
            x = tf.keras.layers.Dropout(self.dropout_end)(x)
        return x

    def get_config(self):
        # ...Add additional configuration parameters for serialization...

        # _______________________________________________________________
        # This method is used to save the configuration of the layer when the model is saved.
        config = super(Conv2DTransposeBN, self).get_config()
        # Add your custom parameters to config dictionary
        return config