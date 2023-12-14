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
    act_end: Activation function for end of the conv + BN -> conv + BN + Act
    batch_norm: Batch normalization
    lrelu_alpha: Leaky ReLU activation function alpha
    '''
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', 
                 activation=None,act_end=None, batch_norm=True, lrelu_alpha=0.3, 
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
            act_func = tf.nn.leaky_relu(alpha=lrelu_alpha)
            self.activation = tf.keras.layers.Activation(act_func)
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


class CustomConv2DLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, act=None, batch_norm=True, bias=False, lrelu_alpha=0.3):
        super(CustomConv2DLayer, self).__init__()
        assert ((act == None) or (act == "relu") or (act == "lrelu"))

        self.conv = tf.keras.layers.Conv2D(filters=filters,
                                           kernel_size=kernel_size,
                                           strides=stride,
                                           padding='valid' if padding == 0 else 'same',
                                           dilation_rate=dilation,
                                           groups=groups,
                                           use_bias=bias)

        self.batch_norm = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.01) if batch_norm else None

        if act == "relu":
            self.activation = tf.keras.layers.ReLU()
        elif act == "lrelu":
            self.activation = tf.keras.layers.LeakyReLU(alpha=lrelu_alpha)
        else:
            self.activation = None

    def call(self, x):
        x = self.conv(x)
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

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
        return tf.keras.layers.multiply([inputs, attention])

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