# -*- coding: utf-8 -*-
"""attention_modules.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1zzjxWzmlw6YhLf_vb2F_8yB3Sdnx9Vjp

Reference:

https://blog.paperspace.com/attention-mechanisms-in-computer-vision-cbam/

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
                 *args, **kwargs):
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

        self.batch_norm = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.01) if batch_norm else None

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
    '''
    def call(self, x):
        max_pool = tf.expand_dims(tf.reduce_max(x, axis=1), axis=1)
        mean_pool = tf.expand_dims(tf.reduce_mean(x, axis=1), axis=1)
        return tf.concat([max_pool, mean_pool], axis=1)

class SpatialGate(tf.keras.layers.Layer):
    '''
    SpatialGate - SAM (Spatial Attention Module)
    Takes input tensors that are the output of a convolutional layer.
    Applies Max. and Avg. pooling operations to compress tensors
    Then applies a Conv. operation that takes a tensor with input shape (2 x H x W)
    and gives a tensor with shape (1 x H x W)
    Last, Sigmoid function is employed to give probabilistic results for a given
    tensor
    '''
    def __init__(self, kernel_size=7, stride=1):
        super(SpatialGate, self).__init__()
        self.compress = ChannelPool()
        self.spatial = CustomConv2DLayer(filters=1, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1) // 2, act=None)

    def call(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = tf.math.sigmoid(x_out)
        return x * scale

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
    '''
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        # Check valid pool types
        self.valid_pool_types = ['avg', 'max', 'lp', 'lse']
        for pool_type in pool_types:
            assert(pool_type in self.valid_pool_types)

        self.gate_channels = gate_channels
        # MLP (Multi Layer Perceptron)
        self.mlp = tf.keras.Sequential([
            #Flatten(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(gate_channels // reduction_ratio),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(gate_channels)
        ])
        self.pool_types = pool_types

    def logsumexp_2d(self, tensor, axis):
        tensor_flatten = tf.reshape(tensor, (tf.shape(tensor)[0], tf.shape(tensor)[1], -1))
        s = tf.reduce_max(tensor_flatten, axis=axis, keepdims=True)
        outputs = s + tf.math.log(tf.reduce_sum(tf.math.exp(tensor_flatten - s), axis=axis, keepdims=True))
        return outputs

    def call(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = tf.reduce_mean(x, axis=[2, 3], keepdims=True)
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = tf.reduce_max(x, axis=[2, 3], keepdims=True)
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = tf.norm(x, ord=2, axis=[2, 3], keepdims=True)
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                lse_pool = self.logsumexp_2d(x, axis=2)
                channel_att_raw = self.mlp(lse_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = tf.math.sigmoid(channel_att_sum)
        #scale = tf.math.sigmoid(channel_att_sum) * tf.expand_dims(tf.expand_dims(tf.constant(1.0), axis=2), axis=3)
        return x * scale

class CBAM(tf.keras.layers.Layer):
    '''
    CBAM (Convolutional Block Attention Module)
    CAM + SAM blocks
    Applies Channel Attention Module for given tensors
    sam_block: sets whether SAM block will be used
    reduction_ratio: The higher the reduction ratio, the fewer the number of
    neurons in the bottleneck
    '''
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], sam_block=True):
        super(CBAM, self).__init__()
        # Check valid pool types
        self.valid_pool_types = ['avg', 'max', 'lp', 'lse']
        for pool_type in pool_types:
            assert(pool_type in self.valid_pool_types)

        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.sam_block = sam_block

        if sam_block:
            self.SpatialGate = SpatialGate()

    def call(self, x):
        x_out = self.ChannelGate(x)
        if self.sam_block:
            x_out = self.SpatialGate(x_out)
        return x_out