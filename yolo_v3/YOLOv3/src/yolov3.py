import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, Input, ZeroPadding2D, LeakyReLU, UpSampling2D

def parse_cfg(cfgfile):
    ''' The code below is a function called parse_cfg() with a parameter named 
    cfgfile used to parse the YOLOv3 configuration fileyolov3.cfg. '''
    with open(cfgfile, 'r') as file:
        lines = [line.rstrip('\n') for line in file if line != '\n' and line[0] != '#']
    holder = {}
    blocks = []
    # Read each line one by one from cfg file
    for line in lines:
        if line[0] == '[':
            line = 'type=' + line[1:-1].rstrip()
            if len(holder) != 0:
                blocks.append(holder)
                holder = {}
        key, value = line.split("=")
        holder[key.rstrip()] = value.lstrip()
    blocks.append(holder)
    return blocks


def YOLOv3Net(cfgfile, model_size, num_classes):

    ''' YOLOv3Net 
    YOLOv3 has 5 layers types in general, they are: “convolutional layer”, 
    “upsample layer”, “route layer”, “shortcut layer”, and “yolo layer”.'''
    blocks = parse_cfg(cfgfile)   # Store all the return attributes in a variable "blocks"
    outputs = {}
    output_filters = []
    filters = []
    out_pred = []
    scale = 0
    inputs = input_image = Input(shape=model_size)
    inputs = inputs / 255.0   # Normalize it to the range of 0-1

    '''The following code performs an iteration over the list blocks. 
    For every iteration, we check the type of the block which corresponds to 
    the type of layer. YOLOv3 has 5 layers types in general, they are: “convolutional layer”, 
    “upsample layer”, “route layer”, “shortcut layer”, and “yolo layer”.'''
    for i, block in enumerate(blocks[1:]):

        # If it is a convolutional layer
        if (block["type"] == "convolutional"):
            ''' Convolutional Layer
            In YOLOv3, there are 2 convolutional layer types, i.e with and without 
            batch normalization layer. The convolutional layer followed by a batch 
            normalization layer uses a leaky ReLU activation layer, otherwise, it 
            uses the linear activation. So, we must handle them for every single 
            iteration we perform.'''
            activation = block["activation"]
            filters = int(block["filters"])
            kernel_size = int(block["size"])
            strides = int(block["stride"])

            # If strideis greater than 1, then downsampling is performed, so we need
            # to adjust the padding.
            if strides > 1:
                inputs = ZeroPadding2D(((1, 0), (1, 0)))(inputs)

            inputs = Conv2D(filters,
                            kernel_size,
                            strides=strides,
                            # verify whether the strides greater than 1, if 
                            # it is true, then downsampling is performed, so we 
                            # need to adjust the padding.
                            padding='valid' if strides > 1 else 'same',
                            name='conv_' + str(i),
                            use_bias = False if ("batch_normalize" in block) else True)(inputs)
            
            # if we find batch_normalizein a block, then add layers 
            # BatchNormalization and LeakyReLU, otherwise, do nothing.
            if "batch_normalize" in block:
                inputs = BatchNormalization(name='bnorm_' + str(i))(inputs)

            if activation == "leaky":
                inputs = LeakyReLU(alpha=0.1, name='leaky_' + str(i))(inputs)

        elif (block["type"] == "upsample"):
            ''' Upsample Layer
            The upsample layer performs upsampling of the previous feature map 
            by a factor of stride. To do this, YOLOv3 uses bilinear upsampling 
            method. So, if we find upsample block, retrieve the stride value and 
            add a layer UpSampling2D by specifying the stride value.
            YOLOv3 uses bilinear upsampling method.'''
            stride = int(block["stride"])
            inputs = UpSampling2D(stride)(inputs)
        
        # If it is a route layer
        elif (block["type"] == "route"):
            ''' Route Layer
            The attribute layers holds a value of -4 which means that if we are in 
            this route block, we need to backward 4 layers and then output the feature 
            map from that layer. However, for the case of the route block whose attribute 
            layers has 2 values, layers contains -1 and 61, we need to concatenate the 
            feature map from a previous layer (-1) and the feature map from layer 61.
            '''
            block["layers"] = block["layers"].split(',')
            start = int(block["layers"][0])
        
            if len(block["layers"]) > 1:
                end = int(block["layers"][1]) - i
                filters = output_filters[i + start] + output_filters[end]  # Index negatif :end - index
                inputs = tf.concat([outputs[i + start], outputs[i + end]], axis=-1)
            else:
                filters = output_filters[i + start]
                inputs = outputs[i + start]

        # Shortcut Layer
        elif block["type"] == "shortcut":
            ''' Shortcut Layer
            In this layer block is to backward 3 layers (-3) as indicated in from value, 
            then take the feature map from that layer, and add it with the feature map from the 
            previous layer.
            '''
            from_ = int(block["from"])
            inputs = outputs[i - 1] + outputs[i + from_]
        
         # Yolo detection layer
        elif block["type"] == "yolo":
            ''' Yolo detection layer
            We perform our detection and do some refining to the bounding boxes.
            '''
            # Get mask attribute
            mask = block["mask"].split(",")
            mask = [int(x) for x in mask]
            # Get anchor attribute
            anchors = block["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            n_anchors = len(anchors)

            out_shape = inputs.get_shape().as_list()

            # YOLOv3 output form: [None, B * grid_size * grid_size, 5 + C]
            # B is the number of anchors
            # C is the number of classes
            inputs = tf.reshape(inputs, [-1, n_anchors * out_shape[1] * out_shape[2], 5 + num_classes])

            # Accessing all boxes attributes
            box_centers = inputs[:, :, 0:2]
            box_shapes = inputs[:, :, 2:4]
            confidence = inputs[:, :, 4:5]
            classes = inputs[:, :, 5:num_classes + 5]

            # Refine Bounding Boxes
            # YOLOv3 network outputs the bounding boxes prediction, we need to refine them 
            # in order to the have the right positions and shapes. Use the sigmoid function 
            # to convert box_centers, confidence, and classes values into range of 0 – 1.
            box_centers = tf.sigmoid(box_centers)
            confidence = tf.sigmoid(confidence)
            classes = tf.sigmoid(classes)

            # Convert box shapes
            '''
            >> a = tf.constant([[1,2,3],[4,5,6]], tf.int32)
            >> c = tf.constant([2,1], tf.int32)
            >> tf.tile(a, c)

            Output: 
            <tf.Tensor: shape=(4,3), dtype=int32, numpy=
            array([[1,2,3],
                   [4,5,6],
                   [1,2,3],
                   [4,5,6]], dtype=int32)>
            '''
            anchors = tf.tile(anchors, [out_shape[1] * out_shape[2], 1])
            box_shapes = tf.exp(box_shapes) * tf.cast(anchors, dtype=tf.float32)   # tf.math.exp() ?

            x = tf.range(out_shape[1], dtype=tf.float32)
            y = tf.range(out_shape[2], dtype=tf.float32)

            # Use a meshgrid to convert the relative positions of the center boxes 
            # into the real positions
            cx, cy = tf.meshgrid(x, y)
            cx = tf.reshape(cx, (-1, 1))
            cy = tf.reshape(cy, (-1, 1))
            cxy = tf.concat([cx, cy], axis=-1)
            cxy = tf.tile(cxy, [1, n_anchors])
            cxy = tf.reshape(cxy, [1, -1, 2])

            strides = (input_image.shape[1] // out_shape[1], input_image.shape[2] // out_shape[2])
            box_centers = (box_centers + cxy) * strides

            # Then, concatenate them all together.
            prediction = tf.concat([box_centers, box_shapes, confidence, classes], axis=-1)

            if scale:
                out_pred = tf.concat([out_pred, prediction], axis=1)
            else:
                out_pred = prediction
                scale = 1

        outputs[i] = inputs
        output_filters.append(filters)

    model = Model(input_image, out_pred)
    model.summary()
    return model