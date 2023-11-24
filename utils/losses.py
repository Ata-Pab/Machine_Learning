import tensorflow as tf
import numpy as np
from keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model

# Show pre-trained model perceptual layers for content/perceptual loss
def show_perceptual_layers_info(model, layers):
    '''
    Prints layers of the pre-trained model that will extract features
    layers: layer number array
    '''
    print(f'Pre-trained {model.name.upper()} Model layers that will be used for feature extraction (for Perceptual Loss): \n')
    for ix, layer_id in enumerate(layers):
        print(f"{layers[ix]}.Layer: {model.layers[layer_id].name}, Trainable: {model.layers[layer_id].trainable}")

    print("\nLast layer of the pre-trained model: ", model.layers[-1].name)

# Init content loss with default VGG16 pre-trained model
def init_perceptual_loss(perp_loss_model='VGG16', perp_layers=None, verbose=0):
  # Set perceptual model
  if perp_loss_model == 'VGG19':
    perp_model = tf.keras.applications.VGG19(input_shape=(224,224,3))
  elif perp_loss_model == 'RESNET50V2':
    perp_model = tf.keras.applications.ResNet50V2(include_top=False,
                    weights='imagenet', input_shape=(224, 224, 3))
  else:
    perp_model = tf.keras.applications.VGG16(input_shape=(224,224,3))

  # perp_model.trainable = False  # No need to set trainable parameter as False

  if verbose > 0:
    show_perceptual_layers_info(perp_model, perp_layers)

  # Set perceptual loss output layers
  if perp_layers != None:
    modelOutputs = [perp_model.layers[i].output for i in perp_layers]
  else:
    modelOutputs = perp_model.layers[-2].output  # Get last layer of the model before prediction layer

  model = Model(perp_model.inputs, modelOutputs)

  if verbose > 1:
      print("\n")
      model.summary()

  return model

def dice_coef(y_true, y_pred):
    y_true_f = layers.Flatten()(y_true)
    y_pred_f = layers.Flatten()(y_pred)
    intersection = tf.math.reduce_sum(y_true_f * y_pred_f)  # or tf.math.reduce_sum(x, axis=[0,1])
    #intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)

def jacard_coef(y_true, y_pred):
    y_true_f = layers.Flatten()(y_true)
    y_pred_f = layers.Flatten()(y_pred)
    intersection = tf.math.reduce_sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (tf.math.reduce_sum(y_true_f) + tf.math.reduce_sum(y_pred_f) - intersection + 1.0)

def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def mse_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    # mse = K.mean(K.square(y_true - y_pred), axis = [1,2,3])
    return mse

def euclid_dis(vectors):
  ''' Calculates Euclidean Distance between two embeddings (vectors)
      Euclidean Distance: sqrt(sum(square(y - x)))

  Arguments:
      vects: List containing two tensors of same length.
  Returns:
      Tensor containing euclidean distance between vectors.
  '''
  x, y = vectors
  sum_square = tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True)
  return tf.sqrt(tf.maximum(sum_square, tf.keras.backend.epsilon()))

def mae_loss(y_true, y_pred):
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    #mae = K.mean(K.abs(y_true - y_pred), axis = [1,2,3])
    return mae

def kl_loss(mean, log_var):
    # kl_loss_result =  -0.5 * K.sum(1 + log_var - K.square(mean) - K.exp(log_var), axis = 1)
    # kl_loss_result =  -0.5 * tf.reduce_sum(1 + log_var - tf.square(mean) - tf.exp(log_var))
    kl_loss_result =  -0.5 * tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var))
    return kl_loss_result

def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def adversarial_loss(y_true, y_pred):
    return -tf.reduce_mean(tf.math.log(y_pred + 1e-8))

def contrastive_loss(y_true, y_pred, margin=1):
    '''Calculates the contrastive loss.

    Arguments:
        y_true: List of labels, each label is of type float32.
        y_pred: List of predictions of same length as of y_true,
                each label is of type float32.
        margin: Integer, defines the baseline for distance for which pairs
                should be classified as dissimilar. - (default is 1).

    Returns:
        A tensor containing contrastive loss as floating point value.
    '''
    y_true = tf.cast(y_true, y_pred.dtype)

    # Loss function Le Cunn
    square_pred = tf.square(y_pred)
    margin_square = tf.square(tf.maximum(margin - (y_pred), 0))
    return tf.reduce_mean((1 - y_true) * margin_square + (y_true) * square_pred)

'''
...Deprecated
def triplet_loss(self, y_true, y_pred):
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    #pos_dist = K.sum(K.square(anchor-positive), axis=1)
    #neg_dist = K.sum(K.square(anchor-negative), axis=1)
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)
    calc_loss = pos_dist - neg_dist + self.margin  # calculated loss
    # The default value of epsilon is 1e-7. This value is used to prevent division by zero 
    # in various operations such as in the computation of gradients during training, or in 
    # the calculation of certain functions like logarithms or exponentials.
    # return K.maximum(calc_loss, 0.0) 
    #return K.maximum(calc_loss, K.epsilon())
    return tf.maximum(calc_loss, tf.keras.backend.epsilon())
'''

def triplet_loss(self, anchor_features, positive_features, negative_features):
    '''
    Triplet Loss
    L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    In Distance Layer: ap_distance = ‖f(A) - f(P)‖², an_distance = ‖f(A) - f(N)‖²
    '''
    ap_distance = tf.reduce_sum(tf.square(anchor_features - positive_features), -1)
    an_distance = tf.reduce_sum(tf.square(anchor_features - negative_features), -1)

    # Computing the Triplet Loss by subtracting both distances and
    # making sure we don't get a negative value.
    loss = ap_distance - an_distance
    loss = tf.maximum(loss + self._margin, tf.keras.backend.epsilon())
    return loss

def binary_cross_entropy_loss(y_true, y_pred, from_logits=False):
    '''
    Binary Cross-Entropy Loss= -(1/N ∑(i to N)[yi⋅log(pi) + (1-yi)⋅log(1-pi)])
    '''
    eps = tf.keras.backend.epsilon()
    # Ensure the predicted values are within the range (0, 1) using a sigmoid activation
    if from_logits:
        y_pred = tf.math.sigmoid(y_pred)

    # Compute binary cross-entropy loss
    loss = -tf.reduce_mean(y_true * tf.math.log(y_pred + eps) + (1 - y_true) * tf.math.log(1 - y_pred + eps))
    return loss

def focal_loss(y_true, y_pred, alpha=1.0, gamma=2.0):    
    '''
    Focal Loss
    FL(p_t) = - (1 - p_t)^gamma * log(p_t)

    p_t is the predicted probability for the true class.
    gamma is a hyperparameter to control the focusing effect.

    The p_t calculation is a way to weigh the predicted probability 
    based on the true class label.
    '''
    epsilon = 1e-7
    y_pred = tf.cast(y_pred, tf.float32)
    # Clip predicted values to avoid log(0) errors
    y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)

    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    focal_loss = -tf.pow(1 - p_t, gamma) * tf.math.log(p_t)

    return tf.reduce_mean(alpha * focal_loss)

# Mad Score
# # Example usage - Calculate MSE (mean squared error) between ground truth and reconstructed data before employing mad_score
# THRESHOLD = 3
# z_scores = mad_score(mse)
# outliers = z_scores > THRESHOLD
def mad_score(points):
    m = np.median(points)
    ad = np.abs(points - m)
    mad = np.median(ad)
    return 0.6745 * ad / mad

# Reference
# https://github.com/keras-team
# 
# https://github.com/mrdbourke/tensorflow-deep-learning/tree/main
# 
# https://www.kaggle.com/code/robinteuwens/anomaly-detection-with-auto-encoders#Unsupervised-Learning-with-Auto-Encoders
# 
# https://pyimagesearch.com/2020/03/02/anomaly-detection-with-keras-tensorflow-and-deep-learning/
# 
# https://www.kaggle.com/code/matheusfacure/semi-supervised-anomaly-detection-survey
# 
# https://medium.com/analytics-vidhya/image-anomaly-detection-using-autoencoders-ae937c7fd2d1
# 
# https://lilianweng.github.io/posts/2018-08-12-vae/

