import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.manifold import TSNE
import numpy as np
import random as rnd
import pandas as pd
import itertools
import cv2
import matplotlib.cm as cm
from utils import utils
from google.colab.patches import cv2_imshow
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score # Evaluation metrics
from sklearn.metrics import classification_report  # Precision, recall, f1-score metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # Classifier Confusion Matrix visualization
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve

# Fit window (Large images do not fit in window for Colab)
WINDOW_FIT_LIMIT = 1000
REF_WINDOW_LIMIT = 1000

# Shows image samples from dataset
def show_image_samples_from_batch(dataset, grid=(4,4), figsize=(10, 10)):
    image_batch = next(iter(dataset))

    fig = plt.figure(figsize=figsize)

    for index, image in enumerate(image_batch):  # Get first batch
      plt.subplot(grid[0], grid[1], index + 1)
      plt.imshow(image[:, :, :])
      plt.axis('off')
      if index >= ((grid[0]*grid[1])-1):
        break

# Display pixel wise imag difference (Valid inputs: img paths or img arrays)
def display_pixel_wise_img_diff(img1_dir, img2_dir, method="mae", threshold=None, channel=0, colorbar=True, verbose=0):
    # Load your two images using TensorFlow
    if type(img1_dir) == str:
        image1 = tf.image.decode_image(tf.io.read_file(img1_dir))
        image2 = tf.image.decode_image(tf.io.read_file(img2_dir))
    else:
        image1 = img1_dir
        image2 = img2_dir

    # Ensure both images have the same shape and data type
    image1 = tf.cast(image1, tf.float32)
    image2 = tf.cast(image2, tf.float32)

    # Compute pixel-wise absolute differences
    assert((method == "mae") or (method == "MAE") or (method == "mse") or (method == "MSE"))
    if (method == "mae") or (method == "MAE"):
        diff = tf.abs(image1 - image2)
    else:
        diff = (tf.square(tf.math.pow(image2, 2) - tf.math.pow(image1, 2)))
    if verbose > 0:  print(f"Difference map shape: {diff.shape}")

    # Define a colormap (e.g., 'jet') and normalize the differences
    cmap = plt.get_cmap('jet')
    norm = plt.Normalize(vmin=0, vmax=tf.reduce_max(diff).numpy())
    norm_diff = norm(diff)
    if verbose > 0:
      print(f"Normalized difference map shape: {norm_diff.shape}")
      print(f"Normalized difference map min, avg, max value: ({np.min(norm_diff)}, {np.mean(norm_diff):.5f}, {np.max(norm_diff)})")

    colored_diff = cmap(norm_diff)
    if verbose > 0:  print(f"Colored difference map shape: {colored_diff.shape}")

    color_diff_plot = colored_diff[:, :, :,channel]

    if threshold != None:
      color_diff_plot[color_diff_plot < threshold] = 0.0

    # Display the color-coded difference map
    plt.imshow(color_diff_plot)
    if colorbar:
      plt.colorbar()
    plt.axis('off')
    plt.show()

# TSNE Anomaly Scatter
# Example Usage
# tsne_scatter(features, labels, dimensions=2, save_as='tsne_2d.png')
'''
Visualising clusters with t-SNE

t-Distributed Stochastic Neighbor Embedding (t-SNE) is a dimensionality reduction technique 
used for visualisations of complex datasets. It maps clusters in high-dimensional data to a 
two- or three dimensional plane so we can get an idea of how easy it will be to discriminate 
between classes. It does this by trying to keep the distance between data points in lower 
dimensions proportional to the probability that these data points are neighbours in the higher 
dimensions.
'''
def tsne_anomaly_scatter(features, labels, dimensions=2, rnd_seed=2, save_as='graph.png'):
    if dimensions not in (2, 3):
        raise ValueError('Make sure that your dimension is 2d or 3d')

    # t-SNE dimensionality reduction
    features_embedded = TSNE(n_components=dimensions, random_state=rnd_seed).fit_transform(features)

    # initialising the plot
    fig, ax = plt.subplots(figsize=(8,8))

    # counting dimensions
    if dimensions == 3: ax = fig.add_subplot(111, projection='3d')

    # Scattering map for Anomaly samples
    ax.scatter(
        *zip(*features_embedded[np.where(labels==1)]),
        marker='o',
        color='r',
        s=2,
        alpha=0.7,
        label='Fraud'
    )

    # Scattering map for Normal samples
    ax.scatter(
        *zip(*features_embedded[np.where(labels==0)]),
        marker='o',
        color='g',
        s=2,
        alpha=0.3,
        label='Clean'
    )

    # storing it to be displayed later
    plt.legend(loc='best')
    plt.savefig(save_as)
    plt.show()

# Visualize Triplet Images from the batches
def show_triplet_images(anchor, positive, negative, size=(10,10), row=3):
  '''Visualize triplets from the batches'''
  def plot_img(ax, image):
    ax.imshow(image)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

  fig = plt.figure(figsize=size)

  axs = fig.subplots(row, 3)
  for i in range(row):
    plot_img(axs[i, 0], anchor[i])
    plot_img(axs[i, 1], positive[i])
    plot_img(axs[i, 2], negative[i])

# Show Confusion Matrix - Classification Evaluation Method
def show_confusion_matrix(y_true, y_pred, classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False, fig_name='confusion_matrix'):
  """Makes a labelled confusion matrix comparing predictions and ground truth labels.

  If classes is passed, confusion matrix will be labelled, if not, integer class values
  will be used.

  Args:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(10, 10)).
    text_size: Size of output figure text (default=15).
    norm: normalize values or not (default=False).
    savefig: save confusion matrix to file (default=False).

  Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.

  Example usage:
    make_confusion_matrix(y_true=test_labels, # ground truth test labels
                          y_pred=y_preds, # predicted labels
                          classes=class_names, # array of class label names
                          figsize=(15, 15),
                          text_size=10)
  """
  # Create the confustion matrix
  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
  '''
  >> A = np.array([2,0,1,8])
  >> A.shape
  Output: (4,)

  >> A[np.newaxis, :]
  Output: array([[2,0,1,8]])

  >> A[:, np.newaxis]
  Output: array([[2],
                 [0],
                 [1],
                 [8]]
  )
  '''
  n_classes = cm.shape[0] # find the number of classes we're dealing with

  # Plot the figure and make it pretty
  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
  fig.colorbar(cax)

  # Are there a list of classes?
  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])

  # Label the axes
  ax.set(title="Confusion Matrix",
         xlabel="Predicted label",
         ylabel="True label",
         xticks=np.arange(n_classes), # create enough axis slots for each class
         yticks=np.arange(n_classes),
         xticklabels=labels, # axes will labeled with class names (if they exist) or ints
         yticklabels=labels)

  # Make x-axis labels appear on bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  # Added: Rotate xticks for readability & increase font size (required due to such a large confusion matrix)
  plt.xticks(rotation=70, fontsize=text_size)
  plt.yticks(fontsize=text_size)

  # Set the threshold for different colors
  threshold = (cm.max() + cm.min()) / 2.

  # Plot the text on each cell
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    if norm:
      plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)
    else:
      plt.text(j, i, f"{cm[i, j]}",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)

  # Save the figure to the current working directory
  if savefig:
    fig_name = fig_name + '.png'
    fig.savefig(fig_name)

# Show Confusion Matrix - Classification Evaluation Method (Alternative - 2)
#def show_confusion_matrix(y_test, y_pred, labels, w_h=(12, 7)):
#  '''This method plots Confusion matrix for classification models with given test dataset and prediction result array'''
#  confMatrix = confusion_matrix(y_test, y_pred)
#  dispConfMatrix = ConfusionMatrixDisplay(confMatrix, display_labels=labels)
#  dispConfMatrix.plot()
#  fig = plt.gcf()
#  w, h = w_h
#  fig.set_size_inches(w, h)

# Plot Randomly Image Predictions
def plot_randomly_img_predictions(img_arr, y_test, y_pred, labels, num_item=2, fig_size=[20,10]):
  plt.figure(figsize=fig_size)
  for img in range(num_item):
      ix = rnd.randint(0, len(img_arr)-1)
      display = plt.subplot(1, num_item, img+1)

  plt.imshow(img_arr[ix], cmap="gray")
  act  = "Act: " + str(labels[(int(y_test[ix]))])
  pred = "Pred: " + str(labels[(int(y_pred[ix]))])

  plt.yticks([])
  plt.title(act)
  plt.ylabel(pred)

  display.get_xaxis().set_visible(False)
  #display.get_yaxis().set_visible(False)

  plt.show()

# Show ROC Curve - Calculate AUC score
def show_ROC_score(self, y_test, pos_prob, kind='fp_tp', plot=False, label='Custom Classifier'):
  if kind == 'fp_tp':   # False Positive-True Positive Curve
    auc_score = roc_auc_score(y_test, pos_prob)
    fp_rate, tp_rate, thresholds = roc_curve(y_test, pos_prob)
    plt_x, plt_y, lbl_x, lbl_y = fp_rate, tp_rate, "False Positive Rate (FP)", "True Positive Rate (TP)"
    # Generate a no skill prediction
    noskill_probs = [0 for _ in range(len(y_test))]
    ns_auc_score = roc_auc_score(y_test, noskill_probs)
    noskill_fp_rate, noskill_tp_rate, noskill_thresholds = roc_curve(y_test, noskill_probs)

  elif kind == 'pre_rec':   # Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(y_test, pos_prob)
    auc_score = auc(recall, precision)
    plt_x, plt_y, lbl_x, lbl_y = recall, precision, "Recall", "Precision"
  else:
    raise ValueError("Use 'fp_tp' or 'pre_rec' as kind parameter")

  print("AUC: ", auc_score)
  if kind == 'fp_tp':
    print("No-skill AUC: ", ns_auc_score)
  print("")

  if plot == True:
    plt.title("ROC Curve")
    plt.plot(plt_x, plt_y, marker='.', label=label)
    if kind == 'fp_tp':
      plt.plot(noskill_fp_rate, noskill_tp_rate, linestyle='--', label='No Skill Classifer')
    plt.xlabel(lbl_x)
    plt.ylabel(lbl_y)
    plt.legend()
    plt.show()

# Print Grid Search Results
def print_grid_search_results(search):
    '''This methods prints Grid Search Results for given search algorithm'''
    print("==== Grid Search Results ====")
    print("best_estimator: ", search.best_estimator_)
    print("best_params:    ", search.best_params_)
    print("best_score:      {:.3f}".format(search.best_score_))

# Print Classification Report - Classification Evaluation Method
def print_eval_parameters(model, y_test, y_pred, labels):
  '''This methods prints all evaluation parameters for classification models'''
  print("====== " + type(model).__name__ +" model Evaluation metrics ======")
  print("Accuracy of model:      {:.3f}".format(accuracy_score(y_test, y_pred)))                    # Accuracy score: (tp + tn) / (tp + fp + tn + fn)
  print("Recall of model:        {:.3f}".format(recall_score(y_test, y_pred, average="micro")))     # Recall score: tp / (tp + fn)
  print("Precision of model:     {:.3f}".format(precision_score(y_test, y_pred, average="micro")))  # Precision score: tp / (tp + fp)
  print("F1 score of model:      {:.3f}".format(f1_score(y_test, y_pred, average="micro")))         # F1 score: 2 * (precision * recall) / (precision + recall)
  # print("Mean accuracy of the model (Score):  {:.3f}".format(model.score(X_train_valid_scl, y_train_valid)))  # Print model Mean Accuracy (score)
  print("Misclassification Number: ", (y_test != y_pred).sum())
  print("\n====== " + type(model).__name__ +" model Detailed Classification Report ======")
  # Print K Nearest Neighbor model's classification report for validation set
  # Report contains; Precision, recal and F1 score values for each label and
  # model's accuracy, macro and weighted average
  print(classification_report(y_test, y_pred, target_names=labels))

# This method prints predicted and actual labels and shows actual image
def show_prediction_result(x_test, y_pred, y_test, labels, n_img=None):
    # If n_img is not set, find random index between 0 and x_test length
    if n_img is None:
        n_img = rnd.randint(0, (len(x_test)- 1))
    print("====== Random Prediction Result ======")
    print("Predicted label: " + labels[y_pred[n_img]], end="")
    print("  -  Actual label: " + labels[y_test[n_img]])
    # If Predicted label and Actual label are not same there is a classification mismatch
    if labels[y_pred[n_img]] != labels[y_test[n_img]]:
        print("There is a classification mismatch here!")
    plt.title("The image of "+ labels[y_test[n_img]] +" from the Dataset")
    # plt.imshow(x_test[n_img], cmap=plt.cm.gray_r)
    plt.imshow(x_test[n_img])
    plt.show()

# Plot the validation and training data separately
# Example Usage - Train a model: model_hist = model.fit(x_train, y_train, epochs=...)
# plot_loss_curves(model_history)
def plot_loss_curves(model_hist, all_in_one=False):
  """
  Returns separate loss curves for training and validation metrics.
  """
  # 
  if all_in_one == True:  # Plots 'loss', 'accuracy', 'val_loss', 'val_accuracy' in the same graph
    pd.DataFrame(model_hist.history).plot(figsize=(10, 7))
  else:
    loss = model_hist.history['loss']
    val_loss = model_hist.history['val_loss']

    accuracy = model_hist.history['accuracy']
    val_accuracy = model_hist.history['val_accuracy']

    epochs = range(len(model_hist.history['loss']))

    # Plot loss
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();

# Model History Comparison (especially after Fine-tuning processes)
def compare_historys(base_model_history, fine_tune_model_history, initial_epochs=5):
    """
    Compares two model history objects.
    """
    # Get base_model history measurements
    acc = base_model_history.history["accuracy"]
    loss = base_model_history.history["loss"]

    print(len(acc))

    val_acc = base_model_history.history["val_accuracy"]
    val_loss = base_model_history.history["val_loss"]

    # Combine base_model history with fine_tune_model_history
    total_acc = acc + fine_tune_model_history.history["accuracy"]
    total_loss = loss + fine_tune_model_history.history["loss"]

    total_val_acc = val_acc + fine_tune_model_history.history["val_accuracy"]
    total_val_loss = val_loss + fine_tune_model_history.history["val_loss"]

    print(len(total_acc))
    print(total_acc)

    # Make plots
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(total_acc, label='Training Accuracy')
    plt.plot(total_val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(total_loss, label='Training Loss')
    plt.plot(total_val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1, initial_epochs-1],
              plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()

def get_misclassified_indexes(y_test, y_pred):
    missed = []
    missed = np.where(y_test[y_test != y_pred])
    return missed[0]

def resize_image_with_aspect_ratio(image, width = None, height = None, inter = cv2.INTER_AREA):
    scl = 0
    (h, w) = image.shape[0], image.shape[1]

    if (width is None) and (height is None):
      raise ValueError("Please set a value for one of width or height parameters")
    elif height is None:
        # Calculate aspect ratio
        ratio = width / float(w)
        scl = (width, int(h * ratio))
    else:
        # Calculate aspect ratio
        ratio = height / float(h)
        scl = (int(w * ratio), height)

    # Resize the image
    resized = cv2.resize(image, scl, interpolation = inter)

    return resized

def overlay_heatmap(heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_JET):
  '''
  Combines Heatmap and the input image
  heatmap: 1 channel image input (W, H, 1), generally used to show gradients
  image: input image
  alpha: overlay ratio
  '''
  # Use jet colormap to colorize heatmap
  jet = cm.get_cmap("jet")  # cm.get_cmap will be deprecated after few releases
  # jet = matplotlib.colormaps.get_cmap("jet")

  # Use RGB values of the colormap
  jet_colors = jet(np.arange(256))[:, :3]  # First 3 channel: R, G, B
  jet_heatmap = jet_colors[heatmap]

  # Create an image with RGB colorized heatmap
  jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
  jet_heatmap = jet_heatmap.resize((image.shape[1], image.shape[0]))
  jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)

  # Superimpose/Combine the heatmap on original image
  superimposed_img = jet_heatmap * alpha + image
  superimposed_img = tf.keras.utils.array_to_img(superimposed_img)

  # Save the superimposed image
  # superimposed_img.save(cam_path)

  # apply the supplied color map to the heatmap and then
  # overlay the heatmap on the input image
  #heatmap = cv2.applyColorMap(heatmap, colormap)
  #superimposed_img = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)

  return (heatmap, superimposed_img)

def visualize_feature_matching(org_img_file, ref_img_file):
  # Load the images
  original_img = cv2.imread(org_img_file)
  reference_img = cv2.imread(ref_img_file)

  # Fit window (Large images do not fit in window for Colab)
  if reference_img.shape[1] > original_img.shape[1]:
    split_ratio = original_img.shape[1] // reference_img.shape[1]
    REF_WINDOW_LIMIT = WINDOW_FIT_LIMIT * split_ratio

  def fit_window(img, limit):
    if img.shape[1] >= limit:
      img = resize_image_with_aspect_ratio(img, width=limit)
    if img.shape[0] >= limit:
      img = resize_image_with_aspect_ratio(img, height=limit)
    return img

  # Rescale images for fitting the window
  original_img = fit_window(original_img, WINDOW_FIT_LIMIT)
  reference_img = fit_window(reference_img, REF_WINDOW_LIMIT)
  original_img_gray = fit_window(cv2.imread(org_img_file, cv2.IMREAD_GRAYSCALE), WINDOW_FIT_LIMIT)
  reference_img_gray = fit_window(cv2.imread(ref_img_file, cv2.IMREAD_GRAYSCALE), REF_WINDOW_LIMIT)

  # Initialize SIFT detector
  sift = cv2.SIFT_create()

  # Detect key points and compute descriptors
  kp1, dst1 = sift.detectAndCompute(original_img_gray, None)
  kp2, dst2 = sift.detectAndCompute(reference_img_gray, None)

  # Initialize the Brute-Force Matcher
  bf = cv2.BFMatcher()

  # Match descriptors
  matches = bf.knnMatch(dst1, dst2, k=2)

  # Apply ratio test
  good_matches = []
  for m, n in matches:
      if m.distance < 0.75 * n.distance:
          good_matches.append(m)

  # Draw matches
  matching_result = cv2.drawMatches(original_img, kp1, reference_img, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
  cv2_imshow(matching_result)

def visualize_feature_heatmap(model, image, conv_layer_name, loss="mae", pool="max",
                              overlay_alpha=0.8, normalize=True, plot_row=5, show=True):
    # Input image shape control
    if len(image.shape) == 3:
        image = tf.expand_dims(image, axis=0)  # expand dim to give input for a model
    elif len(image.shape) < 3:
        raise ValueError("Input image shape length can not be lower than 3")
    
    def normalizing_result(image, normalizing):
        if normalizing:
            return utils.normalize_image(image)
        else:
            return image
    
    # Generate image
    generated_image = model.predict(image)
    # Create model that returns generated image output and the specified conv_layer output
    gradModel = tf.keras.Model(inputs=[model.inputs], outputs= [model.get_layer(conv_layer_name).output, model.output])

    for layer in gradModel.layers:
        layer.trainable=False

    conv2d_block_layer_out, recons_layer_out = gradModel(image)
    conv2d_block_layer_out_gen, recons_layer_out_gen = gradModel(generated_image)

    # Calculate specified conv layer differences for generated and input image
    if loss == "mse":
        conv2d_block_layer_diff = tf.square(conv2d_block_layer_out_gen - conv2d_block_layer_out)
    elif loss == "mae":
        conv2d_block_layer_diff = tf.abs(conv2d_block_layer_out - conv2d_block_layer_out_gen)

    max_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=3, keepdims=True))(conv2d_block_layer_diff)
    mean_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))(conv2d_block_layer_diff)

    max_mean_pool = tf.keras.layers.UpSampling2D(size=(256//mean_pool.shape[1]))(max_pool + mean_pool)
    max_pool = tf.keras.layers.UpSampling2D(size=(256//max_pool.shape[1]))(max_pool)
    mean_pool = tf.keras.layers.UpSampling2D(size=(256//mean_pool.shape[1]))(mean_pool)

    image = normalizing_result((tf.squeeze(image, axis=0).numpy()), normalize)

    if pool == "max":
        pooling_result = max_pool
    elif pool == "avg":
        pooling_result = mean_pool
    elif pool == "max+avg":
        pooling_result = max_mean_pool

    pooling_result = normalizing_result((tf.squeeze(pooling_result, axis=0)), normalize)
                              
    _, org_img_pooling_result = overlay_heatmap((pooling_result.numpy()), image, alpha=overlay_alpha)

    image_matrix = [
        image,
        normalizing_result((tf.squeeze(max_pool, axis=0).numpy()), normalize),
        normalizing_result((tf.squeeze(mean_pool, axis=0).numpy()), normalize),
        normalizing_result((tf.squeeze(max_mean_pool, axis=0).numpy()), normalize),
        org_img_pooling_result
    ]

    label_array = [
        'Original',
        'Max',
        'Avg',
        'Max+Avg',
        'Org+Max+Avg'
    ]

    fig = plt.figure(figsize=(12,12))

    col = (len(label_array) // plot_row) + 1

    if show:
        for subplot in range(plot_row*col):
          plt.subplot(col,5,subplot+1)
          plt.imshow(image_matrix[subplot], cmap='jet')
          plt.axis('off')
          plt.title(label_array[subplot])
          if (subplot+1) == len(label_array):
              break

        plt.show()

    return normalizing_result((tf.squeeze(max_mean_pool, axis=0).numpy()), normalize)

class GradCAM:
  '''
  GRADCAM class
  model: Model object with trained weights
  layer_name: specified layer to be extracting gradients from
  class_id: label
  alpha: transparency factor for a combination of heatmap and original image
  Reference: https://github.com/wiqaaas/youtube/blob/master/Deep_Learning_Using_Tensorflow/Demystifying_CNN/Gradient%20Visualization.ipynb

  Example usage:
  vgg16_model = tf.keras.applications.VGG16(weights="imagenet")
  ...
  preds = vgg16_model.predict(image)
  class_id = np.argmax(preds[0])
  ...
  gradCAM = GradCAM(vgg16_model)
  (heatmap, output) = gradCAM(image, original_image, class_id, alpha=0.8)
  plt.imshow(output)
  '''
  def __init__(self, model, layer_name=None):
    self.model = model
    self.layer_name = layer_name

    if self.layer_name is None:
      self.layer_name = self.find_target_layer()

  def find_target_layer(self):
    # Find the last convolutional layer of the model before Global Average Pooling
    for layer in reversed(self.model.layers):
        # check to see if the layer has a 4D output
        if len(layer.output.shape) == 4:
            return layer.name
    # otherwise, we could not find a 4D layer
    raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")
  
  def __call__(self, model_input_img, original_image, class_id, alpha):
    heatmap = self.compute_heatmap(model_input_img, class_id)
    return self.overlay_heatmap(heatmap, original_image, alpha=alpha) 

  def compute_heatmap(self, image, class_id, eps=1e-8):
    self.class_id = class_id
    self.eps = eps
    self.image = image
    # Construct gradient model by supplying the inputs to the pre-trained model
    # (The output of the final 4D layer in the network, and the output of the
    # softmax activations from the model
    gradModel = tf.keras.Model(inputs=[self.model.inputs], outputs= [self.model.get_layer(self.layer_name).output, self.model.output])

    # Compute the gradients
    with tf.GradientTape() as tape:
      inputs = tf.cast(self.image, tf.float32)  # cast the image tensor to a float-32
      (last_conv_layer_output, preds) = gradModel(inputs)
      loss = preds[:, self.class_id]

    # Compute the gradients for last convolutional layer
    grads = tape.gradient(loss, last_conv_layer_output)

    # Compute guided gradients and get heatmap
    return self.compute_guided_gradients_create_heatmap(grads, last_conv_layer_output)

  def compute_guided_gradients_create_heatmap(self, gradients, last_conv_layer_output):
    # Compute the guided gradients
    cast_conv_layer_output = tf.cast(last_conv_layer_output > 0, tf.float32)
    cast_grads = tf.cast(gradients > 0, tf.float32)
    guided_grads = cast_conv_layer_output * cast_grads * gradients

    # the convolution and guided gradients have a batch dimension
    # (which we don't need) so let's grab the volume itself and
    # discard the batch
    last_conv_layer_output = last_conv_layer_output[0]
    guided_grads = guided_grads[0]

    # compute the average of the gradient values, and using them
    # as weights, compute the ponderation of the filters with
    # respect to the weights
    weights = tf.reduce_mean(guided_grads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, last_conv_layer_output), axis=-1)

    # grab the spatial dimensions of the input image and resize the output class
    # activation map to match the input image dimensions
    (w, h) = (self.image.shape[2], self.image.shape[1])
    heatmap = cv2.resize(cam.numpy(), (w, h))

    # For visualization purpose normalize the heatmap such that
    # all values lie in the range [0, 1], scale the resulting values to the
    # range [0, 255], and then convert to an unsigned 8-bit integer
    numer = heatmap - np.min(heatmap)
    denom = (heatmap.max() - heatmap.min()) + self.eps
    heatmap = numer / denom
    heatmap = (heatmap * 255).astype("uint8")

    # return the resulting heatmap to the calling function
    return heatmap

  # Combine Grad model output and the original image
  def overlay_heatmap(self, heatmap, image, alpha=0.5, colormap=cv2.COLORMAP_JET):
    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")  # cm.get_cmap will be deprecated after few releases
    # jet = matplotlib.colormaps.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]  # First 3 channel: R, G, B
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((image.shape[1], image.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)

    # Superimpose/Combine the heatmap on original image
    superimposed_img = jet_heatmap * alpha + image
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    # superimposed_img.save(cam_path)

    # apply the supplied color map to the heatmap and then
    # overlay the heatmap on the input image
    #heatmap = cv2.applyColorMap(heatmap, colormap)
    #superimposed_img = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)

    return (heatmap, superimposed_img)
  
class GradCAM_AE(GradCAM):
  '''
  GRADCAM_AE class - GradCAM for Autoencoder models
  model: Model object with trained weights
  layer_name: specified layer to be extracting gradients from
  class_id: label
  alpha: transparency factor for a combination of heatmap and original image
  Reference: https://github.com/wiqaaas/youtube/blob/master/Deep_Learning_Using_Tensorflow/Demystifying_CNN/Gradient%20Visualization.ipynb

  Example usage:
  vgg16_model = tf.keras.applications.VGG16(weights="imagenet")
  ...
  preds = vgg16_model.predict(image)
  class_id = np.argmax(preds[0])
  ...
  gradCAM = GradCAM(vgg16_model)
  (heatmap, output) = gradCAM(image, original_image, class_id, alpha=0.8)
  plt.imshow(output)
  '''
  def __init__(self, model, layer_name=None):
    super().__init__(model, layer_name)
  
  def __call__(self, model_input_img, original_image, alpha):
    heatmap = self.compute_heatmap(model_input_img)
    return self.overlay_heatmap(heatmap, original_image, alpha=alpha)

  # Override this method
  def compute_heatmap(self, image, eps=1e-8):
    self.eps = eps
    self.image = image
    # Construct gradient model by supplying the inputs to the pre-trained model
    # (The output of the final 4D layer in the network, and the output of the
    # softmax activations from the model
    gradModel = tf.keras.Model(inputs=[self.model.inputs], outputs= [self.model.get_layer(self.layer_name).output, self.model.output])

    # Compute the gradients
    with tf.GradientTape() as tape:
      inputs = tf.cast(image, tf.float32)  # cast the image tensor to a float-32
      (last_conv_layer_output, reconstructed_output) = gradModel(inputs)
      loss = tf.reduce_mean(tf.square(inputs - reconstructed_output))

    # Compute the gradients for last convolutional layer
    grads = tape.gradient(loss, last_conv_layer_output)

    # Compute guided gradients and get heatmap
    return self.compute_guided_gradients_create_heatmap(grads, last_conv_layer_output)

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