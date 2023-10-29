import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.manifold import TSNE
import numpy as np
import random as rnd
import itertools
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # Classifier Confusion Matrix visualization
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve

# Shows image samples from dataset
def show_image_samples_from_batch(dataset, grid=(4,4), figsize=(10, 10)):
    image_batch = next(iter(dataset))

    fig = plt.figure(figsize=figsize)

    for index, image in enumerate(image_batch):  # Get first batch
      plt.subplot(grid[0], grid[1], index + 1)
      plt.imshow(image[:, :, :])
      plt.axis('off')
      if index >= (len(image_batch)-1):
        break

# Display pixel wise imag difference (Valid inputs: img paths or img arrays)
def display_pixel_wise_img_diff(img1_dir, img2_dir, threshold=None, verbose=0, channel=0):
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
    diff = tf.abs(image1 - image2)
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