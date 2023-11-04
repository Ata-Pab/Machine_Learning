import os
import zipfile
import glob
import re  # Regex for string parsing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
import tensorflow_probability as tfp
from datetime import datetime
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score # Evaluation metrics
from sklearn.metrics import classification_report  # Precision, recall, f1-score metrics


ROT_0   = 0
ROT_90  = 1
ROT_180 = 2
ROT_270 = 3

# Model Layers Describe - Print Model Layer's name, number & trainable state
def describe_model_layers(model):
    for layer_number, layer in enumerate(model.layers):
      print(layer_number, layer.name, layer.trainable)

# Get all image files for valid extensions
def get_image_file_list(dir, ext='jpeg'):
    formats = ['jpg', 'png', 'jpeg', 'JPG', 'JPEG', 'PNG']
    img_list = []

    if ext in formats:
        rgx_for_img = dir + "/*." + ext
        img_list = glob.glob(rgx_for_img)

    return img_list

# Get all image files in specified directory (including sub-folders)
def get_all_img_files_in_directory(data_dir, ext='jpeg', verbose=0):
    img_file_list = []
    for dirpath, dirnames, filenames in os.walk(data_dir):
      if len(filenames) > 0:
        img_file_list.extend(get_image_file_list(dirpath, ext=ext))

      if verbose > 0:
        print(f"There are {len(dirnames)} directories and {len(filenames)} files in '{dirpath}'.")
    
    return img_file_list

def prepare_imgs_as_keras_input(np_array, input_size):
    '''This method prepares a data array as a keras input
    Example of input_size parameter usage: (28,28,1) -> 28x28 pixels gray scale images, (28,28,3) 28x28 pixels RGB images
    Example of np_array parameter usage: np_array.shape = (60000, 28, 28) OR (60000, 784)'''
    w, h, d = input_size
    return np_array.reshape(np_array.shape[0], w, h, d)

# Get all files in specified directory
# # Example usage
# ANNOT_PATH = "/content/Detection/train"
# xml_files = get_all_files_ext(ANNOT_PATH, "xml")
def get_all_files_ext(path, ext, sort=True):
    '''Get all files with the given file extension in the given path'''
    ext = "." + str(ext)
    files = [
        os.path.join(path, file_name)
        for file_name in os.listdir(path)
        if file_name.endswith(ext)
    ]

    if sort == True:
      return sorted(files)
    else:
      return files

# Unzip the downloaded file
def unzip_data(zip_file_name):
    zip_ref = zipfile.ZipFile(zip_file_name, "r")
    zip_ref.extractall()
    zip_ref.close()

# Print working or data directory map - walk through directory
def walk_through_dir(data_dir):
    for dirpath, dirnames, filenames in os.walk(data_dir):
      print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

# Create a function to import an image and resize it to be able to be used with our model
def load_images(filename, img_size=None, scl=True, num_channels=3, rot=ROT_0, accelerator='GPU'):
    # Read in target file (an image)
    if accelerator == 'TPU':
      with open(filename, "rb") as img_file:
        img = img_file.read()
    else:
      img = tf.io.read_file(filename)

    # Decode the read file into a tensor & ensure 3 colour channels
    # (our model is trained on images with 3 colour channels and sometimes images have 4 colour channels)
    img = tf.image.decode_image(img, channels=num_channels)

    # Resize the image (to the same size our model was trained on)
    if img_size != None:
      img = tf.image.resize(img, size = img_size, preserve_aspect_ratio=False)

    # Rotate images 270 degree due to capturing photos in vertical position with iPhone
    if rot != ROT_0:
        img = tf.image.rot90(img, k=rot)

    # Cast to float32
    img = tf.cast(img, tf.float32)

    # Rescale the image (get all values between 0 and 1)
    if scl == True: img = img / 255.
    return img

def load_and_prepare_images(img_file_list, img_size=None, scl=None, num_channels=3, rot=ROT_0, accelerator='GPU'):
    def process_images(filename):
        return load_images(filename, img_size, scl, num_channels, rot, accelerator)

    return np.array(list(map(process_images, img_file_list)))

# Create a new dataset that includes both the original and augmented images
def concatenate_images(original_images, augmented_images):
    return tf.concat([original_images, augmented_images], axis=0)


def create_dataset_pipeline(img_files, batch_size, img_size, scl=True, shuffle=False, num_channels=3, rot=ROT_0, data_augmentation=False, aug_layer=None, data_aug_power=1, accelerator='GPU'):
    # Read images from directory and reshape, scale
    dataset = tf.data.Dataset.from_tensor_slices(load_and_prepare_images(img_files, img_size=img_size, scl=scl, num_channels=num_channels, rot=rot, accelerator=accelerator))
    
    if data_augmentation == True:
        # Apply specified augmentation sequential layer to the image
        def apply_augmentation(image):
            image = aug_layer(image, training=True)  # Apply data augmentation layers
            return image
        
        temp_dataset = dataset
        for _ in range(data_aug_power):
          augmented_dataset = tf.data.Dataset.from_tensor_slices(np.array(list(temp_dataset.map(apply_augmentation))))
          dataset = dataset.concatenate(augmented_dataset)
    # Get image pairs for data pipeline (as an autoencoder input) - Do not create image pairs if you train the model with GANs
    # dataset = dataset.map(lambda image: (image, image), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # Shuffle (only training set) and create batches
    if shuffle == True:
        dataset = dataset.shuffle(len(img_files))
    dataset = dataset.batch(batch_size)
    dataset = dataset.cache()
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

# Detecting and Initializing TPU
def initialize_TPU():
    tpu_address = 'grpc://' + os.environ['COLAB_TPU_ADDR']  # For Google Colab
    # For GCP, replace with the actual TPU address
    # tpu_address = 'grpc://'
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_address)
        print('Running on TPU ', resolver.master())
    except ValueError:
        resolver = None

    if resolver is not None:
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)

    else:
        strategy = tf.distribute.get_strategy()

    print("All devices:  ", tf.config.list_logical_devices('TPU'))
    print("All replicas: ", strategy.num_replicas_in_sync)

    return strategy

# Read TFRecord File
def read_tfRecord_image(filename, feature_description, up_to_batch=None):
    result = []

    raw_dataset = tf.data.TFRecordDataset(filename)

    def _parse_function(example_proto):
      # Parse the input `tf.train.Example` proto using the dictionary above.
      return tf.io.parse_single_example(example_proto, feature_description)

    parsed_dataset = raw_dataset.map(_parse_function)

    if up_to_batch != None:
      for raw_record in parsed_dataset.take(up_to_batch):
          result.append(raw_record)
    else:
      for raw_record in parsed_dataset:
          result.append(raw_record)

    return result
      
def get_replica_dataset(dataset, batch_size, is_training=True):
    # Only shuffle and repeat the dataset in training. The advantage of having an
    # infinite dataset for training is to avoid the potential last partial batch
    # in each epoch, so that you don't need to think about scaling the gradients
    # based on the actual batch size.
    if is_training:
      dataset = dataset.shuffle(50)
      dataset = dataset.repeat()

    #dataset = dataset.batch(batch_size)

    return dataset

# Get prediction analysis for classification model
# # Example Usage
# wrong_preds = get_wrong_predictions(img_file_paths, y_true, y_pred, pred_probs, [class_names[i] for i in y_true], [class_names[i] for i in y_pred])
# wrong_preds.head()
#
# # Visualize some of the most wrong predictions
#
# images_to_view = 16
# start_ix = 11
# plt.figure(figsize=(15,10))
#
# for i, row in enumerate(wrong_preds[start_ix:start_ix+images_to_view].itertuples()):
#   plt.subplot(4, 4, i+1)
#   img = load_and_prep_image(row[1], scale=True)
#   _, _, _, _, pred_prob, y_true, y_pred, _ = row # only interested in a few parameters of each row
#   plt.imshow(img)
#   plt.title(f"actual: {y_true}, pred: {y_pred} \nprob: {pred_prob:.2f}")
#   plt.axis(False)
def get_wrong_predictions(img_file_paths, y_true, y_pred, pred_probs, y_true_class_names=None, y_pred_class_names=None, analysis='bad'):
  assert (analysis == 'bad' or  analysis == 'good'), "Invalid analysis type. Please choose one of 'good', 'bad', analysis types"

  dataframe = {"img_path": img_file_paths,
             "y_true": y_true,
             "y_pred": y_pred,
             "pred_conf": pred_probs.max(axis=1), # get the maximum prediction probability value
             "y_true_classname": y_true_class_names,
             "y_pred_classname": y_pred_class_names}

  pred_df = pd.DataFrame(dataframe)
  pred_df['pred_correct'] = pred_df['y_true'] == pred_df['y_pred']

  # Get Wrong predictions
  if analysis == 'bad':
    wrong_preds = pred_df[pred_df['pred_correct'] == False]
    wrong_preds = wrong_preds.sort_values('pred_conf', ascending=False)[:100]  # Most wrong 100 predictions
    return wrong_preds
  elif analysis == 'good':
    good_preds = pred_df[pred_df['pred_correct'] == True]
    good_preds = good_preds.sort_values('pred_conf', ascending=False)[:100]  # Most correct 100 predictions
    return good_preds


# Parsing Image Annotation Files - PASCAL VOC (xml) format
import xml.etree.ElementTree as ET
# # Example Usage
# IMG_PATH = "/content/Detection/train"    # Contains img files paths
# ANNOT_PATH = "/content/Detection/train"  # Contains xml files paths
#
# class_ids = [
#     "Arduino_Nano",
#     "Heltec_ESP32_Lora",
#     "ESP8266",
#     "Raspberry_Pi_3",
# ]
# class_mapping = dict(zip(range(len(class_ids)), class_ids))
#
# xml_files = get_all_files_ext(ANNOT_PATH, "xml")
# xml_file = xml_files[0]
#
# image_path, boxes, class_ids = parse_annotation(xml_file, IMG_PATH, class_mapping)

def parse_annotation(xml_file, img_path, class_mapping):
    '''
    Reads the XML file and finds the image name and path, iterates over each
    object in the XML file to extract the bounding box coordinates and class
    labels for each object.

    Returns:
    - The image path
    - List of bounding boxes (each represented as a list of four floats: xmin,
    ymin, xmax, ymax),
    - List of class IDs (represented as integers) corresponding to each bounding
    box. The class IDs are obtained by mapping the class labels to integer values
    using a dictionary called class_mapping.
    '''
    tree = ET.parse(xml_file)
    root = tree.getroot()

    image_name = root.find("filename").text
    image_path = os.path.join(img_path, image_name)

    boxes = []
    classes = []

    for obj in root.iter("object"):
      cls = obj.find("name").text
      classes.append(cls)

      bbox = obj.find("bndbox")

      xmin = float(bbox.find("xmin").text)
      ymin = float(bbox.find("ymin").text)
      xmax = float(bbox.find("xmax").text)
      ymax = float(bbox.find("ymax").text)

      boxes.append([xmin, ymin, xmax, ymax])

    class_ids = [
        list(class_mapping.keys())[list(class_mapping.values()).index(cls)]
        for cls in classes
    ]
    return image_path, boxes, class_ids

# Print Grid Search Results
def print_grid_search_results(search):
    '''This methods prints Grid Search Results for given search algorithm'''
    print("==== Grid Search Results ====")
    print("best_estimator: ", search.best_estimator_)
    print("best_params:    ", search.best_params_)
    print("best_score:      {:.3f}".format(search.best_score_))

# Save custom model
# # Example Usage
# save_custom_model(model, "custom_model")
# # Loading the model back
# loaded_model = keras.models.load_model('/path/to/custom_model_2023_08_16_06_45_05.keras')
def save_custom_model(model, name="model", verbose=1):
    # Get Datetime object containing current date and time
    date = str(datetime.now())
    date = (((date.replace("-", "_")).replace(":", "_")).replace(" ", "_")).split(".")[0]

    model_name = str(name) + str(date) + ".keras"
    model.save(model_name)

    if verbose > 0:
        print(f"{model_name} was saved successfully")

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

def change_img_file_format(img_dir, src_format, dest_format):
    formats = ['jpg', 'png', 'jpeg', 'JPG', 'JPEG', 'PNG']
    cnv_rgx_frmt = lambda frmt: '\.' + frmt + '$' 
    rgx_formats = map(cnv_rgx_frmt, formats)

    if any([(re.search(format, img_dir)!=None) for format in rgx_formats]):
        if dest_format in formats:
            if dest_format in img_dir[-5:]:
                print("UserWarning: This image file is already {0} format".format(dest_format))
            else:
                img = Image.open(img_dir)
                img.save()
        else:
            raise ValueError("Invalid image format, use '.png', '.jpeg', '.jpg' file formats")    
    else:
        raise ValueError("Invalid image format, use '.png', '.jpeg', '.jpg' file formats")
    
# Convert HEX code pixel RGB code pixel
def hexc_to_rgbc(hex_code):
    hex_code = hex_code.lstrip('#')
    rgb_img = np.array(tuple(int(hex_code[i:i+2], 16) for i in (0,2,4)))
    return rgb_img

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


     