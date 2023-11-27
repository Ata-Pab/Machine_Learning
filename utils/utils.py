import os
import zipfile
import cv2
import glob
import re  # Regex for string parsing
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random as rnd
from PIL import Image
import tensorflow as tf
#import tensorflow_probability as tfp
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler, StandardScaler


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
def get_all_img_files_in_directory(data_dir, ext='jpeg', exc="", verbose=0):
    '''
    Get all image files in specified directory (including sub-folders)
    data_dir: Investigated directory 
    ext: extension to be found
    exc: exclude directories from search
    verbose: output info
    :return image file list
    '''
    img_file_list = []
    for dirpath, dirnames, filenames in os.walk(data_dir):
      if not((exc != "") and (exc in dirpath)):
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
def load_images(filename, img_size=None, aspect=False, scl=True, num_channels=3, rot=ROT_0, accelerator='GPU'):
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
      if aspect == True:
        width = img.shape[1]  # Get width of the image
        ratio = width / float(width)
        img_size = (width, int(img.shape[0] * ratio))

      img = tf.image.resize(img, size = img_size, preserve_aspect_ratio=False)

    # Rotate images 270 degree due to capturing photos in vertical position with iPhone
    if rot != ROT_0:
        img = tf.image.rot90(img, k=rot)

    # Cast to float32
    img = tf.cast(img, tf.float32)

    # Rescale the image (get all values between 0 and 1)
    if scl == True: img = img / 255.
    return img

def load_and_prepare_images(img_file_list, img_size=None, aspect=False, scl=None, num_channels=3, rot=ROT_0, accelerator='GPU'):
    def process_images(filename):
        return load_images(filename, img_size, aspect, scl, num_channels, rot, accelerator)

    return np.array(list(map(process_images, img_file_list)))

# Create a new dataset that includes both the original and augmented images
def concatenate_images(original_images, augmented_images):
    return tf.concat([original_images, augmented_images], axis=0)


# Create Dataset Pipeline for tf.models
def create_dataset_pipeline(img_files, batch_size, img_size=None, aspect=False, scl=True, shuffle=False, num_channels=3, rot=ROT_0, duplicate=False, aug_layer=None, data_aug_power=1, accelerator='GPU'):
    '''
    img_files: Image file list
    batch_size: Batch size
    img_size: Resize image (h,w)
    aspect: Keep aspect ratio of the image or not
    scl= Scale image. default True
    shuffle= Shuffle images. default False
    num_channels= number of channels. default 3
    rot= Rotate all images in the dataset. default ROT_0
    duplicate= Duplicate input image data. default False
    Note: Use duplicate for unsupervised learning like autoencoders
    aug_layer= tensorflow.keras.models.Sequential([...]). default None
    data_aug_power= How many times data augmentation will be applied to the whole dataset. default 1
    accelerator= 'GPU' or 'TPU'. default 'GPU'
    '''
    # Read images from directory and reshape, scale
    dataset = tf.data.Dataset.from_tensor_slices(load_and_prepare_images(img_files, img_size=img_size, aspect=aspect, scl=scl, num_channels=num_channels, rot=rot, accelerator=accelerator))
    
    if aug_layer != None:
        # Apply specified augmentation sequential layer to the image
        def apply_augmentation(image):
            image = aug_layer(image, training=True)  # Apply data augmentation layers
            if scl:
                image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
            return image
        
        temp_dataset = dataset
        for _ in range(data_aug_power):
          augmented_dataset = tf.data.Dataset.from_tensor_slices(np.array(list(temp_dataset.map(apply_augmentation))))
          dataset = dataset.concatenate(augmented_dataset)

    # Get image pairs for data pipeline (as an autoencoder input) - Do not create image pairs if you train the model with GANs
    if duplicate:
      dataset = dataset.map(lambda image: (image, image), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    # Shuffle (only training set) and create batches
    if shuffle == True:
        if aug_layer != None:
            dataset = dataset.shuffle(len(img_files)*data_aug_power)
        else:
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

# Save custom model
# # Example Usage
# save_custom_model(model, "custom_model")
# # Loading the model back
# loaded_model = keras.models.load_model('/path/to/custom_model_2023_08_16_06_45_05.keras')
def save_custom_model(model, name="model", verbose=1):
    # Get Datetime object containing current date and time
    date = str(datetime.now())
    date = (((date.replace("-", "_")).replace(":", "_")).replace(" ", "_")).split(".")[0]

    model_name = str(name) + "_" + str(date) + ".keras"
    model.save(model_name)

    if verbose > 0:
        print(f"{model_name} was saved successfully")

# Change image file format to valid formats
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

def pip_install_package(package):
    try:
      return __import__(package)
    except ImportError:
      import sys
      import subprocess
      # implement pip as a subprocess:
      subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
      # process output with an API in the subprocess module:
      reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
      installed_packages = [r.decode().split('==')[0] for r in reqs.split()]
      print(installed_packages)

# Divide images into same sized partitions
def patchify_images(img_file_list, patch_size=256, img_size=None, method='CROP', scl=False, cvt_rgb=False, verbose=0):
    '''
    img_file_list: Takes image files' paths as input
    patch_size: Divides all images to speicifed patch size
    img_size: Resize images before patchify
    method: Partitioning method -> 'CROP' or 'RESIZE'
    scl: Scales images
    cvt_rgb: Convert BGR decoding to RGB format
    Return: Image numpy array

    Need patchify package (https://pypi.org/project/patchify/)
    ! pip install patchify
    Modified code of Dr. Sreenivas Bhattiprolu
    '''
    package = 'patchify'

    pip_install_package(package)
       
    from patchify import patchify

    image_dataset = []
    scaler = MinMaxScaler(feature_range=(0,1))

    for image_file in img_file_list:
      image = cv2.imread(image_file, 1)  # Read each image as BGR
      if img_size != None:
        # Resize the image
        image = image.resize(img_size[1], img_size[0])
      if cvt_rgb == True:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      SIZE_X = (image.shape[1]//patch_size)*patch_size # Nearest size divisible by specified patch size
      SIZE_Y = (image.shape[0]//patch_size)*patch_size # Nearest size divisible by specified patch size
      image = Image.fromarray(image)
      if method == 'RESIZE':
        image = image.resize((SIZE_X, SIZE_Y))  # Not recommended for semantic segmentation
      else:
        image = image.crop((0 ,0, SIZE_X, SIZE_Y))  # Crop from top left corner
      image = np.array(image)
      patches_img = patchify(image, (patch_size, patch_size, 3), step=patch_size)  # Step = patch_size for patch_size patches means no overlap

      for i in range(patches_img.shape[0]):
        for j in range(patches_img.shape[1]):
            single_patch_img = patches_img[i,j,:,:]

            if scl ==True:
              # Use minmaxscaler instead of just dividing by 255.
              single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
              # single_patch_img = (single_patch_img.astype('float32')) / 255.

            single_patch_img = single_patch_img[0] # No need to other dimensions.
            image_dataset.append(single_patch_img)

    return np.array(image_dataset)

def unpatchify_img(patches, grid):
  cols, rows = grid[0], grid[1]
  patch_list = []
  num_col_patches = 0

  for row_ix in range(rows): 
    patch_list.append(patches[num_col_patches])
    for col_ix in range(cols-1):
      patch_list[row_ix] = np.concatenate((patch_list[row_ix], patches[num_col_patches+1+col_ix]), axis=1)
    num_col_patches += cols
    if row_ix == 0:
      image = patch_list[row_ix]
    else:
      image = np.concatenate((image, patch_list[row_ix]), axis=0)
  
  return image

def create_labels_for_mask(mask, categories):
  '''
  Set label masks as input in RGB format
  Replace pixels with specific RGB values
  categories: rgb_codes for specified categories
  '''
  label_seg = np.zeros(mask.shape, dtype=np.uint8)
  for ix, category in enumerate(categories):
      label_seg[np.all(mask == category, axis=-1)] = ix

  return label_seg[:,:,0]  # Just return the first channel

# Creates a label map for the given masks with class categories
def rgb_to_2D_label_map(mask_imgs, categories):
    labels = []

    for ix in range(mask_imgs.shape[0]):
      label = create_labels_for_mask(mask_imgs[ix], categories=categories)
      labels.append(label)

    labels = np.array(labels)
    labels = np.expand_dims(labels, axis=3)
    
    return labels

def gaussian_kernel(kernel_size, sigma, n_channels, dtype):
    # https://stackoverflow.com/questions/59286171/gaussian-blur-image-in-dataset-pipeline-in-tensorflow
    x = tf.range(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=dtype)
    g = tf.math.exp(-(tf.pow(x, 2) / (2 * tf.pow(tf.cast(sigma, dtype), 2))))
    g_norm2d = tf.pow(tf.reduce_sum(g), 2)
    g_kernel = tf.tensordot(g, g, axes=0) / g_norm2d
    g_kernel = tf.expand_dims(g_kernel, axis=-1)
    return tf.expand_dims(tf.tile(g_kernel, (1, 1, n_channels)), axis=-1)

def apply_blur(image, kernel_size=3, sigma=2):
    blur = gaussian_kernel(kernel_size, sigma, image.shape[-1], image.dtype)
    image = tf.nn.depthwise_conv2d(image[None], blur, [1,1,1,1], 'SAME')
    return image[0]

def apply_distortion(image, distortion_factor=0.05):
    assert (distortion_factor >= 0.0) and (distortion_factor <= 1.0)
    image += distortion_factor * tf.random.uniform(tf.shape(image))
    assert image.dtype == tf.float32
  
    # Clipping input data to the valid range
    image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    #image_np = image.numpy()
    #image_np[np.where(image_np > 1.0)] = 1.0
    return image

def random_cutout_image(image, min_mask_edge=5, max_mask_edge=20, num_cuts=1, padding=5):
    height, width, channels = image.shape
    tensor_format = False
    assert ((max_mask_edge < height // 2) and (max_mask_edge < width // 2))
  
    if not isinstance(image, np.ndarray):  # If input image is not numpy array (EagerTensor)
      tensor_format = True
      image = image.numpy()
  
    for cut in range(num_cuts):
      mask_size = (rnd.randint(min_mask_edge, max_mask_edge), rnd.randint(min_mask_edge, max_mask_edge))
      bbox = (rnd.randint(0, (height-mask_size[0]-padding)), rnd.randint(0, (width-mask_size[1]-padding)))  # padding=-5, no need to cut out areas that very close to corners of the image      
      image[bbox[0]:(bbox[0]+mask_size[0]),bbox[1]:(bbox[1]+mask_size[1]),:3] = 0
  
    if tensor_format:  # Return as Tensor
      return tf.convert_to_tensor(image, dtype=tf.float32)
    else:
      return image
    
def make_experiment_dir(data_dir):
    exp_num = 0

    for folder in os.listdir(data_dir):
        dirpath = os.path.join(data_dir, folder)
        if (os.path.isdir(dirpath)) and ("experiment_" in dirpath):
            exp_id = int(dirpath.split("experiment_")[1])  # Get experiment ID
            if exp_num < exp_id:
                exp_num = exp_id
    save_dir = data_dir + "/experiment_" + str(exp_num+1)
    os.makedirs(save_dir)
    return save_dir

def write_dict_to_file(_dict, file_dir, sep=": ", head=None):
    text = []
    if head != None:
        text.append(head)
    for key in _dict:
        row = str(key)
        text.append(row + sep + str(_dict[row]))

    with open(file_dir, 'w') as f:
        for line in text:
            f.write(line)
            f.write('\n')

def create_experimental_output(experiment_dict, save_dir):
    try:
        save_dir = make_experiment_dir(save_dir)
        exp_id = save_dir.split("experiment_")[1]  # Get experiment ID
        exp_id = "EXPERIMENT_" + str(exp_id) + " CONFIG\n"
        write_dict_to_file(experiment_dict, (save_dir + "/experiment.txt"), head=exp_id)
        return save_dir
    except:
        print("An error occurred while trying to create experimental output")

# Save experiment checkpoints method should be used in custom train function
def save_experiment_checkpoints(model_list, epoch, save_dir):
    '''
    model_list: List of models used in a custom model
    epoch: current epoch
    save_dir: model weight save directory
    '''
    for model in model_list:
        model.save_weights(os.path.join(save_dir, (model.name + "_epoch_" + str(epoch) + '.h5')))

def load_model_experiment_weights(model_list, epoch, load_dir):
    '''
    model_list: List of models used in a custom model
    epoch: current epoch
    load_dir: model weight load directory
    '''
    for model in model_list:
        model.load_weights(os.path.join(load_dir, (model.name + "_epoch_" + str(epoch) + '.h5')))

def remove_training_weights_except_last_epoch(weights_dir):
    max_trained_epoch = 0
    weights_will_be_removed = []
    for path in os.listdir(weights_dir):
        path = os.path.join(weights_dir, path)
        if (os.path.isfile(path)) and (path[-3:] == '.h5'):
            assert (len(path.split("_")) != 0)
            weights_will_be_removed.append(path)
            epoch_num = int((path.split("_")[-1])[:-3])  # Get trained epoch code

            if epoch_num > max_trained_epoch:
                max_trained_epoch = epoch_num

    for weight_file in weights_will_be_removed:
        if not(("_" + str(max_trained_epoch) + ".h5") in weight_file):
            user = input(f"{weight_file} will be removed. [y/n]: ")
            if user == 'y':
                os.remove(weight_file)

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
#
# https://stackoverflow.com/questions/59286171/gaussian-blur-image-in-dataset-pipeline-in-tensorflow


     