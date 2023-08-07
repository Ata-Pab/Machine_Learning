# Fundamental Libraries for Machine Learning
import numpy as np
import sys
import os
import re  # Regex for string parsing
import random as rnd
import matplotlib.pyplot as plt
from PIL import Image
import glob   # In order to get images as matrices from directory
import pickle

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  # Classifier Confusion Matrix visualization
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score # Evaluation metrics
from sklearn.metrics import classification_report  # Precision, recall, f1-score metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve

# Tensorflow Libraries
import tensorflow as tf
import tensorflow.keras as keras

def prepare_dataset_for_mask_rcnn(path, shuffle=False):
    img_path_list = glob.glob(os.path.join(path, ['*.jpg', '*.png']))
    #rnd.Random(24).shuffle(img_path_list)
    #padding = len(str(len(img_path_list)))  # number of digits for path names - If you have 1000 images your image file names will have 4 digit names like "0005", "0568" 

    #for img_num, img_path in enumerate(img_path_list, 1):
    #    os.rename(img_path, )
    print(img_path_list)


class VisionBasics():
    def __init__(self) -> None:
        self.files = []
        self.formats = ['jpg', 'png', 'jpeg']  # Add valid image formats here
        self._project_directory()
        pass

    def _project_directory(self):
        if (str(sys.platform) == "darwin") or (str(sys.platform) == "linux"):    # MacOS or Linux Environment
            current_dir = os.getcwd() + "/"
        else:   # Windows Environment
            current_dir = os.getcwd() + "\\"
        self.files = os.listdir(current_dir)

    def get_files(self):
        return self.files

    def print_files_of_project_directory(self):
        print(self.files)

    def save_ml_model(self, model, tag='Model'):
        pickle.dump(model, open(tag, 'wb'))

    def load_ml_model(self, directory):
        return pickle.load(open(directory, 'rb'))

    # Example of usage change_img_file_format method:
    # change_img_file_format.open(r'C:\Users\Ron\Desktop\Test\summer.png', )
    # TODO: Continue to write this method
    def change_img_file_format(self, img_dir, src_format, dest_format):
        cnv_rgx_frmt = lambda frmt: '\.' + frmt + '$' 
        rgx_formats = map(cnv_rgx_frmt, self.formats)

        if any([(re.search(format, img_dir)!=None) for format in rgx_formats]):
            if dest_format in self.formats:
                if dest_format in img_dir[-5:]:
                    print("UserWarning: This image file is already {0} format".format(dest_format))
                else:
                    img = Image.open(img_dir)
                    img.save()
            else:
                raise ValueError("Invalid image format, use '.png', '.jpeg', '.jpg' file formats")    
        else:
            raise ValueError("Invalid image format, use '.png', '.jpeg', '.jpg' file formats")

    ''' DEEP LEARNING METHODS FOR COMPUTER VISION '''
    # Load the specified file as a tf.image and preprocess it (see valid image formats)
    # See tf.keras.preprocessing.image_dataset_from_directory (https://keras.io/api/preprocessing/image/)
    def get_imgs_from_directory_as_numpy_array(self, dir, ext, size=None, scl=None, version=2):
        if ext in self.formats:
            rgx_for_img = dir + "/*." + ext
            img_list = glob.glob(rgx_for_img)

            # Scale image
            # scl_img = lambda img: ((img.astype('float32'))/255.0)

            if version < 0:
                # Function V0.1 (Use version 1 if any incompatibility on CUDA GPU version and the tf)
                # Cleanup called error on virtual machine
                if size is None:
                    img_arr = np.array([np.array(Image.open(img)) for img in img_list])
                else:
                    img_arr = np.array([np.array(Image.open(img).resize(size)) for img in img_list])

                img_arr = img_arr.astype('float32')
                if scl == '8bit': img_arr /= 255.0
                    
                return img_arr
            elif version == 1:
                def process_images(filename):
                    img = keras.preprocessing.image.load_img(filename, color_mode="rgb", target_size=size)
                    img_arr = keras.preprocessing.image.img_to_array(img)
                    img_arr = np.array(img_arr).astype('float32')  
                    if scl == '8bit': img_arr /= 255.0
                    return img_arr
                
                return np.array(list(map(process_images, img_list)))
                # https://keras.io/api/preprocessing/image/
                # https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
            else:
                def process_images(filename):
                    image_string = tf.io.read_file(filename)
                    # Decode a JPEG-encoded image to a uint8 tensor
                    # image = tf.image.decode_image(image_string, channels=3)  
                    image = tf.image.decode_image(image_string, channels=0)    
                    # channels: An optional int. Defaults to 0. Number of color channels for the decoded image
                    # Detects whether an image is a BMP, GIF, JPEG, or PNG, and performs the appropriate operation to convert 
                    # the input bytes string into a Tensor of type dtype.
                    image = tf.image.convert_image_dtype(image, tf.float32)
                    image = image.numpy()
                    if size != None:
                        image = tf.image.resize(image, size)
                    if scl == '8bit':
                        image /= 255.0
                    return image

                return list(map(process_images, img_list))
        else:
            raise ValueError("Invalid image format, use '.png', '.jpeg', '.jpg' file formats")

    # This method prepares a data array as a keras input
    # Example of input_size parameter usage: (28,28,1) -> 28x28 pixels gray scale images, (28,28,3) 28x28 pixels RGB images
    # Example of np_array parameter usage: np_array.shape = (60000, 28, 28) OR (60000, 784)
    def prepare_imgs_as_keras_input(self, np_array, input_size):
        w, h, d = input_size
        return np_array.reshape(np_array.shape[0], w, h, d)

    # Visualize triplets from the batches
    def show_triplet_images(self, anchor, positive, negative, size=(10,10), row=3):
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

    ''' CONVENTIONAL CLASSIFICATION ALGORITHMS FOR COMPUTER VISION '''
    def get_misclassified_indexes(self, y_test, y_pred):
        missed = []
        missed = np.where(y_test[y_test != y_pred])
        return missed[0]

    # This methods prints all evaluation parameters for classification models
    def print_eval_parameters(self, model, y_test, y_pred, labels):
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

    # This methods prints Grid Search Results for given search algorithm
    def print_grid_search_results(self, search):
        print("==== Grid Search Results ====")
        print("best_estimator: ", search.best_estimator_)
        print("best_params:    ", search.best_params_)
        print("best_score:      {:.3f}".format(search.best_score_))

    # This method plots Confusion matrix for classification models with given test dataset and prediction result array
    def show_confusion_matrix(self, y_test, y_pred, labels, w_h=(12, 7)):
        confMatrix = confusion_matrix(y_test, y_pred)
        dispConfMatrix = ConfusionMatrixDisplay(confMatrix, display_labels=labels)
        dispConfMatrix.plot()
        fig = plt.gcf()
        w, h = w_h
        fig.set_size_inches(w, h)

    # This method prints predicted and actual labels and shows actual image
    def show_prediction_result(self, x_test, y_pred, y_test, labels, n_img=None):
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

    def plot_randomly_img_predictions(self, img_arr, y_test, y_pred, labels, num_item=2, fig_size=[20,10]):
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


def main():
    prepare_dataset_for_mask_rcnn('/Users/atalaypab/Downloads/package_quality_check/intact/top')

if __name__ == "__main__":
    main()    