import tensorflow as tf
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, recall_score
from sklearn.metrics import precision_score, f1_score, confusion_matrix
from utils import utils

def prepare_evaluation_report(images, gen_images, gnd_images, exp_dict, exp_no, perceptual_input=None, perp_layer_coeffs=[1.0, 0.1, 0.5, 1.0, 0.5], bin_mask_thr=0.5, loss="MAE", channel_diff="mean", eps=1e-6, file="evaluation_report.txt"):
    '''
    Evaluation report method for Anomaly Detection tasks
    images: Input images
    gen_images: Images generated from AD model (reconstructed images)
    gnd_images: Ground truth images
    exp_dict: Experiment dictionary (All experiment specific configurations should be stored here)
    exp_no: Experiment no (1,2,..)
    perceptual_input: Perceptual layer outputs (VGG19, VGG16,...)
    perp_layer_coeffs: Perceptual layers will have the specified coefficients respectively
    bin_mask_thr: Binary loss diff mask threshold (for precision, recall and F1 score calculation)
    loss: Loss (pixel-wise difference) method for anomaly scoring
    channel_diff: To generate heatmap (W, H) from multilayer Loss diff (W, H, C) (max, mean, min, logic_or)
    eps: To avoiding zero-division while MAE+RAT calculation
    file: file save directory
    '''
    mean_iou = []
    mean_precision = []
    mean_recall= []
    mean_f1 = []
    mean_auc = []
    report_txt = ""

    # Convert ground truth image to the binary format
    if len(gnd_images.shape) == 4 and gnd_images.shape[-1] == 3:
        gnd_images = gnd_images[:,:,:,0]  # Get only one channel content
    elif len(gnd_images.shape) < 4:
        raise ValueError('Make sure that your dimension is 4d (batch_size, width, height, channels)')

    assert ((loss=="MAE") or (loss=="MSE") or (loss=="MAE+RAT") or (loss=="MSE+RAT"))

    # Compute Loss diff between test and generated images (Use MAE or MSE)
    if "MAE" in loss:
        loss_diff = tf.abs(images - gen_images)       # MAE
    else:
        loss_diff = tf.square(images - gen_images)    # MSE

    # Rational Loss diff
    # Loss Diff = (Original_Image - Generated_Image) / (Original_Image + Generated_Image + eps)
    if "RAT" in loss:
        # Overlay test image and the generated image
        overlayed_image = gen_images + images + eps
        # Compute rational loss diff
        loss_diff = (loss_diff / overlayed_image)

    # RGB to gray-scale
    assert ((channel_diff=="sum") or (channel_diff=="max") or (channel_diff=="mean"))
    if channel_diff == "sum":
        # Sum of the differences across channels
        loss_diff = tf.reduce_sum(loss_diff, axis=-1)
    elif channel_diff == "max":
        # Maximum of the differences across channels
        loss_diff = tf.reduce_max(loss_diff, axis=-1)
    elif channel_diff == "mean":
        # Mean of the differences across channels
        loss_diff = tf.reduce_mean(loss_diff, axis=-1)
    else:  # logic_or
        # Combine the results along the channel axis - Computes tf.math.logical_or of elements across dimensions of a tensor
        loss_diff = tf.reduce_any((loss_diff > bin_mask_thr), axis=-1)

    # Record all evaluation metrics
    with open(file, "a") as eval_report:
        report_txt = f"====== Experiment {exp_no} Evaluation Report ======\n"
        report_txt += "=== Configs ===\n"
        utils.write_dict_to_file(exp_dict, file, head=report_txt)
        report_txt = "\n=== Evaluation ===\n"
        report_txt += f"Number of test images: {len(gnd_images)}\n"
        report_txt += f"Binary mask Threshold: {bin_mask_thr}\n"
        report_txt += f"Loss difference computation method: {loss}\n"
        report_txt += f"Channel value pooling method: {channel_diff}\n\n"

        for ix, (gnd_image, loss_diff_image) in enumerate(zip(gnd_images, loss_diff)):
            # (Recons. + Perceptual) Loss Approach
            if perceptual_input != None:
                perceptual_out_sample_image = tf.expand_dims(loss_diff_image, axis=0)
                perceptual_out_sample_image = tf.concat((perceptual_out_sample_image, perceptual_input[:,ix,:,:]), axis=0)
                assert (len(perceptual_out_sample_image) == len(perp_layer_coeffs))
                loss_diff_mask = perceptual_out_sample_image[0]*perp_layer_coeffs[0]
                for coeff_ix in range(1,len(perp_layer_coeffs)):
                    loss_diff_mask += perceptual_out_sample_image[coeff_ix]*perp_layer_coeffs[coeff_ix]
            else:
                loss_diff_mask = loss_diff_image  # without perceptual layer contributions

            # Apply Min-max scaling to the Loss Diff mask image
            loss_diff_mask = utils.minmax_scale_tf(loss_diff_mask)

            # Flatten ground truth image
            gnd_image_flat = (tf.cast(tf.reshape(gnd_image, [-1]), dtype=tf.int32)).numpy()
            # Convert ground truth image to the binary mask
            gnd_image_flat_binary = gnd_image_flat.astype(bool)

            # Flatten loss diff image
            loss_diff_mask_flat = (tf.reshape(loss_diff_mask, [-1])).numpy()  # Do not convert dtype to the int32, it represents the logits
            # Convert loss diff image to the binary mask
            loss_diff_mask_flat_bin = (loss_diff_mask_flat > bin_mask_thr).astype(bool)

            # Compute confusion matrix, get TN, FP, FN, TP rates
            tn, fp, fn, tp = confusion_matrix(gnd_image_flat_binary, loss_diff_mask_flat_bin).ravel()
            # Compute px-wise IoU
            px_IoU = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0

            # Append evaluation results to the experiment log
            mean_iou.append(px_IoU)
            mean_recall.append(recall_score(gnd_image_flat_binary, loss_diff_mask_flat_bin))
            mean_precision.append(precision_score(gnd_image_flat_binary, loss_diff_mask_flat_bin))
            mean_f1.append(f1_score(gnd_image_flat_binary, loss_diff_mask_flat_bin))
            mean_auc.append(roc_auc_score(gnd_image_flat, tf.reshape(loss_diff_mask, [-1]).numpy()))
            #precision, recall, _ = precision_recall_curve(gnd_image_flat_binary, loss_diff_mask_flat)
            #mean_auc.append(auc(recall, precision))

            report_txt += (f"{ix}.Test Image\n")
            report_txt += (f"Mean IoU = {mean_iou[ix]:.5f}\n")
            report_txt += (f"Pixel-wise Recall = {mean_recall[ix]:.5f}\n")
            report_txt += (f"Pixel-wise Precision = {mean_precision[ix]:.5f}\n")
            report_txt += (f"Pixel-wise F1 score = {mean_f1[ix]:.5f}\n")
            report_txt += (f"AUC score = {mean_auc[ix]:.5f}\n\n")

        report_txt += "========================\n\n"
        report_txt += (f"Mean IoU: {sum(mean_iou) / len(mean_iou):.5f}\n")
        report_txt += (f"Mean Recall: {sum(mean_recall) / len(mean_recall):.5f}\n")
        report_txt += (f"Mean Precision: {sum(mean_precision) / len(mean_precision):.5f}\n")
        report_txt += (f"Mean F1 score: {sum(mean_f1) / len(mean_f1):.5f}\n")
        report_txt += (f"Mean AUC score: {sum(mean_auc) / len(mean_auc):.5f}")
        eval_report.write(report_txt)

    print(f"Binary mask Threshold: {bin_mask_thr}")
    print(f"Loss difference computation method: {loss}")
    print(f"Channel value pooling method: {channel_diff}")
    print(f"Mean IoU: {sum(mean_iou) / len(mean_iou):.5f}")
    print(f"Mean Recall: {sum(mean_recall) / len(mean_recall):.5f}")
    print(f"Mean Precision: {sum(mean_precision) / len(mean_precision):.5f}")
    print(f"Mean F1 score: {sum(mean_f1) / len(mean_f1):.5f}")
    print(f"Mean AUC score: {sum(mean_auc) / len(mean_auc):.5f}")