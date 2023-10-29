import datetime
import tensorflow as tf

# TF Keras Callbacks

'''
tf.keras.callbacks is a module in TensorFlow that provides various functions that can be used to monitor and modify the behavior of the training process of a neural network. These callbacks can be passed as a list to the fit() method of a tf.keras.Model object, allowing you to customize the training process.

Some of the commonly used callbacks are:
* ModelCheckpoint: This callback saves the model after every epoch, or only when an improvement in performance is observed. This helps you to save the best performing model and avoid overfitting. The ModelCheckpoint callback gives you the ability to save your model, as a whole in the SavedModel format or the weights (patterns) only to a specified directory as it trains. This is helpful if you think your model is going to be training for a long time and you want to make backups of it as it trains. It also means if you think your model could benefit from being trained for longer, you can reload it from a specific checkpoint and continue training from there. For example, say you fit a feature extraction transfer learning model for 5 epochs and you check the training curves and see it was still improving and you want to see if fine-tuning for another 5 epochs could help, you can load the checkpoint, unfreeze some (or all) of the base model layers and then continue training.

* EarlyStopping: This callback stops the training process when the validation loss stops improving. This helps you to avoid overfitting and saves training time.

* TensorBoard: This callback writes a log for TensorBoard, which can be used to visualize the model training process, including metrics like loss and accuracy, and graph visualizations of the model architecture.

* ReduceLROnPlateau: This callback reduces the learning rate when the validation loss stops improving, which can help you to fine-tune the model.

* CSVLogger: This callback writes the training and validation metrics to a CSV file at the end of each epoch, which can be useful for monitoring and analyzing the performance of the model.

These are just a few examples of the many callbacks available in tf.keras.callbacks. You can also create your own custom callbacks by subclassing tf.keras.callbacks.Callback.
'''

# Create Model Checkpoint callback
# # Example usage
# model_history = model.fit(x_train, y_train, epochs=5, ...,
#                          callbacks=[create_model_checkpoint_callback(dir_name='tensorflow_hub', experimodelB0')])

# # Assigning Multiple callbacks
# model_history = model.fit(x_train, y_train, epochs=5, ...,
#                          callbacks=[create_model_checkpoint_callback(dir_name='tensorflow_hub', experimodelB0'),
#                          create_tensorboard_callback(dir_name='tensorflow_hub', experimodelB0')])
#
# # OR
#
# callback_list = [create_early_stopping_callback(), create_model_checkpoint_callback(dir_name='tensorflow_hub', experimodelB0'), create_tensorboard_callback(dir_name='tensorflow_hub', experimodelB0')]
# model_history = model.fit(x_train, y_train, epochs=5, ..., callbacks=callback_list)
def create_model_checkpoint_callback(path, experiment_name, verbose=1):
    # Get Datetime object containing current date and time
    date = str(datetime.now())
    date = (((date.replace("-", "_")).replace(":", "_")).replace(" ", "_")).split(".")[0]

    checkpoint_path = path + "/" + experiment_name + "/" + date + "/checkpoint.ckpt"

    # Create a ModelCheckpoint callback that saves the model's weights only
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True, # set to False to save the entire model
                                                         # If disk space is an issue, saving the weights only is faster and takes up less space than saving the whole model.
                                                         save_best_only=False,   # set to True to save only the best model instead of a model every epoch
                                                         monitor='val_loss',     # Use "loss" or "val_loss" to monitor the model's total loss.
                                                         save_freq='epoch',      # save every epoch
                                                         verbose=1)
    if verbose > 0:
      print(f"Model checkpoint will save to: {checkpoint_path}")
    
    return checkpoint_callback

# Create TensorBoard callback
# # Example usage
# model_history = model.fit(x_train, y_train, epochs=5, ...,
#                          callbacks=[create_tensorboard_callback(dir_name='tensorflow_hub', experimodelB0')])

# %load_ext tensorboard
# %tensorboard --logdir=tensorflow_hub

# # Assigning Multiple callbacks
# model_history = model.fit(x_train, y_train, epochs=5, ...,
#                          callbacks=[create_model_checkpoint_callback(dir_name='tensorflow_hub', experimodelB0'),
#                          create_tensorboard_callback(dir_name='tensorflow_hub', experimodelB0')])
#
# # OR
#
# callback_list = [create_early_stopping_callback(), create_model_checkpoint_callback(dir_name='tensorflow_hub', experimodelB0'), create_tensorboard_callback(dir_name='tensorflow_hub', experimodelB0')]
# model_history = model.fit(x_train, y_train, epochs=5, ..., callbacks=callback_list)
def create_tensorboard_callback(path, experiment_name, verbose=1):
    '''
    TensorBoard is a visualization tool provided with TensorFlow.
    This callback logs events for TensorBoard, including:
    
    * Metrics summary plots
    * Training graph visualization
    * Weight histograms
    * Sampled profiling
    '''
    # Get Datetime object containing current date and time
    date = str(datetime.now())
    date = (((date.replace("-", "_")).replace(":", "_")).replace(" ", "_")).split(".")[0]
    log_dir = path + "/" + experiment_name + "/" + date
    
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        write_graph=True,
        histogram_freq=0,
        update_freq='epoch',
    )
    '''
    update_freq: 'batch' or 'epoch' or integer. When using 'epoch', 
    writes the losses and metrics to TensorBoard after every epoch. If using an
    integer, let's say 1000, all metrics and losses (including custom ones
    added by Model.compile) will be logged to TensorBoard every 1000 batches.
    'batch' is a synonym for 1, meaning that they will be written every batch.
    '''
    
    if verbose > 0:
      print(f"TensorBoard log files will save to: {log_dir}")
    return tensorboard_callback

# Create Early Stopping callback
# # Example usage
# model_history = model.fit(x_train, y_train, epochs=5, ...,
#                          callbacks=[create_early_stopping_callback()])

# # Assigning Multiple callbacks
# model_history = model.fit(x_train, y_train, epochs=5, ...,
#                          callbacks=[create_model_checkpoint_callback(dir_name='tensorflow_hub', experimodelB0'),
#                          create_early_stopping_callback()])
#
# # OR
#
# callback_list = [create_early_stopping_callback(), create_model_checkpoint_callback(dir_name='tensorflow_hub', experimodelB0'), create_tensorboard_callback(dir_name='tensorflow_hub', experimodelB0')]
# model_history = model.fit(x_train, y_train, epochs=5, ..., callbacks=callback_list)

# # Alternatives
# earlystopping = callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True, verbose=2)
# earlystopping = callbacks.EarlyStopping(monitor='val_accuracy', mode='max', patience=5, restore_best_weights=True, verbose=2)
def create_early_stopping_callback(patience=10, verbose=1):
    # Create a EarlyStopping callback that stops the training process when the validation loss stops improving
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01,
                                                         patience=patience,
                                                         mode='auto',
                                                         restore_best_weights=True,
                                                         verbose=verbose)
    '''
    min_delta: Minimum change in the monitored quantity to qualify as an improvement,
    i.e. an absolute change of less than min_delta, will count as no improvement.

    patience: Number of epochs with no improvement after which training will be stopped.

    restore_best_weights: Whether to restore model weights from the epoch with the best
    value of the monitored quantity. If False, the model weights obtained at the last
    step of training are used. An epoch will be restored regardless of the performance
    relative to the baseline. If no epoch improves on baseline, training will run for
    patience epochs and restore weights from the best epoch in that set.
    '''

    return early_stopping_callback

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