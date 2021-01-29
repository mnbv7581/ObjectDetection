import os
import tensorflow as tf
import numpy as np
from loss.loss import YoloLoss
from absl import logging
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),(59, 119), (116, 90), (156, 198), (373, 326)], np.float32) / 416

def grad(model,image,anchors,y_true,classes=80, ignore_thresh=0.5):
    mask = yolo_anchor_masks

    losses = [YoloLoss(anchors[m], classes, ignore_thresh) for m in mask]

    pred_loss = []
    
    with tf.GradientTape() as tape:
        pred = model(image)
        for i, loss in enumerate(losses):
            pred_loss.append(loss(y_true[i], pred[i]))

    return pred_loss, tape.gradient(tf.reduce_sum(pred_loss), model.trainable_variables)

def train(epochs, model, train_dataset, val_dataset, steps_per_epoch_train, steps_per_epoch_val, train_name = 'train'):
    '''
    Train YOLO model for n epochs.
    Eval loss on training and validation dataset.
    Log training loss and validation loss for tensorboard.
    Save best weights during training (according to validation loss).

    Parameters
    ----------
    - epochs : integer, number of epochs to train the model.
    - model : YOLO model.
    - train_dataset : YOLO ground truth and image generator from training dataset.
    - val_dataset : YOLO ground truth and image generator from validation dataset.
    - steps_per_epoch_train : integer, number of batch to complete one epoch for train_dataset.
    - steps_per_epoch_val : integer, number of batch to complete one epoch for val_dataset.
    - train_name : string, training name used to log loss and save weights.
    
    Notes :
    - train_dataset and val_dataset generate YOLO ground truth tensors : detector_mask,
      matching_true_boxes, class_one_hot, true_boxes_grid. Shape of these tensors (batch size, tensor shape).
    - steps per epoch = number of images in dataset // batch size of dataset
    
    Returns
    -------
    - loss history : [train_loss_history, val_loss_history] : list of average loss for each epoch.
    '''
    num_epochs = epochs
    steps_per_epoch_train = steps_per_epoch_train
    steps_per_epoch_val = steps_per_epoch_val
    train_loss_history = []
    val_loss_history = []
    best_val_loss = 1e6

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    metrics = tf.keras.metrics.Mean('loss', dtype=tf.float32)
    summary_writer = tf.summary.create_file_writer(os.path.join('logs/', train_name), flush_millis=20000)
    summary_writer.set_as_default()

    for epoch in range(num_epochs):
        epoch_loss = []
        epoch_val_loss = []
        epoch_val_sub_loss = []

        for batch_idx, (train_x,train_y) in train_dataset:
            losses, gradients = grad(model,train_x,yolo_anchors,train_y)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            epoch_loss.append(losses)
        
        metrics.update_state(epoch_loss)
        avg_loss = metrics.result().numpy()
        train_loss_history.append(metrics.result().numpy())
        tf.summary.scalar('loss', avg_loss, epoch)

        logging.info("Epoch : {}, train_loss : {}".format(
                epoch, avg_loss))1

