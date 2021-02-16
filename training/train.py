import os
import tensorflow as tf
import numpy as np
import time
from loss.loss import YoloLoss
#from absl import logging
import logging

yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),(59, 119), (116, 90), (156, 198), (373, 326)], np.float32) / 416

@tf.function
def grad(model,image,anchors,y_true,losses,optimizer, ignore_thresh=0.5):
    pred_loss = []
    with tf.GradientTape() as tape:
        pred = model(image,training=True)
        #regularization_loss = tf.reduce_sum(model.losses)
        for i, loss in enumerate(losses):
            pred_loss.append(loss(y_true[i], pred[i]))
        total_loss = tf.reduce_sum(pred_loss)
    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return total_loss
def eval(model,image,anchors,y_true,losses,ignore_thresh=0.5):
    pred_loss = []
    pred = model(image,training=True)
    regularization_loss = tf.reduce_sum(model.losses)
    for i, loss in enumerate(losses):
        pred_loss.append(loss(y_true[i], pred[i]))
    total_loss = tf.reduce_sum(pred_loss)
    return total_loss

def build_logging():
    # 로그 생성
    logger = logging.getLogger()
    # 로그의 출력 기준 설정
    logger.setLevel(logging.INFO)
    # log 출력 형식
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # log 출력
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # log를 파일에 출력
    file_handler = logging.FileHandler('train.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def train(epochs, model, train_dataset, val_dataset, classes = 80 ):
    '''
    Parameters
    ----------
    - epochs : integer, number of epochs to train the model.
    - model : model.
    - train_dataset : YOLO ground truth and image generator from training dataset.
    - val_dataset : YOLO ground truth and image generator from validation dataset.
    - classes : Number of Classes

    Returns
    -------
    '''
    num_epochs = epochs
    train_loss_history = []
    val_loss_history = []
    best_val_loss = 1e6

    #optimizer = tf.keras.optimizers.Adam(learning_rate=1e-12, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    metrics = tf.keras.metrics.Mean('loss', dtype=tf.float32)
    val_metrics = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
    losses = [YoloLoss(yolo_anchors[m], classes, 0.5) for m in yolo_anchor_masks]
    # summary_writer = tf.summary.create_file_writer(os.path.join('logs/', train_name), flush_millis=20000)
    # summary_writer.set_as_default()
    
    data_len = len(train_dataset)
    logging = build_logging()

    # for idx in range(100):
    #     train_x, train_y = train_dataset[idx]
    #     total_loss = grad(model,train_x,yolo_anchors,train_y,losses,optimizer)

    #     logging.info("Iter : {}/{}, losses : {}".format(
    #                 idx,
    #                 data_len,
    #                 total_loss))
    
    for epoch in range(num_epochs):
        epoch_loss = []
        epoch_val_loss = []
        epoch_val_sub_loss = []
        logging.info("{} Epoch Training Start".format(epoch))
        for batch_idx, (train_x,train_y) in enumerate(train_dataset):
            total_loss = grad(model,train_x,yolo_anchors,train_y,losses,optimizer)
            
            epoch_loss.append(total_loss)
            metrics.update_state(total_loss)
            avg_loss = metrics.result().numpy()

            if (batch_idx % 50) == 0 :
                logging.info("Iter : {}/{}, losses : {}".format(
                    batch_idx,
                    data_len,
                    avg_loss))
            
        train_loss_history.append(avg_loss)
        #tf.summary.scalar('loss', avg_loss, epoch)

        logging.info("Epoch : {}, train_loss : {}".format(
                epoch, avg_loss))

        logging.info("{} Epoch Evaluation Start".format(epoch))
        for batch_idx, (val_x,val_y) in enumerate(val_dataset):
            total_loss = eval(model,val_x,yolo_anchors,val_y,losses,optimizer)  
            val_metrics.update_state(total_loss)
            val_avg_loss = val_metrics.result().numpy()

            if (batch_idx % 50) == 0 :
                logging.info("Iter : {}/{}, losses : {}".format(
                    batch_idx,
                    data_len,
                    val_avg_loss))

        logging.info("epoch: {} , val: {}".format(
                epoch,
                val_metrics.result().numpy()))
        
        val_loss_history.append(val_avg_loss)


        if best_val_loss > val_avg_loss:
            logging.info("best model save")
            model.save_weights('checkpoint/checkpoint_best.tf')
            
        model.save_weights('checkpoint/checkpoint_{}ep.tf'.format(epoch))
        metrics.reset_states()
        val_metrics.reset_states()
        