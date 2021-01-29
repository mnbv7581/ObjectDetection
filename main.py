import cv2
import numpy as np
import tensorflow as tf
from datasets.coco import COCODataset
from absl import logging
from model.Yolov3 import YoloV3
from utils import preprocess_image, transform_targets
from training.train import train

import argparse

class_names =  ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana","apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake","chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop",
    "mouse","remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator","book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]


yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),(59, 119), (116, 90), (156, 198), (373, 326)], np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])


def main(args):
    data_dir = args.data_dir
    annotation = args.annotation
    image_dir = args.image_dir
    epochs = args.epochs
    batch_size = args.batch_size
    
    if args.train_mode:
       train_dataset = COCODataset(image_dir,data_dir,annotation,batch_size)
       val_dataset = COCODataset(image_dir,data_dir,annotation,batch_size)
       train(epochs,train_dataset,val_dataset)
    else:
        pass

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epochs', default = 5)
    parser.add_argument('--input_size', default = 416)
    parser.add_argument('--batch_size', default = 416)
    parser.add_argument('--data_dir',default='E:/dataset/ObjectDetection/annotations', required=True) #coco or voc
    parser.add_argument('--annotation',default='captions_val2017.json', required=True)
    parser.add_argument('--image_dir', default='E:/dataset/ObjectDetection/annotations')
    parser.add_argument('--lr_rate', default=1e-5)
    parser.add_argument('--cutmix', action='store_true')
    parser.add_argument('--model_path', default = "")
    parser.add_argument('--train_mode', action='store_true')
    main(parser.parse_args())

    # iter = iter(inputs)
    # features, labels = iter.get_next()
    
    # labels = labels[tf.newaxis,...]
    # labels = transform_targets(labels, yolo_anchors, mask, size)
    # features = preprocess_image(features, size)
    # features = features[tf.newaxis,...]
    # avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
    # pred_loss = []
    # loss, _ = grad(yolo,features,yolo_anchors,labels)
    # total_loss = tf.reduce_sum(loss)
    # avg_loss.update_state(total_loss)
    # print(avg_loss.result().numpy())
