import cv2
import numpy as np
import tensorflow as tf
from datasets.coco import COCODataset
from absl import logging
from model.Yolov3 import YoloV3
from model.Yolov4 import YoloV4
#from utils import class_names ,  transform_targets, preprocess_image , yolo_anchors, yolo_anchor_masks , load_fake_dataset
from utils import class_names ,load_darknet_weights
from training.train import train
from training.evaluation import COCOEval
import time
import argparse

weightsyolov3 = 'yolov3.weights'

def main(args):
    data_dir = args.data_dir
    annotation = args.annotation
    image_dir = args.image_dir
    val_annotation = args.val_annotation
    val_image_dir = args.val_image_dir
    epochs = args.epochs
    batch_size = args.batch_size
    input_size = args.input_size
    pretrained = args.pretrained
    dataset_name = args.dataset_name
    model_name = args.model_name
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_visible_devices(gpu, 'GPU')
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    if args.train_mode:
        
       train_dataset = COCODataset(data_dir,image_dir,annotation,batch_size)
       val_dataset = COCODataset(data_dir,val_image_dir,val_annotation,batch_size)
       
       categories_len = len(val_dataset.categories.values())
       
       if model_name=="yolov3":
           yolo_model = YoloV3(size=input_size,classes=categories_len,training=True)

           if pretrained:
              yolo_model.load_weights("checkpoints/yolov3.tf")

       elif model_name=="yolov4":
           yolo_model = YoloV4(size=input_size,classes=categories_len,training=True)
          
       train(epochs,yolo_model,train_dataset,val_dataset, classes = categories_len,dataset_name=dataset_name)
    else:
        val_dataset = COCODataset(data_dir,val_image_dir,val_annotation,batch_size)
        eval = COCOEval("result.json",val_dataset)

if __name__=="__main__":
    #load_fake_dataset()
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epochs', default = 5, type=int)
    parser.add_argument('--input_size', default = 416, type=int)
    parser.add_argument('--batch_size', default = 4, type=int)
    parser.add_argument('--data_dir',default='E:/dataset/ObjectDetection/annotations/') #coco or voc
    parser.add_argument('--annotation',default='instances_train2017.json')
    parser.add_argument('--image_dir', default='E:/dataset/ObjectDetection/train2017/')
    parser.add_argument('--val_annotation',default='instances_val2017.json')
    parser.add_argument('--val_image_dir', default='E:/dataset/ObjectDetection/val2017/')
    parser.add_argument('--lr_rate', default=1e-5, type=float)
    parser.add_argument('--cutmix', action='store_true')
    parser.add_argument('--model_path', default = "")
    parser.add_argument('--train_mode', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--model_name',default="yolov3")
    parser.add_argument('--dataset_name',default="coco")

    main(parser.parse_args())

