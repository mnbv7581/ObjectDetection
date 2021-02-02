import os
import json
import numpy as np
import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from utils import transform_targets, preprocess_image , yolo_anchors, yolo_anchor_masks
class COCODataset(Sequence):

    IMAGE_WIDTH = 416
    IMAGE_HEIGHT = 416
    IMAGE_CHANNEL = 3
    def __init__(self,dataroot, imagedir,jsonfile, batch_size,shuffle=False, *args, **kwargs):
        super(COCODataset, self).__init__(*args, **kwargs)
        self.dataroot = dataroot
        self.imagedir = imagedir
        self.jsonfile = jsonfile
        self.batch_size = batch_size
        self.shuffle=shuffle
        self.imgs = dict()
        self.annos = dict()
        self.categories = dict()
        with open(os.path.join(dataroot,"annotations/"+self.jsonfile), "r") as json_file:
            json_info = json.load(json_file)

            for category in tqdm.tqdm(json_info["categories"]):
                self.categories[category["name"]] = category["id"]

            for img in tqdm.tqdm(json_info["images"]):
                self.imgs[img["id"]] = os.path.join(imagedir,img["file_name"])
                self.annos[img["id"]] = []

            self.instances_max = 0
            for annotation in tqdm.tqdm(json_info["annotations"]):
                stx = annotation["bbox"][0]
                sty = annotation["bbox"][1]
                w = annotation["bbox"][2]
                h = annotation["bbox"][3]
                endx = w - stx
                endy = h - sty
                bbox = np.asarray([stx,sty,endx,endy]+[annotation["category_id"]]) 
                self.annos[annotation["image_id"]].append(bbox)
                if self.instances_max < len(self.annos[annotation["image_id"]]):
                    self.instances_max = len(self.annos[annotation["image_id"]])
            
    def __getitem__(self, index):
        image_ids = list(self.annos.keys())[index*self.batch_size:(index+1)*self.batch_size]
        
        batch_images = np.zeros((self.batch_size,self.IMAGE_HEIGHT,self.IMAGE_WIDTH,self.IMAGE_CHANNEL))
        batch_annotations = np.zeros((self.batch_size,self.instances_max,5))
        for n, image_id in enumerate(image_ids):
            image = tf.image.decode_jpeg(open(self.imgs[image_id], 'rb').read(), channels=3)
            image = preprocess_image(image,self.IMAGE_WIDTH)
            #image = tf.expand_dims(image, axis=0)
            # image = Image.open(self.imgs[image_id]).resize((self.IMAGE_WIDTH,self.IMAGE_HEIGHT))
            # image = np.array(image) / 255.0
            #image.reshape(image.shape[0], image.shape[1], self.IMAGE_CHANNEL)
            batch_images[n, :, :, :] = image

        for n ,image_id in enumerate(image_ids):
            for m, anno in enumerate(self.annos[image_id]):
                batch_annotations[n,m,:] = anno

       
        batch_images        =   tf.convert_to_tensor(batch_images, tf.float32)         
        batch_annotations   =   tf.convert_to_tensor(batch_annotations, tf.float32)            
        batch_annotations   =   transform_targets(batch_annotations, yolo_anchors, yolo_anchor_masks,self.IMAGE_WIDTH) 
        return batch_images, batch_annotations
    
    def __len__(self):
        return int(len(self.imgs)/self.batch_size)