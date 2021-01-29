import os
import json
import numpy as np
import tqdm
from PIL import Image
import tensorflow as tf
from tensorflow.keras.utils import Sequence
import matplotlib.pyplot as plt


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
                bbox = np.asarray(annotation["bbox"]+[annotation["category_id"]]) 
                self.annos[annotation["image_id"]].append(bbox)
                if self.instances_max < len(self.annos[annotation["image_id"]]):
                    self.instances_max = len(self.annos[annotation["image_id"]])
            
    def __getitem__(self, index):
        image_ids = list(self.annos.keys())[index*self.batch_size:(index+1)*self.batch_size]
        
        batch_images = np.zeros((self.batch_size,self.IMAGE_HEIGHT,self.IMAGE_WIDTH,self.IMAGE_CHANNEL))
        batch_annotations = np.zeros((self.batch_size,self.instances_max,5))
        for n, image_id in enumerate(image_ids):
            image = Image.open(self.imgs[image_id]).resize((self.IMAGE_WIDTH,self.IMAGE_HEIGHT))
            image = np.array(image) / 255.0
            image.reshape(image.shape[0], image.shape[1], self.IMAGE_CHANNEL)
            batch_images[n, :, :, :] = image

        for n ,image_id in enumerate(image_ids):
            for m, anno in enumerate(self.annos[image_id]):
                batch_annotations[n,m,:] = anno
                
        return batch_images, batch_annotations
    
    def __len__(self):
        return len(self.imgs)/self.batch_size