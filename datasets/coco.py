import os
import json
import numpy as np
import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import Sequence
from utils import transform_targets, preprocess_image , yolo_anchors, yolo_anchor_masks, load_fake_dataset

class COCODataset(Sequence):

    IMAGE_WIDTH = 416
    IMAGE_HEIGHT = 416
    IMAGE_CHANNEL = 3
    MAX_BBOXES = 100
    def __init__(self,dataroot, imagedir,jsonfile, batch_size,shuffle=False, *args, **kwargs):
        super(COCODataset, self).__init__(*args, **kwargs)
        self.dataroot = dataroot
        self.imagedir = imagedir
        self.jsonfile = jsonfile
        self.batch_size = batch_size
        self.shuffle=shuffle
        self.imgs = dict()
        self.imgs_info = dict()
        self.annos = dict()
        self.categories = dict()
        with open(os.path.join(dataroot,self.jsonfile), "r") as json_file:
            json_info = json.load(json_file)

            for category in tqdm.tqdm(json_info["categories"]):
                self.categories[category["name"]] = category["id"]

            for img in tqdm.tqdm(json_info["images"]):
                self.imgs[img["id"]] = os.path.join(imagedir,img["file_name"])
                self.imgs_info[img["id"]] = {"width":img["width"],"height":img["height"]}
                self.annos[img["id"]] = []

            self.instances_max = 0
            #for annotation in tqdm.tqdm(json_info["annotations"]):
            for annotation in json_info["annotations"]:
                img_width  =  self.imgs_info[annotation["image_id"]]["width"]
                img_height  =  self.imgs_info[annotation["image_id"]]["height"]

                stx =  annotation["bbox"][0]
                sty =  annotation["bbox"][1]
                endx = annotation["bbox"][2] + stx
                endy = annotation["bbox"][3] + sty

                # stx = stx/float(img_width)
                # sty = sty/float(img_height)
                # endx = endx/float(img_width)
                # endy = endy/float(img_height)

                bbox = np.asarray([stx,sty,endx,endy]+[int(annotation["category_id"])],dtype=np.float32)
                self.annos[annotation["image_id"]].append(bbox)
                if self.instances_max < len(self.annos[annotation["image_id"]]):
                    self.instances_max = len(self.annos[annotation["image_id"]])
    def __getitem__(self, index):
        image_ids = list(self.annos.keys())[index*self.batch_size:(index+1)*self.batch_size]
        
        batch_images = np.zeros((self.batch_size,self.IMAGE_HEIGHT,self.IMAGE_WIDTH,self.IMAGE_CHANNEL))
        batch_annotations = np.zeros((self.batch_size,self.MAX_BBOXES,5))
        for n, image_id in enumerate(image_ids):
            image = tf.image.decode_jpeg(open(self.imgs[image_id], 'rb').read(), channels=3)
            image = tf.cast(image,dtype=tf.float32)
            image = preprocess_image(image,self.IMAGE_WIDTH)
            batch_images[n, :, :, :] = image
    
        for n ,image_id in enumerate(image_ids):
            for m, anno in enumerate(self.annos[image_id]):
                batch_annotations[n,m,:] = anno
                
        batch_images        =   tf.convert_to_tensor(batch_images, tf.float32)         
        batch_annotations   =   tf.convert_to_tensor(batch_annotations, tf.float32)    
        #colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        #print(batch_annotations[0,:,0:4])
        # images_with_boxes =  tf.image.draw_bounding_boxes(batch_images[:,:,:,:], batch_annotations[:,:,0:4],colors)
        # for i in range(self.batch_size):
        #     plt.imshow(images_with_boxes[i,:,:,:])
        #     plt.show()
        batch_annotations   =   transform_targets(batch_annotations, yolo_anchors, yolo_anchor_masks,self.IMAGE_WIDTH)
        return batch_images, batch_annotations
    
    def __len__(self):
        return int(len(self.imgs)/self.batch_size)