import os
import tensorflow as tf
import numpy as np
import time
import logging
import json
from collections import Counter
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os, sys, zipfile
import urllib.request
import shutil
import numpy as np
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)



class COCOEval():
    def __init__(self,resfile,datasets):
        self.resfile = resfile
        self.datasets = datasets
        self.annfile = datasets.getCOCOAnnotationPath()
        self.COCOJson = datasets.getCOCOJson()
        self.COCOImageInfo = datasets.getCOCOImageInfo()
        self.COCOResJson = []
    def append(self,image_ids,categori_ids,bboxes,scores,nums):

        for idx, img_id in enumerate(image_ids):
            category, bbox, score =categori_ids[idx], bboxes[idx], scores[idx]
            img_width, img_height = self.COCOImageInfo[img_id]["width"], self.COCOImageInfo[img_id]["height"]
            for num in range(nums[idx]):
                x = float(bbox[num][0]) * img_width
                y = float(bbox[num][1]) * img_height
                w = (float(bbox[num][2]) * img_width) - x
                h = (float(bbox[num][3]) * img_height) - y

                result = {
                    "image_id":img_id,
                    "category_id":int(category[num]),
                    "bbox": [x,y,w,h],
                    "score": float(score[num])
                }
                self.COCOResJson.append(result)

    def save(self):
        with open(os.path.join(self.resfile), "w") as json_file:
            json.dump(self.COCOResJson, json_file)

    def Calculating(self):
        cocoGt=COCO(self.annfile)
        cocoDt=cocoGt.loadRes(self.resfile)

        imgIds=sorted(cocoGt.getImgIds())
        #imgCatIds = sorted(cocoGt.getCatIds())
        cocoEval = COCOeval(cocoGt,cocoDt,"bbox")     
        #cocoEval.params.areaRng = [[0 ** 2, 1e5 ** 2], [0 ** 2, 20 ** 2], [20 ** 2, 30 ** 2], [30 ** 2, 1e5 ** 2]]
        cocoEval.params.imgIds  = imgIds
        #cocoEval.params.catIds  = imgCatIds
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
