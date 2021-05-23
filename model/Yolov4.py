
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Lambda
from utils import yolo_boxes, nonMaximumSuppression
from layers.Darknet import CSPDarknet53, DarknetConv, Upsample
from layers.Yolo import YoloConv, YoloOutput

yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),(59, 119), (116, 90), (156, 198), (373, 326)], np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

def YoloV4(size=None, channels=3, anchors=yolo_anchors,
            masks=yolo_anchor_masks, classes=80, training=False):
    x = inputs = Input([size, size, channels])
    
    #x_36, x_61, x = Darknet(name='yolo_darknet')(x)
    route_2, route_1, x = CSPDarknet53(name='yolo_darknet')(x)
    route = x

    # print(x) # (13, 13, 512)
    # print(route_2) # (26, 26, 512)
    # print(route_1) # (52, 52, 256)

    x = DarknetConv(x,256,1)
    x = Upsample(x)
    route_2 = DarknetConv(route_2,256,1)
    x = tf.concat([x, route_2], axis=-1)
    x = YoloConv(256, name='yolo_conv_0')(x) 

    x = DarknetConv(x,128,1)
    x = Upsample(x)
    route_1 = DarknetConv(x,128,1)
    x = tf.concat([x, route_1], axis=-1)
    x = YoloConv(128, name='yolo_conv_1')(x) 

    #Yolov4 output 1
    route_1 = x
    output_0 = YoloOutput(128, len(masks[2]), classes, name='yolo_output_0')(x)

    x = DarknetConv(x,256,3,strides=2)
    x = tf.concat([x, route_2], axis=-1)
    x = YoloConv(256, name='yolo_conv_2')(x) 
    
    #Yolov4 output 2
    route_2 = x
    output_1 = YoloOutput(256, len(masks[1]), classes, name='yolo_output_1')(x)

    x = DarknetConv(x,512,3,strides=2)
    x = tf.concat([x, route], axis=-1)
    x = YoloConv(512, name='yolo_conv_3')(x) 

    #Yolov4 output 3
    output_2 = YoloOutput(512, len(masks[0]), classes, name='yolo_output_2')(x)


    if training:
        return Model(inputs, (output_2, output_1, output_0), name='yolov4')

    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], classes),
                  name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], classes),
                  name='yolo_boxes_1')(output_1)
    boxes_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], classes),
                  name='yolo_boxes_2')(output_2)

    outputs = Lambda(lambda x: nonMaximumSuppression(x, anchors, masks, classes),
                  name='nonMaximumSuppression')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

    return Model(inputs, outputs, name='yolov4')