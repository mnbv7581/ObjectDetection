import cv2
import numpy as np
import tensorflow as tf
from datasets.coco import COCODataset
from absl import logging
from model.Yolov3 import YoloV3
from utils import preprocess_image, transform_targets
from training.train import grad
weightsyolov3 = 'yolov3.weights' # path to weights file
weights= 'checkpoints/yolov3.tf' # path to checkpoints file
size= 416             #resize images to\
checkpoints = 'checkpoints/yolov3.tf'
num_classes = 80      # number of classes in the model

YOLO_V3_LAYERS = [
    'yolo_darknet',
    'yolo_conv_0',
    'yolo_output_0',
    'yolo_conv_1',
    'yolo_output_1',
    'yolo_conv_2',
    'yolo_output_2',
]

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


def load_darknet_weights(model, weights_file):
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)
    layers = YOLO_V3_LAYERS

    for layer_name in layers:
        sub_model = model.get_layer(layer_name)
        for i, layer in enumerate(sub_model.layers):
            if not layer.name.startswith('conv2d'):
                continue
            batch_norm = None
            if i + 1 < len(sub_model.layers) and \
              sub_model.layers[i + 1].name.startswith('batch_norm'):
                batch_norm = sub_model.layers[i + 1]

            logging.info("{}/{} {}".format(
                sub_model.name, layer.name, 'bn' if batch_norm else 'bias'))

            filters = layer.filters
            size = layer.kernel_size[0]
            in_dim = layer.input_shape[-1]

            if batch_norm is None:
                conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)
            else:
                bn_weights = np.fromfile(
                    wf, dtype=np.float32, count=4 * filters)

                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]

            conv_shape = (filters, in_dim, size, size)
            conv_weights = np.fromfile(
                wf, dtype=np.float32, count=np.product(conv_shape))

            conv_weights = conv_weights.reshape(
                conv_shape).transpose([2, 3, 1, 0])

            if batch_norm is None:
                layer.set_weights([conv_weights, conv_bias])
            else:
                layer.set_weights([conv_weights])
                batch_norm.set_weights(bn_weights)

    assert len(wf.read()) == 0, 'failed to read weigts'
    wf.close()

def load_fake_dataset():
    x_train = tf.image.decode_jpeg(
        open('girl.png', 'rb').read(), channels=3)
    x_train = tf.expand_dims(x_train, axis=0)

    labels = [
        [0.18494931, 0.03049111, 0.9435849,  0.96302897, 0],
        [0.01586703, 0.35938117, 0.17582396, 0.6069674, 56],
        [0.09158827, 0.48252046, 0.26967454, 0.6403017, 67]
    ] + [[0, 0, 0, 0, 0]] * 5
    y_train = tf.convert_to_tensor(labels, tf.float32)
    y_train = tf.expand_dims(y_train, axis=0)

    return tf.data.Dataset.from_tensor_slices((x_train, y_train))


yolo_anchors = np.array([(10, 13), (16, 30), (33, 23), (30, 61), (62, 45),(59, 119), (116, 90), (156, 198), (373, 326)], np.float32) / 416
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])
if __name__=="__main__":
    mask = yolo_anchor_masks
    yolo = YoloV3(classes=num_classes,training=True)
    load_darknet_weights(yolo, weightsyolov3)
    yolo.save_weights(checkpoints)
    image = 'girl.png'     # path to input image

    img = tf.image.decode_image(open(image, 'rb').read(), channels=3)
    img = tf.expand_dims(img, 0)
    img = preprocess_image(img, size)

    inputs = load_fake_dataset()
    inputs.batch(1)

    # inputs = inputs.map(lambda x, y: (
    #     preprocess_image(x, size),
    #     transform_targets(y, yolo_anchors, mask, size)))

    iter = iter(inputs)
    features, labels = iter.get_next()
    
    labels = labels[tf.newaxis,...]
    labels = transform_targets(labels, yolo_anchors, mask, size)
    features = preprocess_image(features, size)
    features = features[tf.newaxis,...]
    avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
    pred_loss = []
    loss, _ = grad(yolo,features,yolo_anchors,labels)
    total_loss = tf.reduce_sum(loss)
    avg_loss.update_state(total_loss)
    print(avg_loss.result().numpy())
