![issue_badge](https://img.shields.io/badge/window-10-blue)
![issue_badge](https://img.shields.io/badge/python-3.6-blue)
![issue_badge](https://img.shields.io/badge/tensorflow-2.3-blue)
![issue_badge](https://img.shields.io/badge/multi--gpu-falied-red)

# ObjectDetection

# introduction
  
  It's a object detection project based on Coco Dataset.
  
# Requirments

  pip install - r requirements.txt
  
# Training

  python main.py --data_dir dataset\annotations --image_dir dataset\train2017 --val_image_dir dataset\val2017 --train_mode --model_name yolov3
  
# Demo
  
  python demo.py

# References
  
  * [YOLOv3](https://arxiv.org/pdf/1804.02767)
  * [YOLOv4](https://arxiv.org/abs/2004.10934)
  * [AlexeyAB Darknet](https://github.com/AlexeyAB/darknet)
  
   #### My project, I referenced the contents of this place.
  
  * [YOLOv3 tensorflow2](https://github.com/zzh8829/yolov3-tf2)
  * [YOLOv4 tensorflow2](https://github.com/hunglc007/tensorflow-yolov4-tflite)
  
