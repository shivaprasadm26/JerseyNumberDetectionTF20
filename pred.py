# -*- coding: utf-8 -*-

import tensorflow as tf
import argparse
import cv2
import matplotlib.pyplot as plt

from yolo.utils.box import visualize_boxes
from yolo.config import ConfigParser
import time


argparser = argparse.ArgumentParser(
    description='test yolov3 network with coco weights')

argparser.add_argument(
    '-c',
    '--config',
    default="configs/svhn.json",
    help='config file')

argparser.add_argument(
    '-i',
    '--image',
    default="tests/dataset/svhn/imgs/BsihopAmat00001_out.jpg",
    help='path to image file')


if __name__ == '__main__':
    args = argparser.parse_args()
    image_path   = args.image
    
    # 1. create yolo model & load weights
    config_parser = ConfigParser(args.config)
    model = config_parser.create_model(skip_detect_layer=False)
    detector = config_parser.create_detector(model)
    
    import os
    
    files = os.listdir("C:/Shiva/PP/tf2-eager-yolo3-master/tf2-eager-yolo3-master/tests/dataset/svhn/testimgs")
    files = os.listdir("C:/Shiva/PP/tf2-eager-yolo3-master/tf2-eager-yolo3-master/tests/dataset/svhn/testimgs")
    for file in files:
        filePath = "tests/dataset/svhn/testimgs/"
        filePath = filePath + file
        # 2. Load image
        image = cv2.imread(filePath)
        image = image[:,:,::-1]
        
        # 3. Run detection
        boxes, labels, probs = detector.detect(image, 0.5)
    
        # 4. draw detected boxes
        visualize_boxes(image, boxes, labels, probs, config_parser.get_labels())

        # 5. plot    
        plt.imshow(image)
        plt.show()
        #.pause(2)
        #plt.close()

 


