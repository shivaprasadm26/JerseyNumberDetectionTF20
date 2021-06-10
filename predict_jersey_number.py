# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 11:40:45 2020

@author: user
"""
import cv2
import numpy as np
import ntpath
import os
import yolo
from sklearn.cluster import AgglomerativeClustering
import tensorflow as tf
import argparse
import cv2
import matplotlib.pyplot as plt
from yolo.config import ConfigParser
from imageai.Detection import ObjectDetection


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#tf.logging.set_verbosity(tf.logging.ERROR)
cluster = AgglomerativeClustering(n_clusters=None, distance_threshold= 0.5, affinity='euclidean', linkage='ward')

threshold = 0.5
argparser = argparse.ArgumentParser(
    description='test yolov3 network with coco weights')

argparser.add_argument(
    '-c',
    '--config',
    default="configs/svhn.json",
    help='config file')


save_img = show_img = False

#Person detection model
def load_objdetection():
    detector = ObjectDetection()
    custom = detector.CustomObjects(person=True)
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(os.path.join(yolo.PROJECT_ROOT, "yolo.h5"))
    detector.loadModel()
    print("detector loaded")
    return  detector, custom



#load number detection weights
def load_numberDetection():
    args = argparser.parse_args()
    config_parser = ConfigParser(args.config)
    model = config_parser.create_model(skip_detect_layer=False)
    detectorNum = config_parser.create_detector(model)
    return detectorNum

def get_digits_boxes_img(image,DETECTOR,DETECTORNUM,CUSTOM):
    im_size = np.shape(image)
    #boxes, probs = yolo.predict(image, float(threshold))
    boxes, labels, probs = DETECTORNUM.detect(image, 0.5)
    #labels = np.argmax(probs, axis=1) if len(probs) > 0 else [] 
    #probs = np.max(probs,axis=1) if len(probs) > 0 else []

    # 4. save detection result

    centers = []
    
    for count in range(len(boxes)):        
        box=boxes[count]        
        x=(((box[0] + box[2]/2)/im_size[0]))
        y=(((box[1] + box[3]/2)/im_size[1]))
        centers.append([x,y])
    centers=np.array(centers)
    jersey_numbers = []

    if len(centers) > 1:
        cluster.fit_predict(centers)
        cluster_labels = cluster.labels_
        clusters = np.unique(cluster_labels)
        
        conf_score=[]
        for c_id in clusters:
            g_centers = centers[list(cluster_labels==c_id)]
            g_labels = labels[list(cluster_labels==c_id)]
            g_probs = probs[list(cluster_labels==c_id)]
            center_x = [center[0] for center in g_centers]
                
            g_labels=g_labels[np.argsort(center_x)]  
            number = int((''.join(str(label) for label in g_labels)))    
            jersey_numbers.append(number)
            conf_score.append(np.mean(g_probs))
    else:
        jersey_numbers = list(labels)
        conf_score = [np.mean(probs)]
    print("{}-boxes are detected".format(len(boxes)))
    print("jersey numbers detected are {}".format(jersey_numbers))



    return jersey_numbers,conf_score

def detect_person_jersey_no(img_path,DETECTOR,DETECTORNUM,CUSTOM):
    print(img_path)
    img = cv2.imread(img_path)
    detections = DETECTOR.detectCustomObjectsFromImage(custom_objects=CUSTOM, input_image=img_path, output_image_path="out.jpg", minimum_percentage_probability=70, extract_detected_objects=False)
    
    jersey_numbers=conf_score=[]
    detNo = 0
    for det in detections:
        bb = det['box_points']
        cropped_im = img[bb[1]:bb[3],bb[0]:bb[2]]
        save_img = show_img = False
        if save_img:
            output_path = os.path.join(ntpath.dirname(img_path) + '/' + (img_path.split("/")[-1]).split('.')[0] +'detNo_' + str(detNo) + '_out.jpg')
            detNo+=1
            cv2.imwrite(output_path, cropped_im)
        if show_img:
            cv2.imshow("jersey number detected image",cropped_im)           
            cv2.waitKey(5000)
            cv2.destroyAllWindows()
        out = get_digits_boxes_img(cropped_im,DETECTOR,DETECTORNUM,CUSTOM)

        #print(out)
        if len(out[0]):
            #print(out)
            jersey_numbers.extend(out[0])
            conf_score.extend(out[1])

    return jersey_numbers,conf_score
            


#img_path = 'D:\\Projects\\RetrieveByJerseyNumber\\data\\person_cropped_set_14_15_16_labelled\\images\\person (49).jpg'
#jersey_numbers,conf_score,labels, boxes= get_digits_boxes(img_path,show_img=False,save_img=False)


#import glob
#
#filenames= glob.glob("D:\Projects\RetrieveByJerseyNumber\data\person_cropped_set_14_15_16_labelled\images\*.jpg")
#for img_path in filenames:
#    jersey_numbers,conf_score,labels, boxes= get_digits_boxes(img_path,show_img=True,save_img=True)

#np.linalg.norm(np.array(centers[0])-np.array(centers[1]))




    


