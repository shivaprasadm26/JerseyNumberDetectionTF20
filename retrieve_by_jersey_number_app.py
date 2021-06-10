# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 14:36:56 2020

@author: user
"""
import glob
import os
import pickle
import shutil
import time

from flask import request
from flask import Flask

import argparse

import os
import glob
from predict_jersey_number import detect_person_jersey_no, load_objdetection, load_numberDetection
app = Flask(__name__)


@app.route('/')
def hello():
    return "welcome! this service is to retrieve player images by jersey number"

DETECTOR = False
DETECTORNUM = False
CUSTOM = False

def JerseyNumberRetrieve(query_jersey_number,SearchSetParentPath,DestPath,NumImages=10):
    '''
    function to get list of images containing players with jersey number same as query_jersey_number

    returns a list of length equal to minimum of number of detected images and NumImages. The list contains 
    path to the images in decreasing order of detection confidence.
    Each element of the list will be a tuple of the form (image_path, detection_confidence)    
    
    '''
    
    detections_file = os.path.join(SearchSetParentPath,"reference_detections.p")
    
    if not os.path.exists(detections_file):
        
        filenames= glob.glob(SearchSetParentPath+ "/*.jpg", recursive = False) 
        filenames.extend(glob.glob(SearchSetParentPath+ "/*.png", recursive = False))
        filenames.extend(glob.glob(SearchSetParentPath+ "/*.bmp", recursive = False))
        filenames.extend(glob.glob(SearchSetParentPath+ "/*.tif", recursive = False))
        filenames.extend(glob.glob(SearchSetParentPath+ "/*.tiff", recursive = False))
        filenames.extend(glob.glob(SearchSetParentPath+ "/*.JPEG", recursive = False))

        reference_detections = []
        for count in range(len(filenames)):
            
            image_path = filenames[count]
            image_name = (image_path.replace('\\','/')).split('/')[-1]
        
            detections = detect_person_jersey_no(filenames[count],DETECTOR,DETECTORNUM,CUSTOM)
            out_dict = {}
            out_dict['path'] = image_path
            out_dict['name'] = image_name
            out_dict['detections'] = detections
            reference_detections.append(out_dict)    
        
        pickle.dump(reference_detections, open(detections_file, "wb" ))
        print("Detections created successfully")
        print("==========>>")
    else:
        reference_detections= pickle.load(open(detections_file, "rb"))
        print("Loaded detections from file")
        print("==========>>")
    matched_files = []
    for detection in reference_detections:
        file_name = detection['path']
        jersey_numbers = detection['detections'][0]
        confidence = detection['detections'][1]
        for j_count in range(len(jersey_numbers)):
            jersey_number=jersey_numbers[j_count]
#             print("jersy numbers",jersey_number)
            if jersey_number == query_jersey_number:
                matched_files.append((file_name,confidence[j_count]))       
    
    matched_files.sort(key = lambda x:x[1], reverse = True)
    print("Found {} images with jersey number {}".format(len(matched_files),query_jersey_number))
    print("==============>>")
    if len(matched_files) > NumImages:
        matched_files = matched_files[:NumImages]
   
    
    #dump the images into user folder with folder name same as jersey number
    DestPath = DestPath.replace("\\","/")
    user_folder = os.path.join(DestPath,str(query_jersey_number))
    print("Retrieving top {} images to {} directory".format(len(matched_files), user_folder))
    print("===================>>")

    if os.path.exists(user_folder):
        shutil.rmtree(user_folder)
        time.sleep(10)
    
    os.mkdir(user_folder)
    for detected_image in matched_files:
        image_name = ((detected_image[0]).replace("\\","/")).split('/')[-1]
        dest_path = os.path.join(user_folder, image_name)
        shutil.copyfile(detected_image[0], dest_path)
    print(" {} images are copied to {} successfully".format(len(matched_files),user_folder))
    print("==========================>>")
    print("done")
    return matched_files, user_folder




@app.route('/retrieve_images/')
def retrieve_by_jersey_number():
    in_json = request.get_json()
    query_jersey_number = in_json['query_jersey_number']
    SearchSetParentPath = in_json['SearchSetParentPath']
    NumImages = in_json['NumImages']
    matched_list, user_folder = JerseyNumberRetrieve(query_jersey_number,SearchSetParentPath,NumImages=NumImages)
    out_message = " {} images are copied to {} successfully".format(len(matched_list),user_folder)

    return out_message

#if __name__ == '__main__':
#    app.run()
    

if __name__ == '__main__':
    # Must be loaded only once
    DETECTOR, CUSTOM = load_objdetection()
    DETECTORNUM = load_numberDetection()

    parser = argparse.ArgumentParser(description='retrieve by jersey number.')
    parser.add_argument("-q","--query_no", type=int, help='Query Jersey Number', required=True)
    parser.add_argument("-s","--search_dir", type=str, help='search set path', required = True)
    parser.add_argument("-n","--no_images", type=int, help='Number of images to be retrieved', default = 10)
    parser.add_argument("-d","--dest_path", type=str, help='destination path to dump output images', required = True)
    args = parser.parse_args()
    args = args
    query_jersey_number = args.query_no
    SearchSetParentPath = args.search_dir
    NumImages = args.no_images
    DestPath = args.dest_path
    
    #print("Modified")
    #print(args)SA
    #APPi
    #matched_list,user_folder = JerseyNumberRetrieve("2","C://Shiva//PP//Rod//Image//2021-05-04","2","C://Shiva//PP//Rod//Image")
    matched_list,user_folder = JerseyNumberRetrieve(query_jersey_number=query_jersey_number,SearchSetParentPath =SearchSetParentPath,NumImages=NumImages,DestPath=DestPath)
