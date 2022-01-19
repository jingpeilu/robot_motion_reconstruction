import glob
import numpy as np
from os.path import dirname, join, abspath
import transforms3d.quaternions as quaternions
import transforms3d.euler as euler
import os,sys
import cv2
import torch
import yaml
from hed_test import *
from attrdict import AttrDict
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

def preprocess_img(img):
    # preprocessing the image 
    from matplotlib import cm
    
    viridis = cm.get_cmap('viridis')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    edge_img = hed_test.extract_edge(img)
    canvas = np.zeros((edge_img.shape[0],edge_img.shape[1],3))
    for i in range(edge_img.shape[0]):
        for j in range(edge_img.shape[1]):
            canvas[i,j] = viridis(edge_img[i,j])[:3]
            
    output = canvas*255
    return output.astype(np.uint8)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, help='spot_dance')
args = parser.parse_args()


if args.dataset == "spot_dance":
    cfg_file = "pretrained_weights/spot/standard.yaml"
    with open(cfg_file, 'r') as f:
        cfg = AttrDict( yaml.safe_load(f) )
    cfg.path = cfg_file
    cfg.time = "0"

    hed_test = HED_test(cfg,"pretrained_weights/spot/epoch_29.pth")

    #### load keypoint detector ####
    from keypoint_detection import Keypoint_detector
    kp_detector = Keypoint_detector(weights_path="pretrained_weights/spot/pose_cfg_test.yamlsnapshot-1030000" , 
                                    config="pretrained_weights/spot/pose_cfg_test.yaml")

    img_list = glob.glob("../dataset/spot_dance/*.jpg")
    img_list.sort(key=lambda f:int(f.split("/")[-1].split(".")[-2]))

    # extracting keypoints
    predictions = np.zeros((len(img_list),17,2))
    for i in tqdm(range(len(img_list))):
        file_name = img_list[i]
        img = cv2.imread(file_name)
        imgScale = 0.25
        newX,newY = img.shape[1]*imgScale, img.shape[0]*imgScale
        resized_img = cv2.resize(img,(int(newX),int(newY)))
        resized_img = preprocess_img(resized_img)

        results = kp_detector.predict_single_image(resized_img)
        points_predicted = results[:,:2] * 1.2
        points_predicted = points_predicted.astype(int)
        predictions[i] = points_predicted

        ###save imgs for visualization###

        #scores = results[:,2]
        #points_predicted_ori = results[:,:2] * (1/imgScale)
        #img = kp_detector.overwrite_image(img, points_predicted_ori, scores)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #plt.imsave("test/%d.png"%i , img)

    print("Saving spot keypoints...")
    np.save("outputs/spot_dance_2d_keypoints.npy",predictions)


    # initializing the camera parameters
    
    # initial guess of intrinsic camera
    P = np.array([[-554.25625842,    0.        ,  320.        ],
       [   0.        , -554.25625842,  240.        ],
       [   0.        ,    0.        ,    1.        ]])
       
       
    points_3d = np.array([[ 4.0199992e-01, -2.3283064e-09,  3.5001636e-03],
       [-4.2300019e-01, -2.6542693e-08,  4.4996142e-03],
       [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00],
       [ 2.2300006e-01,  3.9115548e-08,  8.0500066e-02],
       [-2.2300011e-01, -2.8405339e-08,  8.0499947e-02]])


    prediction_vecs = np.zeros((predictions.shape[0],6))
    for i in range(predictions.shape[0]):

        # keypoint 2d position
        points_predicted = predictions[i]
        points_2d = points_predicted[[0,1,2,3,4]]

        # using EPnP to find the cam-to-base transform
        retval,rvec,tvec = cv2.solvePnP(points_3d,points_2d.astype(float),P,distCoeffs = None, flags = 1)
        R,_ = cv2.Rodrigues(rvec)
        
        prediction_vecs[i,:3] = rvec.reshape(-1)
        prediction_vecs[i,3:] = tvec.reshape(-1)

    print("Saving initial camera parameters...")
    np.save("outputs/spot_dance_extrinsic.npy",prediction_vecs)
    np.save("outputs/spot_dance_intrinsic.npy",P)

else:
    print("Unknown dataset...")

