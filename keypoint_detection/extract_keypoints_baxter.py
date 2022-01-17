import glob
import numpy as np
from os.path import dirname, join, abspath
import transforms3d.quaternions as quaternions
import transforms3d.euler as euler
import os,sys
import cv2

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True, help='baxter_poses or baxter_wobbler')
args = parser.parse_args()


if args.dataset == "baxter_wobbler":
    from keypoint_detection import Keypoint_detector
    # loading pretrained weights for keypoint detector
    kp_detector = Keypoint_detector(weights_path="pretrained_weights/baxter/pose_cfg_test.yamlsnapshot-1030000" , 
                                    config="pretrained_weights/baxter/pose_cfg_test.yaml")

    img_list = glob.glob("../dataset/baxter_wobbler/images/*.png")
    img_list.sort(key=lambda f:int(f.split("/")[-1].split(".")[-2]))

    # extracting keypoints
    predictions = np.zeros((len(img_list),14,2))
    for i in range(len(img_list)):
        file_name = img_list[i]
        img = cv2.imread(file_name)
        imgScale = 0.25
        newX,newY = img.shape[1]*imgScale, img.shape[0]*imgScale
        resized_img = cv2.resize(img,(int(newX),int(newY)))

        results = kp_detector.predict_single_image(resized_img)
        points_predicted = results[:,:2] * 4
        points_predicted = points_predicted.astype(int)
        predictions[i] = points_predicted

    print("Saving baxter keypoints...")
    np.save("outputs/baxter_wobbler_2d_keypoints.npy",predictions)


    # initializing the camera parameters

    # initial guess of intrinsic camera
    P = np.array([[9.42658407e+02,  0.00000000e+00,  1.02400000e+03],
        [ 0.00000000e+00, 9.42658407e+02,  7.68000000e+02],
        [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

    # baxter FK
    from baxter_fk import Baxter_FK
    baxter = Baxter_FK("keypoint_config.json")

    feature_ids = [0,21,22,23,24,25,26,27]

    # keypoint 3d position
    points_3d = np.zeros((1,3))
    for i in feature_ids:
        points_3d = np.vstack((points_3d,baxter.get_3d_position_to_bl(i,[0,0,0,0,0,0,0]).reshape(-1)[:3]))
    points_3d = points_3d[1:,:].astype(np.float32)


    prediction_vecs = np.zeros((predictions.shape[0],6))
    for i in range(predictions.shape[0]):

        # keypoint 2d position
        points_predicted = predictions[i]
        points_2d = points_predicted[[0,7,8,9,10,11,12,13]]

        # using EPnP to find the cam-to-base transform
        retval,rvec,tvec = cv2.solvePnP(points_3d,points_2d.astype(float),P,distCoeffs = None, flags = 1)
        R,_ = cv2.Rodrigues(rvec)
        
        prediction_vecs[i,:3] = rvec.reshape(-1)
        prediction_vecs[i,3:] = tvec.reshape(-1)

    print("Saving initial camera parameters...")
    np.save("outputs/baxter_wobbler_extrinsic.npy",prediction_vecs)
    np.save("outputs/baxter_wobbler_intrinsic.npy",P)


elif args.dataset == "baxter_poses":
    from keypoint_detection import Keypoint_detector
    # loading pretrained weights for keypoint detector
    kp_detector = Keypoint_detector(weights_path="pretrained_weights/baxter/pose_cfg_test.yamlsnapshot-1030000" , 
                                    config="pretrained_weights/baxter/pose_cfg_test.yaml")

    img_list = glob.glob("../dataset/baxter_poses/images/*.png")
    img_list.sort(key=lambda f:int(f.split("/")[-1].split(".")[-2]))

    # extracting keypoints
    predictions = np.zeros((len(img_list),14,2))
    for i in range(len(img_list)):
        file_name = img_list[i]
        img = cv2.imread(file_name)
        imgScale = 0.25
        newX,newY = img.shape[1]*imgScale, img.shape[0]*imgScale
        resized_img = cv2.resize(img,(int(newX),int(newY)))

        results = kp_detector.predict_single_image(resized_img)
        points_predicted = results[:,:2] * (1/imgScale)
        points_predicted = points_predicted.astype(int)
        predictions[i] = points_predicted

    print("Saving baxter keypoints...")
    np.save("outputs/baxter_poses_2d_keypoints.npy",predictions)


    # initializing the camera parameters

    # initial guess of intrinsic camera
    P = np.array([[9.42658407e+02,  0.00000000e+00,  1.02400000e+03],
        [ 0.00000000e+00, 9.42658407e+02,  7.68000000e+02],
        [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])

    # baxter FK
    from baxter_fk import Baxter_FK
    baxter = Baxter_FK("keypoint_config.json")

    feature_ids = [0,21,22,23,24,25,26,27]

    # keypoint 3d position
    points_3d = np.zeros((1,3))
    for i in feature_ids:
        points_3d = np.vstack((points_3d,baxter.get_3d_position_to_bl(i,[0,0,0,0,0,0,0]).reshape(-1)[:3]))
    points_3d = points_3d[1:,:].astype(np.float32)


    prediction_vecs = np.zeros(6)
    i = 0
    # keypoint 2d position
    points_predicted = predictions[i]
    points_2d = points_predicted[[0,7,8,9,10,11,12,13]]

    # using EPnP to find the cam-to-base transform
    retval,rvec,tvec = cv2.solvePnP(points_3d,points_2d.astype(float),P,distCoeffs = None, flags = 1)
    R,_ = cv2.Rodrigues(rvec)

    prediction_vecs[:3] = rvec.reshape(-1)
    prediction_vecs[3:] = tvec.reshape(-1)

    print("Saving initial camera parameters...")
    np.save("outputs/baxter_poses_extrinsic.npy",prediction_vecs)
    np.save("outputs/baxter_poses_intrinsic.npy",P)


else:
    print("Unknown dataset...")

