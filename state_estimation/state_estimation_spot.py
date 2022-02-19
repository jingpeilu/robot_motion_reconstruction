import torch
import numpy as np
from spot_utils import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_l', type=int, default = 500, help='number of iteration for the estimation algorithm')
parser.add_argument('--n_q', type=int, default = 5, help='number of iteration for solving joint constraint')
parser.add_argument('--n_c', type=int, default = 5, help='number of iteration for solving camera constraint')
args = parser.parse_args()

w_q = 10
w_c = 1

dtype = torch.float
device = torch.device("cpu")

kp_2d = np.load("../keypoint_detection/outputs/spot_dance_2d_keypoints.npy")[:,5:]
b_T_cam_vecs = np.load("../keypoint_detection/outputs/spot_dance_extrinsic.npy")
n = kp_2d.shape[0]


# initialization
x_t = np.array([0.0, 0.5, -1.0, 0.0, 0.5, -1.0, 0.0, 0.5, -1.0, 0.0, 0.5, -1.0])
x_t = np.tile(x_t,(n,1))
x_t = torch.tensor(x_t, device=device, dtype=dtype, requires_grad=True)
k_param = torch.tensor(np.pi/3, device=device, dtype=dtype, requires_grad=True)
b_T_cam_torch = torch.tensor(b_T_cam_vecs, device=device, dtype=dtype, requires_grad=True)


def loss_fn(kp_2d, proj_kp, x_t, b_T_cam_torch , w_q, w_c):
    
    # keypoint loss
    loss_1 = torch.zeros(kp_2d.shape[0])
    for j in range(kp_2d.shape[0]):
        
        loss_j =torch.zeros(kp_2d.shape[1])
        for i in range(kp_2d.shape[1]):
            err = proj_kp[j,i,:].reshape(-1) - torch.tensor(kp_2d[j,i,:]).float().reshape(-1)
            loss_j[i] = torch.sqrt(torch.sum(err**2))
            
        loss_1[j] = torch.sum(loss_j)
        
    keypoint_loss = torch.sum(loss_1) / kp_2d.shape[0]

            
    # joint constraint loss
    loss_2 = torch.tensor(0)
    for i in range(x_t.shape[0]-1):
        err = torch.sum(torch.relu(torch.abs(x_t[i+1] - x_t[i]) - torch.tensor(0.1)))
        loss_2 = loss_2 + w_q*err #[50,10,5,1]
        
    l_q = loss_2
    
    # camera constraint loss
    loss_3 = torch.tensor(0)
    for i in range(b_T_cam_torch.shape[0]-1):
        err = torch.sum(torch.relu(torch.abs(b_T_cam_torch[i+1] - b_T_cam_torch[i]) - torch.tensor(0.1)))
        loss_3 = loss_3 + w_c*err #[50,10,5,1]
        
    l_c = loss_3
        
    print("keypoint loss: " + str(float(keypoint_loss)))
    print("joint constraint loss: " + str(float(l_q)))
    print("camera constraint loss: " + str(float(l_c)))

    return keypoint_loss + l_q + l_c




for num_loop in range(args.n_l):
    print(num_loop)

    
    # optmizing the joint states
    for itr in range(args.n_q):
        lr = 0.0001


        proj_kp = torch.zeros((n,12,2))
        for i in range(n):
            R = axi_angle_to_rot_matrix(b_T_cam_torch[i,:3])
            tvec = b_T_cam_torch[i,3:]
            pred_pose = torch.eye(4)
            pred_pose[:3,:3] = R
            pred_pose[0,3] = tvec[0]
            pred_pose[1,3] = tvec[1]
            pred_pose[2,3] = tvec[2]

            proj_kp_i = proj(x_t[i], pred_pose, k_param)
            proj_kp[i] = proj_kp_i

        z = loss_fn(kp_2d, proj_kp, x_t, b_T_cam_torch , w_q , w_c )

        #print(itr)
        print("loss:" + str(float(z)))
        z.backward()
        with torch.no_grad():
            x_t -= lr*torch.nan_to_num(x_t.grad)
            #print(torch.max(torch.abs(x_grad)))
            #print(torch.max(torch.abs(x_t)))
            
            x_t.grad = None
        
        print("---------------------")

    np.save("outputs/spot_x_itr%d.npy" % num_loop,x_t.detach().numpy())
    
    
    # optmizing the camera parameters
    for itr in range(args.n_c):
        lr_k = 0.00001
        lr_c = 0.0001


        proj_kp = torch.zeros((n,12,2))
        for i in range(n):
            R = axi_angle_to_rot_matrix(b_T_cam_torch[i,:3])
            tvec = b_T_cam_torch[i,3:]
            pred_pose = torch.eye(4)
            pred_pose[:3,:3] = R
            pred_pose[0,3] = tvec[0]
            pred_pose[1,3] = tvec[1]
            pred_pose[2,3] = tvec[2]

            proj_kp_i = proj(x_t[i], pred_pose, k_param)
            proj_kp[i] = proj_kp_i

        z = loss_fn(kp_2d, proj_kp, x_t, b_T_cam_torch , w_q , w_c )

        #print(itr)
        print("loss:" + str(float(z)))
        z.backward()

        with torch.no_grad():
            k_param -= lr_k*torch.nan_to_num(k_param.grad)
            b_T_cam_torch -= lr_c*torch.nan_to_num(b_T_cam_torch.grad)

            k_param.grad = None
            b_T_cam_torch.grad = None

        print("---------------------")

    np.save("outputs/spot_k_itr%d.npy" % num_loop ,k_param.detach().numpy())
    np.save("outputs/spot_c_itr%d.npy" % num_loop ,b_T_cam_torch.detach().numpy())
    
    if z.detach().numpy() < 10:
            break
