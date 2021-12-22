import torch
from utils import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--n_l', type=int, default = 200, help='number of iteration for the estimation algorithm')
parser.add_argument('--n_q', type=int, default = 10, help='number of iteration for solving joint constraint')
parser.add_argument('--n_c', type=int, default = 5, help='number of iteration for solving camera constraint')
args = parser.parse_args()

kp_2d = np.load("../keypoint_detection/outputs/baxter_poses_2d_keypoints.npy")[:,:7]
b_T_cam_vecs = np.load("../keypoint_detection/outputs/baxter_poses_extrinsic.npy")
n = kp_2d.shape[0]

# initialization
x_t = np.random.uniform(low = [-0.2, -0.6, -0.2,  0.6, -0.1, 1.1, -0.1],
                  high = [ 0.2, -0.4,  0.2,  0.8,  0.1, 1.3,  0.2], 
                  size = (n,7))
x_t = torch.tensor(x_t)
#x_t = x_t.to(device='cuda')
k_param = torch.tensor([9.42658407e+02, 9.42658407e+02, 1.02400000e+03, 7.68000000e+02])
#k_param = k_param.to(device='cuda')
b_T_cam_torch = torch.tensor(b_T_cam_vecs)
#b_T_cam_torch = b_T_cam_torch.to(device='cuda')


def loss_fn(kp_2d, proj_kp, x_t, b_T_cam_torch , w_q, w_c):
    
    # keypoint loss
    loss_1 = np.zeros(kp_2d.shape[0]).astype(object)
    for j in range(kp_2d.shape[0]):
        
        loss_j =np.zeros(kp_2d.shape[1]).astype(object)
        for i in range(kp_2d.shape[1]):
            err = proj_kp[j,i,:].reshape(-1) - kp_2d[j,i,:].reshape(-1)
            loss_j[i] = torch.sqrt(np.sum(err**2))
            
        loss_1[j] = np.sum(loss_j)
        
    keypoint_loss = np.sum(loss_1) / kp_2d.shape[0]

            
    # joint constraint loss
    loss_2 = torch.tensor(0)
    for i in range(x_t.shape[0]-1):
        err = torch.sum(torch.relu(torch.abs(x_t[i+1] - x_t[i]) - torch.tensor(0.02)))
        loss_2 = loss_2 + w_q*err #[50,10,5,1]
        
    l_q = loss_2
        
    print("keypoint loss: " + str(float(keypoint_loss)))
    print("joint constraint loss: " + str(float(l_q)))

    return keypoint_loss + l_q




for num_loop in range(args.n_l):
    print(num_loop)

    
    # optmizing the joint states
    for itr in range(args.n_q):
        lr = 0.001
        x_t.requires_grad_(True)


        proj_kp = np.zeros((n,7,2)).astype(object)
        for i in range(n):
            R = axi_angle_to_rot_matrix(b_T_cam_torch[:3])
            tvec = b_T_cam_torch[3:]
            pred_pose = np.identity(4).astype(object)
            pred_pose[:3,:3] = R
            pred_pose[0,3] = tvec[0]
            pred_pose[1,3] = tvec[1]
            pred_pose[2,3] = tvec[2]

            proj_kp_i = proj(x_t[i], pred_pose, k_param)
            proj_kp[i] = proj_kp_i

        z = loss_fn(kp_2d, proj_kp, x_t, b_T_cam_torch , w_q = 1, w_c = 10)

        #print(itr)
        print("loss:" + str(float(z)))
        z.backward()
        x_grad = x_t.grad.data
        x_t = x_t.clone().detach()
        x_t = x_t - lr*x_grad
        print("---------------------")

    np.save("outputs/x_itr%d.npy" % num_loop,x_t.detach().numpy())
    
    # optmizing the camera parameters
    for itr in range(args.n_c):
        lr_k = 0.00001
        lr_c = 0.0001
        
        k_param.requires_grad_(True)    
        b_T_cam_torch.requires_grad_(True)   



        proj_kp = np.zeros((n,7,2)).astype(object)
        for i in range(n):
            R = axi_angle_to_rot_matrix(b_T_cam_torch[:3])
            tvec = b_T_cam_torch[3:]
            pred_pose = np.identity(4).astype(object)
            pred_pose[:3,:3] = R
            pred_pose[0,3] = tvec[0]
            pred_pose[1,3] = tvec[1]
            pred_pose[2,3] = tvec[2]

            proj_kp_i = proj(x_t[i], pred_pose, k_param)
            proj_kp[i] = proj_kp_i

        z = loss_fn(kp_2d, proj_kp, x_t, b_T_cam_torch , w_q = 1, w_c = 10)

        #print(itr)
        print("loss:" + str(float(z)))
        z.backward()
        k_grad = k_param.grad.data
        b_T_cam_grad = b_T_cam_torch.grad.data
        k_param = k_param.clone().detach()    
        k_param = k_param - lr_k*k_grad
        b_T_cam_torch = b_T_cam_torch.clone().detach()    
        b_T_cam_torch = b_T_cam_torch - lr_c*b_T_cam_grad
        print("---------------------")

    np.save("outputs/k_itr%d.npy" % num_loop ,k_param.detach().numpy())
    np.save("outputs/c_itr%d.npy" % num_loop ,b_T_cam_torch.detach().numpy())
    
    if z.detach().numpy() < 10:
            break
