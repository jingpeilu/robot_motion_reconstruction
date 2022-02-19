import numpy as np
import torch
import matplotlib.pyplot as plt
import json


def get_coor_by_P(k_param, v_pos):
    
    # find the field of view for x and y axis

    P = torch.zeros((3,3))
    P[0,0] = k_param[0]
    P[0,2] = k_param[2]
    P[1,1] = k_param[1]
    P[1,2] = k_param[3]
    P[2,2] = 1.0
    x = P @ v_pos.reshape((3,1))
    x = x/x[2]
    x = x[:2].reshape(-1)
    return x


def dehomogenize_3d(vec):
    vec = vec.reshape((-1,1))
    vec = vec/vec[3]
    return vec[:3]

def T_from_DH(alp,a,d,the):
    '''
    Transformation matrix fron DH
    '''
    T = torch.zeros((4,4))
    T[0,0] = torch.cos(the)
    T[0,1] = -torch.sin(the)
    T[0,2] = 0
    T[0,3] = a
    T[1,0] = torch.sin(the)*torch.cos(alp)
    T[1,1] = torch.cos(the)*torch.cos(alp)
    T[1,2] = -torch.sin(alp)
    T[1,3] = -d*torch.sin(alp)

    T[2,0] = torch.sin(the)*torch.sin(alp)
    T[2,1] = torch.cos(the)*torch.sin(alp)
    T[2,2] = torch.cos(alp)
    T[2,3] = d*torch.cos(alp)
    T[3,0] = 0
    T[3,1] = 0
    T[3,2] = 0
    T[3,3] = 1
    return T

def get_bl_T_Jn(n, theta):
    '''
    Get joint to base(left) transform using baxter FK
    FK source: https://www.ohio.edu/mechanical-faculty/williams/html/pdf/BaxterKinematics.pdf
    n = 6 for J6 to base
    n = 8 for EE to base
    '''
    assert n in [0,2,4,6,8]
    bl_T_0 = torch.tensor([[1,0,0,0],
                    [0,1,0,0],
                    [0,0,1,0.27035],
                    [0,0,0,1]])
    T_7_ee = torch.tensor([[1,0,0,0],
                    [0,1,0,0],
                    [0,0,1,0.3683],
                    [0,0,0,1]])

    T_0_1 = T_from_DH(torch.tensor(0.0),      torch.tensor(0.0),   torch.tensor(0.0),    theta[0])
    T_1_2 = T_from_DH(torch.tensor(-np.pi/2), torch.tensor(0.069), torch.tensor(0.0),    theta[1]+np.pi/2)
    T_2_3 = T_from_DH(torch.tensor(np.pi/2),  torch.tensor(0.0),   torch.tensor(0.36435),theta[2])
    T_3_4 = T_from_DH(torch.tensor(-np.pi/2), torch.tensor(0.069), torch.tensor(0.0),    theta[3])
    T_4_5 = T_from_DH(torch.tensor(np.pi/2),  torch.tensor(0.0),   torch.tensor(0.37429),theta[4])
    T_5_6 = T_from_DH(torch.tensor(-np.pi/2), torch.tensor(0.010), torch.tensor(0.0),    theta[5])
    T_6_7 = T_from_DH(torch.tensor(np.pi/2),  torch.tensor(0.0),   torch.tensor(0.0),    theta[6])
    if n == 0:
        T = T_0_1
    elif n == 2:
        T = bl_T_0  @ T_0_1 @ T_1_2
    elif n == 4:
        T = bl_T_0  @ T_0_1 @ T_1_2 @ T_2_3 @ T_3_4
    elif n == 6:
        T = bl_T_0  @ T_0_1 @ T_1_2 @ T_2_3 @ T_3_4 @ T_4_5 @ T_5_6
    elif n == 8:
        T = bl_T_0  @ T_0_1 @ T_1_2 @ T_2_3 @ T_3_4 @ T_4_5 @ T_5_6 @ T_6_7 @ T_7_ee
    else:
        raise Exception("Invalid joint number")
    return T

def get_position_wrt_joints(p_id):
    # Opening JSON file 
    with open("../keypoint_detection/keypoint_config.json") as json_file: 
        data = json.load(json_file) 

    name = "point_" + str(p_id)
    position = data[name]["position"]
    parent = data[name]["parent"]


    return position, parent


def get_3d_position_to_bl( p_id,theta):

    position_to_joint, parent_joint = get_position_wrt_joints(p_id)
    # list to p_vec
    position_to_joint = torch.tensor(np.hstack((position_to_joint,[1])).reshape((4,1)))
    if parent_joint == "J0":
        position_to_bl = get_bl_T_Jn(0,theta) @ position_to_joint
    elif parent_joint == "J2":
        position_to_bl = get_bl_T_Jn(2,theta) @ position_to_joint
    elif parent_joint == "J4":
        position_to_bl = get_bl_T_Jn(4,theta) @ position_to_joint
    elif parent_joint == "J6":
        position_to_bl = get_bl_T_Jn(6,theta) @ position_to_joint
    elif parent_joint == "EE":
        position_to_bl = get_bl_T_Jn(8,theta) @ position_to_joint
    else:
        raise Exception("Sorry, no parent joint found")
    return position_to_bl

def proj(x, b_T_cam, k_param):
    # projecting keypoints to image
    # using [0,4,7,9,13,15,18] keypoints from keypoint json file
    # skipping reading from file process to speed up the code
    p = torch.zeros((7,2))
    p[0] = get_coor_by_P( k_param, dehomogenize_3d(b_T_cam @ get_bl_T_Jn(0,x) @ torch.tensor([0.0,0.0,0.0,1.0]).reshape((4,1)) ))
    p[1] = get_coor_by_P( k_param, dehomogenize_3d(b_T_cam @ get_bl_T_Jn(2,x) @ torch.tensor([-0.02700508, 0.04893792, -0.0228409,1]).reshape((4,1)) ))
    p[2] = get_coor_by_P( k_param, dehomogenize_3d(b_T_cam @ get_bl_T_Jn(2,x) @ torch.tensor([0.05922043, -0.18791741, -0.00197341,1]).reshape((4,1)) ))
    p[3] = get_coor_by_P( k_param, dehomogenize_3d(b_T_cam @ get_bl_T_Jn(4,x) @ torch.tensor([0.0, 0.0, 0.0,1.0]).reshape((4,1)) ))
    p[4] = get_coor_by_P( k_param, dehomogenize_3d(b_T_cam @ get_bl_T_Jn(4,x) @ torch.tensor([-0.02277625, -0.18333554, -0.01677856,1]).reshape((4,1)) ))
    p[5] = get_coor_by_P( k_param, dehomogenize_3d(b_T_cam @ get_bl_T_Jn(6,x) @ torch.tensor([0.0, 0.0, 0.0 ,1.0]).reshape((4,1)) ))
    p[6] = get_coor_by_P( k_param, dehomogenize_3d(b_T_cam @ get_bl_T_Jn(8,x) @ torch.tensor([-0.00543057919, 0.000110387802, -0.112797022,1]).reshape((4,1)) ))
    return p


def axi_angle_to_rot_matrix(k):
    angle = torch.norm(k)
    k = k/angle
    c = torch.cos(angle)
    s = torch.sin(angle)
    t = 1-c
    m00 = c + k[0]*k[0]*t
    m11 = c + k[1]*k[1]*t
    m22 = c + k[2]*k[2]*t
    tmp1 =  k[0]*k[1]*t
    tmp2 = k[2]*s
    m10 = tmp1 + tmp2
    m01 = tmp1 - tmp2
    tmp1 =  k[0]*k[2]*t
    tmp2 = k[1]*s
    m20 = tmp1 - tmp2
    m02 = tmp1 + tmp2    
    tmp1 = k[1]*k[2]*t
    tmp2 =  k[0]*s
    m21 = tmp1 + tmp2
    m12 = tmp1 - tmp2
    R = torch.zeros((3,3))
    R[0,0] = m00
    R[0,1] = m01
    R[0,2] = m02
    R[1,0] = m10
    R[1,1] = m11
    R[1,2] = m12
    R[2,0] = m20
    R[2,1] = m21
    R[2,2] = m22
    return R
