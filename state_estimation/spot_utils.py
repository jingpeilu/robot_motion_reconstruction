import torch
import numpy as np



'''
Forward kinematic transformations for a quadriped robot. 

References:
1. Inverse Kinematic Analysis Of A Quadruped Robot.
   Sen, Muhammed Arif & Bakircioglu, Veli & Kalyoncu, Mete. (2017). 
   International Journal of Scientific & Technology Research. 6. 
   
2. https://github.com/mike4192/spot_micro_kinematics_python/blob/master/utilities/spot_micro_kinematics.py

'''
def rotx(angle):
    """Create a 3x3 numpy rotation matrix about the x axis

    rotxMatrix = np.array(
        [   [1,             0,              0],
            [0,      torch.cos(ang),      -torch.sin(ang)],
            [0,      torch.sin(ang),       torch.cos(ang)]  ])
    """
    rotxMatrix = torch.zeros((3,3))
    rotxMatrix[0,0] = 1.0
    rotxMatrix[1,1] = torch.cos(angle)
    rotxMatrix[1,2] = -torch.sin(angle)
    rotxMatrix[2,1] = torch.sin(angle)
    rotxMatrix[2,2] = torch.cos(angle)

    return rotxMatrix

def roty(angle):
    """Create a 3x3 numpy rotation matrix about the y axis
    
    rotyMatrix = np.array(
        [   [ torch.cos(ang),      0,       torch.sin(ang)],
            [       0,       1,              0],
            [-torch.sin(ang),      0,       torch.cos(ang)]  ])
    """
    
    rotyMatrix = torch.zeros((3,3))
    rotyMatrix[0,0] = torch.cos(angle)
    rotyMatrix[0,2] = torch.sin(angle)
    rotyMatrix[1,1] = 1.0
    rotyMatrix[2,0] = -torch.sin(angle)
    rotyMatrix[2,2] = torch.cos(angle)
        
    return rotyMatrix


def rotz(angle):
    """Create a 3x3 numpy rotation matrix about the z axis

    
    rotzMatrix = np.array(
        [   [torch.cos(ang),   -torch.sin(ang),             0],
            [torch.sin(ang),    torch.cos(ang),             0],
            [       0,           0,             1]  ])
    """
    rotzMatrix = torch.zeros((3,3))
    rotzMatrix[0,0] = torch.cos(angle)
    rotzMatrix[0,1] = -torch.sin(angle)
    rotzMatrix[1,0] = torch.sin(angle)
    rotzMatrix[1,1] = torch.cos(angle)
    rotzMatrix[2,2] = 1.0


    return rotzMatrix

def rotxyz(x_ang,y_ang,z_ang):
    """Creates a 3x3 numpy rotation matrix from three rotations done in the order
    of x, y, and z in the local coordinate frame as it rotates.

    """
    return rotx(x_ang) @ roty(y_ang) @ rotz(z_ang)


def get_T_m():
    T_m = torch.eye(4)
    T_m[:3,:3] = rotx(torch.tensor(np.pi/2))
    return T_m

def t_rightback(t_m,l,w):
    '''Creates a 4x4 numpy homogeneous transformation matrix representing coordinate system and 
    position of the rightback leg of a quadriped. Assumes legs postioned in corners of a rectangular
    plane defined by a width and length 
    
    temp_homog_transf = np.block( [ [ roty(np.pi/2), np.array([[-l/2],[0],[w/2]])  ],
                                    [np.array([0,0,0,1])] ]    )
    '''
    temp_homog_transf = torch.eye(4)
    temp_homog_transf[:3,:3] = roty(torch.tensor(np.pi/2))
    temp_homog_transf[0,3] = -l/2
    temp_homog_transf[2,3] = w/2
    return t_m @ temp_homog_transf

def t_rightfront(t_m,l,w):
    '''
    temp_homog_transf = np.block( [ [ roty(np.pi/2), np.array([[l/2],[0],[w/2]])  ],
                                    [np.array([0,0,0,1])] ]    )
    '''
    temp_homog_transf = torch.eye(4)
    temp_homog_transf[:3,:3] = roty(torch.tensor(np.pi/2))
    temp_homog_transf[0,3] = l/2
    temp_homog_transf[2,3] = w/2
    return t_m @ temp_homog_transf

def t_leftback(t_m,l,w):

    '''
    temp_homog_transf = np.block( [ [ roty(-np.pi/2), np.array([[-l/2],[0],[-w/2]])  ],
                                    [np.array([0,0,0,1])] ]    )
    '''
    temp_homog_transf = torch.eye(4)
    temp_homog_transf[:3,:3] = roty(torch.tensor(-np.pi/2))
    temp_homog_transf[0,3] = -l/2
    temp_homog_transf[2,3] = -w/2
    return t_m @ temp_homog_transf

def t_leftfront(t_m,l,w):
    '''
    temp_homog_transf = np.block( [ [ roty(-np.pi/2), np.array([[l/2],[0],[-w/2]])  ],
                                    [np.array([0,0,0,1])] ]    )
    '''
    temp_homog_transf = torch.eye(4)
    temp_homog_transf[:3,:3] = roty(torch.tensor(-np.pi/2))
    temp_homog_transf[0,3] = l/2
    temp_homog_transf[2,3] = -w/2
    return t_m @ temp_homog_transf

def t_0_to_1(theta1,l1):
    theta1 = -theta1
    '''Create the homogeneous transformation matrix for joint 0 to 1 for a quadriped leg.
    Args:
        theta1: Rotation angle in radians of the hip joint
        l1: Length of the hip joint link
    Returns:
        A 4x4 numpy matrix. Homogeneous transform from joint 0 to 1

    t_01 = np.block( [ [ rotz(theta1), np.array([[-l1*np.cos(theta1)],[-l1*np.sin(theta1)],[0]])  ],
                                        [np.array([0,0,0,1])] ]    )
                                        
    '''
    t_01 = torch.eye(4)
    t_01[:3,:3] = rotz(theta1)
    t_01[0,3] = -l1*torch.cos(theta1)
    t_01[1,3] = -l1*torch.sin(theta1)
    return t_01

def t_1_to_2():
    '''Create the homogeneous transformation matrix for joint 1 to 2 for a quadriped leg.
    Args:
        None
    Returns:
        A 4x4 numpy matrix. Homogeneous transform from joint 1 to 2

     
    t_12 = np.array([[ 0,  0, -1,  0],
                     [-1,  0,  0,  0],
                     [ 0,  1,  0,  0],
                     [ 0,  0,  0,  1]])
    '''
    
    t_12 = torch.tensor([[ 0.0,  0.0, -1.0,  0.0],
                         [-1.0,  0.0,  0.0,  0.0],
                         [ 0.0,  1.0,  0.0,  0.0],
                         [ 0.0,  0.0,  0.0,  1.0]])
    
    return t_12

def t_2_to_3(theta2,l2):
    '''Create the homogeneous transformation matrix for joint 1 to 2 for a quadriped leg.
    Args:
        theta2: Rotation angle in radians of the leg joint
        l2: Length of the upper leg link
    Returns:
        A 4x4 numpy matrix. Homogeneous transform from joint 2 to 3

        t_23 = np.block( [ [ rotz(theta2), np.array([[l2*np.cos(theta2)],[l2*np.sin(theta2)],[0]])  ],
                                        [np.array([0,0,0,1])] ]    )
    '''
    t_23 = torch.eye(4)
    t_23[:3,:3] = rotz(theta2)
    t_23[0,3] = l2*torch.cos(theta2)
    t_23[1,3] = l2*torch.sin(theta2)
    return t_23

def t_3_to_4(theta3,l3):
    '''Create the homogeneous transformation matrix for joint 3 to 4 for a quadriped leg.
    Args:
        theta3: Rotation angle in radians of the knee joint
        l3: Length of the lower leg link
    Returns:
        A 4x4 numpy matrix. Homogeneous transform from joint 3 to 4
        
        t_34 = np.block( [ [ rotz(theta3), np.array([[l3*np.cos(theta3)],[l3*np.sin(theta3)],[0]])  ],
                                        [np.array([0,0,0,1])] ]    )
    '''
    
    t_34 = torch.eye(4)
    t_34[:3,:3] = rotz(theta3)
    t_34[0,3] = l3*torch.cos(theta3)
    t_34[1,3] = l3*torch.sin(theta3)
    
    return t_34


def front_left_in_base(theta):
    ''' Compute joint position in robot base frame
    Args:
        theta: Rotation angle in radians of the joints
    Returns:
        hip, knee, and foot position
    '''
    T_b_1 = t_leftfront(T_m, l, w) @ t_0_to_1(theta[0],l1)
    T_b_3 = T_b_1 @ t_1_to_2() @ t_2_to_3(theta[1],l2)
    T_b_4 = T_b_3 @ t_3_to_4(theta[2],l3)
    hip = T_b_1 @ torch.tensor([0.0,0.0,0.0,1.0]).reshape(-1,1)
    knee = T_b_3 @ torch.tensor([0,-0.025,0,1]).reshape(-1,1)
    foot = T_b_4 @ torch.tensor([0.0,0.0,0.0,1.0]).reshape(-1,1)
    
    return hip, knee, foot


def rear_left_in_base(theta):
    T_b_1 = t_leftback(T_m, l, w) @ t_0_to_1(theta[0],l1)
    T_b_3 = T_b_1 @ t_1_to_2() @ t_2_to_3(theta[1],l2)
    T_b_4 = T_b_3 @ t_3_to_4(theta[2],l3)
    
    hip = T_b_1 @ torch.tensor([0.0,0.0,0.0,1.0]).reshape(-1,1)
    knee = T_b_3 @ torch.tensor([0,-0.025,0,1]).reshape(-1,1)
    foot = T_b_4 @ torch.tensor([0.0,0.0,0.0,1.0]).reshape(-1,1)
    
    return hip, knee, foot

    
    
def front_right_in_base(theta):
    theta = -1 * theta
    T_b_1 = t_rightfront(T_m, l, w) @ t_0_to_1(theta[0],l1)
    T_b_3 = T_b_1 @ t_1_to_2() @ t_2_to_3(theta[1],l2)
    T_b_4 = T_b_3 @ t_3_to_4(theta[2],l3)
    
    hip = T_b_1 @ torch.tensor([0.0,0.0,0.0,1.0]).reshape(-1,1)
    knee = T_b_3 @ torch.tensor([0,0.025,0,1]).reshape(-1,1)
    foot = T_b_4 @ torch.tensor([0.0,0.0,0.0,1.0]).reshape(-1,1)

    return hip, knee, foot


def rear_right_in_base(theta):
    theta = -1 * theta
    
    T_b_1 = t_rightback(T_m, l, w) @ t_0_to_1(theta[0],l1)
    T_b_3 = T_b_1 @ t_1_to_2() @ t_2_to_3(theta[1],l2)
    T_b_4 = T_b_3 @ t_3_to_4(theta[2],l3)
    
    hip = T_b_1 @ torch.tensor([0.0,0.0,0.0,1.0]).reshape(-1,1)
    knee = T_b_3 @ torch.tensor([0,0.025,0,1]).reshape(-1,1)
    foot = T_b_4 @ torch.tensor([0.0,0.0,0.0,1.0]).reshape(-1,1)

    return hip, knee, foot
    
    
T_m = get_T_m()
l = torch.tensor(0.29785*2)
w = torch.tensor(0.05500*2)
l1 = torch.tensor(0.110945)
l2 = torch.tensor(0.3205)
l3 = torch.tensor(0.37)
######################################################################################


def get_camera_matrix(fov, x_resolution = torch.tensor(640.0), y_resolution = torch.tensor(480.0)):
    if (x_resolution/y_resolution) > 1:
        fov_x = fov
        fov_y = 2 * torch.arctan(torch.tan(fov/2) / (x_resolution/y_resolution))
    else:
        fov_x = 2 * torch.arctan(torch.tan(fov/2) / (x_resolution/y_resolution))
        fov_y = fov
        
    P = torch.eye(3)
    P[0,0] = -(x_resolution/2)/torch.tan(0.5*fov_x)
    P[0,2] = (x_resolution/2)
    P[1,1] = -(y_resolution/2)/torch.tan(0.5*fov_y)
    P[1,2] = (y_resolution/2)

    return P

def get_coor_by_P(v_pos, fov = torch.tensor(np.pi/3)):
    
    # find the field of view for x and y axis
    P = get_camera_matrix(fov)
    
    x = P @ v_pos.reshape((3,1))
    x = x/x[2]
    x = x[:2].reshape(-1)
    return x

def dehomogenize_3d(vec):
    vec = vec.reshape((-1,1))
    vec = vec/vec[3]
    return vec[:3]
    
    
    
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
    
    
def proj(x, b_T_cam, k_param):
    # projecting keypoints to image
    
    p = torch.zeros((12,2))

    p[0] = get_coor_by_P(dehomogenize_3d(b_T_cam @ front_left_in_base(x[0:3])[0]), k_param)
    p[1] = get_coor_by_P(dehomogenize_3d(b_T_cam @ front_left_in_base(x[0:3])[1]), k_param)
    p[2] = get_coor_by_P(dehomogenize_3d(b_T_cam @ front_left_in_base(x[0:3])[2]), k_param)
    
    p[3] = get_coor_by_P(dehomogenize_3d(b_T_cam @ front_right_in_base(x[3:6])[0]), k_param)
    p[4] = get_coor_by_P(dehomogenize_3d(b_T_cam @ front_right_in_base(x[3:6])[1]), k_param)
    p[5] = get_coor_by_P(dehomogenize_3d(b_T_cam @ front_right_in_base(x[3:6])[2]), k_param)
    
    p[6] = get_coor_by_P(dehomogenize_3d(b_T_cam @ rear_left_in_base(x[6:9])[0]), k_param)
    p[7] = get_coor_by_P(dehomogenize_3d(b_T_cam @ rear_left_in_base(x[6:9])[1]), k_param)
    p[8] = get_coor_by_P(dehomogenize_3d(b_T_cam @ rear_left_in_base(x[6:9])[2]), k_param)
    
    p[9] = get_coor_by_P(dehomogenize_3d(b_T_cam @ rear_right_in_base(x[9:12])[0]), k_param)
    p[10] = get_coor_by_P(dehomogenize_3d(b_T_cam @ rear_right_in_base(x[9:12])[1]), k_param)
    p[11] = get_coor_by_P(dehomogenize_3d(b_T_cam @ rear_right_in_base(x[9:12])[2]), k_param)

    
    return p
