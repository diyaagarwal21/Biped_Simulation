import numpy as np
from scipy.linalg import expm

def twist_matrix(v, omega):
    """
    Constructs the 4x4 twist matrix for given linear and angular components.
    
    :param v: 3x1 numpy array, representing the linear velocity part of the twist.
    :param omega: 3x1 numpy array, representing the angular velocity part of the twist.
    :return: 4x4 numpy array representing the twist matrix.
    """
    # Skew-symmetric matrix for omega
    omega_hat = np.array([
        [0, -omega[2], omega[1]],
        [omega[2], 0, -omega[0]],
        [-omega[1], omega[0], 0]
    ])
    
    # Build the twist matrix
    twist_hat = np.block([
        [omega_hat, v.reshape((3,1))],
        [0, 0, 0, 0]
    ])
    
    return twist_hat


def forward_kinematics_product_Lie(d):
    """
    Forward Kinematics for a biped with two hips and two knees.
    :param d: Mujoco Data containing joint positions
    :return: End Effector Transformation Matrix
    """
    # Joint angles for the hips and knees
    theta_hip1 = d.qpos[0]  # Hip 1 angle
    theta_knee1 = d.qpos[1] # Knee 1 angle
    theta_hip2 = d.qpos[2]  # Hip 2 angle
    theta_knee2 = d.qpos[3] # Knee 2 angle

    # Define rotation matrices for each joint
    R_hip1 = expm(np.array([
        [0, -theta_hip1, 0],
        [theta_hip1, 0, 0],
        [0, 0, 0]
    ]))
    
    R_knee1 = expm(np.array([
        [0, -theta_knee1, 0],
        [theta_knee1, 0, 0],
        [0, 0, 0]
    ]))
    
    R_hip2 = expm(np.array([
        [0, -theta_hip2, 0],
        [theta_hip2, 0, 0],
        [0, 0, 0]
    ]))
    
    R_knee2 = expm(np.array([
        [0, -theta_knee2, 0],
        [theta_knee2, 0, 0],
        [0, 0, 0]
    ]))
    
    # Define displacement vectors for each link
    d_hip1 = np.array([[0], [0], [1]])  # Hip1 to Knee1
    d_knee1 = np.array([[0], [0], [1]]) # Knee1 to foot
    d_hip2 = np.array([[0], [0], [1]])  # Hip2 to Knee2
    d_knee2 = np.array([[0], [0], [1]]) # Knee2 to foot
    
    # Transformation matrices for each joint in the chain
    g_base_hip1 = np.block([[R_hip1, d_hip1], [0, 0, 0, 1]])
    g_hip1_knee1 = np.block([[R_knee1, d_knee1], [0, 0, 0, 1]])
    g_base_hip2 = np.block([[R_hip2, d_hip2], [0, 0, 0, 1]])
    g_hip2_knee2 = np.block([[R_knee2, d_knee2], [0, 0, 0, 1]])
    
    # Compute the end effector position for each leg
    g_leg1 = g_base_hip1 @ g_hip1_knee1
    g_leg2 = g_base_hip2 @ g_hip2_knee2
    
    return g_leg1, g_leg2

def forward_kinematics_product_exp(joint_angles):
    """
    Forward Kinematics using product of exponentials for a biped with two hips and two knees.
    :param joint_angles: List of joint angles [hip1, knee1, hip2, knee2]
    :return: End Effector Transformation Matrices for each leg
    """
    # Number of joints
    nj = 4
    hip1, knee1, hip2, knee2 = joint_angles
    
    # Define the axes of rotation for each joint (z-axis for simplicity)
    omegas = [
        np.array([0, 0, 1]),  # Hip1 axis
        np.array([0, 0, 1]),  # Knee1 axis (aligned with Hip1)
        np.array([0, 0, 1]),  # Hip2 axis
        np.array([0, 0, 1])   # Knee2 axis (aligned with Hip2)
    ]
    
    # Define points on each axis of rotation (relative positions)
    q_all = [
        np.array([0, 0, 0]),               # Hip1 origin
        np.array([0, 0, 1]),               # Knee1 origin relative to Hip1
        np.array([0, 0, 0]),               # Hip2 origin
        np.array([0, 0, 1])                # Knee2 origin relative to Hip2
    ]
    
    # Define the end-effector position relative to the last joint in each leg
    end_effector_transform = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, -1],  # Assume foot is 1 unit below the knee
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    # Compute forward kinematics for each leg
    g_leg1 = np.eye(4)
    for i, theta in enumerate([hip1, knee1]):
        v = np.cross(-omegas[i], q_all[i])
        twist_hat = twist_matrix(v, omegas[i])
        g_current = expm(twist_hat * theta)
        g_leg1 = g_leg1 @ g_current
    g_leg1 = g_leg1 @ end_effector_transform
    
    g_leg2 = np.eye(4)
    for i, theta in enumerate([hip2, knee2], start=2):
        v = np.cross(-omegas[i], q_all[i])
        twist_hat = twist_matrix(v, omegas[i])
        g_current = expm(twist_hat * theta)
        g_leg2 = g_leg2 @ g_current
    g_leg2 = g_leg2 @ end_effector_transform
    
    return g_leg1, g_leg2
