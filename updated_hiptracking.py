import math
import time
import mujoco
import mujoco.viewer
import numpy as np

m = mujoco.MjModel.from_xml_path('STLs/scene.xml')
d = mujoco.MjData(m)

d.qpos[0] = 0.5

l1 = 0.253158 #m
l2 = 0.2475 #m

frequency = 0.5
amplitude = 0.4

Kp = 2.5 
Kd = 1
    

# Mujoco Simulation
with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()

  # qpos_prev = 0

  while viewer.is_running():

    # qpos_prev = d.qpos[0]
           
    step_start = time.time()

    #q_des come from sinusoidal wave
    hip_y = max(-1,(amplitude * math.sin(2 * np.pi * frequency * step_start)))


     # Check if the desired position is within reachable limits
    if hip_y > l1 + l2:
        raise ValueError("Desired hip position out of reachable range")
    
    #would a negative hip_y be 0 - hip_y then
    # if hip_y < 0:
    #     hip_y = 0 - hip_y

    # Calculate theta1 using the law of cosines
    cos_theta1 = (l2**2 - l1**2 - hip_y**2) / (-2 * l1 * hip_y)
    theta1 = np.arccos(np.clip(cos_theta1, -1.0, 1.0))
    hip_angle = theta1

    # Calculate theta2 using the law of cosines
    cos_theta2 = (hip_y**2 - l1**2 - l2**2) / (-2 * l1 * l2)
    theta2 = np.arccos(np.clip(cos_theta2, -1.0, 1.0))
    knee_angle = -(np.pi - theta2)

    # Calculate theta1 based on theta2 and y_des
    # dq_des = d.qpos[0] - qpos_prev
    # dq_des = amplitude * 2 * np.py * frequency * math.cos(2 * np.py * frequency * step_start)

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    # d.ctrl[0] =  (Kp*(theta1 - d.qpos[1])) + (Kd * (-d.qvel[1]))
    # d.ctrl[1] = -((Kp*(2 * q_des - d.qpos[1])) + (Kd * (-d.qvel[1])))

    # Move right if q_des is positive
    if hip_y > 0:
         # Apply control to bring it back to zero (origin) without overshooting
        d.ctrl[1] = ((Kp * (knee_angle - d.qpos[2])) + (Kd * (-d.qvel[2])))
        d.ctrl[0] =  (Kp*(hip_angle - d.qpos[1])) + (Kd * (-d.qvel[1]))
        # d.ctrl[0] = (Kp_2 * (0 - d.qpos[0])) + (Kd * (-d.qvel[0]))
    # Return to origin if q_des is zero
    else:
        d.ctrl[1] = ((Kp * (0 - d.qpos[2])) + (Kd * (-d.qvel[2])))
        d.ctrl[0] =  (Kp*(0 - d.qpos[1])) + (Kd * (-d.qvel[1]))
       
        # d.ctrl[0] = (Kp_2 * (q_des - d.qpos[0])) + (Kd * (-d.qvel[0]))
    #   # Proportional Gain
    # Kp = np.array([200, 200, 200, 100, 100, 100])
    
    # # Derivative Gain
    # Kd = np.array([10, 10, 10, 10, 10, 10])
    
    # Number of Joints
    # nj = d.qpos.shape[0]
    # # Torques
    # torques = np.zeros(nj)
    # # Loop through each joint
    # for i in range(nj):
    #     # Compute the Torque
    #     torques[i] = (Kp[i]*(q_des[i] - d.qpos[i])) + (Kd[i] * (dq_des[i] - d.qvel[i]))

    #how to assign something to hip and knee
    #create a PD controller so gain is good
    #plotting hip properly

    mujoco.mj_step(m, d)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)