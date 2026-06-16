import math
import time
import mujoco
import mujoco.viewer
import numpy as np

m = mujoco.MjModel.from_xml_path('STLs/scene.xml')
d = mujoco.MjData(m)

d.qpos[0] = 0.5

frequency = 0.5
amplitude = 2

Kp = 0

Kd = 0.2
    

# Mujoco Simulation
with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()

  # qpos_prev = 0

  while viewer.is_running():

    # qpos_prev = d.qpos[0]
           
    step_start = time.time()

    #q_des come from sinusoidal wave
    q_des = (amplitude * math.sin(2 * np.pi * frequency * step_start))
    # dq_des = d.qpos[0] - qpos_prev
    # dq_des = amplitude * 2 * np.py * frequency * math.cos(2 * np.py * frequency * step_start)

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.


    d.ctrl[0] =  (Kp*(q_des - d.qpos[1])) + (Kd * (-d.qvel[1]))
    # d.ctrl[1] = -((Kp*(2 * q_des - d.qpos[1])) + (Kd * (-d.qvel[1])))

    # Move right if q_des is positive
    if q_des > 0:
        d.ctrl[1] = ((Kp * (0 - d.qpos[2])) + (Kd * (-d.qvel[2])))
        # d.ctrl[0] = (Kp_2 * (0 - d.qpos[0])) + (Kd * (-d.qvel[0]))
    # Return to origin if q_des is zero
    else:
        # Apply control to bring it back to zero (origin) without overshooting
        d.ctrl[1] = ((Kp * (q_des - d.qpos[2])) + (Kd * (-d.qvel[2])))
        # d.ctrl[0] = (Kp_2 * (q_des - d.qpos[0])) + (Kd * (-d.qvel[0]))


    # d.ctrl[0] =  (Kp*(q_des - d.qpos[1])) + (Kd * (-d.qvel[1]))
    # d.ctrl[1] = -((Kp*(2 * q_des - d.qpos[1])) + (Kd * (-d.qvel[1])))

    # # Move right if q_des is positive
    # if q_des > 0:
    #     d.ctrl[1] = - ((Kp * (0 - d.qpos[2])) + (Kd * (-d.qvel[2])))
    #     # d.ctrl[0] = (Kp_2 * (0 - d.qpos[0])) + (Kd * (-d.qvel[0]))
    # # Return to origin if q_des is zero
    # else:
    #     # Apply control to bring it back to zero (origin) without overshooting
    #     d.ctrl[1] = - ((Kp * (q_des - d.qpos[2])) + (Kd * (-d.qvel[2])))
    
    # if q_des < 0:
    #   # Apply control to move towards q_des, but only if the deviation is large enough
    #   if abs(q_des - d.qpos[0]) > threshold:
    #       d.ctrl[0] = (Kp * (q_des - d.qpos[0])) + (Kd * (-d.qvel[0]))
    #   else:
    #       d.ctrl[0] = 0  # No control needed if close to desired position
    # # Return to origin if q_des is zero
    # else:
    #   # Apply control to bring it back to zero (origin) without overshooting
    #   if abs(d.qpos[0]) > threshold:
    #       d.ctrl[0] = (Kp * (0 - d.qpos[0])) + (Kd * (-d.qvel[0]))
    #   else:
    #       d.ctrl[0] = 0  # No control needed if close to the origin

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