import math
import time
import mujoco
import mujoco.viewer
import numpy as np

m = mujoco.MjModel.from_xml_path('STLs/scene.xml')
d = mujoco.MjData(m)   # simulation state

z0 = 0.304
# d.qpos[0] = 0
# # d.qpos[2] = 0
# d.qpos[1] = 0.395

# d.qpos[3] = -0.8
# d.qpos[4] = 1.2

# d.qpos[5] = -0.8
# d.qpos[6] = 1.2

# gets right foot id
rfoot_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_foot_contact")
# gets left foot id
lfoot_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_foot_contact")
# gets Hip ID
hip_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Hip")

mujoco.mj_forward(m, d)

print("COM", d.subtree_com[hip_id])
print("right foot", d.site_xpos[rfoot_id])
print("left foot", d.site_xpos[lfoot_id])

# Prints all the joint names
for i in range(m.njnt):
    print(i, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i))

# returns qdot (joint velocity)
def ik(target_swing, target_com_z, target_stance_x, swing_foot, stance_foot):
    # get current foot x position
    swing_pos = d.site_xpos[swing_foot].copy()
    error_swing = target_swing - swing_pos
    # get jacobian from mujoco (jacsite --> populates jacp and jacr)
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jacp, jacr, swing_foot)

    # z COM constraint
    com = d.subtree_com[hip_id]
    z_error = target_com_z - com[2]
    jac_com = np.zeros((3, m.nv))
    mujoco.mj_jacSubtreeCom(m, d, jac_com, hip_id)

    # stance foot x position constraint
    # stance_x = d.site_xpos[stance_foot][0]
    # error_stance_x = target_stance_x - stance_x
    # jacp_stance = np.zeros((3, m.nv))
    # jacr_stance = np.zeros((3, m.nv))
    # mujoco.mj_jacSite(m, d, jacp_stance, jacr_stance, stance_foot)

    # get pitch angle stabilization
    hip_body = d.xmat[hip_id].reshape(3,3)
    # pitch angle from rotation matrix
    pitch = np.arctan2(-hip_body[2,0],np.sqrt(hip_body[0,0]**2 + hip_body[1,0]**2))
    pitch_des = 0.0
    pitch_error = pitch_des - pitch
    # numerical Jacobian for pitch
    J_pitch = np.zeros(m.nv)

    # MuJoCo rotational Jacobian of the pelvis
    jacp_body = np.zeros((3,m.nv))
    jacr_body = np.zeros((3,m.nv))
    mujoco.mj_jacBody(m,d,jacp_body,jacr_body,hip_id)
    # pitch corresponds to y-axis rotation
    J_pitch = jacr_body[1,:]

    # stack jacobians and errors
    # J = np.vstack([jacp[0,:], jacp[2,:], jac_com[2,:], jacp_stance[0,:]])
    # error = np.array([error_swing[0], error_swing[2], z_error, error_stance_x])
    J = np.vstack([jac_com[2,:], jacp[0,:], jacp[2,:], J_pitch])
    error = np.array([z_error, error_swing[0], error_swing[2], pitch_error])

    J = J[:,0:4]

    # pseudoinverse
    # Jpinv = np.linalg.pinv(J)

    # damped inverse
    lam = 0.005
    Jpinv = J.T @ np.linalg.inv(J @ J.T + lam**2 * np.eye(4))

    # get qdot
    K = 1
    qdot = Jpinv @ (K*error)
    # print("qdot =",qdot)
    return qdot

# Generates a swing foot trajectory and returns the target position.
def foot_trajectory(start_pos, landing_x, phase):
    target = start_pos.copy()  # (3x1) array

    # x trajectory: linear interpolation (Can change later)
    target[0] = start_pos[0] + phase * (landing_x - start_pos[0])

    # z trajectory
    swing_height = 0.11    # in m
    target[2] = start_pos[2] + swing_height * np.sin(np.pi * phase)

    return target

"""
    Gets the center of mass state (for HLIP --> [p,v])
    Returns:
    p --> horizontal COM position relative to stance foot
    v --> horizontal COM velocity
    """
def get_com_state(prev_p, step_duration):
    # Whole robot COM (world coordinates)
    com = d.subtree_com[hip_id].copy()

    # Current stance foot (left foot)
    foot = d.site_xpos[lfoot_id].copy()

    # Horizontal COM relative to stance foot
    p = com[0] - foot[0]

    # Velocity (differentiated)
    if prev_p is None:
        v = 0.0
    else:
        v = (p-prev_p)/step_duration
    return p, v

# UNUSUED
# def hlip(p,v):
#     g = 9.81
#     z0 = 0.304
#     T = 0.5
#     lamb = np.sqrt(g/z0)

#     c1 = 1/2 * (p + 1/lamb * v)
#     c2 = 1/2 * (p  - 1/lamb * v)
#     p_end = c1*np.exp(lamb*T) + c2*np.exp(-lamb*T)
#     v_end = lamb*(c1*np.exp(lamb*T) - c2*np.exp(-lamb*T))

def get_hlip_orbit():
    g = 9.81

    v_d = 0.05  # desired walking speed
    T_SSP = 0.2
    T_DSP = 0.0

    T = T_SSP + T_DSP
    lamb = np.sqrt(g/z0)

    sigma1 = lamb / np.tanh(T_SSP*lamb/2)  # orbital slope

    # p* and v* are the desired periodic orbit (hlip state)
    p_star = (v_d*T)/(2 + T_DSP*sigma1)
    v_star = sigma1*p_star

    u_star = v_d*T
    return p_star, v_star, u_star

# Returns u --> the step size (target)
def hlip_controller(p,v):
    Kp = -1
    Kv = 0.1
    p_star, v_star, u_star = get_hlip_orbit()
    x = np.array([p,v])
    x_star = np.array([p_star,v_star])
    K = np.array([Kp,Kv])

    # u is the stepping stabilization for the orbit
    u = u_star + K @ (x-x_star)
    return u

def main():
    # Mujoco Simulation
    with mujoco.viewer.launch_passive(m, d) as viewer:
        step_start = time.time()

        stance_foot = lfoot_id
        swing_foot = rfoot_id

        # gets initial right and left hip and knee position
        q_cmd = d.qpos[0:4].copy()

        dt = m.opt.timestep
        prev_p = None  # for COM state

        step_duration = 0.2     # seconds
        start_pos = d.site_xpos[swing_foot].copy()
        stance_target = d.site_xpos[stance_foot][0].copy()

        # calculate first HLIP step
        p, v = get_com_state(prev_p, dt)
        u = hlip_controller(p, v)

        stance_x = d.site_xpos[stance_foot][0]
        landing_x = stance_x + u

        print("FIRST STEP")
        print("u =", u)
        print("landing =", landing_x)

        while viewer.is_running():
            # step phase
            elapsed = time.time() - step_start
            phase = elapsed / step_duration

            # new step starts
            if phase >= 1.0:
                step_start = time.time()  # reset time
                phase = 0.0

                stance_foot, swing_foot = swing_foot, stance_foot

                # Current foot position becomes the start
                start_pos = d.site_xpos[swing_foot].copy()
                stance_target = d.site_xpos[stance_foot][0].copy()

                # HLIP based on current COM posiiton
                p, v = get_com_state(prev_p, step_duration)
                prev_p = p  # save current COM position for next step
                u = hlip_controller(p, v)  # get the next step size
            
                stance_x = d.site_xpos[stance_foot][0]
                landing_x = stance_x + u

                print("----------------")
                print("NEW STEP")
                print("swing foot =", swing_foot)
                print("stance foot =", stance_foot)
                print("p =",p)
                print("v =",v)
                print("u =",u)
                print("landing_x =",landing_x)
            
            # Calculate the swing foot trajectory
            target_swing = foot_trajectory(start_pos,landing_x,phase)

            # get qdot from ik(target) function
            qdot = ik(target_swing, z0, stance_target, swing_foot, stance_foot)  # (2, )

            # integrate to get position for the necessary joints
            # set position actuators for the necessary joint
            q_cmd += qdot * dt
            # d.ctrl[0] = q_cmd[0]
            # d.ctrl[1] = q_cmd[1]
            # d.ctrl[2] = q_cmd[2]
            # d.ctrl[3] = q_cmd[3]

            # d.ctrl[0] = 0.3   # right hip
            # d.ctrl[1] = -0.5  # right knee

            # d.ctrl[2] = -0.3  # left hip
            # d.ctrl[3] = -0.5  # left knee

            mujoco.mj_step(m, d)
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()