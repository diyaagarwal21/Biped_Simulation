import math
import time
import mujoco
import mujoco.viewer
import numpy as np
import matplotlib.pyplot as plt

m = mujoco.MjModel.from_xml_path('STLs/scene.xml')
d = mujoco.MjData(m)   # simulation state

# gets right foot id
rfoot_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_foot_contact")
# gets left foot id
lfoot_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_foot_contact")
# gets Hip ID
hip_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Hip")

d.qpos[0] = 0.4
# d.qpos[1] = -0.6

mujoco.mj_forward(m, d)

print("COM", d.subtree_com[hip_id])
print("right foot", d.site_xpos[rfoot_id])
print("left foot", d.site_xpos[lfoot_id])

# Prints all the joint names
for i in range(m.njnt):
    print(i, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i))

def swing_trajectory(start, end, phase, height):
    pos = start + phase * (end - start)

    pos[2] += height * np.sin(np.pi * phase)

    return pos

def circle_trajectory(center, radius, phase):
    theta = 2 * np.pi * phase

    pos = center.copy()

    pos[0] += radius * np.cos(theta)   # x
    pos[2] += radius * np.sin(theta)   # z

    return pos

def line_trajectory(center, amp, phase):
    pos = center.copy()
    pos[0] += amp * np.sin(2*np.pi*phase)
    return pos

# Generates a swing foot trajectory and returns the target position.
def foot_trajectory(start_pos, landing_x, phase):
    target = start_pos.copy()  # (3x1) array

    # x trajectory: linear interpolation (Can change later)
    target[0] = start_pos[0] + phase * (landing_x - start_pos[0])

    # z trajectory
    swing_height = 0.05   # in m
    target[2] = start_pos[2] + swing_height * np.sin(np.pi * phase)

    return target

# returns qdot (joint velocity)
def ik(target_swing, target_com_z, swing_foot):
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

    # stack jacobians and errors
    J = np.vstack([jacp[0,:], jacp[2,:], jac_com[2,:]])
    error = np.array([error_swing[0], error_swing[2], z_error])
    # J = np.vstack([jacp[0,:], jacp[2,:]])
    # error = np.array([error_swing[0], error_swing[2]])
    print("||error|| =", np.linalg.norm(error))
    print("Jacobian: ", J)

    J = J[:,3:7]

    # pseudoinverse
    # Jpinv = np.linalg.pinv(J)

    # damped inverse
    lam = 0.01
    Jpinv = J.T @ np.linalg.inv(J @ J.T + lam**2 * J.shape[0])

    # get qdot
    K = 15
    qdot = Jpinv @ (K*error)
    # print("qdot =",qdot)
    return qdot

def get_hlip_orbit():
    g = 9.81
    z0 = 0.356
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
        dt = m.opt.timestep
        v = (p-prev_p)/dt
    return p, v

    
def main():
    with mujoco.viewer.launch_passive(m, d) as viewer:

        mujoco.mj_forward(m, d)
        dt = m.opt.timestep
        q_cmd = d.qpos[3:7].copy()

        # start_pos = d.site_xpos[rfoot_id].copy()

        # step 1: Fixed target for IK
        # target = np.array([
        #     start_pos[0] + 0.05,   # 5 cm forward
        #     start_pos[1],
        #     start_pos[2] + 0.03    # 3 cm upward
        # ])

        # step 2: foot trajectory
        # swing_start_time = time.time()
        # T = 5
        # landing_x = start_pos[0] + 0.08

        # step 3: switching feet
        # Initial stance/swing feet
        stance_foot = lfoot_id
        swing_foot = rfoot_id
        # Initial swing
        swing_start_time = time.time()
        T = 0.5
        start_pos = d.site_xpos[swing_foot].copy()
        landing_x = start_pos[0] + 0.08

        desired_log = []
        actual_log = []

        plt.ion()
        fig, ax = plt.subplots()

        line_desired, = ax.plot([], [], '--', label="Desired")
        line_actual, = ax.plot([], [], label="Actual")

        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.legend()
        ax.axis("equal")

        while viewer.is_running():

            # step 2: foot trajectory
            phase = np.clip((time.time() - swing_start_time) / T, 0, 1)
            target = foot_trajectory(start_pos,landing_x,phase)

            print("phase =", phase)
            print("target =", target)

            # Current foot position
            foot_pos = d.site_xpos[swing_foot].copy()

            # IK
            qdot = ik(target, 0.356, swing_foot)

            # Integrate joints directly (pure kinematics)
            # d.qpos[3:7] += qdot *dt
            q_cmd += qdot *dt
            d.ctrl[0] = q_cmd[0]
            d.ctrl[1] = q_cmd[1]
            d.ctrl[2] = q_cmd[2]
            d.ctrl[3] = q_cmd[3]

            # Update forward kinematics
            # mujoco.mj_forward(m, d)
            mujoco.mj_step(m, d)

            # Finished this swing?
            if phase >= 1.0:
                # Switch stance and swing feet
                stance_foot, swing_foot = swing_foot, stance_foot
                # New swing begins now
                swing_start_time = time.time()
                # Starting point of the new swing
                start_pos = d.site_xpos[swing_foot].copy()
                # Temporary fixed step length
                landing_x = start_pos[0] + 0.08

            # Logging
            foot_pos = d.site_xpos[swing_foot].copy()
            error = target - foot_pos
            print("contacts:", d.ncon)

            print("q:", d.qpos[:4])
            print("qdot:", qdot)
            print("target:", target)
            print("foot:", foot_pos)
            print("error:", error)
            print("||error||:", np.linalg.norm(error))
            print()

            desired_log.append(target.copy())
            actual_log.append(foot_pos.copy())

            desired = np.array(desired_log)
            actual = np.array(actual_log)

            line_desired.set_data(desired[:, 0], desired[:, 2])
            line_actual.set_data(actual[:, 0], actual[:, 2])

            ax.relim()
            ax.autoscale_view()

            fig.canvas.draw()
            fig.canvas.flush_events()

            viewer.sync()

            time.sleep(dt)

if __name__ == "__main__":
    main()