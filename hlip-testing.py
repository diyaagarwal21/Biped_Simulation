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

mujoco.mj_forward(m, d)

print("COM", d.subtree_com[hip_id])
print("right foot", d.site_xpos[rfoot_id])
print("left foot", d.site_xpos[lfoot_id])

# Prints all the joint names
for i in range(m.njnt):
    print(i, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i))

def swing_trajectory(start, end, phase):
    pos = start + phase*(end-start)

    # add foot clearance
    pos[2] += 0.05*np.sin(np.pi*phase)

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

def main():
    # Mujoco Simulation
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        desired_log = []
        actual_log = []

        d.qpos[0] = 0.8     # hip
        d.qpos[1] = -1     # knee

        mujoco.mj_forward(m, d)

        q_des = d.qpos.copy()

        start_pos = d.site_xpos[rfoot_id].copy()
        target_pos = np.array([0.1, 0.0525, 0.6])

        plt.ion()

        fig, ax = plt.subplots()

        line_desired, = ax.plot([], [], '--', label="desired")
        line_actual, = ax.plot([], [], label="actual")

        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.legend()
        ax.axis("equal")

        while viewer.is_running():
            step_duration = 8  # seconds
                
            # phase = min((time.time() - start) / step_duration, 1.0)            
            # target = swing_trajectory(start_pos, target_pos, phase)       

            # trying a circle trajectory
            phase = ((time.time() - start) / step_duration) % 1.0
            center = start_pos.copy()
            radius = 0.01
            target = circle_trajectory(center, radius, phase)   
            print(m.opt.timestep)  

            # target = line_trajectory(center, 0.03, phase)
            ###################


            foot_pos = d.site_xpos[rfoot_id].copy()
            error = target - foot_pos
            jacp = np.zeros((3, m.nv))
            jacr = np.zeros((3, m.nv))

            print("||error|| =", np.linalg.norm(error))


            mujoco.mj_jacSite(
                m,
                d,
                jacp,
                jacr,
                rfoot_id
            )

            K = 20
            # dq = K*np.linalg.pinv(jacp) @ error
            # dq = np.clip(dq, -0.05, 0.05)
            # q_des += dq
            dt = m.opt.timestep
            error = np.array([
                target[0] - foot_pos[0],   # x error
                target[2] - foot_pos[2]    # z error
            ])

            J = jacp[[0, 2], 0:2]
            print("J: ", J)

            print("cond(J):", np.linalg.cond(J))

            lam = 0.01
            J_pinv = J.T @ np.linalg.inv(J @ J.T + lam**2 * np.eye(2))
            qdot = K * J_pinv @ error
            qdot = np.clip(qdot, -2.0, 2.0)
            q_des[:2] += qdot * dt

            com = d.subtree_com[hip_id].copy()

            print("COM:", com)

            d.ctrl[:] = q_des
            # d.ctrl[0] = 0.5 * np.sin(2*np.pi*0.5*(time.time()-start))
            # d.ctrl[1] = -0.5 * np.sin(2*np.pi*0.5*(time.time()-start))
            # d.ctrl[0] = 0.3      # hip
            # d.ctrl[1] = -0.6     # knee
            # print("target:", target)
            # print("foot:", foot_pos)
            # print("error:", error)
            print("q_des:", q_des)
            print("q_actual:", d.qpos[:4])
            print("qdot:", qdot)
    

            desired_log.append(target.copy())
            actual_log.append(foot_pos.copy())

            mujoco.mj_step(m, d)
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

            desired = np.array(desired_log)
            actual = np.array(actual_log)

            print("ACtuator force: ", d.actuator_force)

            line_desired.set_data(
                desired[:,0],
                desired[:,2]
            )

            line_actual.set_data(actual[:,0],actual[:,2])

            ax.relim()
            ax.autoscale_view()

            fig.canvas.draw()
            fig.canvas.flush_events()


if __name__ == "__main__":
    main()