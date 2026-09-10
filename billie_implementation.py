import time
import mujoco
import mujoco.viewer
import numpy as np
import pinocchio as pin
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, CubicHermiteSpline

from Controllers.pd_controller import PDController
from Controllers.feedback_linearization_controller import FeedbackLinearization
from Controllers.linear_quadratic_regulizer_controller import TVLQR


# 1. Initilialize model in mujoco
controller = "PD" 
# choose "PD", "FBL", "TVLQR"
m = mujoco.MjModel.from_xml_path('STLs/scene.xml')
d = mujoco.MjData(m)

joint_names = ["knee_joint_l", "hip_joint_l", "hip_joint_r", "knee_joint_r"]
joint_ids = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name) for name in joint_names])
hip_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Hip")

joint_qpos_adr = m.jnt_qposadr[joint_ids]
joint_qvel_adr = m.jnt_dofadr[joint_ids]


mujoco.mj_forward(m,d)

# Load pinocchio model and contstruct desired trajectories
urdf_path = r"C:\Users\julie\OneDrive - Georgia Institute of Technology\courses\Capstone\Code\Biped_Simulation\Controllers\URDF\biped_billie_linkage_pinned.urdf"
biped = pin.buildModelFromUrdf(str(urdf_path))
pin_d = biped.createData()
step_duration = 0.2
n_steps = 2

if n_steps == 1:
    x_ref = np.load(r"C:\Users\julie\OneDrive - Georgia Institute of Technology\courses\Capstone\Code\Biped_Simulation\Data\Xsol_0.2_periodic_0.2_wheight.npy")
    x_ref[:, 1] *= -1
    x_ref[:, 2] *= -1 
    x_ref[:, 6] *= -1 
    x_ref[:, 7] *= -1
    x1 = x_ref
    N_step = (len(x_ref) + 1) // n_steps
    dt = np.linspace(0.0, step_duration*n_steps, x_ref.shape[0])
    x_reconstruct1 = CubicHermiteSpline(dt, x_ref[:, :5], x_ref[:, 5:], axis=0)
    a_ref = x_reconstruct1(dt, 2)

    tau = np.zeros((x_ref.shape[0], biped.nv))
    for k in range(x_ref.shape[0]):
        tau[k] = pin.rnea(biped, pin_d, x_ref[k, :5], x_ref[k, 5:], a_ref[k])
    u1 = tau[:, 1:]
    # u1 = np.load(r"C:\Users\julie\OneDrive - Georgia Institute of Technology\courses\Capstone\Code\Biped_Simulation\Data\Usol_0.2_periodic_0.2_wheight.npy")
    u_reconstruct1 = CubicSpline(dt, u1, axis=0)

if n_steps == 2:
    x_ref = np.load(r"C:\Users\julie\OneDrive - Georgia Institute of Technology\courses\Capstone\Code\Biped_Simulation\Data\Xsol_full_0.2_periodic_0.2_wheight.npy")
    x_ref[:, 1] *= -1
    x_ref[:, 2] *= -1 
    x_ref[:, 6] *= -1 
    x_ref[:, 7] *= -1
    N_step = (len(x_ref) + 1) // n_steps
    x1 = x_ref[:N_step]
    x2 = x_ref[N_step:]

    dt1 = np.linspace(0, step_duration, len(x1))
    dt2 = np.linspace(0, step_duration, len(x2))
    x_reconstruct1 = CubicHermiteSpline(dt1, x1[:, :5], x1[:, 5:], axis=0)
    x_reconstruct2 = CubicHermiteSpline(dt2, x2[:, :5], x2[:, 5:], axis=0)

    a1 = x_reconstruct1(dt1, 2)
    a2 = x_reconstruct2(dt2, 2)
    tau1 = np.zeros((x1.shape[0], biped.nv))
    tau2 = np.zeros((x2.shape[0], biped.nv))
    for k in range(x1.shape[0]):
        tau1[k] = pin.rnea(biped, pin_d, x1[k, :5], x1[k, 5:], a1[k])
        tau2[k] = pin.rnea(biped, pin_d, x2[k, :5], x2[k, 5:], a2[k])
    u1 = tau1[:, 1:]
    u2 = tau2[:, 1:]
    u_reconstruct1 = CubicSpline(dt1, u1, axis=0)
    u_reconstruct2 = CubicSpline(dt2, u2, axis=0)

    x_ref = x_ref


# 3. Initialize pinocchio controller
torso_id = biped.getFrameId("Torso")

# PD controller
if controller == "PD":
    Kp = np.array([
            [2, 0,   0,  0],
            [0, 2, 0,  0],
            [0, 0,  2,  0],
            [0, 0,   0, 2]])
    Kd = np.array([
            [0.1, 0,  0,  0],
            [0,  0.1, 0,  0],
            [0,  0, 0.1,  0],
            [0,  0,  0, 0.1]])
    C = PDController(Kp, Kd)

elif controller == "FBL":
    # kp = np.diag([70,50,150,250])
    # kd = np.diag([10, 10, 10, 50])
    kp = np.diag([150,150,150,250])
    kd = np.diag([10, 10, 10, 50])
    C = FeedbackLinearization(biped, kp, kd)

elif controller == "TVLQR":
    Q = np.diag([900, 900, 900, 900, 900,
                 200, 200, 200, 200, 200])
    R = np.eye(x_ref.shape[1]//2 - 1) * 0.01
    C = TVLQR(biped, x1, u1, Q, R, step_duration, N_step)

# helper functions
def _mujoco_joint_states():
    q = {}
    dq = {}
    for name in joint_names:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        q[name] = d.qpos[m.jnt_qposadr[jid]]
        dq[name] = d.qvel[m.jnt_dofadr[jid]]

    return q, dq

def _mujoco_to_pin_mapping():
    q, dq = _mujoco_joint_states()
    pitch_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "base_pitch")
    base_pitch = d.qpos[m.jnt_qposadr[pitch_id]]
    base_pitch_dot = d.qvel[m.jnt_dofadr[pitch_id]]

    q_pin = np.zeros(5)
    q_pin[1] = q["knee_joint_l"]
    q_pin[2] = q["hip_joint_l"]
    q_pin[3] =  q["hip_joint_r"]
    q_pin[4] =  q["knee_joint_r"]
    q_pin[0] = - base_pitch + q_pin[1] + q_pin[2]

    v_pin = np.zeros(5)
    v_pin[1] = dq["knee_joint_l"]
    v_pin[2] = dq["hip_joint_l"]
    v_pin[3] =  dq["hip_joint_r"]
    v_pin[4] =  dq["knee_joint_r"]
    v_pin[0] = - base_pitch_dot + v_pin[1] + v_pin[2]

    return q_pin, v_pin

def _pin_to_mujoco_mapping(u_pin):
    # [LK, LH, RH, RK] -> [RH, RK, LH, LK]
    u_mj = np.zeros(4)
    u_mj[0] =  u_pin[2]
    u_mj[1] =  u_pin[3]
    u_mj[2] = u_pin[1]
    u_mj[3] = u_pin[0]
    return u_mj

def _base_state(q_pin, z0_mj):
    # get [base_x, base_z, base_pitch] for the mujoco model from the pinocchio states
    pin.forwardKinematics(biped, pin_d, q_pin)
    pin.updateFramePlacements(biped, pin_d)

    foot_x = 0.0
    foot_z = 0.00
    T = pin_d.oMf[torso_id]
    R = T.rotation

    base_x = foot_x + T.translation[0]
    base_z = foot_z + T.translation[2] - z0_mj
    base_pitch = np.arctan2(R[0, 2], R[0, 0])

    return base_x, base_z, base_pitch

def interpolation(t):
    step_idx = min(int(t // step_duration), n_steps - 1)
    t_phase = t - step_idx * step_duration
    t_phase = np.clip(t_phase, 0.0, step_duration)
    if step_idx % 2 == 0:
        x_reconstruct = x_reconstruct1
        u_reconstruct = u_reconstruct1
    else:
        x_reconstruct = x_reconstruct2
        u_reconstruct = u_reconstruct2
        
    p_ref = x_reconstruct(t_phase)
    v_ref = x_reconstruct(t_phase, 1)
    a_ref = x_reconstruct(t_phase, 2)
    u_ref = u_reconstruct(t_phase)
    return  p_ref, v_ref, u_ref, a_ref

class Logger:
    def __init__(self):
        self.time = []
        self.x_ref = []
        self.x_real = []
        self.u_ref = []
        self.u_real = []

    def log(self, t, x_ref, x_real, u_ref, u_real):
        self.time.append(t)
        self.x_ref.append(np.asarray(x_ref).copy())
        self.x_real.append(np.asarray(x_real).copy())
        self.u_ref.append(np.asarray(u_ref).copy())
        self.u_real.append(np.asarray(u_real).copy())

    def reset(self):
        self.__init__()

    def plot(self):
        time = np.asarray(self.time)
        x_ref = np.asarray(self.x_ref)
        x_real = np.asarray(self.x_real)
        u_ref = np.asarray(self.u_ref)
        u_real = np.asarray(self.u_real)

        state_names = ["Left ankle", "Left knee", "Left hip", "Right hip", "Right knee"]

        actuator_names = ["Left knee", "Left hip", "Right hip", "Right knee"]

        for i, name in enumerate(state_names):
            plt.figure(figsize=(10, 4))
            plt.plot(time, x_ref[:, i], label="Reference")
            plt.plot(time, x_real[:, i], "--", label="Measured")
            plt.xlabel("Time [s]")
            plt.ylabel("Position [rad]")
            plt.title(name)
            plt.grid()
            plt.legend()

        for i, name in enumerate(actuator_names):
            plt.figure(figsize=(10, 4))
            plt.plot(time, u_ref[:, i], label="Feedforward reference")
            plt.plot(time, u_real[:, i], "--", label="Applied torque")
            plt.xlabel("Time [s]")
            plt.ylabel("Torque [Nm]")
            plt.title(f"{name} torque")
            plt.grid()
            plt.legend()

        plt.show()

# 4. Set and save initial state to loop simulation
q0_pin = x_ref[0, :5]
d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "hip_joint_r")]] =  q0_pin[3]
d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "knee_joint_r")]] =  q0_pin[4]
d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "hip_joint_l")]] = q0_pin[2]
d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "knee_joint_l")]] = q0_pin[1]

v0_pin = x_ref[0, 5:]
d.qvel[m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "hip_joint_r")]] =  v0_pin[3]
d.qvel[m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "knee_joint_r")]] =  v0_pin[4]
d.qvel[m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "hip_joint_l")]] = v0_pin[2]
d.qvel[m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "knee_joint_l")]] = v0_pin[1]

z0_mj = m.body_pos[hip_id, 2] #- 0.13
b_x0, b_z0, b_p0 = _base_state(q0_pin, z0_mj)
d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "base_x")]] = b_x0
d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "base_z")]] = b_z0
d.qpos[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "base_pitch")]] = b_p0
eps = 1e-5
b_x1, b_z1, b_p1 = _base_state(q0_pin + eps * v0_pin, z0_mj)
# base_v = np.array([(b_x1 - b_x0) / eps, (b_z1 - b_z0) / eps, (b_p1 - b_p0) / eps])
base_v = np.array([(b_x1 - b_x0) / eps, 0.0 / eps, (b_p1 - b_p0) / eps])
for name, value in zip(["base_x", "base_z", "base_pitch"], base_v):
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
    d.qvel[m.jnt_dofadr[jid]] = value

mujoco.mj_forward(m, d)
q_pin_rt, v_pin_rt = _mujoco_to_pin_mapping()
initial_qpos = d.qpos.copy()
initial_qvel = d.qvel.copy()

p_ref, v_ref, uff, a_ref = interpolation(0.0)

# Pinocchio acceleration
tau_pin = np.zeros(biped.nv)
tau_pin[1:] = uff
qdd_pin = pin.aba(biped, pin_d, p_ref, v_ref, tau_pin)

# MuJoCo acceleration
d.ctrl[:] = _pin_to_mujoco_mapping(uff)
mujoco.mj_forward(m, d)

# Map MuJoCo joint accelerations to Pinocchio convention
qdd_mj_pin = np.zeros(5)

pitch_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "base_pitch")
base_pitch_dd = d.qacc[m.jnt_dofadr[pitch_id]]

# 5. Simulate the controlled gait
def main():
    logger = Logger()
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():

            if d.time >= n_steps * step_duration:
                # logger.plot()
                mujoco.mj_resetData(m, d)
                d.qpos[:] = initial_qpos
                d.qvel[:] = initial_qvel
                mujoco.mj_forward(m, d)
                logger.reset()
                break

            step_idx = int(d.time // step_duration)
            t_phase = d.time % step_duration # 0: left stance leg / 1: right stance leg

            p_measured, v_measured = _mujoco_to_pin_mapping()
            p_ref, v_ref, uff, a_ref = interpolation(d.time)
            mujoco.mj_forward(m, d)

            if controller == "PD":
                u_pin = C.compute(p_ref, v_ref, p_measured, v_measured, uff, a_ref)

            elif controller == "TVLQR" or controller == "FBL":
                u_pin = C.compute(p_ref, v_ref, p_measured, v_measured, uff, a_ref, t_phase, step_idx)

            u_pin = np.clip(u_pin, -0.7, 0.7)
            u_mj = _pin_to_mujoco_mapping(u_pin) # tf control back to mujoco convention, expected order [RH, RK, LH, LK]

            d.ctrl[:] = u_mj

            mujoco.mj_step(m, d)
            logger.log(d.time, p_ref, p_measured, uff, u_pin)
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()