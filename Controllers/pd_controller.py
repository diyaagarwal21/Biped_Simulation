import numpy as np

class PDController:
    def __init__(self, kp, kd, add_feedforward=True):
        self.Kp = np.asarray(kp, dtype=float)
        self.Kd = np.asarray(kd, dtype=float)
        self.add_feedforward = add_feedforward

    def compute(self, p_ref, v_ref, p_measured, v_measured, uff, a_ref):
        p_ref = p_ref[1:]
        v_ref = v_ref[1:]
        p_measured = p_measured[1:]
        v_measured = v_measured[1:]

        position_error = np.asarray(p_ref) - np.asarray(p_measured)
        velocity_error = np.asarray(v_ref) - np.asarray(v_measured)

        ufb = self.Kp @ position_error + self.Kd @ velocity_error
        u = uff + ufb

        return ufb
