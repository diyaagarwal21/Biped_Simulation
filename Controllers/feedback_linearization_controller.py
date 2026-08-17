import numpy as np
import pinocchio as pin

# TODO: use mass matrix and everything from pinnochio


class FeedbackLinearization:
    def __init__(self, model, kp, kd):
        self.model = model
        self.data = model.createData()
        self.Kp = np.asarray(kp, dtype=float)
        self.Kd = np.asarray(kd, dtype=float)

        self.passive_idx = 0

    def compute(self, p_ref, v_ref, p_measured, v_measured, uff, a_ref, time):
        # mass matrix from pinocchio model
        M = np.zeros((self.model.nq, self.model.nq))
        M = pin.crba(self.model, self.data, p_measured)
        M = (M + M.T) / 2.0

        # nonlinear term from pinocchio model
        NL = pin.nonLinearEffects(self.model, self.data, p_measured, v_measured)

        M11 = M[0:1, 0:1]
        M12 = M[0:1, 1:]
        M21 = M[1:, 0:1]
        M22 = M[1:, 1:]

        NL1 = NL[0:1]
        NL2 = NL[1:]

        position_error = p_ref[1:] - p_measured[1:]
        velocity_error = v_ref[1:] - v_measured[1:]
        desired_acceleration = (a_ref[1:] + self.Kp @ position_error + self.Kd @ velocity_error)

        M11_inv_M12 = np.linalg.solve(M11, M12)
        M11_inv_NL1 = np.linalg.solve(M11, NL1)
        reduced_mass = M22 - M21 @ M11_inv_M12
        reduced_bias = NL2 - M21 @ M11_inv_NL1
        ufb = reduced_mass @ desired_acceleration + reduced_bias

        return ufb