import numpy as np
import pinocchio as pin
from scipy.interpolate import interp1d


class TVLQR:
    def __init__(self, biped, Xref, Uref, Q, R, step_duration, N):
        self.biped = biped
        self.data = biped.createData()
        self.Xref = Xref
        self.Uref = Uref
        self.Q = Q
        self.R = R
        self.step_duration = step_duration
        self.N = N
        self.m = Uref.shape[1]
        self.nq = biped.nq

        self.t_grid = np.linspace(0.0, step_duration, self.N)
        self.dt_grid = np.diff(self.t_grid)

        self.S = np.zeros((self.nq, self.m))
        self.S[1:1 + self.m, :] = np.eye(self.m)

        self.K = self._compute_K()
        K_zh = self.K.reshape(self.N, -1)
        self.K_reconstructed = interp1d(self.t_grid, K_zh, axis=0, kind="previous", bounds_error=False, fill_value=(K_zh[0], K_zh[-1]))

        self.Pu = np.array([[0,0,0,1],
                            [0,0,1,0],
                            [0,1,0,0],
                            [1,0,0,0]])
        Relabel = np.array([[1,1,1,1,1],
                        [0,0,0,0,-1],
                        [0,0,0,-1,0],
                        [0,0,-1,0,0],
                        [0,-1,0,0,0]])
        self.Px = np.block([[Relabel, np.zeros_like(Relabel)],
                            [np.zeros_like(Relabel), Relabel]])

    def _compute_K(self):
        K = np.zeros((self.N, self.m, self.nq*2))
        P = self.Q.copy()

        for k in range(self.N-2, -1, -1):
            p_k, v_k = self.Xref[k, :self.nq], self.Xref[k, self.nq:]
            u_k = self.Uref[k, :]
            dt_k = self.dt_grid[k] if k < self.N - 1 else self.dt_grid[-1]

            Ak, Bk = self._linearize(p_k, v_k, u_k)
            A_d = np.eye(self.nq*2) + Ak * dt_k
            B_d = Bk * dt_k

            S_gain = self.R + B_d.T @ P @ B_d
            K_k = np.linalg.solve(S_gain, B_d.T @ P @ A_d)
            P = self.Q + A_d.T @ P @ A_d - A_d.T @ P @ B_d @ K_k
            K[k] = K_k
            
        return K

    def _linearize(self, p, v, u):
        # gives A and B in evaluation point
        tau = self.S @ u
        pin.computeABADerivatives(self.biped, self.data, p, v, tau)
        ddp_dp = self.data.ddq_dq
        ddp_dv = self.data.ddq_dv
        ddp_dtau = self.data.Minv @ self.S

        A = np.zeros((self.nq*2, self.nq*2))
        A[:self.nq, self.nq:] = np.eye(self.nq)
        A[self.nq:, :self.nq] = ddp_dp
        A[self.nq:, self.nq:] = ddp_dv

        B = np.zeros((self.nq*2, self.m))
        B[self.nq:, :] = ddp_dtau
        return A, B

    def _gain_at_t(self, t_phase, step_idx):
        t_phase = np.clip(t_phase, self.t_grid[0], self.t_grid[-1])
        K = self.K_reconstructed(t_phase).reshape(self.m, 2*self.nq)

        if step_idx % 2 == 0:
            # print("step 1")
            return K
        else:
            # print("step 2")
            return self.Pu @ K @ self.Px
    

    def compute(self, p_ref, v_ref, p_measured, v_measured, uff, a_ref, t_phase, step_idx):
        e_x = np.hstack([p_ref, v_ref]) - np.hstack([p_measured, v_measured])
        K_t = self._gain_at_t(t_phase, step_idx)

        ufb = K_t @ e_x
        u = uff + ufb
        return u