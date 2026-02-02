
import torch
import numpy as np
from tqdm import tqdm

class GoogleEKF:
    """
    Python implementation of Android's SensorFusion.cpp (9-axis EKF).
    
    State:
        x0: Quaternion (W, X, Y, Z) - World to Body
        x1: Gyro Bias  (X, Y, Z)
        
    Covariance P (6x6):
        [0][0]: Orientation Error (3x3)
        [1][1]: Bias Error (3x3)
    """
    def __init__(self, device='cpu', 
                 gyro_var=1e-7, 
                 gyro_bias_var=1e-12, 
                 acc_stdev=0.015, 
                 mag_stdev=0.1):
        self.device = device
        
        # Parameters (from Fusion.cpp)
        self.GYRO_VAR = gyro_var      # rad^2/s^2
        self.GYRO_BIAS_VAR = gyro_bias_var # rad^2/s^3 (RW)
        self.ACC_STDEV = acc_stdev    # m/s^2
        self.MAG_STDEV = mag_stdev      # uT
        
        self.DEFAULT_BIAS = torch.zeros(3, device=device)
        self.reset()
        
    def reset(self):
        self.q = torch.tensor([1., 0., 0., 0.], device=self.device) # w,x,y,z
        self.b = torch.zeros(3, device=self.device)
        self.P = torch.zeros(6, 6, device=self.device)
        self.init_done = False
        
    def skew(self, v):
        """ Skew symmetric matrix from vector v """
        x, y, z = v
        return torch.tensor([
            [0, -z, y],
            [z, 0, -x],
            [-y, x, 0]
        ], device=self.device)
        
    def quat_to_matrix(self, q):
        # R(q) World -> Body
        w, x, y, z = q
        xx, yy, zz = x*x, y*y, z*z
        xy, xz, yz = x*y, x*z, y*z
        wx, wy, wz = w*x, w*y, w*z
        
        return torch.tensor([
            [1 - 2*(yy+zz), 2*(xy+wz),     2*(xz-wy)],
            [2*(xy-wz),     1 - 2*(xx+zz), 2*(yz+wx)],
            [2*(xz+wy),     2*(yz-wx),     1 - 2*(xx+yy)]
        ], device=self.device)
    
    def predict(self, w_meas, dt):
        """
        w_meas: Gyro measurement (rad/s)
        dt: Time step (s)
        """
        # 1. State Prediction
        # we = w - b
        we = w_meas - self.b
        
        # Analytical Quaternion Update (Exact integration of constant w)
        w_norm = torch.norm(we)
        if w_norm < 1e-6:
            k0 = 1.0 - w_norm**2 * dt**2 / 8.0
            k1 = 0.5 * dt * (1.0 - w_norm**2 * dt**2 / 24.0) # Taylor
        else:
            k0 = torch.cos(0.5 * w_norm * dt)
            k1 = torch.sin(0.5 * w_norm * dt) / w_norm
            
        dq = torch.tensor([k0, k1*we[0], k1*we[1], k1*we[2]], device=self.device)
        
        # q_next = q * dq (Standard multiplication)
        # Note: Fusion.cpp uses O(we)*q, which is matrix mult.
        # Here we do quat mult: q_new = q_old * dq_local
        # q = [w, v]
        qw, qv = self.q[0], self.q[1:]
        dqw, dqv = dq[0], dq[1:]
        
        new_w = qw*dqw - torch.dot(qv, dqv)
        new_v = qw*dqv + dqw*qv + torch.cross(qv, dqv)
        self.q = torch.cat([new_w.unsqueeze(0), new_v])
        
        if self.q[0] < 0: # Canonical form
            self.q = -self.q
        self.q = torch.nn.functional.normalize(self.q, dim=0)

        # 2. Covariance Prediction
        # Phi (6x6 State Transition Jacobian)
        # Phi = [ I   -dt*I ]
        #       [ 0    I    ]
        # (Simplified approximation from Fusion.cpp)
        
        Phi = torch.eye(6, device=self.device)
        Phi[0:3, 3:6] = -torch.eye(3, device=self.device) * dt
        
        # Process Noise Q (6x6)
        # Q_theta = GYRO_VAR * dt
        # Q_bias  = GYRO_BIAS_VAR * dt
        Q = torch.zeros(6, 6, device=self.device)
        Q[0:3, 0:3] = torch.eye(3, device=self.device) * (self.GYRO_VAR * dt)
        Q[3:6, 3:6] = torch.eye(3, device=self.device) * (self.GYRO_BIAS_VAR * dt)
        
        # P = Phi * P * Phi' + Q
        self.P = Phi @ self.P @ Phi.T + Q
        
    def update(self, z, Bi, sigma_r):
        """
        z: Measurement in Body Frame (Accel/Mag)
        Bi: Reference Vector in World Frame (Gravity/North)
        sigma: Measurement Noise Std Dev
        """
        # Predicted Measurement: Bb = R * Bi
        R = self.quat_to_matrix(self.q)
        Bb = R @ Bi
        
        # Residual e = z - Bb
        e = z - Bb
        
        # Sensitivity H = [ [Bb]x   0 ]
        # H is 3x6
        # Fusion.cpp: L = crossMatrix(Bb, 0)
        L = self.skew(Bb)
        H = torch.zeros(3, 6, device=self.device)
        H[0:3, 0:3] = L
        
        # Kalman Gain K = P * H' * (H * P * H' + R)^-1
        R_cov = torch.eye(3, device=self.device) * (sigma_r**2)
        S = H @ self.P @ H.T + R_cov
        
        # Robust Inverse
        try:
            S_inv = torch.linalg.inv(S)
        except:
            S_inv = torch.eye(3, device=self.device) * (1.0/(sigma_r**2))
            
        K = self.P @ H.T @ S_inv
        
        # Update State
        dx = K @ e
        
        dq_err = dx[0:3]
        db_err = dx[3:6]
        
        # Apply orientation correction
        # q_new = q * dq_err
        # dq_err is small rotation vector. q_err ~ [1, dq/2]
        dq_w_val = torch.sqrt(torch.clamp(1.0 - torch.norm(dq_err)**2 / 4.0, min=0.0))
        q_err = torch.cat([dq_w_val.unsqueeze(0), 0.5 * dq_err])
        
        # Multiply: q_next = q_prev * q_err_local?
        # AOSP: q += getF(q)*(0.5*dq)
        # getF(q) maps 3D vector to 4D quaternion derivative.
        # Effectively: q_new = q + 0.5 * q * (0, dq)
        # Yes.
        qw, qv = self.q[0], self.q[1:]
        dew, dev = 0.0, 0.5 * dq_err
        
        new_w = qw*dew - torch.dot(qv, dev)
        new_v = qw*dev + dew*qv + torch.cross(qv, dev)
        self.q = self.q + torch.cat([new_w.unsqueeze(0), new_v])
        self.q = torch.nn.functional.normalize(self.q, dim=0)
        
        # Apply bias correction
        self.b = self.b + db_err
        
        # Update Covariance
        # P = (I - K*H) * P
        I6 = torch.eye(6, device=self.device)
        self.P = (I6 - K @ H) @ self.P
        
        # Force Symmetry (AOSP does checkState)
        self.P = 0.5 * (self.P + self.P.T)
        
    def solve(self, time, gyro, accel, mag):
        """
        Process batches of data.
        time: (N,)
        gyro: (N, 3)
        accel: (N, 3)
        mag: (N, 3)
        Returns: q_pred (N, 4), bias_pred (N, 3)
        """
        N = len(time)
        q_out = []
        b_out = []
        
        # Initialize
        # AOSP calculates mean of first few samples to init R
        init_samples = min(50, N)
        a_mean = accel[:init_samples].mean(axis=0)
        m_mean = mag[:init_samples].mean(axis=0)
        
        a_mean /= np.linalg.norm(a_mean)
        m_mean /= np.linalg.norm(m_mean)
        
        # TRIAD / Cross Product Init
        # Up = Accel (Gravity) -> Vector pointing UP in Body if we hold it still?
        # No, Accel measures Reaction. Gravity is Down (0,0,-1) in World?
        # AOSP: Nominal Gravity = 9.81.
        # update(unityA, Ba, p). Ba = [0, 0, 1].
        # So they assume Accel = [0, 0, 1] (Reaction Force) when stationary upright.
        
        # Init Orientation R:
        # Up = a_mean.
        # East = cross(m_mean, Up).
        # North = cross(Up, East).
        # R = [East, North, Up].
        # This R maps World (ENU) to Body?
        # R * [1,0,0] = East (body).
        # So R columns are World bases in Body.
        # So R is World->Body.
        
        up = torch.tensor(a_mean, device=self.device, dtype=torch.float32)
        m_vec = torch.tensor(m_mean, device=self.device, dtype=torch.float32)
        east = torch.cross(m_vec, up)
        east = torch.nn.functional.normalize(east, dim=0)
        north = torch.cross(up, east)
        
        R_init = torch.stack([east, north, up], dim=1) # Columns
        # Convert matrix to quat
        # Implementation of matrixToQuat needed?
        # Can use standard lib or assume identity for now if lazy?
        # Let's implement robust mat2quat.
        tr = R_init[0,0] + R_init[1,1] + R_init[2,2]
        if tr > 0:
            S = torch.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (R_init[2,1] - R_init[1,2]) / S
            qy = (R_init[0,2] - R_init[2,0]) / S
            qz = (R_init[1,0] - R_init[0,1]) / S
        else:
            # max diag
            if (R_init[0,0] > R_init[1,1]) and (R_init[0,0] > R_init[2,2]):
                S = torch.sqrt(1.0 + R_init[0,0] - R_init[1,1] - R_init[2,2]) * 2
                qw = (R_init[2,1] - R_init[1,2]) / S
                qx = 0.25 * S
                qy = (R_init[0,1] + R_init[1,0]) / S
                qz = (R_init[0,2] + R_init[2,0]) / S
            elif R_init[1,1] > R_init[2,2]:
                S = torch.sqrt(1.0 + R_init[1,1] - R_init[0,0] - R_init[2,2]) * 2
                qw = (R_init[0,2] - R_init[2,0]) / S
                qx = (R_init[0,1] + R_init[1,0]) / S
                qy = 0.25 * S
                qz = (R_init[1,2] + R_init[2,1]) / S
            else:
                S = torch.sqrt(1.0 + R_init[2,2] - R_init[0,0] - R_init[1,1]) * 2
                qw = (R_init[1,0] - R_init[0,1]) / S
                qx = (R_init[0,2] + R_init[2,0]) / S
                qy = (R_init[1,2] + R_init[2,1]) / S
                qz = 0.25 * S
        
        self.q = torch.stack([qw, qx, qy, qz])
        self.q = torch.nn.functional.normalize(self.q, dim=0)
        
        # Loop
        g_ref = torch.tensor([0., 0., 1.], device=self.device) # Gravity Up (Reaction)
        m_ref = torch.tensor([0., 1., 0.], device=self.device) # Magnetic North
        
        for i in tqdm(range(N), desc="Google EKF"):
            dt = time[i] - time[i-1] if i > 0 else 0.01
            dt = float(dt)
            w = torch.tensor(gyro[i], device=self.device, dtype=torch.float32)
            a = torch.tensor(accel[i], device=self.device, dtype=torch.float32)
            m = torch.tensor(mag[i], device=self.device, dtype=torch.float32)
            
            # Predict
            self.predict(w, dt)
            
            # Update Accel
            a_norm = torch.norm(a)
            if abs(a_norm - 9.81) < 2.0: # Stationaryish
                self.update(a/a_norm, g_ref, self.ACC_STDEV)
                
            # Update Mag
            m_norm = torch.norm(m)
            self.update(m/m_norm, m_ref, self.MAG_STDEV)
            
            q_out.append(self.q.cpu().numpy())
            b_out.append(self.b.cpu().numpy())
            
        return np.array(q_out), np.array(b_out)

