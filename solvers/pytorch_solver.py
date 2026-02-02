
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import tqdm
from .base_solver import BaseSolver, OrientationTrajectory
from data_loader import SensorData
from noise_db import NoiseParams

class PyTorchSolver(BaseSolver):
    def solve(self, data: SensorData, noise_params: NoiseParams) -> OrientationTrajectory:
        """
        Solves for orientation using a custom PyTorch optimizer.
        Minimizes:
          L = w_g * || q_curr * gyro_delta - q_next ||^2  (Dynamics)
            + w_a * || Rot(q) * G - accel ||^2           (Gravity)
            + w_m * || Rot(q) * M - mag ||^2             (Compass)
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"PyTorchSolver using device: {device}")
        
        # Optimize Bias?
        # User requested rigorous factor graph approach.
        # State: Orientation (q_t), Gyro Bias (b_w), Accel Bias (b_a).
        # We assume static bias for the duration of the clip (80s) is a reasonable first approx, 
        # or slowly varying (Random Walk). Static is much more stable for optimization.
        optimize_bias = True

        # Convert data to Tensor
        # Downsample further if needed for speed, but user asked for 50Hz which is fine.
        accel = torch.tensor(data.accel, dtype=torch.float32, device=device)
        gyro = torch.tensor(data.gyro, dtype=torch.float32, device=device)
        mag = torch.tensor(data.mag, dtype=torch.float32, device=device)
        dt = float(np.mean(np.diff(data.time)))
        N = len(data.time)

        # Initialize Quaternions (Identity or aligned with gravity)
        # q = [w, x, y, z]
        # Better: Initialize by integrating gyro only first
        quats_init = torch.zeros((N, 4), dtype=torch.float32, device=device)
        quats_init[:, 0] = 1.0 # Identity
        
        # Optimization variable
        quats_init[:, 0] = 1.0 # Identity
        
        # Parameters
        quats = nn.Parameter(quats_init)
        
        # Bias Parameters (Static for now)
        # gyro_bias: 1x3
        # accel_bias: 1x3
        gyro_bias = nn.Parameter(torch.zeros(1, 3, dtype=torch.float32, device=device))
        accel_bias = nn.Parameter(torch.zeros(1, 3, dtype=torch.float32, device=device))
        
        params_list = [quats, gyro_bias, accel_bias]
        
        # Optimizer
        # L-BFGS is good for batch optimization, but Adam is robust.
        # We will use Adam for a few epochs.
        optimizer = optim.Adam(params_list, lr=0.01)
        
        # Weights (Inverse Variance)
        # For Bias Estimation, we need 'q' to be driven by Mag/Accel (Absolute Refs) 
        # to observe the Gyro drift. So w_gyro must NOT be too high initially.
        w_accel = 1.0 / (noise_params.accel_noise_sigma ** 2)
        w_mag   = 1.0 / (noise_params.mag_noise_sigma ** 2)
        
        # Heuristic: w_gyro should be comparable to w_mag to allow bending.
        # True physics weight (1/sigma*dt^2) is too stiff (~1e8).
        # We use a relaxed weight for the solver.
        w_gyro  = 2.0 * w_mag # Tuned for observability
        
        # Bias Priors (Regularization)
        # Constrain bias to be close to 0 (or previous estimate).
        # sigma_rw usually small.
        w_Gb = 1.0 / (noise_params.gyro_bias_sigma ** 2)
        w_Ab = 1.0 / (noise_params.accel_bias_sigma ** 2)
        
        # Gravity vector (assume phone held upright-ish, gravity is DOWN in ENU?)
        # ENU: Gravity is usually [0, 0, -9.81].
        # Accel measures Reaction force -> [0, 0, 9.81] when still.
        # Let's normalize ref vectors
        m_ref = torch.tensor([0.0, 1.0, 0.0], device=device)
        
        # IMPROVEMENT: Estimate m_ref from initial data!
        # Assuming the first second is relatively static or representative.
        # 1. Calculate Initial Roll/Pitch from Accel
        # 2. De-rotate Mag to horizontal
        # 3. Calculate Initial Yaw
        # 4. Construct q0
        # 5. m_ref = q0 * m_0 * q0_inv
        
        # Or simpler: We define World Frame such that Gravity is [0,0,1] and Mag is [0, my, mz].
        # We can just compute the angle between Gravity and Mag in Body frame, and preserve it in World frame.
        
        # Let's do the full initialization:
        print("Initializing Logic from Sensor Data...")
        try:
            # Take first 50 samples or 1 sec
            init_N = min(50, N)
            a_mean = np.mean(data.accel[:init_N], axis=0)
            m_mean = np.mean(data.mag[:init_N], axis=0)
            a_mean /= np.linalg.norm(a_mean)
            m_mean /= np.linalg.norm(m_mean)
            
            # 1. Align Gravity (Body-Z approx) to World-Z ([0,0,1])
            # R_bg (Body to GravityAligned)
            # We want R * a_mean = [0, 0, 1].
            # Vector aligning rotation.
            from scipy.spatial.transform import Rotation as Rot
            
            # Simple approach: TRIAD or just arithmetic
            # Z_w = a_mean (Wait, Accelerometer measures +1g UP when static flat? YES. +Z)
            # So a_mean is UP.
            # Z_b = a_mean
            
            # Construct a basis from a_mean and m_mean
            # Z = a_mean
            # X = cross(m_mean, Z).normalized (East)
            # Y = cross(Z, X) (North)
            
            z_axis = a_mean
            x_axis = np.cross(m_mean, z_axis)
            if np.linalg.norm(x_axis) < 1e-6:
                x_axis = np.array([1,0,0]) # singular (mag || gravity)
            x_axis /= np.linalg.norm(x_axis)
            y_axis = np.cross(z_axis, x_axis)
            
            # This Rotation R_wb (World to Body) matrix has columns [X, Y, Z] expressed in Body Frame.
            # R_wb = [X, Y, Z]
            # R_bw = R_wb.T
            # Our q is Body to World. So R_bw.
            
            R_matrix = np.stack([x_axis, y_axis, z_axis], axis=1) # Columns are body axes? No.
            # The columns of R (Body->World) are the Body axes expressed in World.
            # Here we have World Axes expressed in Body.
            # X_w_in_b = x_axis
            # Y_w_in_b = y_axis
            # Z_w_in_b = z_axis
            # So R_wb = [x_axis, y_axis, z_axis].
            # We want R_bw (Body to World).
            # R_bw = R_wb.T
            
            # Let's verify:
            # R_bw @ z_axis (which is Z_w in b) = [0,0,1]?
            # row3 of R_bw is z_axis.T. z_axis.T @ z_axis = 1. Correct.
            
            R_bw = R_matrix.T
            q0_scipy = Rot.from_matrix(R_bw).as_quat() # x,y,z,w
            q0_init = torch.tensor([q0_scipy[3], q0_scipy[0], q0_scipy[1], q0_scipy[2]], dtype=torch.float32, device=device)
            
            # 4. Integrate Gyro for better initialization (Dead Reckoning)
            # This prevents 360-degree wrapping issues in optimization
            print("Integrating Gyro for initialization...")
            q_curr = q0_init.clone()
            quats_init[0] = q_curr
            
            # Simple Euler integration
            for k in range(1, N):
                w = gyro[k-1] # (3,)
                theta = torch.norm(w) * dt
                if theta < 1e-6:
                    dq = torch.tensor([1.0, 0, 0, 0], device=device)
                else:
                    axis = w / theta
                    dq_w = torch.cos(theta/2)
                    dq_xyz = axis * torch.sin(theta/2)
                    dq = torch.cat([dq_w.unsqueeze(0), dq_xyz])
                
                # q_next = q_curr * dq
                w1, x1, y1, z1 = q_curr
                w2, x2, y2, z2 = dq
                
                qw = w1*w2 - x1*x2 - y1*y2 - z1*z2
                qx = w1*x2 + x1*w2 + y1*z2 - z1*y2
                qy = w1*y2 - x1*z2 + y1*w2 + z1*x2
                qz = w1*z2 + x1*y2 - y1*x2 + z1*w2
                
                q_curr = torch.stack([qw, qx, qy, qz])
                q_curr = torch.nn.functional.normalize(q_curr, dim=0)
                quats_init[k] = q_curr
                
            quats = nn.Parameter(quats_init)
            
            # Re-optimizer needed because we changed parameter
            params_list = [quats, gyro_bias, accel_bias]
            optimizer = optim.Adam(params_list, lr=0.01)

        except Exception as e:
            print(f"Initialization Warning: {e}. Using defaults.")

        # Normalize measurements

        # Normalize measurements
        accel_norm = torch.nn.functional.normalize(accel, dim=1)
        mag_norm = torch.nn.functional.normalize(mag, dim=1)
        
        # Training Loop
        steps = 500
        
        
        # --- Iterative Analytic Optimization ---
        # Iteration Loop
        n_outer_iters = 10
        
        # We handle Accel Bias as fixed 0
        accel_bias.requires_grad = False
        accel_bias.data.fill_(0.0)
        
        for outer_k in range(n_outer_iters):
             # Step 1: Optimize Q (Adam)
             gyro_bias.requires_grad = False 
             
             # Lower LR to prevent jagged trajectories
             optimizer = optim.Adam([quats], lr=0.005)
             
             # Optimization Loop
             steps = 100
             # pbar = tqdm.tqdm(range(steps), desc=f"Iter {outer_k+1}")
             for i in range(steps):
                optimizer.zero_grad()
                q_norm = torch.nn.functional.normalize(quats, p=2, dim=1)
                
                # 1. Gyro
                q_t = q_norm[:-1]
                q_next_pred = q_norm[1:]
                
                w_corr = gyro[:-1] - gyro_bias
                w = w_corr
                theta = torch.norm(w, dim=1, keepdim=True) * dt
                
                # Use F.normalize for stability
                axis = torch.nn.functional.normalize(w, dim=1)
                
                dq_w = torch.cos(theta/2)
                dq_xyz = axis * torch.sin(theta/2)
                dq = torch.cat([dq_w, dq_xyz], dim=1)
                
                # Explicitly normalize dq to prevent drift > 1.0
                dq = torch.nn.functional.normalize(dq, dim=1)
                
                w1,x1,y1,z1 = q_t[:,0], q_t[:,1], q_t[:,2], q_t[:,3]
                w2,x2,y2,z2 = dq[:,0], dq[:,1], dq[:,2], dq[:,3]
                qn_w = w1*w2 - x1*x2 - y1*y2 - z1*z2
                qn_x = w1*x2 + x1*w2 + y1*z2 - z1*y2
                qn_y = w1*y2 - x1*z2 + y1*w2 + z1*x2
                qn_z = w1*z2 + x1*y2 - y1*x2 + z1*w2
                q_next_calc = torch.stack([qn_w, qn_x, qn_y, qn_z], dim=1)
                
                dot = torch.sum(q_next_pred * q_next_calc, dim=1)
                loss_gyro = torch.mean(1.0 - dot**2) * w_gyro
                
                # 2. Accel
                w,x,y,z = q_norm[:,0], q_norm[:,1], q_norm[:,2], q_norm[:,3]
                rx = 2*(x*z - w*y)
                ry = 2*(y*z + w*x)
                rz = 1 - 2*(x*x + y*y)
                pred_accel = torch.stack([rx, ry, rz], dim=1)
                loss_accel = torch.nn.functional.mse_loss(pred_accel, accel_norm) * w_accel

                # 3. Mag
                mx, my, mz = m_ref[0], m_ref[1], m_ref[2]
                pred_mag_x = (1 - 2*y*y - 2*z*z)*mx + (2*x*y + 2*w*z)*my + (2*x*z - 2*w*y)*mz
                pred_mag_y = (2*x*y - 2*w*z)*mx     + (1 - 2*x*x - 2*z*z)*my + (2*y*z + 2*w*x)*mz
                pred_mag_z = (2*x*z + 2*w*y)*mx     + (2*y*z - 2*w*x)*my     + (1 - 2*x*x - 2*y*y)*mz
                pred_mag = torch.stack([pred_mag_x, pred_mag_y, pred_mag_z], dim=1)
                loss_mag = torch.nn.functional.smooth_l1_loss(pred_mag, mag_norm) * w_mag
                
                loss = loss_gyro + loss_accel + loss_mag
                loss.backward()
                optimizer.step()

             # Step 2: Analytic Bias Update
             with torch.no_grad():
                 q_opt = torch.nn.functional.normalize(quats, p=2, dim=1)
                 q_t = q_opt[:-1]
                 q_next = q_opt[1:]
                 
                 qi_w, qi_x, qi_y, qi_z = q_t[:,0], -q_t[:,1], -q_t[:,2], -q_t[:,3]
                 qn_w, qn_x, qn_y, qn_z = q_next[:,0], q_next[:,1], q_next[:,2], q_next[:,3]
                 d_w = qi_w*qn_w - qi_x*qn_x - qi_y*qn_y - qi_z*qn_z
                 d_x = qi_w*qn_x + qi_x*qn_w + qi_y*qn_z - qi_z*qn_y
                 d_y = qi_w*qn_y - qi_x*qn_z + qi_y*qn_w + qi_z*qn_x
                 d_z = qi_w*qn_z + qi_x*qn_y - qi_y*qn_x + qi_z*qn_w
                 
                 # Fix Double Cover (Shortest Path)
                 mask_neg = d_w < 0
                 d_w[mask_neg] *= -1
                 d_x[mask_neg] *= -1
                 d_y[mask_neg] *= -1
                 d_z[mask_neg] *= -1
                 
                 theta = 2.0 * torch.acos(torch.clamp(d_w, -1.0, 1.0))
                 sin_half = torch.sqrt(torch.clamp(1.0 - d_w**2, min=1e-6))
                 
                 # w_q stats
                 w_q_x = (d_x / sin_half) * theta / dt
                 w_q_y = (d_y / sin_half) * theta / dt
                 w_q_z = (d_z / sin_half) * theta / dt
                 w_q = torch.stack([w_q_x, w_q_y, w_q_z], dim=1)
                 
                 # Update Bias with Damping
                 diff = gyro[:-1] - w_q
                 bias_est = torch.mean(diff, dim=0)
                 bias_est = torch.clamp(bias_est, -0.5, 0.5)
                 
                 # EMA Update: bias = 0.9*old + 0.1*new
                 alpha = 0.2
                 new_bias = (1 - alpha) * gyro_bias.data.squeeze() + alpha * bias_est
                 
                 # print(f"Iter {outer_k+1} Bias Est: {new_bias.cpu().numpy()}")
                 gyro_bias.data = new_bias.unsqueeze(0)
            
        # Result
        with torch.no_grad():
            final_q = torch.nn.functional.normalize(quats, p=2, dim=1).cpu().numpy()
            final_gb = gyro_bias.cpu().numpy()
            final_ab = accel_bias.cpu().numpy()
            print(f"Full Optimization Complete.")
            print(f"Estimated Gyro Bias: {final_gb[0]}")
            print(f"Estimated Accel Bias: {final_ab[0]}")
            
        # Refine with L-BFGS for final convergence?
        # Adam is usually enough.

        # Covariance Estimation (KF Smoother / RTS)
        # We perform a linearized Error-State Kalman Smoother pass on the optimized trajectory.
        # This provides the correct marginal covariances (Sigma).
        
        print("Estimating Covariance (RTS Smoother)...")
        # Pass bias to smoother to correct measurements effectively (or just orientation covariance around the mean)
        covariances = self.compute_covariance_rts(final_q, data, noise_params, dt, m_ref, final_gb, final_ab)
        
        return OrientationTrajectory(data.time, final_q, covariances)

    def skew(self, v):
        """ Returns 3x3 skew symmetric matrix from 3-vector """
        x, y, z = v
        return np.array([[0, -z, y],
                         [z, 0, -x],
                         [-y, x, 0]])

    def compute_covariance_rts(self, quats, data, noise_params, dt, m_ref_tensor, gyro_bias=None, accel_bias=None):
        N = len(quats)
        
        # 1. Setup Parameters
        # Process Noise Q (Gyro integration error) using small angle approx
        # sigma_theta = sigma_gyro * dt
        # Q = diag(sigma_theta**2)
        q_gyro_var = (noise_params.gyro_noise_sigma * dt) ** 2
        Q = np.eye(3) * q_gyro_var
        
        # Measurement Noise R
        ra_var = noise_params.accel_noise_sigma ** 2
        rm_var = noise_params.mag_noise_sigma ** 2
        R_cov = np.diag([ra_var]*3 + [rm_var]*3) # 6x6
        
        # Initial Covariance (High uncertainty or steady state)
        P_filt = [None] * N
        P_pred = [None] * N
        
        P_curr = np.eye(3) * 0.1 # Initial uncertainty
        
        # Forward Pass (Filtering)
        # g_ref = [0,0,1], m_ref = [0,1,0] (Approximation)
        g_ref = np.array([0, 0, 1])
        
        # Ensure m_ref is numpy and normalized
        if torch.is_tensor(m_ref_tensor):
             m_ref = m_ref_tensor.cpu().numpy()
        else:
             m_ref = np.array(m_ref_tensor)
        
        # Pre-compute R matrices?
        from scipy.spatial.transform import Rotation as Rot
        # quats is Nx4 (w,x,y,z). Scipy needs (x,y,z,w)
        qs_scipy = quats[:, [1,2,3,0]]
        Rs = Rot.from_quat(qs_scipy).as_matrix() # Nx3x3
        
        # Measurements
        accel = data.accel
        mag = data.mag
        
        # Apply Bias correction if provided
        if gyro_bias is not None:
             # Gyro bias affects the state equation (Q), not measurement directly here,
             # but we already used the optimized trajectory which accounts for it.
             # The uncertainty Q is about the NOISE, not the bias (which is now 'known').
             pass
             
        if accel_bias is not None:
             accel = accel - accel_bias
             
        # Normalize just in case, though noise models usually assume raw unit? 
        # Actually our loss used normalized. Let's normalize data to match 'unit vector' assumption.
        accel = accel / (np.linalg.norm(accel, axis=1, keepdims=True) + 1e-6)
        mag = mag / (np.linalg.norm(mag, axis=1, keepdims=True) + 1e-6)

        for k in range(N):
            # 1. Prediction (Time Update)
            # x_{k} = x_{k-1} + u (Rotation integration) represented by quat.
            # Error state: d_theta_{k} = d_theta_{k-1} + noise.
            # F = I
            if k == 0:
                P_minus = P_curr # P_0
            else:
                P_minus = P_filt[k-1] + Q
            
            P_pred[k] = P_minus
            
            # 2. Update (Measurement Update)
            # H = [H_accel; H_mag] (6x3)
            # z_pred = R^T * g
            # H = [z_pred]_x 
            
            # Accel
            R_k = Rs[k] # Body to World? No, R maps Body to World.
            # z = R_k.T @ g_ref
            z_pred_a = R_k.T @ g_ref
            H_a = self.skew(z_pred_a)
            
            # Mag
            z_pred_m = R_k.T @ m_ref
            H_m = self.skew(z_pred_m)
            
            H = np.vstack([H_a, H_m]) # 6x3
            
            # Kalman Gain
            # K = P H^T (H P H^T + R)^-1
            S = H @ P_minus @ H.T + R_cov
            K = P_minus @ H.T @ np.linalg.inv(S)
            
            start_P = P_minus
            # Update P
            # P_plus = (I - K H) P_minus
            # Use Joseph form for stability? Or simple:
            KH = K @ H
            P_plus = (np.eye(3) - KH) @ P_minus
            
            # Symmetrize
            P_plus = (P_plus + P_plus.T) / 2.0
            
            P_filt[k] = P_plus
            
        # Backward Pass (RTS Smoother)
        P_smooth = [None] * N
        P_smooth[-1] = P_filt[-1]
        
        for k in range(N-2, -1, -1):
            P_k_k = P_filt[k]
            P_kplus_k = P_pred[k+1] # P_{k+1 | k}
            P_kplus_N = P_smooth[k+1]
            
            # C_k = P_{k|k} F^T P_{k+1|k}^-1. F=I.
            # C_k = P_{k|k} * inv(P_{k+1|k})
            
            C_k = P_k_k @ np.linalg.inv(P_kplus_k)
            
            # P_{k|N} = P_{k|k} + C_k (P_{k+1|N} - P_{k+1|k}) C_k^T
            P_smooth[k] = P_k_k + C_k @ (P_kplus_N - P_kplus_k) @ C_k.T
            
        return np.array(P_smooth)
