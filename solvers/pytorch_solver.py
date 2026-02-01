
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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
        quats = nn.Parameter(quats_init)
        
        # Optimizer
        # L-BFGS is good for batch optimization, but Adam is robust.
        # We will use Adam for a few epochs.
        optimizer = optim.Adam([quats], lr=0.01)
        
        # Weights (Inverse Variance)
        w_accel = 1.0 / (noise_params.accel_noise_sigma ** 2)
        w_gyro  = 1.0 / (noise_params.gyro_noise_sigma ** 2) 
        w_mag   = 1.0 / (noise_params.mag_noise_sigma ** 2)
        
        # Gravity vector (assume phone held upright-ish, gravity is DOWN in ENU?)
        # ENU: Gravity is usually [0, 0, -9.81].
        # Accel measures Reaction force -> [0, 0, 9.81] when still.
        # Let's normalize ref vectors
        g_ref = torch.tensor([0.0, 0.0, 1.0], device=device) 
        # Mag vector (Assume North for initialization, but this might be wrong)
        # We'll optimizing for consistency. 
        # Or better: Assume Magnetic North = [0, 1, -1] roughly (dip).
        # Let's use [0, 1, 0] as North approximation for now.
        m_ref = torch.tensor([0.0, 1.0, 0.0], device=device)

        # Normalize measurements
        accel_norm = torch.nn.functional.normalize(accel, dim=1)
        mag_norm = torch.nn.functional.normalize(mag, dim=1)
        
        # Training Loop
        steps = 500
        for i in range(steps):
            optimizer.zero_grad()
            
            # Normalize current estimation to unit quaternion
            q_norm = torch.nn.functional.normalize(quats, p=2, dim=1)
            
            # 1. Gyro Factor (Between)
            # q_{t+1} approx q_t * delta_q
            # delta_q = [cos(w*dt/2), sin(w*dt/2)*axis]
            # Small angle approx: [1, w_x*dt/2, w_y*dt/2, w_z*dt/2]
            
            # Use data[0:N-1] to predict data[1:N]
            q_t = q_norm[:-1]
            q_next_pred = q_norm[1:]
            
            w = gyro[:-1]
            theta = torch.norm(w, dim=1, keepdim=True) * dt
            axis = w / (torch.norm(w, dim=1, keepdim=True) + 1e-6)
            
            # Axis-Angle to Quat
            dq_w = torch.cos(theta/2)
            dq_xyz = axis * torch.sin(theta/2)
            dq = torch.cat([dq_w, dq_xyz], dim=1)
            
            # Quaternion multiplication: q_new = q_old * dq
            # (Standard w,x,y,z mult)
            w1, x1, y1, z1 = q_t[:,0], q_t[:,1], q_t[:,2], q_t[:,3]
            w2, x2, y2, z2 = dq[:,0], dq[:,1], dq[:,2], dq[:,3]
            
            q_next_calc_w = w1*w2 - x1*x2 - y1*y2 - z1*z2
            q_next_calc_x = w1*x2 + x1*w2 + y1*z2 - z1*y2
            q_next_calc_y = w1*y2 - x1*z2 + y1*w2 + z1*x2
            q_next_calc_z = w1*z2 + x1*y2 - y1*x2 + z1*w2
            
            q_next_calc = torch.stack([q_next_calc_w, q_next_calc_x, q_next_calc_y, q_next_calc_z], dim=1)
            
            # Loss: 1 - <q_pred, q_calc>^2 (Geodesic distance approximation)
            # dot product
            dot = torch.sum(q_next_pred * q_next_calc, dim=1)
            loss_gyro = torch.mean(1.0 - dot**2) * w_gyro
            
            # 2. Accel Factor (Prior per node)
            # R(q)^T * g_ref = accel_meas
            # Rotate g_ref by q_inv (which is q_conjugate) -> should match linear accel in body frame
            
            # Rotate vector by quaternion: v' = q_inv * v * q
            # Or just use Rotation Matrix function
            # R(q) maps Body to World.
            # Accel is in Body. R(q)^T * [0,0,1] should vary.
            # Actually accel measures g in Body.
            # So g_body = R(q)^T * g_world.
            # g_world = [0,0,1].
            # R^T * z_axis = last row of R.
            
            # Convert q to R
            w, x, y, z = q_norm[:,0], q_norm[:,1], q_norm[:,2], q_norm[:,3]
            
            # Formula for rotated vector using quaternion algebra is likely faster
            # v_rot = v + 2*cross(q_xyz, cross(q_xyz, v) + q_w*v)
            # Here optimize for [0, 0, 1]
            # Ry = row 3 of R
            rx = 2*(x*z - w*y)
            ry = 2*(y*z + w*x)
            rz = 1 - 2*(x*x + y*y)
            pred_accel = torch.stack([rx, ry, rz], dim=1)
            
            loss_accel = torch.nn.functional.mse_loss(pred_accel, accel_norm) * w_accel
            
            # 3. Mag Factor
            # Similar to accel, but with North
            # For robustness, we use a robust loss (Huber/L1) if mag_noise is high
            # Assume m_ref = [0, 1, 0] (North)
            # Rx = 2*(x*y + w*z)
            # Ry = 1 - 2*(x*x + z*z)
            # Rz = 2*(y*z - w*x)
            # pred_mag = torch.stack([Rx, Ry, Rz], dim=1)
            
            # Using generic rotation for arbitrary m_ref
            # q_inv = [w, -x, -y, -z]
            # ...
            # Let's just assume North for now
            pred_mag_x = 2*(x*y + w*z)
            pred_mag_y = 1 - 2*(x*x + z*z)
            pred_mag_z = 2*(y*z - w*x)
            pred_mag = torch.stack([pred_mag_x, pred_mag_y, pred_mag_z], dim=1)
            
            # Robust Loss for Mag: SmoothL1
            loss_mag = torch.nn.functional.smooth_l1_loss(pred_mag, mag_norm) * w_mag
            
            total_loss = loss_gyro + loss_accel + loss_mag
            
            total_loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                print(f"Step {i}: Loss {total_loss.item():.4f} (G:{loss_gyro.item():.4f} A:{loss_accel.item():.4f} M:{loss_mag.item():.4f})")

        # Result
        with torch.no_grad():
            final_q = torch.nn.functional.normalize(quats, p=2, dim=1).cpu().numpy()

        # Covariance Estimation (Laplace Approximation)
        # H approx J^T J.
        # We have a huge state N x 4 ~ 4N. Inverting 4Nx4N is too slow.
        # However, the graph structure is block-tridiagonal (Markov chain).
        # The uncertainty usually grows or stays bounded by measurement noise.
        # We will use a simplified marginal estimation: 
        # The local uncertainty is roughly proportional to the inverse of the factor weights sum.
        # Sigma^2 ~ 1 / (w_accel + w_mag) (if we ignore temporal correlation for a moment)
        # But Gyro ties them.
        # A better approximation for visualization: Use the noise params to set the bounds!
        # If the filter works, the posterior covariance should be close to the `accel/mag` noise when stationary,
        # and grow slightly during motion.
        
        # Let's populate with a baseline covariance derived from measurement noise.
        # It's better than zeros.
        # Var(Angle) ~ (sigma_accel / g)^2 + (sigma_mag / B)^2 roughly.
        
        baseline_sigma_sq = (noise_params.accel_noise_sigma / 9.81)**2
        # If indoor, mag noise is high, so uncertainty is dominated by mag noise in Yaw.
        yaw_sigma_sq = (noise_params.mag_noise_sigma / 30.0)**2 # Assuming ~30uT field strength
        
        # We'll return a constant covariance for visualization purposes if full marginals are too expensive.
        # This gives the user the "Expected Accuracy" of their device.
        
        covariances = np.zeros((N, 3, 3))
        covariances[:, 0, 0] = baseline_sigma_sq # Roll
        covariances[:, 1, 1] = baseline_sigma_sq # Pitch
        covariances[:, 2, 2] = yaw_sigma_sq      # Yaw
        
        return OrientationTrajectory(data.time, final_q, covariances)
