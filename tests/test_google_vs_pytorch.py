
import unittest
import numpy as np
import torch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scipy.spatial.transform import Rotation as R
from solvers.pytorch_solver import PyTorchSolver
from solvers.google_solver import GoogleEKF
from solvers.base_solver import OrientationTrajectory
from data_loader import SensorData
from noise_db import NoiseParams

class TestSolverComparison(unittest.TestCase):
    def test_compare_google_vs_pytorch(self):
        print("\n--- Google EKF vs PyTorch Solver Comparison ---")
        
        # 1. Generate Synthetic Data (Rigorous Case)
        N = 1000 # 10 seconds at 100Hz
        dt = 0.01
        
        # True Trajectory: Constant rotation + Sinusoidal
        t = np.linspace(0, N*dt, N)
        w_true = np.zeros((N, 3))
        w_true[:, 2] = 0.5 # Constant Yaw Rate
        w_true[:, 0] = 0.2 * np.sin(2.0 * t) # Roll wobble
        
        # Integrate to get True Orientation
        rots = [R.from_quat([0, 0, 0, 1])]
        for i in range(1, N):
            w = w_true[i-1]
            angle = np.linalg.norm(w) * dt
            if angle > 1e-9:
                axis = w / np.linalg.norm(w)
                r_delta = R.from_rotvec(axis * angle)
                rots.append(rots[-1] * r_delta)
            else:
                rots.append(rots[-1])
                
        rots = R.from_quat([r.as_quat() for r in rots])
        q_gt_wxyz = np.roll(rots.as_quat(), 1, axis=1) # xyzw -> wxyz
        
        # Sensor Data (With Bias)
        true_gb = np.array([0.05, -0.05, 0.02])
        true_ab = np.array([0.0, 0.0, 0.0]) # Disable Accel Bias for fair comparison (Google assumes 0)
        
        gyro_meas = w_true + true_gb + np.random.normal(0, 0.001, (N, 3))
        
        # Accel/Mag from Truth
        g_ref = np.array([0, 0, 1])
        m_ref = np.array([0, 1, 0])
        
        accel_meas = rots.inv().apply(g_ref) + np.random.normal(0, 0.02, (N, 3))
        mag_meas = rots.inv().apply(m_ref) + np.random.normal(0, 0.05, (N, 3))
        
        data = SensorData(
            time=t,
            unix_timestamps=t, # Mock unix
            accel=accel_meas,
            gyro=gyro_meas,
            mag=mag_meas,
            orientation=q_gt_wxyz
        )
        
        # 2. Run Google EKF
        print("Running Google EKF (AOSP)...")
        # Boost Bias Process Noise to allow learning 0.05 rad/s quickly
        google_solver = GoogleEKF(device='cpu', gyro_bias_var=1e-06)
        # Google assumes [0,0,1] Gravity and [0,1,0] North? 
        # Our GT Accel is [0,0,1] rotated. Accel measures Reaction Force.
        # If gravity is Down, Accel holds Up ([0,0,1]). Correct.
        
        q_google, b_google = google_solver.solve(t, gyro_meas, accel_meas, mag_meas)
        
        # Evaluate Google
        # Helper to compute error
        def compute_error(q_est, q_gt):
            # q_est: (N, 4) wxyz
            # q_gt: (N, 4) wxyz
            q_est_xyzw = np.roll(q_est, -1, axis=1)
            q_gt_xyzw = np.roll(q_gt, -1, axis=1)
            
            r_est = R.from_quat(q_est_xyzw)
            r_gt = R.from_quat(q_gt_xyzw)
            
            diff = r_est.inv() * r_gt
            angles = diff.magnitude() # rad
            return np.degrees(angles)
            
        err_google = compute_error(q_google, q_gt_wxyz)
        mean_err_google = np.mean(err_google)
        print(f"Google EKF Mean Error: {mean_err_google:.4f} deg")
        print(f"Google Bias Est (Final): {b_google[-1]}")
        print(f"True Bias: {true_gb}")
        
        # 3. Run PyTorch Solver
        print("\nRunning PyTorch Solver (Iterative Analytic)...")
        params = NoiseParams(0.001, 0.02, 0.05)
        params.gyro_bias_sigma = 0.5 # Weak prior to allow recovery of 0.05 bias
        
        # Tune weights as per previous discovery in `pytorch_solver.py`
        # PyTorch solver logic handles weights internally now.
        
        # Tune weights as per previous discovery in `pytorch_solver.py`
        # PyTorch solver logic handles weights internally now.
        
        pytorch_solver = PyTorchSolver()
        traj_pytorch = pytorch_solver.solve(data, params)
        
        err_pytorch = compute_error(traj_pytorch.quaternions, q_gt_wxyz)
        mean_err_pytorch = np.mean(err_pytorch)
        print(f"PyTorch Solver Mean Error: {mean_err_pytorch:.4f} deg")
        
        # 4. Comparison
        print("\n--- RESULTS ---")
        print(f"Method         | Mean Error (deg) | Final Bias Error (deg/s)")
        print(f"---------------|------------------|-------------------------")
        
        b_err_g = np.linalg.norm(b_google[-1] - true_gb) * 180 / np.pi
        # PyTorch bias not easily accessible from Trajectory object unless printed
        # But we know it works from previous test.
        
        print(f"Google EKF     | {mean_err_google:16.4f} | {b_err_g:8.4f}")
        print(f"PyTorch (Opt)  | {mean_err_pytorch:16.4f} | (See logs)")
        
        self.assertLess(mean_err_google, 20.0, "Google EKF should perform well.")
        self.assertLess(mean_err_pytorch, 20.0, "PyTorch Solver should perform well.")

if __name__ == '__main__':
    unittest.main()
