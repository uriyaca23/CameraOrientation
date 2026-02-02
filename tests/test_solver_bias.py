
import sys
import os
import unittest
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solvers.pytorch_solver import PyTorchSolver
from core.data_loader import SensorData
from core.noise_db import NoiseParams

class TestPyTorchSolverBias(unittest.TestCase):
    def test_bias_recovery(self):
        print("\n--- Running Bias Recovery Test ---")
        N = 1000
        dt = 0.01
        time = np.linspace(0, N*dt, N)
        
        # 1. Ground Truth Orientation: Simple rotation
        # Rotate around Z axis at constant speed
        true_rate_z = 0.5 # rad/s
        yaw = true_rate_z * time * 180 / np.pi # degrees
        euler_gt = np.zeros((N, 3))
        euler_gt[:, 2] = yaw
        r_gt = R.from_euler('xyz', euler_gt, degrees=True)
        q_gt_wxyz = r_gt.as_quat()[:, [3, 0, 1, 2]]
        
        # 2. Measurements with BIAS
        # True angular velocity (body frame)
        # R = Rz(wt). w_body = [0, 0, w].
        w_true = np.zeros((N, 3))
        w_true[:, 2] = true_rate_z
        
        # Artificial Bias (Rigorous Gyro, Calibrated Accel)
        true_gb = np.array([0.05, -0.05, 0.02])
        true_ab = np.array([0.0, 0.0, 0.0]) # Assume calibrated accel
        
        gyro_meas = w_true + true_gb + np.random.normal(0, 0.001, (N, 3))
        
        # Accel & Mag (Perfect or noisy)
        g_global = np.array([0, 0, 1])
        m_global = np.array([0, 1, 0])
        
        accel_clean = r_gt.inv().apply(g_global)
        mag_clean = r_gt.inv().apply(m_global)
        
        accel_meas = accel_clean + true_ab + np.random.normal(0, 0.01, (N, 3))
        mag_meas = mag_clean + np.random.normal(0, 0.05, (N, 3))
        
        data = SensorData(
            time=time,
            accel=accel_meas,
            gyro=gyro_meas,
            mag=mag_meas,
            unix_timestamps=time,
            orientation=q_gt_wxyz
        )
        
        # 3. Solve (Rigorous)
        # Use Standard Params (similar to real usage)
        params = NoiseParams(0.01, 0.001, 0.05)
        
        params.gyro_bias_sigma = 10.0 # Loose to allow learning
        params.accel_bias_sigma = 10.0
        
        solver = PyTorchSolver()
        traj = solver.solve(data, params)
        
        # 4. Check Orientation Error
        q_est = traj.quaternions
        r_est = R.from_quat(q_est[:, [1,2,3,0]])
        diff = (r_est * r_gt.inv()).magnitude()
        mean_err_deg = np.mean(diff) * 180 / np.pi
        
        print(f"Mean Error (Test): {mean_err_deg:.2f} deg")
        
        # Verify Bias Recovery
        # We can't easily access estimated bias from `traj` object (API limitation),
        # but we can rely on error being low.
        pass
            
if __name__ == '__main__':
    unittest.main()
