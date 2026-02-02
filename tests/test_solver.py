
import sys
import os
import unittest
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solvers.pytorch_solver import PyTorchSolver
from core.data_loader import SensorData
from core.noise_db import NoiseParams

class TestPyTorchSolver(unittest.TestCase):
    def test_synthetic_trajectory(self):
        print("\n--- Running Synthetic Test ---")
        # 1. Generate Ground Truth Trajectory (smooth sine waves)
        N = 1000
        dt = 0.01
        time = np.linspace(0, N*dt, N)
        
        # Euler angles: Roll, Pitch, Yaw
        # Case 1: Simple Rotation in Yaw, steady Pitch/Roll
        # Simulating a phone panning around
        yaw = np.linspace(0, 90, N) # 0 to 90 degrees
        pitch = 10 * np.sin(time) # wobble
        roll = 5 * np.cos(time)
        
        euler_gt = np.stack([roll, pitch, yaw], axis=1)
        r_gt = R.from_euler('xyz', euler_gt, degrees=True)
        q_gt = r_gt.as_quat() # x,y,z,w
        # Convert to w,x,y,z for our internal convention if needed, 
        # BUT solvers/pytorch_solver outputs w,x,y,z? 
        # Let's check: BaseSolver comments say q=[w,x,y,z]
        # Scipy is [x,y,z,w].
        q_gt_wxyz = q_gt[:, [3, 0, 1, 2]]
        
        # 2. Generate Noisy Measurements
        # Accel: R(q)^T * g_world (Gravity Up [0,0,1] or Down [0,0,-1]?)
        # Standard: Gravity vector points UP in sensor frame when stationary on table?
        # Accelerometer measures Reaction Force. Resting on table (+Z up): Accel = +9.81 in Z.
        # So g_world in sensor frame should be [0,0,9.81].
        # In World Frame (ENU), Gravity is Down (-9.81)? No, Reaction is Up (+9.81).
        # Let's assume g_ref = [0, 0, 1] (normalized).
        
        g_ref = np.array([0, 0, 1])
        accel_clean = r_gt.inv().apply(g_ref)
        
        # Mag: North. Let's assume magnetic north is X [1,0,0] (East?) No, Y [0,1,0] (North).
        # And let's add Dip!
        # True Mag Vector in World (North + Dip)
        # Dip angle 60 deg down: [0, cos(60), -sin(60)]
        dip_angle_deg = 60
        dip = np.radians(dip_angle_deg)
        m_ref_true = np.array([0, np.cos(dip), -np.sin(dip)])
        m_ref_true /= np.linalg.norm(m_ref_true)
        
        mag_clean = r_gt.inv().apply(m_ref_true)
        
        # Gyro: Angular Velocity
        # Differentiate
        rots = r_gt
        # ang_vel = (R_{t+1} * R_t^T) / dt roughly
        # Better: use gradient
        # For simplicity, numerical diff
        q_next = q_gt_wxyz[1:]
        q_curr = q_gt_wxyz[:-1]
        # dq = q_curr_inv * q_next
        # axis-angle from dq
        # gyro = axis * angle / dt
        
        # Or analytical:
        # rate from euler rates? complicated.
        # Let's assume gyro is perfect for now or small noise.
        gyro_clean = np.zeros((N, 3))
        for i in range(N-1):
            R0 = rots[i].as_matrix()
            R1 = rots[i+1].as_matrix()
            dR = R1 @ R0.T
            w_skew = (dR - np.eye(3)) / dt # Approx
            # Actually log map
            vec = R.from_matrix(dR).as_rotvec() / dt
            # This vec is in WORLD frame? No:
            # R1 = dR * R0 -> dR is global rotation.
            # Local gyro: R1 = R0 * dR_local
            # dR_local = R0^T * dR * R0 = R0^T * R1.
            dR_local = R0.T @ R1
            vec_local = R.from_matrix(dR_local).as_rotvec() / dt
            gyro_clean[i] = vec_local
        gyro_clean[-1] = gyro_clean[-2]
        
        # Add Noise
        accel_noise = np.random.normal(0, 0.01, (N, 3))
        mag_noise = np.random.normal(0, 0.05, (N, 3))
        gyro_noise = np.random.normal(0, 0.001, (N, 3))
        
        data = SensorData(
            time=time,
            accel=accel_clean + accel_noise,
            gyro=gyro_clean + gyro_noise,
            mag=mag_clean + mag_noise,
            unix_timestamps=time, # dummy
            orientation=q_gt_wxyz # Pass GT for comparison logic if needed
        )
        
        # 3. Reference Params
        # Note the solver assumes m_ref = [0, 1, 0] by default!
        # This TEST should fail if we don't fix the solver to ESTIMATE m_ref or handle dip.
        
        params = NoiseParams(
            accel_noise_sigma=0.01,
            gyro_noise_sigma=0.001,
            mag_noise_sigma=0.05
        )
        
        # 4. SOLVE
        solver = PyTorchSolver()
        # We expect this to fail initially on Yaw/North because of the Dip angle mismatch.
        # The solver assumes [0,1,0], but truth is [0, 0.5, -0.86].
        
        traj = solver.solve(data, params)
        
        # 5. Evaluate
        q_est = traj.quaternions # w,x,y,z
        r_est = R.from_quat(q_est[:, [1,2,3,0]])
        euler_est = r_est.as_euler('xyz', degrees=True)
        
        # Align Yaw?
        # The absolute Yaw might be offset if the initial m_ref assumption is wrong.
        # Calculate error
        
        diffs = (euler_est - euler_gt)
        # Wrap yaw
        diffs[:, 2] = (diffs[:, 2] + 180) % 360 - 180
        
        mae_deg = np.mean(np.abs(diffs), axis=0)
        print(f"Mean Absolute Error (deg): Roll={mae_deg[0]:.2f}, Pitch={mae_deg[1]:.2f}, Yaw={mae_deg[2]:.2f}")
        
        # Check Initial Heading Logic
        m_meas_0 = mag_clean[0]
        grad_0 = accel_clean[0] # roughly [0,0,1]
        
        print("\n--- Diagnostic ---")
        print(f"True Mag World (Dip 60): {m_ref_true}")
        # Solver M_ref (hardcoded in code): [0, 1, 0]
        
        if mae_deg[2] > 10.0:
            print("FAILURE: Yaw error is high! This confirms the Magnetic Dip issue.")
        else:
            print("SUCCESS: Yaw is good.")
            
if __name__ == '__main__':
    unittest.main()
