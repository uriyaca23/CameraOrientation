import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import sys

# Ensure parent directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_loader import DataLoader
from solvers.pytorch_solver import PyTorchSolver
from core.noise_db import NoiseParams

def quat_error_degrees(q1, q2):
    """Computes angular distance between two quaternion arrays in degrees."""
    # Ensure q1, q2 are (N, 4)
    # distance = 2 * arccos( |<q1, q2>| )
    dot = np.abs(np.sum(q1 * q2, axis=1))
    dot = np.clip(dot, -1.0, 1.0)
    angle_rad = 2 * np.arccos(dot)
    return np.degrees(angle_rad)

def main():
    # Hardcoded path for test convenience, matching the user's current context
    log_path = r"c:\Users\uriya\PycharmProjects\CameraOrientation\data\google_pixel_10\indoor\exp1_uriya_apartment\sensorLog_fA4bFhD_ZGc_20260201T191641.txt"
    video_path = r"c:\Users\uriya\PycharmProjects\CameraOrientation\data\google_pixel_10\indoor\exp1_uriya_apartment\PXL_20260201_171650791.mp4"
    
    print("Loading data...")
    loader = DataLoader(target_freq=30.0)
    data = loader.load_data(log_path, video_path)
    
    print(f"Loaded {len(data.time)} samples.")
    
    # Noise params (Indoor)
    noise_params = NoiseParams(
        accel_noise_sigma=0.008, 
        gyro_noise_sigma=0.002, 
        mag_noise_sigma=20.0
    )
    
    print("Running PyTorchSolver (Using Raw Accel/Gyro/Mag)...")
    solver = PyTorchSolver()
    trajectory = solver.solve(data, noise_params)
    
    # Compare with Device Orientation (data.orientation)
    # Check if data.orientation is populated
    if data.orientation is None or np.all(data.orientation == 0):
        print("Error: Device orientation data missing or empty.")
        return

    # Device Orientation format: w, x, y, z or x, y, z, w?
    # Log: 0.73, -0.002, 0.010, 0.68.
    # 0.73 is likely w. So [w, x, y, z].
    # PyTorchSolver returns [w, x, y, z].
    
    q_est = trajectory.quaternions
    q_ref = data.orientation
    
    # Align Initial Rotation?
    # The solver initializes to Identity (unfortunately). 
    # The device orientation is in World frame (referenced to Magnetic North/Gravity).
    # There will be a constant offset.
    # Let's compute the offsets at t=0 and remove it from q_est for comparison purposes.
    # q_est_aligned = q_est * q_offset
    # We want q_est[0] * q_offset = q_ref[0] => q_offset = q_est[0]^-1 * q_ref[0]
    
    r_est0 = R.from_quat([q_est[0,1], q_est[0,2], q_est[0,3], q_est[0,0]])
    r_ref0 = R.from_quat([q_ref[0,1], q_ref[0,2], q_ref[0,3], q_ref[0,0]])
    
    offset = r_est0.inv() * r_ref0
    
    # Apply offset to all q_est
    r_est_all = R.from_quat(q_est[:, [1,2,3,0]])
    r_est_aligned = r_est_all * offset
    q_est_aligned_scipy = r_est_aligned.as_quat() # x, y, z, w
    # Convert back to w, x, y, z for manual error func
    q_est_aligned = np.stack([q_est_aligned_scipy[:,3], q_est_aligned_scipy[:,0], q_est_aligned_scipy[:,1], q_est_aligned_scipy[:,2]], axis=1)
    
    errors = quat_error_degrees(q_est_aligned, q_ref)
    rmse = np.sqrt(np.mean(errors**2))
    
    print(f"Comparison Results:")
    print(f"Mean Angular Error: {np.mean(errors):.2f} degrees")
    print(f"RMSE: {rmse:.2f} degrees")
    print(f"Max Error: {np.max(errors):.2f} degrees")
    
    # Plot
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot RPY
    rpy_est = r_est_aligned.as_euler('xyz', degrees=True)
    r_ref = R.from_quat(q_ref[:, [1,2,3,0]])
    rpy_ref = r_ref.as_euler('xyz', degrees=True)
    
    axs[0].plot(data.time, rpy_est[:,0], label='Solver Roll', color='r', linestyle='--')
    axs[0].plot(data.time, rpy_est[:,1], label='Solver Pitch', color='g', linestyle='--')
    axs[0].plot(data.time, rpy_est[:,2], label='Solver Yaw', color='b', linestyle='--')
    
    axs[0].plot(data.time, rpy_ref[:,0], label='Device Roll', color='r', alpha=0.5)
    axs[0].plot(data.time, rpy_ref[:,1], label='Device Pitch', color='g', alpha=0.5)
    axs[0].plot(data.time, rpy_ref[:,2], label='Device Yaw', color='b', alpha=0.5)
    axs[0].set_title("Orientation (RPY) Comparison (Aligned at t=0)")
    axs[0].legend()
    
    axs[1].plot(data.time, errors, color='k')
    axs[1].set_title("Angular Error (deg)")
    axs[1].set_xlabel("Time (s)")
    axs[1].grid(True)
    
    output_img = "test_comparison.png"
    plt.tight_layout()
    plt.savefig(output_img)
    print(f"Saved comparison plot to {output_img}")

if __name__ == "__main__":
    main()
