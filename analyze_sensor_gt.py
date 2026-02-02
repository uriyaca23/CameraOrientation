
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import argparse
from data_loader import DataLoader
import os

def compute_gt_angular_velocity(times, quats):
    """
    Computes angular velocity from orientation quaternions using central differences.
    q = [w, x, y, z]
    """
    # Create Rotation objects
    rots = R.from_quat(quats[:, [1,2,3,0]]) # xyzw
    
    # Calculate relative rotations matching the time steps
    # R_{i+1} = R_i * dR_i  =>  dR_i = R_i^T * R_{i+1}
    # But better to use: rotvec from R_i to R_{i+1}
    # angle_axis = (R_i^T * R_{i+1}).as_rotvec()
    
    # Vectorized approach
    # R_diff = R[:-1].inv() * R[1:]
    R_diff = rots[:-1].inv() * rots[1:]
    rotvecs = R_diff.as_rotvec() # Angle*Axis ~ w * dt
    
    dt = np.diff(times)
    # Avoid div by zero
    dt[dt == 0] = 1e-6
    
    ang_vel = rotvecs / dt[:, None]
    
    # Pad to match length
    ang_vel = np.vstack([ang_vel, ang_vel[-1]])
    
    return ang_vel

def analyze_sync_and_bias(data, out_dir):
    print("--- Analyzing Sensor vs GT Alignment ---")
    
    if getattr(data, 'orientation', None) is None or len(data.orientation) == 0:
        print("Error: No GT orientation found in data.")
        return

    t = data.time
    gyro_meas = data.gyro # Raw Gyro
    accel_meas = data.accel

    # 1. Compute GT Angular Velocity
    print("Computing GT Angular Velocity...")
    gyro_gt = compute_gt_angular_velocity(t, data.orientation)

    # 2. Analyze Gyro
    # Plot Components
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    labels = ['x', 'y', 'z']
    
    bias_est = []
    
    for i in range(3):
        ax = axes[i]
        ax.plot(t, gyro_meas[:, i], 'r-', label='Measured', alpha=0.7)
        ax.plot(t, gyro_gt[:, i], 'k--', label='GT Derived', alpha=0.7)
        
        # Simple Bias Estimate (Mean Difference)
        # Filter high frequency? No, simple mean diff.
        diff = gyro_meas[:, i] - gyro_gt[:, i]
        bias = np.mean(diff)
        bias_est.append(bias)
        
        # Correlation
        corr = np.corrcoef(gyro_meas[:, i], gyro_gt[:, i])[0, 1]
        
        ax.set_title(f"Gyro {labels[i]} (Corr: {corr:.3f}, Bias: {bias:.4f} rad/s)")
        ax.legend()
        ax.grid(True)
        ax.set_ylabel("rad/s")
        
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Gyroscope: Measured vs GT Derived")
    fig.savefig(os.path.join(out_dir, "analysis_gyro.png"))
    print(f"Estimated Gyro Bias: {bias_est}")

    # 3. Analyze Accel (Gravity)
    # Expected Gravity input in Body Frame
    # R(q)^T * [0, 0, 1] (Assuming World Z is UP? Or Down?)
    # Usually: Gravity vector in World is [0, 0, 9.8] or [0, 0, -9.8]?
    # Accelerometer measures Reaction Force. Stationary flat: +9.8 on Z.
    # So Gravity-caused-Accel in World is [0, 0, 9.8].
    # Expected Accel = R_bw * [0, 0, 1] * 9.8.
    # R_bw = rots.inv().
    
    rots = R.from_quat(data.orientation[:, [1,2,3,0]])
    # R maps Body -> World. We want World -> Body for vectors.
    g_world = np.array([0, 0, 1]) # Normalized
    
    # Apply inverse rotation to global vector
    g_body_expected = rots.inv().apply(g_world)
    
    # Accel Normalized
    accel_norm = data.accel / (np.linalg.norm(data.accel, axis=1, keepdims=True) + 1e-6)
    
    fig2, axes2 = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for i in range(3):
        ax = axes2[i]
        ax.plot(t, accel_norm[:, i], 'r-', label='Measured (Norm)', alpha=0.5)
        ax.plot(t, g_body_expected[:, i], 'k--', label='Expected Gravity', alpha=0.8)
        
        ax.set_title(f"Accel {labels[i]} Direction")
        ax.legend()
        ax.grid(True)
        
    axes2[-1].set_xlabel("Time (s)")
    fig2.suptitle("Accelerometer Direction vs GT Gravity Direction")
    fig2.savefig(os.path.join(out_dir, "analysis_accel.png"))
    
    plt.close('all')
    print("Analysis plots saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--offset", type=float, default=0.0)
    args = parser.parse_args()
    
    loader = DataLoader()
    data = loader.load_data(args.log, args.video, additional_offset_s=args.offset)
    
    analyze_sync_and_bias(data, ".")
