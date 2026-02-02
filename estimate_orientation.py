"""
Simple IMU to Orientation Estimator

Converts raw IMU sensor data to Yaw-Pitch-Roll (YPR) orientation using
the Google EKF sensor fusion algorithm.

Usage:
    from estimate_orientation import estimate_orientation
    
    ypr, cov, timestamps = estimate_orientation(
        gyro=gyro_data,      # Nx3 array [rad/s]
        accel=accel_data,    # Nx3 array [m/s²]
        mag=mag_data,        # Nx3 array [µT] (optional)
        timestamps=times,    # N array [seconds]
        model="pixel_9",     # Optional: phone model for noise params
        is_indoor=True       # Indoor/outdoor environment
    )
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing import Optional, Tuple
import sys
import os

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.noise_db import noise_db, NoiseParams
from solvers.google_solver import GoogleEKF


def estimate_orientation(
    gyro: np.ndarray,
    accel: np.ndarray,
    timestamps: np.ndarray,
    mag: Optional[np.ndarray] = None,
    model: Optional[str] = None,
    is_indoor: bool = True,
    sampling_rate: float = 100.0
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Estimate orientation from IMU data.
    
    Args:
        gyro: Nx3 array of gyroscope readings [rad/s]
        accel: Nx3 array of accelerometer readings [m/s²]
        timestamps: N array of timestamps [seconds]
        mag: Nx3 array of magnetometer readings [µT] (optional, zeros if None)
        model: Phone model name for noise parameters (e.g., "pixel_9", "galaxy_s24")
               If None, uses generic parameters.
        is_indoor: True for indoor (high mag noise), False for outdoor
        sampling_rate: IMU sampling rate in Hz (used for noise calculations)
    
    Returns:
        ypr: Nx3 array of Yaw, Pitch, Roll angles [degrees]
        cov: Nx3x3 array of orientation covariance matrices [rad²] (if available)
        timestamps: N array of timestamps (passthrough for convenience)
    
    Example:
        >>> import numpy as np
        >>> N = 1000
        >>> t = np.linspace(0, 10, N)
        >>> gyro = np.random.randn(N, 3) * 0.01  # Small random motion
        >>> accel = np.tile([0, 0, 9.81], (N, 1))  # Gravity pointing down
        >>> ypr, cov, _ = estimate_orientation(gyro, accel, t, model="pixel_9")
        >>> print(f"Final orientation: Y={ypr[-1,0]:.1f}°, P={ypr[-1,1]:.1f}°, R={ypr[-1,2]:.1f}°")
    """
    # Validate inputs
    n_samples = len(timestamps)
    assert gyro.shape == (n_samples, 3), f"gyro shape mismatch: expected ({n_samples}, 3)"
    assert accel.shape == (n_samples, 3), f"accel shape mismatch: expected ({n_samples}, 3)"
    
    if mag is None:
        mag = np.zeros((n_samples, 3))
    else:
        assert mag.shape == (n_samples, 3), f"mag shape mismatch: expected ({n_samples}, 3)"
    
    # Get noise parameters
    if model is not None:
        params = noise_db.get_params(model, is_indoor=is_indoor, sampling_rate_hz=sampling_rate)
    else:
        # Use generic midrange parameters
        params = noise_db.get_params("generic", is_indoor=is_indoor, sampling_rate_hz=sampling_rate)
    
    # Initialize EKF with device-specific noise parameters
    ekf = GoogleEKF(
        gyro_var=params.gyro_noise_sigma ** 2,
        gyro_bias_var=params.gyro_bias_instability ** 2 if params.gyro_bias_instability > 0 else 1e-10,
        acc_stdev=params.accel_noise_sigma,
        mag_stdev=params.mag_noise_sigma
    )
    
    # Process all samples
    quaternions = []
    covariances = []
    
    import torch
    
    for i in range(n_samples):
        # Calculate dt
        if i == 0:
            dt = 1.0 / sampling_rate
        else:
            dt = timestamps[i] - timestamps[i-1]
            if dt <= 0:
                dt = 1.0 / sampling_rate
        
        # Convert to torch tensors
        w = torch.tensor(gyro[i], dtype=torch.float32)
        a = torch.tensor(accel[i], dtype=torch.float32)
        
        # Update EKF
        ekf.predict(w, dt)
        
        # Accel update (normalize and use gravity reference)
        a_norm = torch.norm(a)
        if abs(a_norm - 9.81) < 2.0:  # Roughly stationary
            g_ref = torch.tensor([0., 0., 1.])
            ekf.update(a / a_norm, g_ref, ekf.ACC_STDEV)
        
        if np.linalg.norm(mag[i]) > 1e-3:
            m = torch.tensor(mag[i], dtype=torch.float32)
            m_ref = torch.tensor([0., 1., 0.])
            ekf.update(m / torch.norm(m), m_ref, ekf.MAG_STDEV)
        
        # Store results
        quaternions.append(ekf.q.detach().cpu().numpy().copy())
        covariances.append(ekf.P[:3, :3].detach().cpu().numpy().copy())  # Orientation covariance (3x3)
    
    quaternions = np.array(quaternions)  # Nx4 [w, x, y, z]
    covariances = np.array(covariances)  # Nx3x3
    
    # Convert quaternions to YPR (Yaw, Pitch, Roll) in degrees
    # scipy uses [x, y, z, w] format, we have [w, x, y, z]
    rotations = R.from_quat(quaternions[:, [1, 2, 3, 0]])  # Reorder to [x, y, z, w]
    ypr = rotations.as_euler('ZYX', degrees=True)  # Returns [yaw, pitch, roll]
    
    return ypr, covariances, timestamps


def estimate_orientation_simple(
    gyro: np.ndarray,
    accel: np.ndarray,
    dt: float = 0.01,
    model: Optional[str] = None,
    is_indoor: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Simplified interface with constant timestep.
    
    Args:
        gyro: Nx3 gyroscope [rad/s]
        accel: Nx3 accelerometer [m/s²]
        dt: Constant time step [seconds]
        model: Phone model name
        is_indoor: Indoor/outdoor flag
    
    Returns:
        ypr: Nx3 Yaw, Pitch, Roll [degrees]
        cov: Nx3x3 covariance matrices [rad²]
    """
    n = len(gyro)
    timestamps = np.arange(n) * dt
    ypr, cov, _ = estimate_orientation(
        gyro=gyro,
        accel=accel,
        timestamps=timestamps,
        model=model,
        is_indoor=is_indoor,
        sampling_rate=1.0/dt
    )
    return ypr, cov


# =============================================================================
# Command-line interface
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Estimate orientation from IMU data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python estimate_orientation.py --input data.npz --model pixel_9 --indoor
  python estimate_orientation.py --demo
        """
    )
    parser.add_argument("--input", type=str, help="Path to .npz file with gyro, accel, timestamps")
    parser.add_argument("--model", type=str, default=None, help="Phone model (e.g., pixel_9, galaxy_s24)")
    parser.add_argument("--indoor", action="store_true", help="Indoor environment (default)")
    parser.add_argument("--outdoor", action="store_true", help="Outdoor environment")
    parser.add_argument("--output", type=str, default=None, help="Output .npz file for results")
    parser.add_argument("--demo", action="store_true", help="Run demo with synthetic data")
    
    args = parser.parse_args()
    
    is_indoor = not args.outdoor
    
    if args.demo:
        # Demo with synthetic stationary data
        print("Running demo with synthetic IMU data...")
        
        n = 500
        dt = 0.01
        t = np.arange(n) * dt
        
        # Stationary phone: zero gyro, gravity pointing down
        gyro = np.random.randn(n, 3) * 0.001  # Small noise
        accel = np.tile([0, 0, 9.81], (n, 1)) + np.random.randn(n, 3) * 0.01
        
        ypr, cov, _ = estimate_orientation(
            gyro=gyro,
            accel=accel,
            timestamps=t,
            model=args.model,
            is_indoor=is_indoor
        )
        
        print(f"\nResults (stationary phone):")
        print(f"  Model: {args.model or 'generic'}")
        print(f"  Environment: {'indoor' if is_indoor else 'outdoor'}")
        print(f"  Samples: {n}")
        print(f"\nFinal Orientation:")
        print(f"  Yaw:   {ypr[-1, 0]:7.2f}° ± {np.sqrt(cov[-1, 0, 0]) * 180/np.pi:.2f}°")
        print(f"  Pitch: {ypr[-1, 1]:7.2f}° ± {np.sqrt(cov[-1, 1, 1]) * 180/np.pi:.2f}°")
        print(f"  Roll:  {ypr[-1, 2]:7.2f}° ± {np.sqrt(cov[-1, 2, 2]) * 180/np.pi:.2f}°")
        
    elif args.input:
        # Load data from file
        print(f"Loading data from {args.input}...")
        data = np.load(args.input)
        
        gyro = data['gyro']
        accel = data['accel']
        timestamps = data.get('timestamps', np.arange(len(gyro)) * 0.01)
        mag = data.get('mag', None)
        
        print(f"Processing {len(gyro)} samples...")
        
        ypr, cov, _ = estimate_orientation(
            gyro=gyro,
            accel=accel,
            timestamps=timestamps,
            mag=mag,
            model=args.model,
            is_indoor=is_indoor
        )
        
        print(f"\nFinal Orientation:")
        print(f"  Yaw:   {ypr[-1, 0]:7.2f}°")
        print(f"  Pitch: {ypr[-1, 1]:7.2f}°")
        print(f"  Roll:  {ypr[-1, 2]:7.2f}°")
        
        if args.output:
            np.savez(args.output, ypr=ypr, covariance=cov, timestamps=timestamps)
            print(f"\nResults saved to {args.output}")
    else:
        parser.print_help()
