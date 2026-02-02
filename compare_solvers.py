import numpy as np
import matplotlib.pyplot as plt
from data_loader import DataLoader
from noise_db import noise_db
from solvers.pytorch_solver import PyTorchSolver
import argparse

# Try importing GTSAMSolver
try:
    from solvers.gtsam_solver import GTSAMSolver
    GTSAM_AVAILABLE = True
except ImportError:
    print("Warning: GTSAMSolver or gtsam library not found. Skipping GTSAM.")
    GTSAM_AVAILABLE = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, required=True)
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--device", type=str, default="pixel_10")
    args = parser.parse_args()

    # Load Data
    loader = DataLoader(target_freq=30.0)
    data = loader.load_data(args.log, args.video)
    noise_params = noise_db.get_params(args.device, is_indoor=True)

    # 1. Run PyTorch Solver
    print("Running PyTorchSolver...")
    solver_torch = PyTorchSolver()
    traj_torch = solver_torch.solve(data, noise_params)

    if not GTSAM_AVAILABLE:
        print("Cannot compare: GTSAM Unavailable.")
        return

    # 2. Run GTSAM Solver
    print("Running GTSAMSolver...")
    solver_gtsam = GTSAMSolver()
    try:
        traj_gtsam = solver_gtsam.solve(data, noise_params)
    except Exception as e:
        print(f"GTSAM Execution failed: {e}")
        return

    # 3. Compare
    print("Comparing trajectories...")
    # Calculate angular difference
    diffs = []
    for i in range(len(traj_torch.timestamps)):
        q1 = traj_torch.quaternions[i] # w, x, y, z
        q2 = traj_gtsam.quaternions[i] # w, x, y, z
        
        # Dot product
        dot = np.abs(np.dot(q1, q2))
        if dot > 1.0: dot = 1.0
        angle_diff_deg = 2 * np.arccos(dot) * 180.0 / np.pi
        diffs.append(angle_diff_deg)
        
    diffs = np.array(diffs)
    print(f"Mean Difference: {np.mean(diffs):.4f} deg")
    print(f"Max Difference: {np.max(diffs):.4f} deg")

    # Plot
    plt.figure()
    plt.plot(traj_torch.timestamps, diffs)
    plt.title("Angular Difference: PyTorch vs GTSAM")
    plt.xlabel("Time (s)")
    plt.ylabel("Difference (deg)")
    plt.savefig("solver_comparison.png")
    print("Saved solver_comparison.png")

if __name__ == "__main__":
    main()
