
import sys
import os
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
import torch
from core.data_loader import DataLoader
from core.noise_db import noise_db
from solvers.pytorch_solver import PyTorchSolver

def main():
    log_path = r"data\google_pixel_10\indoor\exp1_uriya_apartment\sensorLog_fA4bFhD_ZGc_20260201T191641.txt"
    video_path = r"data\google_pixel_10\indoor\exp1_uriya_apartment\PXL_20260201_171650791.mp4"
    offset = 1.0
    
    print("Loading data...")
    loader = DataLoader(target_freq=30.0)
    data = loader.load_data(log_path, video_path, additional_offset_s=offset)
    
    print("Solving...")
    solver = PyTorchSolver()
    noise_params = noise_db.get_params("pixel_10", is_indoor=True)
    trajectory = solver.solve(data, noise_params)
    
    if trajectory.covariances is None:
        print("Covariances is NONE")
    else:
        cov = trajectory.covariances
        print(f"Covariance Shape: {cov.shape}")
        print(f"Max Val: {np.max(cov)}")
        print(f"Min Val: {np.min(cov)}")
        print(f"Mean Val: {np.mean(cov)}")
        
        diags = np.diagonal(cov, axis1=1, axis2=2)
        print(f"Diagonal Mean: {np.mean(diags, axis=0)}")
        print(f"Diagonal Max: {np.max(diags, axis=0)}")

if __name__ == "__main__":
    main()
