
import os
import argparse
from data_loader import DataLoader
from noise_db import noise_db
from solvers.pytorch_solver import PyTorchSolver
from visualizer import Visualizer

def main():
    parser = argparse.ArgumentParser(description="Phone Orientation Estimator")
    parser.add_argument("--device", type=str, default="pixel_10", help="Phone model name")
    parser.add_argument("--indoor", action="store_true", help="Use indoor noise parameters")
    parser.add_argument("--log", type=str, required=True, help="Path to sensor log .txt")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    
    args = parser.parse_args()
    
    print(f"Loading data from {args.log}...")
    loader = DataLoader(target_freq=30.0)
    # Note: manual_time_offset_sec could be an argument if we knew it
    data = loader.load_data(args.log, args.video)
    print(f"Loaded {len(data.time)} samples ({data.time[-1]:.2f}s).")
    
    print("Retrieving noise parameters...")
    # Fix: User argparse "indoor" is boolean, assume True means Indoor
    noise_params = noise_db.get_params(args.device, args.indoor)
    print(f"Noise Params: Acc={noise_params.accel_noise_sigma}, Mag={noise_params.mag_noise_sigma}")
    
    print("Solving for orientation (PyTorch)...")
    solver = PyTorchSolver()
    trajectory = solver.solve(data, noise_params)
    print("Solved.")
    
    print("Starting Visualization...")
    # Pass original video path for reference, but Visualizer currently expects 'assets/video.mp4' 
    # (We assume it's copied there by the setup script)
    viz = Visualizer(trajectory, args.video)
    viz.run()

if __name__ == "__main__":
    main()
