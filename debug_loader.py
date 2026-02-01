from data_loader import DataLoader
import os

log_path = "data/google_pixel_10/indoor/exp1_uriya_apartment/sensorLog_fA4bFhD_ZGc_20260201T191641.txt"
video_path = "data/google_pixel_10/indoor/exp1_uriya_apartment/PXL_20260201_171650791.mp4"

print(f"Testing DataLoader with:")
print(f"Log: {log_path}")
print(f"Video: {video_path}")

try:
    loader = DataLoader()
    data = loader.load_data(log_path, video_path)
    print("\n--- Result ---")
    print(f"Data Points: {len(data.time)}")
    print(f"Time Range: {data.time[0]:.2f}s to {data.time[-1]:.2f}s")
    print(f"Unix Timestamp Start: {data.unix_timestamps[0]:.0f}")
except Exception as e:
    print(f"\nERROR: {e}")
