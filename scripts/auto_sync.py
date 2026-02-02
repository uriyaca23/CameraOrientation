
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loader import DataLoader
import argparse

def get_optical_flow_magnitude(video_path, max_frames=900):
    """
    Extracts average optical flow magnitude per frame from video.
    Returns: time_v (sec), flow_mag
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    ret, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    flow_magnitudes = []
    
    print(f"Extracting Optical Flow from {video_path}...")
    
    count = 0
    while count < max_frames:
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Dense Optical Flow (Farneback)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        
        # Mean Magnitude
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_magnitudes.append(np.mean(mag))
        
        prev_gray = gray
        count += 1
        
    cap.release()
    
    t_v = np.arange(len(flow_magnitudes)) / fps
    return t_v, np.array(flow_magnitudes)

def find_best_offset(log_path, video_path, duration=30.0):
    # 1. Load Gyro Data
    loader = DataLoader(target_freq=30.0) # Match video approx
    # Load with 0 offset initially
    data = loader.load_data(log_path, video_path, additional_offset_s=0.0)
    
    # Gyro Magnitude
    gyro_mag = np.linalg.norm(data.gyro, axis=1)
    # Remove Bias/Gravity roughly (High Pass or just assume correlation works)
    # Gyro measures rotational velocity. Optical flow measures pixel motion ~ rotation.
    # So direct correlation should work.
    
    # 2. Get Optical Flow
    t_video, flow_mag = get_optical_flow_magnitude(video_path, max_frames=int(duration*30))
    
    # 3. Resample Gyro to Video Time
    # Data time is already aligned to video start (0.0 = Start of Video)
    # But checking if there is a shift.
    
    # Interpolate Gyro to Video Frames
    gyro_interp = np.interp(t_video, data.time, gyro_mag)
    
    # Normalize
    a = (gyro_interp - np.mean(gyro_interp)) / (np.std(gyro_interp) + 1e-6)
    b = (flow_mag - np.mean(flow_mag)) / (np.std(flow_mag) + 1e-6)
    
    # Cross Correlation
    correlation = correlate(a, b, mode='full')
    lags = correlation_lags(a.size, b.size, mode='full')
    
    best_lag_idx = np.argmax(correlation)
    best_lag_frames = lags[best_lag_idx]
    
    fps = 30.0 # From video ideally, but loader used 30.0 target
    offset_seconds = best_lag_frames / fps
    
    # Lag > 0 means signal A (Gyro) is shifted to right? 
    # if A[t] = B[t+tau], lag is -tau.
    # We want to find shift S such that Gyro(t-S) matches Video(t).
    # If Gyro peaks later than Video, we need to shift Gyro left (negative offset).
    # correlate(a, b). Argmax positive means A is AHEAD of B? 
    # Let's verify manually.
    
    print(f"Best Correlation Lag: {best_lag_frames} frames ({offset_seconds:.3f} s)")
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.plot(t_video, a, label='Gyro Mag (Norm)')
    plt.plot(t_video, b, label='Optical Flow (Norm)')
    plt.plot(t_video - offset_seconds, a, label='Gyro Shifted', linestyle='--')
    plt.legend()
    plt.title(f"Sync Alignment (Offset: {offset_seconds:.3f}s)")
    plt.savefig('sync_debug.png')
    
    # The `data_loader` applies `additional_offset_s` to the timestamp calculation.
    # video_start_unix_adjusted = video_start_unix + additional_offset_s.
    # If offset_seconds is positive (Gyro matches Video if shifted right?), 
    # wait.
    # If Gyro happens at t=2 and Video at t=1. Gyro is late.
    # We want Gyro at t=1. So we subtract 1s from Gyro time.
    # Loader logic:
    # video_start = video_start_orig + additional_offset
    # if we increase video_start, we essentially crop LATER into the gyro stream.
    # So if Gyro is Late (Event at 2s), and Video is Early (Event at 1s).
    # We want to map Video 1s to Gyro 2s.
    # Video T=0 corresponds to Real T.
    # Gyro T=0 corresponds to Real T + Delta.
    
    # Return raw offset, user interprets.
    return offset_seconds

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--video", required=True)
    args = parser.parse_args()
    
    offset = find_best_offset(args.log, args.video)
    print(f"SUGGESTED_OFFSET={offset}")
