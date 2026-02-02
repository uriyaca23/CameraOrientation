import cv2
import numpy as np
import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from core.data_loader import DataLoader
import matplotlib.pyplot as plt

def find_video_start_event(video_path, threshold=10):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    first_light_frame = None
    
    frame_idx = 0
    brightness_history = []
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        brightness_history.append(brightness)
        
        if brightness > threshold and first_light_frame is None:
             first_light_frame = frame_idx
             # Continue a bit to see plot
        
        frame_idx += 1
        if frame_idx > fps * 5: # Check first 5 seconds
            break
            
    cap.release()
    
    time_of_event = first_light_frame / fps if first_light_frame is not None else None
    return time_of_event, brightness_history, fps

def find_sensor_start_event(data, threshold=0.1):
    # Accel magnitude changing from gravity (approx 9.8) or Gyro moving from 0
    # Let's use Gyro magnitude
    gyro_mag = np.linalg.norm(data.gyro, axis=1)
    
    # Find index where magnitude > threshold
    idx = np.where(gyro_mag > threshold)[0]
    if len(idx) > 0:
        first_motion_idx = idx[0]
        return data.time[first_motion_idx], gyro_mag
    return None, gyro_mag

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, required=True)
    parser.add_argument("--video", type=str, required=True)
    args = parser.parse_args()
    
    print(f"Analyzing sync for:\nVideo: {args.video}\nLog: {args.log}")
    
    # 1. Analyze Video (Light)
    # Detect when phone is picked up (dark -> light)
    print("Analyzing Video Brightness...")
    video_event_time, brightness, fps = find_video_start_event(args.video, threshold=30)
    print(f"Video Light Event detected at: {video_event_time}s")
    
    # 2. Analyze Sensor (Motion)
    print("Loading Sensor Data...")
    loader = DataLoader(target_freq=50.0) # High freq for precision
    data = loader.load_data(args.log, args.video)
    
    print("Analyzing Sensor Motion (Gyro)...")
    sensor_event_time, gyro_mag = find_sensor_start_event(data, threshold=0.5)
    print(f"Sensor Motion Event detected at: {sensor_event_time}s")
    
    if video_event_time is not None and sensor_event_time is not None:
        diff = video_event_time - sensor_event_time
        print(f"\n--- SYNC RESULT ---")
        print(f"Video Event: {video_event_time:.3f}s")
        print(f"Sensor Event: {sensor_event_time:.3f}s")
        print(f"Recommended Offset (Add to sensor time): {diff:.3f}s")
        print(f"If Video is ahead (Event happened later in video time line), we need to delay sensor data (add +diff).")
    else:
        print("Could not detect events in both streams.")

    # Plot
    plt.figure()
    
    # Brightness
    t_vid = np.arange(len(brightness)) / fps
    plt.subplot(2,1,1)
    plt.plot(t_vid, brightness, label='Video Brightness')
    if video_event_time:
        plt.axvline(video_event_time, color='r', linestyle='--', label='Light On')
    plt.legend()
    plt.title('Video Synchronization')
    
    # Gyro
    plt.subplot(2,1,2)
    plt.plot(data.time, gyro_mag, label='Gyro Magnitude')
    if sensor_event_time:
        plt.axvline(sensor_event_time, color='r', linestyle='--', label='Motion Start')
    plt.xlim([0, 5])
    plt.legend()
    plt.title('Sensor Synchronization')
    
    plt.tight_layout()
    plt.savefig('sync_analysis.png')
    print("Saved sync_analysis.png")

if __name__ == "__main__":
    main()
