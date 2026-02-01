
import pandas as pd
import numpy as np
import datetime
from dataclasses import dataclass
from typing import List, Tuple, Optional
import os
import re
import cv2

@dataclass
class SensorData:
    """
    Container for synced and resampled sensor data.
    Timestamps are in seconds (relative to video start).
    """
    time: np.ndarray      # T x 1
    accel: np.ndarray     # T x 3 (x, y, z)
    gyro: np.ndarray      # T x 3 (x, y, z)
    mag: np.ndarray       # T x 3 (x, y, z)
    unix_timestamps: np.ndarray # T x 1

class DataLoader:
    def __init__(self, target_freq: float = 50.0):
        self.target_freq = target_freq

    def _parse_video_timestamp(self, video_path: str) -> float:
        """
        Parses Pixel filename format: PXL_YYYYMMDD_HHMMSSmmm.mp4
        Returns Unix timestamp (float). Assumes filename time is UTC.
        """
        basename = os.path.basename(video_path)
        # Regex for PXL_20260201_171650791.mp4
        match = re.search(r'PXL_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})(\d{3})', basename)
        if match:
            y, m, d, H, M, S, ms = map(int, match.groups())
            dt = datetime.datetime(y, m, d, H, M, S, ms*1000, tzinfo=datetime.timezone.utc)
            return dt.timestamp()
        
        # Fallback or other formats?
        print(f"Warning: Could not parse timestamp from video filename {basename}. Assuming 0 offset.")
        return 0.0

    def get_video_duration(self, video_path: str) -> float:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0.0
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        if fps > 0:
            return frames / fps
        return 0.0

    def load_data(self, sensor_log_path: str, video_path: str) -> SensorData:
        # 1. Parse Sensor Log and establish Uptime -> Unix mapping
        df_raw, uptime_to_unix_offset_ms = self._parse_log_file(sensor_log_path)
        
        if df_raw.empty:
            raise ValueError("No valid data found in sensor log.")

        # 2. Get Video Start and Duration
        video_start_unix = self._parse_video_timestamp(video_path)
        video_duration = self.get_video_duration(video_path)
        
        print(f"Video Start (Unix): {video_start_unix:.3f}")
        print(f"Video Duration: {video_duration:.2f}s")
        
        # 3. Define Time Grid relative to VIDEO
        # We need to map Video Start Unix -> Uptime
        # uptime = unix - offset
        video_start_uptime_ms = (video_start_unix * 1000.0) - uptime_to_unix_offset_ms
        video_end_uptime_ms = video_start_uptime_ms + (video_duration * 1000.0)
        
        print(f"Cropping Data to Uptime: {video_start_uptime_ms:.0f} to {video_end_uptime_ms:.0f}")
        
        # 4. Filter Dataframes (optimization: filter before interpolation)
        # Give a small buffer
        buffer_ms = 500
        df_raw = df_raw[(df_raw['timestamp_ms'] >= video_start_uptime_ms - buffer_ms) & 
                        (df_raw['timestamp_ms'] <= video_end_uptime_ms + buffer_ms)]
        
        # 5. Resample
        dt_ms = 1000.0 / self.target_freq
        # Strictly from video start to end
        time_grid_ms = np.arange(video_start_uptime_ms, video_end_uptime_ms, dt_ms)
        
        # Split types
        acc_df = df_raw[df_raw['type'] == 'ACC'].sort_values('timestamp_ms')
        gyr_df = df_raw[df_raw['type'] == 'GYR'].sort_values('timestamp_ms')
        mag_df = df_raw[df_raw['type'] == 'MAG'].sort_values('timestamp_ms')

        # Interpolate
        acc_interp = self._interp_sensor(time_grid_ms, acc_df)
        gyr_interp = self._interp_sensor(time_grid_ms, gyr_df)
        mag_interp = self._interp_sensor(time_grid_ms, mag_df)
        
        # 6. Normalize Time (t=0 is Video Start)
        t_normalized = (time_grid_ms - video_start_uptime_ms) / 1000.0
        unix_timestamps = (time_grid_ms + uptime_to_unix_offset_ms) / 1000.0

        return SensorData(
            time=t_normalized,
            accel=acc_interp,
            gyro=gyr_interp,
            mag=mag_interp,
            unix_timestamps=unix_timestamps
        )

    def _parse_log_file(self, path: str) -> Tuple[pd.DataFrame, float]:
        data_records = []
        uptime_to_unix_offset_ms = 0.0
        found_sync = False
        
        columns = ['timestamp_ms', 'type', 'x', 'y', 'z']
        
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2: continue
                try:
                    uptime_ms = int(parts[0])
                    msg_type = parts[1]
                    
                    if msg_type in ['ACC', 'GYR', 'MAG']:
                        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                        data_records.append((uptime_ms, msg_type, x, y, z))
                        
                    elif msg_type == 'RSSCELL' and not found_sync:
                         # Use RSSCELL unix timestamp to sync
                         # parts[0] is unix_ms for this specific line type?
                         # The file format puts content in uptime column sometimes?
                         # Based on previous analysis:
                         # Line 114: 1769966201790 (Unix) RSSCELL LTE ...
                         # We need to find the uptime of the previous line (Line 113: 376017461)
                         # So Uptime ~ 376017461 corresponds to Unix 1769966201790
                         # Offset = Unix - Uptime
                         unix_ts = float(parts[0])
                         if data_records:
                             last_uptime = data_records[-1][0]
                             if unix_ts > last_uptime * 2: # Heuristic check
                                 uptime_to_unix_offset_ms = unix_ts - last_uptime
                                 found_sync = True
                except ValueError:
                    continue
                    
        df = pd.DataFrame(data_records, columns=columns)
        return df, uptime_to_unix_offset_ms

    def _interp_sensor(self, target_time_ms: np.ndarray, source_df: pd.DataFrame) -> np.ndarray:
        if source_df.empty:
            return np.zeros((len(target_time_ms), 3))
        t_src = source_df['timestamp_ms'].values
        vals_src = source_df[['x', 'y', 'z']].values
        
        x_new = np.interp(target_time_ms, t_src, vals_src[:, 0])
        y_new = np.interp(target_time_ms, t_src, vals_src[:, 1])
        z_new = np.interp(target_time_ms, t_src, vals_src[:, 2])
        
        return np.stack([x_new, y_new, z_new], axis=1)

if __name__ == "__main__":
    print("DataLoader updated.")
