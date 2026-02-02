import pandas as pd
import numpy as np
import datetime
from dataclasses import dataclass
from typing import List, Tuple, Optional
import os
import re
import cv2
from scipy.spatial.transform import Rotation as R

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
    orientation: np.ndarray # T x 4 (w, x, y, z) - Device Ground Truth
    unix_timestamps: np.ndarray # T x 1

class DataLoader:
    def __init__(self, target_freq: float = 30.0): # Reduced default freq to 30Hz or user defined
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
        
        # Try finding simpler PXL format or just return 0
        match_simple = re.search(r'PXL_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})', basename)
        if match_simple:
            y, m, d, H, M, S = map(int, match_simple.groups())
            dt = datetime.datetime(y, m, d, H, M, S, 0, tzinfo=datetime.timezone.utc)
            return dt.timestamp()

        print(f"Warning: Could not parse timestamp from video filename {basename}. Assuming 0.")
        return 0.0

    def _parse_log_timestamp(self, log_path: str) -> float:
        """
        Parses Sensor Log filename: sensorLog_ID_ID_YYYYMMDDThhmmss.txt
        Returns Unix timestamp (float). Tries to infer Timezone match with Video if possible, 
        but initially returns naive or UTC interpretation.
        """
        basename = os.path.basename(log_path)
        # Regex for sensorLog_..._20260201T191641.txt
        match = re.search(r'_(\d{8})T(\d{6})', basename)
        if match:
            date_part, time_part = match.groups()
            y, m, d = int(date_part[:4]), int(date_part[4:6]), int(date_part[6:])
            H, M, S = int(time_part[:2]), int(time_part[2:4]), int(time_part[4:])
            
            # Create a naive datetime first
            dt = datetime.datetime(y, m, d, H, M, S)
            # We don't know the timezone yet. Return simple timestamp assuming it *might* be UTC for now,
            # adjustment regarding video happens later.
            return dt.replace(tzinfo=datetime.timezone.utc).timestamp()
            
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

    def load_data(self, sensor_log_path: str, video_path: str, additional_offset_s: float = 0.0) -> SensorData:
        # 1. Parse Video Timestamp (UTC)
        video_start_unix = self._parse_video_timestamp(video_path)
        video_duration = self.get_video_duration(video_path)
        
        print(f"Video Start (Video Filename UTC): {video_start_unix:.3f}")
        print(f"Video Duration: {video_duration:.2f}s")

        # 2. Parse Log Timestamp (Filename)
        log_start_unix_naive = self._parse_log_timestamp(sensor_log_path)
        
        # 3. Heuristic Timezone Adjustment
        # If Log is e.g. 19:16 and Video is 17:16, difference is ~7200s.
        # We assume they were recorded roughly at the same time (> 10 mins diff is suspicious for this task).
        diff = log_start_unix_naive - video_start_unix
        
        # Round diff to nearest hour (3600s)
        tz_offset_seconds = round(diff / 3600.0) * 3600.0
        
        # If the diff suggests a timezone offset (e.g. 1h, 2h, 3h...), subtract it from the log timestamp 
        # to bring it to UTC equivalent of the Video.
        # Example: Log 19:00, Video 17:00. Diff +2h. Log UTC = 19:00 - 2h = 17:00.
        log_start_unix_utc = log_start_unix_naive - tz_offset_seconds
        
        if abs(log_start_unix_utc - video_start_unix) > 600:
             print(f"Warning: Even after timezone adjustment, Log and Video start times differ by {log_start_unix_utc - video_start_unix:.1f}s.")
             # Fallback: Trust the raw diff? No, stick to adjustment.
        
        print(f"Log Filename Raw Timestamp: {log_start_unix_naive:.3f}")
        print(f"Detected Timezone Offset: {tz_offset_seconds/3600:.0f} hours")
        print(f"Log Start (Adjusted UTC): {log_start_unix_utc:.3f}")

        # 4. Parse Log File Content
        df_raw, first_uptime_ms = self._parse_log_file(sensor_log_path)
        
        if df_raw.empty:
            raise ValueError("No valid data found in sensor log.")

        # 5. Establish Sync: Uptime -> Unix
        # We assume the First Line of the Log corresponds closely to the File Creation timestamp.
        # Uptime (First Line) <-> Log Start (UTC)
        # Thus: unix = (uptime - first_uptime) / 1000 + log_start_utc
        # offset = unix*1000 - uptime
        # We want: uptime_to_unix_offset_ms
        # unix * 1000 = uptime + offset
        # offset = log_start_unix_utc * 1000.0 - first_uptime_ms
        
        uptime_to_unix_offset_ms = (log_start_unix_utc * 1000.0) - first_uptime_ms
        print(f"Calculated Uptime Offset: {uptime_to_unix_offset_ms:.0f} ms")

        # 6. Define Time Grid relative to VIDEO
        # Video Start Frame Uptime:
        video_start_uptime_ms = (video_start_unix * 1000.0) - uptime_to_unix_offset_ms
        video_end_uptime_ms = video_start_uptime_ms + (video_duration * 1000.0)
        
        print(f"Cropping Data to Uptime: {video_start_uptime_ms:.0f} to {video_end_uptime_ms:.0f}")
        
        buffer_ms = 1000
        # Apply additional offset: Shift sensor time.
        # If offset is +1s, it means sensor events happen 1s later than expected relative to video.
        # Or does it mean we shift the data?
        # Let's say we want to align: sensor_t_new = sensor_t_old + offset.
        # So we just add offset to unix_timestamps or adjust calc?
        # Actually easier to adjust video_start_uptime_ms or the offsets.
        
        # We adjust uptime_to_unix_offset_ms by the offset amount.
        # If we need to delay sensor data (sensor is ahead of video, i.e., sensor says t=0 but video says t=1),
        # we need to shift sensor timestamps.
        
        # Let's simple apply it to the final result time grid.
        # Actually, let's just add it to the sync calculation.
        
        # uptime_to_unix_offset_ms connects Uptime to Unix.
        # video_start_uptime_ms is derived from video_start_unix.
        
        # If we add offset to video_start_unix, we shift the video start time.
        # video_start_unix_adjusted = video_start_unix + additional_offset_s
        
        video_start_uptime_ms = ((video_start_unix + additional_offset_s) * 1000.0) - uptime_to_unix_offset_ms
        video_end_uptime_ms = video_start_uptime_ms + (video_duration * 1000.0)

        df_raw = df_raw[(df_raw['timestamp_ms'] >= video_start_uptime_ms - buffer_ms) & 
                        (df_raw['timestamp_ms'] <= video_end_uptime_ms + buffer_ms)]
        
        if df_raw.empty:
             print("Warning: Cropped dataframe is empty! Check synchronization logic.")

        # 7. Resample
        dt_ms = 1000.0 / self.target_freq
        # Ensure we cover the full duration including the last frame
        time_grid_ms = np.arange(video_start_uptime_ms, video_end_uptime_ms + dt_ms, dt_ms)
        
        # Split & Sort
        acc_df = df_raw[df_raw['type'] == 'ACC'].sort_values('timestamp_ms')
        gyr_df = df_raw[df_raw['type'] == 'GYR'].sort_values('timestamp_ms')
        mag_df = df_raw[df_raw['type'] == 'MAG'].sort_values('timestamp_ms')
        ori_df = df_raw[df_raw['type'] == 'ORI'].sort_values('timestamp_ms')

        # Interpolate
        # We assume 3-component vectors for Acc/Gyr/Mag
        acc_interp = self._interp_vectors(time_grid_ms, acc_df, ['x', 'y', 'z'])
        gyr_interp = self._interp_vectors(time_grid_ms, gyr_df, ['x', 'y', 'z'])
        mag_interp = self._interp_vectors(time_grid_ms, mag_df, ['x', 'y', 'z'])
        
        # Slerp for Orientation? For now, simple component-wise linear interp is OK for small steps 
        # (needs renormalization afterwards)
        ori_interp = self._interp_vectors(time_grid_ms, ori_df, ['v1', 'v2', 'v3', 'v4'])
        # Normalize quaternions
        norms = np.linalg.norm(ori_interp, axis=1, keepdims=True)
        # Avoid division by zero, set to identity [1,0,0,0] if norm is small (missing chunks)
        valid_mask = norms > 1e-6
        ori_interp[~valid_mask.flatten()] = [1,0,0,0]
        ori_interp[valid_mask.flatten()] /= norms[valid_mask.flatten()]
        
        t_normalized = (time_grid_ms - video_start_uptime_ms) / 1000.0
        unix_timestamps = (time_grid_ms + uptime_to_unix_offset_ms) / 1000.0

        return SensorData(
            time=t_normalized,
            accel=acc_interp,
            gyro=gyr_interp,
            mag=mag_interp,
            orientation=ori_interp,
            unix_timestamps=unix_timestamps
        )

    def _parse_log_file(self, path: str) -> Tuple[pd.DataFrame, float]:
        data_records = []
        first_uptime_ms = None
        
        # We need flexible columns because ORI has 4 vals, others have 3
        # Let's verify format.
        # ACC: timestamp, type, x, y, z
        # ORI: timestamp, type, x, y, z, w (or w,x,y,z?)
        # Looking at log: 376017337	ORI	0.7302354	-0.0024199497	0.010854652	0.6831051
        # This is 4 values.
        
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 3: continue
                try:
                    uptime_ms = int(parts[0])
                    msg_type = parts[1]
                    
                    if first_uptime_ms is None:
                        first_uptime_ms = uptime_ms
                    
                    if msg_type in ['ACC', 'GYR', 'MAG']:
                        if len(parts) >= 5:
                            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                            data_records.append({'timestamp_ms': uptime_ms, 'type': msg_type, 'x': x, 'y': y, 'z': z})
                            
                    elif msg_type == 'ORI':
                        if len(parts) >= 6:
                            v1, v2, v3, v4 = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                            data_records.append({'timestamp_ms': uptime_ms, 'type': msg_type, 
                                                 'v1': v1, 'v2': v2, 'v3': v3, 'v4': v4})

                except ValueError:
                    continue
                    
        df = pd.DataFrame(data_records)
        return df, (first_uptime_ms if first_uptime_ms is not None else 0.0)

    def _interp_vectors(self, target_time_ms: np.ndarray, source_df: pd.DataFrame, cols: List[str]) -> np.ndarray:
        if source_df.empty:
            return np.zeros((len(target_time_ms), len(cols)))
            
        t_src = source_df['timestamp_ms'].values
        vals_src = source_df[cols].values
        
        result = []
        for i in range(len(cols)):
             interp_col = np.interp(target_time_ms, t_src, vals_src[:, i])
             result.append(interp_col)
             
        return np.stack(result, axis=1)

if __name__ == "__main__":
    print("DataLoader updated.")
