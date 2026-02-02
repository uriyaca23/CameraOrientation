
import unittest
import os
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.data_loader import DataLoader

class TestDataLoader(unittest.TestCase):
    def test_load_sample_data(self):
        # Path to the specific sample file we analyzed
        # We assume the user has this path structure based on previous exploration
        base_path = r"c:\Users\uriya\PycharmProjects\CameraOrientation\data\google_pixel_10\indoor\exp1_uriya_apartment"
        log_file = os.path.join(base_path, "sensorLog_fA4bFhD_ZGc_20260201T191641.txt")
        video_file = os.path.join(base_path, "PXL_20260201_171650791.mp4")
        
        if not os.path.exists(log_file):
            print(f"Skipping test_load_sample_data: File not found at {log_file}")
            return

        loader = DataLoader(target_freq=50.0)
        data = loader.load_data(log_file, video_file)
        
        # Checks
        self.assertTrue(len(data.time) > 0, "Time array should not be empty")
        self.assertTrue(data.accel.shape[1] == 3, "Accel should have 3 columns")
        self.assertTrue(data.gyro.shape[1] == 3, "Gyro should have 3 columns")
        self.assertTrue(data.mag.shape[1] == 3, "Mag should have 3 columns")
        self.assertEqual(len(data.time), len(data.accel), "Time and Accel length mismatch")
        
        # Check frequency
        dt = np.diff(data.time)
        mean_dt = np.mean(dt)
        expected_dt = 1.0 / 50.0
        self.assertAlmostEqual(mean_dt, expected_dt, delta=1e-4, msg=f"Resampling interval incorrect. Got {mean_dt}, expected {expected_dt}")

        print(f"Loaded {len(data.time)} samples.")
        print(f"Duration: {data.time[-1] - data.time[0]:.2f} seconds")
        print(f"First Unix Timestamp: {data.unix_timestamps[0]}")
        
        # Check if Unix timestamp is reasonable (2026 -> 1769...)
        # 1.7e12 ms
        self.assertGreater(data.unix_timestamps[0], 1.7e12, "Unix timestamp seems too small (not in ms?)")
        
if __name__ == '__main__':
    unittest.main()
