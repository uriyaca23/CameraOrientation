
import unittest
import numpy as np
import os
import sys
sys.path.append(r"c:\Users\uriya\PycharmProjects\CameraOrientation")

from visualizer import get_phone_mesh
from solvers.pytorch_solver import PyTorchSolver
from data_loader import SensorData
from noise_db import noise_db

class TestRobust(unittest.TestCase):
    
    def test_phone_mesh_generation(self):
        """Test if phone mesh returns valid plotly traces."""
        q_identity = np.array([1.0, 0.0, 0.0, 0.0]) # w, x, y, z
        traces = get_phone_mesh(q_identity)
        
        # Should have traces for Body, Screen, Axises
        self.assertGreater(len(traces), 2, "Visualizer should return body and screen traces")
        
        # Check if traces are Mesh3d
        mesh_types = [t.type for t in traces if t.type == 'mesh3d']
        self.assertGreater(len(mesh_types), 1, "Should contain Mesh3d maps")

    def test_solver_uncertainty(self):
        """Test if solver returns non-zero covariance."""
        # Mock Data
        N = 100
        time = np.linspace(0, 2, N)
        accel = np.zeros((N, 3)); accel[:, 2] = 9.81 # Down
        gyro = np.zeros((N, 3))
        mag = np.zeros((N, 3)); mag[:, 1] = 30.0 # North
        
        data = SensorData(time, accel, gyro, mag, time*1000)
        params = noise_db.get_params("generic", is_indoor=True)
        
        solver = PyTorchSolver()
        traj = solver.solve(data, params)
        
        # Check covariance
        self.assertEqual(traj.covariances.shape, (N, 3, 3))
        self.assertTrue(np.all(traj.covariances[:, 0, 0] > 0), "Roll uncertainty should be > 0")
        self.assertTrue(np.all(traj.covariances[:, 2, 2] > 0), "Yaw uncertainty should be > 0 (especially indoor)")
        # Indoor mag noise is high, so yaw variance should be larger than roll
        self.assertGreater(traj.covariances[0, 2, 2], traj.covariances[0, 0, 0], "Yaw uncertainty should be higher indoors")

    def test_video_sync_logic(self):
        """Verify video path exists in assets."""
        asset_video = r"c:\Users\uriya\PycharmProjects\CameraOrientation\assets\video.mp4"
        self.assertTrue(os.path.exists(asset_video), "Video file must exist in assets for Dash to work")

if __name__ == '__main__':
    unittest.main()
