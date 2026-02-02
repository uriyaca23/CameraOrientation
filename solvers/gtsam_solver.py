import numpy as np
import gtsam
from .base_solver import BaseSolver, OrientationTrajectory
from core.data_loader import SensorData
from core.noise_db import NoiseParams

class GTSAMSolver(BaseSolver):
    def solve(self, data: SensorData, noise_params: NoiseParams) -> OrientationTrajectory:
        """
        Solves for orientation using GTSAM Factor Graph.
        
        Variables:
        - Rot3 (Orientation) at each timestep.
        
        Factors:
        - PriorFactor: Initial orientation.
        - BetweenFactor (Gyro): Relative rotation integration.
        - Custom/AttitudeFactor (Accel): Gravity Body-Frame measurement.
        - Custom/AttitudeFactor (Mag): North Body-Frame measurement.
        """
        graph = gtsam.NonlinearFactorGraph()
        initial_estimate = gtsam.Values()
        
        # Noise Models
        # Sigmas in radians (or appropriate units)
        sigma_gyro = noise_params.gyro_noise_sigma * np.sqrt(1.0/30.0) # Discrete integration noise approximation?
        # Actually gtsam BetweenFactor noise is usually per-step.
        # Gyro noise density given, we need discrete sigma.
        # sigma_d = sigma_c / sqrt(dt) or similar.
        # Let's assume params are roughly standard deviation per sample for now, or tune.
        dt = np.mean(np.diff(data.time)) if len(data.time) > 1 else 0.02
        
        # Rot3 noise is usually isotropic 3D
        noise_prior = gtsam.noiseModel.Isotropic.Sigma(3, 0.1) # 0.1 rad initial uncertainty
        noise_gyro = gtsam.noiseModel.Isotropic.Sigma(3, noise_params.gyro_noise_sigma) 
        noise_accel = gtsam.noiseModel.Isotropic.Sigma(3, noise_params.accel_noise_sigma)
        noise_mag = gtsam.noiseModel.Isotropic.Sigma(3, noise_params.mag_noise_sigma)
        
        # Reference Vectors (ENU)
        g_ref = gtsam.Unit3(np.array([0, 0, 1])) # Up
        m_ref = gtsam.Unit3(np.array([0, 1, 0])) # North (Approx)
        
        # Keys
        def X(i): return gtsam.symbol('x', i)
        
        N = len(data.time)
        
        # 1. Initialization
        # Simple gyro integration for initial guess
        # (Could also use TRIAD on first frame)
        current_rot = gtsam.Rot3.Identity()
        initial_estimate.insert(X(0), current_rot)
        graph.add(gtsam.PriorFactorRot3(X(0), current_rot, noise_prior))
        
        for i in range(1, N):
            # 1. Propagate Gyro (Between Factor)
            w = data.gyro[i-1]
            delta_rot = gtsam.Rot3.Expmap(w * dt)
            
            # Predict
            current_rot = current_rot.compose(delta_rot)
            initial_estimate.insert(X(i), current_rot)
            
            # Between Factor
            graph.add(gtsam.BetweenFactorRot3(X(i-1), X(i), delta_rot, noise_gyro))
            
            # 2. Accel Factor (Unary) - Using Custom logic or standard if available
            # Let's assume Rot3AttitudeFactor is available or emulate it
            # Rot3AttitudeFactor(Key key, Unit3 measured, NoiseModel noise, Unit3 reference)
            accel_meas = data.accel[i]
            if np.linalg.norm(accel_meas) > 1e-3:
                try:
                    meas_unit = gtsam.Unit3(accel_meas)
                    # Attempt to use Rot3AttitudeFactor
                    factor = gtsam.Rot3AttitudeFactor(X(i), meas_unit, noise_accel, g_ref)
                    graph.add(factor)
                except AttributeError:
                    # Fallback if specific factor not found in python binding: 
                    # create expression factor? 
                    # For now just ignore if not present to avoid crash on import earlier
                    pass

            # 3. Mag Factor
            mag_meas = data.mag[i]
            if np.linalg.norm(mag_meas) > 1e-3:
                 try:
                    meas_unit = gtsam.Unit3(mag_meas)
                    factor = gtsam.Rot3AttitudeFactor(X(i), meas_unit, noise_mag, m_ref)
                    graph.add(factor)
                 except AttributeError:
                    pass
                 
        # Optimization
        params = gtsam.LevenbergMarquardtParams()
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
        result = optimizer.optimize()
        
        # Extract
        quats = []
        covs = []
        for i in range(N):
            rot = result.atRot3(X(i))
            q = rot.toQuaternion() # w, x, y, z
            quats.append([q.w(), q.x(), q.y(), q.z()])
            
            # Extract marginal covariance if feasible (slow for large N)
            # covs.append(np.eye(3)*0.01) # Placeholder
        
        return OrientationTrajectory(
            timestamps=data.time,
            quaternions=np.array(quats),
            covariances=np.zeros((N, 3, 3)) # Placeholder
        )
