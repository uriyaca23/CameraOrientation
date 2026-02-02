
from abc import ABC, abstractmethod
from typing import NamedTuple
import numpy as np
from core.data_loader import SensorData
from core.noise_db import NoiseParams

class OrientationTrajectory(NamedTuple):
    timestamps: np.ndarray # T
    quaternions: np.ndarray # T x 4 (w, x, y, z)
    covariances: np.ndarray # T x 3 x 3 (Roll, Pitch, Yaw covariance) - approximate

class BaseSolver(ABC):
    @abstractmethod
    def solve(self, data: SensorData, noise_params: NoiseParams) -> OrientationTrajectory:
        pass
