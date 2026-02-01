
import json
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class NoiseParams:
    """
    Stores noise parameters for a specific phone model or condition.
    Variances are in relevant units squared (e.g., (m/s^2)^2, (rad/s)^2, (Tesla)^2 or normalized).
    """
    accel_noise_sigma: float  # std dev, m/s^2
    gyro_noise_sigma: float   # std dev, rad/s
    mag_noise_sigma: float    # std dev, microTesla or normalized units
    
    # Bias stability (random walk) - optional for simple models but good for factors
    accel_bias_sigma: float = 1e-4
    gyro_bias_sigma: float = 1e-5

class NoiseDatabase:
    """
    Manages noise parameters for different devices and environments (Indoor/Outdoor).
    """
    
    # Default parameters (based on typical smartphone IMU noise - e.g. BMI160, LSM6DSO)
    _DEFAULTS = {
        "generic": {
            "outdoor": NoiseParams(accel_noise_sigma=0.02, gyro_noise_sigma=0.005, mag_noise_sigma=0.5),
            "indoor":  NoiseParams(accel_noise_sigma=0.02, gyro_noise_sigma=0.005, mag_noise_sigma=50.0) # High uncertainty for indoor mag
        },
        "pixel_10": { # Example high quality phone
             "outdoor": NoiseParams(accel_noise_sigma=0.008, gyro_noise_sigma=0.002, mag_noise_sigma=0.3),
             "indoor":  NoiseParams(accel_noise_sigma=0.008, gyro_noise_sigma=0.002, mag_noise_sigma=20.0)
        }
    }

    def __init__(self):
        self.db: Dict[str, Dict[str, NoiseParams]] = self._DEFAULTS

    def get_params(self, device_model: str, is_indoor: bool) -> NoiseParams:
        """
        Retrieve noise parameters for a device.
        Falls back to 'generic' if model not found.
        """
        key = device_model.lower().replace(" ", "_")
        environment = "indoor" if is_indoor else "outdoor"
        
        # Normalize key or fuzzy match could go here
        if key not in self.db:
            # Try to find a partial match or default
            key = "generic"
            
        return self.db[key][environment]

# Global instance
noise_db = NoiseDatabase()
