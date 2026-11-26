"""EpiCare Branch 1: seizure risk forecasting from wearable data (My Seizure Gauge)."""
from .config import Config
from .dataset import PatientData
from .forecaster import PatientForecaster

__all__ = ["Config", "PatientData", "PatientForecaster"]
__version__ = "0.1.0"
