"""Reproducible vehicular task-offloading research simulator."""

from .config import SimulationConfig
from .domain import OffloadAction, OffloadResult, Task, VehicleState

__all__ = [
    "OffloadAction",
    "OffloadResult",
    "SimulationConfig",
    "Task",
    "VehicleState",
]

__version__ = "0.1.0"
