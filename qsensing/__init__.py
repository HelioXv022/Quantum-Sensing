"""Toy model of collective AC sensing with a quantum sensor array."""

from .model import (
    SensorCase,
    analyze_case,
    conventional_time,
    detection_time,
    effective_signal,
    generate_cases,
    qss_time,
    sensor_grid,
    success_probability,
    trace_distance,
)

__all__ = [
    "SensorCase",
    "analyze_case",
    "conventional_time",
    "detection_time",
    "effective_signal",
    "generate_cases",
    "qss_time",
    "sensor_grid",
    "success_probability",
    "trace_distance",
]
