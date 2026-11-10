"""Charging sessions on a half-hourly grid (SGT)."""
from dataclasses import dataclass

DT_H = 0.5  # length of one market period in hours


@dataclass(frozen=True)
class Session:
    """One vehicle plugged in from `arrival` (inclusive) to `departure` (exclusive) periods."""

    vehicle_id: str
    arrival: int
    departure: int
    energy_kwh: float  # energy the driver needs before departure
    max_kw: float  # charger / on-board charger limit

    def __post_init__(self):
        if not 0 <= self.arrival < self.departure:
            raise ValueError(f"{self.vehicle_id}: need 0 <= arrival < departure")
        if self.energy_kwh < 0 or self.max_kw <= 0:
            raise ValueError(f"{self.vehicle_id}: energy must be >= 0 and max_kw > 0")

    @property
    def max_deliverable_kwh(self) -> float:
        return self.max_kw * (self.departure - self.arrival) * DT_H
