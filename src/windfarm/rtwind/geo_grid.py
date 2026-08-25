"""WGS84 ↔ local planning grid (same convention as data_ingest.crop_srtm_to_grid).

Grid axes:
- x increases east
- y increases south (row 0 is northern edge)
- origin (center_lat, center_lon) is the geometric center of the grid
"""

from __future__ import annotations

import math
from dataclasses import dataclass


METERS_PER_DEG_LAT = 111_320.0


@dataclass(slots=True)
class GeoGrid:
    center_lat: float
    center_lon: float
    width: int
    height: int
    resolution_m: float

    @property
    def meters_per_deg_lon(self) -> float:
        return METERS_PER_DEG_LAT * math.cos(math.radians(self.center_lat))

    @property
    def half_w_m(self) -> float:
        return 0.5 * (self.width - 1) * self.resolution_m

    @property
    def half_h_m(self) -> float:
        return 0.5 * (self.height - 1) * self.resolution_m

    def bounds(self) -> dict[str, float]:
        """Geographic bounds of cell centers (south, west, north, east)."""
        mlat = METERS_PER_DEG_LAT
        mlon = self.meters_per_deg_lon
        return {
            "south": self.center_lat - self.half_h_m / mlat,
            "north": self.center_lat + self.half_h_m / mlat,
            "west": self.center_lon - self.half_w_m / mlon,
            "east": self.center_lon + self.half_w_m / mlon,
        }

    def latlon_to_xy(self, lat: float, lon: float) -> tuple[float, float]:
        """Continuous grid coords (x east, y south)."""
        east_m = (lon - self.center_lon) * self.meters_per_deg_lon
        north_m = (lat - self.center_lat) * METERS_PER_DEG_LAT
        x = east_m / self.resolution_m + 0.5 * (self.width - 1)
        y = -north_m / self.resolution_m + 0.5 * (self.height - 1)
        return x, y

    def xy_to_latlon(self, x: float, y: float) -> tuple[float, float]:
        east_m = (x - 0.5 * (self.width - 1)) * self.resolution_m
        north_m = (0.5 * (self.height - 1) - y) * self.resolution_m
        lat = self.center_lat + north_m / METERS_PER_DEG_LAT
        lon = self.center_lon + east_m / self.meters_per_deg_lon
        return lat, lon

    def latlon_to_cell(self, lat: float, lon: float) -> tuple[int, int] | None:
        x, y = self.latlon_to_xy(lat, lon)
        ix, iy = int(round(x)), int(round(y))
        if ix < 0 or iy < 0 or ix >= self.width or iy >= self.height:
            return None
        return ix, iy

    def contains_latlon(self, lat: float, lon: float, *, margin_cells: float = 0.0) -> bool:
        x, y = self.latlon_to_xy(lat, lon)
        return (
            -margin_cells <= x < self.width + margin_cells
            and -margin_cells <= y < self.height + margin_cells
        )
