import rasterio
from pathlib import Path


FILE_NAME = "Singapore_DEM.tif"


class ElevationMap:
    def __init__(self, base_path: Path):
        filepath = base_path / FILE_NAME
        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} does not exist!")

        self.file = filepath

        # Open the geotiff file using rasterio, read and load the contents into memory
        with rasterio.open(self.file) as src:
            self.data = src.read(1)
            self.transform = src.transform

    def __call__(self, lat: float, lon: float) -> float:
        row, col = rasterio.transform.rowcol(self.transform, lon, lat)
        return self.data[row, col]


emap = ElevationMap(Path("geotiff"))
