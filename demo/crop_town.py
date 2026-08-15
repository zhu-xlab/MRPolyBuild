"""Crop the demo tile (Saint-Martin-de-Crau, France) from a PlanetScope scene.

The input scene is the public ESA PlanetScope Visual sample:
https://earth.esa.int/eogateway/missions/planetscope/sample-data
"""

import sys

import rasterio
from rasterio.windows import Window


CENTER = (646320, 4833156)  # Saint-Martin-de-Crau center (UTM 31N)
SIZE = 2048


def main():
    src_path = sys.argv[1] if len(sys.argv) > 1 else (
        'demo/raw/PSScene_Visual/20230709_093825_63_24c5/visual/'
        '20230709_093825_63_24c5_3B_Visual.tif')
    out = sys.argv[2] if len(sys.argv) > 2 else (
        'demo/data/saint_martin_de_crau_town.tif')

    with rasterio.open(src_path) as d:
        bounds = d.bounds
        col = int((CENTER[0] - bounds.left) / 3 - SIZE / 2)
        row = int((bounds.top - CENTER[1]) / 3 - SIZE / 2)
        col = max(0, min(col, d.width - SIZE))
        row = max(0, min(row, d.height - SIZE))
        win = Window(col, row, SIZE, SIZE)
        data = d.read(window=win)
        t = d.window_transform(win)
        with rasterio.open(
                out, 'w', driver='GTiff', width=SIZE, height=SIZE,
                count=3, dtype=data.dtype, crs=d.crs, transform=t,
                compress='deflate') as o:
            o.write(data[:3])
    print('wrote', out)


if __name__ == '__main__':
    main()
