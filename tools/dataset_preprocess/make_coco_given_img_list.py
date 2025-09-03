import os
import numpy as np
import geopandas as gpd
import pdb
import fiona
import shapely
from fiona.crs import from_epsg
import shutil
from tqdm import tqdm
import glob
import rasterio
import subprocess
from rasterio.transform import xy, rowcol
from planettools.utils import polygon_utils
import json

basemap_name = 'global_quartely_2023q2'
coco_dict = {
    'info': {'district': 'Planet Basemap', 'description': 'Basemap data from PlanetScope captured in the 2023 quarter 2', 'contributor': 'tanmlh'},
    'categories': [{'id': 100, 'name': 'building'}],
    'images': [],
    'annotations': [],
}

data_root = '/home/fahong/Datasets/ai4eo3/planet_data_download/basemap/dataset_2023q2_v3/train'
tif_pattern = f'{data_root}/img/*.tif'
out_dir = f'{data_root}/coco'
out_path = f'{data_root}/coco/coco_no_ann.json'

tif_paths = glob.glob(tif_pattern.replace('{}', '*'))
tif_names = [x.split('/')[-1] for x in tif_paths]

img_id = 0

for tif_path in tqdm(tif_paths):
    tif_name = tif_path.split('/')[-1].split('.')[0]

    H, W = rasterio.open(tif_path).shape
    rel_path = os.path.relpath(tif_path, data_root)
    tif_info = {'id': img_id, 'file_name': rel_path, 'width': W, 'height': H}
    coco_dict['images'].append(tif_info)
    coco_dict['annotations'] = []

    img_id += 1

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

with open(out_path, 'w') as f_json:
    json.dump(coco_dict, f_json)
