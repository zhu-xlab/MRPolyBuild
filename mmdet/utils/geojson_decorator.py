# Copyright (c) MRPolyBuild authors. All rights reserved.
"""Self-contained GeoJSON reader used by the Planet evaluation scripts.

This is a trimmed, dependency-light reimplementation of the utility that was
previously shipped inside the private ``planettool`` repository.  Only the
parts needed by ``tools/eval_planet_metrics_by_geojson_list.py`` are kept so
that the evaluation script can run from a plain MRPolyBuild checkout.
"""

import math
import numpy as np
import shapely
from shapely.geometry import shape, mapping
from shapely import ops
from affine import Affine
import geopandas as gpd


def compute_overlap_matrix(boxes1, boxes2):
    """Return a boolean (N, M) matrix: True where boxes overlap.

    Args:
        boxes1: (N, 4) array of (minx, miny, maxx, maxy).
        boxes2: (M, 4) array of (minx, miny, maxx, maxy).
    """
    boxes1 = boxes1[:, np.newaxis, :]
    boxes2 = boxes2[np.newaxis, :, :]

    x_overlap = np.logical_not(
        (boxes1[..., 2] < boxes2[..., 0]) | (boxes1[..., 0] > boxes2[..., 2])
    )
    y_overlap = np.logical_not(
        (boxes1[..., 3] < boxes2[..., 1]) | (boxes1[..., 1] > boxes2[..., 3])
    )
    return np.logical_and(x_overlap, y_overlap)


class GeoJSONDecorator:
    """Minimal GeoJSON loader exposing the API used by the eval scripts."""

    def __init__(self, geojson_path, crs='EPSG:4326'):
        self.geojson_path = geojson_path
        self.crs = crs
        self.loaded = False
        self.shp_loaded = False
        self.features = []
        self.geom_shps = []

    # ------------------------------------------------------------------ #
    # loading
    # ------------------------------------------------------------------ #
    def load_features(self):
        if self.loaded:
            return self.features

        self.features = []
        gdf = gpd.read_file(self.geojson_path)
        for _, row in gdf.iterrows():
            geom = row.geometry
            props = {
                k: (None if v is None else v.item() if hasattr(v, 'item')
                    else v)
                for k, v in row.items() if k != 'geometry'
            }
            self.features.append({
                'type': 'Feature',
                'geometry': mapping(geom),
                'properties': props,
            })

        self.loaded = True
        return self.features

    @staticmethod
    def json2coco(geom_json):
        """Flatten a polygon ring list into COCO-style flat coordinate lists."""
        if geom_json['type'] == 'Polygon':
            coords_list = geom_json['coordinates']
            new_coords_list = []
            for coords in coords_list:
                new_coords_list.append(np.array(coords).reshape(-1).tolist())
            return new_coords_list
        if geom_json['type'] == 'MultiPolygon':
            # Should not happen after to_coco_anns splits multipolygons.
            raise ValueError('MultiPolygon should be split before json2coco')
        return None

    def load_geom_shps(self, reload=True, fix_invalid=True,
                       unfold_multi_polygon=True):
        self.load_features()
        if self.shp_loaded and not reload:
            return self.geom_shps

        self.geom_shps = []
        for feature in self.features:
            geom_json = feature['geometry']
            if geom_json is None:
                continue
            cur_shp = shape(geom_json)
            if fix_invalid and not cur_shp.is_valid:
                cur_shp = cur_shp.buffer(0.)

            if cur_shp.geom_type == 'Polygon':
                self.geom_shps.append(cur_shp)
            elif cur_shp.geom_type == 'MultiPolygon':
                if unfold_multi_polygon:
                    self.geom_shps.extend(list(cur_shp.geoms))
                else:
                    self.geom_shps.append(cur_shp)
            else:
                continue

        self.geom_shps = [x for x in self.geom_shps if not x.is_empty]
        self.shp_loaded = True
        return self.geom_shps

    def get_num_features(self):
        self.load_features()
        return len(self.features)

    def get_bounds(self, reload=False):
        self.load_geom_shps(reload)
        return np.array([x.bounds for x in self.geom_shps])

    def filter_invalid(self):
        self.load_geom_shps(reload=True)
        valid_idxes = [
            i for i, shp in enumerate(self.geom_shps) if shp.is_valid
        ]
        self.features = [self.features[x] for x in valid_idxes]

    # ------------------------------------------------------------------ #
    # coordinate conversion
    # ------------------------------------------------------------------ #
    @staticmethod
    def geojson_to_pixel_3857(polygons_json, bound, gsd=None, img_size=None):
        """Convert GeoJSON polygons to pixel coordinates.

        Args:
            polygons_json: list of GeoJSON Feature dicts.
            bound: (min_x, min_y, max_x, max_y).
            gsd: ground sampling distance in meters per pixel.
            img_size: (height, width) if GSD is not available.

        Returns:
            (transformed_features, (image_width, image_height)).
        """
        assert gsd is not None or img_size is not None, \
            'Must provide either gsd or img_size'

        min_x, min_y, max_x, max_y = bound

        if img_size is None:
            image_width = int((max_x - min_x) / gsd)
            image_height = int((max_y - min_y) / gsd)
            transform = Affine.translation(-min_x, max_y) * \
                Affine.scale(1 / gsd, -1 / gsd)
        else:
            image_height, image_width = img_size
            gsd_x = (max_x - min_x) / image_width
            gsd_y = (max_y - min_y) / image_height
            transform = Affine.translation(-min_x, max_y) * \
                Affine.scale(gsd_x, -gsd_y)

        def transform_coords(ring):
            return [list(map(int, transform * (x, y))) for x, y in ring]

        def transform_polygon(poly):
            exterior = transform_coords(poly.exterior.coords)
            interiors = [
                transform_coords(ring.coords) for ring in poly.interiors
            ]
            return [exterior] + interiors

        transformed_polygons = []
        for feature in polygons_json:
            geom = feature.get('geometry', {})
            properties = feature.get('properties', {})
            if not geom:
                continue

            shapely_geom = shape(geom)
            if isinstance(shapely_geom, shapely.geometry.Polygon):
                pixel_coords = transform_polygon(shapely_geom)
                geom_type = 'Polygon'
            elif isinstance(shapely_geom, shapely.geometry.MultiPolygon):
                pixel_coords = [
                    transform_polygon(p) for p in shapely_geom.geoms
                ]
                geom_type = 'MultiPolygon'
            else:
                continue

            transformed_polygons.append({
                'type': 'Feature',
                'geometry': {
                    'type': geom_type,
                    'coordinates': pixel_coords,
                },
                'properties': properties,
            })

        return transformed_polygons, (image_width, image_height)

    def to_pixel_coords(self, bound, gsd=None, img_size=None):
        self.load_features()
        new_features, (_, _) = self.geojson_to_pixel_3857(
            self.features, bound, gsd, img_size)
        self.features = new_features
        return new_features

    # ------------------------------------------------------------------ #
    # COCO conversion
    # ------------------------------------------------------------------ #
    def to_coco_anns(self, geom_type='coco', fix_invalid=True):
        self.load_features()

        coco_dict = {'masks': [], 'scores': []}
        for feature in self.features:
            geom_json = feature.get('geometry', {})
            properties = feature.get('properties', {})

            if not geom_json:
                continue

            shp = shape(geom_json)
            if fix_invalid and not shp.is_valid:
                shp = shp.buffer(0.0)

            if shp.is_empty:
                continue

            if isinstance(shp, shapely.geometry.Polygon):
                polygons = [shp]
            elif isinstance(shp, shapely.geometry.MultiPolygon):
                polygons = list(shp.geoms)
            else:
                continue

            if 'scores' in properties:
                score = properties['scores']
            elif 'confidence' in properties:
                score = max(properties['confidence'], 0.0)
            else:
                score = 1.0

            for poly in polygons:
                if poly.is_empty:
                    continue
                poly_json = mapping(poly)
                geom_coco = (
                    self.json2coco(poly_json)
                    if geom_type == 'coco' else poly_json
                )
                coco_dict['masks'].append(geom_coco)
                coco_dict['scores'].append(score)

        return coco_dict

    # ------------------------------------------------------------------ #
    # IoU
    # ------------------------------------------------------------------ #
    @staticmethod
    def create_grid(x_min, y_min, x_max, y_max, h, w):
        W = x_max - x_min
        H = y_max - y_min
        n_x = math.ceil(W / w)
        n_y = math.ceil(H / h)
        w_adj = W / n_x
        h_adj = H / n_y

        grids = []
        for i in range(n_x):
            for j in range(n_y):
                grids.append((
                    x_min + i * w_adj,
                    y_min + j * h_adj,
                    x_min + (i + 1) * w_adj,
                    y_min + (j + 1) * h_adj,
                ))
        return np.array(grids)

    @staticmethod
    def bbox_to_polygon(bbox):
        min_x, min_y, max_x, max_y = bbox
        return shapely.geometry.Polygon([
            (min_x, min_y),
            (min_x, max_y),
            (max_x, max_y),
            (max_x, min_y),
            (min_x, min_y),
        ])

    def iou(self, trg_geojson, grid_size=(1024, 1024)):
        A_shps = self.load_geom_shps(fix_invalid=True)
        B_shps = trg_geojson.load_geom_shps(fix_invalid=True)

        if len(A_shps) == 0 or len(B_shps) == 0:
            A_union = ops.unary_union(A_shps)
            B_union = ops.unary_union(B_shps)
            return 0, (0, A_union.area + B_union.area)

        bounds_A = self.get_bounds()
        bounds_B = trg_geojson.get_bounds()
        bounds_min = np.concatenate([bounds_A, bounds_B]).min(axis=0)[:2]
        bounds_max = np.concatenate([bounds_A, bounds_B]).max(axis=0)[2:]
        bounds = np.concatenate([bounds_min, bounds_max])

        grids = self.create_grid(
            bounds[0], bounds[1], bounds[2], bounds[3],
            grid_size[0], grid_size[1])
        overlap_mat_A = compute_overlap_matrix(grids, bounds_A)
        overlap_mat_B = compute_overlap_matrix(grids, bounds_B)

        intersect_area = 0
        union_area = 0
        for i in range(len(grids)):
            cur_A_idxes = overlap_mat_A[i].nonzero()[0]
            cur_B_idxes = overlap_mat_B[i].nonzero()[0]

            cur_A_shps = [A_shps[x] for x in cur_A_idxes]
            cur_B_shps = [B_shps[x] for x in cur_B_idxes]

            bound_shp = self.bbox_to_polygon(grids[i])

            cur_A_union = ops.unary_union(cur_A_shps)
            cur_B_union = ops.unary_union(cur_B_shps)

            crop_A = cur_A_union.intersection(bound_shp)
            crop_B = cur_B_union.intersection(bound_shp)

            if isinstance(crop_A, shapely.geometry.collection.GeometryCollection):
                crop_A = shapely.geometry.collection.GeometryCollection(
                    [x for x in crop_A.geoms if x.area > 0])

            if isinstance(crop_B, shapely.geometry.collection.GeometryCollection):
                crop_B = shapely.geometry.collection.GeometryCollection(
                    [x for x in crop_B.geoms if x.area > 0])

            cur_intersect_area = crop_A.intersection(crop_B).area
            cur_union_area = crop_A.union(crop_B).area

            intersect_area += cur_intersect_area
            union_area += cur_union_area

        if union_area <= 0:
            iou = 0
        else:
            iou = intersect_area / union_area

        return iou, (intersect_area, union_area)
