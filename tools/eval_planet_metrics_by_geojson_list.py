import os
import pdb
import glob
import json
import argparse
import shapely
import numpy as np
from tqdm import tqdm
from mmdet.structures.mask import PolygonMasks
from mmdet.evaluation.functional import eval_poly_map
import mmdet.utils.tanmlh_polygon_utils as polygon_utils
from mmdet.utils.geojson_decorator import GeoJSONDecorator
import geopandas as gpd
from prettytable import PrettyTable


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate paired Planet GeoJSON predictions.")
    parser.add_argument(
        "--product-name",
        action="append",
        dest="product_names",
        help="Product/output name. Can be passed multiple times.",
    )
    parser.add_argument(
        "--pred-geojson-pattern",
        default="/home/Datasets/so2sat03/Global3D_v2/MRPolyBuild/data/test_6k/{product_name}/geojson/*.geojson",
        help="Prediction GeoJSON glob. May contain {product_name}.",
    )
    parser.add_argument(
        "--gt-geojson-pattern",
        default="/home/Datasets/so2sat03/Global3D_v2/MRPolyBuild/data/test_6k/osm/geojson/*.geojson",
        help="Ground-truth GeoJSON glob.",
    )
    parser.add_argument(
        "--out-base-path",
        default="/home/Datasets/so2sat03/Global3D_v2/MRPolyBuild/data/test_6k/metrics/{product_name}",
        help="Metric output directory. May contain {product_name}.",
    )
    parser.add_argument(
        "--match-mode",
        default="gt_driven",
        choices=["common", "gt_driven"],
        help="Pair predictions and GT by common files or by GT files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute metrics even if cached JSON already exists.",
    )
    return parser.parse_args()


args = parse_args()

# product_name = 'inf_ins_v2_8x_planet-test-5k'
# product_name = 'inf_ins_v2_8x_thre05_planet-test-5k'
# product_name = 'inf_gcp_ins_v2_8x_late-stop_planet-test-5k'
# product_name = 'gcp_right-v2_50e_8x/'

# product_name = 'clsm'
# product_name = 'gcp'
# product_name = 'gcp_seg_pbc'
# product_name = 'gcp_diff01'
# product_name = 'gcp_seg_ct'
# product_name = 'gcp_no-rht'
# product_name = 'gcp_diff003'
# product_name = 'gcp_diff02'
# product_name = 'gcp_diff-inf'
# product_name = 'gcp_seg_pbc_late'
# product_name = 'gcp_late-stop'
# product_name = 'gcp_no-pfs'
# product_name = 'microsoft'
# product_name = 'google'

# product_name = 'gcp_no-pfs'
# product_name = 'gcp_ct_late-stop'
# product_name = 'gcp_pbc_late-stop'

# product_name = 'google25d/gcp_pbc_thr005'
# product_name = 'google25d/gcp_ct'
# product_name = 'google25d/gcp'

# product_names = ['google25d/gcp', 'google25d/gcp_pbc_thr005', 'google25d/gcp_ct']
# product_names = ['gcp_fixed', 'gcp_seg_pbc', 'gcp_seg_ct']
# product_names = ['gcp_early_pbc-thr005', 'gcp_seg_ct']
# product_names = ['gcp_early']
# product_names = ['gcp_early-stop_diff010', 'gcp_early-stop_diff020', 'gcp_early-stop_diff003']
# product_names = ['gcp_diff003', 'gcp_diff01', 'gcp_diff02']
# product_names = ['gcp_diff003', 'gcp_diff01', 'gcp_diff02']
# product_names = ['gcp_no-pfs', 'gcp_no-rht']
# product_names = ['gcp_ct-thr05', 'gcp_ct-thr06']
# product_names = ['gcp_4x']
# product_names = ['gcp_early-stop_diff-inf']
# product_names = ['microsoft']
# product_names = ['google']
# product_names = ['gcp_early-stop_no-pfs', 'gcp_early_no-rht']
product_names = args.product_names or ['hisup_hrnet48_aug8x']

for product_name in product_names:

# ---------------- 閰嶇疆璺緞 ---------------- #
# pred_geojson_pattern = f'/home/fahong/Datasets/ai4eo3/planet_data_download/basemap/dataset_2023q2_v3/test_16k/{product_name}/geojson/*.geojson'
# gt_geojson_pattern = '/home/fahong/Datasets/ai4eo3/planet_data_download/basemap/dataset_2023q2_v3/test_16k/osm/geojson_with_continent/*.geojson'
# out_base_path = f'/home/fahong/Datasets/ai4eo3/planet_data_download/basemap/dataset_2023q2_v3/test_16k/metrics/{product_name}'

# pred_geojson_pattern = f'/home/fahong/Datasets/ai4eo3/planet_data_download/basemap/dataset_2023q2_v3/test/{product_name}/geojson/*.geojson'
# gt_geojson_pattern = '/home/fahong/Datasets/ai4eo3/planet_data_download/basemap/dataset_2023q2_v3/test/osm/geojson_with_continent/*.geojson'
# out_base_path = f'/home/fahong/Datasets/ai4eo3/planet_data_download/basemap/dataset_2023q2_v3/test/metrics/{product_name}'

# pred_geojson_pattern = f'/home/fahong/Datasets/ai4eo3/planet_data_download/basemap/dataset_2023q2_v3/test_40k/{product_name}/geojson/*.geojson'
# gt_geojson_pattern = '/home/fahong/Datasets/ai4eo3/planet_data_download/basemap/dataset_2023q2_v3/test_40k/osm/geojson_with_continent/*.geojson'
# out_base_path = f'/home/fahong/Datasets/ai4eo3/planet_data_download/basemap/dataset_2023q2_v3/test_40k/metrics/{product_name}'

    pred_geojson_pattern = args.pred_geojson_pattern.format(product_name=product_name)
    gt_geojson_pattern = args.gt_geojson_pattern.format(product_name=product_name)
    out_base_path = args.out_base_path.format(product_name=product_name)

    os.makedirs(out_base_path, exist_ok=True)

# ---------------- 鎺у埗閰嶅妯″紡 ---------------- #
    match_mode = args.match_mode  # option: "common" or "gt_driven"

# ---------------- define filters ---------------- #
    filters = [
            # {"name": "Africa", "continent": "Africa"},
            # {"name": "Europe", "continent": "Europe"},
            # {"name": "Asia", "continent": "Asia"},
            # {"name": "South_America", "continent": "South America"},
            # {"name": "North_America", "continent": "North America"},
            # {"name": "Oceania", "continent": "Oceania"},
            # {"name": "China", "countries": ['China']},
        # {"name": "East_Asia", "countries": ['China', 'Mongolia', 'Japan', 'Taiwan', 'South Korea', 'North Korea']},
        {"name": "Globe"}
    ]

    metrics = ['iou', 'AP30', 'AP50', 'AR30', 'AR50', 'MTA', 'PoLiS', 'N-ratio', 'MSE_BN']

# ---------------- 璇诲彇鎵€鏈夋枃浠跺苟閰嶅 ---------------- #
    pred_geojson_paths = sorted(glob.glob(pred_geojson_pattern))
    gt_geojson_paths = sorted(glob.glob(gt_geojson_pattern))

    pred_dict = {os.path.basename(p): p for p in pred_geojson_paths}
    gt_dict = {os.path.basename(p): p for p in gt_geojson_paths}

    if match_mode == "common":
        common_files = sorted(list(set(pred_dict.keys()) & set(gt_dict.keys())))
    elif match_mode == "gt_driven":
        common_files = sorted(list(gt_dict.keys()))
    else:
        raise ValueError("Unsupported match_mode setting!")

    print(f"Found {len(common_files)} matched geojson pairs (mode: {match_mode})")


# ---------------- 棰勮鍙栨瘡涓枃浠剁殑region灞炴€?---------------- #
    file_region_info = {}
    for file_idx, filename in enumerate(tqdm(common_files, desc="Pre-scanning regions")):
        gt_path = gt_dict[filename]
        gdf = gpd.read_file(gt_path)
        if len(gdf) == 0:
            continue

        first_row = gdf.iloc[0]
        file_region_info[filename] = {
            "Continent": first_row.get('Continent', 'Unknown'),
            "Country": first_row.get('Sovereign', 'Unknown')
        }

# ---------------- summary table ---------------- #
    summary_results = []

# ---------------- 閫愪釜filter鎵ц璇勪及 ---------------- #
    for filter_item in filters:
        print(f"\nRunning evaluation for filter: {filter_item['name']}")

        out_path = os.path.join(out_base_path, f"overall_{filter_item['name']}.json")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"Cached result found at {out_path}, loading...")
            with open(out_path, 'r') as f:
                result_dict = json.load(f)
        else:
            selected_files = []
            for filename, region in file_region_info.items():
                continent_match = filter_item.get("continent") == region.get('Continent') if "continent" in filter_item else True
                country_match = region.get('Country') in filter_item.get("countries", [region.get('Country')])
                if continent_match and country_match:
                    selected_files.append(filename)

            print(f"Selected {len(selected_files)} files for this filter.")

            all_det_polys = []
            all_det_scores = []
            all_annotations = []
            total_intersection = 0.0
            total_union = 0.0
            total_pred_area = 0
            total_gt_area = 0
            gsd = 1.0
            num_gt_files = 0
            num_pred_files = 0
            all_mtas = [] if 'MTA' in metrics else None
            all_polis = [] if 'PoLiS' in metrics else None

            for filename in tqdm(selected_files, desc="Processing"):
                gt_geojson = GeoJSONDecorator(gt_dict[filename], crs='EPSG:3857')
                pred_geojson = GeoJSONDecorator(pred_dict[filename], crs='EPSG:3857') if filename in pred_dict else None

                pred_polys = pred_geojson.load_geom_shps(fix_invalid=True) if pred_geojson else []
                gt_polys = gt_geojson.load_geom_shps(fix_invalid=True) if gt_geojson else []

                if gt_geojson.get_num_features() == 0:
                    continue
                    pred_coco_anns = [pred_geojson.to_coco_anns(geom_type='json')] if pred_geojson else [[]]
                    gt_coco_ann = {'masks': PolygonMasks([], 0, 0)}
                    all_det_polys.append([x['masks'] for x in pred_coco_anns] if pred_geojson else [[]])
                    all_det_scores.append(np.array([x.get('scores', 1.0) for x in pred_coco_anns]) if pred_geojson else np.array([[]]))

                    all_annotations.append(gt_coco_ann)
                    if pred_geojson:
                        total_union += sum([x.area for x in pred_polys])
                        total_pred_area += sum([x.area for x in pred_polys])
                    continue

                num_gt_files += 1

                bound = np.concatenate([gt_geojson.get_bounds()[:, :2].min(axis=0), gt_geojson.get_bounds()[:, 2:].max(axis=0)])
                gt_geojson.to_pixel_coords(bound, gsd)
                # gt_geojson.filter_invalid()
                gt_coco_anns = [gt_geojson.to_coco_anns(geom_type='json')]
                for ann in gt_coco_anns:
                    ann['masks'] = PolygonMasks.from_json(ann['masks'], 0, 0)

                if pred_geojson:
                    num_pred_files += 1
                    pred_geojson.to_pixel_coords(bound, gsd)
                    # pred_geojson.filter_invalid()
                    pred_coco_anns = [pred_geojson.to_coco_anns(geom_type='json')]
                    pred_masks = [x['masks'] for x in pred_coco_anns]
                    pred_scores = np.array([x.get('scores', 1.0) for x in pred_coco_anns])

                    # pred_masks = [GeoJSONDecorator.json2coco(shapely.geometry.mapping(x)) for x in pred_polys]
                else:
                    pred_masks = [[]]
                    pred_scores = np.array([[]])

                all_det_polys.append(pred_masks)
                all_det_scores.append(pred_scores)
                all_annotations.append(gt_coco_anns[0])

                if 'MTA' in metrics:
                    mtas = polygon_utils.compute_polygon_contour_measures(pred_polys, gt_polys, sampling_spacing=2.0, min_precision=0.5, max_stretch=2)
                    all_mtas.extend([x for x in mtas if x is not None and not np.isnan(x)])

                if 'PoLiS' in metrics:
                    polis = polygon_utils.compute_polys_measure(pred_polys, gt_polys)
                    all_polis.extend(polis)

                if pred_geojson:
                    _, (inter_area, union_area) = pred_geojson.iou(gt_geojson)
                    total_intersection += inter_area
                    total_union += union_area
                    total_pred_area += sum([x.area for x in pred_polys])
                    total_gt_area += sum([x.area for x in gt_polys])

                else:
                    total_union += sum([x.area for x in gt_polys])
                    total_gt_area += sum([x.area for x in gt_polys])

            if len(all_det_polys) > 0:
                _, ap_result_30 = eval_poly_map(all_det_polys, all_det_scores, all_annotations, iou_thr=0.3)
                _, ap_result_50 = eval_poly_map(all_det_polys, all_det_scores, all_annotations, iou_thr=0.5)
            else:
                ap_result_30 = [{'num_gts': 0,'num_dets': 0,'recall': [],'precision': [],'ap': 0}]
                ap_result_50 = [{'num_gts': 0,'num_dets': 0,'recall': [],'precision': [],'ap': 0}]

            overall_iou = total_intersection / total_union if total_union > 0 else 0.0
            precision = total_intersection / (total_pred_area + 1e-8)
            recall = total_intersection / (total_gt_area + 1e-8)
            N_ratio = float(ap_result_50[0]['num_dets'] / (ap_result_50[0]['num_gts'] + 1e-8))
            completeness = num_pred_files / (num_gt_files + 1e-8)

            result_dict = {
                "matched_files": len(selected_files),
                "AP50": float(ap_result_50[0]['ap']),
                "AP30": float(ap_result_30[0]['ap']),
                "AR50": float(ap_result_50[0]['recall'][-1]) if len(ap_result_50[0]['recall']) > 0 else 0.0,
                "AR30": float(ap_result_30[0]['recall'][-1]) if len(ap_result_30[0]['recall']) > 0 else 0.0,
                "N_ratio": N_ratio,
                'MTA': float(np.mean(all_mtas)) if all_mtas else -1,
                'PoLiS': float(np.mean(all_polis)) if all_polis else -1,
                "IoU": overall_iou,
                "Precision": precision,
                "Recall": recall,
                "Completeness": completeness
            }

            with open(out_path, 'w') as f:
                json.dump(result_dict, f, indent=2)
            print(f"Saved result to {out_path}")

        flat_result = {"name": filter_item['name']}
        flat_result.update(result_dict)
        summary_results.append(flat_result)

# ---------------- final table ---------------- #
    table = PrettyTable()
    field_names = list(summary_results[0].keys())
    field_names = [x for x in field_names if x != 'Overall_IoU']
    table.field_names = field_names

    for res in summary_results:
        row = []
        for k in field_names:
            v = res[k]
            if isinstance(v, float):
                if k == 'Overall_IoU': # skip it since it's renamed as IoU
                    continue
                if k == 'N_ratio':
                    exp = f"{v:.2f}"
                elif k == 'MTA' or k == 'PoLiS':
                    exp = f"{v:.1f}"
                else:
                    exp = f"{v*100:.1f}"

                row.append(exp)
            else:
                row.append(str(v))
        table.add_row(row)

    print("\n========== Summary Results ==========")
    print(table)

    # save table as text file
    with open(os.path.join(out_base_path, "summary_table.txt"), 'w') as f:
        f.write(str(table))

    with open(os.path.join(out_base_path, "summary_table.html"), 'w') as f:
        f.write(table.get_html_string())

