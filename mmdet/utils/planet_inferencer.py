import queue
import threading
import concurrent.futures
import torch
import pdb
import numpy as np
from tqdm import tqdm
import os
import time
import mmdet.utils.tanmlh_polygon_utils as polygon_utils
import resource
import rasterio
from rasterio.transform import Affine


class InferencePipeline:
    def __init__(self, model, num_images, save_cfg, cpu_workers=None):
        """
        Main thread-driven inference pipeline with all GPU operations in main thread
        
        Args:
            model: Inference model
            num_images: Maximum number of images to process
            save_cfg: Result saving configuration
            cpu_workers: Number of CPU worker threads
        """
        self.model = model
        self.num_images = num_images
        self.save_cfg = save_cfg
        self.cpu_workers = cpu_workers or max(1, os.cpu_count() - 1)
        
        # Queues for task coordination
        self.gpu_task_queue = queue.Queue()    # GPU task queue
        
        # State tracking
        self.submitted_count = 0
        self.completed_count = 0
        self.active_tasks = 0  # Tracks active in-process tasks
        
        # Time statistics
        self.time_stats = {
            'total': 0.0,
            'data_preprocessor': 0.0,
            'predict_sem_seg': 0.0,
            'predict_mosaic_sem_seg': 0.0,
            'predict_seg2ins': 0.0,
            'predict_sample_segments': 0.0,
            'predict_gcp': 0.0,
            'predict_assemble_segments': 0.0,
            'predict_dp': 0.0
        }

    def run(self, data_loader):
        """
        Run the main thread-driven inference pipeline
        
        Args:
            data_loader: Data loader object
            
        Returns:
            Number of images processed
        """
        # Create output directory if saving enabled
        if self.save_cfg.get('save_results', False):
            out_dir = self.save_cfg['out_dir']
            flag_dir = os.path.join(out_dir, 'flag')
            out_geojson_dir = os.path.join(out_dir, 'geojson')
            out_tif_dir = os.path.join(out_dir, 'tif')

            os.makedirs(out_dir, exist_ok=True)
            os.makedirs(flag_dir, exist_ok=True)
            os.makedirs(out_geojson_dir, exist_ok=True)
            os.makedirs(out_tif_dir, exist_ok=True)

        
        # Create thread pool for CPU tasks

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.cpu_workers) as executor:
            self.executor = executor
        # with concurrent.futures.ProcessPoolExecutor(max_workers=self.cpu_workers) as executor:
        #     self.executor = executor
            
            # Start total time tracking
            total_start = time.perf_counter()

            dataset = data_loader.dataset

            with tqdm(total=self.num_images, desc='Processing images (0 active tasks)') as pbar:
                # Create progress bar

                # Process all images
                # for idx, data_batch in enumerate(data_loader):
                for idx in range(self.num_images):
                    data_info = dataset.get_data_info(idx)
                    file_name = data_info['img_path'].split('/')[-1].split('.')[0]
                    start_flag_path = os.path.join(flag_dir, f'{file_name}.start')
                    finish_flag_path = os.path.join(flag_dir, f'{file_name}.finish')

                    # file_name = data_batch['data_samples'][0].metainfo['img_path'].split('/')[-1].split('.')[0]
                    # start_flag_path = os.path.join(flag_dir, f'{file_name}.start')
                    # finish_flag_path = os.path.join(flag_dir, f'{file_name}.finish')

                    if os.path.exists(start_flag_path) or os.path.exists(finish_flag_path):
                        print(f'{file_name} is under processing or has already been processed, skip it.')
                        pbar.update(1)
                        continue

                    with open(start_flag_path, 'w') as _:
                        pass

                    data_batch = dataset[idx]
                    for key in data_batch.keys():
                        data_batch[key] = [data_batch[key]]

                    # Update progress description
                    pbar.set_description(f"Processing images ({self.active_tasks} active tasks)")
                    
                    # Process pending GPU tasks
                    self._process_pending_gpu_tasks(pbar)
                    
                    # Step 1: Data preprocessing
                    preprocess_start = time.perf_counter()
                    with torch.no_grad():
                        data = self.model.data_preprocessor(data_batch)
                        imgs = data['inputs']
                        batch_data_samples = data['data_samples']
                    self.time_stats['data_preprocessor'] += time.perf_counter() - preprocess_start
                    
                    # Step 2: Semantic segmentation (GPU)
                    semseg_start = time.perf_counter()
                    with torch.no_grad():
                        results = self.model.predict_sem_seg(imgs, batch_data_samples)
                    self.time_stats['predict_sem_seg'] += time.perf_counter() - semseg_start
                    # pbar.update(1)  # Update progress bar (once per completed image)
                    if 'gt_instances' in results[0]:
                        del results[0].gt_instances
                    
                    # Extract metadata
                    img_path = batch_data_samples[0].metainfo['img_path']
                    transform = batch_data_samples[0].metainfo['tif_meta']['transform']
                    crs = batch_data_samples[0].metainfo['tif_meta']['crs']
                    
                    # Submit CPU stage1 task (mosaic_sem_seg + seg2ins + sample_segments)
                    self._submit_cpu_stage1_task(
                        imgs,
                        results,
                        img_path,
                        transform,
                        crs
                    )
                    self.submitted_count += 1
                    self.active_tasks += 1
                
                # Process remaining tasks after all images submitted
                while self.completed_count < self.submitted_count:
                    # Update progress description
                    pbar.set_description(f"Finishing processing ({self.active_tasks} active tasks)")
                    
                    # Process GPU tasks
                    self._process_pending_gpu_tasks(pbar)
                    
                    # Brief pause if no tasks to process
                    if self.gpu_task_queue.empty():
                        time.sleep(0.01)
            
            # Calculate total time
            self.time_stats['total'] = time.perf_counter() - total_start
        
        # Print time statistics
        self._print_time_statistics()
        
        return self.completed_count
    
    def _submit_cpu_stage1_task(self, imgs, results, img_path, transform, crs):
        """Submit CPU stage1 task to thread pool"""
        self.executor.submit(
            self._process_cpu_stage1,
            imgs,
            results,
            img_path,
            transform,
            crs
        )
    
    def _process_cpu_stage1(self, imgs, results, img_path, transform, crs):
        """CPU stage1: mosaic_sem_seg, seg2ins, and sample_segments processing"""
        try:
            imgs_cpu = imgs.cpu()
            # mosaic_sem_seg processing
            mosaic_start = time.perf_counter()
            with torch.no_grad():
                results = self.model.predict_mosaic_sem_seg(imgs_cpu, results)
            self.time_stats['predict_mosaic_sem_seg'] += time.perf_counter() - mosaic_start


            cpu_throttle = threading.Semaphore(2)
            # seg2ins processing
            seg2ins_start = time.perf_counter()
            with torch.no_grad():
                with cpu_throttle:
                    results = self.model.seg_poly_head.predict_seg2ins(imgs_cpu, results)

            self.time_stats['predict_seg2ins'] += time.perf_counter() - seg2ins_start
            
            # sample_segments processing
            sample_start = time.perf_counter()
            with torch.no_grad():
                results = self.model.seg_poly_head.poly_head.predict_sample_segments(imgs_cpu, results)
            self.time_stats['predict_sample_segments'] += time.perf_counter() - sample_start
            
            # Add to GPU task queue
            self.gpu_task_queue.put({
                'type': 'gcp',
                'imgs': imgs,
                'results': results,
                'img_path': img_path,
                'transform': transform,
                'crs': crs
            })

        except Exception as e:
            print(f"CPU stage1 task error: {e}")
            self.active_tasks -= 1  # Reduce active task count
    
    def _process_pending_gpu_tasks(self, pbar):
        """Process pending tasks in GPU queue"""
        while not self.gpu_task_queue.empty():
            # Get next GPU task
            task = self.gpu_task_queue.get_nowait()
            
            if task['type'] == 'gcp':
                # GCP processing (GPU)
                gcp_start = time.perf_counter()
                with torch.no_grad():
                    gcp_results = self.model.seg_poly_head.poly_head.predict_gcp(
                        task['imgs'],
                        task['results']
                    )
                self.time_stats['predict_gcp'] += time.perf_counter() - gcp_start
                
                # Submit CPU stage2 task
                self._submit_cpu_stage2_task(
                    task['imgs'],
                    gcp_results,
                    task['img_path'],
                    task['transform'],
                    task['crs']
                )
            
            elif task['type'] == 'dp':
                # DP processing (GPU)
                dp_start = time.perf_counter()
                with torch.no_grad():
                    dp_results = self.model.seg_poly_head.poly_head.predict_dp(
                        task['imgs'],
                        task['results'],
                    )
                self.time_stats['predict_dp'] += time.perf_counter() - dp_start
                
                # Submit save results task to background
                self.executor.submit(
                    self._save_results,
                    dp_results,
                    task['img_path'],
                    task['transform'],
                    task['crs']
                )
                
                # Update completion state
                self.completed_count += 1
                self.active_tasks -= 1
                pbar.update(1)  # Update progress bar (once per completed image)
    
    def _submit_cpu_stage2_task(self, imgs, results, img_path, transform, crs):
        """Submit CPU stage2 task to thread pool"""
        self.executor.submit(
            self._process_cpu_stage2, 
            imgs,
            results,
            img_path,
            transform,
            crs
        )
    
    def _process_cpu_stage2(self, imgs, results, img_path, transform, crs):
        """CPU stage2: assemble_segments processing"""
        try:
            # assemble_segments processing
            assemble_start = time.perf_counter()
            with torch.no_grad():
                results = self.model.seg_poly_head.poly_head.predict_assemble_segments(imgs, results)
            self.time_stats['predict_assemble_segments'] += time.perf_counter() - assemble_start
            
            # Add to GPU task queue
            self.gpu_task_queue.put({
                'type': 'dp',
                'imgs': imgs,
                'results': results,
                'img_path': img_path,
                'transform': transform,
                'crs': crs
            })
        except Exception as e:
            print(f"CPU stage2 task error: {e}")
            self.active_tasks -= 1  # Reduce active task count

    @staticmethod
    def extract_float_from_tensor(self, tensor_str: str) -> float:
        """
            tensor_str (str): tensor字符串，如"tensor(0.6510, dtype=torch.float64)"
            float: 提取的浮点数
        """
        # 使用正则表达式匹配浮点数
        match = re.search(r"[-+]?\d*\.\d+|\d+", tensor_str)
        if match:
            return float(match.group())
        else:
            return tensor_str

    def _save_results(self, results, img_path, transform, crs):
        """Save results to GeoJSON file in a background thread"""
        if not self.save_cfg.get('save_results', False):
            return

        try:
            poly_jsons = results[0].pred_instances['segmentations']
            file_name = results[0].metainfo['img_path'].split('/')[-1].split('.')[0]
            height = results[0].seg_probs.shape[2]  # 图像高度
            width = results[0].seg_probs.shape[3]   # 图像宽度
            scores = results[0].scores
            if type(scores) == torch.Tensor:
                scores = [self.extract_float_from_tensor(score) for score in scores]

            out_scale = self.save_cfg.get('out_poly_scale', 1.0)

            out_dir = self.save_cfg['out_dir']
            flag_dir = os.path.join(out_dir, 'flag')
            out_geojson_dir = os.path.join(out_dir, 'geojson')
            out_tif_dir = os.path.join(out_dir, 'tif')
            out_geojson_path = os.path.join(out_geojson_dir, file_name + '.geojson')
            out_tif_path = os.path.join(out_tif_dir, file_name + '.tif')

            file_name = results[0].metainfo['img_path'].split('/')[-1].split('.')[0]
            start_flag_path = os.path.join(flag_dir, f'{file_name}.start')
            finish_flag_path = os.path.join(flag_dir, f'{file_name}.finish')

            if os.path.exists(os.path.join(finish_flag_path)):
                print(f'{file_name} has already been processed, skip it.')
                return

            # Save polygons
            polygon_utils.save_polygons(
                poly_jsons,
                transform,
                crs,
                out_geojson_path,
                out_scale,
                properties=dict(scores=scores)
            )

            original_transform = results[0].metainfo['tif_meta']['transform']

            pixel_size_x = original_transform.a / 8
            pixel_size_y = original_transform.e / 8

            adjusted_transform = Affine(
                pixel_size_x,            # 新的x方向分辨率
                0,                       # 无旋转
                original_transform.c,    # 左上角x坐标保持不变
                0,                       # 无旋转
                pixel_size_y,            # 新的y方向分辨率
                original_transform.f      # 左上角y坐标保持不变
            )

            if adjusted_transform.e > 0:
                # 确保y方向分辨率为负值（图像从上到下）
                adjusted_transform = Affine(
                    adjusted_transform.a,
                    adjusted_transform.b,
                    adjusted_transform.c,
                    adjusted_transform.d,
                    -adjusted_transform.e,
                    adjusted_transform.f
                )
            seg_probs = (results[0].seg_probs[0,1] * 255).clip(0, 255).to(torch.uint8) # (H, W)
            profile = {
                'driver': 'GTiff',
                'height': height,
                'width': width,
                'count': 1,  # 单波段
                'dtype': rasterio.uint8,
                'crs': crs,
                'transform': adjusted_transform,
                'compress': 'lzw',  # 压缩
                'nodata': None         # 无数据值
            }

            if os.path.exists(os.path.join(finish_flag_path)):
                print(f'{file_name} has already been processed, skip it.')
                return

            with rasterio.open(out_tif_path, 'w', **profile) as dst:
                dst.write(seg_probs, 1)

            with open(finish_flag_path, 'w') as _:
                pass

        except Exception as e:
            print(f"Error saving results for {img_path}: {e}")
    
    def _print_time_statistics(self):
        """Print detailed time usage statistics"""
        print("\nTime Usage Statistics:")
        print("=" * 50)
        print(f"{'Total processing time':<30}: {self.time_stats['total']:.2f} seconds")
        print(f"{'Data preprocessing':<30}: {self.time_stats['data_preprocessor']:.2f} seconds")
        print(f"{'Semantic segmentation':<30}: {self.time_stats['predict_sem_seg']:.2f} seconds")
        print(f"{'Mosaic sem seg':<30}: {self.time_stats['predict_mosaic_sem_seg']:.2f} seconds")
        print(f"{'Seg2ins processing':<30}: {self.time_stats['predict_seg2ins']:.2f} seconds")
        print(f"{'Sample segments':<30}: {self.time_stats['predict_sample_segments']:.2f} seconds")
        print(f"{'GCP prediction':<30}: {self.time_stats['predict_gcp']:.2f} seconds")
        print(f"{'Assemble segments':<30}: {self.time_stats['predict_assemble_segments']:.2f} seconds")
        print(f"{'DP prediction':<30}: {self.time_stats['predict_dp']:.2f} seconds")
        print("=" * 50)

# Usage example
if __name__ == "__main__":
    # Assume model, data_loader, and arguments are defined
    save_cfg = {
        'save_results': True,
        'out_dir': './output',
        'out_poly_scale': 1.0
    }
    
    pipeline = InferencePipeline(
        model=model,
        num_images=100,
        save_cfg=save_cfg,
        cpu_workers=4
    )
    
    processed_count = pipeline.run(data_loader)
    print(f"Successfully processed {processed_count} images")
