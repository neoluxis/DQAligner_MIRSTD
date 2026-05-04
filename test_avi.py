import argparse
import sys

# 提前定义parse_args_avi()，在导入其他模块之前解析参数
def parse_args_avi():
    """AVI处理参数"""
    parser = argparse.ArgumentParser(description='AVI Video Processing with Target Tracking')
    # 继承test.py的参数
    parser.add_argument('--DataPath', type=str, default='./dataset/', help='Dataset path [default: ./dataset/]')
    parser.add_argument('--dataset', type=str, default='NUDT-MIRSDT',
                        help='Dataset name [dafult: NUDT-MIRSDT],IRDST,TSIRMT')
    parser.add_argument('--saveDir', type=str, default='./results/', help='Save path [defaule: ./results/]')
    parser.add_argument('--weight_path', type=str,
                        # default='results/NUDT-MIRSDT/DQAligner/weight_NUDT-MIRSDT.pth',
                        default='results/IRDST/DQAligner_DeepSupFalse_adafocal_2026_04_25__00_02_55/Epoch_5_0.81499_best.pth',
                        help='model weight path')
    parser.add_argument('--model', type=str, default='DQAligner_test_visual',
                        help='ResUNet_DTUM, DNANet_DTUM, ACM, ALCNet, ResUNet, DNANet, ISNet, UIU')
    parser.add_argument('--fullySupervised', default=True)
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument('--DataParallel', default=False, help='Use one gpu or more')
    
    # 新增AVI处理参数
    parser.add_argument('--avi_path', type=str,
                        # default='./TEST_video/',
                        default='../CST_AntiUAV/avi/train/building_19.avi',
                        help='Input AVI video path or directory [default: ./TEST_video/]')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Output directory for results [default: test/{dataset}/CST/]')
    parser.add_argument('--max_age', type=int, default=30,
                        help='Maximum age for tracking [default: 30]')
    parser.add_argument('--iou_threshold', type=float, default=0.3,
                        help='IoU threshold for data association [default: 0.3]')
    parser.add_argument('--mask_threshold', type=float, default=0.5,
                        help='Mask binarization threshold for contour extraction and visualization [default: 0.5]')
    parser.add_argument('--min_box_width', type=int, default=3,
                        help='Minimum detected box width in pixels [default: 3]')
    parser.add_argument('--min_box_height', type=int, default=3,
                        help='Minimum detected box height in pixels [default: 3]')
    parser.add_argument('--min_area', type=float, default=0.0,
                        help='Minimum contour area to keep a detection [default: 0.0]')
    parser.add_argument('--min_score', type=float, default=0.0,
                        help='Minimum average mask score inside a box [default: 0.0]')
    parser.add_argument('--max_center_distance', type=float, default=12.0,
                        help='Maximum center distance in pixels for track association [default: 12.0]')
    parser.add_argument('--pad_to_512', action='store_true',
                        help='Pad frames to 512x512 before inference. Disabled by default.')
    parser.add_argument('--custom_mean', type=str, default='auto',
                        help='Custom normalization mean for dataset CUST. Use a number or auto [default: auto].')
    parser.add_argument('--custom_std', type=str, default='auto',
                        help='Custom normalization std for dataset CUST. Use a number or auto [default: auto].')
    parser.add_argument('--cust_recalc', type=int, default=20,
                        help='Sliding window length for CUST auto statistics [default: 20].')
    parser.add_argument('--recalc_overlap', type=int, default=10,
                        help='Overlap length between adjacent CUST recalculation windows [default: 10].')
    
    args = parser.parse_args()
    if args.cust_recalc <= 0:
        parser.error('--cust_recalc must be > 0')
    if args.recalc_overlap < 0 or args.recalc_overlap >= args.cust_recalc:
        parser.error('--recalc_overlap must satisfy 0 <= recalc_overlap < cust_recalc')
    print("model: %s, dataset: %s, avi_path: %s" % (args.model, args.dataset, args.avi_path))
    
    # 设置输出路径
    if args.output_path is None:
        args.output_path = args.saveDir + f'{args.dataset}/test_avi_results/'
    
    return args


# 在导入其他重型模块之前，先处理参数
_args_avi = parse_args_avi()

# 清空sys.argv避免test.py中的parse_args再次处理
# 只保留脚本名称
sys.argv = [sys.argv[0]]

import cv2
import numpy as np
import torch
import os
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image
import time
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from collections import defaultdict, deque

# 导入复用的模块
from test import Trainer, seed_pytorch
from utils.metric_basic import *
from model.DQAligner import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.autograd.set_detect_anomaly(True)

BOX_THICKNESS = 1
TEXT_SCALE = 0.35
TEXT_THICKNESS = 1
TRACK_LINE_THICKNESS = 1
TRACK_POINT_RADIUS = 1
TRACK_HISTORY_LENGTH = 20
TEXT_OUTLINE_THICKNESS = 2
TEXT_OUTLINE_COLOR = (16, 16, 16)


class TargetTracker:
    """多目标跟踪器，保持目标ID和颜色一致"""
    
    def __init__(self, max_age=30, iou_threshold=0.3, mask_threshold=0.5,
                 min_box_width=3, min_box_height=3, min_area=0.0, min_score=0.0,
                 max_center_distance=12.0):
        self.max_age = max_age
        self.iou_threshold = iou_threshold
        self.mask_threshold = mask_threshold
        self.min_box_width = min_box_width
        self.min_box_height = min_box_height
        self.min_area = min_area
        self.min_score = min_score
        self.max_center_distance = max_center_distance
        self.next_id = 1
        self.tracks = {}  # {track_id: {'box': ..., 'color': ..., 'age': ..., 'score': ...}}
        self.colors = self._generate_colors(100)
    
    def _generate_colors(self, num_colors):
        """生成更亮、更适合低分辨率视频的颜色序列"""
        base_colors = [
            (255, 255, 0),
            (0, 255, 255),
            (255, 128, 0),
            (0, 255, 128),
            (255, 64, 255),
            (128, 255, 0),
            (0, 160, 255),
            (255, 0, 128),
        ]
        colors = []
        for i in range(num_colors):
            colors.append(base_colors[i % len(base_colors)])
        return colors
    
    def _compute_iou(self, box1, box2):
        """计算两个框的IoU"""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / (union_area + 1e-6)

    def _box_center(self, box):
        x1, y1, x2, y2 = box
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)

    def _create_kalman_state(self, center):
        state = np.array([center[0], center[1], 0.0, 0.0], dtype=np.float32)
        covariance = np.diag([4.0, 4.0, 9.0, 9.0]).astype(np.float32)
        return state, covariance

    def _predict_kalman(self, track):
        transition = np.array(
            [[1.0, 0.0, 1.0, 0.0],
             [0.0, 1.0, 0.0, 1.0],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32
        )
        process_noise = np.diag([0.8, 0.8, 1.5, 1.5]).astype(np.float32)
        state = transition @ track['kf_state']
        covariance = transition @ track['kf_cov'] @ transition.T + process_noise
        track['kf_state'] = state
        track['kf_cov'] = covariance
        track['center'] = state[:2].copy()
        return state[:2].copy()

    def _update_kalman(self, track, measurement_center):
        measurement_matrix = np.array(
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float32
        )
        measurement_noise = np.diag([2.5, 2.5]).astype(np.float32)
        measurement = np.asarray(measurement_center, dtype=np.float32)
        state = track['kf_state']
        covariance = track['kf_cov']
        innovation = measurement - (measurement_matrix @ state)
        innovation_cov = measurement_matrix @ covariance @ measurement_matrix.T + measurement_noise
        kalman_gain = covariance @ measurement_matrix.T @ np.linalg.inv(innovation_cov)
        updated_state = state + kalman_gain @ innovation
        identity = np.eye(4, dtype=np.float32)
        updated_covariance = (identity - kalman_gain @ measurement_matrix) @ covariance
        track['kf_state'] = updated_state
        track['kf_cov'] = updated_covariance
        track['center'] = updated_state[:2].copy()
        track['velocity'] = updated_state[2:].copy()
        return updated_state[:2].copy()

    def _center_to_box(self, center, width, height):
        half_w = width / 2.0
        half_h = height / 2.0
        return (
            float(center[0] - half_w),
            float(center[1] - half_h),
            float(center[0] + half_w),
            float(center[1] + half_h)
        )

    def _predict_box(self, track):
        pred_center = np.asarray(track.get('center', self._box_center(track['box'])), dtype=np.float32)
        width = track.get('width', max(1.0, track['box'][2] - track['box'][0]))
        height = track.get('height', max(1.0, track['box'][3] - track['box'][1]))
        return self._center_to_box(pred_center, width, height)

    def _association_cost(self, track, det_box, det_score, det_area):
        pred_box = self._predict_box(track)
        pred_center = self._box_center(pred_box)
        det_center = self._box_center(det_box)
        center_distance = float(np.linalg.norm(pred_center - det_center))

        distance_gate = self.max_center_distance * (1.0 + 0.25 * min(track.get('age', 0), 4))
        if center_distance > distance_gate:
            return None

        iou = self._compute_iou(pred_box, det_box)
        area_ratio = det_area / max(track['area'], 1e-6)
        area_cost = min(abs(np.log(max(area_ratio, 1e-6))), 2.0) / 2.0
        score_cost = abs(det_score - track['score'])
        normalized_distance = center_distance / max(distance_gate, 1e-6)

        # For tiny targets, center distance is more reliable than IoU.
        cost = 0.65 * normalized_distance + 0.25 * (1.0 - iou) + 0.07 * area_cost + 0.03 * score_cost

        if iou < self.iou_threshold and center_distance > self.max_center_distance:
            return None

        return cost

    def _create_track(self, box, score, area):
        color_idx = (self.next_id - 1) % len(self.colors)
        center = self._box_center(box)
        width = float(max(1.0, box[2] - box[0]))
        height = float(max(1.0, box[3] - box[1]))
        kf_state, kf_cov = self._create_kalman_state(center)
        self.tracks[self.next_id] = {
            'box': box,
            'color': self.colors[color_idx],
            'age': 0,
            'score': score,
            'area': area,
            'center': center,
            'velocity': np.zeros(2, dtype=np.float32),
            'width': width,
            'height': height,
            'predicted_box': box,
            'kf_state': kf_state,
            'kf_cov': kf_cov,
            'hits': 1,
            'history': [center.copy()]
        }
        self.next_id += 1
    
    def _extract_boxes(self, mask):
        """从掩模中提取边界框和得分"""
        # 二值化
        binary_mask = (mask > self.mask_threshold).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            contour_mask = np.zeros((h, w), dtype=np.uint8)
            shifted_contour = contour - np.array([[[x, y]]], dtype=contour.dtype)
            cv2.drawContours(contour_mask, [shifted_contour], -1, 1, thickness=-1)
            contour_pixels = mask[y:y+h, x:x+w][contour_mask.astype(bool)]
            score = float(contour_pixels.mean()) if contour_pixels.size > 0 else 0.0
            if w < self.min_box_width or h < self.min_box_height:
                continue
            if area < self.min_area:
                continue
            if score < self.min_score:
                continue

            box = (x, y, x + w, y + h)
            detections.append((box, score, area))
        
        return detections
    
    def update(self, mask):
        """更新跟踪器，返回跟踪结果"""
        detections = self._extract_boxes(mask)
        track_ids = list(self.tracks.keys())
        for track_id in track_ids:
            predicted_center = self._predict_kalman(self.tracks[track_id])
            width = self.tracks[track_id].get('width', max(1.0, self.tracks[track_id]['box'][2] - self.tracks[track_id]['box'][0]))
            height = self.tracks[track_id].get('height', max(1.0, self.tracks[track_id]['box'][3] - self.tracks[track_id]['box'][1]))
            self.tracks[track_id]['predicted_box'] = self._center_to_box(predicted_center, width, height)
        
        # 匈牙利算法匹配
        if len(self.tracks) == 0:
            # 如果没有已有的轨迹，直接创建新轨迹
            for box, score, area in detections:
                self._create_track(box, score, area)
        else:
            # 计算基于运动预测的关联代价矩阵
            num_tracks = len(self.tracks)
            num_dets = len(detections)
            cost_matrix = np.full((num_tracks, num_dets), 1e6, dtype=np.float32)
            
            track_ids = list(self.tracks.keys())
            for i, track_id in enumerate(track_ids):
                for j, (det_box, det_score, det_area) in enumerate(detections):
                    cost = self._association_cost(self.tracks[track_id], det_box, det_score, det_area)
                    if cost is not None:
                        cost_matrix[i, j] = cost
            
            # 匈牙利算法
            if num_tracks > 0 and num_dets > 0:
                track_indices, det_indices = linear_sum_assignment(cost_matrix)
            else:
                track_indices = np.array([], dtype=np.int64)
                det_indices = np.array([], dtype=np.int64)
            
            prev_track_ids = list(track_ids)
            matched_tracks = set()
            matched_det_indices = set()
            for track_idx, det_idx in zip(track_indices, det_indices):
                if cost_matrix[track_idx, det_idx] >= 1e5:
                    continue

                track_id = track_ids[track_idx]
                box, score, area = detections[det_idx]
                new_center = self._box_center(box)
                updated_center = self._update_kalman(self.tracks[track_id], new_center)
                width = float(max(1.0, box[2] - box[0]))
                height = float(max(1.0, box[3] - box[1]))

                self.tracks[track_id]['box'] = box
                self.tracks[track_id]['score'] = score
                self.tracks[track_id]['age'] = 0
                self.tracks[track_id]['area'] = area
                self.tracks[track_id]['center'] = updated_center
                self.tracks[track_id]['width'] = width
                self.tracks[track_id]['height'] = height
                self.tracks[track_id]['predicted_box'] = box
                self.tracks[track_id]['hits'] = self.tracks[track_id].get('hits', 0) + 1
                history = self.tracks[track_id].setdefault('history', [])
                history.append(updated_center.copy())
                if len(history) > TRACK_HISTORY_LENGTH:
                    del history[:-TRACK_HISTORY_LENGTH]
                matched_tracks.add(track_id)
                matched_det_indices.add(int(det_idx))
            
            # 处理未匹配的检测（创建新轨迹）
            for j, (box, score, area) in enumerate(detections):
                if j not in matched_det_indices:
                    self._create_track(box, score, area)
            
            # 增加未匹配轨迹的年龄
            for track_id in prev_track_ids:
                if track_id not in matched_tracks:
                    self.tracks[track_id]['age'] += 1
                    self.tracks[track_id]['box'] = self.tracks[track_id].get('predicted_box', self.tracks[track_id]['box'])
                    history = self.tracks[track_id].setdefault('history', [])
                    history.append(self.tracks[track_id]['center'].copy())
                    if len(history) > TRACK_HISTORY_LENGTH:
                        del history[:-TRACK_HISTORY_LENGTH]
            
            # 移除过期轨迹
            expired_ids = [tid for tid, track in self.tracks.items() if track['age'] > self.max_age]
            for tid in expired_ids:
                del self.tracks[tid]
        
        return self.tracks.copy()


class AVIProcessor:
    """处理AVI视频文件的推理和输出"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = 512
        self.use_cust_profile = args.dataset.upper() == 'CUST'
        self.cust_auto_mean = False
        self.cust_auto_std = False
        self.cust_recent_frames = deque(maxlen=args.cust_recalc)
        self.cust_frames_since_recalc = 0
        self.cust_recalc_step = max(1, args.cust_recalc - args.recalc_overlap)
        norm_profile = args.dataset
        if norm_profile not in ['IRDST', 'TSIRMT', 'NUDT-MIRSDT']:
            weight_path_lower = args.weight_path.lower()
            if 'irdst' in weight_path_lower or 'tsirmt' in weight_path_lower:
                norm_profile = 'IRDST'
            elif 'nudt-mirsdt' in weight_path_lower:
                norm_profile = 'NUDT-MIRSDT'

        if self.use_cust_profile:
            self.test_mean, self.cust_auto_mean = self._parse_custom_stat(args.custom_mean, default_value=0.0)
            self.test_std, self.cust_auto_std = self._parse_custom_stat(args.custom_std, default_value=255.0)
            self.test_std = max(self.test_std, 1e-6)
        elif norm_profile in ['IRDST', 'TSIRMT']:
            self.test_mean = 94.96572
            self.test_std = 37.13109
        elif norm_profile == 'NUDT-MIRSDT':
            self.test_mean = 105.4025
            self.test_std = 26.6452
        else:
            self.test_mean = 0.0
            self.test_std = 255.0
        
        # 加载模型
        self.net = DQAligner(input_channels=1, num_frames=5, train_mode=True, key_mode='last')
        self.net = self.net.to(self.device)
        
        # 加载权重
        checkpoint = torch.load(args.weight_path, map_location=self.device, weights_only=False)
        # 处理两种情况：state_dict或完整模型
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            self.net.load_state_dict(checkpoint['state_dict'])
        elif isinstance(checkpoint, dict):
            self.net.load_state_dict(checkpoint)
        else:
            # checkpoint是完整的模型对象
            self.net = checkpoint
        
        self.net.eval()
        
        # 初始化跟踪器（使用args中的参数）
        self.tracker = TargetTracker(
            max_age=args.max_age,
            iou_threshold=args.iou_threshold,
            mask_threshold=args.mask_threshold,
            min_box_width=args.min_box_width,
            min_box_height=args.min_box_height,
            min_area=args.min_area,
            min_score=args.min_score,
            max_center_distance=args.max_center_distance
        )
        
        # 视频属性 - 使用更稳定的编码格式
        self.fps = 16
        # 尝试使用MP4V编码，如果不可用则使用MJPG
        try:
            self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        except:
            self.fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        
        # 推理缓存（用于多帧输入）
        self.frame_buffer = []
        self.feat_prop = None
        self.cat_flag = 0

    def _parse_custom_stat(self, value, default_value):
        if isinstance(value, str) and value.lower() == 'auto':
            return float(default_value), True
        return float(value), False

    def _update_cust_stats(self, frame_gray):
        self.cust_recent_frames.append(frame_gray.astype(np.float32))
        self.cust_frames_since_recalc += 1

        if not (self.cust_auto_mean or self.cust_auto_std):
            return

        should_recalc = False
        if len(self.cust_recent_frames) == 1:
            should_recalc = True
        elif len(self.cust_recent_frames) < self.args.cust_recalc:
            should_recalc = True
        elif self.cust_frames_since_recalc >= self.cust_recalc_step:
            should_recalc = True

        if not should_recalc:
            return

        stacked = np.stack(self.cust_recent_frames, axis=0)
        if self.cust_auto_mean:
            self.test_mean = float(stacked.mean())
        if self.cust_auto_std:
            self.test_std = max(float(stacked.std()), 1.0)
        self.cust_frames_since_recalc = 0
    
    def _preprocess_frame(self, frame, target_size=512):
        """预处理单帧"""
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        h, w = frame.shape
        if self.use_cust_profile:
            self._update_cust_stats(frame)
        frame = frame.astype(np.float32)
        if self.test_std > 0:
            frame = (frame - self.test_mean) / self.test_std
        else:
            frame = frame / 255.0

        if self.args.pad_to_512:
            frame_pad = np.zeros((target_size, target_size), dtype=np.float32)
            frame_pad[0:h, 0:w] = frame
            frame = frame_pad

        frame = torch.from_numpy(frame).float().unsqueeze(0).unsqueeze(0)
        # 保持在CPU上，避免GPU内存占用
        return frame, h, w
    
    def _infer_frame(self, frame_tensor):
        """对单帧进行推理"""
        # 扩展为5帧（简单复制）
        if len(self.frame_buffer) < 5:
            # 只保留引用，不复制数据
            self.frame_buffer.append(frame_tensor)
            if len(self.frame_buffer) < 5:
                return None
        else:
            # 移除最旧的帧以释放内存
            old_frame = self.frame_buffer.pop(0)
            del old_frame
            self.frame_buffer.append(frame_tensor)
        
        # 堆叠为序列（沿着通道维度dim=1，不是dim=0）
        seq_tensor = torch.cat(self.frame_buffer, dim=1).unsqueeze(0).to(self.device)
        
        try:
            with torch.no_grad():
                outputs = self.net(seq_tensor, self.feat_prop, self.cat_flag, False)
                
                if isinstance(outputs, (list, tuple)):
                    pred = outputs[1].squeeze(2)
                    # 分离feat_prop以避免内存泄漏
                    if outputs[2] is not None:
                        self.feat_prop = outputs[2].detach() if torch.is_tensor(outputs[2]) else outputs[2]
                    else:
                        self.feat_prop = None
                else:
                    pred = outputs
                    self.feat_prop = None
                
                self.cat_flag = 1
                # 将掩模转换为numpy并立即清理GPU内存
                pred_mask = torch.sigmoid(pred).squeeze(0).squeeze(0).cpu().numpy()
                del pred, seq_tensor, outputs
        finally:
            # 确保清理seq_tensor
            torch.cuda.empty_cache()
        
        return pred_mask
    
    def process_avi(self, input_video_path, output_dir):
        """处理AVI视频"""
        print(f"Processing video: {input_video_path}")
        
        # 打开输入视频
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {input_video_path}")
            return
        
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        self.fps = fps if fps > 0 else 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video info: {frame_width}x{frame_height}, {self.fps} FPS, {total_frames} frames")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化输出视频写入器
        bbox_output = os.path.join(output_dir, 'bbox_output.avi')
        mask_output = os.path.join(output_dir, 'mask_output.avi')
        concat_output = os.path.join(output_dir, 'concat_output.avi')
        
        try:
            writer_bbox = cv2.VideoWriter(bbox_output, self.fourcc, self.fps, (frame_width, frame_height))
            writer_mask = cv2.VideoWriter(mask_output, self.fourcc, self.fps, (frame_width, frame_height))
            writer_concat = cv2.VideoWriter(concat_output, self.fourcc, self.fps, (frame_width * 2, frame_height))
            
            if not all([writer_bbox.isOpened(), writer_mask.isOpened(), writer_concat.isOpened()]):
                print("Warning: Some video writers failed to open, trying with MJPG codec")
                writer_bbox = cv2.VideoWriter(bbox_output, cv2.VideoWriter_fourcc(*'MJPG'), self.fps, (frame_width, frame_height))
                writer_mask = cv2.VideoWriter(mask_output, cv2.VideoWriter_fourcc(*'MJPG'), self.fps, (frame_width, frame_height))
                writer_concat = cv2.VideoWriter(concat_output, cv2.VideoWriter_fourcc(*'MJPG'), self.fps, (frame_width * 2, frame_height))
        except Exception as e:
            print(f"Error creating video writers: {e}")
            return
        
        # 重置跟踪器和推理缓存
        self.tracker = TargetTracker(
            max_age=self.args.max_age,
            iou_threshold=self.args.iou_threshold,
            mask_threshold=self.args.mask_threshold,
            min_box_width=self.args.min_box_width,
            min_box_height=self.args.min_box_height,
            min_area=self.args.min_area,
            min_score=self.args.min_score,
            max_center_distance=self.args.max_center_distance
        )
        self.frame_buffer = []
        self.feat_prop = None
        self.cat_flag = 0
        self.cust_recent_frames.clear()
        self.cust_frames_since_recalc = 0
        
        frame_idx = 0
        pbar = tqdm(total=total_frames, desc="Processing frames")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                try:
                    # 预处理
                    frame_tensor, h, w = self._preprocess_frame(frame)
                    
                    # 推理
                    pred_mask = self._infer_frame(frame_tensor)
                    
                    if pred_mask is not None:
                        # 调整掩模大小到原始尺寸
                        if pred_mask.shape != (h, w):
                            pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_LINEAR)
                        
                        # 更新跟踪器
                        tracks = self.tracker.update(pred_mask)
                        
                        # 生成三个输出
                        bbox_frame = frame.copy()
                        mask_frame = np.zeros((h, w, 3), dtype=np.uint8)
                        
                        # 绘制检测框和跟踪ID
                        for track_id, track_data in tracks.items():
                            x1, y1, x2, y2 = track_data['box']
                            color = track_data['color']
                            score = track_data['score']
                            history = track_data.get('history', [])
                            is_predicted_only = track_data.get('age', 0) > 0

                            if len(history) >= 2:
                                pts = np.array([[int(p[0]), int(p[1])] for p in history], dtype=np.int32)
                                cv2.polylines(bbox_frame, [pts], False, color, TRACK_LINE_THICKNESS)
                            for point in history:
                                cv2.circle(
                                    bbox_frame,
                                    (int(point[0]), int(point[1])),
                                    TRACK_POINT_RADIUS,
                                    color,
                                    -1
                                )
                            
                            # 在bbox_frame上绘制
                            cv2.rectangle(
                                bbox_frame,
                                (int(x1), int(y1)),
                                (int(x2), int(y2)),
                                color,
                                BOX_THICKNESS
                            )
                            label = f"ID:{track_id} S:{score:.2f}"
                            if is_predicted_only:
                                label += " P"
                            text_y = max(8, int(y1) - 3)
                            cv2.putText(
                                bbox_frame,
                                label,
                                (int(x1), text_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                TEXT_SCALE,
                                TEXT_OUTLINE_COLOR,
                                TEXT_OUTLINE_THICKNESS
                            )
                            cv2.putText(
                                bbox_frame,
                                label,
                                (int(x1), text_y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                TEXT_SCALE,
                                color,
                                TEXT_THICKNESS
                            )
                        
                        # 生成掩模可视化
                        mask_binary = (pred_mask > self.args.mask_threshold).astype(np.uint8) * 255
                        mask_frame[:, :, 0] = mask_binary
                        mask_frame[:, :, 1] = mask_binary
                        mask_frame[:, :, 2] = mask_binary
                        
                        # 左侧显示带目标框的视频，右侧显示掩模可视化
                        concat_frame = np.hstack([bbox_frame, mask_frame])
                        
                        # 写入输出视频
                        writer_bbox.write(bbox_frame)
                        writer_mask.write(mask_frame)
                        writer_concat.write(concat_frame)
                        
                        # 清理临时数组
                        del bbox_frame, mask_frame, concat_frame, pred_mask
                    else:
                        # 推理缓存未满，先不输出
                        pass
                    
                    # 定期清理内存
                    if frame_idx % 10 == 0:
                        torch.cuda.empty_cache()
                    
                    frame_idx += 1
                    pbar.update(1)
                    
                except Exception as e:
                    print(f"Error processing frame {frame_idx}: {e}")
                    frame_idx += 1
                    pbar.update(1)
                    continue
        
        finally:
            cap.release()
            writer_bbox.release()
            writer_mask.release()
            writer_concat.release()
            pbar.close()
            torch.cuda.empty_cache()
        
        print(f"\nProcessing complete!")
        print(f"Output videos saved to: {output_dir}")
        print(f"  - Bounding Box: {bbox_output}")
        print(f"  - Mask: {mask_output}")
        print(f"  - Concatenated: {concat_output}")


def main():
    # 使用全局的_args_avi参数
    args = _args_avi
    seed_pytorch(args.seed)
    
    # 创建AVI处理器
    processor = AVIProcessor(args)
    
    # 处理视频
    avi_path = args.avi_path
    output_path = args.output_path
    
    # 检查输入是目录还是单个文件
    if os.path.isdir(avi_path):
        # 处理目录中的所有视频
        video_files = [f for f in os.listdir(avi_path) if f.endswith(('.avi', '.mp4', '.mov'))]
        if not video_files:
            print(f"No video files found in {avi_path}")
            return
        
        for video_file in video_files:
            video_filepath = os.path.join(avi_path, video_file)
            output_subdir = os.path.join(output_path, video_file.split('.')[0])
            processor.process_avi(video_filepath, output_subdir)
    elif os.path.isfile(avi_path):
        # 处理单个文件
        video_name = os.path.basename(avi_path).split('.')[0]
        output_subdir = os.path.join(output_path, video_name)
        processor.process_avi(avi_path, output_subdir)
    else:
        print(f"Error: AVI path not found: {avi_path}")
        print("Usage: python test_avi.py --avi_path <path_to_video> --output_path <output_dir>")


if __name__ == '__main__':
    main()
