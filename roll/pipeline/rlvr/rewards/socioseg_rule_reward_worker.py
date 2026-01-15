import json
import re
import numpy as np
import torch
import ast
from math import exp
from scipy.optimize import linear_sum_assignment
from typing import Optional, Union
from roll.configs.worker_config import WorkerConfig
from roll.distributed.executor.worker import Worker
from roll.distributed.scheduler.decorator import Dispatch, register
from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.strategy import InferenceStrategy, TrainStrategy
from roll.models.model_providers import default_tokenizer_provider

def _batch_iou(boxes1, boxes2):
    """Calculates Intersection over Union (IoU) for batches of boxes."""
    x11, y11, x12, y12 = np.split(boxes1, 4, axis=1)
    x21, y21, x22, y22 = np.split(boxes2, 4, axis=1)
    
    xA = np.maximum(x11, np.transpose(x21))
    yA = np.maximum(y11, np.transpose(y21))
    xB = np.minimum(x12, np.transpose(x22))
    yB = np.minimum(y12, np.transpose(y22))
    
    interArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)
    box1Area = (x12 - x11 + 1) * (y12 - y11 + 1)
    box2Area = (x22 - x21 + 1) * (y22 - y21 + 1)
    
    unionArea = box1Area + np.transpose(box2Area) - interArea
    iou = interArea / np.maximum(unionArea, 1e-6) # Avoid division by zero
    return iou

def _batch_l1_distance(boxes1, boxes2):
    """Calculates mean L1 distance for batches of boxes."""
    boxes1 = boxes1[:, np.newaxis, :]
    boxes2 = boxes2[np.newaxis, :, :]
    return np.mean(np.abs(boxes1 - boxes2), axis=2)

def _multi_s1_format_reward(predict_str: str) -> float:
    """Calculates the format reward for a single prediction string."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    match = re.fullmatch(pattern, predict_str, re.DOTALL)
    thinking_format_reward = 1.0 if match else 0.0
    
    segmentation_format_reward = 0.0
    try:
        json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
        if not json_match:
            return thinking_format_reward
        
        data = json.loads(json_match.group(1))
        if not data: # Handle empty list in answer
            return thinking_format_reward

        data_cnt = len(data)
        total_cur_reward = 0.0
        
        for item in data:
            cur_reward = 0.0
            if item.keys() == {'bbox_2d'}:
                bbox_2d = item['bbox_2d']
                if isinstance(bbox_2d, list) and len(bbox_2d) == 4:
                    cur_reward += 1.0
            
            total_cur_reward += cur_reward
        
        segmentation_format_reward = total_cur_reward / data_cnt
    except Exception:
        pass # Return 0.0 for segmentation part on error
        
    return thinking_format_reward + segmentation_format_reward

def _multi_s2_format_reward(predict_str: str, bbox_text: str) -> float:
    """Calculates the format reward for a single prediction string."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    match = re.fullmatch(pattern, predict_str, re.DOTALL)
    thinking_format_reward = 1.0 if match else 0.0
    
    segmentation_format_reward = 0.0
    try:
        json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
        if not json_match:
            return thinking_format_reward
        
        data = json.loads(json_match.group(1))
        stage1_bbox_text = bbox_text.replace("'", '"')
        stage1_bbox_data = json.loads(stage1_bbox_text)
        if not data: # Handle empty list in answer
            return thinking_format_reward

        data_cnt = len(data)
        total_cur_reward = 0.0
        
        if len(stage1_bbox_data) != data_cnt:
            return thinking_format_reward
        
        for item, stage1_item in zip(data, stage1_bbox_data):
            cur_reward = 0.0
            stage1_bbox_2d= stage1_item['bbox_2d']
            if 'bbox_2d' in item.keys() and 'points' in item.keys():
                bbox_2d = item['bbox_2d']
                point_2d = item['points']
                if isinstance(bbox_2d, list) and len(bbox_2d) == 4 and isinstance(point_2d, list):
                    flag = True
                    # 判断bbox是否和stage1_bbox_2d相同
                    if stage1_bbox_2d != bbox_2d:
                        flag = False
                    for point in point_2d:
                        if not isinstance(point, list) or len(point) != 2:
                            flag = False

                        if point[0] <= bbox_2d[0] or point[0] >= bbox_2d[2] or point[1] <= bbox_2d[1] or point[1] >= bbox_2d[3]:
                            flag = False
                            break
                    if flag:
                        cur_reward += 1.0
            
            total_cur_reward += cur_reward
        
        segmentation_format_reward = total_cur_reward / data_cnt
    except Exception:
        pass # Return 0.0 for segmentation part on error
        
    return thinking_format_reward + segmentation_format_reward

def _multi_s1_accuracy_reward(predict_str: str, ground_truth: str) -> float:
    """Calculates the accuracy reward using Hungarian matching."""
    max_accuracy_reward = 0.0
    MAX_OBJECTS = 120
    
    try:
        gt_data = json.loads(ground_truth.replace("'", '"'))
        gt_bboxes = np.array([item['bbox_2d'] for item in gt_data])
        
        json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
        if not json_match:
            return 0.0
        pred_data = json.loads(json_match.group(1))
        if not pred_data: # Handle empty prediction
            return 0.0
                
        pred_bboxes = np.array([item['bbox_2d'] for item in pred_data])

        # Truncate if exceeding max objects
        if len(pred_bboxes) > MAX_OBJECTS:
            pred_bboxes = pred_bboxes[:MAX_OBJECTS]
        
        if len(gt_bboxes) > MAX_OBJECTS:
            gt_bboxes = gt_bboxes[:MAX_OBJECTS]
        
        if len(pred_bboxes) == 0 or len(gt_bboxes) == 0:
            return 0.0

        # Calculate metrics using the helper functions
        iou_matrix = _batch_iou(pred_bboxes, gt_bboxes)
        l1_matrix = _batch_l1_distance(pred_bboxes, gt_bboxes)
        
        # Calculate reward components
        iou_reward = (iou_matrix > 0.5).astype(float)
        bbox_l1_reward = (l1_matrix < 10).astype(float)

        cost_matrix = 2.0 - iou_reward - bbox_l1_reward
        
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Calculate total reward from matched pairs
        total_reward = len(row_indices) - cost_matrix[row_indices, col_indices].sum()
        
        # Normalize by the max number of objects to penalize mismatches in count
        max_length = max(len(pred_bboxes), len(gt_bboxes))
        if max_length == 0: return 0.0
        
        max_accuracy_reward = total_reward / max_length
        
    except Exception:
        pass # Return 0.0 on any error
        
    return max_accuracy_reward

def _multi_s2_accuracy_reward(mask: np.ndarray, gt_mask: np.ndarray) -> float:

    if not isinstance(mask, np.ndarray) or not isinstance(gt_mask, np.ndarray):
        return 0.0
    
    if mask.shape != gt_mask.shape:
        return 0.0

    mask = mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    intersection = np.logical_and(mask, gt_mask).sum()
    union = np.logical_or(mask, gt_mask).sum()

    if union == 0:
        return 0.0

    iou = intersection / union
    return iou

def _batch_points_distance(points1, points2):
    """Calculates Euclidean distance for batches of points."""
    points1 = points1[:, np.newaxis, :]
    points2 = points2[np.newaxis, :, :]
    dist = np.sqrt(np.sum((points1 - points2)**2, axis=2))
    return dist


def _multi_s1_length_reward(predict_str: str, ground_truth: str) -> float:
    try:
        gt_data = json.loads(ground_truth.replace("'", '"'))
        gt_bboxes = np.array([item['bbox_2d'] for item in gt_data])
        gt_length = len(gt_bboxes)

        json_match = re.search(r'<answer>\s*(.*?)\s*</answer>', predict_str, re.DOTALL)
        if not json_match:
            return 0.0
        
        pred_data = json.loads(json_match.group(1))
        pred_bboxes = np.array([item['bbox_2d'] for item in pred_data])
        pred_length = len(pred_bboxes)

        J = gt_length
        K = pred_length

        if J == 0 and K == 0:
            return 1.0
        elif J == 0 and K > 0:
            return 0.0
        else: # J > 0
            return np.exp(-2 * abs(K - J) / J)

    except (json.JSONDecodeError, re.error, ValueError, IndexError, TypeError, SyntaxError, KeyError) as e:
        return 0.0
    
def _multi_s2_length_reward(text: str) -> float:
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.MULTILINE)
    if not match:
        return 0
    reward = 0
    answer_content = match.group(1).strip()
    try:
        parsed_answer = json.loads(answer_content)
        for group in parsed_answer:
            if 'points' not in group:
                continue
            length = len(group['points'])
            ideal = 2
            sigma = 2
            reward += exp(-((length - ideal) ** 2) / (2 * sigma ** 2))
            
        reward = reward / len(parsed_answer) if parsed_answer else 0
        return reward
    except Exception:
        return 0


class SocioSegRuleRewardWorker(Worker):

    def __init__(self, worker_config: WorkerConfig):
        super().__init__(worker_config=worker_config)
        self.rank_info.dp_rank = self.rank_info.rank
        self.rank_info.dp_size = self.rank_info.world_size
        self.tokenizer = default_tokenizer_provider(model_args=self.worker_config.model_args)
        self.strategy: Optional[Union[InferenceStrategy, TrainStrategy]] = None
        self.format_pattern = self.worker_config.format_pattern

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def initialize(self, pipeline_config):
        pass

    @register(dispatch_mode=Dispatch.DP_MP_COMPUTE, clear_cache=False)
    def compute_rewards_split(self, data: DataProto):
        """
        Computes rewards for both satellite and map responses.
        
        Returns a DataProto containing reward tensors and metrics for both response types.
        """

        map_response_text_list = self.tokenizer.batch_decode(data.batch["map_responses"], skip_special_tokens=False)
        sat_response_text_list = self.tokenizer.batch_decode(data.batch["sat_responses"], skip_special_tokens=False)

        map_pred_mask_list = data.non_tensor_batch['map_mask']
        sat_pred_mask_list = data.non_tensor_batch['sat_mask']
        gt_mask_list = data.non_tensor_batch["gt_mask"]
        gt_bbox_list = data.non_tensor_batch["gt_bbox"]
        bbox_text_list = data.non_tensor_batch["bboxs_text"]
        gt_point_list = data.non_tensor_batch["gt_point"]


        map_format_rewards = []
        map_length_rewards = []
        map_accuracy_rewards = []
        map_seg_iou_accuracies = []


        for response, pred_mask, gt_mask, gt_bbox in zip(map_response_text_list, map_pred_mask_list, gt_mask_list, gt_bbox_list):
            response = response.replace("<|endoftext|>", "").replace("<|im_end|>", "").replace("<pad>", "")
            
            format_reward = _multi_s1_format_reward(response)
            map_format_rewards.append(format_reward)

            length_reward = _multi_s1_length_reward(response, gt_bbox)
            map_length_rewards.append(length_reward)

            accuracy_reward = _multi_s1_accuracy_reward(response, gt_bbox)
            map_accuracy_rewards.append(accuracy_reward)
            
            gt_mask_np = np.array(gt_mask.convert("L"))
            iou_accuracy = _multi_s2_accuracy_reward(pred_mask, gt_mask_np)
            map_seg_iou_accuracies.append(iou_accuracy)


        sat_format_rewards = []
        sat_length_rewards = []
        sat_accuracy_rewards = []
        sat_seg_iou_accuracies = []

        for response, bbox_text, pred_mask, gt_mask, gt_point in zip(sat_response_text_list, bbox_text_list, sat_pred_mask_list, gt_mask_list, gt_point_list):
            response = response.replace("<|endoftext|>", "").replace("<|im_end|>", "").replace("<pad>", "")
            
            format_reward = _multi_s2_format_reward(response, bbox_text)
            sat_format_rewards.append(format_reward)

            length_reward = _multi_s2_length_reward(response)
            sat_length_rewards.append(length_reward)

            gt_mask_np = np.array(gt_mask.convert("L"))
            accuracy_reward = _multi_s2_accuracy_reward(pred_mask, gt_mask_np)
            sat_accuracy_rewards.append(accuracy_reward)
            sat_seg_iou_accuracies.append(accuracy_reward)

        sat_format_rewards = torch.tensor(sat_format_rewards, dtype=torch.float16)
        sat_length_rewards = torch.tensor(sat_length_rewards, dtype=torch.float16)
        sat_accuracy_rewards = torch.tensor(sat_accuracy_rewards, dtype=torch.float16)
        sat_seg_iou_accuracies = torch.tensor(sat_seg_iou_accuracies, dtype=torch.float16)
        sat_sum_rewards = (
            sat_format_rewards + sat_length_rewards + sat_accuracy_rewards
        )

        map_format_rewards = torch.tensor(map_format_rewards, dtype=torch.float16)
        map_length_rewards = torch.tensor(map_length_rewards, dtype=torch.float16)
        map_accuracy_rewards = torch.tensor(map_accuracy_rewards, dtype=torch.float16)
        map_seg_iou_accuracies = torch.tensor(map_seg_iou_accuracies, dtype=torch.float16)
        map_sum_rewards = (
            map_format_rewards + map_length_rewards + map_accuracy_rewards
        )

        metrics = {
            "sat_format_reward_mean": sat_format_rewards.mean().item(),
            "sat_length_reward_mean": sat_length_rewards.mean().item(),
            "sat_accuracy_reward_mean": sat_accuracy_rewards.mean().item(),
            "sat_seg_iou_accuracy_mean": sat_seg_iou_accuracies.mean().item(),
            "map_format_reward_mean": map_format_rewards.mean().item(),
            "map_length_reward_mean": map_length_rewards.mean().item(),
            "map_accuracy_reward_mean": map_accuracy_rewards.mean().item(),
            "map_seg_iou_accuracy_mean": map_seg_iou_accuracies.mean().item(),
        }

        output = DataProto.from_dict(
            tensors={
                "seg_iou_rewards": sat_accuracy_rewards,
                "sat_response_level_rewards": sat_sum_rewards,
                "map_response_level_rewards": map_sum_rewards,
            },
            meta_info={"metrics": metrics}
        )
        
        return output
