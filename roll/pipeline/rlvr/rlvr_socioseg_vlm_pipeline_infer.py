import numpy as np
import json
import os
from typing import Any, Dict, List, Optional, Union

from PIL import Image, ImageDraw
import warnings
import ray
import torch
import cv2
import datasets
from collections import defaultdict
from transformers import ProcessorMixin, AutoConfig
from transformers.image_utils import load_images
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers import BatchFeature, ProcessorMixin
from transformers.data.data_collator import pad_without_fast_tokenizer_warning

from datasets import load_from_disk, load_dataset
from ray.util.timer import _Timer
from torch.utils.data import DataLoader
from tqdm import tqdm

from roll.datasets.collator import DataCollatorWithPaddingForMultiSeg
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.generate_scheduler import GenerateScheduler
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_processor_provider
from roll.pipeline.base_pipeline import BasePipeline
from roll.pipeline.rlvr.rlvr_config import SocioSegConfig
from roll.utils.checkpoint_manager import download_model
from roll.utils.constants import GENERATE_SCHEDULER_NAME, RAY_NAMESPACE
from roll.utils.functionals import (
    reduce_metrics,
    RunningMoments,
)
from roll.utils.kl_controller import get_kl_controller
from roll.utils.logging import get_logger

from roll.datasets.dataset import SocioSegDataset
from roll.pipeline.multi_utils import parse_points_text_from_content

logger = get_logger()

def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Computes Intersection over Union (IoU) for binary masks."""
    # Ensure masks are boolean
    pred_mask_bool = pred_mask > 0
    gt_mask_bool = gt_mask > 0
    
    intersection = np.logical_and(pred_mask_bool, gt_mask_bool).sum()
    union = np.logical_or(pred_mask_bool, gt_mask_bool).sum()
    
    if union == 0:
        # Both masks are empty, perfect match
        return 1.0
        
    return intersection / union

def compute_giou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Computes Generalized IoU (GIoU) for binary masks."""
    # Ensure masks are boolean
    pred_mask_bool = pred_mask > 0
    gt_mask_bool = gt_mask > 0

    intersection = np.logical_and(pred_mask_bool, gt_mask_bool).sum()
    union = np.logical_or(pred_mask_bool, gt_mask_bool).sum()
    
    if union == 0:
        return 1.0

    iou = intersection / union
    
    # Find the smallest enclosing box C
    all_points = np.argwhere(np.logical_or(pred_mask_bool, gt_mask_bool))
    if all_points.shape[0] == 0:
        return 1.0
        
    y_min, x_min = all_points.min(axis=0)
    y_max, x_max = all_points.max(axis=0)
    
    enclosing_area = (x_max - x_min + 1) * (y_max - y_min + 1)
    
    if enclosing_area == 0:
        return iou # Should not happen if union > 0
        
    return iou - (enclosing_area - union) / enclosing_area

def compute_ciou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Computes Complete IoU (CIoU) based on bounding boxes of masks."""
    # Convert boolean or 0/255 masks to uint8 0/1 for findContours
    pred_mask_u8 = (pred_mask > 0).astype(np.uint8)
    gt_mask_u8 = (gt_mask > 0).astype(np.uint8)

    pred_contours, _ = cv2.findContours(pred_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pred_box = cv2.boundingRect(np.concatenate(pred_contours)) if pred_contours else None

    gt_contours, _ = cv2.findContours(gt_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    gt_box = cv2.boundingRect(np.concatenate(gt_contours)) if gt_contours else None

    if pred_box is None and gt_box is None: return 1.0
    if pred_box is None or gt_box is None: return 0.0

    px, py, pw, ph = pred_box
    gx, gy, gw, gh = gt_box

    # Bounding Box IoU
    ix1, iy1 = max(px, gx), max(py, gy)
    ix2, iy2 = min(px + pw, gx + gw), min(py + ph, gy + gh)
    inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union_area = (pw * ph) + (gw * gh) - inter_area
    if union_area == 0: return 1.0 if inter_area > 0 else 0.0
    iou = inter_area / union_area

    # Enclosing Box C
    cx1, cy1 = min(px, gx), min(py, gy)
    cx2, cy2 = max(px + pw, gx + gw), max(py + ph, gy + gh)
    
    # Center distance penalty
    pcx, pcy = px + pw / 2, py + ph / 2
    gcx, gcy = gx + gw / 2, gy + gh / 2
    center_dist_sq = (pcx - gcx)**2 + (pcy - gcy)**2
    c_diag_sq = (cx2 - cx1)**2 + (cy2 - cy1)**2
    if c_diag_sq == 0: return iou
    
    # Aspect ratio penalty
    with warnings.catch_warnings(): # Suppress RuntimeWarning for division by zero
        warnings.simplefilter("ignore")
        v = (4 / (np.pi**2)) * ((np.arctan(gw / gh) - np.arctan(pw / ph))**2)
    
    alpha = v / ((1 - iou) + v + 1e-7)
    
    return iou - (center_dist_sq / c_diag_sq + alpha * v)


def format_prompt_1(prompt, processor, use_image=True, prompt_image_token=None):

    question_template = (
                        "You will be given two images. The first is a map and the second is a corresponding satellite image."
                        "Please find '{prompt}' with bboxs." \
                        "Compare the difference between object(s) and find the most closely matched object(s)." \
                        "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags. Please use English."  \
                        "Output the bbox(es) in JSON format." \
                        "i.e., <think>thinking process here </think>" \
                        "<answer>{answer}</answer>"
                    )
    answer = "[{\"bbox_2d\": [bx1,by1,bx2,by2]}, {\"bbox_2d\": [bx3,by3,bx4,by4]}]"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": question_template.format(prompt=prompt, answer=answer)},
            ]
            if use_image and not prompt_image_token
            else [
                {"type": "text", "text": question_template.format(prompt=prompt, answer=answer)},
            ],  # image_token has been included in prompt
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if prompt_image_token:
        text = text.replace(prompt_image_token, "<|vision_start|><|image_pad|><|vision_end|>")
    return text

def format_prompt_2(prompt, bboxs, processor, use_image=True, prompt_image_token=None):

    question_template = (
                        "You will be given two images. The first is a map and the second is a corresponding satellite image."
                        "Now some bbox(s) and the results after SAM segmentation for \"{prompt}\" have been rendered on these two images."
                        "The found bbox(s) are: {bboxs}."
                        "Please add some points appropriately to each bbox to better represent the area of interest."
                        "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
                        "i.e., <think> thinking process here </think>"
                        "<answer>{answer}</answer>"
                    )
    answer = "[{\"bbox_2d\": [bx1,by1,bx2,by2], \"points\": [[px1,py1],[px2,py2],[px3,py3]]}, {\"bbox_2d\": [bx3,by3,bx4,by4], \"points\": [[px4,py4],[px5,py5],[px6,py6]}]"
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "image"},
                {"type": "text", "text": question_template.format(prompt=prompt, bboxs=bboxs, answer=answer)},
            ]
            if use_image and not prompt_image_token
            else [
                {"type": "text", "text": question_template.format(prompt=prompt, bboxs=bboxs, answer=answer)},
            ],  # image_token has been included in prompt
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if prompt_image_token:
        text = text.replace(prompt_image_token, "<|vision_start|><|image_pad|><|vision_end|>")
    return text

def process_image(images: List[Image.Image], processor: ProcessorMixin):
    # same as qwen2-vl image processor
    image_processor = processor.image_processor
    factor = (
        image_processor.patch_size * image_processor.merge_size
        if "Qwen" in image_processor.image_processor_type
        else 28
    )
    def resize_image(image):
        height, width = image.height, image.width
        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=factor,
            min_pixels=image_processor.min_pixels,
            max_pixels=image_processor.max_pixels,
        )
        return image.resize((resized_width, resized_height), resample=image_processor.resample)
    return [resize_image(image) for image in images]

def count_components_opencv(image_list: List[Image.Image]) -> List[int]:
    counts = []
    for img in image_list:
        np_image = np.array(img)
        gray_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        _ , binary_mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        counts.append(num_labels - 1)
    return counts

def get_bboxes(image_list: List[Image.Image]) -> str:
    all_bboxes_list = []
    
    for img in image_list:
        np_image = np.array(img)
        
        if len(np_image.shape) == 2:
            gray_image = np_image
        else:
            gray_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        
        _ , binary_mask = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes_list = []
        for contour in contours:
            if cv2.contourArea(contour) > 10:
                x, y, w, h = cv2.boundingRect(contour)

                bbox_dict = {
                    "bbox_2d": list([x, y, x + w, y + h])
                }
            
                bboxes_list.append(bbox_dict)
        bboxes_list_str = json.dumps(bboxes_list)
        all_bboxes_list.append(bboxes_list_str)

    return all_bboxes_list

def encode_function(data_i, processor, id_key, prompt_key, label_key, image_map_key, image_sat_key):
    n = len(data_i[prompt_key])

    image_list = []
    image_flag = []
    for i in range(n):
        images = []
        ok = True
        for image in [data_i[image_map_key][i], data_i[image_sat_key][i]]:
            try:
                out = load_images([image], timeout=None)
                out = process_image(out, processor)
                images.append(out[0])
            except Exception:
                images.append(Image.new("RGB", (224, 224)))
                ok = False
        image_list.append(images)
        image_flag.append(ok)
    
    id_list = [data_i.get(id_key, [f"id_{i}"])[i] for i in range(n)]

    question_list = []
    for idx, instruct in enumerate(data_i[prompt_key]):
        question_list.append(instruct)
    
    sat_image_list = []
    map_image_list = []
    for sat_image, map_image in zip(data_i[image_sat_key], data_i[image_map_key]):
        try:
            sat_out = load_images(sat_image if isinstance(sat_image, (list, tuple)) else [sat_image], timeout=None)
            map_out = load_images(map_image if isinstance(map_image, (list, tuple)) else [map_image], timeout=None)
        except Exception:
            sat_out = [Image.new("RGB", (756,756))]
            map_out = [Image.new("RGB", (756,756))]
            
        sat_out = process_image(sat_out, processor)
        map_out = process_image(map_out, processor)
        
        sat_image_list.append(sat_out)
        map_image_list.append(map_out)
    
    try:
        label_out = load_images(data_i[label_key] if isinstance(data_i[label_key], (list, tuple)) else [data_i[label_key]], timeout=None)
        sat_seg_out = load_images(data_i[image_sat_key] if isinstance(data_i[image_sat_key], (list, tuple)) else [data_i[image_sat_key]], timeout=None)
    except Exception:
        label_out = [Image.new("RGB", (756,756))]
        sat_seg_out = [Image.new("RGB", (756,756))]

    object_gt = count_components_opencv(label_out)

    bbox_gt = get_bboxes(label_out)

    map_text_list = []
    for idx, instruct in enumerate(data_i[prompt_key]):
        text = format_prompt_1(instruct, processor, use_image=image_flag[idx], prompt_image_token=None)
        map_text_list.append(text)

    encodings = {
        "id": id_list,                # [n]
        "prompt_map": map_text_list,          # [n]
        "question": question_list,
        "gt_mask": label_out,       # [n]
        "gt_bbox": bbox_gt,             # [n]
        "gt_object": object_gt,          # [n]
        "image_sat": sat_image_list,             # [n]
        "image_map": map_image_list,             # [n]
        "seg_image": sat_seg_out,             # [n]
        "image": image_list,             # [n][2]
        "image_flag": image_flag,        # [n]
        "tag": [""] * n                  # [n]
    }
    return encodings


FILEEXT2TYPE = {
    "arrow": "arrow",
    "csv": "csv",
    "json": "json",
    "jsonl": "json",
    "parquet": "parquet",
    "txt": "text",
}


def get_dataset(data_args, encode_function, processor, features=None, get_eval=False):
    cache_path = getattr(data_args, "cache_path", None)
    if cache_path:
        cache_path = os.path.join(cache_path, "test")
    if cache_path and os.path.exists(cache_path):
        dataset = load_from_disk(cache_path)
        return dataset
    data_path = None
    data_name = data_args.file_name
    data_files = []
    dataset_dir = getattr(data_args, "dataset_dir", ".")
    local_path: str = os.path.join(dataset_dir, data_name)

    # if load from local
    # dataset_builder = SocioSegDataset()
    # test_path = os.path.join(local_path, "test")
    # dataset = datasets.Dataset.from_generator(
    #     dataset_builder._generate_examples,
    #     gen_kwargs={"data_dir": test_path},
    #     features=dataset_builder.info.features
    # )

    # if load from huggingface
    dataset = load_dataset("vvangfaye/SocioSeg")["test"]

    remove_columns = list(dataset.features.keys() - features.keys())

    id_key = getattr(data_args, "id") if getattr(data_args, "id", None) else "id"
    prompt_key = getattr(data_args, "prompt") if getattr(data_args, "prompt", None) else "problem"
    label_key = getattr(data_args, "mask_label") if getattr(data_args, "mask_label", None) else "mask_label"
    image_map_key = getattr(data_args, "map_image") if getattr(data_args, "map_image", None) else "map_image"
    image_sat_key = getattr(data_args, "sat_image") if getattr(data_args, "sat_image", None) else "sat_image"
    print(f"Begin : {dataset}")
    dataset = dataset.map(
        lambda data: encode_function(data, processor, id_key, prompt_key, label_key, image_map_key, image_sat_key),
        batched=True,
        batch_size=100,
        num_proc=32,
        features=features,
        remove_columns=remove_columns,
        desc="Encoding dataset",
    )
    print(f"Encoding: {dataset}")
    if cache_path:
        dataset.save_to_disk(cache_path)
    return dataset


def get_dataloader(dataset, batch_size, data_collator):
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,  # larger shm for bigger num_workers
        collate_fn=data_collator,
    )
    return dataloader


def get_extra_data_provider(model_name_or_path: str, processor=None):
    model_name_or_path = download_model(model_name_or_path)
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    if "qwen2" in config.model_type:
        import types

        from transformers import BatchFeature  # help define a object to accesss attr
        from transformers.models.qwen2_vl import Qwen2VLForConditionalGeneration, Qwen2VLModel

        dummy_self = BatchFeature(
            {
                "config": BatchFeature(
                    {
                        "vision_config": BatchFeature({"spatial_merge_size": processor.image_processor.merge_size}),
                        "image_token_id": processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"),
                        "video_token_id": processor.tokenizer.convert_tokens_to_ids("<|video_pad|>"),
                        "vision_start_token_id": processor.tokenizer.convert_tokens_to_ids("<|vision_start|>"),
                    }
                )
            }
        )
        if hasattr(Qwen2VLForConditionalGeneration, "get_rope_index"):
            get_rope_index = types.MethodType(Qwen2VLForConditionalGeneration.get_rope_index, dummy_self)
        else:
            get_rope_index = types.MethodType(Qwen2VLModel.get_rope_index, dummy_self)

        def extra_data_provider(
            input_ids: torch.LongTensor,
            image_grid_thw: Optional[torch.LongTensor] = None,
            video_grid_thw: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
        ):
            rope_index = get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask)[0]
            # (3, bsz, seqlen) -> (bsz, 3, seqlen) to put it into DataProto,
            # transpose it batck to (3, bsz, seqlen) before forward for model
            rope_index = rope_index.transpose(0, 1)
            return {"position_ids": rope_index}

        return extra_data_provider

    def default_extra_data_provider(
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        bsz, seqlen = input_ids.shape
        position_ids = torch.arange(seqlen, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(bsz, -1)
        if attention_mask is not None:
            position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        return {"position_ids": position_ids}

    return default_extra_data_provider

def render_image(
    bboxes_json: str,
    images: List[Image.Image],
    mask: Union[np.ndarray, Image.Image]
) -> List[Image.Image]:
    rendered_images = []
    processed_mask_overlay = None
    try:
        if isinstance(mask, Image.Image):
            mask_array = np.array(mask.convert('L'))
        else:
            mask_array = np.array(mask)

        if images:
            first_image_width, first_image_height = images[0].size
            overlay_np = np.zeros((first_image_height, first_image_width, 4), dtype=np.uint8)
            mask_array = cv2.resize(mask_array, (first_image_width, first_image_height), interpolation=cv2.INTER_NEAREST)
            mask_array = mask_array > 0
            alpha_value = int(255 * 0.4)
            mask_color = [255, 0, 0, alpha_value]
            overlay_np[mask_array] = mask_color
            processed_mask_overlay = Image.fromarray(overlay_np, 'RGBA')
        else:
            print("warning: images is empty")

    except Exception as e:
        print(f"error: {e}")
        processed_mask_overlay = None

    bboxes = []
    try:
        bbox_data: List[Dict[str, Any]] = json.loads(bboxes_json)
        if isinstance(bbox_data, list):
            for item in bbox_data:
                if isinstance(item, dict) and 'bbox_2d' in item and len(item['bbox_2d']) == 4:
                    bboxes.append(item['bbox_2d'])
                else:
                    print(f"warning: item is not a dict or bbox_2d is not in item")
    except (json.JSONDecodeError, TypeError) as e:
        print(f"error: {e}")
        bboxes = []

    for i, image in enumerate(images):
        current_rendered_image = image.copy().convert("RGBA")
        if bboxes:
            draw = ImageDraw.Draw(current_rendered_image)
            for bbox in bboxes:
                if len(bbox) != 4:
                    continue
                try:
                    shape = [(bbox[0], bbox[1]), (bbox[2], bbox[3])]
                    draw.rectangle(shape, outline="blue", width=2)
                except Exception as e:
                    continue

        if processed_mask_overlay:
            try:
                if current_rendered_image.size != processed_mask_overlay.size:
                     resized_mask = processed_mask_overlay.resize(current_rendered_image.size, Image.Resampling.LANCZOS)
                     current_rendered_image = Image.alpha_composite(current_rendered_image, resized_mask)
                else:
                     current_rendered_image = Image.alpha_composite(current_rendered_image, processed_mask_overlay)
            except ValueError as e:
                print(f"error: {e}")

        final_image = current_rendered_image.convert("RGB")
        rendered_images.append(final_image)

    return rendered_images

def draw_visual_prompt(image: Image.Image, mask: Union[np.ndarray, Image.Image], visual_prompt) -> Image.Image:

    prompt_data = visual_prompt

    rendered_image = image.copy().convert("RGBA")
    width, height = rendered_image.size

    try:
        if isinstance(mask, Image.Image):
            mask_array = np.array(mask.convert('L'))
        else:
            mask_array = np.array(mask)

        overlay_np = np.zeros((height, width, 4), dtype=np.uint8)
        
        if mask_array.shape[0] != height or mask_array.shape[1] != width:
            mask_array = cv2.resize(mask_array, (width, height), interpolation=cv2.INTER_NEAREST)
        
        bool_mask = mask_array > 0
        
        alpha_value = int(255 * 0.4)
        mask_color = [255, 0, 0, alpha_value]  # R, G, B, A
        
        overlay_np[bool_mask] = mask_color
        mask_overlay = Image.fromarray(overlay_np, 'RGBA')
        rendered_image = Image.alpha_composite(rendered_image, mask_overlay)

    except Exception as e:
        print(f"error: {e}")

    draw = ImageDraw.Draw(rendered_image)

    if "box" in prompt_data:
        bbox = prompt_data["box"]
        bbox = bbox.tolist()
        if isinstance(bbox, list) and len(bbox) == 4:
            shape = [(bbox[0], bbox[1]), (bbox[2], bbox[3])]
            draw.rectangle(shape, outline="blue", width=2)

    if "point_coords" in prompt_data and "point_labels" in prompt_data:
        points = prompt_data.get("point_coords", [])
        labels = prompt_data.get("point_labels", [])
        radius = 5

        for point, label in zip(points, labels):
            point = point.tolist()
            if isinstance(point, list) and len(point) == 2:
                x, y = point
                point_bbox = [x - radius, y - radius, x + radius, y + radius]
                if label == 1:
                    fill_color = "green"
                else:
                    fill_color = "red"
                draw.ellipse(point_bbox, fill=fill_color, outline=None)
    return rendered_image.convert("RGB")    

class SocioSegInferPipeline(BasePipeline):
    def __init__(self, pipeline_config: SocioSegConfig):
        super().__init__(pipeline_config)
        self.pipeline_config = pipeline_config

        self.processor = default_processor_provider(self.pipeline_config.actor_train.model_args)
        # set max_pixels to avoid image token num is larger than prompt length
        self.processor.image_processor.max_pixels, self.processor.image_processor.min_pixels = (
            getattr(self.pipeline_config.actor_train.model_args, "max_pixels", 768 * 768),
            getattr(self.pipeline_config.actor_train.model_args, "min_pixels", 56 * 56),
        )
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"
        # regularized data filed
        features = datasets.Features(
            {
                # only support single image temporarily since sglang usage
                "id": datasets.Value("string"),
                "prompt_map": datasets.Value("string"),
                "question": datasets.Value("string"),
                "gt_mask": datasets.Image(decode=True),
                "seg_image": datasets.Image(decode=True),  # sat_image segmentation
                "gt_object": datasets.Value("int32"),  # number of objects in ground truth
                "gt_bbox": datasets.Value("string"),  # {"bbox_2d": [x1, y1, w1, h1]},
                "image_sat": datasets.Sequence(datasets.Image(decode=True)),
                "image_map": datasets.Sequence(datasets.Image(decode=True)),
                "image": datasets.Sequence(datasets.Image(decode=True)),  # [sat_image, map_image]
                # for text and multi-modal mixed data usage, indicating valid image
                "image_flag": datasets.Value("bool"),
                # for area seperated validation, dummy currently
                "tag": datasets.Value("string"),
            }
        )
        dataset = get_dataset(
            self.pipeline_config.actor_train.data_args, encode_function, self.processor, features, get_eval=False
        )
            
        self.extra_data_provider = get_extra_data_provider(
            self.pipeline_config.actor_train.model_args.model_name_or_path, processor=self.processor
        )
        data_collator = DataCollatorWithPaddingForMultiSeg(
            tokenizer=self.tokenizer,
            processor=self.processor,
            extra_data_provider=self.extra_data_provider,
            max_length=self.pipeline_config.prompt_length,
            image_key="image",
            padding="max_length",
            gt_object_key="gt_object",
            gt_bbox_key="gt_bbox",
            
        )
        self.dataloader = get_dataloader(dataset, self.pipeline_config.rollout_batch_size, data_collator)
        
        max_steps = len(self.dataloader) * self.pipeline_config.actor_train.training_args.num_train_epochs
        self.pipeline_config.set_max_steps(max_steps=max_steps)

        self.seg_infer: Any = Cluster(
            name=self.pipeline_config.seg_infer.name,
            worker_cls=self.pipeline_config.seg_infer.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.seg_infer,
        )

        self.actor_train: Any = Cluster(
            name="actor_train_actor",
            worker_cls=self.pipeline_config.actor_train.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_train,
        )

        self.actor_infer: Any = Cluster(
            name="actor_infer_actor",
            worker_cls=self.pipeline_config.actor_infer.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_infer,
        )
        

        self.generate_scheduler = GenerateScheduler.options(
            name=f"{GENERATE_SCHEDULER_NAME}_{self.actor_infer.cluster_name}",
            get_if_exists=True,
            namespace=RAY_NAMESPACE,
        ).remote()
        
        refs: List[ray.ObjectRef] = []
        refs.extend(self.actor_train.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)
        
        refs = []
        refs.extend(self.actor_infer.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)
        
        refs = []
        refs.extend(self.seg_infer.initialize(pipeline_config=self.pipeline_config, blocking=False, tokenizer=self.tokenizer))
        ray.get(refs)
        
        # TODO: if don't define actor_train and set pair, the actor_infer will output garbled text. why?
        self.set_model_update_pair(
            src_cluster=self.actor_train,
            tgt_cluster=self.actor_infer,
            frequency=self.pipeline_config.actor_train.model_update_frequency,
        )

        self.running = RunningMoments()

    @torch.no_grad()
    def run(self):
        global_step = 0

        tps_timer = _Timer(window_size=5)
        actor_infer_timer1 = _Timer(window_size=5)
        actor_infer_response_timer1 = _Timer(window_size=5)
        actor_infer_timer2 = _Timer(window_size=5)
        actor_infer_response_timer2 = _Timer(window_size=5)
        seg_infer_timer = _Timer(window_size=5)
        actor_train_timer = _Timer(window_size=5)

        all_iou_acc = []
        all_ciou_acc = []
        all_giou_acc = []
        for batch_dict in tqdm(self.dataloader):
            metrics = {}
            with tps_timer:

                
                model_update_metrics: Dict = self.model_update(global_step)
                metrics.update(model_update_metrics)

                batch_dict: Dict
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                batch.meta_info = {
                    "global_step": global_step,
                    # mark here to make megatron get_data_input broadcast with non_batch_tensor
                    "_broadcast_non_tensor_batch": True,
                }

                with actor_infer_timer1, actor_infer_response_timer1:
                    # donot support hf/deepspeed infer generate which use
                    # multi_modal_inputs tensors
                    gen_batch = batch.pop(
                        batch_keys=["map_input_ids", "map_attention_mask", "map_position_ids"],
                        non_tensor_batch_keys=(
                            ["multi_modal_map_data"] if "multi_modal_map_data" in batch.non_tensor_batch else []
                        ),
                    )
                    gen_batch.rename("map_input_ids", "input_ids")
                    gen_batch.rename("map_attention_mask", "attention_mask")
                    gen_batch.rename("map_position_ids", "position_ids")
                    gen_batch.non_tensor_batch["multi_modal_data"] = gen_batch.non_tensor_batch.pop("multi_modal_map_data")

                    # gen_batch = batch.pop(
                    #     batch_keys=["input_ids", "attention_mask", "position_ids"],
                    #     non_tensor_batch_keys=(
                    #         ["multi_modal_data"] if "multi_modal_data" in batch.non_tensor_batch else []
                    #     ),
                    # )
                    
                    gen_batch.meta_info = {"global_step": global_step}
                    gen_batch.meta_info["response_callback_fn"] = self.generate_scheduler.report_response.remote
                    generate_output: DataProto = ray.get(
                        self.generate_scheduler.generate.remote(
                            data=gen_batch,
                            actor_cluster=self.actor_infer,
                            pipeline_config=self.pipeline_config,
                        ),
                        timeout=self.pipeline_config.rpc_timeout,
                    )
                    metrics.update(reduce_metrics(generate_output.meta_info.pop("metrics", {})))

                # generate_output is repeated by num_return_sequences, thus
                # reset batch.batch before union to make batch size same,
                generate_output.rename(old_keys="input_ids", new_keys="map_input_ids")
                generate_output.rename(old_keys="attention_mask", new_keys="map_attention_mask")
                generate_output.rename(old_keys="position_ids", new_keys="map_position_ids")
                generate_output.rename(old_keys="responses", new_keys="map_responses")
                generate_output.rename(old_keys="response_mask", new_keys="map_response_mask")
                generate_output.rename(old_keys="prompts", new_keys="map_prompts")
                generate_output.rename(old_keys="prompt_mask", new_keys="map_prompt_mask")

                # repeat num_return_sequences for fields not in gen_batch
                # which has been repeated in generate_scheduler
                # breakpoint()
                for key, value in batch.non_tensor_batch.items():
                    batch.non_tensor_batch[key] = np.repeat(
                        value, self.actor_infer.worker_config.generating_args.num_return_sequences
                    )
                    
                batch.batch = generate_output.batch
                batch = batch.union(generate_output)

                with seg_infer_timer:
                    seg_batch = batch.pop(
                        batch_keys=["map_responses", "map_prompts"],
                        non_tensor_batch_keys=['seg_image']
                    )
                    seg_batch_refs: List[ray.ObjectRef] = self.seg_infer.segment_v4_map(seg_batch, blocking=False)
                
                seg_batch_out: DataProto = DataProto.materialize_concat(data_refs=seg_batch_refs)
                batch = batch.union(seg_batch_out)
                
                batch.non_tensor_batch["map_mask"] = batch.non_tensor_batch.pop("mask")
                batch.non_tensor_batch["map_visual_prompt"] = batch.non_tensor_batch.pop("visual_prompt")
                batch.non_tensor_batch.pop("response_text")
                
                # breakpoint()
                response_list = self.tokenizer.batch_decode(batch.batch["map_responses"], skip_special_tokens=False)
                bboxs_text_list = []
                for response in response_list:            
                    bboxs_text = parse_points_text_from_content(response)
                    bboxs_text_list.append(bboxs_text)
                
                sat_text_list = []
                for instruct, points in zip(batch.non_tensor_batch["question"], bboxs_text_list):
                    text = format_prompt_2(instruct, points, self.processor)
                    sat_text_list.append(text)
                    
                # This section is rewritten to mimic the second snippet's iterative approach.

                sat_padded_features = defaultdict(list)
                un_padded_features = defaultdict(list)
                mm_feature_keys = set()

                # Iterate over each sample, analogous to `for feature in features:` in the reference snippet.
                zipped_features = zip(
                    sat_text_list,
                    bboxs_text_list,
                    batch.non_tensor_batch["image"],
                    batch.non_tensor_batch["map_mask"],
                )

                for text, bboxs_text, image_sat, mask in zipped_features:
                    rd_image = render_image(bboxs_text, image_sat, mask)
                    
                    # Call the processor for each sample individually.
                    sat_model_inputs: BatchFeature = self.processor(
                        images=rd_image,
                        text=text,
                    )
                    # remove non-tensor feature, e.g. tbstars2_moe_vista has sat_prompt in processor output
                    for key in ["prompt_sat"]:
                        if key in sat_model_inputs:
                            sat_model_inputs.pop(key)
                            
                    padded_keys = ["input_ids", "attention_mask", "labels"]
                    # Collect features that require padding into separate lists.
                    for key in filter(lambda k: k in sat_model_inputs, padded_keys):
                        sat_padded_features[key].append(sat_model_inputs.pop(key)[0])

                    # mm feature fileds can be different because of mixed data
                    mm_feature_keys = mm_feature_keys.union(sat_model_inputs.keys())
                    
                    # to tensors except padded_keys which would be converted after padding
                    sat_model_inputs.convert_to_tensors(tensor_type='pt')

                    # allow mixed text and multi-modal data
                    # tensors in multi_modal_inputs dict have bsz=1 and should be
                    # concat at dim=0 before model forward
                    un_padded_features["multi_modal_sat_inputs"].append(dict(sat_model_inputs))
                    
                    # inputs for infer engine, not tensors
                    un_padded_features["multi_modal_sat_data"].append(
                        {
                            "prompt_token_ids":  # different with input_ids
                            self.tokenizer.encode(text, add_special_tokens=False),
                            "multi_modal_data": {
                                "image": [rd_image] if not isinstance(rd_image, list) else rd_image,
                            },
                        }
                        )

                # Pad the collected text-based features together into a single batch.
                sat_batch = pad_without_fast_tokenizer_warning(
                    self.tokenizer,
                    sat_padded_features,
                    padding='max_length',
                    max_length=self.pipeline_config.prompt_length,
                    pad_to_multiple_of=None,
                    return_tensors='pt',
                )
                sat_batch.update(un_padded_features)

                # The logic for handling extra data provider remains, now working with the constructed batch.
                sat_fun_params = ['input_ids', 'attention_mask', 'image_grid_thw']
                sat_kwargs = {}
                for key in sat_fun_params:
                    if key in sat_batch:
                        sat_kwargs[key] = sat_batch[key]
                    elif key in mm_feature_keys:
                        mm_inputs = [inputs[key] for inputs in sat_batch["multi_modal_sat_inputs"] if key in inputs]
                        if mm_inputs:
                            sat_kwargs[key] = torch.concat(mm_inputs, dim=0)
                        else:
                            # Replicating the original code's behavior to fail if a required key is missing.
                            print(f"Warning: {key} not found in any multi-modal inputs, using default value.")
                            exit()
                    else:
                        print(f"Warning: {key} not found in batch, using default value.")
                        exit()

                sat_extra_data = self.extra_data_provider(**sat_kwargs)
                sat_extra_data['position_ids'] = sat_extra_data.pop('position_ids')
                sat_extra_data['attention_mask'] = sat_kwargs.pop('attention_mask')
                sat_extra_data['input_ids'] = sat_kwargs.pop('input_ids')
                sat_batch.update(sat_extra_data)

                sat_batch['bboxs_text'] = bboxs_text_list
                
                for key in sat_batch:
                    if isinstance(sat_batch[key], (torch.Tensor, np.ndarray)):
                        assert sat_batch[key].shape[0] == sat_batch["input_ids"].shape[0]
                    else:
                        assert len(sat_batch[key]) == sat_batch["input_ids"].shape[0]
                        sat_batch[key] = np.array(sat_batch[key], dtype=object)

                sat_batch: DataProto = DataProto.from_single_dict(sat_batch)
                
                batch = batch.union(sat_batch)

                # infer 2
                with actor_infer_timer2, actor_infer_response_timer2:
                    # donot support hf/deepspeed infer generate which use
                    # multi_modal_inputs tensors
                    gen_batch = batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=(
                            ["multi_modal_sat_data"] if "multi_modal_sat_data" in batch.non_tensor_batch else []
                        ),
                    )
                    gen_batch.non_tensor_batch["multi_modal_data"] = gen_batch.non_tensor_batch.pop("multi_modal_sat_data")
                    ori_num_return_sequences = self.pipeline_config.actor_infer.generating_args.num_return_sequences
                    self.pipeline_config.actor_infer.generating_args.num_return_sequences = 1
                    gen_batch.meta_info = {"global_step": global_step}
                    gen_batch.meta_info["response_callback_fn"] = self.generate_scheduler.report_response.remote
                    generate_output: DataProto = ray.get(
                        self.generate_scheduler.generate.remote(
                            data=gen_batch,
                            actor_cluster=self.actor_infer,
                            pipeline_config=self.pipeline_config,
                        ),
                        timeout=self.pipeline_config.rpc_timeout,
                    )
                    self.pipeline_config.actor_infer.generating_args.num_return_sequences = ori_num_return_sequences
                    metrics.update(reduce_metrics(generate_output.meta_info.pop("metrics", {})))

                # no rename(for faster)
    
                batch = batch.union(generate_output)
                    
                with seg_infer_timer:
                    seg_batch = batch.pop(
                        batch_keys=["responses", "prompts"],
                        non_tensor_batch_keys=['seg_image']
                    )
                    seg_batch_refs: List[ray.ObjectRef] = self.seg_infer.segment_v4_sat(seg_batch, blocking=False)
                
                seg_batch_out: DataProto = DataProto.materialize_concat(data_refs=seg_batch_refs)
                seg_batch_out.meta_info.pop("metrics")
                batch = batch.union(seg_batch_out)
                batch.non_tensor_batch["sat_mask"] = batch.non_tensor_batch.pop("mask")
                batch.non_tensor_batch["sat_visual_prompt"] = batch.non_tensor_batch.pop("visual_prompt")

                iou_list, ciou_list, giou_list = [], [], []
                map_response_list = self.tokenizer.batch_decode(batch.batch["map_responses"], skip_special_tokens=False)
                sat_response_list = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=False)
                for i in range(len(batch)):
                    gt_mask = np.array(batch[i].non_tensor_batch["gt_mask"].convert("L"))

                    iou = compute_iou(
                        batch[i].non_tensor_batch["sat_mask"],
                        gt_mask
                    )
                    ciou = compute_ciou(
                        batch[i].non_tensor_batch["sat_mask"],
                        gt_mask
                    )
                    giou = compute_giou(
                        batch[i].non_tensor_batch["sat_mask"],
                        gt_mask
                    )

                    iou_list.append(iou)
                    ciou_list.append(ciou)
                    giou_list.append(giou)

                    try:
                        map_visual_prompt = batch[i].non_tensor_batch["map_visual_prompt"][0]
                        sat_visual_prompt = batch[i].non_tensor_batch["sat_visual_prompt"][0]
                    except:
                        map_visual_prompt = {}
                        sat_visual_prompt = {}
                    image_s1 = draw_visual_prompt(batch[i].non_tensor_batch["seg_image"], batch[i].non_tensor_batch["map_mask"], map_visual_prompt)
                    image_s2 = draw_visual_prompt(batch[i].non_tensor_batch["seg_image"], batch[i].non_tensor_batch["sat_mask"], sat_visual_prompt)
                    mask_s1 = batch[i].non_tensor_batch["map_mask"]
                    mask_s2 = batch[i].non_tensor_batch["sat_mask"]
                    save_id = batch[i].non_tensor_batch["id"]
                    save_dir1 = f".output/infer/result/stage1/"
                    save_dir2 = f".output/infer/result/stage2/"
                    save_dir3 = f".output/infer/result/render1/"
                    save_dir4 = f".output/infer/result/render2/"
                    os.makedirs(save_dir1, exist_ok=True)
                    os.makedirs(save_dir2, exist_ok=True)
                    os.makedirs(save_dir3, exist_ok=True)
                    os.makedirs(save_dir4, exist_ok=True)
                    # mask is numpy array
                    mask_s1 = mask_s1.astype(np.uint8) * 255
                    mask_s2 = mask_s2.astype(np.uint8) * 255
                    cv2.imwrite(f"{save_dir1}/{save_id}.png", mask_s1)
                    cv2.imwrite(f"{save_dir2}/{save_id}.png", mask_s2)
                    image_s1.save(f"{save_dir3}/{save_id}.png")
                    image_s2.save(f"{save_dir4}/{save_id}.png")
                    
                    # save response
                    with open(f"{save_dir1}/{save_id}.txt", "w") as f:
                        f.write(map_response_list[i])
                    with open(f"{save_dir2}/{save_id}.txt", "w") as f:
                        f.write(sat_response_list[i])

                print(f"iou_acc: {np.mean(iou_list)}, ciou_acc: {np.mean(ciou_list)}, giou_acc: {np.mean(giou_list)}")

                all_iou_acc.extend(iou_list)
                all_ciou_acc.extend(ciou_list)
                all_giou_acc.extend(giou_list)
        print(f"iou_acc: {np.mean(all_iou_acc)}, ciou_acc: {np.mean(all_ciou_acc)}, giou_acc: {np.mean(all_giou_acc)}")
        with open("iou_acc.txt", "w") as f:
            f.write(f"iou_acc: {np.mean(all_iou_acc)}, ciou_acc: {np.mean(all_ciou_acc)}, giou_acc: {np.mean(all_giou_acc)}")

    

    