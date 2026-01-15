import numpy as np
import json
import os
from typing import Any, Dict, List, Optional

import ray
import torch
import cv2
import datasets
import PIL.Image as Image
from collections import defaultdict
from transformers import ProcessorMixin, AutoConfig
from transformers.image_utils import load_images
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
from transformers import BatchFeature, PreTrainedTokenizerBase, ProcessorMixin
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from transformers.utils import PaddingStrategy

from datasets import load_dataset, load_from_disk
from codetiming import Timer
from ray.util.timer import _Timer
from torch.utils.data import DataLoader
from tqdm import tqdm

from roll.datasets.collator import DataCollatorWithPaddingForMultiSeg
from roll.distributed.executor.cluster import Cluster
from roll.distributed.scheduler.generate_scheduler import GenerateScheduler
from roll.distributed.scheduler.protocol import DataProto
from roll.models.model_providers import default_processor_provider
from roll.pipeline.base_pipeline import BasePipeline
from roll.pipeline.rlvr.aoireason_config import AOIReasonConfig
from roll.utils.checkpoint_manager import download_model
from roll.utils.constants import GENERATE_SCHEDULER_NAME, RAY_NAMESPACE
from roll.utils.functionals import (
    apply_kl_penalty,
    compute_advantage,
    reduce_metrics,
    masked_mean,
    RunningMoments,
    compute_clip_fraction,
    group_reward_norm,
    expand_to_token_level,
)
from roll.utils.kl_controller import get_kl_controller
from roll.utils.logging import get_logger

from roll.datasets.dataset import AOIReasonDataset
from roll.pipeline.multi_utils import parse_points_text_from_content

logger = get_logger()


def format_prompt(prompt, processor, use_image=True, prompt_image_token=None):
    question_template1 = (
                        "You are an expert in geospatial analysis. You will be given two images: "
                        "the first is a map and the second is a corresponding satellite image. \n"
                        "Your task is finding some points to locate the areas of interest {prompt} on the map and satellite image.\n"
                        "Please provide your reasoning process within <think></think> tags and the final list of points in "
                        "<answer></answer> tags using JSON format.\n"
                        "i.e., <answer>[{\"points\": [[25,115], [35, 102], [31, 32]]}, {\"points\": [[30,110], [40, 100]]}]</answer>"
                    )
    question_template2 = (
                        "You will be given two images. The first is a map and the second is a corresponding satellite image."
                        "Please find \"{prompt}\" with points."
                        "Compare the difference between object(s) and find the most closely matched object(s)."
                        "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
                        "Output the point(s) inside the interested object(s) in JSON format."
                        "i.e., <think> thinking process here </think>"
                        "<answer>{answer}</answer>"
                    )
    answer = "[{\"point_2d\": [[30,110], [40, 100]]}, {\"point_2d\": [[302,410], [402, 410]]}]"
    question_template3 = (
                        "You will be given two images. The first is a map and the second is a corresponding satellite image."
                        "Please find \"{prompt}\" with point set(s)."
                        "Compare the difference between object(s) and find the most closely matched object(s)."
                        "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
                        "For each set, please output 1~5 points [x, y], each enclosed in [], and finally, enclose all sets in [] (with a total of three layers of [] nesting)."
                        "i.e., <think> thinking process here </think>"
                        "<answer>{answer}</answer>"
                    )
    answer = "[[[x11,y11], [x12, y12], [x13, y13], [x14, y14]], [[x21,y21], [x22, y22], [x13, y13]]]"
    
    question_template = question_template3
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

def format_prompt_sat(prompt, points, processor, use_image=True, prompt_image_token=None):
    question_template1 = (
                        "You are an expert in geospatial analysis. You will be given two images: "
                        "the first is a map and the second is a corresponding satellite image. \n"
                        "Your task is finding some points to locate the areas of interest {prompt} on the map and satellite image.\n"
                        "Please provide your reasoning process within <think></think> tags and the final list of points in "
                        "<answer></answer> tags using JSON format.\n"
                        "i.e., <answer>[{\"points\": [[25,115], [35, 102], [31, 32]]}, {\"points\": [[30,110], [40, 100]]}]</answer>"
                    )
    question_template2 = (
                        "You will be given two images. The first is a map and the second is a corresponding satellite image."
                        "Please find \"{prompt}\" with points."
                        "Compare the difference between object(s) and find the most closely matched object(s)."
                        "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
                        "Output the point(s) inside the interested object(s) in JSON format."
                        "i.e., <think> thinking process here </think>"
                        "<answer>{answer}</answer>"
                    )
    answer = "[{\"point_2d\": [[30,110], [40, 100]]}, {\"point_2d\": [[302,410], [402, 410]]}]"
    question_template3 = (
                        "You will be given a satellite image."
                        "Now some point sets for \"{prompt}\" have been found on the map. Please add, delete, or modify the point sets on the satellite image to better represent the area of interest."
                        "The found point sets are: {points}."
                        "Compare the difference between object(s) and find the most closely matched object(s)."
                        "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
                        "For each set, please output 1~5 points [x, y], each enclosed in [], and finally, enclose all sets in [] (with a total of three layers of [] nesting)."
                        "i.e., <think> thinking process here </think>"
                        "<answer>{answer}</answer>"
                    )
    answer = "[[[x11,y11], [x12, y12], [x13, y13], [x14, y14]], [[x21,y21], [x22, y22], [x13, y13]]]"
    
    question_template = question_template3
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question_template.format(prompt=prompt, points=points, answer=answer)},
            ]
            if use_image and not prompt_image_token
            else [
                {"type": "text", "text": question_template.format(prompt=prompt, points=points, answer=answer)},
            ],  # image_token has been included in prompt
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if prompt_image_token:
        text = text.replace(prompt_image_token, "<|vision_start|><|image_pad|><|vision_end|>")
    return text

def format_prompt_map(prompt, processor, use_image=True, prompt_image_token=None):
    question_template1 = (
                        "You are an expert in geospatial analysis. You will be given two images: "
                        "the first is a map and the second is a corresponding satellite image. \n"
                        "Your task is finding some points to locate the areas of interest {prompt} on the map and satellite image.\n"
                        "Please provide your reasoning process within <think></think> tags and the final list of points in "
                        "<answer></answer> tags using JSON format.\n"
                        "i.e., <answer>[{\"points\": [[25,115], [35, 102], [31, 32]]}, {\"points\": [[30,110], [40, 100]]}]</answer>"
                    )
    question_template2 = (
                        "You will be given two images. The first is a map and the second is a corresponding satellite image."
                        "Please find \"{prompt}\" with points."
                        "Compare the difference between object(s) and find the most closely matched object(s)."
                        "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
                        "Output the point(s) inside the interested object(s) in JSON format."
                        "i.e., <think> thinking process here </think>"
                        "<answer>{answer}</answer>"
                    )
    answer = "[{\"point_2d\": [[30,110], [40, 100]]}, {\"point_2d\": [[302,410], [402, 410]]}]"
    question_template3 = (
                        "You will be given a map image."
                        "Please find \"{prompt}\" with point set(s)."
                        "Compare the difference between object(s) and find the most closely matched object(s)."
                        "Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
                        "For each set, please output 1~5 points [x, y], each enclosed in [], and finally, enclose all sets in [] (with a total of three layers of [] nesting)."
                        "i.e., <think> thinking process here </think>"
                        "<answer>{answer}</answer>"
                    )
    answer = "[[[x11,y11], [x12, y12], [x13, y13], [x14, y14]], [[x21,y21], [x22, y22], [x13, y13]]]"
    
    question_template = question_template3
    messages = [
        {
            "role": "user",
            "content": [
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
        image_list.append(images)  # 保证每个元素都是两个图
        image_flag.append(ok)
    
    id_list = [data_i.get(id_key, [f"id_{i}"])[i] for i in range(n)]
    
    text_list = []
    for idx, instruct in enumerate(data_i[prompt_key]):
        text = format_prompt(instruct, processor, use_image=image_flag[idx], prompt_image_token=None)
        text_list.append(text)

    map_text_list = []
    for idx, instruct in enumerate(data_i[prompt_key]):
        text = format_prompt_map(instruct, processor, use_image=image_flag[idx], prompt_image_token=None)
        map_text_list.append(text)

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
            sat_out = [Image.new("RGB", (224,224))]
            map_out = [Image.new("RGB", (224,224))]
            
        sat_out = process_image(sat_out, processor)
        map_out = process_image(map_out, processor)
        
        sat_image_list.append(sat_out)
        map_image_list.append(map_out)
    
    try:
        label_out = load_images(data_i[label_key] if isinstance(data_i[label_key], (list, tuple)) else [data_i[label_key]], timeout=None)
        sat_seg_out = load_images(data_i[image_sat_key] if isinstance(data_i[image_sat_key], (list, tuple)) else [data_i[image_sat_key]], timeout=None)
    except Exception:
        label_out = [Image.new("RGB", (224,224))]
        sat_seg_out = [Image.new("RGB", (224,224))]

    object_gt = count_components_opencv(label_out)

    encodings = {
        "id": id_list,                # [n]
        "prompt": text_list,          # [n]
        "prompt_map": map_text_list,          # [n]
        "question": question_list,
        "gt_mask": label_out,       # [n]
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
        cache_path = os.path.join(cache_path, "val" if get_eval else "train")
    if cache_path and os.path.exists(cache_path):
        dataset = load_from_disk(cache_path)
        return dataset
    data_path = None
    data_name = data_args.file_name
    data_files = []
    dataset_dir = getattr(data_args, "dataset_dir", ".")
    local_path: str = os.path.join(dataset_dir, data_name)
    dataset_builder = AOIReasonDataset()
    train_path = os.path.join(local_path, "train")
    dataset = datasets.Dataset.from_generator(
        dataset_builder._generate_examples,
        gen_kwargs={"data_dir": train_path},
        features=dataset_builder.info.features
    )
    remove_columns = list(dataset.features.keys() - features.keys())
    # TODO: add fileds into config dataclass, actually these config attrs cannot
    # be used temporarily and equal to hard-code
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
        from transformers.models.qwen2_vl import Qwen2VLForConditionalGeneration

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
        get_rope_index = types.MethodType(Qwen2VLForConditionalGeneration.get_rope_index, dummy_self)

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


class AOIReasonVLMPipelineMulti(BasePipeline):
    def __init__(self, pipeline_config: AOIReasonConfig):
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
                "prompt": datasets.Value("string"),
                "prompt_sat": datasets.Value("string"),
                "prompt_map": datasets.Value("string"),
                "question": datasets.Value("string"),
                "gt_mask": datasets.Image(decode=True),
                "seg_image": datasets.Image(decode=True),  # sat_image segmentation
                "gt_object": datasets.Value("int32"),  # number of objects in ground truth
                "image_sat": datasets.Sequence(datasets.Image(decode=True)),
                "image_map": datasets.Sequence(datasets.Image(decode=True)),
                "image": datasets.Sequence(datasets.Image(decode=True)),  # [sat_image, map_image]
                # for text and multi-modal mixed data usage, indicating valid image
                "image_flag": datasets.Value("bool"),
                # for area seperated validation, dummy currently
                "tag": datasets.Value("string"),
            }
        )
        # breakpoint()
        dataset = get_dataset(
            self.pipeline_config.actor_train.data_args, encode_function, self.processor, features, get_eval=False
        )
        val_dataset = None
        if self.pipeline_config.validation.data_args:
            val_dataset = get_dataset(
                self.pipeline_config.validation.data_args, encode_function, self.processor, features, get_eval=True
            )
            
        self.extra_data_provider = get_extra_data_provider(
            self.pipeline_config.actor_train.model_args.model_name_or_path, processor=self.processor
        )
        data_collator = DataCollatorWithPaddingForMultiSeg(
            tokenizer=self.tokenizer,
            processor=self.processor,
            extra_data_provider=self.extra_data_provider,
            max_length=self.pipeline_config.prompt_length,
            padding="max_length",
        )
        self.dataloader = get_dataloader(dataset, self.pipeline_config.rollout_batch_size, data_collator)
        
        # i = 0
        # for batch in self.dataloader:
        #     batch_true: DataProto = DataProto.from_single_dict(batch)
        #     print("batch", i)
        #     i += 1
        #     continue

        self.val_dataloader = None
        if val_dataset:
            self.val_dataloader = get_dataloader(val_dataset, len(val_dataset), data_collator)
        max_steps = len(self.dataloader) * self.pipeline_config.actor_train.training_args.num_train_epochs
        self.pipeline_config.set_max_steps(max_steps=max_steps)

        self.seg_infer: Any = Cluster(
            name=self.pipeline_config.seg_infer.name,
            worker_cls=self.pipeline_config.seg_infer.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.seg_infer,
        )

        self.actor_train: Any = Cluster(
            name=self.pipeline_config.actor_train.name,
            worker_cls=self.pipeline_config.actor_train.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_train,
        )
        self.actor_infer: Any = Cluster(
            name=self.pipeline_config.actor_infer.name,
            worker_cls=self.pipeline_config.actor_infer.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.actor_infer,
        )
        self.reference: Any = Cluster(
            name=self.pipeline_config.reference.name,
            worker_cls=self.pipeline_config.reference.worker_cls,
            resource_manager=self.resource_manager,
            worker_config=self.pipeline_config.reference,
        )
        self.rewards: Dict[str, Any] = {
            key: Cluster(
                name=f"reward-{key}",
                worker_cls=worker_config.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=worker_config,
            )
            for key, worker_config in self.pipeline_config.rewards.items()
        }
        self.reward: Any = self.rewards[list(self.rewards.keys())[0]]
        if self.pipeline_config.adv_estimator == "gae":
            self.critic: Any = Cluster(
                name=self.pipeline_config.critic.name,
                worker_cls=self.pipeline_config.critic.worker_cls,
                resource_manager=self.resource_manager,
                worker_config=self.pipeline_config.critic,
            )

        self.generate_scheduler = GenerateScheduler.options(
            name=f"{GENERATE_SCHEDULER_NAME}_{self.actor_infer.cluster_name}",
            get_if_exists=True,
            namespace=RAY_NAMESPACE,
        ).remote()

        self.kl_ctrl = get_kl_controller(
            init_kl_coef=self.pipeline_config.init_kl_coef,
            target_kl=self.pipeline_config.target_kl,
            kl_horizon=self.pipeline_config.kl_horizon,
        )
        
        refs: List[ray.ObjectRef] = []
        refs.extend(self.actor_train.initialize(pipeline_config=self.pipeline_config, blocking=False))
        if self.pipeline_config.adv_estimator == "gae":
            refs.extend(self.critic.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)

        refs = []
        refs.extend(self.actor_infer.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)
        
        refs = []
        refs.extend(self.seg_infer.initialize(pipeline_config=self.pipeline_config, blocking=False, tokenizer=self.tokenizer))
        ray.get(refs)

        refs = []
        refs.extend(self.reference.initialize(pipeline_config=self.pipeline_config, blocking=False))
        refs.extend(self.reward.initialize(pipeline_config=self.pipeline_config, blocking=False))
        ray.get(refs)

        self.set_model_update_pair(
            src_cluster=self.actor_train,
            tgt_cluster=self.actor_infer,
            frequency=self.pipeline_config.actor_train.model_update_frequency,
        )

        if self.pipeline_config.adv_estimator == "gae":
            self.set_checkpoint_clusters(self.actor_train, self.critic)
        else:
            self.set_checkpoint_clusters(self.actor_train)

        self.running = RunningMoments()

    @torch.no_grad()
    def run(self):
        global_step = 0

        # throughput for tokens per second
        tps_timer = _Timer(window_size=5)
        actor_infer_timer1 = _Timer(window_size=5)
        actor_infer_response_timer1 = _Timer(window_size=5)
        actor_infer_timer2 = _Timer(window_size=5)
        actor_infer_response_timer2 = _Timer(window_size=5)
        seg_infer_timer = _Timer(window_size=5)
        actor_train_timer = _Timer(window_size=5)

        for epoch in range(int(self.pipeline_config.actor_train.training_args.num_train_epochs)):
            logger.info(f"epoch {epoch} start...")
            for batch_dict in tqdm(self.dataloader):
                if global_step <= self.state.step:
                    global_step += 1
                    continue

                logger.info(f"pipeline step {global_step} start...")

                metrics = {}
                with tps_timer:
                    if self.pipeline_config.adv_estimator == "gae":
                        self.critic.offload_states(blocking=True)
                    self.actor_train.offload_states(blocking=True)
                    model_update_metrics: Dict = self.model_update(global_step)
                    metrics.update(model_update_metrics)

                    if self.val_dataloader and global_step % self.pipeline_config.eval_steps == 0:
                        metrics.update(self.val())

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
                    for key, value in batch.non_tensor_batch.items():
                        batch.non_tensor_batch[key] = np.repeat(
                            value, self.actor_infer.worker_config.generating_args.num_return_sequences
                        )
                        
                    batch.batch = generate_output.batch
                    batch = batch.union(generate_output)
                       
                    response_list = self.tokenizer.batch_decode(batch.batch["map_responses"], skip_special_tokens=False)
                    points_text_list = []
                    for response in response_list:            
                        # coords_list 的例子: [[10, 20], [30, 40]] 或 []
                        # 检查response类型
                        points_text = parse_points_text_from_content(response)
                        points_text_list.append(points_text)
                    
                    sat_text_list = []
                    for instruct, points in zip(batch.non_tensor_batch["question"], points_text_list):
                        text = format_prompt_sat(instruct, points, self.processor)
                        sat_text_list.append(text)
                        
                    # This section is rewritten to mimic the second snippet's iterative approach.

                    sat_padded_features = defaultdict(list)
                    un_padded_features = defaultdict(list)
                    mm_feature_keys = set()

                    # Iterate over each sample, analogous to `for feature in features:` in the reference snippet.
                    zipped_features = zip(
                        sat_text_list,
                        batch.non_tensor_batch["seg_image"]
                    )

                    for text, image_sat in zipped_features:
                        # Call the processor for each sample individually.
                        sat_model_inputs: BatchFeature = self.processor(
                            images=image_sat,
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
                                    "image": [image_sat] if not isinstance(image_sat, list) else image_sat,
                                },
                            }
                            )

                    # This part now runs after the loop, operating on the collected features.

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
                    
                    for key in sat_batch:
                        if isinstance(sat_batch[key], (torch.Tensor, np.ndarray)):
                            assert sat_batch[key].shape[0] == sat_batch["input_ids"].shape[0]
                        else:
                            assert len(sat_batch[key]) == sat_batch["input_ids"].shape[0]
                            sat_batch[key] = np.array(sat_batch[key], dtype=object)

                    sat_batch: DataProto = DataProto.from_single_dict(sat_batch)
                    
                    batch = batch.union(sat_batch)
                    
                    # for i in range(len(batch)):
                    #     batch[i].non_tensor_batch["response_text"] = self.tokenizer.decode(
                    #         batch[i].batch["responses"][0], skip_special_tokens=True
                    #     )
                    #     batch[i].non_tensor_batch["prompts"] = self.tokenizer.decode(
                    #         batch[i].batch["prompts"][0], skip_special_tokens=True
                    #     )
                    #     # add sat_image and mask
                    #     batch[i].non_tensor_batch["sat_image"] = batch[i].non_tensor_batch["image_sat"][0]
                    #     batch[i].non_tensor_batch["mask"] = batch[i].non_tensor_batch["seg_image"][0]

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
                        seg_batch_refs: List[ray.ObjectRef] = self.seg_infer.segment(seg_batch, blocking=False)
                    
                    seg_batch_out: DataProto = DataProto.materialize_concat(data_refs=seg_batch_refs)
                    batch = batch.union(seg_batch_out)
                    # save sat_image and mask
                    
                    # for i in range(len(batch)):
                    #     id = batch[i].non_tensor_batch['id']
                    #     response_text = batch[i].non_tensor_batch['response_text']
                    #     image = batch[i].non_tensor_batch['seg_image']
                    #     mask = batch[i].non_tensor_batch['mask']
                    #     # mask[mask> 0] = 1  # convert mask to binary
                    #     # save image and mask 
                    #     image.save(f"./save_dir/{id}_{global_step}_{i}_sat_image.png")
                    #     mask_image = Image.fromarray(mask * 255)  # convert to PIL image
                    #     mask_image.save(f"./save_dir/{id}_{global_step}_{i}_mask.png")
                    #     # save reponse text
                    #     with open(f"./save_dir/{id}_{global_step}_{i}_response_text.txt", "w", encoding="utf-8") as f:
                    #         f.write(response_text)
                    
                    # breakpoint()
                    with Timer(name="cal_ref_log_probs_reward", logger=None) as cal_timer:
                        map_batch = batch.pop(
                            batch_keys=["map_attention_mask", "map_position_ids", "map_input_ids", "map_responses", "map_prompts", "map_response_mask", "map_prompt_mask"],
                            non_tensor_batch_keys=['multi_modal_map_inputs']
                        )
                        map_batch.rename("map_input_ids", "input_ids")
                        map_batch.rename("map_attention_mask", "attention_mask")
                        map_batch.rename("map_position_ids", "position_ids")
                        map_batch.rename("map_responses", "responses")
                        map_batch.rename("map_response_mask", "response_mask")
                        map_batch.rename("map_prompts", "prompts")
                        map_batch.rename("map_prompt_mask", "prompt_mask")
                        map_batch.non_tensor_batch["multi_modal_inputs"] = map_batch.non_tensor_batch.pop("multi_modal_map_inputs")
                        
                        sat_batch = batch.pop(
                            batch_keys=["attention_mask", "position_ids", "input_ids", "responses", "prompts", "response_mask", "prompt_mask"],
                            non_tensor_batch_keys=['multi_modal_sat_inputs']
                        )
                        sat_batch.non_tensor_batch["multi_modal_inputs"] = sat_batch.non_tensor_batch.pop("multi_modal_sat_inputs")
                        
                        map_ref_log_probs_refs: List[ray.ObjectRef] = self.reference.compute_log_probs(
                            map_batch, blocking=False
                        )
                        
                        sat_ref_log_probs_refs: List[ray.ObjectRef] = self.reference.compute_log_probs(
                            sat_batch, blocking=False
                        )

                        batch.batch["sat_responses"] = sat_batch.batch["responses"]
                        batch.batch["map_responses"] = map_batch.batch["responses"]
                        
                        rewards_refs: List[ray.ObjectRef] = self.reward.compute_rewards(batch, blocking=False)
                        
                        map_ref_log_probs = DataProto.materialize_concat(data_refs=map_ref_log_probs_refs)
                        sat_ref_log_probs = DataProto.materialize_concat(data_refs=sat_ref_log_probs_refs)
                        
                        rewards = DataProto.materialize_concat(data_refs=rewards_refs)
                        
                        metrics.update(reduce_metrics(sat_ref_log_probs.meta_info.pop("metrics", {})))
                        # metrics.update(reduce_metrics(map_ref_log_probs.meta_info.pop("metrics", {})))
                        metrics.update(reduce_metrics(rewards.meta_info.pop("metrics", {})))
                        sat_ref_log_probs.rename(old_keys="log_probs", new_keys="ref_log_probs")
                        map_ref_log_probs.rename(old_keys="log_probs", new_keys="ref_log_probs")
                        
                        sat_batch = sat_batch.union(sat_ref_log_probs)
                        map_batch = map_batch.union(map_ref_log_probs)

                        sat_batch = sat_batch.union(rewards)
                        map_batch = map_batch.union(rewards)

                    metrics["time/ref_log_probs_values_reward"] = cal_timer.last

                    with Timer(name="cal_old_log_probs_values", logger=None) as cal_old_logpb_timer:
                        batch.meta_info["is_offload_states"] = False
                        # if self.pipeline_config.adv_estimator == "gae":
                        #     values_refs: List[ray.ObjectRef] = self.critic.compute_values(batch, blocking=False)
                        # breakpoint() 
                        map_old_log_probs_refs: List[ray.ObjectRef] = self.actor_train.compute_log_probs(
                            map_batch, blocking=False
                        )
                        map_old_log_probs = DataProto.materialize_concat(data_refs=map_old_log_probs_refs)
                        
                        sat_old_log_probs_refs: List[ray.ObjectRef] = self.actor_train.compute_log_probs(
                            sat_batch, blocking=False
                        )
                        
                        sat_old_log_probs = DataProto.materialize_concat(data_refs=sat_old_log_probs_refs)
                        # if self.pipeline_config.adv_estimator == "gae":
                        #     values = DataProto.materialize_concat(data_refs=values_refs)
                        #     batch = batch.union(values)
                        #     metrics.update(reduce_metrics(values.meta_info.pop("metrics", {})))

                        map_batch.batch["old_log_probs"] = map_old_log_probs.batch["log_probs"]
                        sat_batch.batch["old_log_probs"] = sat_old_log_probs.batch["log_probs"]
                        metrics.update(reduce_metrics(map_old_log_probs.meta_info.pop("metrics", {})))
                        metrics.update(reduce_metrics(sat_old_log_probs.meta_info.pop("metrics", {})))

                    metrics["time/old_log_probs"] = cal_old_logpb_timer.last

                    with Timer(name="adv", logger=None) as timer:
                        # if self.pipeline_config.use_reward_scaling:
                        #     self.running.update(batch.batch["response_level_rewards"])
                        #     reward_scaling_factor = (
                        #         self.running.std + torch.finfo(batch.batch["response_level_rewards"].dtype).eps
                        #     )
                        #     if self.pipeline_config.use_reward_norm:
                        #         batch.batch["response_level_rewards"] = (
                        #             batch.batch["response_level_rewards"] - self.running.mean
                        #         ) / reward_scaling_factor
                        #     else:
                        #         batch.batch["response_level_rewards"] /= (
                        #             reward_scaling_factor  # do not -= mean since advantage will be normalized again
                        #         )

                        if self.pipeline_config.reward_clip:
                            map_reward_clip_frac = compute_clip_fraction(
                                values=map_batch.batch["response_level_rewards"],
                                clip_max=self.pipeline_config.reward_clip,
                                clip_min=-self.pipeline_config.reward_clip,
                            )
                            sat_reward_clip_frac = compute_clip_fraction(
                                values=sat_batch.batch["response_level_rewards"],
                                clip_max=self.pipeline_config.reward_clip,
                                clip_min=-self.pipeline_config.reward_clip,
                            )
                            metrics["critic/map_reward_clip_frac"] = map_reward_clip_frac
                            metrics["critic/sat_reward_clip_frac"] = sat_reward_clip_frac
                            
                            map_batch.batch["response_level_rewards"] = torch.clamp(
                                map_batch.batch["response_level_rewards"],
                                min=-self.pipeline_config.reward_clip,
                                max=self.pipeline_config.reward_clip,
                            ) 
                            sat_batch.batch["response_level_rewards"] = torch.clamp(
                                sat_batch.batch["response_level_rewards"],
                                min=-self.pipeline_config.reward_clip,
                                max=self.pipeline_config.reward_clip,
                            )
                            
                        if self.pipeline_config.adv_estimator == "grpo":
                            map_batch = group_reward_norm(
                                map_batch,
                                n_sample=self.pipeline_config.actor_infer.generating_args.num_return_sequences,
                                div_std=True,
                            )
                            sat_batch = group_reward_norm(
                                sat_batch,
                                n_sample=self.pipeline_config.actor_infer.generating_args.num_return_sequences,
                                div_std=True,
                            )

                        if not self.pipeline_config.use_kl_loss:  # not grpo's kl loss
                            batch, kl_metrics = apply_kl_penalty(
                                data=batch, kl_ctrl=self.kl_ctrl, kl_penalty=self.pipeline_config.kl_penalty
                            )
                        else:
                            map_token_level_rewards = expand_to_token_level(data=map_batch)
                            map_batch.batch["token_level_rewards"] = map_token_level_rewards
                            sat_token_level_rewards = expand_to_token_level(data=sat_batch)
                            sat_batch.batch["token_level_rewards"] = sat_token_level_rewards
                            kl_metrics = {}
                            
                        if self.pipeline_config.reward_clip:
                            map_reward_clip_frac = compute_clip_fraction(
                                values=map_batch.batch["token_level_rewards"],
                                clip_max=self.pipeline_config.reward_clip,
                                clip_min=-self.pipeline_config.reward_clip,
                            )
                            sat_reward_clip_frac = compute_clip_fraction(
                                values=sat_batch.batch["token_level_rewards"],
                                clip_max=self.pipeline_config.reward_clip,
                                clip_min=-self.pipeline_config.reward_clip,
                            )
                            metrics["critic/map_token_reward_clip_frac"] = map_reward_clip_frac
                            metrics["critic/sat_token_reward_clip_frac"] = sat_reward_clip_frac
                            
                            map_batch.batch["token_level_rewards"] = torch.clamp(
                                map_batch.batch["token_level_rewards"],
                                min=-self.pipeline_config.reward_clip,
                                max=self.pipeline_config.reward_clip,
                            )
                            sat_batch.batch["token_level_rewards"] = torch.clamp(
                                sat_batch.batch["token_level_rewards"],
                                min=-self.pipeline_config.reward_clip,
                                max=self.pipeline_config.reward_clip,
                            )
                        # breakpoint()
                        map_batch = compute_advantage(
                            data=map_batch,
                            gamma=self.pipeline_config.gamma,
                            lambd=self.pipeline_config.lambd,
                            adv_estimator=self.pipeline_config.adv_estimator,
                            advantage_clip=self.pipeline_config.advantage_clip,
                            whiten_advantages=self.pipeline_config.whiten_advantages,
                            whiten_rewards=self.pipeline_config.whiten_rewards,
                        )

                        sat_batch = compute_advantage(
                            data=sat_batch,
                            gamma=self.pipeline_config.gamma,
                            lambd=self.pipeline_config.lambd,
                            adv_estimator=self.pipeline_config.adv_estimator,
                            advantage_clip=self.pipeline_config.advantage_clip,
                            whiten_advantages=self.pipeline_config.whiten_advantages,
                            whiten_rewards=self.pipeline_config.whiten_rewards,
                        )
                        metrics.update(reduce_metrics(batch.meta_info.pop("metrics", {})))

                    metrics.update(kl_metrics)
                    metrics["time/adv"] = timer.last

                    if self.pipeline_config.adv_estimator == "gae":
                        critic_train_metrics_refs: List[ray.ObjectRef] = self.critic.train_step(batch, blocking=False)

                    with actor_train_timer:
                        # implement critic warmup
                        if not hasattr(self, "critic") or self.pipeline_config.critic_warmup <= global_step:
                            # update actor
                            map_actor_train_metrics_refs = self.actor_train.train_step(map_batch, blocking=False)
                            sat_actor_train_metrics_refs = self.actor_train.train_step(sat_batch, blocking=False)
                            map_actor_train_metrics: DataProto = DataProto.materialize_concat(
                                data_refs=map_actor_train_metrics_refs
                            )
                            sat_actor_train_metrics: DataProto = DataProto.materialize_concat(
                                data_refs=sat_actor_train_metrics_refs
                            )
                            metrics.update(reduce_metrics(map_actor_train_metrics.meta_info.pop("metrics", {})))
                            metrics.update(reduce_metrics(sat_actor_train_metrics.meta_info.pop("metrics", {})))
                    
                    if self.pipeline_config.adv_estimator == "gae":
                        critic_train_metrics = DataProto.materialize_concat(data_refs=critic_train_metrics_refs)
                        metrics.update(reduce_metrics(critic_train_metrics.meta_info.pop("metrics", {})))

                    tps_timer.push_units_processed(n=torch.sum(sat_batch.batch["attention_mask"]).detach().item())
                    actor_infer_timer1.push_units_processed(n=torch.sum(sat_batch.batch["attention_mask"]).detach().item())
                    actor_infer_response_timer1.push_units_processed(
                        n=torch.sum(sat_batch.batch["response_mask"]).detach().item()
                    )
                    actor_train_timer.push_units_processed(n=torch.sum(sat_batch.batch["attention_mask"]).detach().item())

                batch = batch.union(sat_batch)
                data_metrics = compute_data_metrics(batch=batch)
                metrics.update(data_metrics)
                metrics["system/tps"] = tps_timer.mean_throughput
                metrics["system/actor_infer/tps"] = actor_infer_timer1.mean_throughput
                metrics["system/actor_infer/response/tps"] = actor_infer_response_timer1.mean_throughput
                metrics["system/actor_train/tps"] = actor_train_timer.mean_throughput
                metrics["system/tps_gpu"] = tps_timer.mean_throughput / self.resource_manager.num_gpus
                metrics["system/actor_infer/tps_gpu"] = actor_infer_timer1.mean_throughput / self.actor_infer.world_size
                metrics["system/actor_infer/response/tps_gpu"] = (
                    actor_infer_response_timer1.mean_throughput / self.actor_infer.world_size
                )
                metrics["system/actor_train/tps_gpu"] = actor_train_timer.mean_throughput / self.actor_train.world_size
                metrics["system/actor_infer/tps_dp"] = actor_infer_timer1.mean_throughput / self.actor_infer.dp_size
                metrics["system/actor_infer/response/tps_dp"] = (
                    actor_infer_response_timer1.mean_throughput / self.actor_infer.dp_size
                )
                metrics["system/actor_train/tps_dp"] = actor_train_timer.mean_throughput / self.actor_train.dp_size
                metrics["system/samples"] = (global_step + 1) * batch.batch.shape[0]

                # do ckpt
                self.state.step = global_step
                self.state.log_history.append(metrics)

                self.do_checkpoint(global_step=global_step)

                self.tracker.log(values=metrics, step=global_step)

                if global_step % self.pipeline_config.logging_steps == 0:
                    if int(os.environ.get("RAY_PROFILING", "0")):
                        timeline_dir = os.path.join(self.pipeline_config.profiler_output_dir, "timeline")
                        os.makedirs(timeline_dir, exist_ok=True)
                        ray.timeline(
                            filename=os.path.join(timeline_dir, f"timeline-step-{global_step}.json"),
                        )

                    prompt_ids = generate_output.batch["prompts"]
                    response_ids = generate_output.batch["responses"]

                    generate_res = []
                    # skip_special_tokens=True would output without image token, maybe do not skip
                    prompts = self.tokenizer.batch_decode(prompt_ids, skip_special_tokens=True)
                    responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
                    for prompt, prompt_id, response, response_id in zip(
                        prompts,
                        prompt_ids,
                        responses,
                        response_ids,
                    ):
                        generate_res.append(
                            {
                                "prompt": prompt,
                                # "prompt_id": prompt_id.tolist(),
                                "response": response,
                                # "response_id": response_id.tolist(),
                            }
                        )
                    logger.info(json.dumps(generate_res[:10], ensure_ascii=False))
                    logger.info(json.dumps(metrics, ensure_ascii=False))

                logger.info(f"pipeline step {global_step} finished")
                global_step += 1

                if global_step >= self.pipeline_config.max_steps:
                    logger.info(f"pipeline step {global_step} finished, reached max steps: {self.pipeline_config.max_steps}")
                    return

            logger.info(f"epoch {epoch} finished")
        logger.info("pipeline complete!")

    @torch.no_grad()
    def val(self):
        # throughput for tokens per second
        tps_timer = _Timer(window_size=5)
        metrics = {}
        epoch_batch = []
        for batch_dict in tqdm(self.val_dataloader):
            with tps_timer:
                batch_dict: Dict
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                gen_batch = batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["multi_modal_data"] if "multi_modal_data" in batch.non_tensor_batch else [],
                )
                gen_batch.meta_info["is_offload_states"] = False
                gen_batch.meta_info["response_callback_fn"] = self.generate_scheduler.report_response.remote
                generate_output: DataProto = ray.get(
                    self.generate_scheduler.generate.remote(
                        data=gen_batch,
                        actor_cluster=self.actor_infer,
                        pipeline_config=self.pipeline_config,
                    ),
                    timeout=self.pipeline_config.rpc_timeout,
                )
                batch.batch = generate_output.batch
                batch = batch.union(generate_output)

                for key, value in batch.non_tensor_batch.items():
                    batch.non_tensor_batch[key] = np.repeat(
                        value, self.actor_infer.worker_config.generating_args.num_return_sequences
                    )

                with Timer(name="cal_reward", logger=None) as cal_timer:
                    rewards = ray.get(self.reward.workers[0].compute_rewards.remote(batch))
                    batch = batch.union(rewards)
                logger.info(
                    json.dumps(
                        {"val_correct/mean": (batch.batch["scores"] == 1).detach().float().mean().item()},
                        ensure_ascii=False,
                    )
                )
                epoch_batch.append(batch)

        if len(epoch_batch) == 0:
            logger.info(f"len(self.val_dataloader): {len(self.val_dataloader)}, skip val...")
            return {}

        epoch_batch = DataProto.concat(epoch_batch)
        logger.info(f"total eval information: {epoch_batch}")
        logger.info(f"total eval information --- scores mean: {epoch_batch.batch['scores'].mean().item()} "
                    f"scores: {epoch_batch.batch['scores'].tolist()}")
        metrics[ f"val_correct/mean"] =  (epoch_batch.batch["scores"] == 1).detach().float().mean().item()
        return metrics


def compute_data_metrics(batch):
    sequence_score = batch.batch["seg_iou_rewards"]
    sequence_reward = batch.batch["token_level_rewards"].sum(-1)
    sequence_reward_mean = batch.batch["token_level_rewards"].mean(-1)

    max_response_length = batch.batch["responses"].shape[-1]
    advantages = batch.batch["advantages"]
    prompt_mask = batch.batch["prompt_mask"].bool()
    response_mask = batch.batch["response_mask"][:, 1:].bool()
    raw_advantages = batch.batch["raw_advantages"]
    prompt_length = prompt_mask.sum(-1).float()  # (batch_size,)
    response_length = response_mask.sum(-1).float()  # (batch_size,)
    returns = batch.batch["returns"]

    metrics = {
        # score
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        "critic/rewards_mean/mean": torch.mean(sequence_reward_mean).detach().item(),
        "critic/rewards_mean/max": torch.max(sequence_reward_mean).detach().item(),
        "critic/rewards_mean/min": torch.min(sequence_reward_mean).detach().item(),
        # adv
        "critic/advantages/mean": masked_mean(advantages, response_mask).detach().item(),
        "critic/advantages/max": torch.max(advantages[response_mask]).detach().item(),
        "critic/advantages/min": torch.min(advantages[response_mask]).detach().item(),
        # raw_adv
        "critic/raw_advantages/mean": masked_mean(raw_advantages, response_mask).detach().item(),
        "critic/raw_advantages/max": torch.max(raw_advantages[response_mask]).detach().item(),
        "critic/raw_advantages/min": torch.min(raw_advantages[response_mask]).detach().item(),
        # returns
        "critic/returns/mean": masked_mean(returns, response_mask).detach().item(),
        "critic/returns/max": torch.max(returns[response_mask]).detach().item(),
        "critic/returns/min": torch.min(returns[response_mask]).detach().item(),
        # response length
        "tokens/response_length/mean": torch.mean(response_length).detach().item(),
        "tokens/response_length/max": torch.max(response_length).detach().item(),
        "tokens/response_length/min": torch.min(response_length).detach().item(),
        # prompt length
        "tokens/prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "tokens/prompt_length/max": torch.max(prompt_length).detach().item(),
        "tokens/prompt_length/min": torch.min(prompt_length).detach().item(),
    }

    if "values" in batch.batch.keys():
        values = batch.batch["values"]
        # values
        metrics.update(
            {
                "critic/values/mean": masked_mean(values, response_mask).detach().item(),
                "critic/values/max": torch.max(values[response_mask]).detach().item(),
                "critic/values/min": torch.min(values[response_mask]).detach().item(),
            }
        )
    return metrics
