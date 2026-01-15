import os
import json
from typing import Union, Callable, Dict
import datasets 
from datasets import Dataset, IterableDataset, load_dataset
from transformers import PreTrainedTokenizer

from roll.configs.data_args import DataArguments
from roll.utils.logging import get_logger

logger = get_logger()

REGISTERED_DATASETS: Dict[str, Callable[[DataArguments], Union[Dataset, IterableDataset]]] = {}

def register_dataset(key: str):
    def decorator(func: Callable[[DataArguments], Union[Dataset, IterableDataset]]):
        if key in REGISTERED_DATASETS:
            raise ValueError(f"Dataset type '{key}' already exists!")
        REGISTERED_DATASETS[key] = func
        return func
    return decorator

def get_dataset(data_args: "DataArguments"):
    key = data_args.dataset_type
    if key not in REGISTERED_DATASETS:
        raise ValueError(
            f"Dataset type '{key}' is not found! Available datasets: {list(REGISTERED_DATASETS.keys())}"
        )
        
    dataset_paths = []
    if data_args.file_name:
        dataset_paths.extend(data_args.file_name)

    logger.info(f'load_dataset_paths: {chr(10)} {chr(10).join(dataset_paths)}')
    logger.info(f'prompt column: {data_args.prompt}  label column: {data_args.response}')

    return REGISTERED_DATASETS[key](dataset_paths, data_args)


@register_dataset("default")
@register_dataset("json")
def default_json_dataset(
    dataset_paths: "DataPaths",
    data_args: "DataArguments",
) -> Union["Dataset", "IterableDataset"]:
    return datasets.load_dataset('json', data_files=dataset_paths)['train']


class SocioSegDataset(datasets.GeneratorBasedBuilder):
    """
    - train/
      - <id_1>/
        - map.png
        - sat.png
        - mask.png
        - question.json
      - <id_2>/
        ...
    - val/
      - <id_3>/
        ...
    """

    def _info(self):
        # 在这里定义数据集的特征，即每一条数据的结构
        return datasets.DatasetInfo(
            description="SocioSeg Dataset",
            features=datasets.Features({
                "id": datasets.Value("string"), 
                "problem": datasets.Value("string"),
                "map_image": datasets.Image(),
                "sat_image": datasets.Image(),
                "stage2_seg_image": datasets.Image(),
                "mask_label": datasets.Image(),
                "segzero_label": datasets.Value("string"),
                "stage1_answer": datasets.Value("string"),
            }),
        )

    def _split_generators(self, dl_manager):
        data_dir = self.config.data_dir
        if not data_dir or not os.path.isdir(data_dir):
            raise ValueError(f"please provide a valid data_dir")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_dir": os.path.join(data_dir, "train")},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"data_dir": os.path.join(data_dir, "val")},
            ),
        ]

    def _generate_examples(self, data_dir):
        example_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

        for idx, example_id in enumerate(example_dirs):
            example_path = os.path.join(data_dir, example_id)

            json_path = os.path.join(example_path, "question.json")
            map_path = os.path.join(example_path, "map.png")
            sat_path = os.path.join(example_path, "sat.png")
            mask_path = os.path.join(example_path, "mask.png")
            segzero_path = os.path.join(example_path, "two_point.json")

            #
            stage1_answer_path = os.path.join(example_path, "stage1_answer.json")

            if not all(os.path.exists(p) for p in [json_path, map_path, sat_path, mask_path]):
                print(f"warning {example_path} is not complete")
                continue

            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    question_data = json.load(f)
                    problem_text = question_data.get("problem", "")
            except Exception as e:
                print(f"error: {e}")
                continue

            if os.path.exists(segzero_path):
                with open(segzero_path, "r", encoding="utf-8") as f:
                    segzero_data = json.load(f)
            else:
                segzero_data = None

            if os.path.exists(stage1_answer_path):
                with open(stage1_answer_path, "r", encoding="utf-8") as f:
                    stage1_answer_data = json.load(f)
                    stage1_answer_text = stage1_answer_data.get("stage1_answer", "")
                stage2_seg_path = os.path.join(example_path.replace("stage2_", ""), "sat.png")
            else:
                stage1_answer_text = None
                stage2_seg_path = None

            yield {
                "id": example_id,
                "problem": problem_text,
                "map_image": map_path,
                "sat_image": sat_path,
                "mask_label": mask_path,
                "segzero_label": segzero_data,
                "stage1_answer": stage1_answer_text,
                "stage2_seg_image": stage2_seg_path
            }