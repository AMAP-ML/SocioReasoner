import torch
import numpy as np
import cv2

from roll.distributed.scheduler.protocol import DataProto
from roll.distributed.strategy.strategy import InferenceStrategy
from roll.utils.logging import get_logger
from roll.utils.offload_states import OffloadStateType, offload_module, reload_module
from roll.datasets.collator import collate_fn_to_dict_list

logger = get_logger()

class SegInferStrategy(InferenceStrategy):
    strategy_name = "seg_infer"

    def __init__(self, worker: "Worker"):
        super().__init__(worker)
        self.model = None

    def initialize(self, model_provider):

        self.model = model_provider(model_args=self.worker_config.model_args, is_trainable=False)

        print("SAMStrategy initialized with model and predictor.")

    def segment(self, batch: DataProto) -> dict:
        num_microbatches = batch.batch.batch_size[0]
        micro_batches = batch.chunk(chunks=num_microbatches)
        
        masks = []
        scores = []
        
        for data in micro_batches:
            image = data.non_tensor_batch['seg_image'][0]
            visual_prompt= data.non_tensor_batch['visual_prompt'][0]
        
            image = image.resize((756, 756))
            mask = np.zeros((756, 756), dtype=np.uint8)
            
            if len(visual_prompt) == 0:
                mask = np.zeros((768, 768), dtype=np.uint8)
                masks.append({
                    "mask": mask
                })
                continue
                
            self.model.set_image(image)
            
            for single_vp in visual_prompt:
                try:
                    prompt = {}
                    if 'point_coords' in single_vp and 'point_labels' in single_vp:
                        prompt['point_coords'] = single_vp['point_coords']
                        prompt['point_labels'] = single_vp['point_labels']
                    if 'box' in single_vp:
                        # The model's predict function expects the key to be 'box'
                        prompt['box'] = single_vp['box']
                    pred_masks, scores, _ = self.model.predict(**prompt)
                    best_mask = pred_masks[np.argmax(scores)]
                    mask = np.logical_or(mask, best_mask).astype(np.uint8)
                    # mask += pred_masks[0].astype(np.uint8) 
                except:
                    continue

            mask = cv2.resize(mask, (768, 768), interpolation=cv2.INTER_NEAREST)

            masks.append({
                "mask": mask
            })

        results = collate_fn_to_dict_list(masks)
        return results

    def load_states(self, *args, **kwargs):
        reload_module(self.model.model, device="cuda", non_blocking=True)

    def offload_states(self, include=None, non_blocking=False):
        if include is None or OffloadStateType.model_params in include:
            offload_module(self.model.model, device="cpu", pin_memory=True, non_blocking=non_blocking)
        torch.cuda.empty_cache()