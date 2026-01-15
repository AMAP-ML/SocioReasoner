import gc
from typing import Optional

import torch
from vllm.worker.worker import Worker
from vllm.device_allocator.cumem import CuMemAllocator

from roll.third_party.vllm.worker_helper import WorkerHelper
from roll.utils.logging import get_logger

logger = get_logger()


class Worker073(WorkerHelper, Worker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def wake_up(self, tags: Optional[list[str]] = None) -> None:
        allocator = CuMemAllocator.get_instance()
        allocator.wake_up(tags)
