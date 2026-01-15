from typing import Optional

from vllm.device_allocator.cumem import CuMemAllocator, create_and_map, libcudart

def wake_up_with_tags(self, tags: Optional[list[str]] = None) -> None:
    """
    Wake up the allocator from sleep mode.
    All data that is previously offloaded will be loaded back to GPU
    memory, and the rest of the data will have empty memory.

    :param tags: The tags of the memory allocation that will be loaded
        back to GPU memory. If None, all memory allocation will be loaded
        back to GPU memory.
    """
    for ptr, data in self.pointer_to_data.items():
        if tags is None or data.tag in tags:
            handle = data.handle
            create_and_map(handle)
            if data.cpu_backup_tensor is not None:
                cpu_backup_tensor = data.cpu_backup_tensor
                if cpu_backup_tensor is not None:
                    size_in_bytes = cpu_backup_tensor.numel(
                    ) * cpu_backup_tensor.element_size()
                    cpu_ptr = cpu_backup_tensor.data_ptr()
                    libcudart.cudaMemcpy(ptr, cpu_ptr, size_in_bytes)
                    data.cpu_backup_tensor = None

assert CuMemAllocator.instance is None
CuMemAllocator.wake_up = wake_up_with_tags

__all__ = []
