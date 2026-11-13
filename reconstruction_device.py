"""Explicit compute-device checks for the single-view reconstruction control."""
import torch


def checked_device(requested):
    """Reject an unavailable accelerator instead of silently running on CPU."""
    device = torch.device(requested)
    if device.type == 'cpu':
        if device.index is not None:
            raise ValueError('Use cpu without a device index')
    elif device.type == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable in this PyTorch environment')
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise ValueError(f'CUDA device {index} does not exist')
        device = torch.device('cuda', index)
    elif device.type == 'mps':
        if device.index is not None:
            raise ValueError('Use mps without a device index')
        if not torch.backends.mps.is_available():
            raise ValueError('MPS is unavailable in this PyTorch environment')
    else:
        raise ValueError('Choose cpu, mps, cuda or cuda:<index>')
    return device


def synchronize(device):
    """Finish queued accelerator work before reporting elapsed wall time."""
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    elif device.type == 'mps':
        torch.mps.synchronize()


def release_unused_memory(device):
    """Release cached encoder memory after its tensors have been deleted."""
    if device.type == 'cuda':
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    elif device.type == 'mps':
        torch.mps.empty_cache()


def runtime_receipt(device):
    receipt = {'torch': torch.__version__, 'device': str(device),
               'torch_cuda_runtime': torch.version.cuda}
    if device.type == 'cuda':
        properties = torch.cuda.get_device_properties(device)
        receipt.update(gpu=properties.name, total_vram_bytes=properties.total_memory,
                       peak_tensor_memory_bytes=torch.cuda.max_memory_allocated(device),
                       peak_reserved_memory_bytes=torch.cuda.max_memory_reserved(device))
    return receipt
