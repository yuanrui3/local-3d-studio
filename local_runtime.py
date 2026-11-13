"""Workspace-local configuration and CPU mesh extraction for TripoSR."""
from pathlib import Path
import os
import sys
import tempfile
import types

ROOT = Path(__file__).resolve().parent


def configure(offline=True):
    os.environ['HF_HOME'] = str(ROOT / '.cache' / 'huggingface')
    os.environ['U2NET_HOME'] = str(ROOT / '.cache' / 'rembg')
    # Numba adds long generated filenames, so use a short Windows cache path.
    numba_cache = (Path(tempfile.gettempdir()) / 'local-3d-studio-numba'
                   if os.name == 'nt' else ROOT / '.cache' / 'numba')
    os.environ.setdefault('NUMBA_CACHE_DIR', str(numba_cache))
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    os.environ['DO_NOT_TRACK'] = '1'
    if offline:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
    else:
        os.environ.pop('HF_HUB_OFFLINE', None)
        os.environ.pop('TRANSFORMERS_OFFLINE', None)
    sys.path.insert(0, str(ROOT / 'vendor' / 'TripoSR'))


def install_cpu_marching_cubes():
    """Supply torchmcubes's coordinate convention using scikit-image.

    Upstream's helper reverses the coordinates returned by torchmcubes.
    scikit-image returns array-axis coordinates, so reverse here first to
    preserve upstream's final XYZ convention. All extraction happens on CPU.
    """
    import numpy as np
    import torch
    from skimage.measure import marching_cubes

    def extract(volume, threshold):
        array = volume.detach().float().cpu().numpy()
        vertices, faces, _, _ = marching_cubes(
            array, level=threshold, allow_degenerate=False,
            gradient_direction='descent',
        )
        return (
            torch.from_numpy(np.ascontiguousarray(vertices[:, [2, 1, 0]])),
            torch.from_numpy(np.ascontiguousarray(faces.astype(np.int64))),
        )

    module = types.ModuleType('torchmcubes')
    module.marching_cubes = extract
    sys.modules['torchmcubes'] = module
