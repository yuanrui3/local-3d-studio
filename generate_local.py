"""Offline single-view TripoSR control, with bounded-memory CPU meshing.

Neural inference and density queries can use CPU, Apple MPS or NVIDIA CUDA.
Marching cubes always runs locally on CPU through scikit-image.
"""
import argparse
import gc
import hashlib
import io
import json
import time
from pathlib import Path
from local_runtime import ROOT, configure, install_cpu_marching_cubes

configure()
import numpy as np
import torch
import trimesh
from PIL import Image
from omegaconf import OmegaConf
from reconstruction_device import (checked_device, synchronize,
                                   release_unused_memory, runtime_receipt)

install_cpu_marching_cubes()
from tsr.system import TSR


def recorded_model_provenance(sources):
    """Copy download pins without exposing cache paths or claiming a new hash check."""
    required = {('stabilityai/TripoSR', 'config.yaml'),
                ('stabilityai/TripoSR', 'model.ckpt'),
                ('facebook/dino-vitb16', 'config.json')}
    assets = []
    for item in sources:
        if (item.get('repo'), item.get('file')) in required:
            assets.append({key: item.get(key) for key in (
                'repo', 'revision', 'file', 'sha256', 'weights_manifest_sha256')})
    source = next(item for item in json.loads((ROOT/'vendor/sources.json').read_text())['repositories']
                  if item['path'] == 'vendor/TripoSR')
    return {
        'status': 'download_receipts_copied_not_reverified_for_this_inference',
        'assets': assets,
        'upstream_source': {'url': source['url'], 'commit': source['commit'],
                            'status': 'recorded_vendor_pin_not_rechecked_for_this_inference'},
        'limitations': 'Missing SHA256 fields denote legacy receipts, not verified weight provenance.',
    }


def load_model(sources=None):
    if sources is None:
        sources = json.loads((ROOT/'models'/'sources.json').read_text())
    local = {x['file']:x['local_path'] for x in sources if x.get('repo')=='stabilityai/TripoSR'}
    # Upstream asks the Hub for the DINO config during construction. Resolve
    # that one asset from the pinned local receipt; no network during inference.
    import tsr.models.tokenizers.image as tokenizer_module
    dino_config = next(x['local_path'] for x in sources
                       if x.get('repo')=='facebook/dino-vitb16')
    def local_dino_config(repo_id, filename):
        if repo_id != 'facebook/dino-vitb16' or filename != 'config.json':
            raise ValueError('Unexpected model asset requested')
        return dino_config
    tokenizer_module.hf_hub_download = local_dino_config
    config = OmegaConf.load(local['config.yaml'])
    OmegaConf.resolve(config)
    model = TSR(config)
    # memory-mapped, restricted tensor loading avoids a second full checkpoint
    # allocation and prevents arbitrary pickle globals from executing.
    state = torch.load(local['model.ckpt'], map_location='cpu',
                       weights_only=True, mmap=True)
    model.load_state_dict(state, assign=True)
    del state
    gc.collect()
    return model.eval()


def extract_mesh(model, scene, resolution, threshold=25.0, chunk=8192):
    from skimage.measure import marching_cubes
    radius = float(model.renderer.cfg.radius)
    coords = torch.linspace(-radius,radius,resolution,device=scene.device)
    density = np.empty((resolution,resolution,resolution),dtype=np.float32)
    # Slice grid instead of materializing all positions/features at once.
    for x in range(resolution):
        yy,zz = torch.meshgrid(coords,coords,indexing='ij')
        points = torch.stack([torch.full_like(yy,coords[x]),yy,zz],-1).reshape(-1,3)
        vals = model.renderer.query_triplane(model.decoder,points,scene)['density_act']
        density[x] = vals.detach().cpu().numpy().reshape(resolution,resolution)
        if x%16==0:
            print(f'Extracting surface: {x}/{resolution} slices',flush=True)
    if not density.min() < threshold < density.max():
        raise RuntimeError(f'No surface at threshold {threshold}; density range {density.min()}..{density.max()}')
    vertices,faces,_,_ = marching_cubes(density,level=threshold,
                                      allow_degenerate=False,gradient_direction='descent')
    vertices = vertices/(resolution-1)*(2*radius)-radius
    colors = []
    for start in range(0,len(vertices),chunk):
        points = torch.tensor(vertices[start:start+chunk].copy(),device=scene.device)
        rgb = model.renderer.query_triplane(model.decoder,points,scene)['color']
        colors.append(rgb.detach().cpu().numpy())
    rgba = np.column_stack([np.clip(np.concatenate(colors)*255,0,255).astype(np.uint8),
                            np.full(len(vertices),255,dtype=np.uint8)])
    mesh = trimesh.Trimesh(vertices=vertices,faces=faces,vertex_colors=rgba,process=False)
    mesh.fix_normals()
    return mesh


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('image',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs'/'local-v001')
    parser.add_argument('--device',default='cpu',
                        help='cpu (default), mps, cuda or cuda:<index>; no silent fallback')
    parser.add_argument('--resolution',type=int,default=128)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    if not 32 <= args.resolution <= 256:
        parser.error('resolution must be between 32 and 256')
    if args.threads < 1:
        parser.error('threads must be positive')
    try:
        device=checked_device(args.device)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(device)
    start=time.monotonic()
    print('Loading pretrained TripoSR weights locally',flush=True)
    sources=json.loads((ROOT/'models'/'sources.json').read_text())
    model=load_model(sources)
    model.renderer.set_chunk_size(8192)
    model.to(device)
    synchronize(device)
    print(f'Model loaded in {time.monotonic()-start:.1f}s; reconstructing on {device}',flush=True)
    # Decode the same immutable byte string recorded in the manifest.
    input_bytes=args.image.read_bytes()
    input_sha256=hashlib.sha256(input_bytes).hexdigest()
    with Image.open(io.BytesIO(input_bytes)) as encoded:
        source=encoded.convert('RGB')
    with torch.inference_mode():
        scene=model([source],device=str(device))[0]
        synchronize(device)
        print(f'Inference finished at {time.monotonic()-start:.1f}s',flush=True)
        # Keep just the decoder/renderer and the triplane during extraction.
        del model.image_tokenizer,model.tokenizer,model.backbone,model.post_processor
        gc.collect()
        release_unused_memory(device)
        mesh=extract_mesh(model,scene,args.resolution)
        synchronize(device)
    # Drop only tiny disconnected reconstruction debris, retaining substantial
    # components (e.g. a shoe separated from a trouser cuff).
    components=mesh.split(only_watertight=False)
    area_total=sum(piece.area for piece in components)
    retained=[piece for piece in components if piece.area >= area_total*0.01]
    removed=len(components)-len(retained)
    mesh=trimesh.util.concatenate(retained)
    if not np.isfinite(mesh.vertices).all() or len(mesh.faces)==0:
        raise RuntimeError('Invalid mesh output')
    # TripoSR is Z-up. Export glTF's Y-up convention without changing scale.
    mesh.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2,[1,0,0]))
    mesh.export(args.output/'model.glb')
    mesh.export(args.output/'model.ply')
    elapsed=time.monotonic()-start
    manifest={'backend':'TripoSR','weights':'stabilityai/TripoSR','device':str(device),
              'runtime':runtime_receipt(device),
              'model_provenance':recorded_model_provenance(sources),
              'mesh_extraction':{'density_queries':str(device),
                                 'marching_cubes':'scikit-image CPU',
                                 'compiled_torchmcubes_used':False},
              'input':str(args.image.resolve()),'input_sha256':input_sha256,
              'input_sha256_scope':'Exact encoded image file passed to TripoSR before RGB decoding; this may be a processed cutout/padded derivative, not the original observed media.',
              'elapsed_seconds':round(elapsed,2),
              'resolution':args.resolution,'vertices':len(mesh.vertices),'faces':len(mesh.faces),
              'bounds':mesh.bounds.tolist(),'watertight':bool(mesh.is_watertight),
              'up_axis':'Y','scale':'arbitrary; not a measured human height',
              'tiny_components_removed':removed,
              'rigged':False,'color':'vertex colors','source_views_used':1,
              'limitations':['Single-image estimated geometry; side reference not fused.',
                             'Unobserved rear surfaces are synthesized by pretrained model.',
                             'No wardrobe separation or skeletal animation.']}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)


if __name__ == '__main__':
    main()
