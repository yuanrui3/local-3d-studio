"""Fixed-camera clothed-mesh silhouette audit against the user's source outline.

This audits the observed front projection, not unseen anatomy. The supplied
camera is reused unchanged for every candidate and is never fit to this outline.
"""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
import trimesh
from evaluate_reconstruction import outline_mask,to_source,mask_metrics
from refine_multiview import silhouette

ROOT=Path(__file__).resolve().parent


def evaluate(mesh_path,camera_manifest,prepared,annotation_path,output):
    mesh=trimesh.load(mesh_path,force='mesh',process=False)
    calibration=json.loads(camera_manifest.read_text())
    camera=calibration['front_camera']
    receipts=json.loads((prepared/'manifest.json').read_text())
    receipt=next(x for x in receipts if Path(x['source']).stem=='front')
    source_size=tuple(receipt['source_size'])
    output_size=(round(source_size[0]/source_size[1]*1024),1024)
    raster=silhouette(mesh.vertices,mesh.faces,camera,size=1024)
    candidate=to_source(Image.fromarray(np.uint8(raster)*255),receipt,source_size,output_size)
    annotation=json.loads(annotation_path.read_text())
    reference=outline_mask(annotation['front'],output_size)
    exterior=ndimage.binary_fill_holes(candidate)
    metrics=mask_metrics(exterior,ndimage.binary_fill_holes(reference))
    topology=mesh.copy();topology.merge_vertices(merge_tex=True,merge_norm=True)
    parts=topology.split(only_watertight=False)
    areas=np.array([part.area for part in parts])
    substantial=int((areas>=max(1e-10,topology.area*.001)).sum())
    finite=bool(np.isfinite(mesh.vertices).all() and np.isfinite(mesh.vertex_normals).all())
    report={
        'metric_version':'1.1','stage':'known_view_mesh_projection',
        'model':str(mesh_path),'camera':camera,'camera_manifest':str(camera_manifest),
        'camera_policy':'Supplied camera is fixed during this audit, not fitted to the user outline. Compare models only after checking whether the camera and crop are shared or separately source-calibrated.',
        'metrics':metrics,'raw_mask_metrics':mask_metrics(candidate,reference),
        'boundary_tolerance_sensitivity':{str(t):mask_metrics(exterior,reference,t)['boundary_f1'] for t in [.005,.01,.02]},
        'geometry':{'vertices':len(mesh.vertices),'triangles':len(mesh.faces),
                    'finite':finite,'substantial_surface_components':substantial,
                    'component_area_fractions':sorted((areas/max(topology.area,1e-10)).tolist(),reverse=True)[:10],
                    'watertight':bool(topology.is_watertight)},
        'known_view_gate_passes':bool(finite and substantial==1 and metrics['passes_known_view_gate']),
        'held_out_reconstruction_success':None,
        'limitations':['This photograph participated in reconstruction and camera alignment. It is not a held-out view.',
                       'A 2D silhouette cannot measure depth, internal anatomy, real-world scale or rear-view correctness.',
                       'Coarse human polygon has uncertain hair edges and does not specify interior gaps.',
                       'Only the supplied front projection is scored.']}
    output.mkdir(parents=True,exist_ok=True)
    (output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    overlay=np.zeros((*reference.shape,3),np.uint8)
    overlay[candidate&reference]=[75,150,100]
    overlay[candidate&~reference]=[230,95,65]
    overlay[reference&~candidate]=[75,150,245]
    Image.fromarray(overlay).save(output/'mesh-outline-comparison.png')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh',type=Path,required=True)
    parser.add_argument('--camera-manifest',type=Path,required=True)
    parser.add_argument('--prepared',type=Path,default=ROOT/'outputs/prepared')
    parser.add_argument('--annotations',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(evaluate(args.mesh,args.camera_manifest,args.prepared,args.annotations,args.output),indent=2))
