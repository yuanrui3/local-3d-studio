"""Local person segmentation and padding, without changing the person."""
import argparse
import json
import os
from pathlib import Path
from local_runtime import ROOT, configure

configure()
os.environ['OMP_NUM_THREADS'] = '4'
import numpy as np
from PIL import Image, ImageOps
import rembg
from scipy import ndimage


def semantic_person_support(original,backend='semantic',pose=None,details=None):
    # Torchvision and rembg/Numba link separate OpenMP runtimes on this Mac.
    # Isolate inference rather than disabling duplicate-runtime safety checks.
    import subprocess,sys,tempfile
    with tempfile.TemporaryDirectory(prefix='local3d-semantic-') as folder:
        source=Path(folder)/'source.png';mask=Path(folder)/'mask.png'
        original.save(source)
        command=[sys.executable,str(ROOT/'semantic_person.py'),str(source),str(mask),backend]
        if pose is not None:
            pose_file=Path(folder)/'pose.json';pose_file.write_text(json.dumps(pose));command.append(str(pose_file))
        result=subprocess.run(command,
                              capture_output=True,text=True,timeout=180)
        if result.returncode:
            raise RuntimeError('Local person inference failed: '+result.stderr[-1500:])
        metadata=json.loads(result.stdout)
        if details is not None:details.update(metadata)
        value=np.asarray(Image.open(mask))
        return (value if backend in ('instance-probability','sam') else value>0),metadata['components']


def mask_quality(alpha, semantic_components=1, removed_fraction=0., recovered_fraction=0., agreement_iou=1.):
    ys,xs=np.where(alpha>127)
    if not len(xs):raise RuntimeError('Extraction produced an empty subject.')
    h,w=alpha.shape;warnings=[]
    fraction=len(xs)/(h*w)
    if min(xs.min(),ys.min(),w-1-xs.max(),h-1-ys.max())<3:
        warnings.append('The subject touches the source image edge; head, feet or limbs may be cropped.')
    if semantic_components>1:warnings.append('More than one substantial person region was detected. Check which person was kept.')
    if fraction<.04:warnings.append('The subject occupies little of the photo; fine detail may be unreliable.')
    if max(removed_fraction,recovered_fraction)>.15:warnings.append('The segmentation models disagreed substantially. Inspect the cutout carefully.')
    return {'foreground_fraction':round(fraction,4),'semantic_person_components':semantic_components,
            'semantic_removed_fraction':round(removed_fraction,4),
            'recovered_foreground_fraction':round(recovered_fraction,4),'mask_agreement_iou':round(agreement_iou,4),
            'warnings':warnings,
            'review_required':'Check the complete head, hands, clothing and shoes; segmentation does not certify anatomical accuracy.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('images', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs'/'prepared')
    parser.add_argument('--keep-detached',action='store_true',help='Keep disconnected foreground components.')
    parser.add_argument('--semantic',action='store_true',help='Constrain foreground to a semantic person mask before reconstruction.')
    parser.add_argument('--instance',action='store_true',help='Experimental instance-person support; evaluate before adopting.')
    parser.add_argument('--instance-refine',action='store_true',help='Experimental instance confidence and image-edge refinement.')
    parser.add_argument('--sam',action='store_true',help='Experimental instance-box prompted SAM segmentation, entirely local.')
    parser.add_argument('--pose-record',type=Path,help='Existing local pose record for one input; enables conservative partial-mask recovery.')
    args = parser.parse_args()
    if args.pose_record and len(args.images)!=1:parser.error('--pose-record applies to exactly one source image.')
    pose=json.loads(args.pose_record.read_text()) if args.pose_record else None
    args.output.mkdir(parents=True, exist_ok=True)
    model = ROOT / '.cache' / 'rembg' / 'u2net_human_seg.onnx'
    if not model.exists():
        raise SystemExit('Missing segmentation weights. Run fetch_models.py first.')
    session = rembg.new_session('u2net_human_seg', providers=['CPUExecutionProvider'])
    manifest=args.output/'manifest.json'
    receipts=json.loads(manifest.read_text()) if manifest.exists() else []
    for path in args.images:
        print(f'Segmenting {path.name}', flush=True)
        original = ImageOps.exif_transpose(Image.open(path)).convert('RGB')
        rgba = rembg.remove(original, session=session, post_process_mask=True)
        alpha = np.asarray(rgba.getchannel('A'))
        initial_mask=alpha>127
        components=1;removed=0.;sam_details={}
        if args.sam:
            before=max(1,int((alpha>127).sum()))
            alpha,components=semantic_person_support(original,'sam',pose,sam_details)
            removed=max(0.,1-int((alpha>127).sum())/before)
        elif args.instance_refine:
            from extraction_refinement import refine_person_edges
            probability,components=semantic_person_support(original,'instance-probability')
            before=max(1,int((alpha>127).sum()))
            alpha=refine_person_edges(original,alpha,probability)
            removed=max(0.,1-int((alpha>127).sum())/before)
        elif args.semantic or args.instance:
            person,components=semantic_person_support(original,'instance' if args.instance else 'semantic')
            before=max(1,int((alpha>127).sum()))
            alpha=np.where(person,alpha,0).astype(np.uint8)
            removed=1-int((alpha>127).sum())/before
        # Remove detached background islands, preserving
        # the original soft edge around the largest connected person region.
        labels,count = ndimage.label(alpha > 32)
        if not count:
            raise RuntimeError(f'No person found in {path}')
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        support = ndimage.binary_dilation(labels == sizes.argmax(),iterations=3)
        if not args.keep_detached:
            alpha = np.where(support,alpha,0).astype(np.uint8)
        # rembg can zero RGB where its original alpha was zero. Replacing only
        # that alpha would recover black pixels instead of the missing shoe.
        # Always composite refined masks with the untouched source RGB.
        rgba=original.convert('RGBA')
        rgba.putalpha(Image.fromarray(alpha))
        final_mask=alpha>127
        removed=float((initial_mask&~final_mask).sum()/max(1,initial_mask.sum()))
        recovered=float((final_mask&~initial_mask).sum()/max(1,final_mask.sum()))
        agreement=float((initial_mask&final_mask).sum()/max(1,(initial_mask|final_mask).sum()))
        quality=mask_quality(alpha,components,removed,recovered,agreement)
        if args.sam:quality['pose_recovery']=sam_details.get('pose_recovery',{'attempted':False,'accepted':False})
        ys, xs = np.where(alpha > 32)
        if xs.size == 0:
            raise RuntimeError(f'No person found in {path}')
        bbox = (int(xs.min()), int(ys.min()), int(xs.max()+1), int(ys.max()+1))
        observed_bbox=bbox
        size=int(np.ceil(max(bbox[2]-bbox[0],bbox[3]-bbox[1])/.85))
        ox=(size-bbox[2]+bbox[0])//2;oy=(size-bbox[3]+bbox[1])//2
        # Crop the full padded square, not just the old tight bounding box.
        # Recovered extremities can extend beyond a legacy mask's old bounds.
        square=rgba.crop((bbox[0]-ox,bbox[1]-oy,bbox[0]-ox+size,bbox[1]-oy+size))
        square.thumbnail((1024,1024), Image.Resampling.LANCZOS)
        cutout = args.output / f'{path.stem}-cutout.png'
        square.save(cutout)
        background = Image.new('RGBA', square.size, (128,128,128,255))
        composite = Image.alpha_composite(background, square).convert('RGB')
        prepared = args.output / f'{path.stem}-input.png'
        composite.save(prepared)
        receipts=[r for r in receipts if Path(r['source']).stem!=path.stem]
        receipts.append({'source':str(path.resolve()),'cutout':str(cutout.resolve()),
                         'input':str(prepared.resolve()),'source_bbox':bbox,
                         'observed_subject_bbox':observed_bbox,
                         'method':('MaskRCNN box + SAM ViT-B' if args.sam else 'u2net_human_seg'+(' + MaskRCNN confidence + GrabCut image edges' if args.instance_refine else ' + MaskRCNN instance-person support' if args.instance else ' + DeepLabV3 semantic person support' if args.semantic else ''))+', local CPU, no generative edits',
                         'source_size':original.size,'square_size':size,'quality':quality})
        print(f'Saved {prepared}', flush=True)
    (args.output/'manifest.json').write_text(json.dumps(receipts,indent=2)+'\n')


if __name__ == '__main__':
    main()
