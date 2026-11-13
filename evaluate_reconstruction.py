"""Independent human-outline checks; these do not certify unseen 3D anatomy."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageOps
from scipy import ndimage

ROOT=Path(__file__).resolve().parent


def outline_mask(points,size):
    mask=Image.new('L',size)
    ImageDraw.Draw(mask).polygon([(x*size[0],y*size[1]) for x,y in points],fill=255)
    return np.asarray(mask)>127


def mask_metrics(candidate,reference,tolerance_fraction=.01):
    """IoU plus symmetric boundary error, normalized by annotated subject height."""
    if candidate.shape!=reference.shape:raise ValueError('Mask sizes differ')
    if not reference.any():raise ValueError('Empty reference mask')
    ys=np.where(reference)[0];height=int(ys.max()-ys.min()+1)
    union=np.logical_or(candidate,reference).sum()
    iou=float(np.logical_and(candidate,reference).sum()/union)
    if not candidate.any():return {'silhouette_iou':0.,'boundary_f1':0.,
        'boundary_tolerance_fraction_of_height':tolerance_fraction,
        'mean_boundary_error_fraction_of_height':None,'boundary_95th_error_fraction_of_height':None,
        'substantial_connected_components':0,'vertical_region_foreground_recall':{},
        'minimum_region_recall':0.,'passes_known_view_gate':False}
    cb=candidate^ndimage.binary_erosion(candidate)
    rb=reference^ndimage.binary_erosion(reference)
    c_to_r=ndimage.distance_transform_edt(~rb)[cb]
    r_to_c=ndimage.distance_transform_edt(~cb)[rb]
    tolerance=max(1.,height*tolerance_fraction)
    precision=float((c_to_r<=tolerance).mean());recall=float((r_to_c<=tolerance).mean())
    f1=2*precision*recall/max(1e-12,precision+recall)
    symmetric=np.r_[c_to_r,r_to_c]/height
    labels,count=ndimage.label(candidate);sizes=np.bincount(labels.ravel())[1:]
    meaningful=int((sizes>=max(5,candidate.sum()*.002)).sum())
    # A missing shoe can occupy few pixels and disappear in the global IoU.
    # Report coarse vertical regions separately; these are image bands, not
    # an anatomical part detector (raised arms and bent poses cross bands).
    regional_recall={}
    for name,lo,hi in [('top',0,.15),('upper',.15,.55),('lower',.55,.85),('bottom',.85,1.)]:
        start=int(ys.min()+lo*height);stop=int(ys.min()+hi*height)
        region=reference[start:stop]
        if region.any():regional_recall[name]=float((candidate[start:stop]&region).sum()/region.sum())
    minimum_recall=min(regional_recall.values())
    return {'silhouette_iou':iou,'boundary_f1':f1,
            'boundary_tolerance_fraction_of_height':tolerance_fraction,
            'mean_boundary_error_fraction_of_height':float(symmetric.mean()),
            'boundary_95th_error_fraction_of_height':float(np.percentile(symmetric,95)),
            'substantial_connected_components':meaningful,
            'vertical_region_foreground_recall':regional_recall,
            'minimum_region_recall':minimum_recall,
            'passes_known_view_gate':bool(iou>=.90 and f1>=.90 and meaningful==1 and minimum_recall>=.90)}


def to_source(mask,receipt,source_size,output_size):
    x0,y0,x1,y1=receipt['source_bbox'];width=x1-x0;height=y1-y0
    square=receipt.get('square_size',int(np.ceil(max(width,height)/.85)))
    ox=(square-width)//2;oy=(square-height)//2
    sx=source_size[0]/output_size[0];sy=source_size[1]/output_size[1]
    coefficients=(sx*mask.width/square,0,(-x0+ox)*mask.width/square,
                  0,sy*mask.height/square,(-y0+oy)*mask.height/square)
    return np.asarray(mask.transform(output_size,Image.Transform.AFFINE,coefficients,Image.Resampling.BILINEAR))>127


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--annotations',type=Path,required=True)
    parser.add_argument('--view',choices=['front','side'],default='front')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    path=args.annotations.resolve()
    annotation=json.loads(path.read_text())
    receipt=next(r for r in json.loads((args.prepared/'manifest.json').read_text()) if Path(r['source']).stem==args.view)
    with Image.open(receipt['source']) as image:
        source_size = ImageOps.exif_transpose(image).size
    output_size=(round(source_size[0]/source_size[1]*1024),1024)
    reference=outline_mask(annotation[args.view],output_size)
    candidate=to_source(Image.open(args.prepared/f'{args.view}-cutout.png').getchannel('A'),receipt,source_size,output_size)
    # The annotation UI records one outer polygon, not interior rings. Compare
    # exterior silhouettes separately, so correctly extracted gaps are not
    # rewarded for being filled in. Never change the actual segmentation.
    outer_candidate=ndimage.binary_fill_holes(candidate)
    outer_reference=ndimage.binary_fill_holes(reference)
    metrics=mask_metrics(outer_candidate,outer_reference)
    report={'metric_version':'1.1','view':args.view,'stage':'foreground_extraction','metrics':metrics,
            'raw_mask_metrics':mask_metrics(candidate,reference),
            'boundary_tolerance_sensitivity':{str(t):mask_metrics(outer_candidate,outer_reference,t)['boundary_f1'] for t in [.005,.01,.02]},
            'annotation_topology':'One exterior polygon; enclosed gaps are ignored only in exterior-silhouette scoring. Raw masks remain untouched and raw-mask metrics are also reported.',
            'annotation':str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name,
            'gate_definition':'Provisional known-view gate: silhouette IoU >= .90, boundary F1 >= .90 within 1% subject-height tolerance, one substantial component, foreground recall >= .90 in every vertical band. Bands are image regions, not inferred anatomical parts.',
            'independence':'User drew the reference outline without model predictions. Source photograph is also a reconstruction input; this is not held-out camera validation.',
            'uncertainty':'Coarse polygon annotations omit some hair strands and small gaps. Thresholds are engineering acceptance gates, not validated anatomical-accuracy estimates.',
            'reconstruction_success':'Not established by extraction metrics alone. Requires separate geometry checks and compatible held-out views.'}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    overlay=np.zeros((*reference.shape,3),np.uint8)
    overlay[candidate&reference]=[75,150,100]
    overlay[candidate&~reference]=[230,95,65]
    overlay[reference&~candidate]=[75,150,245]
    Image.fromarray(overlay).save(args.output/'mask-comparison.png')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
