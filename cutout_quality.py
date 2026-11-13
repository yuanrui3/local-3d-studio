"""Check local pose landmarks against a cutout without changing its pixels.

Landmark heatmap scores are uncalibrated. Coverage is a review diagnostic, not
proof of visibility, identity, complete clothing, or anatomical correctness.
"""
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def source_to_cutout(points,receipt,cutout_size):
    """Map normalized EXIF-oriented source coordinates through padded cropping."""
    source_size=np.asarray(receipt['source_size'],float)
    x0,y0,x1,y1=receipt['source_bbox'];square=float(receipt['square_size'])
    if square<=0 or np.any(source_size<=0):raise ValueError('Invalid source-to-cutout mapping')
    offset=np.array([(square-x1+x0)//2,(square-y1+y0)//2])
    origin=np.array([x0,y0])-offset
    return (np.asarray(points,float)*source_size-origin)*np.asarray(cutout_size)/square


def box_iou(a,b):
    a=np.asarray(a,float);b=np.asarray(b,float)
    overlap=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    union=np.maximum(0,a[2:]-a[:2]).prod()+np.maximum(0,b[2:]-b[:2]).prod()-overlap
    return float(overlap/max(union,1e-12))


def unavailable(reason):
    return {'status':'unavailable','coverage_fraction':None,'checked_landmarks':0,
            'warnings':[reason],'summary':'Automatic landmark coverage could not be assessed. Review the complete cutout.'}


def check_cutout(alpha,receipt,pose,score_threshold=4.,tolerance_fraction=.01):
    if not pose or not pose.get('person_detected'):
        return unavailable('No usable person pose estimate was available for this reference.')
    source=np.asarray(receipt['source_size'],float);pose_size=np.asarray(pose['image_size'],float)
    if np.any(source<=0) or np.any(pose_size<=0) or abs((source[0]/source[1])/(pose_size[0]/pose_size[1])-1)>.01:
        return unavailable('Pose and cutout image coordinates do not match.')
    alpha=np.asarray(alpha)
    if alpha.ndim!=2:raise ValueError('Expected a single-channel cutout alpha mask')
    h,w=alpha.shape;mask=alpha>127
    if not mask.any():return unavailable('The cutout is empty.')
    observed=np.asarray(receipt.get('observed_subject_bbox',receipt['source_bbox']),float)/np.tile(source,2)
    agreement=box_iou(observed,pose['person_box_normalized'])
    if agreement<.3:
        result=unavailable('Pose and cutout describe different image regions; confirm the intended person.')
        result.update(status='subject_region_mismatch',person_box_iou=agreement)
        return result
    subject_height=(pose['person_box_normalized'][3]-pose['person_box_normalized'][1])*source[1]*h/receipt['square_size']
    if subject_height<=0:return unavailable('The pose estimate has an invalid person extent.')
    boundary=mask&~ndimage.binary_erosion(mask);y,x=np.where(boundary)
    tree=cKDTree(np.column_stack([x+.5,y+.5]));checked=[];ignored=[]
    tolerance=max(2.,tolerance_fraction*subject_height)
    for name,point in pose.get('keypoints',{}).items():
        score=float(point.get('heatmap_score',-1));xy=np.asarray(point.get('xy',[]),float)
        if score<score_threshold or not np.isfinite(score) or xy.shape!=(2,) or not np.isfinite(xy).all():
            ignored.append(name);continue
        pixel=source_to_cutout(xy,receipt,(w,h))
        inside=bool(0<=pixel[0]<w and 0<=pixel[1]<h)
        distance=0. if inside and mask[int(pixel[1]),int(pixel[0])] else float(tree.query(pixel)[0])
        checked.append({'name':name,'heatmap_score':score,'distance_to_mask_pixels':distance,
                        'distance_fraction_of_person_height':distance/subject_height,
                        'inside_canvas':inside,'near_mask':bool(inside and distance<=tolerance)})
    if not checked:return unavailable('All pose landmarks were too uncertain for a cutout coverage check.')
    outside=[point['name'] for point in checked if not point['near_mask']]
    warnings=[]
    if outside:warnings.append('The cutout may exclude the estimated '+', '.join(name.replace('_',' ') for name in outside)+'. Inspect those areas.')
    if ignored:warnings.append('Uncertain landmarks were not checked: '+', '.join(name.replace('_',' ') for name in ignored)+'.')
    summary=f'{len(checked)-len(outside)} of {len(checked)} estimated joint positions are near the retained subject. Check hair, fingers, clothing and shoes visually.'
    return {'status':'checked' if not ignored else 'partial','method':'Pose landmark proximity to extracted alpha',
            'checked_landmarks':len(checked),'ignored_landmarks':ignored,'outside_landmarks':outside,
            'coverage_fraction':float((len(checked)-len(outside))/len(checked)),
            'person_box_iou':agreement,'tolerance_pixels':tolerance,'tolerance_fraction_of_person_height':tolerance_fraction,
            'score_threshold':score_threshold,'landmarks':checked,'warnings':warnings,'summary':summary,
            'limitations':'Heuristic pose-to-mask consistency only. Low-score or occluded joints, fingers, toes, hair and clothing edges are not verified. No identity check or anatomical accuracy claim.'}
