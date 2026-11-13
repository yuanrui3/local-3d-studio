"""Conservative automatic prompts for a visibly truncated person mask.

Pose points select observed image regions, not an anatomical template. These
heuristics cannot establish correct clothing edges; cutout review is required.
"""
import numpy as np
from scipy import ndimage

POINTS = ['nose', 'left_hip', 'right_hip', 'left_knee', 'right_knee',
          'left_ankle', 'right_ankle']


def diagnostics(mask, pose):
    mask = np.asarray(mask, bool)
    if mask.ndim != 2 or not mask.any() or not pose or not pose.get('person_detected'):
        return None
    # Match the preparation pipeline's component cleanup. Tiny detached skirt
    # fragments must not make an upper-body-only mask look full-height.
    labels, _ = ndimage.label(mask)
    areas = np.bincount(labels.ravel()); areas[0] = 0
    mask = mask & ndimage.binary_dilation(labels == areas.argmax(), iterations=3)
    if pose.get('substantial_people', 1) != 1:
        return None
    h, w = mask.shape
    size = np.asarray(pose.get('image_size', []), float)
    if size.shape != (2,) or np.any(size <= 0) or abs((w/h)/(size[0]/size[1])-1) > .01:
        return None
    box = np.asarray(pose.get('person_box_normalized', []), float)
    if box.shape != (4,) or not np.isfinite(box).all() or np.any(box[2:] <= box[:2]):
        return None
    box = box*np.tile([w, h], 2)
    ys, xs = np.where(mask)
    person_height = box[3]-box[1]
    margin = max(2., person_height*.03)
    near_box = ((xs >= box[0]-margin) & (xs <= box[2]+margin) &
                (ys >= box[1]-margin) & (ys <= box[3]+margin))
    if near_box.mean() < .95:
        return None
    distance = ndimage.distance_transform_edt(~mask)
    points, missing, names = [], [], []
    for name in POINTS:
        p = pose.get('keypoints', {}).get(name, {})
        xy = np.asarray(p.get('xy', []), float)
        score=p.get('heatmap_score', 0)
        if not np.isfinite(score) or score < 4 or xy.shape != (2,) or not np.isfinite(xy).all():
            continue
        x, y = xy*[w, h]
        if not 0 <= x < w or not 0 <= y < h:
            continue
        points.append([float(x), float(y)]); names.append(name)
        if distance[int(y), int(x)] > max(2., person_height*.01):
            missing.append(name)
    return {'box': box.tolist(), 'points': points, 'point_names': names,
            'missing': missing, 'height_ratio': float((ys.max()-ys.min()+1)/person_height),
            'foreground_pixels': int(mask.sum()), 'inside_person_box_fraction': float(near_box.mean())}


def recovery_prompt(mask, pose):
    d = diagnostics(mask, pose)
    if (not d or len(d['points']) < 4 or d['height_ratio'] >= .8 or
            len([n for n in d['missing'] if n != 'nose']) < 3):
        return None
    return d


def accept_recovery(before, after, pose):
    old, new = diagnostics(before, pose), diagnostics(after, pose)
    if not old or not new:
        return False, {'reason': 'Recovered mask and detected person do not agree.'}
    retained = float((np.asarray(before, bool) & np.asarray(after, bool)).sum()/max(1, np.asarray(before, bool).sum()))
    accepted = (new['height_ratio'] >= .85 and len(new['missing']) < len(old['missing']) and
                len(new['missing']) <= 1 and retained >= .95 and
                new['foreground_pixels'] <= old['foreground_pixels']*8)
    return bool(accepted), {'before': old, 'after': new, 'original_foreground_retained': retained,
                            'reason': 'Pose consistency improved; visual review still required.' if accepted else
                                      'Recovery did not satisfy the conservative coverage/preservation checks.'}
