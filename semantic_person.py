"""Isolated CPU semantic segmentation, avoiding mixed OpenMP runtimes."""
import json,sys
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
ROOT=Path(__file__).resolve().parent

def sam_person_support(original, pose=None):
    """Automatic instance bounding box prompts a local SAM image-edge mask."""
    import gc,torch,zipfile
    probability,components=instance_person_support(original,probability=True)
    gc.collect()
    sys.path.insert(0,str(ROOT/'vendor/segment-anything'))
    from segment_anything import SamPredictor,sam_model_registry
    checkpoint=ROOT/'models/segmentation/sam_vit_b_01ec64.pth'
    model=sam_model_registry['vit_b']()
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=zipfile.is_zipfile(checkpoint)))
    model.eval()
    sample=original.copy();sample.thumbnail((1280,1280),Image.Resampling.LANCZOS)
    prob=np.asarray(Image.fromarray(probability).resize(sample.size,Image.Resampling.BILINEAR))/255.
    ys,xs=np.where(prob>.5)
    if not len(xs):raise RuntimeError('No person bounding box available for local segmentation.')
    margin=max(2,round((ys.max()-ys.min())*.01))
    box=np.array([max(0,xs.min()-margin),max(0,ys.min()-margin),
                  min(sample.width-1,xs.max()+margin),min(sample.height-1,ys.max()+margin)])
    predictor=SamPredictor(model)
    with torch.inference_mode():
        predictor.set_image(np.asarray(sample))
        masks,scores,_=predictor.predict(box=box,multimask_output=False)
        recovery={'attempted':False,'accepted':False}
        if pose:
            from sam_recovery import recovery_prompt,accept_recovery,diagnostics
            prompt=recovery_prompt(masks[0],pose)
            recovery['diagnostic']=diagnostics(masks[0],pose)
            if prompt:
                candidate,_,_=predictor.predict(box=np.array(prompt['box']),
                    point_coords=np.array(prompt['points']),
                    point_labels=np.ones(len(prompt['points']),dtype=int),multimask_output=False)
                accepted,details=accept_recovery(masks[0],candidate[0],pose)
                recovery.update(attempted=True,accepted=accepted,details=details)
                if accepted:masks=candidate
    # Scores are the model's estimates, not independently measured accuracy.
    mask=Image.fromarray(np.uint8(masks[0])*255).resize(original.size,Image.Resampling.BILINEAR)
    return np.asarray(mask),components,recovery

def instance_person_support(original, probability=False):
    import torch
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
    from torchvision.transforms.functional import to_tensor
    torch.set_num_threads(4)
    weights=ROOT/'models/segmentation/maskrcnn_resnet50_fpn_v2_coco-73cbd019.pth'
    model=maskrcnn_resnet50_fpn_v2(weights=None,weights_backbone=None,min_size=640,max_size=960).eval()
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True,mmap=True))
    sample=original.copy();sample.thumbnail((960,960))
    with torch.inference_mode():result=model([to_tensor(sample)])[0]
    indices=torch.where((result['labels']==1)&(result['scores']>=.75))[0]
    if not len(indices):raise RuntimeError('No confident person instance found. Choose a clearer photo.')
    areas=(result['masks'][indices,0]>.5).sum((1,2))
    index=indices[areas.argmax()]
    if probability:
        confidence=result['masks'][index,0].numpy()
        support=np.asarray(Image.fromarray(np.uint8(np.clip(confidence,0,1)*255)).resize(original.size,Image.Resampling.BILINEAR))
        return support,int((areas>areas.max()*.18).sum())
    support=ndimage.binary_dilation(result['masks'][index,0].numpy()>.4,iterations=5)
    support=np.asarray(Image.fromarray(np.uint8(support)*255).resize(original.size,Image.Resampling.NEAREST))>0
    substantial=int((areas>areas.max()*.18).sum())
    return support,substantial


def semantic_person_support(original):
    """Semantic person evidence rejects salient equipment mistaken for a person."""
    import torch
    from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
    torch.set_num_threads(4)
    weights=ROOT/'models/segmentation/deeplabv3_resnet50_coco-cd0a2569.pth'
    if not weights.exists():raise RuntimeError('Missing local semantic person weights. Run fetch_models.py --only deeplabv3 first.')
    model=deeplabv3_resnet50(weights=None,weights_backbone=None,aux_loss=True).eval()
    # Official checkpoint uses the legacy (non-ZIP) serialization format.
    model.load_state_dict(torch.load(weights,map_location='cpu',weights_only=True))
    sample=original.copy();sample.thumbnail((768,768))
    tensor=DeepLabV3_ResNet50_Weights.DEFAULT.transforms()(sample).unsqueeze(0)
    with torch.inference_mode():prob=model(tensor)['out'][0].softmax(0)[15].numpy()
    labels,count=ndimage.label(prob>.35)
    sizes=np.bincount(labels.ravel());sizes[0]=0
    if not count or sizes.max()<100:raise RuntimeError('No confident person region found. Choose a clearer, closer photo.')
    main=labels==sizes.argmax()
    components=int(np.sum(sizes>sizes.max()*.18))
    # Conservative margin preserves soft hair/clothing edges; U2Net supplies detail.
    support=ndimage.binary_dilation(main,iterations=7)
    support=np.asarray(Image.fromarray(np.uint8(support)*255).resize(original.size,Image.Resampling.NEAREST))>0
    return support,components


if __name__=='__main__':
    original=Image.open(sys.argv[1]).convert('RGB')
    backend=sys.argv[3] if len(sys.argv)>3 else 'semantic'
    metadata={}
    if backend=='sam':
        pose=json.loads(Path(sys.argv[4]).read_text()) if len(sys.argv)>4 else None
        support,components,recovery=sam_person_support(original,pose)
        metadata['pose_recovery']=recovery
    elif backend=='instance-probability':support,components=instance_person_support(original,probability=True)
    else:
        support,components=(instance_person_support if backend=='instance' else semantic_person_support)(original)
        support=np.uint8(support)*255
    Image.fromarray(support).save(sys.argv[2])
    print(json.dumps(dict(metadata,components=components)))
