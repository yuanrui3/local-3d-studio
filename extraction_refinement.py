"""Non-generative image-edge refinement of a locally detected person instance."""
import numpy as np
from PIL import Image
from scipy import ndimage


def refine_person_edges(original, saliency, person_probability):
    """Use independent person evidence to recover saliency omissions (e.g. shoes).

    GrabCut uses image colors and instance confidence, never human evaluation
    outlines. It can still fail on occlusion or similar foreground/background.
    """
    import cv2
    image=original.copy();image.thumbnail((1024,1024),Image.Resampling.LANCZOS)
    rgb=np.asarray(image)
    probability=np.asarray(Image.fromarray(person_probability).resize(image.size,Image.Resampling.BILINEAR))/255.
    salient=np.asarray(Image.fromarray(saliency).resize(image.size,Image.Resampling.BILINEAR))/255.
    support=ndimage.binary_dilation(probability>.1,iterations=5)
    labels=np.full(probability.shape,cv2.GC_PR_BGD,np.uint8)
    labels[probability>.4]=cv2.GC_PR_FGD
    # An intersection-only seed reproduces saliency's missing-shoe error.
    # Independent high-confidence instance pixels must be allowed to recover it.
    instance_interior=ndimage.binary_erosion(probability>.8,iterations=4)
    agreement_interior=ndimage.binary_erosion((probability>.5)&(salient>.95),iterations=2)
    labels[instance_interior|agreement_interior]=cv2.GC_FGD
    labels[~support]=cv2.GC_BGD
    if not (labels==cv2.GC_FGD).any():raise ValueError('No reliable foreground seed. Choose a clearer photo.')
    cv2.grabCut(rgb,labels,None,np.zeros((1,65),np.float64),np.zeros((1,65),np.float64),4,cv2.GC_INIT_WITH_MASK)
    foreground=(labels==cv2.GC_FGD)|(labels==cv2.GC_PR_FGD)
    # Keep the dominant instance; do not restore disconnected scene objects.
    components,count=ndimage.label(foreground)
    sizes=np.bincount(components.ravel());sizes[0]=0
    if not count:raise ValueError('Person edge refinement produced no foreground.')
    foreground=components==sizes.argmax()
    alpha=np.asarray(Image.fromarray(np.uint8(foreground)*255).resize(original.size,Image.Resampling.BILINEAR))
    return alpha
