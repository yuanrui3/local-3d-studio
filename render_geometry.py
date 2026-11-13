"""Fixed-camera neutral surface renders for geometry regression review."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import trimesh
from refine_multiview import rasterize


def render(mesh,camera,size=512):
    v=mesh.vertices;n=mesh.vertex_normals;s,cx,cy=np.array(camera)*size/1024
    tiles=[]
    for title,horizontal,depth,eye in [('Front',-v[:,2],v[:,0],[1,.35,-.25]),
                                      ('Side',-v[:,0],-v[:,2],[.25,.35,-1]),
                                      ('Back',v[:,2],-v[:,0],[-1,.35,.25])]:
        light=np.array(eye);light=light/np.linalg.norm(light)
        shade=np.clip(n@light,0,1)*.65+.25
        uv=np.column_stack([horizontal*s+cx,-v[:,1]*s+cy])
        z,pixels=rasterize(uv,depth,mesh.faces,shade[:,None],size,size)
        mask=z>-1e9
        canvas=np.full((size,size,3),[30,33,39],np.uint8)
        canvas[mask]=(pixels[mask]*np.array([200,210,220])).clip(0,255)
        image=Image.fromarray(canvas);ImageDraw.Draw(image).text((15,12),title,fill='white')
        tiles.append(image)
    joined=Image.new('RGB',(size*3,size))
    for i,tile in enumerate(tiles):joined.paste(tile,(i*size,0))
    return joined


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mesh',type=Path,required=True)
    p.add_argument('--camera-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    render(trimesh.load(args.mesh,force='mesh',process=False),json.loads(args.camera_manifest.read_text())['front_camera']).save(args.output)
