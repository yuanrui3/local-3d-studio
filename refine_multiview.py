"""Local two-view clothed torso refinement and visibility-aware texture baking.

Keeps a TripoSR shape prior; fits only corresponding visible garment widths.
This is sparse, approximately calibrated multi-view fitting, not photogrammetry
or Gaussian splatting. All processing uses local open-source libraries.
"""
import json
import argparse
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.optimize import minimize, least_squares
import trimesh
from numba import njit

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/multiview-v002'


@njit(cache=True)
def rasterize(uv, depth, faces, attributes, height, width):
    """Orthographic triangle rasterization with a nearest-surface depth test."""
    zbuf=np.full((height,width),-1e10,np.float32)
    out=np.zeros((height,width,attributes.shape[1]),np.float32)
    for face in faces:
        ia,ib,ic=face
        ax,ay=uv[ia];bx,by=uv[ib];cx,cy=uv[ic]
        denom=(by-cy)*(ax-cx)+(cx-bx)*(ay-cy)
        if abs(denom)<1e-9: continue
        xmin=max(0,int(np.floor(min(ax,bx,cx))))
        xmax=min(width-1,int(np.ceil(max(ax,bx,cx))))
        ymin=max(0,int(np.floor(min(ay,by,cy))))
        ymax=min(height-1,int(np.ceil(max(ay,by,cy))))
        for y in range(ymin,ymax+1):
            for x in range(xmin,xmax+1):
                a=((by-cy)*(x+.5-cx)+(cx-bx)*(y+.5-cy))/denom
                b=((cy-ay)*(x+.5-cx)+(ax-cx)*(y+.5-cy))/denom
                c=1-a-b
                if min(a,b,c)<-1e-6:continue
                z=a*depth[ia]+b*depth[ib]+c*depth[ic]
                if z>zbuf[y,x]:
                    zbuf[y,x]=z
                    out[y,x]=a*attributes[ia]+b*attributes[ib]+c*attributes[ic]
    return zbuf,out


def project(vertices, camera, view='front', vertical_map=None):
    s,cx,cy=camera
    if view=='front':
        u=-vertices[:,2]*s+cx
        depth=vertices[:,0]
    else:
        u=-vertices[:,0]*s+cx
        depth=-vertices[:,2]
    v=-vertices[:,1]*s+cy
    if vertical_map is not None:
        v=np.interp(v,vertical_map[:,0],vertical_map[:,1])
    return np.column_stack((u,v)),depth


def silhouette(vertices,faces,camera,view='front',size=256,vertical_map=None):
    uv,depth=project(vertices,camera,view,vertical_map)
    z,_=rasterize(uv*size/1024,depth,faces,np.zeros((len(vertices),1)),size,size)
    return z>-1e9


def calibrate_front(mesh,rgba):
    target=np.array(Image.fromarray(rgba[:,:,3]).resize((256,256)))>127
    # Ignore the tiny gym bracket connected to the hair mask.
    labels,n=ndimage.label(target);counts=np.bincount(labels.ravel());counts[0]=0
    target=labels==counts.argmax()
    def objective(p):
        candidate=silhouette(mesh.vertices,mesh.faces,p)
        iou=np.logical_and(candidate,target).sum()/np.logical_or(candidate,target).sum()
        return 1-iou
    result=minimize(objective,[832,512,512],method='Powell',
                    bounds=[(770,900),(485,535),(490,535)],
                    options={'maxiter':30,'xtol':0.04,'ftol':1e-5})
    return result.x,1-objective(result.x)


def fit_shape(mesh,annotation,camera):
    rows=annotation['rows']; s,cx,cy=camera
    heights=np.array([(cy-r['front_y'])/s for r in rows])
    target_width=np.array([(r['front_right']-r['front_left'])/s for r in rows])
    zcenters=np.array([(cx-(r['front_right']+r['front_left'])/2)/s for r in rows])
    xcenter=[];base_width=[];base_depth=[];selections=[]
    for y,z,w in zip(heights,zcenters,target_width):
        # Exclude arms with the observed front torso contour.
        select=(abs(mesh.vertices[:,1]-y)<.009)&(abs(mesh.vertices[:,2]-z)<w*.54)
        section=mesh.vertices[select]
        selections.append(select)
        if len(section)<10:raise RuntimeError('Too few torso vertices for a reliable section')
        base_width.append(np.ptp(section[:,2]))
        base_depth.append(np.ptp(section[:,0]))
        xcenter.append((section[:,0].max()+section[:,0].min())/2)
    base_width=np.array(base_width);base_depth=np.array(base_depth);xcenter=np.array(xcenter)
    observed=np.array([r['side_right'] is not None for r in rows])
    target_depth=np.array([(r['side_right']-r['side_left'])/s if r['side_right'] is not None else d
                           for r,d in zip(rows,base_depth)])
    def residual(parameters):
        width,depth=parameters[:len(rows)],parameters[len(rows):]
        data=np.r_[((base_width*width-target_width)*s/4),
                   ((base_depth*depth-target_depth)*s/4)[observed]]
        prior=(parameters-1)*3
        smooth=np.r_[np.diff(width,n=2),np.diff(depth,n=2)]*5
        return np.r_[data,prior,smooth]
    n=len(rows)
    result=least_squares(residual,np.ones(2*n),
                         bounds=(np.r_[np.full(n,.9),np.full(n,.85)],
                                 np.r_[np.full(n,1.1),np.full(n,1.15)]))
    sw,sd=result.x[:n],result.x[n:]
    # Unknown upper-back depth stays anchored to the nearest observed section.
    sd[~observed]=1
    order=np.argsort(heights)
    def interp(values,y):return np.interp(y,heights[order],np.array(values)[order])
    verts=mesh.vertices.copy();ys=verts[:,1]
    zc=interp(zcenters,ys);half=interp(target_width,ys)/2
    feather=np.clip((1.10-abs(verts[:,2]-zc)/half)/.15,0,1)
    feather*=np.clip((ys-(heights.min()-.03))/.03,0,1)
    feather*=np.clip(((heights.max()+.03)-ys)/.03,0,1)
    xc=interp(xcenter,ys)
    verts[:,0]=xc+(verts[:,0]-xc)*(1+feather*(interp(sd,ys)-1))
    verts[:,2]=zc+(verts[:,2]-zc)*(1+feather*(interp(sw,ys)-1))
    refined=mesh.copy();refined.vertices=verts
    actual_width=np.array([np.ptp(verts[select,2]) for select in selections])
    actual_depth=np.array([np.ptp(verts[select,0]) for select in selections])
    side_cx=float(np.median([(r['side_left']+r['side_right'])/2+s*x
                            for r,x,ok in zip(rows,xcenter,observed) if ok]))
    metrics={'before_width_rmse_pixels':float(np.sqrt(np.mean(((base_width-target_width)*s)**2))),
             'after_width_rmse_pixels':float(np.sqrt(np.mean(((actual_width-target_width)*s)**2))),
             'before_depth_rmse_pixels':float(np.sqrt(np.mean(((base_depth-target_depth)[observed]*s)**2))),
             'after_depth_rmse_pixels':float(np.sqrt(np.mean(((actual_depth-target_depth)[observed]*s)**2))),
             'width_factors':sw.tolist(),'depth_factors':sd.tolist(),
             'max_vertex_displacement_arbitrary_units':float(np.linalg.norm(verts-mesh.vertices,axis=1).max()),
             'metric_scope':'Fit residual on selected garment sections, not 3D ground-truth accuracy.'}
    return refined,np.array([s,side_cx,cy]),metrics


def bilinear(array,uv):
    coords=[uv[:,1]-.5,uv[:,0]-.5]
    if array.ndim==2:return ndimage.map_coordinates(array,coords,order=1,mode='constant',cval=0)
    return np.stack([ndimage.map_coordinates(array[:,:,i],coords,order=1,mode='constant',cval=0)
                     for i in range(array.shape[2])],axis=1)


def bake(mesh,front,side,front_camera,side_camera,vertical_map,annotations,size=2048):
    import xatlas
    print('Unwrapping UV atlas',flush=True)
    atlas=xatlas.Atlas();atlas.add_mesh(mesh.vertices.astype(np.float32),mesh.faces.astype(np.uint32))
    options=xatlas.PackOptions();options.resolution=size;options.padding=6
    atlas.generate(pack_options=options)
    mapping,faces,uv=atlas[0]
    verts=mesh.vertices[mapping];vertex_normals=mesh.vertex_normals[mapping]
    base=mesh.visual.vertex_colors[mapping,:3].astype(float)/255
    attrs=np.column_stack([verts,vertex_normals,base])
    uv_pixels=np.column_stack([uv[:,0]*size,(1-uv[:,1])*size])
    depth,canvas=rasterize(uv_pixels,np.zeros(len(verts)),faces,attrs,size,size)
    valid=depth>-1e9
    values=canvas[valid];points=values[:,:3];normals=values[:,3:6]
    colors=values[:,6:9].copy();source_counts={}
    for name,photo,camera,warp,axis,sign in [
        ('side',side,side_camera,vertical_map,2,-1),
        ('front',front,front_camera,None,0,1)]:
        coords,z=project(points,camera,name,warp)
        vertex_uv,vertex_z=project(mesh.vertices,camera,name,warp)
        zbuffer,_=rasterize(vertex_uv,vertex_z,mesh.faces,np.zeros((len(mesh.vertices),1)),1024,1024)
        visible=(abs(bilinear(zbuffer,coords)-z)<.012)
        sampled=bilinear(photo.astype(np.float32)/255,coords)
        alpha=sampled[:,3]
        facing=np.clip((normals[:,axis]*sign-.10)/.55,0,1)
        weight=facing*visible*(alpha>.98)
        if name=='side':
            # Side arm occludes the torso. Only observed red garment pixels
            # contribute; never bake the arm or punching bag onto the body.
            red=(sampled[:,0]>sampled[:,1]*1.3)&(sampled[:,0]>sampled[:,2]*1.1)
            inside=(coords[:,0]>415)&(coords[:,0]<550)&(coords[:,1]>265)&(coords[:,1]<555)
            front_uv,_=project(points,front_camera,'front')
            rows=annotations['rows']
            row_y=[r['front_y'] for r in rows]
            left=np.interp(front_uv[:,1],row_y,[r['front_left'] for r in rows])
            right=np.interp(front_uv[:,1],row_y,[r['front_right'] for r in rows])
            torso=(front_uv[:,0]>left+3)&(front_uv[:,0]<right-3)
            base_red=(values[:,6]>values[:,7]*1.7)&(values[:,6]>values[:,8]*1.25)
            # Both source and target must be garment. Otherwise a pose change
            # would project red shorts onto an arm occupying the same ray.
            weight*=red*inside*torso*base_red*.65
        else:
            # Feather the source cutout boundary to avoid background bleed.
            interior=ndimage.distance_transform_edt(photo[:,:,3]>250)
            weight*=np.clip(bilinear(interior,coords)/5,0,1)
        colors=colors*(1-weight[:,None])+sampled[:,:3]*weight[:,None]
        source_counts[name]=int((weight>.1).sum())
    texture=np.zeros((size,size,3),np.uint8);texture[valid]=np.clip(colors*255,0,255).astype(np.uint8)
    # Fill small chart gutters for bilinear filtering without changing UVs.
    distance,nearest=ndimage.distance_transform_edt(~valid,return_indices=True)
    padding=(distance<=6)&~valid
    texture[padding]=texture[nearest[0][padding],nearest[1][padding]]
    image=Image.fromarray(texture);image.save(OUT/'albedo.png')
    material=trimesh.visual.material.PBRMaterial(baseColorTexture=image,metallicFactor=0,roughnessFactor=1,doubleSided=False)
    result=trimesh.Trimesh(vertices=verts,faces=faces,vertex_normals=vertex_normals,process=False)
    result.visual=trimesh.visual.TextureVisuals(uv=uv,material=material)
    return result,source_counts


def render_debug(mesh,front,side,fcam,scam,warp,annotations,label):
    tiles=[]
    for name,photo,camera,vwarp in [('front',front,fcam,None),('side',side,scam,warp)]:
        uv,z=project(mesh.vertices,camera,name,vwarp)
        depth,color=rasterize(uv,z,mesh.faces,mesh.visual.vertex_colors[:,:3]/255.,1024,1024)
        im=np.full((1024,1024,3),235,np.uint8);mask=depth>-1e9;im[mask]=np.clip(color[mask]*255,0,255).astype(np.uint8)
        original=Image.fromarray(photo).convert('RGB');draw=ImageDraw.Draw(original)
        for row in annotations['rows']:
            y=row[name+'_y'];left=row[name+'_left'];right=row[name+'_right']
            draw.ellipse((left-3,y-3,left+3,y+3),fill='cyan')
            if right is not None:draw.line((left,y,right,y),fill='cyan',width=2)
        tiles.extend([original.resize((384,384)),Image.fromarray(im).resize((384,384))])
    sheet=Image.new('RGB',(1536,420),'#202128')
    for i,tile in enumerate(tiles):sheet.paste(tile,(i*384,30))
    d=ImageDraw.Draw(sheet)
    for i,text in enumerate(['Front reference · annotated garment','Front projection · '+label,'Side reference · partial constraints','Side projection · '+label]):d.text((i*384+10,10),text,fill='white')
    sheet.save(OUT/f'{label}-fit.jpg')


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',type=Path,default=ROOT/'outputs/local-v001/model.glb')
    parser.add_argument('--front',type=Path,required=True)
    parser.add_argument('--side',type=Path,required=True)
    parser.add_argument('--camera',type=float,nargs=3,metavar=('SCALE','CX','CY'),help='Use a fixed reviewed camera instead of fitting one to this cutout.')
    parser.add_argument('--annotations',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args();OUT=args.output
    start=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    annotations=json.loads(args.annotations.read_text())
    front=np.array(Image.open(args.front).convert('RGBA').resize((1024,1024),Image.Resampling.LANCZOS))
    side=np.array(Image.open(args.side).convert('RGBA').resize((1024,1024),Image.Resampling.LANCZOS))
    mesh=trimesh.load(args.base,force='mesh')
    if args.camera:
        camera=np.array(args.camera)
        if not np.isfinite(camera).all() or camera[0]<=0:raise ValueError('Invalid fixed camera')
        target=np.array(Image.fromarray(front[:,:,3]).resize((256,256)))>127
        projected=silhouette(mesh.vertices,mesh.faces,camera)
        iou=float((target&projected).sum()/(target|projected).sum())
    else:
        print('Calibrating approximate front camera',flush=True)
        camera,iou=calibrate_front(mesh,front)
    print('Camera',camera,'silhouette IoU',iou,flush=True)
    refined,side_camera,metrics=fit_shape(mesh,annotations,camera)
    warp=np.array(annotations['vertical_correspondences'],float)
    render_debug(mesh,front,side,camera,side_camera,warp,annotations,'before')
    render_debug(refined,front,side,camera,side_camera,warp,annotations,'after')
    assert np.isfinite(refined.vertices).all() and refined.is_watertight
    refined.export(OUT/'geometry.ply')
    textured,counts=bake(refined,front,side,camera,side_camera,warp,annotations)
    # Source photos already contain lighting. Preserve their appearance instead
    # of applying the viewer's lighting a second time.
    def unlit(tree):
        tree.setdefault('extensionsUsed',[]).append('KHR_materials_unlit')
        for material in tree.get('materials',[]):
            material.setdefault('extensions',{})['KHR_materials_unlit']={}
    from trimesh.exchange.gltf import export_glb
    (OUT/'model.glb').write_bytes(export_glb(trimesh.Scene(textured),tree_postprocessor=unlit))
    manifest={'method':'TripoSR prior + sparse two-view garment silhouette fitting + visibility-aware photo texture baking',
              'gaussian_splatting':False,'photos_used':2,'video_frames_used':0,
              'front_camera':camera.tolist(),'side_camera':side_camera.tolist(),'camera_model':'approximate orthographic',
              'camera_selection':'fixed reviewed camera' if args.camera else 'optimized against input cutout',
              'front_silhouette_iou_before':iou,'fit':metrics,'texture_pixels_from_sources':counts,
              'appearance':'unlit photographic texture; lighting is baked into source images',
              'elapsed_seconds':round(time.monotonic()-start,2),'faces':len(textured.faces),
              'texture_resolution':[2048,2048],'rigged':False,
              'limitations':['Partial torso constraints only; changing limb poses are excluded.',
                             'No rear source: rear remains inferred from the single-image prior.',
                             'Approximate calibration; lower fit residual is not measured 3D accuracy.',
                             'Texture includes photographic lighting; view transitions may show seams.',
                             'No clothing separation or animation.']}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)


if __name__=='__main__':main()
