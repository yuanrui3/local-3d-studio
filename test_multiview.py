import unittest
import numpy as np
from refine_multiview import rasterize,project


class ProjectionTests(unittest.TestCase):
    def test_known_orthogonal_views_and_local_pose_alignment(self):
        points=np.array([[.1,.2,.3],[-.2,-.1,.4]])
        front,depth=project(points,[100,50,60])
        np.testing.assert_allclose(front,[[20,40],[10,70]])
        np.testing.assert_allclose(depth,[.1,-.2])
        side,depth=project(points,[100,50,60],'side',np.array([[0,10],[100,110]]))
        np.testing.assert_allclose(side,[[40,50],[70,80]])
        np.testing.assert_allclose(depth,[-.3,-.4])

    def test_nearer_surface_occludes_farther_texture(self):
        uv=np.array([[1.,1.],[6.,1.],[1.,6.]]*2)
        depth=np.array([1.,1.,1.,2.,2.,2.])
        faces=np.array([[0,1,2],[3,4,5]])
        colors=np.array([[1.,0.,0.]]*3+[[0.,1.,0.]]*3)
        z,render=rasterize(uv,depth,faces,colors,8,8)
        self.assertEqual(z[2,2],2)
        np.testing.assert_allclose(render[2,2],[0,1,0])
        self.assertLess(z[7,7],-1e9)


if __name__=='__main__':unittest.main()
