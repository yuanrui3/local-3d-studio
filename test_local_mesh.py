"""Regression checks for the CPU extraction adapter's XYZ/color convention."""
import types
import unittest
import numpy as np
import torch
from generate_local import extract_mesh, recorded_model_provenance


class AnalyticRenderer:
    cfg=types.SimpleNamespace(radius=1.0)
    def query_triplane(self, decoder, points, scene):
        center=points.new_tensor([0.23,-0.14,0.37])
        density=25.0+0.3**2-((points-center)**2).sum(-1,keepdim=True)
        return {'density_act':density,'color':(points+1)/2}


class ExtractionTests(unittest.TestCase):
    def test_model_provenance_retains_pins_without_machine_local_weight_paths(self):
        import json
        sources=[{'repo':'stabilityai/TripoSR','revision':'a'*40,'file':'model.ckpt',
                  'sha256':'b'*64,'weights_manifest_sha256':'c'*64,
                  'local_path':'temporary/cache/model.ckpt'},
                 {'model':'unrelated.pth','file':'temporary/unrelated.pth'}]
        receipt=recorded_model_provenance(sources)
        self.assertEqual(receipt['assets'],[{key:sources[0][key] for key in
                                           ('repo','revision','file','sha256','weights_manifest_sha256')}])
        self.assertEqual(receipt['upstream_source']['commit'],'107cefdc244c39106fa830359024f6a2f1c78871')
        self.assertIn('not_reverified',receipt['status'])
        self.assertNotIn('temporary',json.dumps(receipt))

    def test_asymmetric_center_and_colors_survive_glb_export(self):
        import tempfile
        import trimesh
        model=types.SimpleNamespace(renderer=AnalyticRenderer(),decoder=None)
        mesh=extract_mesh(model,torch.empty(1),32)
        np.testing.assert_allclose(mesh.bounds.mean(0),[0.23,-0.14,0.37],atol=0.01)
        self.assertTrue(mesh.is_watertight)
        self.assertGreater(mesh.volume,0)
        np.testing.assert_allclose(mesh.visual.vertex_colors[:,:3]/255,
                                   (mesh.vertices+1)/2,atol=1/255+1e-5)
        with tempfile.TemporaryDirectory() as folder:
            from pathlib import Path
            output=Path(folder)/'model.glb'
            mesh.export(output)
            restored=trimesh.load(output,force='mesh')
            self.assertEqual(len(restored.faces),len(mesh.faces))
            np.testing.assert_allclose(restored.bounds,mesh.bounds,atol=1e-6)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA not available')
    def test_cuda_density_queries_keep_cpu_meshing_xyz_and_color(self):
        model=types.SimpleNamespace(renderer=AnalyticRenderer(),decoder=None)
        cpu=extract_mesh(model,torch.empty(1),32)
        # An off-centre analytic surface detects axis swaps and a CPU fallback
        # cannot satisfy a renderer that explicitly requires CUDA points.
        class CudaRenderer(AnalyticRenderer):
            def query_triplane(self,decoder,points,scene):
                if points.device.type!='cuda':
                    raise AssertionError('density/color queries left CUDA')
                return super().query_triplane(decoder,points,scene)
        model.renderer=CudaRenderer()
        checked=extract_mesh(model,torch.empty(1,device='cuda'),32)
        np.testing.assert_allclose(checked.bounds,cpu.bounds,atol=1e-5)
        self.assertEqual(len(checked.faces),len(cpu.faces))
        self.assertTrue(checked.is_watertight)
        self.assertGreater(checked.volume,0)
        np.testing.assert_allclose(checked.visual.vertex_colors[:,:3]/255,
                                   (checked.vertices+1)/2,atol=1/255+1e-5)


if __name__=='__main__':
    unittest.main()
