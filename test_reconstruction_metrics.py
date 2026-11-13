import unittest
import numpy as np
from evaluate_reconstruction import mask_metrics

class ReconstructionMetricsTests(unittest.TestCase):
    def test_exact_outline_is_perfect(self):
        a=np.zeros((100,100),bool);a[10:90,30:70]=True
        result=mask_metrics(a,a)
        self.assertEqual(result['silhouette_iou'],1)
        self.assertEqual(result['boundary_f1'],1)
        self.assertTrue(result['passes_known_view_gate'])
    def test_shifted_and_empty_shapes_fail(self):
        a=np.zeros((100,100),bool);a[10:90,30:70]=True
        self.assertFalse(mask_metrics(np.roll(a,12,axis=1),a)['passes_known_view_gate'])
        self.assertFalse(mask_metrics(np.zeros_like(a),a)['passes_known_view_gate'])
    def test_floaters_do_not_hide_in_overlap_score(self):
        a=np.zeros((100,100),bool);a[10:90,30:70]=True
        b=a.copy();b[15:20,80:85]=True
        result=mask_metrics(b,a)
        self.assertGreater(result['silhouette_iou'],.98)
        self.assertEqual(result['substantial_connected_components'],2)
        self.assertFalse(result['passes_known_view_gate'])
    def test_missing_extremity_does_not_hide_in_global_overlap(self):
        reference=np.zeros((120,120),bool)
        reference[10:90,30:90]=True
        reference[90:110,40:45]=True
        reference[90:110,75:80]=True
        candidate=reference.copy();candidate[90:110,75:80]=False
        result=mask_metrics(candidate,reference)
        self.assertGreater(result['silhouette_iou'],.97)
        self.assertLess(result['minimum_region_recall'],.6)
        self.assertFalse(result['passes_known_view_gate'])

if __name__=='__main__':unittest.main()
