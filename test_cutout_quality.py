import copy,unittest
import numpy as np
from cutout_quality import check_cutout,source_to_cutout


class CutoutCoverageTests(unittest.TestCase):
    def setUp(self):
        self.receipt={'source_size':[200,400],'source_bbox':[40,40,160,360],
                      'observed_subject_bbox':[40,40,160,360],'square_size':400}
        self.pose={'person_detected':True,'image_size':[200,400],
                   'person_box_normalized':[.2,.1,.8,.9],
                   'keypoints':{name:{'xy':xy,'heatmap_score':10.} for name,xy in
                     [('nose',[.5,.15]),('left_wrist',[.3,.55]),('left_ankle',[.4,.85]),('right_ankle',[.6,.85])]}}
        self.alpha=np.zeros((200,200),np.uint8);self.alpha[20:181,70:131]=255

    def test_padding_and_resize_map_source_coordinates(self):
        np.testing.assert_allclose(source_to_cutout([[.5,.5],[0,0]],self.receipt,(200,200)),[[100,100],[50,0]])
        receipt=dict(self.receipt,square_size=401)
        np.testing.assert_allclose(source_to_cutout([.5,.5],receipt,(201,201)),np.array([200,200])*201/401)

    def test_removed_lower_legs_flag_both_ankles(self):
        full=check_cutout(self.alpha,self.receipt,self.pose)
        self.assertEqual(full['coverage_fraction'],1.)
        damaged=self.alpha.copy();damaged[155:]=0
        result=check_cutout(damaged,self.receipt,self.pose)
        self.assertEqual(set(result['outside_landmarks']),{'left_ankle','right_ankle'})
        self.assertEqual(result['coverage_fraction'],.5)
        self.assertTrue(any('may exclude' in warning for warning in result['warnings']))

    def test_uncertain_pose_is_not_claimed_as_missing_or_verified(self):
        damaged=self.alpha.copy();damaged[155:]=0;pose=copy.deepcopy(self.pose)
        for name in ['left_ankle','right_ankle']:pose['keypoints'][name]['heatmap_score']=.1
        result=check_cutout(damaged,self.receipt,pose)
        self.assertEqual(result['status'],'partial');self.assertEqual(result['checked_landmarks'],2)
        self.assertFalse(result['outside_landmarks']);self.assertEqual(len(result['ignored_landmarks']),2)
        self.assertTrue(any('Uncertain' in warning for warning in result['warnings']))

    def test_wrong_subject_region_or_wrong_coordinates_are_not_success(self):
        other=dict(self.receipt,observed_subject_bbox=[0,0,20,30])
        result=check_cutout(self.alpha,other,self.pose)
        self.assertEqual(result['status'],'subject_region_mismatch');self.assertIsNone(result['coverage_fraction'])
        wrong=dict(self.pose,image_size=[400,200])
        self.assertEqual(check_cutout(self.alpha,self.receipt,wrong)['status'],'unavailable')


if __name__=='__main__':unittest.main()
