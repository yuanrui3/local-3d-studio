import copy
import unittest

import numpy as np

from sam_recovery import recovery_prompt, accept_recovery


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.complete=np.zeros((200,100),bool);self.complete[20:181,30:71]=True
        self.partial=self.complete.copy();self.partial[90:]=False
        coords={'nose':[.5,.15],'left_hip':[.4,.5],'right_hip':[.6,.5],
                'left_knee':[.4,.7],'right_knee':[.6,.7],
                'left_ankle':[.4,.85],'right_ankle':[.6,.85]}
        self.pose={'person_detected':True,'substantial_people':1,'image_size':[100,200],
                   'person_box_normalized':[.3,.1,.71,.91],
                   'keypoints':{n:{'xy':xy,'heatmap_score':8.} for n,xy in coords.items()}}

    def test_partial_lower_body_can_be_recovered(self):
        self.assertIsNotNone(recovery_prompt(self.partial,self.pose))
        accepted,receipt=accept_recovery(self.partial,self.complete,self.pose)
        self.assertTrue(accepted)
        self.assertEqual(receipt['original_foreground_retained'],1.)

    def test_complete_or_low_confidence_masks_do_not_trigger(self):
        self.assertIsNone(recovery_prompt(self.complete,self.pose))
        weak=copy.deepcopy(self.pose)
        for p in weak['keypoints'].values():p['heatmap_score']=2.
        self.assertIsNone(recovery_prompt(self.partial,weak))

    def test_detached_fragments_do_not_hide_a_truncated_main_subject(self):
        fragments=self.partial.copy();fragments[165:180,40:45]=True
        self.assertIsNotNone(recovery_prompt(fragments,self.pose))

    def test_ambiguous_or_mismatched_person_does_not_trigger(self):
        multiple=dict(self.pose,substantial_people=2)
        self.assertIsNone(recovery_prompt(self.partial,multiple))
        other=dict(self.pose,person_box_normalized=[0,.1,.2,.91])
        self.assertIsNone(recovery_prompt(self.partial,other))
        self.assertIsNone(recovery_prompt(self.partial,dict(self.pose,person_box_normalized=[])))

    def test_background_spill_and_loss_of_original_region_are_rejected(self):
        self.assertFalse(accept_recovery(self.partial,np.ones_like(self.complete),self.pose)[0])
        missing=self.complete.copy();missing[20:60]=False
        self.assertFalse(accept_recovery(self.partial,missing,self.pose)[0])


if __name__=='__main__':unittest.main()
