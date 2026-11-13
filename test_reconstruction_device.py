"""Accelerator selection must fail clearly rather than create mislabeled CPU runs."""
import unittest
from unittest.mock import patch
import torch
from reconstruction_device import checked_device, synchronize, runtime_receipt


class DeviceTests(unittest.TestCase):
    def test_cpu_default_and_receipt(self):
        device=checked_device('cpu')
        self.assertEqual(str(device),'cpu')
        self.assertEqual(runtime_receipt(device)['device'],'cpu')

    def test_missing_cuda_fails_instead_of_falling_back(self):
        with patch('torch.cuda.is_available',return_value=False):
            with self.assertRaisesRegex(ValueError,'CUDA is unavailable'):
                checked_device('cuda')

    def test_explicit_cuda_index_is_validated(self):
        with patch('torch.cuda.is_available',return_value=True), \
             patch('torch.cuda.device_count',return_value=1), \
             patch('torch.cuda.current_device',return_value=0):
            self.assertEqual(str(checked_device('cuda')),'cuda:0')
            with self.assertRaisesRegex(ValueError,'does not exist'):
                checked_device('cuda:1')

    def test_unavailable_mps_and_unsupported_device_fail(self):
        with patch('torch.backends.mps.is_available',return_value=False):
            with self.assertRaisesRegex(ValueError,'MPS is unavailable'):
                checked_device('mps')
        with self.assertRaisesRegex(ValueError,'Choose cpu'):
            checked_device('meta')

    def test_cuda_timing_waits_on_the_selected_device(self):
        device=torch.device('cuda:1')
        with patch('torch.cuda.synchronize') as finish:
            synchronize(device)
            finish.assert_called_once_with(device)


if __name__=='__main__':
    unittest.main()
