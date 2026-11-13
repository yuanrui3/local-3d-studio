"""Downloader regressions: corrupt/interrupted weights must never become usable."""
import hashlib
import http.client
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import fetch_models


class FetchModelsTests(unittest.TestCase):
    def setUp(self):
        temp_root = fetch_models.ROOT / '.runtime' / 'test-temp'
        temp_root.mkdir(parents=True, exist_ok=True)
        self.folder = tempfile.TemporaryDirectory(dir=temp_root)
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.payload = b'official model bytes\x00' * 12
        self.asset = {
            'id': 'test-model', 'repo': 'official/model', 'revision': 'a' * 40,
            'file': 'model.ckpt', 'destination': 'models/test/model.ckpt',
            'url': 'https://official.invalid/model.ckpt',
            'sha256': hashlib.sha256(self.payload).hexdigest(),
            'md5': hashlib.md5(self.payload).hexdigest(),
            'size_bytes': len(self.payload), 'license': {'model_license': 'test'},
            'checksum_source': 'test fixture',
        }
        self.destination = self.root / self.asset['destination']

    def fetch(self, **kwargs):
        return fetch_models.fetch_asset(self.asset, self.root, **kwargs)

    def parts(self):
        return list(self.root.rglob('*.part'))

    def test_streamed_download_and_offline_reuse_preserve_hf_receipt_contract(self):
        sizes = []

        class Stream(io.BytesIO):
            def read(inner, size=-1):
                sizes.append(size)
                self.assertFalse(self.destination.exists())
                return super().read(size)

        with patch.object(fetch_models, 'BLOCK_SIZE', 17):
            receipt = self.fetch(opener=lambda *a, **k: Stream(self.payload))
        self.assertEqual(self.destination.read_bytes(), self.payload)
        self.assertEqual(set(sizes), {17})
        self.assertEqual(receipt['repo'], self.asset['repo'])
        self.assertEqual(receipt['revision'], self.asset['revision'])
        self.assertEqual(receipt['file'], 'model.ckpt')
        self.assertEqual(receipt['local_path'], str(self.destination))
        self.assertEqual(receipt['sha256'], self.asset['sha256'])
        self.assertEqual(receipt['md5'], self.asset['md5'])
        self.assertEqual(self.parts(), [])
        self.fetch(verify_only=True, opener=lambda *a, **k: self.fail('Unexpected network'))

    def test_existing_corrupt_file_is_rejected_without_network(self):
        self.destination.parent.mkdir(parents=True)
        self.destination.write_bytes(b'corrupt')
        with self.assertRaisesRegex(RuntimeError, 'Existing .* failed verification'):
            self.fetch(opener=lambda *a, **k: self.fail('Unexpected network'))
        self.assertEqual(self.destination.read_bytes(), b'corrupt')

    def test_interrupted_repair_preserves_existing_bytes_and_cleans_partial(self):
        self.destination.parent.mkdir(parents=True)
        self.destination.write_bytes(b'corrupt but preserved')

        class Interrupted(io.BytesIO):
            def read(inner, size=-1):
                self.assertEqual(self.destination.read_bytes(), b'corrupt but preserved')
                if inner.tell():
                    raise ConnectionError('interrupted transfer')
                return super().read(5)

        with patch.object(fetch_models.time, 'sleep'):
            with self.assertRaises(ConnectionError):
                self.fetch(repair=True, opener=lambda *a, **k: Interrupted(self.payload))
        self.assertEqual(self.destination.read_bytes(), b'corrupt but preserved')
        self.assertEqual(self.parts(), [])
        self.fetch(repair=True, opener=lambda *a, **k: io.BytesIO(self.payload))
        self.assertEqual(self.destination.read_bytes(), self.payload)

    def test_bad_digest_truncation_and_oversize_never_publish_final_path(self):
        for payload, pattern in [
            (b'x' * len(self.payload), 'sha256'),
            (self.payload[:-1], 'sha256'),
            (self.payload + b'excess', 'exceeded'),
        ]:
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(RuntimeError, pattern):
                    self.fetch(opener=lambda *a, **k: io.BytesIO(payload))
                self.assertFalse(self.destination.exists())
                self.assertEqual(self.parts(), [])

    def test_rembg_md5_is_checked_independently_of_sha256(self):
        self.asset['md5'] = '0' * 32
        with self.assertRaisesRegex(RuntimeError, 'md5'):
            self.fetch(opener=lambda *a, **k: io.BytesIO(self.payload))
        self.assertFalse(self.destination.exists())
        self.assertEqual(self.parts(), [])

    def test_remote_disconnect_retries_from_zero_and_digest_error_does_not_retry(self):
        calls = []

        class Disconnected(io.BytesIO):
            def read(inner, size=-1):
                if inner.tell():
                    raise http.client.RemoteDisconnected('remote closed')
                return super().read(7)

        def opener(*args, **kwargs):
            calls.append(1)
            # The failed attempt's partial file must be removed before retrying.
            self.assertEqual(len(self.parts()), 1)
            return Disconnected(self.payload) if len(calls) == 1 else io.BytesIO(self.payload)

        with patch.object(fetch_models.time, 'sleep'):
            self.fetch(opener=opener)
        self.assertEqual(len(calls), 2)
        self.assertEqual(self.destination.read_bytes(), self.payload)
        self.assertEqual(self.parts(), [])
        self.destination.unlink()
        calls.clear()

        def bad_digest(*args, **kwargs):
            calls.append(1)
            return io.BytesIO(b'x' * len(self.payload))

        with self.assertRaisesRegex(RuntimeError, 'sha256'):
            self.fetch(opener=bad_digest)
        self.assertEqual(len(calls), 1)
        self.assertFalse(self.destination.exists())
        self.assertEqual(self.parts(), [])

    def test_missing_verify_only_and_escaping_path_fail_without_network(self):
        with self.assertRaises(FileNotFoundError):
            self.fetch(verify_only=True, opener=lambda *a, **k: self.fail('Unexpected network'))
        self.asset['destination'] = '../outside.ckpt'
        with self.assertRaisesRegex(ValueError, 'escapes project'):
            self.fetch(opener=lambda *a, **k: self.fail('Unexpected network'))

    def test_failed_receipt_replace_retains_previous_receipt(self):
        target = self.root / 'sources.json'
        target.write_text('[{"id":"previous"}]', encoding='utf-8')
        with patch.object(fetch_models.os, 'replace', side_effect=PermissionError('busy')):
            with self.assertRaises(PermissionError):
                fetch_models.write_receipts([{'id': 'new'}], target)
        self.assertEqual(json.loads(target.read_text()), [{'id': 'previous'}])
        self.assertEqual(self.parts(), [])

    def test_official_manifest_has_all_pins_and_keeps_expected_runtime_paths(self):
        assets = {a['id']: a for a in fetch_models.load_manifest()['assets']}
        self.assertEqual(len(assets), 8)
        self.assertEqual(assets['u2net-human-seg']['destination'], '.cache/rembg/u2net_human_seg.onnx')
        for asset in assets.values():
            self.assertEqual(len(asset['sha256']), 64)
            self.assertTrue(asset['url'].startswith('https://'))
            self.assertTrue(asset['license'])
        self.assertEqual(assets['triposr-model']['revision'], '5b521936b01fbe1890f6f9baed0254ab6351c04a')


if __name__ == '__main__':
    unittest.main()
