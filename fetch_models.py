"""Fetch pinned official model assets locally, with atomic SHA256 verification.

This command only downloads public weights/configuration; it never reads or
uploads reference media. Inference continues to use models/sources.json offline.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from local_runtime import ROOT

MANIFEST = ROOT / 'weights-manifest.json'
BLOCK_SIZE = 1024 * 1024
MAX_DOWNLOAD_BYTES = 5 * 1024**3
DOWNLOAD_ATTEMPTS = 3


def file_digests(path):
    """Hash bounded chunks, including rembg's independently published MD5."""
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    size = 0
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(BLOCK_SIZE), b''):
            sha256.update(block)
            md5.update(block)
            size += len(block)
    return {'sha256': sha256.hexdigest(), 'md5': md5.hexdigest(), 'size_bytes': size}


def check_digests(asset, actual):
    for key in ('sha256', 'md5', 'size_bytes'):
        expected = asset.get(key)
        if expected is not None and actual[key] != expected:
            raise RuntimeError(f"{asset['id']} failed pinned {key}: "
                               f"expected {expected}, got {actual[key]}")


def asset_destination(asset, root):
    root = Path(root).resolve()
    destination = (root / asset['destination']).resolve()
    if destination == root or not destination.is_relative_to(root):
        raise ValueError(f"Asset destination escapes project: {asset['destination']}")
    return destination


def fetch_asset(asset, root=ROOT, *, repair=False, verify_only=False, opener=urlopen):
    """Return a receipt only after the complete local file passes its pins.

    Existing corrupt files fail closed unless --repair is requested. Replacement
    is atomic, so a failed transfer cannot replace an existing file or leave a
    partial model under its final name.
    """
    destination = asset_destination(asset, root)
    if destination.is_file():
        actual = file_digests(destination)
        try:
            check_digests(asset, actual)
        except RuntimeError:
            if verify_only or not repair:
                raise RuntimeError(f"Existing {destination} failed verification; "
                                   "use --repair to replace it from the pinned official URL")
        else:
            return make_receipt(asset, destination, actual)
    elif destination.exists():
        raise RuntimeError(f'Model destination is not a regular file: {destination}')
    if verify_only:
        raise FileNotFoundError(f'Missing model asset: {destination}')
    if not asset['url'].startswith('https://'):
        raise ValueError('Model downloads require HTTPS')
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            actual = download_once(asset, destination, opener)
            return make_receipt(asset, destination, actual)
        except (HTTPError, URLError, ConnectionError, TimeoutError, http.client.HTTPException) as error:
            if not retryable_network_error(error) or attempt == DOWNLOAD_ATTEMPTS:
                raise
            print(f"{asset['id']}: transient {type(error).__name__}; "
                  f"retrying ({attempt + 1}/{DOWNLOAD_ATTEMPTS})", flush=True)
            time.sleep(2 ** (attempt - 1))


def retryable_network_error(error):
    if isinstance(error, HTTPError):
        return error.code in (408, 425, 429, 500, 502, 503, 504)
    return isinstance(error, (URLError, ConnectionError, TimeoutError, http.client.HTTPException))


def download_once(asset, destination, opener):
    """Start at byte zero each time; digest/size errors are never retried."""
    part = None
    try:
        # Unique names allow unrelated downloads without sharing partial state.
        with tempfile.NamedTemporaryFile(mode='wb', prefix=destination.name + '.',
                                         suffix='.part', dir=destination.parent,
                                         delete=False) as output:
            part = Path(output.name)
            sha256, md5, size = hashlib.sha256(), hashlib.md5(), 0
            request = Request(asset['url'], headers={'User-Agent': 'local-clothed-3d-studio/1'})
            with opener(request, timeout=60) as response:
                for block in iter(lambda: response.read(BLOCK_SIZE), b''):
                    size += len(block)
                    if size > asset.get('size_bytes', MAX_DOWNLOAD_BYTES):
                        raise RuntimeError(f"{asset['id']} exceeded its pinned download size")
                    output.write(block)
                    sha256.update(block)
                    md5.update(block)
            output.flush()
            os.fsync(output.fileno())
        actual = {'sha256': sha256.hexdigest(), 'md5': md5.hexdigest(), 'size_bytes': size}
        check_digests(asset, actual)
        os.replace(part, destination)
        part = None
        return actual
    finally:
        if part is not None:
            part.unlink(missing_ok=True)


def make_receipt(asset, destination, actual):
    # Preserve the original receipt keys consumed by generate_local and tooling.
    receipt = {
        'id': asset['id'], 'url': asset['url'], 'sha256': actual['sha256'],
        'size_bytes': actual['size_bytes'], 'local_path': str(destination),
        'verified_at_utc': datetime.now(timezone.utc).isoformat(),
        'verification': 'pinned-sha256', 'license': asset['license'],
        'checksum_source': asset['checksum_source'],
    }
    if 'repo' in asset:
        receipt.update(repo=asset['repo'], revision=asset['revision'], file=asset['file'])
    else:
        receipt.update(model=asset['file'], file=str(destination))
    if asset.get('md5'):
        receipt['md5'] = actual['md5']
    return receipt


def load_manifest(path=MANIFEST):
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    if manifest.get('schema_version') != 1:
        raise ValueError('Unsupported weights manifest version')
    assets = manifest['assets']
    if len({a['id'] for a in assets}) != len(assets):
        raise ValueError('Duplicate weight asset id')
    for asset in assets:
        if len(asset['sha256']) != 64 or any(c not in '0123456789abcdef' for c in asset['sha256']):
            raise ValueError(f"Invalid SHA256 for {asset['id']}")
        asset_destination(asset, ROOT)
    return manifest


def write_receipts(receipts, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    part = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='\n',
                                         prefix=target.name + '.', suffix='.part',
                                         dir=target.parent, delete=False) as output:
            part = Path(output.name)
            output.write(json.dumps(receipts, indent=2) + '\n')
            output.flush()
            os.fsync(output.fileno())
        os.replace(part, target)
        part = None
    finally:
        if part is not None:
            part.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', action='append', metavar='ASSET_ID',
                        help='Download/verify only these assets (repeatable); default: all')
    parser.add_argument('--repair', action='store_true', help='Atomically replace corrupt existing files')
    parser.add_argument('--verify-only', action='store_true', help='Verify locally without network access')
    args = parser.parse_args()
    manifest = load_manifest()
    selected = set(args.only or [a['id'] for a in manifest['assets']])
    unknown = selected - {a['id'] for a in manifest['assets']}
    if unknown:
        parser.error('Unknown asset id(s): ' + ', '.join(sorted(unknown)))
    target = ROOT / 'models' / 'sources.json'
    # Revalidate preserved partial-run receipts before they can be reused offline.
    receipts = []
    if target.exists() and args.only:
        previous = json.loads(target.read_text(encoding='utf-8'))
        previous_ids = {r.get('id') for r in previous}
        for asset in manifest['assets']:
            if asset['id'] not in selected and asset['id'] in previous_ids:
                receipts.append(fetch_asset(asset, verify_only=True))
    for asset in manifest['assets']:
        if asset['id'] in selected:
            print(f"Verifying {asset['id']}" if args.verify_only else f"Fetching {asset['id']}", flush=True)
            receipts.append(fetch_asset(asset, repair=args.repair, verify_only=args.verify_only))
    manifest_digest = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    for receipt in receipts:
        receipt['weights_manifest_sha256'] = manifest_digest
        receipt['license_limitations'] = manifest['license_limitations']
    write_receipts(receipts, target)
    print(f'Verified {len(receipts)} local assets; receipts: {target}', flush=True)


if __name__ == '__main__':
    main()
