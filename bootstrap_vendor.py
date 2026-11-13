"""Restore pinned upstream code, never media or model weights.

Run --check for a read-only check. Existing checkouts at other revisions are
left untouched.
"""
import argparse
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent


def git(path, *args, check=True):
    return subprocess.run(['git', '-C', str(path), *args], check=check,
                          text=True, capture_output=True)


def restore(source, check_only):
    path = ROOT / source['path']
    if not path.exists():
        if check_only:
            raise RuntimeError(f"Missing {source['path']}; run without --check")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.mkdir()
        git(path, 'init')
        git(path, 'remote', 'add', 'origin', source['url'])
        if source.get('sparse_paths'):
            # Optional code-only sources can exclude upstream demo media before
            # blobs are retrieved. Existing checkouts are never reconfigured.
            git(path, 'config', 'remote.origin.promisor', 'true')
            git(path, 'config', 'remote.origin.partialclonefilter', 'blob:none')
            git(path, 'fetch', '--filter=blob:none', '--depth', '1', 'origin', source['commit'])
            git(path, 'sparse-checkout', 'init', '--no-cone')
            git(path, 'sparse-checkout', 'set', '--no-cone', *source['sparse_paths'])
        else:
            git(path, 'fetch', '--depth', '1', 'origin', source['commit'])
        git(path, 'checkout', '--detach', source['commit'])
    top = Path(git(path, 'rev-parse', '--show-toplevel').stdout.strip()).resolve()
    if top != path.resolve():
        raise RuntimeError(f"{source['path']} is not its own Git checkout")
    head = git(path, 'rev-parse', 'HEAD').stdout.strip()
    if head != source['commit']:
        raise RuntimeError(f"{source['path']} is at {head}; expected {source['commit']}. "
                           'Existing checkout left untouched.')
    for relative_patch in source.get('patches', []):
        patch = ROOT / relative_patch
        if git(path, 'apply', '--reverse', '--check', str(patch), check=False).returncode == 0:
            continue
        if check_only:
            raise RuntimeError(f"{relative_patch} is not applied")
        git(path, 'apply', '--check', str(patch))
        git(path, 'apply', str(patch))
    print(f"OK {source['path']} @ {head}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='No downloads or modifications')
    args = parser.parse_args()
    for source in json.loads((ROOT / 'vendor/sources.json').read_text(encoding='utf-8'))['repositories']:
        try:
            restore(source, args.check)
        except subprocess.CalledProcessError as error:
            parser.exit(1, f"Could not restore {source['path']}: {error.stderr or error}\n")
        except RuntimeError as error:
            parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
