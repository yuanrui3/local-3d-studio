from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import bootstrap_vendor as b


class SparseRestoration(unittest.TestCase):
    def source(self):
        return {'path': 'ignored/upstream', 'url': 'https://example.invalid/source.git',
                'commit': 'a'*40, 'sparse_paths': ['/hmr4d/', '/LICENSE']}

    def test_missing_check_only_creates_nothing_and_does_not_call_git(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with patch.object(b,'ROOT',root),patch.object(b,'git') as git:
                with self.assertRaisesRegex(RuntimeError,'Missing'):b.restore(self.source(),True)
                git.assert_not_called()
            self.assertEqual(list(root.iterdir()),[])

    def test_new_sparse_source_filters_blobs_before_checkout_and_never_executes_source(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source=self.source(); target=root/source['path']; calls=[]
            def git(path,*args,**kwargs):
                self.assertEqual(path,target);calls.append(args)
                value=str(target)if args==('rev-parse','--show-toplevel')else source['commit']if args==('rev-parse','HEAD')else''
                return SimpleNamespace(stdout=value,returncode=0)
            with patch.object(b,'ROOT',root),patch.object(b,'git',side_effect=git):b.restore(source,False)
            fetch=('fetch','--filter=blob:none','--depth','1','origin',source['commit'])
            sparse=('sparse-checkout','set','--no-cone','/hmr4d/','/LICENSE')
            checkout=('checkout','--detach',source['commit'])
            self.assertLess(calls.index(fetch),calls.index(sparse));self.assertLess(calls.index(sparse),calls.index(checkout))
            self.assertTrue(all(c[0]in {'init','remote','config','fetch','sparse-checkout','checkout','rev-parse'}for c in calls))

    def test_existing_foreign_revision_and_files_are_preserved_without_reconfiguration(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);source=self.source();target=root/source['path'];target.mkdir(parents=True)
            keep=target/'local-work.txt';keep.write_bytes(b'preserve');calls=[]
            def git(path,*args,**kwargs):
                calls.append(args);return SimpleNamespace(stdout=str(target)if args[-1]=='--show-toplevel'else'b'*40,returncode=0)
            with patch.object(b,'ROOT',root),patch.object(b,'git',side_effect=git):
                with self.assertRaisesRegex(RuntimeError,'left untouched'):b.restore(source,False)
            self.assertEqual(keep.read_bytes(),b'preserve');self.assertEqual([c[0]for c in calls],['rev-parse','rev-parse'])


if __name__=='__main__':unittest.main()
