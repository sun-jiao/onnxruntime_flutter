"""Exercise local-pod preparation without network or real native binaries."""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(shutil.which('ruby'), 'Ruby is needed for the CocoaPods downloader test')
class MacosDownloadTest(unittest.TestCase):
    def test_local_pod_preparation_verifies_repairs_and_rejects_corruption(self):
        with tempfile.TemporaryDirectory(prefix='ort-ruby-') as temporary:
            root = Path(temporary)
            plugin = root / 'plugin'
            plugin.mkdir()
            files = {'macos/libonnxruntime.dylib': 'upstream/lib/libonnxruntime.dylib',
                     'macos/LICENSE': 'upstream/LICENSE',
                     'macos/ThirdPartyNotices.txt': 'upstream/ThirdPartyNotices.txt'}
            payload = io.BytesIO()
            with tarfile.open(fileobj=payload, mode='w:gz') as tar:
                for member in files.values():
                    data = member.encode()
                    info = tarfile.TarInfo(member)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
            digest = hashlib.sha256(payload.getvalue()).hexdigest()
            entry = root / 'cache' / digest
            entry.mkdir(parents=True)
            (entry / 'archive').write_bytes(payload.getvalue())
            manifest = {'libraries': {'macos/libonnxruntime.dylib': hashlib.sha256(
                b'upstream/lib/libonnxruntime.dylib').hexdigest()}, 'archives': [{
                'target': 'macos-universal2', 'sha256': digest,
                'url': 'https://invalid.invalid/must-not-download', 'files': files}]}
            (root / 'manifest.json').write_text(json.dumps(manifest))
            command = ['ruby', '-r', str(ROOT / 'macos/download_runtime.rb'), '-e',
                       'OnnxruntimeDownload.prepare(ARGV[0], ARGV[1])',
                       str(plugin), str(root / 'manifest.json')]
            env = dict(os.environ, ORT_CACHE_DIR=str(root / 'cache'))
            def run():
                return subprocess.run(command, env=env, capture_output=True, text=True)
            result = run()
            self.assertEqual(result.returncode, 0, result.stderr)
            library = plugin / '.onnxruntime/libonnxruntime.dylib'
            self.assertEqual(library.read_bytes(), b'upstream/lib/libonnxruntime.dylib')
            library.write_bytes(b'corrupt stage')
            result = run()
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(library.read_bytes(), b'upstream/lib/libonnxruntime.dylib')
            (entry / 'macos-extracted/upstream/lib/libonnxruntime.dylib').write_bytes(b'bad cache')
            result = run()
            self.assertEqual(result.returncode, 0, result.stderr)
            shutil.rmtree(entry / 'macos-extracted')
            (entry / 'archive').write_bytes(b'corrupt archive')
            result = run()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('checksum mismatch', result.stderr)
