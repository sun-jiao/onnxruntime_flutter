"""Run real browser inference against a checksum-pinned ONNX Runtime Web.

Requires Flutter, Python 3 and Chrome (or CHROME_EXECUTABLE). The first run
needs network access; --runtime-archive accepts an offline copy of the npm tarball.
"""
import argparse
import functools
import hashlib
import http.server
from pathlib import Path
import shutil
import subprocess
import tarfile
import threading
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
VERSION = '1.23.2'
SHA256 = '0b8707d7efab9a2bea63564d66cad79e897cfd0bd627e5ae000488ffc1c45c7d'
URL = f'https://registry.npmjs.org/onnxruntime-web/-/onnxruntime-web-{VERSION}.tgz'
FILES = ('ort.min.js', 'ort-wasm-simd-threaded.jsep.mjs', 'ort-wasm-simd-threaded.jsep.wasm')


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      '.mjs': 'application/javascript', '.wasm': 'application/wasm'}

    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        super().end_headers()

    def log_message(self, *_args):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-archive', type=Path)
    args = parser.parse_args()
    cache = ROOT / '.dart_tool' / 'ort_web' / VERSION
    cache.mkdir(parents=True, exist_ok=True)
    archive = args.runtime_archive or cache / 'runtime.tgz'
    if not archive.exists():
        print(f'Downloading ONNX Runtime Web {VERSION}', flush=True)
        with urllib.request.urlopen(URL, timeout=120) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != SHA256:
            raise ValueError('Downloaded ONNX Runtime Web checksum mismatch')
        archive.write_bytes(data)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise ValueError(f'ONNX Runtime Web checksum mismatch: {archive}')
    # Extract only the three named regular files; never trust archive paths.
    with tarfile.open(archive, 'r:gz') as package:
        for name in FILES:
            member = package.getmember(f'package/dist/{name}')
            if not member.isfile():
                raise ValueError(f'Unexpected archive entry: {member.name}')
            with package.extractfile(member) as source, (cache / name).open('wb') as target:
                shutil.copyfileobj(source, target)
    for fixture in (ROOT / 'test' / 'fixtures').glob('*.onnx'):
        shutil.copyfile(fixture, cache / fixture.name)
    for fixture in (ROOT / 'example' / 'assets' / 'models').glob('test_types_*.pb'):
        shutil.copyfile(fixture, cache / fixture.name)
    server = http.server.ThreadingHTTPServer(
        ('127.0.0.1', 0), functools.partial(Handler, directory=str(cache)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    flutter = shutil.which('flutter')
    if not flutter:
        raise RuntimeError('Flutter is not on PATH')
    try:
        return subprocess.run([
            flutter, 'test', '--no-pub', '--platform', 'chrome',
            f'--dart-define=ORT_WEB_URL=http://127.0.0.1:{server.server_port}',
            '--reporter', 'expanded', 'test/web',
        ], cwd=ROOT).returncode
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    raise SystemExit(main())
