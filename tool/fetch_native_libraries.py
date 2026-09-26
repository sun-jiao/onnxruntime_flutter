"""Fetch a pinned desktop runtime into a cache; never populate source folders."""
import argparse
import os
from pathlib import Path
import platform
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def host_target(system=None, machine=None):
    system = (system or platform.system()).lower()
    machine = (machine or platform.machine()).lower()
    if machine in ('arm64', 'aarch64'):
        arch = 'arm64'
    elif machine in ('x64', 'x86_64', 'amd64'):
        arch = 'x64'
    else:
        raise ValueError(f'Unsupported test architecture: {machine}')
    if system == 'darwin':
        return 'macos-universal2'
    if system not in ('linux', 'windows'):
        raise ValueError(f'Unsupported test platform: {system}')
    return f'{system}-{arch}'


def fetch_runtime(target=None, cache=None):
    target = target or host_target()
    cache = Path(cache or os.environ.get('ORT_CACHE_DIR', ROOT / 'build/native')).resolve()
    with tempfile.TemporaryDirectory(prefix='ort-fetch-') as temporary:
        output = Path(temporary) / 'library-path.txt'
        subprocess.run(['cmake', f'-DORT_TARGET={target}', f'-DORT_CACHE_DIR={cache}',
                        f'-DORT_OUTPUT={output}', '-P',
                        str(ROOT / 'tool/fetch_native_libraries.cmake')], check=True)
        return Path(output.read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', choices=['linux-x64', 'linux-arm64', 'windows-x64',
                                           'windows-arm64', 'macos-universal2'])
    parser.add_argument('--cache-dir')
    args = parser.parse_args()
    print(fetch_runtime(args.target, args.cache_dir))
