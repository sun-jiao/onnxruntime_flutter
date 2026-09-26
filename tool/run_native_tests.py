"""Run the native Dart suite with the library for this host's architecture."""
import argparse
import os
from pathlib import Path
import platform
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def test_environment(system, machine, root, environment, runtime_dir=None,
                     runtime_version=None):
    if bool(runtime_dir) != bool(runtime_version):
        raise ValueError('An alternate runtime needs both its directory and version')
    system = system.lower()
    arch = machine.lower()
    if arch in ('amd64', 'x86_64', 'x64'):
        arch = 'x64'
    elif arch in ('aarch64', 'arm64'):
        arch = 'arm64'
    else:
        raise ValueError(f'Unsupported test architecture: {machine}')
    platforms = {
        'linux': ('linux', 'LD_LIBRARY_PATH', ':', 'libonnxruntime.so.1.15.1'),
        'windows': ('windows', 'PATH', ';', 'onnxruntime.dll'),
        'darwin': ('macos', 'DYLD_LIBRARY_PATH', ':', 'libonnxruntime.1.15.1.dylib'),
    }
    if system not in platforms:
        raise ValueError(f'Unsupported test platform: {system}')
    folder, key, separator, filename = platforms[system]
    directory = Path(root) / folder
    if system != 'darwin' and arch == 'arm64':
        directory /= 'arm64'
    if runtime_dir:
        directory = Path(runtime_dir).resolve()
    if not (directory / filename).is_file():
        raise ValueError(f'Native test library is missing: {directory / filename}')
    env = dict(environment)
    # Do not let a stale test expectation disguise the bundled runtime version.
    env.pop('ORT_TEST_VERSION', None)
    if runtime_version:
        env['ORT_TEST_VERSION'] = runtime_version
    env[key] = str(directory) + (separator + env[key] if env.get(key) else '')
    return env


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-dir')
    parser.add_argument('--runtime-version')
    args = parser.parse_args(argv)
    try:
        env = test_environment(platform.system(), platform.machine(), ROOT,
                               os.environ, args.runtime_dir, args.runtime_version)
    except ValueError as error:
        parser.error(str(error))
    # Resolve flutter.bat explicitly on Windows as subprocess does not apply
    # PATHEXT to an extensionless program consistently across Python versions.
    flutter = shutil.which('flutter', path=env.get('PATH'))
    if flutter is None:
        parser.error('Flutter is not on PATH')
    return subprocess.run([flutter, 'test', '--no-pub', '--reporter', 'expanded'],
                          cwd=ROOT, env=env).returncode


if __name__ == '__main__':
    raise SystemExit(main())
