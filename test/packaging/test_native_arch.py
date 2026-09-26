"""Offline CMake target selection and bundled ELF/PE architecture checks."""
import pathlib
import json
import hashlib
import io
import tarfile
import shutil
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tool"))
from generate_native_downloads import generate
import struct
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


class NativeArchitectureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='ort-offline-')
        cls.root = pathlib.Path(cls.temporary.name)
        cls.cache = cls.root / 'cache'
        cls.cache.mkdir()
        for folder in ['linux', 'windows', 'cmake']:
            (cls.root / folder).mkdir()
        for name in ['linux/CMakeLists.txt', 'windows/CMakeLists.txt',
                     'cmake/onnxruntime_arch.cmake', 'cmake/onnxruntime_download.cmake']:
            shutil.copyfile(ROOT / name, cls.root / name)
        manifest = json.loads((ROOT / 'tool/native_runtime_versions.json').read_text())
        cls.expected = {}
        for archive in manifest['archives']:
            payload = io.BytesIO()
            with tarfile.open(fileobj=payload, mode='w:gz') as tar:
                for destination, member in archive['files'].items():
                    content = (archive['target'] + member).encode()
                    info = tarfile.TarInfo(member)
                    info.size = len(content)
                    tar.addfile(info, io.BytesIO(content))
                    if destination in manifest['libraries']:
                        manifest['libraries'][destination] = hashlib.sha256(content).hexdigest()
            archive['sha256'] = hashlib.sha256(payload.getvalue()).hexdigest()
            archive['url'] = 'https://invalid.invalid/must-not-download'
            entry = cls.cache / archive['sha256']
            entry.mkdir()
            (entry / 'archive').write_bytes(payload.getvalue())
            root = next(iter(archive['files'].values())).split('/')[0]
            cls.expected[archive['target']] = entry / root / 'lib'
        (cls.root / 'cmake/onnxruntime_downloads.cmake').write_text(generate(manifest))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def configure(self, platform, *, target=None, processor='x86_64', generator=''):
        with tempfile.TemporaryDirectory(prefix='ort-cmake-') as temporary:
            directory = pathlib.Path(temporary)
            # Use a host compiler even when checking Windows packaging: these
            # plugins only select prebuilt files and do not compile native code.
            settings = '\n'.join([
                f'set(CMAKE_SYSTEM_PROCESSOR "{processor}")',
                f'set(CMAKE_GENERATOR_PLATFORM "{generator}")',
                *([f'set(FLUTTER_TARGET_PLATFORM "{target}")'] if target else []),
            ])
            (directory / 'CMakeLists.txt').write_text(
                'cmake_minimum_required(VERSION 3.14)\n'
                'project(ort_arch_test LANGUAGES CXX)\n'
                + settings + '\n'
                + f'set(ORT_CACHE_DIR "{self.cache.as_posix()}")\n'
                + f'add_subdirectory("{self.root.as_posix()}/{platform}" plugin)\n'
                + 'file(WRITE "${CMAKE_BINARY_DIR}/selected.txt" '
                '"${onnxruntime_bundled_libraries}")\n')
            result = subprocess.run(
                ['cmake', '-S', str(directory), '-B', str(directory / 'build')],
                capture_output=True, text=True)
            selected = directory / 'build/selected.txt'
            return result, selected.read_text() if selected.exists() else None

    def test_explicit_targets_override_host_architecture(self):
        for platform in ['linux', 'windows']:
            for arch, host in [('arm64', 'x86_64'), ('x64', 'aarch64')]:
                with self.subTest(platform=platform, arch=arch):
                    result, selected = self.configure(
                        platform, target=f'{platform}-{arch}', processor=host)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    filename = ('onnxruntime.dll' if platform == 'windows'
                                else 'libonnxruntime.so.1.30.0')
                    expected = self.expected[f'{platform}-{arch}']
                    libraries = [pathlib.Path(p) for p in selected.split(';')]
                    shared = ('onnxruntime_providers_shared.dll' if platform == 'windows'
                              else 'libonnxruntime_providers_shared.so')
                    self.assertEqual(libraries, [expected / filename, expected / shared])
                    self.assertTrue(all(p.is_file() for p in libraries))

    def test_legacy_target_processor_fallback(self):
        for platform in ['linux', 'windows']:
            for processor in ['aarch64', 'ARM64', 'AMD64', 'x86_64']:
                with self.subTest(platform=platform, processor=processor):
                    result, selected = self.configure(platform, processor=processor)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(any('arm64' in part or 'aarch64' in part for part in pathlib.Path(selected.split(';')[0]).parts),
                                     processor.lower() in ['arm64', 'aarch64'])

    def test_windows_generator_platform_precedes_host_processor(self):
        for generator, processor in [('ARM64', 'AMD64'), ('x64', 'ARM64')]:
            result, selected = self.configure(
                'windows', processor=processor, generator=generator)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(any('arm64' in part or 'aarch64' in part for part in pathlib.Path(selected.split(';')[0]).parts),
                             generator == 'ARM64')

    def test_unsupported_targets_fail_instead_of_bundling_x64(self):
        for platform, kwargs in [
            ('linux', {'target': 'linux-riscv64'}),
            ('windows', {'target': 'windows-x86'}),
            ('linux', {'processor': 'armv7l'}),
            ('windows', {'generator': 'Win32'}),
        ]:
            with self.subTest(platform=platform, kwargs=kwargs):
                result, _ = self.configure(platform, **kwargs)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Unsupported ONNX Runtime', result.stderr)

    def test_corrupt_archive_is_rejected_without_network(self):
        entry = self.expected['linux-x64'].parent.parent
        shutil.rmtree(self.expected['linux-x64'].parent, ignore_errors=True)
        archive = entry / 'archive'
        original = archive.read_bytes()
        try:
            archive.write_bytes(b'corrupt')
            result, _ = self.configure('linux', target='linux-x64')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('checksum mismatch', result.stderr)
        finally:
            archive.write_bytes(original)

    def test_bad_download_is_not_promoted_to_cached_archive(self):
        entry = self.expected['linux-x64'].parent.parent
        shutil.rmtree(self.expected['linux-x64'].parent, ignore_errors=True)
        archive = entry / 'archive'
        original = archive.read_bytes()
        archive.unlink()
        invalid = self.root / 'invalid-download'
        invalid.write_bytes(b'not the pinned archive')
        metadata = self.root / 'cmake/onnxruntime_downloads.cmake'
        original_metadata = metadata.read_text()
        try:
            metadata.write_text(original_metadata.replace(
                'https://invalid.invalid/must-not-download', invalid.as_uri()))
            result, _ = self.configure('linux', target='linux-x64')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('checksum mismatch', result.stderr)
            self.assertFalse(archive.exists())
            self.assertFalse((entry / 'archive.part').exists())
        finally:
            archive.write_bytes(original)
            metadata.write_text(original_metadata)

    def test_modified_extracted_library_is_repaired_from_verified_archive(self):
        result, selected = self.configure('linux', target='linux-x64')
        self.assertEqual(result.returncode, 0, result.stderr)
        library = pathlib.Path(selected.split(';')[0])
        original = library.read_bytes()
        library.write_bytes(b'corrupt extracted library')
        result, _ = self.configure('linux', target='linux-x64')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(library.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
