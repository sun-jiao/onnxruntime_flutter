"""Offline CMake target selection and bundled ELF/PE architecture checks."""
import pathlib
import struct
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


class NativeArchitectureTest(unittest.TestCase):
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
                + f'add_subdirectory("{ROOT.as_posix()}/{platform}" plugin)\n'
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
                                else 'libonnxruntime.so.1.15.1')
                    expected = ROOT / platform
                    if arch == 'arm64':
                        expected /= 'arm64'
                    self.assertEqual(pathlib.Path(selected), expected / filename)
                    self.assertTrue(pathlib.Path(selected).is_file())

    def test_legacy_target_processor_fallback(self):
        for platform in ['linux', 'windows']:
            for processor in ['aarch64', 'ARM64', 'AMD64', 'x86_64']:
                with self.subTest(platform=platform, processor=processor):
                    result, selected = self.configure(platform, processor=processor)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual('arm64' in pathlib.Path(selected).parts,
                                     processor.lower() in ['arm64', 'aarch64'])

    def test_windows_generator_platform_precedes_host_processor(self):
        for generator, processor in [('ARM64', 'AMD64'), ('x64', 'ARM64')]:
            result, selected = self.configure(
                'windows', processor=processor, generator=generator)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual('arm64' in pathlib.Path(selected).parts,
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

    def test_bundled_elf_and_pe_machine_types(self):
        for arch, elf_machine, pe_machine in [('', 62, 0x8664), ('arm64', 183, 0xAA64)]:
            with self.subTest(arch=arch or 'x64'):
                elf = (ROOT / 'linux' / arch / 'libonnxruntime.so.1.15.1').read_bytes()
                self.assertEqual(elf[:6], b'\x7fELF\x02\x01')
                self.assertEqual(struct.unpack_from('<H', elf, 18)[0], elf_machine)
                pe = (ROOT / 'windows' / arch / 'onnxruntime.dll').read_bytes()
                self.assertEqual(pe[:2], b'MZ')
                offset = struct.unpack_from('<I', pe, 0x3C)[0]
                self.assertEqual(pe[offset:offset + 4], b'PE\x00\x00')
                self.assertEqual(struct.unpack_from('<H', pe, offset + 4)[0], pe_machine)


if __name__ == '__main__':
    unittest.main()
