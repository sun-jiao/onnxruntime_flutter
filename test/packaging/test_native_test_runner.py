import importlib.util
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tool"))
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = pathlib.Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('native_runner', ROOT / 'tool/run_native_tests.py')
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class NativeTestRunnerTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temporary.name)
        for folder, filename in [('linux', 'libonnxruntime.so.1.30.0'),
                                 ('linux/arm64', 'libonnxruntime.so.1.30.0'),
                                 ('windows', 'onnxruntime.dll'),
                                 ('windows/arm64', 'onnxruntime.dll'),
                                 ('macos', 'libonnxruntime.1.23.2.dylib')]:
            (self.root / folder).mkdir(parents=True, exist_ok=True)
            (self.root / folder / filename).touch()
        self.addCleanup(self.temporary.cleanup)

    def test_each_desktop_architecture_uses_matching_library_and_keeps_search_path(self):
        for system, machine, folder, variable, separator in [
            ('Linux', 'x86_64', 'linux', 'LD_LIBRARY_PATH', ':'),
            ('Linux', 'aarch64', 'linux/arm64', 'LD_LIBRARY_PATH', ':'),
            ('Windows', 'AMD64', 'windows', 'PATH', ';'),
            ('Windows', 'ARM64', 'windows/arm64', 'PATH', ';'),
            ('Darwin', 'arm64', 'macos', 'DYLD_LIBRARY_PATH', ':'),
            ('Darwin', 'x86_64', 'macos', 'DYLD_LIBRARY_PATH', ':'),
        ]:
            with self.subTest(system=system, machine=machine):
                original = {variable: 'existing', 'UNRELATED': 'kept',
                            'ORT_TEST_VERSION': 'stale',
                            'ORT_TEST_LIBRARY_PATH': 'stale-library'}
                env = RUNNER.test_environment(system, machine, self.root, original)
                self.assertEqual(env[variable], str(self.root / folder) + separator + 'existing')
                self.assertEqual(env['UNRELATED'], 'kept')
                self.assertNotIn('ORT_TEST_VERSION', env)
                self.assertEqual(original[variable], 'existing')
                library = pathlib.Path(env['ORT_TEST_LIBRARY_PATH'])
                self.assertTrue(library.is_absolute())
                self.assertTrue(library.is_file())
                self.assertEqual(library.parent, (self.root / folder).resolve())
                self.assertEqual(original['ORT_TEST_LIBRARY_PATH'], 'stale-library')

    def test_alternate_runtime_is_explicit_and_missing_library_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                RUNNER.test_environment('Linux', 'x64', ROOT, {}, temporary)
            with self.assertRaises(ValueError):
                RUNNER.test_environment('Linux', 'x64', ROOT, {}, temporary, '1.23.2')
            (pathlib.Path(temporary) / 'libonnxruntime.so.1.30.0').touch()
            env = RUNNER.test_environment('Linux', 'x64', ROOT, {}, temporary, '1.23.2')
            self.assertEqual(env['ORT_TEST_VERSION'], '1.23.2')
            self.assertEqual(env['LD_LIBRARY_PATH'], str(pathlib.Path(temporary).resolve()))
            self.assertEqual(env['ORT_TEST_LIBRARY_PATH'],
                             str((pathlib.Path(temporary) / 'libonnxruntime.so.1.30.0').resolve()))

    def test_windows_download_path_with_spaces_overrides_system_dll_search(self):
        directory = self.root / 'download cache' / 'windows-x64'
        directory.mkdir(parents=True)
        library = directory / 'onnxruntime.dll'
        library.touch()
        env = RUNNER.test_environment('Windows', 'AMD64', self.root,
                                      {'PATH': r'C:\Windows\System32'},
                                      downloaded_dir=directory)
        self.assertEqual(env['ORT_TEST_LIBRARY_PATH'], str(library.resolve()))


    def test_unknown_architecture_never_silently_uses_x64(self):
        with self.assertRaises(ValueError):
            RUNNER.test_environment('Linux', 'riscv64', ROOT, {})

    def test_flutter_failure_is_the_runner_exit_status(self):
        with patch.object(RUNNER, 'fetch_runtime', return_value=self.root / 'linux'), \
             patch.object(RUNNER.platform, 'system', return_value='Linux'), \
             patch.object(RUNNER.platform, 'machine', return_value='x86_64'), \
             patch.object(RUNNER.shutil, 'which', return_value='/sdk/flutter'), \
             patch.object(RUNNER.subprocess, 'run',
                          return_value=subprocess.CompletedProcess([], 7)) as run:
            self.assertEqual(RUNNER.main([]), 7)
            self.assertEqual(run.call_args.args[0][1:],
                             ['test', '--no-pub', '--reporter', 'expanded'])
            self.assertEqual(run.call_args.kwargs['cwd'], ROOT)
            self.assertTrue(run.call_args.kwargs['env']['LD_LIBRARY_PATH'].startswith(str(self.root / 'linux')))


if __name__ == '__main__':
    unittest.main()
