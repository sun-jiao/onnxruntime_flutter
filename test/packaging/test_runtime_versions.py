"""Guard native version changes independently of the Dart API version."""
import importlib.util
import json
import pathlib
import re
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


class RuntimeVersionTest(unittest.TestCase):
    def test_platform_dependencies_and_bundles_match_recorded_versions(self):
        contract = json.loads((ROOT / 'tool/native_runtime_versions.json').read_text())
        versions = contract['platforms']
        android = (ROOT / 'android/build.gradle').read_text()
        self.assertEqual(re.search(r'onnxruntime-android:([0-9.]+)', android)[1],
                         versions['android'])
        ios = (ROOT / 'ios/onnxruntime.podspec').read_text()
        self.assertEqual(re.search(r"'onnxruntime-objc', '([0-9.]+)'", ios)[1],
                         versions['ios'])
        self.assertFalse((ROOT / f"linux/libonnxruntime.so.{versions['linux']}").exists())
        self.assertFalse((ROOT / f"macos/libonnxruntime.{versions['macos']}.dylib").exists())
        spec = importlib.util.spec_from_file_location('generator', ROOT / 'tool/generate_native_downloads.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertEqual((ROOT / 'cmake/onnxruntime_downloads.cmake').read_text(), module.generate(contract))
        for filename in contract['libraries']:
            self.assertFalse((ROOT / filename).exists(), 'Native binaries must not be committed or published')

    def test_official_archive_inventory_covers_all_bundled_libraries(self):
        contract = json.loads((ROOT / 'tool/native_runtime_versions.json').read_text())
        restored = set()
        for archive in contract['archives']:
            self.assertTrue(archive['url'].startswith(
                'https://github.com/microsoft/onnxruntime/releases/download/'))
            self.assertRegex(archive['sha256'], r'^[0-9a-f]{64}$')
            for destination, member in archive['files'].items():
                self.assertNotIn(destination, restored)
                restored.add(destination)
                self.assertFalse((ROOT / destination).exists())
                self.assertNotIn('..', pathlib.PurePosixPath(member).parts)
        self.assertTrue(set(contract['libraries']).issubset(restored))
        for filename in contract['libraries']:
            folder = pathlib.PurePosixPath(filename).parent
            for notice in ['LICENSE', 'ThirdPartyNotices.txt']:
                self.assertIn(str(folder / notice), restored)


if __name__ == '__main__':
    unittest.main()
