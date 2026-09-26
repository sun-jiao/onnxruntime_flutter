"""Guard native version changes independently of the Dart API version."""
import hashlib
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
        self.assertTrue((ROOT / f"linux/libonnxruntime.so.{versions['linux']}").is_file())
        self.assertTrue((ROOT / f"macos/libonnxruntime.{versions['macos']}.dylib").is_file())
        for filename, digest in contract['bundled_libraries'].items():
            with self.subTest(filename=filename):
                self.assertEqual(hashlib.sha256((ROOT / filename).read_bytes()).hexdigest(),
                                 digest, 'Update and validate the compatibility matrix when replacing a library')


if __name__ == '__main__':
    unittest.main()
