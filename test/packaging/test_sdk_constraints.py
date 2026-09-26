"""Keep advertised SDK support consistent with runtime dependency requirements."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[2]


def environment(path):
    text = path.read_text()
    section = re.search(r'^environment:\n((?:[ \t].*\n|\n)+)', text, re.M)[1]
    return dict(re.findall(r"^  (sdk|flutter):\s*['\"]([^'\"]+)['\"]", section, re.M))


def version(text):
    return tuple(map(int, text.split('.')))


def bounds(constraint):
    match = re.fullmatch(r'>=(\d+\.\d+\.\d+)(?: <(\d+\.\d+\.\d+))?', constraint)
    if not match:
        raise AssertionError(f'Unrecognized SDK range: {constraint}')
    return version(match[1]), version(match[2]) if match[2] else None


def allows(constraint, candidate):
    lower, upper = bounds(constraint)
    return lower <= candidate and (upper is None or candidate < upper)


class SdkConstraintsTest(unittest.TestCase):
    def test_runtime_sdk_matches_ffi_requirement_without_dev_tool_restrictions(self):
        # ffi 2.2.0 requires Dart >=3.7.0 <4.0.0. Development-only ffigen
        # constraints must not unnecessarily raise the published runtime floor.
        manifest = (ROOT / 'pubspec.yaml').read_text()
        self.assertRegex(manifest, r'(?m)^  ffi: \^2\.2\.0$')
        sdk = environment(ROOT / 'pubspec.yaml')['sdk']
        for supported in [(3, 7, 0), (3, 10, 0), (3, 11, 0)]:
            self.assertTrue(allows(sdk, supported), supported)
        for unsupported in [(2, 17, 0), (3, 0, 0), (3, 6, 2), (4, 0, 0)]:
            self.assertFalse(allows(sdk, unsupported), unsupported)

    def test_flutter_floor_includes_the_required_dart_sdk(self):
        # Flutter 3.29.0 is the first stable release with Dart 3.7.0.
        constraint = environment(ROOT / 'pubspec.yaml')['flutter']
        self.assertFalse(allows(constraint, (3, 27, 4)))
        self.assertTrue(allows(constraint, (3, 29, 0)))
        self.assertTrue(allows(constraint, (3, 47, 2)))

    def test_example_does_not_advertise_an_older_sdk_than_its_library(self):
        library = environment(ROOT / 'pubspec.yaml')
        example = environment(ROOT / 'example/pubspec.yaml')
        self.assertEqual(bounds(example['sdk']), bounds(library['sdk']))
        self.assertEqual(bounds(example['flutter']), bounds(library['flutter']))


if __name__ == '__main__':
    unittest.main()
