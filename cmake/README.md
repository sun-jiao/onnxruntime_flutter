# Native dependency downloads

The repository and pub package contain no ONNX Runtime desktop binaries.
Linux/Windows CMake downloads only the selected target's official release during
configuration, verifies the archive and library SHA-256 hashes, and packages the
runtime plus provider support library. Target selection uses Flutter's target
platform before the host processor. Unsupported targets fail explicitly.

macOS keeps the 1.23.2 universal2 dylib. Its local CocoaPods spec runs
`macos/download_runtime.rb` before CocoaPods enumerates vendored libraries.
This is intentional: CocoaPods does not run `prepare_command` for local/path
pods. The helper uses macOS's Ruby, curl and tar, verifies SHA-256, and copies the
library/notices into the ignored `macos/.onnxruntime/` staging directory.
No additional Python or CMake installation is required for macOS app builds.

## Cache and offline builds

Set `ORT_CACHE_DIR` to an absolute writable directory to share downloaded archives
between builds, platforms and the native test runner. Defaults are:

- Linux/Windows builds: `<CMake build directory>/_deps/onnxruntime`.
- macOS CocoaPods: `~/.cache/onnxruntime_flutter`.
- Native Dart test runner: `<repository>/build/native`.

Each artifact has its own SHA-256-named cache directory with an `archive` file.
An empty cache requires network access to GitHub releases. After a successful
fetch, subsequent builds can use the cache offline. To prepare a cache on a
connected machine (Python 3 and CMake required for this helper):

```sh
python3 tool/fetch_native_libraries.py --target linux-x64 --cache-dir /path/to/cache
python3 tool/fetch_native_libraries.py --target windows-arm64 --cache-dir /path/to/cache
python3 tool/fetch_native_libraries.py --target macos-universal2 --cache-dir /path/to/cache
```

Copy the cache to the build machine and set `ORT_CACHE_DIR` there. Targets also
include `linux-arm64` and `windows-x64`. Downloads/extraction are serialized per
artifact; interrupted downloads are never promoted to verified archives. A bad
archive fails with a checksum error; remove that archive to retry the download.
Modified extracted libraries are repaired from the verified cached archive.
Neither downloader writes runtime binaries into tracked source locations.

## Version maintenance and tests

`tool/native_runtime_versions.json` is the source of truth for versions, archive
URLs/hashes, member mappings and library hashes. After editing it, regenerate the
CMake 3.10-compatible constants with `python3 tool/generate_native_downloads.py`.

Run offline downloader/target-selection tests:

```sh
python3 -m unittest discover -s test/packaging -v
```

Then run `python3 tool/run_native_tests.py`; it fetches the pinned library for the
host before executing Flutter tests. App builds do not invoke this Python helper.
Linux/Windows install the upstream license/notices under `share/onnxruntime`;
macOS preserves them next to the staged dylib for CocoaPods license handling.

See [runtime compatibility](../tool/RUNTIME_COMPATIBILITY.md) for versions,
system minimums, Dart compatibility and device-validation limits.
