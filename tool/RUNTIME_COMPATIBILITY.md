# Native runtime compatibility

## Pinned versions and requirements

| Platform | ONNX Runtime | Native requirement |
| --- | --- | --- |
| Android | 1.30.0 | API 24+; ARM32, ARM64, x86, x64 in the upstream AAR |
| iOS | 1.30.0 | iOS 15.1+ (upstream CocoaPods requirement) |
| Linux | 1.30.0 | x64/ARM64; glibc 2.28+, libstdc++ with GLIBCXX_3.4.22 |
| macOS | 1.23.2 | macOS 13.4+; universal Intel/Apple Silicon library |
| Windows | 1.30.0 | x64/ARM64; matching Microsoft Visual C++ runtime |

macOS deliberately uses the last universal2 release, 1.23.2. Newer upstream
GitHub releases provide only ARM64 macOS archives. Both slices of the downloaded
1.23.2 library declare a 13.4 minimum in LC_BUILD_VERSION. Do not replace this
library with an ARM64-only archive without explicitly dropping Intel support.

The Android example uses AGP 8.13.2, Gradle 8.14.4, Kotlin 2.3.21 and JVM target
17. Run Gradle with JDK 17 or 21 (JDK 25 requires Gradle 9). Compile/target SDK is
36. Keep the AGP 8 line to avoid requiring all consuming Flutter plugins to
migrate to AGP 9's built-in Kotlin/new DSL. The FFI plugin itself does not apply
Kotlin or pin the host application's AGP/NDK. It packages prebuilt libraries;
changing the host NDK cannot rebuild or change their alignment. The upstream
Android ARM64/x64 libraries have 16 KB-aligned LOAD segments.

## Dart compatibility and observable differences

Public Dart methods, enums, ownership, sync/async semantics and default C API 14
are unchanged. The only Dart implementation change is the private desktop
library filenames. The headers under `src/onnxruntime/` and generated Dart
bindings intentionally remain the C API 14 snapshot: upgrading a native runtime
does not require regenerating or expanding the exposed API table.

`OrtEnv.version` now reports the versions above. New runtimes can change model
acceptance, execution-provider availability, graph optimizations, performance,
floating-point results and native error messages. These native differences
cannot be ruled out by preserving Dart signatures. For example, invalid input
names now report `Invalid input name: missing`, versus 1.15.1's
`Invalid Feed Input Name`. Native errors continue to pass through unchanged.
Applications should regression-test their own models on target devices.

ORT 1.30's Android AAR adds a Java telemetry startup provider and network
permissions (`INTERNET` and `ACCESS_NETWORK_STATE`). The plugin manifest removes
the provider, keeping FFI initialization lazy. The upstream network permissions
remain in the merged app manifest; apps which do not use networking can remove
them in their own manifest. This is not a claim about all upstream native telemetry.

## Reproducibility and validation

`native_runtime_versions.json` records platform versions, library SHA-256 hashes,
and official archive URLs/hashes/member mappings. Desktop binaries are downloaded
on demand and are not committed or included in the pub package. Linux/Windows use
CMake; macOS prepares the universal dylib when CocoaPods evaluates the local Pod.
See [download/cache instructions](../cmake/README.md), including offline builds.
The first build needs GitHub access unless its cache was prefilled. Libraries
still ship inside the final app; the change removes them from the source package.

Run `python3 -m unittest discover -s test/packaging -v`, then after
`flutter pub get`, run `python3 tool/run_native_tests.py`. The test runner fetches
the host's pinned library automatically. Offline packaging tests use tiny local
archives to check architecture selection, checksum failure, cache repair and
manifest consistency without relying on committed binaries. Native tests cover
C API 14, fixed outputs, metadata, strings, sequences, maps, errors, memory and
isolate lifetime. Fixtures use IR 8 / opset 13.

To test another runtime, place it under the current loader's filename in an
isolated directory and run:

```sh
python3 tool/run_native_tests.py --runtime-dir /path/to/runtime --runtime-version 1.23.2
```

The version override is used only by tests. CI tests downloaded Linux x64/ARM64,
Windows x64 and macOS ARM64, plus Linux 1.23.2 as a wrapper compatibility check
for the macOS runtime version. Host tests do not replace iOS/Android device tests,
Windows ARM64 execution, or CoreML/NNAPI/provider-specific model validation.

## Upstream references

- [ONNX Runtime 1.30.0](https://github.com/microsoft/onnxruntime/releases/tag/v1.30.0)
- [ONNX Runtime 1.23.2 universal2](https://github.com/microsoft/onnxruntime/releases/tag/v1.23.2)
- [iOS Pod specification](https://trunk.cocoapods.org/api/v1/pods/onnxruntime-objc/specs/1.30.0)
- [AGP 8.13.2 compatibility](https://developer.android.com/build/releases/agp-8-13-0-release-notes)
- [Gradle 8.14.4](https://docs.gradle.org/8.14.4/release-notes.html)
- [Kotlin compatibility](https://kotlinlang.org/docs/gradle-configure-project.html)

## Local upgrade validation (2026-09-26)

- Linux x64 with downloaded ORT 1.30.0: all 71 Dart native tests passed.
- Offline packaging/downloader tests cover target selection, checksum rejection,
  corrupted-cache repair and absence of source-tree binaries.
- Android example Debug and Release builds passed using JDK 21. Release APK
  passed `zipalign -c -P 16 4`; packaged ORT ARM64/x64 LOAD segments are 16 KB
  aligned. The merged manifest does not contain the telemetry initializer.
- Linux Release build passed; both packaged ONNX libraries match manifest hashes.
- Dart analysis reported no errors/warnings, with two existing doc-comment infos.
- CocoaPods dependency analysis generated both Apple lockfiles. Full Apple
  installation/build could not be completed on this Linux host without the
  macOS Flutter engine/Xcode. Exported iOS C API entry points and macOS universal
  architecture/minimum OS metadata were inspected directly.
- Windows and Apple runtime execution was not performed locally. Android/Apple
  build jobs have been added to CI, but their remote results are not yet known.

## Download migration validation

After removing source-tree binaries, all 71 Dart native tests and the Linux
Release example build passed with the downloaded cache. All 14 offline tests
passed, including CMake download hash rejection, corrupt-cache repair, architecture
selection and the Ruby local-pod preparation flow. A real empty-cache Linux
download succeeded. All five official desktop archives were extracted/verified
through the new cache flow; CocoaPods recognized the staged macOS dylib and
regenerated its lockfile. macOS/Windows application execution still needs their
native hosts. `flutter pub publish --dry-run` listed a 2 MB compressed package
with no desktop ONNX libraries; its only two warnings were uncommitted changes
and the intentionally deleted tracked files. No package was published.
