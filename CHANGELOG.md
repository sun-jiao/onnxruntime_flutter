## Unreleased

- Add runAsyncOrThrow and structured asynchronous inference diagnostics without changing legacy empty-list failures.

- Add opt-in model input/output descriptions, tensor shape/type getters and input validation on native and Web backends.

- Add a conditional ONNX Runtime Web WASM backend while retaining the native Dart API.
- Support existing asynchronous tensor inference, model names/metadata and release calls on Web.
- Add a runnable Web example and browser regression tests; document Web-specific limitations.

## Unreleased

* Upgrade ONNX Runtime to 1.30.0 on Android, iOS, Linux and Windows; upgrade macOS to 1.23.2 universal2 to retain Intel support.
* Require Android API 24+, iOS 15.1+, macOS 13.4+, and Linux glibc 2.28+.
* Update the Android example to AGP 8.13.2, Gradle 8.14.4, Kotlin 2.3.21 and JVM 17; verify 16 KB native alignment.
* Download and cache desktop runtime/provider libraries during native builds instead of committing binaries or publishing them in the pub package. Pin archive/library SHA-256 hashes and support prefilled offline caches.
* Preserve public Dart APIs and C API 14. Runtime version strings, native errors, supported models and numerical results can change with ONNX Runtime; see tool/RUNTIME_COMPATIBILITY.md.

## 1.4.1

* Fixes a memory leak when creating tensor.

## 1.4.0

* Fixes a memory leak when creating tensor.

## 1.3.0

* Attempts to support macOS, Windows and Linux.

## 1.2.0

* Compatible with Gradle8.

## 1.1.0

* Exposes some methods of input and output name.
* Adds some documentation comments.

## 1.0.0

* Initial release.
