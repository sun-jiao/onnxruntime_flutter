<p align="center"><img width="50%" src="https://github.com/microsoft/onnxruntime/raw/main/docs/images/ONNX_Runtime_logo_dark.png" /></p>

# OnnxRuntime Plugin
[![pub package](https://img.shields.io/pub/v/onnxruntime.svg)](https://pub.dev/packages/onnxruntime)

## Overview

Flutter plugin for OnnxRuntime via native FFI and WebAssembly provides an easy, flexible, and fast Dart API to integrate Onnx models in flutter apps across mobile, desktop, and web platforms.

| **Platform**      | Android       | iOS | Linux | macOS | Windows | Web |
|-------------------|---------------|-----|-------|-------|---------|-----|
| **Compatibility** | API level 24+ | 15.1+ | glibc 2.28+ | 13.4+ | *       | Modern browsers |
| **Architecture**  | arm32/arm64/x86/x64 | * | x64/arm64 | x64/arm64 | x64/arm64 | WASM CPU |

*: [Consistent with Flutter](https://docs.flutter.dev/reference/supported-platforms)

Desktop builds download and cache the verified native library for the target
architecture; no desktop runtime binaries are shipped in the pub package. The
first build needs network access or a prefilled cache. See
[desktop packaging](cmake/README.md) for binary provenance and validation.

## Key Features

* Multi-platform Support for Android, iOS, Linux, macOS, Windows, and Web (WASM CPU).
* Flexibility to use any Onnx Model.
* Acceleration using multi-threading.
* Similar structure as OnnxRuntime Java and C# API.
* Inference speed is not slower than native Android/iOS Apps built using the Java/Objective-C API.
* Run inference in different isolates to prevent jank in UI thread.

## Getting Started

Requires Dart >=3.7.0 <4.0.0 and Flutter >=3.29.0, matching the SDK
requirement of the runtime dependency `ffi 2.2.0`. Flutter 3.29 ships with
Dart 3.7; see the [official release archive](https://docs.flutter.dev/release/archive-whats-new).

For repository development, use the Flutter version pinned in CI. Development
tools such as `ffigen` have higher SDK requirements than the published library's
runtime dependencies.

In your flutter project add the dependency:

```yml
dependencies:
  ...
  onnxruntime: x.y.z
```

## Usage example

### Import

```dart
import 'package:onnxruntime/onnxruntime.dart';
```

### Initializing environment

```dart
OrtEnv.instance.init();
```

### Creating the Session

```dart
final sessionOptions = OrtSessionOptions();
const assetFileName = 'assets/models/test.onnx';
final rawAssetFile = await rootBundle.load(assetFileName);
final bytes = rawAssetFile.buffer.asUint8List();
final session = OrtSession.fromBuffer(bytes, sessionOptions!);
```

### Performing inference

```dart
final shape = [1, 2, 3];
final inputOrt = OrtValueTensor.createTensorWithDataList(data, shape);
final inputs = {'input': inputOrt};
final runOptions = OrtRunOptions();
final outputs = await _session?.runAsync(runOptions, inputs);
inputOrt.release();
runOptions.release();
outputs?.forEach((element) {
  element?.release();
});
```

### Releasing environment

```dart
OrtEnv.instance.release();
```


### Web setup

Web uses [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html)
1.23.2 with the WebAssembly CPU backend. The public import, synchronous
`OrtSession.fromBuffer` constructor, tensor factories, options, `runAsync`, and
`release` calls stay the same. Native platforms continue using the existing FFI
implementation.

Load the runtime **before** Flutter in your application's `web/index.html`:

```html
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/ort.min.js"></script>
<script>
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/';
  ort.env.wasm.numThreads = 1;
</script>
<script src="flutter_bootstrap.js" async></script>
```

For offline deployments or a restrictive CSP, host `ort.min.js`,
`ort-wasm-simd-threaded.jsep.mjs`, and `ort-wasm-simd-threaded.jsep.wasm` from that
same npm package/version on your own server and change both URLs. Serve `.mjs`
as JavaScript and `.wasm` as `application/wasm`. A different runtime version is
not covered by the regression tests. No model data is uploaded by this library.

Load ONNX model bytes using `rootBundle.load` or HTTP, then use the existing API:

```dart
final asset = await rootBundle.load('assets/models/model.onnx');
final options = OrtSessionOptions();
final session = OrtSession.fromBuffer(
  asset.buffer.asUint8List(asset.offsetInBytes, asset.lengthInBytes), options);
options.release();
final input = OrtValueTensor.createTensorWithDataList(
  Float32List.fromList([1, 2]), [1, 2]);
final runOptions = OrtRunOptions();
try {
  final outputs = await session.runAsync(runOptions, {'input': input});
  try {
    // Inspect outputs; an empty list indicates an asynchronous inference error.
    print(outputs?.map((output) => output?.value).toList());
  } finally {
    outputs?.forEach((output) => output?.release());
  }
} finally {
  input.release();
  runOptions.release();
  session.release();
}
```

Web compatibility details:

- `run()` keeps its signature but throws `UnsupportedError` on Web: the upstream
  JavaScript inference API is asynchronous. Use `runAsync()` on shared code paths.
  The first run also initializes the WASM session. Errors are logged and return
  `[]`, matching the native asynchronous API. Accepted runs finish before release.
- Input/output names, counts and custom metadata are available immediately from
  ONNX protobuf bytes. ORT-format models and external weight files are not
  supported by this constructor on Web. Full model/operator validation happens
  during the first run.
- Numeric, boolean and string tensors are supported. Use typed lists such as
  `Float32List` or `Int32List` to select a numeric type explicitly. Plain
  `List<int>` becomes int64. JavaScript Dart cannot represent every 64-bit integer;
  inputs/outputs outside ±9007199254740991 throw instead of silently rounding.
  Dart `Int64List`/`Uint64List` construction itself is unavailable in JS builds.
  Float16/BFloat16 value extraction, sequences, maps and sparse tensors are not
  supported by this Web adapter.
- Native pointers, addresses, `fromFile`, and isolate construction have no browser
  equivalent and throw `UnsupportedError`. `isolateSession` is `null`. WASM CPU is
  reported as `OrtProvider.cpu`; other existing provider append methods return
  `false`. No GPU provider is exposed through the existing API.
- Thread count is global to the Web runtime, fixed when the first session starts.
  Configure it in HTML or before the first run. One thread works without special
  HTTP headers; multiple threads require browser cross-origin isolation (COOP /
  COEP). Native affinity/spinning controls are unsupported. `setTerminate()` stops
  queued runs using those options; it cannot interrupt JavaScript already running.
- The browser adapter does not create a Dart isolate. Inference can occupy the
  main thread; configuring ONNX Runtime's `ort.env.wasm.proxy = true` before
  initialization moves WASM work to its worker (subject to your hosting/CSP setup).

The example includes a Web entry point: run `flutter run -d chrome` from
`example/`. Browser inference regression tests run with
`python3 tool/run_web_tests.py`; see [test instructions](test/README.md).

### QNN execution provider

`appendQnnProvider()` returns `false` when the loaded runtime does not include
QNN. With a QNN-enabled runtime it registers `QNN` using the HTP backend's
standard library name (`QnnHtp.dll` on Windows, `libQnnHtp.so` elsewhere).
Install the matching Qualcomm backend and its dependencies on the native library
search path. Native registration errors propagate as for the other providers.
The bundled CPU runtimes do not acquire QNN support merely by calling this method.
See the [upstream QNN documentation](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html).

### Native runtime versions

Android, iOS, Linux and Windows use ORT 1.30.0. macOS uses ORT 1.23.2
to preserve Intel and Apple Silicon support. The Dart wrapper
uses C API 14. Model support and numerical results may differ between versions;
see the [compatibility contract and tests](tool/RUNTIME_COMPATIBILITY.md).

### Regression tests

After `flutter pub get`, run `python3 tool/run_native_tests.py` on a desktop host.
The runner selects the correct native library for the host architecture.
See [test coverage and CI](test/README.md) for packaging tests, the native runtime
matrix, and device-validation limits.
