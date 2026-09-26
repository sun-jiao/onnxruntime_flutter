<p align="center"><img width="50%" src="https://github.com/microsoft/onnxruntime/raw/main/docs/images/ONNX_Runtime_logo_dark.png" /></p>

# OnnxRuntime Plugin
[![pub package](https://img.shields.io/pub/v/onnxruntime.svg)](https://pub.dev/packages/onnxruntime)

## Overview

Flutter plugin for OnnxRuntime via `dart:ffi` provides an easy, flexible, and fast Dart API to integrate Onnx models in flutter apps across mobile and desktop platforms.

| **Platform**      | Android       | iOS | Linux | macOS | Windows |
|-------------------|---------------|-----|-------|-------|---------|
| **Compatibility** | API level 21+ | *   | *     | *     | *       |
| **Architecture**  | arm32/arm64   | *   | x64/arm64 | * | x64/arm64 |

*: [Consistent with Flutter](https://docs.flutter.dev/reference/supported-platforms)

Desktop builds select the native library for the target architecture. See
[desktop packaging](cmake/README.md) for binary provenance and validation.

## Key Features

* Multi-platform Support for Android, iOS, Linux, macOS, Windows, and Web(Coming soon).
* Flexibility to use any Onnx Model.
* Acceleration using multi-threading.
* Similar structure as OnnxRuntime Java and C# API.
* Inference speed is not slower than native Android/iOS Apps built using the Java/Objective-C API.
* Run inference in different isolates to prevent jank in UI thread.

## Getting Started

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


### QNN execution provider

`appendQnnProvider()` returns `false` when the loaded runtime does not include
QNN. With a QNN-enabled runtime it registers `QNN` using the HTP backend's
standard library name (`QnnHtp.dll` on Windows, `libQnnHtp.so` elsewhere).
Install the matching Qualcomm backend and its dependencies on the native library
search path. Native registration errors propagate as for the other providers.
The bundled CPU runtimes do not acquire QNN support merely by calling this method.
See the [upstream QNN documentation](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html).

### Native runtime versions

Android currently uses ORT 1.23.2; the other platforms use 1.15.1. The Dart wrapper
uses C API 14. Model support and numerical results may differ between versions;
see the [compatibility contract and tests](tool/RUNTIME_COMPATIBILITY.md).

### Regression tests

After `flutter pub get`, run `python3 tool/run_native_tests.py` on a desktop host.
The runner selects the correct native library for the host architecture.
See [test coverage and CI](test/README.md) for packaging tests, the native runtime
matrix, and device-validation limits.
