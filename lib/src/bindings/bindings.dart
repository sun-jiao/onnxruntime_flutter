import 'dart:ffi';
import 'dart:io';
import 'package:onnxruntime/src/bindings/onnxruntime_bindings_generated.dart';

final DynamicLibrary _dylib = () {
  // The native test runner pins the exact library, including in worker isolates.
  // In particular, Windows may find a system DLL before directories on PATH.
  if (Platform.isWindows || Platform.isLinux || Platform.isMacOS) {
    final testLibrary = Platform.environment['ORT_TEST_LIBRARY_PATH'];
    if (testLibrary != null && testLibrary.isNotEmpty) {
      return DynamicLibrary.open(testLibrary);
    }
  }

  if (Platform.isAndroid) {
    return DynamicLibrary.open('libonnxruntime.so');
  }

  if (Platform.isIOS) {
    return DynamicLibrary.process();
  }

  if (Platform.isMacOS) {
    return DynamicLibrary.open('libonnxruntime.1.23.2.dylib');
  }

  if (Platform.isWindows) {
    return DynamicLibrary.open('onnxruntime.dll');
  }

  if (Platform.isLinux) {
    return DynamicLibrary.open('libonnxruntime.so.1.30.0');
  }

  throw UnsupportedError('Unknown platform: ${Platform.operatingSystem}');
}();

/// OnnxRuntime Bindings
final onnxRuntimeBinding = OnnxRuntimeBindings(_dylib);
