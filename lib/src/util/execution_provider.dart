import 'dart:ffi' as ffi;
import 'dart:io';

import 'package:ffi/ffi.dart';
import 'package:onnxruntime/src/bindings/onnxruntime_bindings_generated.dart'
    as bg;
import 'package:onnxruntime/src/ort_provider.dart';
import 'package:onnxruntime/src/ort_status.dart';
import 'package:onnxruntime/src/util/native_memory.dart';

// Internal entry point also permits testing provider registration without QNN
// hardware. No new options are added to the public session-options API.
bool appendExecutionProvider(
    ffi.Pointer<bg.OrtApi> api,
    ffi.Pointer<bg.OrtSessionOptions> sessionOptions,
    OrtProvider provider,
    Map<String, String> providerOptions,
    {required List<OrtProvider> Function() availableProviders}) {
  return usingNative((arena) {
    final String providerName;
    var options = providerOptions;
    switch (provider) {
      case OrtProvider.qnn:
        if (!availableProviders().contains(OrtProvider.qnn)) {
          return false;
        }
        providerName = 'QNN';
        options = {
          'backend_path': Platform.isWindows ? 'QnnHtp.dll' : 'libQnnHtp.so',
          ...providerOptions,
        };
        break;
      case OrtProvider.xnnpack:
        providerName = 'XNNPACK';
        break;
      default:
        return false;
    }
    final name = providerName.toNativeUtf8(allocator: arena).cast<ffi.Char>();
    final keys = arena<ffi.Pointer<ffi.Char>>(options.length);
    final values = arena<ffi.Pointer<ffi.Char>>(options.length);
    var i = 0;
    for (final entry in options.entries) {
      keys[i] = entry.key.toNativeUtf8(allocator: arena).cast<ffi.Char>();
      values[i] = entry.value.toNativeUtf8(allocator: arena).cast<ffi.Char>();
      ++i;
    }
    final status = api.ref.SessionOptionsAppendExecutionProvider.asFunction<
        bg.OrtStatusPtr Function(
            ffi.Pointer<bg.OrtSessionOptions>,
            ffi.Pointer<ffi.Char>,
            ffi.Pointer<ffi.Pointer<ffi.Char>>,
            ffi.Pointer<ffi.Pointer<ffi.Char>>,
            int)>()(sessionOptions, name, keys, values, options.length);
    OrtStatus.checkOrtStatus(status);
    return true;
  });
}
