import 'dart:ffi';

import 'package:ffi/ffi.dart';
import 'package:onnxruntime/src/bindings/onnxruntime_bindings_generated.dart'
    as bg;
import 'package:onnxruntime/src/ort_status.dart';
import 'package:onnxruntime/src/util/native_memory.dart';

// Keep native ownership at this boundary: metadata belongs to ORT, the returned
// string belongs to its allocator, and holders/key bytes belong to the arena.
String readModelMetadata(Pointer<bg.OrtApi> api, Pointer<bg.OrtSession> session,
    Pointer<bg.OrtAllocator> allocator, String key) {
  return usingNative((arena) {
    final metadata = arena<Pointer<bg.OrtModelMetadata>>();
    try {
      final metadataStatus = api.ref.SessionGetModelMetadata.asFunction<
          bg.OrtStatusPtr Function(Pointer<bg.OrtSession>,
              Pointer<Pointer<bg.OrtModelMetadata>>)>()(session, metadata);
      OrtStatus.checkOrtStatus(metadataStatus);

      final value = arena<Pointer<Char>>();
      try {
        final lookupStatus = api.ref.ModelMetadataLookupCustomMetadataMap
                .asFunction<
                    bg.OrtStatusPtr Function(
                        Pointer<bg.OrtModelMetadata>,
                        Pointer<bg.OrtAllocator>,
                        Pointer<Char>,
                        Pointer<Pointer<Char>>)>()(metadata.value, allocator,
            key.toNativeUtf8(allocator: arena).cast(), value);
        OrtStatus.checkOrtStatus(lookupStatus);
        if (value.value == nullptr) {
          // Preserve the existing ffi.toDartString error for a missing key.
          throw UnsupportedError(
              "Operation 'toDartString' not allowed on a 'nullptr'.");
        }
        return value.value.cast<Utf8>().toDartString();
      } finally {
        if (value.value != nullptr) {
          final status = api.ref.AllocatorFree.asFunction<
              bg.OrtStatusPtr Function(Pointer<bg.OrtAllocator>,
                  Pointer<Void>)>()(allocator, value.value.cast());
          OrtStatus.checkOrtStatus(status);
        }
      }
    } finally {
      // Nested finally blocks also unwind if AllocatorFree reports an error.
      if (metadata.value != nullptr) {
        api.ref.ReleaseModelMetadata
                .asFunction<void Function(Pointer<bg.OrtModelMetadata>)>()(
            metadata.value);
      }
    }
  });
}
